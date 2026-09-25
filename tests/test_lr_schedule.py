"""Check actual optimizer updates in every benchmark arm, including the LR boundary."""
from argparse import ArgumentParser
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from core import rf, ayf_rf, ayf, edm
from core.lr_schedule import make_lr_scheduler
from toy import uniform_rf, gaussian_rf, edm_baseline
from toy.lr_schedule import add_lr_arguments, apply_lr_overrides
from toy.common import run_config_record


class ConstantTeacher(torch.nn.Module):
    def forward(self, z, t):
        return torch.ones_like(z) * .1


class ScheduleTests(unittest.TestCase):
    def test_defaults_match_across_teachers_and_students(self):
        for module in (uniform_rf, gaussian_rf, edm_baseline):
            cfg = module.Config(device='cpu')
            teacher_prefix = 'edm_teacher' if module is edm_baseline else 'teacher'
            self.assertEqual(getattr(cfg, f'{teacher_prefix}_steps'), 80000)
            self.assertEqual(getattr(cfg, f'{teacher_prefix}_lr'), 1e-3)
            self.assertEqual(cfg.student_steps, 80000)
            self.assertEqual(cfg.student_lr, 1e-3)
            for role in ('teacher', 'student'):
                self.assertEqual(getattr(cfg, f'{role}_lr_drop_step'), 40000)
                self.assertEqual(getattr(cfg, f'{role}_final_lr'), 1e-4)
                self.assertEqual(run_config_record(cfg)[f'{role}_lr_drop_step'], 40000)

    def test_all_six_training_paths_drop_after_exactly_two_updates_without_reset(self):
        original = torch.optim.AdamW
        rates, optimizers = [], []

        class RecordingAdamW(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                optimizers.append(self)

            def step(self, *args, **kwargs):
                rates.append(self.param_groups[0]['lr'])
                return super().step(*args, **kwargs)

        for module in (uniform_rf, gaussian_rf, edm_baseline):
            cfg = module.Config(device='cpu', batch_size=4, log_every=100,
                                student_steps=4, teacher_lr_drop_step=2, student_lr_drop_step=2)
            for role in ('teacher', 'student'):
                with self.subTest(arm=module.__name__, role=role):
                    rates.clear(); optimizers.clear()
                    torch.manual_seed(7)
                    sampler = lambda n: torch.rand(n, 2) * 2 - 1
                    if role == 'student':
                        model = ayf_rf.RFIntervalFlowMap(2, hidden=8, depth=1, num_freq=4)
                        train = ayf_rf.train_student_interval if module is uniform_rf else ayf.train_student_ayf
                        call = lambda: train(model, ConstantTeacher(), sampler, cfg)
                    elif module is edm_baseline:
                        cfg.edm_teacher_steps = 4
                        model = edm.EDMPrecondDenoiser(2, hidden=8, depth=1, sigma_data=cfg.sigma_data)
                        call = lambda: edm.train_edm_teacher(model, sampler, cfg)
                    else:
                        cfg.teacher_steps = 4
                        model = rf.RFTeacher(2, hidden=8, depth=1, num_freq=4)
                        train = rf.train_teacher if module is uniform_rf else ayf.train_teacher_gaussian
                        call = lambda: train(model, sampler, cfg)
                    with patch('torch.optim.AdamW', RecordingAdamW):
                        call()
                    self.assertEqual(rates, [1e-3, 1e-3, 1e-4, 1e-4])
                    self.assertEqual(len(optimizers), 1)
                    self.assertEqual(next(iter(optimizers[0].state.values()))['step'].item(), 4)
                    self.assertFalse(model.training)

    def test_configs_without_schedule_keep_constant_lr(self):
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.001)
        self.assertIsNone(make_lr_scheduler(opt, SimpleNamespace(), 'teacher'))
        self.assertEqual(opt.param_groups[0]['lr'], .001)

    def test_cli_overrides_map_edm_teacher_and_allow_constant_lr(self):
        parser = ArgumentParser()
        add_lr_arguments(parser)
        args = parser.parse_args(['--teacher-lr', '.0002', '--teacher-final-lr', '.00002',
                                  '--teacher-lr-drop-step', '20', '--student-lr-drop-step', '-1'])
        cfg = edm_baseline.Config(device='cpu')
        apply_lr_overrides(cfg, args, teacher_lr_field='edm_teacher_lr')
        self.assertEqual(cfg.edm_teacher_lr, .0002)
        self.assertEqual(cfg.teacher_final_lr, .00002)
        self.assertEqual(cfg.teacher_lr_drop_step, 20)
        self.assertIsNone(cfg.student_lr_drop_step)
