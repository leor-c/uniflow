"""Regression checks for refinement and unmodified diagnostic sampling."""

import unittest
from unittest.mock import patch

import torch

from core import rf, ayf_rf
from toy.uniform_rf import Config, make_plot, main
from toy.common import run_config_record


class RefinementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.cfg = Config(
            device="cpu", hidden=8, depth=1, batch_size=4,
            teacher_steps=3, teacher_refine_steps=2,
            student_steps=3, student_refine_steps=2, log_every=100,
        )

    def test_teacher_refines_before_reflow_with_separate_learning_rate(self):
        teacher = rf.RFTeacher(2, hidden=8, depth=1, num_freq=4)
        before = [p.detach().clone() for p in teacher.parameters()]
        rates = []
        original = torch.optim.AdamW

        class RecordingAdamW(original):
            def step(self, *args, **kwargs):
                rates.append(self.param_groups[0]["lr"])
                return super().step(*args, **kwargs)

        with patch("torch.optim.AdamW", RecordingAdamW), patch(
            "core.rf.training.train_teacher_reflow",
            side_effect=lambda *a, **k: self.assertEqual(len(rates), 5),
        ) as reflow:
            rf.train_teacher(teacher, lambda n: torch.rand(n, 2), self.cfg)
        self.assertEqual(rates, [1e-3] * 3 + [1e-4] * 2)
        reflow.assert_called_once()
        self.assertFalse(teacher.training)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, teacher.parameters())))
        self.assertEqual(self.cfg.teacher_steps, 3)
        self.assertEqual(self.cfg.teacher_lr, 1e-3)

    def test_student_refinement_continues_warmup_and_preserves_teacher(self):
        self.cfg.train_mode = "emd"
        self.cfg.tangent_warmup_steps = 4
        teacher = rf.RFTeacher(2, hidden=8, depth=1, num_freq=4)
        student = ayf_rf.RFIntervalFlowMap(2, hidden=8, depth=1, num_freq=4)
        before = [p.detach().clone() for p in teacher.parameters()]
        warmups = []

        def loss(model, *args, **kwargs):
            warmups.append(kwargs["tangent_warmup"])
            return sum(p.square().mean() for p in model.parameters())

        with patch("core.ayf_rf.training.ayf_emd_loss", side_effect=loss):
            ayf_rf.train_student_interval(student, teacher, lambda n: torch.rand(n, 2), self.cfg)
        self.assertEqual(warmups, [0.0, 0.25, 0.5, 0.75, 0.99])
        for a, b in zip(before, teacher.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(student.training)
        self.assertEqual(self.cfg.student_steps, 3)

    def test_refinement_and_sampling_settings_are_recorded(self):
        record = run_config_record(self.cfg)
        self.assertEqual(record["teacher_refine_steps"], 2)
        self.assertEqual(record["student_refine_lr"], 1e-4)
        self.assertEqual(record["teacher_sample_steps"], 256)
        self.assertEqual(record["sample_clamp"], True)

    def test_checkpoint_cannot_be_overwritten_by_evaluation_samples(self):
        cfg = Config(metrics_out="/tmp/rf_test.json", checkpoint_out="/tmp/rf_test.pt")
        with self.assertRaisesRegex(ValueError, "checkpoint must differ"):
            main(cfg)


class SamplingTests(unittest.TestCase):
    def test_clamp_flag_controls_teacher_and_both_student_outputs(self):
        class IdentityAE:
            encode = decode = staticmethod(lambda z: z)

        raw = torch.tensor([[1.5, -1.5]])
        for clamp in (False, True):
            cfg = Config(device="cpu", save_figure=False, sample_clamp=clamp)
            with self.subTest(clamp=clamp), patch(
                "toy.uniform_rf.R.sample_teacher_latents", return_value=raw
            ), patch(
                "toy.uniform_rf.A.sample_student_latents_one_step", return_value=raw
            ), patch(
                "toy.uniform_rf.A.sample_student_latents_k_step",
                side_effect=lambda *a, **kw: raw.clamp(-1, 1) if kw["clamp"] else raw,
            ), patch(
                "toy.uniform_rf.evaluate_density_outputs", return_value={}
            ) as evaluate, patch("toy.uniform_rf.save_evaluation_samples"), patch(
                "toy.uniform_rf.save_metrics"
            ):
                make_plot(IdentityAE(), None, None, lambda n: torch.zeros(n, 2), cfg)
                outputs = evaluate.call_args.args[2]
                for name in ("teacher", "student_1_step", "student_8_step"):
                    torch.testing.assert_close(outputs[name], raw.clamp(-1, 1) if clamp else raw)
