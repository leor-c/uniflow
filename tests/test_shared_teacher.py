"""The shared-architecture teacher: one class for teacher and student.

``core.ayf_rf.train_map_teacher`` fits an ``RFIntervalFlowMap`` as the RF
teacher, so the student can be initialized with a full weight copy instead of
from scratch. These tests pin the three things that make that work: the
velocity identification is exact, every tensor copies, and the copy is what the
toy arm actually performs when the flag is on.
"""

import tempfile
import unittest
from unittest.mock import patch

import torch

from core import ayf_rf, rf
from toy.uniform_rf import Config, main


def _cfg(**overrides):
    base = dict(device="cpu", latent_dim=2, hidden=8, depth=1, batch_size=4,
                teacher_steps=3, student_steps=3, log_every=100)
    base.update(overrides)
    return Config(**base)


class VelocityViewTests(unittest.TestCase):
    def test_velocity_view_is_exactly_the_s_equals_one_slope(self):
        flow_map = ayf_rf.RFIntervalFlowMap(2, hidden=8, depth=1, num_freq=4)
        view = ayf_rf.VelocityView(flow_map)
        z = torch.randn(5, 2)
        t = torch.rand(5, 1)
        self.assertTrue(torch.equal(
            view(z, t), flow_map.slope(z, t, torch.ones_like(t))))

    def test_velocity_view_accepts_a_flat_time_vector_like_rfteacher(self):
        flow_map = ayf_rf.RFIntervalFlowMap(2, hidden=8, depth=1, num_freq=4)
        view = ayf_rf.VelocityView(flow_map)
        z = torch.randn(5, 2)
        flat, column = torch.rand(5), None
        column = flat[:, None]
        self.assertTrue(torch.equal(view(z, flat), view(z, column)))

    def test_velocity_view_shares_weights_rather_than_copying_them(self):
        flow_map = ayf_rf.RFIntervalFlowMap(2, hidden=8, depth=1, num_freq=4)
        view = ayf_rf.VelocityView(flow_map)
        self.assertIs(view.flow_map, flow_map)


class MapTeacherTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = _cfg()

    def test_training_updates_the_map_and_returns_a_velocity_view(self):
        flow_map = ayf_rf.RFIntervalFlowMap(2, hidden=8, depth=1, num_freq=4)
        before = [p.detach().clone() for p in flow_map.parameters()]
        view = ayf_rf.train_map_teacher(
            flow_map, lambda n: torch.rand(n, 2) * 2 - 1, self.cfg)
        self.assertIsInstance(view, ayf_rf.VelocityView)
        self.assertIs(view.flow_map, flow_map)
        self.assertTrue(any(not torch.equal(a, b)
                            for a, b in zip(before, flow_map.parameters())))

    def test_target_time_is_sampled_not_pinned_to_one(self):
        """Pinning s=1 leaves emb(s-t) a function of t alone, which the student
        then has to unlearn. The schedule must be the student's own."""
        seen = []
        real = ayf_rf.training.sample_ayf_time_pair

        def recording(*args, **kwargs):
            pair = real(*args, **kwargs)
            seen.append(pair[1])
            return pair

        with patch.object(ayf_rf.training, "sample_ayf_time_pair", recording):
            ayf_rf.train_map_teacher(
                ayf_rf.RFIntervalFlowMap(2, hidden=8, depth=1, num_freq=4),
                lambda n: torch.rand(n, 2) * 2 - 1, _cfg(teacher_steps=8))
        self.assertTrue(seen, "teacher never drew a target time")
        self.assertTrue(any(not torch.allclose(s, torch.ones_like(s))
                            for s in seen))

    def test_teacher_weights_load_into_a_student_of_the_same_config(self):
        kwargs = dict(hidden=8, depth=1, num_freq=4)
        teacher_map = ayf_rf.RFIntervalFlowMap(2, **kwargs)
        student = ayf_rf.RFIntervalFlowMap(2, **kwargs)
        student.load_state_dict(teacher_map.state_dict())
        for (name, a), b in zip(student.state_dict().items(),
                                teacher_map.state_dict().values()):
            self.assertTrue(torch.equal(a, b), f"{name} did not copy")

    def test_a_separate_rfteacher_cannot_supply_the_conditioning_projection(self):
        """The reason the shared class exists: RFTeacher conditions on emb(t)
        alone, so its conditioning projection is a third of the width the map
        needs -- and that one tensor is the *only* thing standing between them.
        Checked on both backbones, which name the projection differently."""
        for arch, extra in (("mlp", {}),
                            ("dit", dict(latent_shape=(1, 4, 4), heads=2))):
            with self.subTest(arch=arch):
                dim = 2 if arch == "mlp" else 16
                kwargs = dict(hidden=8, depth=1, num_freq=4, arch=arch, **extra)
                teacher = rf.RFTeacher(dim, **kwargs).state_dict()
                student = ayf_rf.RFIntervalFlowMap(dim, **kwargs).state_dict()
                mismatched = [k for k in student
                              if k not in teacher
                              or teacher[k].shape != student[k].shape]
                self.assertEqual(len(mismatched), 1, mismatched)
                name = mismatched[0]
                # Three time embeddings in, one out: exactly 3x the fan-in.
                self.assertEqual(student[name].shape[1], 3 * teacher[name].shape[1])


class ToyArmWiringTests(unittest.TestCase):
    def _run(self, shared):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(ae_steps=2, save_figure=False, shared_teacher=shared,
                       teacher_sample_steps=2, eval_teacher_ode_steps=2,
                       interval_teacher_ode_steps=2,
                       out=f"{tmp}/samples.png", metrics_out=f"{tmp}/metrics.json",
                       out_dir=tmp)
            captured = {}
            real_load = ayf_rf.RFIntervalFlowMap.load_state_dict

            def recording(self, state, *a, **k):
                captured["copied"] = True
                return real_load(self, state, *a, **k)

            with patch.object(ayf_rf.RFIntervalFlowMap, "load_state_dict",
                              recording), \
                    patch("toy.uniform_rf.save_evaluation_samples"), \
                    patch("toy.uniform_rf.evaluate_density_outputs",
                          return_value={}):
                main(cfg)
            return captured.get("copied", False)

    def test_flag_on_copies_teacher_weights_into_the_student(self):
        self.assertTrue(self._run(True))

    def test_flag_off_leaves_the_student_randomly_initialized(self):
        self.assertFalse(self._run(False))

    def test_flag_is_recorded_with_the_results(self):
        from toy.common import run_config_record
        self.assertIn("shared_teacher", run_config_record(_cfg(shared_teacher=True)))


if __name__ == "__main__":
    unittest.main()
