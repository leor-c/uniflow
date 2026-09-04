"""
The three benchmark arms must run the *same* AYF algorithm.

Only the forward process differs between arms (uniform-prior RF, Gaussian-prior
RF, EDM probability flow). The AYF objective, its tangent normalisation, and
the interval schedule are shared, and these tests pin that down: a divergence
here silently turns the benchmark into a comparison of distillation recipes
rather than of base distributions.

Since the EDM teacher is wrapped as a flow-matching velocity field
(``core.edm.EDMFlowVelocity``), that arm now runs ``core.ayf`` itself rather
than a parallel implementation, so the sharing is structural.
"""

import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

import core.ayf as ayf_package
import core.ayf.losses as ayf_gaussian
import core.ayf_rf.losses as ayf_uniform
from core.ayf_rf.losses import normalize_tangent
from core.ayf_rf.training import parse_train_mode, resolve_train_mode
from core.ayf_rf.models import RFIntervalFlowMap
from core.ayf_rf.sampling import sample_ayf_time_pair

BATCH, DIM = 8, 4


class _ConstantTeacher(nn.Module):
    """RF teacher stub returning a fixed velocity for any (z, t)."""

    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.velocity = velocity

    def forward(self, z, t):
        return self.velocity.clone()


class AYFConsistencyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z = torch.randn(BATCH, DIM)
        self.t = torch.full((BATCH, 1), 0.3)
        self.s = torch.full((BATCH, 1), 0.75)
        self.v = torch.randn(BATCH, DIM)

    def _student(self):
        torch.manual_seed(123)
        return RFIntervalFlowMap(DIM, hidden=16, depth=2, num_freq=4, max_freq=4.0)

    def _grads(self, student, loss):
        loss.backward()
        return torch.cat(
            [p.grad.flatten() for p in student.parameters() if p.grad is not None],
        )

    def _gaussian(self, kind):
        student = self._student()
        loss_fn = getattr(ayf_gaussian, f"ayf_{kind}_loss")
        with patch.object(ayf_gaussian, "sample_noise", return_value=self.z.clone()), \
                patch.object(
                    ayf_gaussian, "sample_ayf_time_pair",
                    return_value=(self.t.clone(), self.s.clone()),
                ):
            loss = loss_fn(
                student, _ConstantTeacher(self.v), lambda b: self.z.clone(),
                BATCH, DIM, "cpu", jvp_dt_ref=1.0,
            )
        return loss, self._grads(student, loss)

    def _uniform(self, kind):
        student = self._student()
        loss_fn = getattr(ayf_uniform, f"ayf_{kind}_loss")
        with patch.object(ayf_uniform, "sample_uniform_noise", return_value=self.z.clone()), \
                patch.object(
                    ayf_uniform, "sample_ayf_time_pair",
                    return_value=(self.t.clone(), self.s.clone()),
                ):
            loss = loss_fn(
                student, _ConstantTeacher(self.v), lambda b: self.z.clone(),
                BATCH, DIM, "cpu", jvp_dt_ref=1.0,
            )
        return loss, self._grads(student, loss)

    def test_emd_objective_is_identical_across_arms(self):
        """Given the same (z, t, s, v), the EMD losses must agree."""
        loss, grad = self._uniform("emd")
        reference_loss, reference_grad = self._gaussian("emd")
        torch.testing.assert_close(loss, reference_loss, atol=1e-9, rtol=0.0)
        torch.testing.assert_close(grad, reference_grad, atol=1e-7, rtol=0.0)

    def test_lmd_objective_is_identical_across_arms(self):
        """Given the same (z, t, s, v), the LMD losses must agree."""
        loss, grad = self._uniform("lmd")
        reference_loss, reference_grad = self._gaussian("lmd")
        torch.testing.assert_close(loss, reference_loss, atol=1e-9, rtol=0.0)
        torch.testing.assert_close(grad, reference_grad, atol=1e-7, rtol=0.0)

    def test_edm_arm_reuses_the_shared_ayf_implementation(self):
        """The EDM arm must distil through core.ayf, not a parallel copy."""
        from toy import edm_baseline

        self.assertIs(edm_baseline.A, ayf_package)
        self.assertIs(ayf_package.RFIntervalFlowMap, RFIntervalFlowMap)
        source = (
            edm_baseline.main.__code__.co_names
            + edm_baseline.make_plot.__code__.co_names
        )
        self.assertIn("train_student_ayf", source)
        self.assertIn("EDMFlowVelocity", source)

    def test_every_arm_uses_the_one_interval_schedule(self):
        """Both loss modules must reference the same schedule function."""
        self.assertIs(ayf_gaussian.sample_ayf_time_pair, sample_ayf_time_pair)
        self.assertIs(ayf_uniform.sample_ayf_time_pair, sample_ayf_time_pair)

    def test_every_arm_uses_the_one_tangent_normalisation(self):
        self.assertIs(ayf_gaussian.normalize_tangent, normalize_tangent)
        self.assertIs(ayf_uniform.normalize_tangent, normalize_tangent)

    def test_interval_schedule_is_a_valid_forward_pair(self):
        t, s = sample_ayf_time_pair(4096, "cpu", eps=1e-4)
        self.assertTrue(bool((t >= 0.0).all()))
        self.assertTrue(bool((s <= 1.0).all()))
        self.assertTrue(bool((s > t).all()))


class TrainModeTests(unittest.TestCase):
    """``train_mode`` decides which AYF terms are active; weights only mix them.

    AYF reports LMD as the more stable objective on 2D toys and EMD as the
    better one on images, so both must be reachable, identically, in every arm.
    """

    def test_parses_single_and_combined_terms(self):
        self.assertEqual(parse_train_mode("emd"), frozenset({"emd"}))
        self.assertEqual(parse_train_mode("lmd"), frozenset({"lmd"}))
        self.assertEqual(parse_train_mode("emd+lmd"), frozenset({"emd", "lmd"}))
        self.assertEqual(
            parse_train_mode("ct+lmd", allow_ct=True), frozenset({"ct", "lmd"}),
        )

    def test_rejects_unknown_empty_and_contradictory_modes(self):
        for mode in ("", "eemd", "emd+bogus", "ct"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                parse_train_mode(mode)
        with self.assertRaises(ValueError):
            parse_train_mode("emd+ct", allow_ct=True)

    def test_active_terms_require_positive_weights(self):
        cfg = SimpleNamespace(train_mode="lmd", emd_weight=1.0, lmd_weight=0.0)
        with self.assertRaises(ValueError):
            resolve_train_mode(cfg)
        cfg = SimpleNamespace(train_mode="emd", emd_weight=0.0, lmd_weight=1.0)
        with self.assertRaises(ValueError):
            resolve_train_mode(cfg)

    def test_inactive_term_weight_is_inert(self):
        """The default lmd_weight must not switch LMD on behind train_mode."""
        cfg = SimpleNamespace(train_mode="emd", emd_weight=1.0, lmd_weight=1.0)
        self.assertEqual(resolve_train_mode(cfg), frozenset({"emd"}))

    def test_every_toy_arm_defaults_to_emd_only(self):
        from toy import edm_baseline, gaussian_rf, uniform_rf

        for module in (uniform_rf, gaussian_rf, edm_baseline):
            cfg = module.Config(device="cpu")
            with self.subTest(arm=module.__name__):
                self.assertEqual(cfg.train_mode, "emd")
                self.assertEqual(
                    resolve_train_mode(cfg, allow_ct=True), frozenset({"emd"}),
                )

    def test_every_toy_arm_can_select_lmd_the_same_way(self):
        from toy import edm_baseline, gaussian_rf, uniform_rf

        for module in (uniform_rf, gaussian_rf, edm_baseline):
            for mode, expected in (("lmd", {"lmd"}), ("emd+lmd", {"emd", "lmd"})):
                cfg = module.Config(device="cpu")
                cfg.train_mode = mode
                with self.subTest(arm=module.__name__, mode=mode):
                    self.assertEqual(
                        resolve_train_mode(cfg, allow_ct=True), frozenset(expected),
                    )


class StopGradientTests(unittest.TestCase):
    """AYF-EMD's target is a stopgrad copy: theta- = stopgrad(theta).

    Without it the model can lower the loss by shrinking its own Jacobian
    instead of satisfying the consistency condition, which is the degeneracy
    the stopgrad exists to prevent.
    """

    def test_emd_target_carries_no_gradient(self):
        import core.ayf.losses as gaussian
        from core.ayf.losses import ayf_emd_loss

        seen = {}
        real = gaussian.normalize_tangent

        def spy(tangent, *a, **k):
            seen["requires_grad"] = tangent.requires_grad
            return real(tangent, *a, **k)

        student = RFIntervalFlowMap(DIM, hidden=16, depth=2, num_freq=4, max_freq=4.0)
        with patch.object(gaussian, "normalize_tangent", side_effect=spy):
            ayf_emd_loss(
                student, _ConstantTeacher(torch.randn(BATCH, DIM)),
                lambda b: torch.randn(b, DIM), BATCH, DIM, "cpu",
            ).backward()

        self.assertFalse(seen["requires_grad"], "EMD tangent must be stopgrad'd")
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in student.parameters()))


class FourierParityTests(unittest.TestCase):
    """The Fourier settings must be identical across arms.

    ff_input_scale differs by design (bounded latents sit near sqrt(1/3),
    standardized ones near 1) precisely so that one nominal scale means the same
    effective frequency everywhere. An arm-specific ff_scale would undo that and
    make the comparison uncontrolled.
    """

    def test_all_toy_arms_share_the_fourier_scales(self):
        from toy import edm_baseline, gaussian_rf, uniform_rf

        configs = [m.Config(device="cpu") for m in (uniform_rf, gaussian_rf, edm_baseline)]
        for field in ("ff_size", "ae_ff_scale", "teacher_ff_scale", "student_ff_scale"):
            values = {getattr(c, field) for c in configs}
            with self.subTest(field=field):
                self.assertEqual(len(values), 1, f"{field} differs across arms: {values}")

    def test_input_scale_matches_each_arm_latent_convention(self):
        import math
        from toy import edm_baseline, gaussian_rf, uniform_rf

        self.assertAlmostEqual(uniform_rf.Config(device="cpu").ff_input_scale,
                               math.sqrt(1.0 / 3.0))
        for module in (gaussian_rf, edm_baseline):
            self.assertEqual(module.Config(device="cpu").ff_input_scale, 1.0)


class AutoencoderWidthTests(unittest.TestCase):
    """The AE trunk width is separate from the teacher/student width.

    ``hidden`` used to feed both, so widening the flow models silently widened
    the autoencoder too -- and the autoencoder determines the latent geometry
    every arm is then measured on, so that coupling made a "network capacity"
    change into a benchmark change.
    """

    def test_toy_arms_wire_ae_hidden_to_the_autoencoder(self):
        import ast

        for name in ("uniform_rf", "gaussian_rf", "edm_baseline"):
            source = ast.parse(pathlib.Path(f"toy/{name}.py").read_text())
            calls = [n for n in ast.walk(source)
                     if isinstance(n, ast.Call)
                     and getattr(n.func, "id", None) == "ToyAE"]
            self.assertEqual(len(calls), 1, f"{name}: expected one ToyAE construction")
            bound = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
            with self.subTest(arm=name):
                self.assertEqual(bound.get("hidden"), "cfg.ae_hidden")
                self.assertEqual(bound.get("depth"), "cfg.ae_depth")

    def test_ae_width_is_independent_of_the_flow_width(self):
        from toy.common import ToyAE

        narrow = ToyAE(latent_dim=2, hidden=32, depth=2, bounded=True)
        wide = ToyAE(latent_dim=2, hidden=256, depth=2, bounded=True)
        self.assertEqual(narrow.encoder_net.net[0].out_features, 32)
        self.assertEqual(wide.encoder_net.net[0].out_features, 256)

    def test_ae_hidden_is_recorded_in_the_run_config(self):
        from toy.common import run_config_record
        from toy import uniform_rf

        cfg = uniform_rf.Config(device="cpu")
        cfg.ae_hidden = 64
        record = run_config_record(cfg)
        self.assertEqual(record["ae_hidden"], 64)
        self.assertEqual(record["hidden"], cfg.hidden)


class ReflowTests(unittest.TestCase):
    """ReFlow refits the teacher on its own ODE coupling (Liu et al. 2022)."""

    @staticmethod
    def _cfg(rounds, **kw):
        from types import SimpleNamespace
        base = dict(latent_dim=DIM, device="cpu", batch_size=8, teacher_lr=1e-3,
                    teacher_steps=2, log_every=10 ** 9, reflow_rounds=rounds,
                    reflow_steps=2, reflow_pairs=16, reflow_ode_steps=2)
        base.update(kw)
        return SimpleNamespace(**base)

    def _teacher(self):
        from core.rf import RFTeacher
        torch.manual_seed(0)
        return RFTeacher(DIM, 16, 2)

    def test_zero_rounds_leaves_the_teacher_untouched(self):
        from core.rf import train_teacher_reflow

        teacher = self._teacher()
        before = [p.clone() for p in teacher.parameters()]
        train_teacher_reflow(teacher, self._cfg(0))
        for a, b in zip(before, teacher.parameters()):
            self.assertTrue(torch.equal(a, b), "reflow_rounds=0 must be a no-op")

    def test_one_round_trains_and_leaves_the_teacher_in_eval(self):
        from core.rf import train_teacher_reflow

        teacher = self._teacher()
        before = [p.clone() for p in teacher.parameters()]
        train_teacher_reflow(teacher, self._cfg(1))
        self.assertTrue(any(not torch.equal(a, b)
                            for a, b in zip(before, teacher.parameters())))
        self.assertFalse(teacher.training)

    def test_pairs_are_detached_and_shaped(self):
        from core.rf import generate_reflow_pairs, sample_uniform_noise

        teacher = self._teacher()
        z0, z1 = generate_reflow_pairs(
            teacher, 10, DIM, "cpu", sample_uniform_noise,
            ode_steps=2, batch_size=4,
        )
        self.assertEqual(z0.shape, (10, DIM))
        self.assertEqual(z1.shape, (10, DIM))
        # pairs are training data for the next round, never a gradient path
        self.assertFalse(z0.requires_grad or z1.requires_grad)
        self.assertFalse(torch.allclose(z0, z1), "the ODE must move the samples")


class TangentWarmupTests(unittest.TestCase):
    """AYF Algorithm 1 line 8: r scales only the (s-t) dF/dt term.

    So r=0 is pure flow matching -- drive the slope onto the teacher velocity --
    and r=1 is the full map-consistency tangent. That is what lets training
    start from a random student without the teacher-initialised weights the
    paper uses.
    """

    def test_zero_warmup_is_exactly_flow_matching(self):
        import core.ayf.losses as gaussian
        from core.ayf.losses import ayf_emd_loss, normalize_tangent

        torch.manual_seed(0)
        z = torch.randn(6, DIM)
        t = torch.full((6, 1), 0.3)
        s = torch.full((6, 1), 0.8)
        student = RFIntervalFlowMap(DIM, hidden=16, depth=2, num_freq=4, max_freq=4.0)
        teacher = _ConstantTeacher(torch.full((6, DIM), 2.0))

        with torch.no_grad():
            expected = normalize_tangent(
                student.slope(z, t, s) - torch.full((6, DIM), 2.0)
            ).pow(2).mean()

        with patch.object(gaussian, "sample_noise", return_value=z), \
                patch.object(gaussian, "sample_ayf_time_pair", return_value=(t, s)):
            loss = ayf_emd_loss(
                student, teacher, lambda b: z, 6, DIM, "cpu", tangent_warmup=0.0,
            )
        torch.testing.assert_close(loss, expected)

    def test_schedule_reaches_r_max_not_one(self):
        warmup_steps, warmup_max = 100, 0.99
        r = [min(warmup_max, (step - 1) / float(warmup_steps))
             for step in (1, 50, 100, 200)]
        self.assertEqual(r[0], 0.0)
        self.assertAlmostEqual(r[1], 0.49)
        self.assertEqual(r[-1], warmup_max)


class TangentNormalizationTests(unittest.TestCase):
    """AYF normalises the tangent by its own norm: df/dt / (||df/dt|| + 0.1)."""

    def test_normalises_per_sample_by_its_own_norm(self):
        tangent = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
        out = normalize_tangent(tangent)
        torch.testing.assert_close(out[0], torch.tensor([3.0, 4.0]) / 5.1)
        torch.testing.assert_close(out[1], torch.zeros(2))

    def test_denominator_carries_no_gradient(self):
        tangent = torch.tensor([[3.0, 4.0]], requires_grad=True)
        normalize_tangent(tangent).sum().backward()
        # d/dx [x / c] with c detached at 5.1
        torch.testing.assert_close(tangent.grad, torch.full((1, 2), 1.0 / 5.1))

    def test_squared_loss_saturates_for_large_tangents(self):
        small = normalize_tangent(torch.full((1, 1), 1e-4)).pow(2).mean()
        huge = normalize_tangent(torch.full((1, 1), 1e6)).pow(2).mean()
        self.assertLess(small.item(), 1e-5)
        self.assertAlmostEqual(huge.item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
