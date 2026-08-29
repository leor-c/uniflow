import math
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import core.ayf.losses as ayf_gaussian
from core.ayf.losses import ayf_emd_loss
from core.ayf.training import train_student_ayf
from core.ayf_rf.models import RFIntervalFlowMap
from core.edm.flow import EDMFlowVelocity
from core.edm.models import EDMConsistencyModel, EDMPrecondDenoiser
from core.edm.sampling import sample_edm_student_latents_k_step
from core.edm.utils import edm_karras_sigmas, edm_sample_log_sigma
from toy.edm_baseline import Config as EDMConfig, main as edm_main
from toy.gaussian_rf import Config as GaussianRFConfig, main as gaussian_rf_main


class _RecordingStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.log_sigmas = []

    def forward(self, z, log_sigma):
        self.inputs.append(z.clone())
        self.log_sigmas.append(log_sigma.clone())
        return torch.zeros_like(z)


class _ConstantSlopeStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def slope(self, z, time_start, time_end):
        return self.value.expand_as(z)

    def forward(self, z, time_start, time_end):
        return z + (time_end - time_start) * self.slope(z, time_start, time_end)


class _ZeroResidualDenoiser(EDMPrecondDenoiser):
    """Denoiser whose network residual vanishes, as a trained one does at
    extreme noise levels. Makes the flow-velocity endpoints exact."""

    def network_residual(self, scaled_input, log_sigma):
        return torch.zeros_like(scaled_input)


class _EncoderConfigured(Exception):
    """Stops a benchmark after its autoencoder constructor is captured."""


class EDMTeacherTests(unittest.TestCase):
    def test_teacher_log_noise_distribution_is_untruncated(self):
        draws = torch.tensor([[-10.0], [10.0]])
        with patch("core.edm.utils.torch.randn", return_value=draws):
            log_sigma = edm_sample_log_sigma(
                batch_size=2, device="cpu", p_mean=-1.2, p_std=1.2,
            )

        torch.testing.assert_close(log_sigma, -1.2 + 1.2 * draws)

    def test_unit_latent_edm_config_exactly_rescales_published_noise_levels(self):
        cfg = EDMConfig(device="cpu")

        self.assertEqual(cfg.sigma_data, 1.0)
        self.assertEqual(cfg.sigma_min, 2.0 * 0.002)
        self.assertEqual(cfg.sigma_max, 2.0 * 80.0)
        self.assertAlmostEqual(cfg.edm_p_mean, -1.2 + math.log(2.0))
        self.assertEqual(cfg.edm_p_std, 1.2)

    def test_edm_encoder_matches_gaussian_rf_encoder(self):
        constructor_args = []

        def capture_encoder(**kwargs):
            constructor_args.append(kwargs)
            raise _EncoderConfigured

        benchmarks = (
            ("toy.gaussian_rf.ToyAE", gaussian_rf_main, GaussianRFConfig(device="cpu")),
            ("toy.edm_baseline.ToyAE", edm_main, EDMConfig(device="cpu")),
        )
        for target, benchmark_main, config in benchmarks:
            with patch(target, side_effect=capture_encoder), patch(
                    "builtins.print",
            ), self.assertRaises(_EncoderConfigured):
                benchmark_main(config)

        self.assertEqual(constructor_args[0], constructor_args[1])
        self.assertFalse(constructor_args[1]["bounded"])

    def test_edm_arm_shares_the_gaussian_rf_base_distribution(self):
        """Under the flow-matching view the base law is the RF arm's, and its
        scale must match sigma_data because the latents carry unit scale."""
        cfg = EDMConfig(device="cpu")
        self.assertEqual(cfg.base_std, GaussianRFConfig(device="cpu").base_std)
        self.assertEqual(cfg.base_std, cfg.sigma_data)


class EDMFlowVelocityTests(unittest.TestCase):
    """The AYF arm distils the EDM teacher through the flow-matching change of
    variables sigma = (1 - t) / t, not through a preconditioned sigma-space
    state. See core/edm/flow.py."""

    def setUp(self):
        torch.manual_seed(0)
        self.denoiser = EDMPrecondDenoiser(
            latent_dim=4, hidden=16, depth=2, sigma_data=1.0,
        )
        self.z = torch.randn(6, 4)

    def test_matches_the_direct_definition_in_the_interior(self):
        # A wide clamp makes the conditioning noise level exact.
        flow = EDMFlowVelocity(self.denoiser, sigma_min=1e-6, sigma_max=1e6)
        for t_value in (0.05, 0.2, 0.5, 0.8, 0.95):
            t = torch.full((6, 1), t_value)
            direct = (self.denoiser(self.z / t, ((1 - t) / t).log()) - self.z) / (1 - t)
            with self.subTest(t=t_value):
                torch.testing.assert_close(flow(self.z, t), direct, atol=1e-5, rtol=0.0)

    def test_endpoints_are_finite_where_the_direct_form_is_not(self):
        flow = EDMFlowVelocity(self.denoiser, sigma_min=0.004, sigma_max=160.0)
        for t_value in (0.0, 1.0):
            t = torch.full((6, 1), t_value)
            direct = (self.denoiser(self.z / t, ((1 - t) / t).log()) - self.z) / (1 - t)
            with self.subTest(t=t_value):
                self.assertFalse(bool(torch.isfinite(direct).all()))
                self.assertTrue(bool(torch.isfinite(flow(self.z, t)).all()))

    def test_endpoint_limits_match_the_linear_interpolant(self):
        """With a vanishing residual, v(z, 0) = -z_0 and v(z, 1) = z_1."""
        flow = EDMFlowVelocity(
            _ZeroResidualDenoiser(latent_dim=4, hidden=16, depth=2, sigma_data=1.0),
            sigma_min=0.004, sigma_max=160.0,
        )
        torch.testing.assert_close(flow(self.z, torch.zeros(6, 1)), -self.z)
        torch.testing.assert_close(flow(self.z, torch.ones(6, 1)), self.z)

    def test_velocity_magnitude_stays_bounded_across_the_interval(self):
        """The old preconditioned state collapsed to O((sigma_data/sigma)^2) at
        the noise end, drowning the distillation target in teacher error."""
        flow = EDMFlowVelocity(self.denoiser, sigma_min=0.004, sigma_max=160.0)
        magnitudes = [
            flow(self.z, torch.full((6, 1), t)).norm(dim=-1).mean().item()
            for t in (0.0, 0.01, 0.1, 0.5, 0.9, 1.0)
        ]
        self.assertGreater(min(magnitudes), 0.1)
        self.assertLess(max(magnitudes) / min(magnitudes), 20.0)

    def test_rejects_an_invalid_sigma_range(self):
        with self.assertRaises(ValueError):
            EDMFlowVelocity(self.denoiser, sigma_min=1.0, sigma_max=1.0)


class EDMAYFStudentTests(unittest.TestCase):
    def test_student_is_the_shared_rf_interval_flow_map(self):
        student = RFIntervalFlowMap(latent_dim=4, hidden=16, depth=2,
                                    num_freq=4, max_freq=4.0)
        z = torch.randn(5, 4)
        time = torch.rand(5, 1)
        torch.testing.assert_close(student(z, time, time), z, atol=0.0, rtol=0.0)

    def test_emd_loss_through_the_flow_wrapper_is_finite_and_differentiable(self):
        torch.manual_seed(0)
        teacher = EDMFlowVelocity(
            EDMPrecondDenoiser(latent_dim=4, hidden=16, depth=2, sigma_data=1.0),
            sigma_min=0.004, sigma_max=160.0,
        )
        student = RFIntervalFlowMap(4, hidden=16, depth=2, num_freq=4, max_freq=4.0)

        loss = ayf_emd_loss(
            student, teacher, lambda b: torch.randn(b, 4),
            batch_size=3, latent_dim=4, device="cpu", time_eps=1e-4,
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [p.grad for p in student.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_emd_uses_ayf_tangent_normalisation(self):
        """
        AYF normalises the tangent by its own norm, df/dt / (||df/dt|| + 0.1),
        rather than by the teacher-velocity scale.

        With C(z, t, s) = z + (s - t) a and a constant slope a, the source-time
        JVP along (v, 1) is ``v - a``. For a one-dimensional latent and v = 2 at
        a = 0 the tangent norm is 2, so

            loss = ((v - a) / 2.1)^2   and   dloss/da = -2 * (2 / 2.1) / 2.1.
        """
        student = _ConstantSlopeStudent()
        z = torch.zeros(2, 1)
        t = torch.full((2, 1), 0.2)
        s = torch.full((2, 1), 0.8)

        with patch.object(ayf_gaussian, "sample_noise", return_value=z), \
                patch.object(
                    ayf_gaussian, "sample_ayf_time_pair", return_value=(t, s),
                ):
            loss = ayf_emd_loss(
                student,
                lambda z_in, t_in: torch.full_like(z_in, 2.0),
                lambda b: torch.zeros(b, 1),
                batch_size=2, latent_dim=1, device="cpu",
            )

        loss.backward()
        torch.testing.assert_close(
            student.value.grad, torch.tensor(-2.0 * (2.0 / 2.1) / 2.1),
        )

    def test_student_training_reads_the_shared_ayf_config_fields(self):
        cfg = EDMConfig(device="cpu")
        cfg.student_steps = 1
        cfg.log_every = 10 ** 9
        recorded = {}

        def fake_loss(*args, **kwargs):
            recorded.update(kwargs)
            return torch.zeros((), requires_grad=True)

        student = RFIntervalFlowMap(2, hidden=8, depth=2)
        teacher = EDMFlowVelocity(
            EDMPrecondDenoiser(latent_dim=2, hidden=8, depth=2, sigma_data=1.0),
            sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max,
        )
        with patch("core.ayf.training.ayf_emd_loss", side_effect=fake_loss):
            train_student_ayf(
                student, teacher, lambda b: torch.zeros(b, 2), cfg,
            )

        self.assertEqual(recorded["base_std"], cfg.base_std)
        self.assertEqual(recorded["interval_p_mean"], cfg.interval_p_mean)
        self.assertEqual(recorded["interval_p_std"], cfg.interval_p_std)


class EDMConsistencyReferenceTests(unittest.TestCase):
    """The endpoint-consistency student is kept as a reference arm."""

    def test_reference_consistency_student_is_identity_at_sigma_min(self):
        sigma_min = 0.002
        student = EDMConsistencyModel(
            latent_dim=4, hidden=8, depth=2, sigma_min=sigma_min,
            num_freq=4, max_freq=4.0,
        )
        z = torch.randn(3, 4)
        log_sigma = torch.full((3, 1), math.log(sigma_min))

        torch.testing.assert_close(student(z, log_sigma), z, atol=1e-6, rtol=0.0)

    def test_multistep_sampler_uses_fresh_variance_corrected_noise(self):
        student = _RecordingStudent()
        n, latent_dim, k = 2, 3, 3
        sigma_min, sigma_max, rho = 0.1, 2.0, 7.0
        initial_noise = torch.ones(n, latent_dim)
        fresh_noises = [
            torch.full((n, latent_dim), 2.0),
            torch.full((n, latent_dim), 3.0),
        ]

        with patch("core.edm.sampling.torch.randn", return_value=initial_noise), patch(
                "core.edm.sampling.torch.randn_like", side_effect=fresh_noises,
        ) as fresh_noise:
            output = sample_edm_student_latents_k_step(
                student, n=n, latent_dim=latent_dim, device="cpu", k=k,
                sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
            )

        sigmas = edm_karras_sigmas(k + 1, sigma_min, sigma_max, rho, "cpu")
        scale_1 = torch.sqrt(sigmas[1].square() - sigma_min ** 2)
        scale_2 = torch.sqrt(sigmas[2].square() - sigma_min ** 2)

        self.assertEqual(len(student.inputs), k)
        self.assertEqual(fresh_noise.call_count, k - 1)
        torch.testing.assert_close(student.inputs[0], sigmas[0] * initial_noise)
        torch.testing.assert_close(student.inputs[1], scale_1 * fresh_noises[0])
        torch.testing.assert_close(student.inputs[2], scale_2 * fresh_noises[1])
        torch.testing.assert_close(output, torch.zeros_like(output))


if __name__ == "__main__":
    unittest.main()
