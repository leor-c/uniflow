import math
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from core.edm.losses import _sample_edm_ayf_state, edm_ayf_emd_loss
from core.edm.training import train_edm_student_ayf
from core.edm.models import (
    EDMConsistencyModel,
    EDMIntervalFlowMap,
    EDMPrecondDenoiser,
)
from core.edm.sampling import (
    sample_edm_ayf_latents_k_step,
    sample_edm_student_latents_k_step,
)
from core.edm.utils import (
    edm_karras_sigmas,
    edm_log_sigma_to_time,
    edm_time_to_log_sigma,
)
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


class _RecordingIntervalStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.intervals = []

    def forward(self, z, time_start, time_end):
        self.intervals.append((time_start.clone(), time_end.clone()))
        return z


class _ConstantSlopeStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def slope(self, z, time_start, time_end):
        return self.value.expand_as(z)

    def forward(self, z, time_start, time_end):
        return z + (time_end - time_start) * self.slope(z, time_start, time_end)


class _EncoderConfigured(Exception):
    """Stops a benchmark after its autoencoder constructor is captured."""


class EDMTests(unittest.TestCase):
    def test_normalized_time_log_sigma_round_trip(self):
        times = torch.tensor([[0.0], [0.3], [1.0]])
        log_sigma = edm_time_to_log_sigma(times, sigma_min=0.01, sigma_max=12.0)
        restored = edm_log_sigma_to_time(log_sigma, sigma_min=0.01, sigma_max=12.0)

        torch.testing.assert_close(log_sigma[0], torch.tensor([math.log(12.0)]))
        torch.testing.assert_close(log_sigma[-1], torch.tensor([math.log(0.01)]))
        torch.testing.assert_close(restored, times)

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
                "builtins.print"
            ), self.assertRaises(_EncoderConfigured):
                benchmark_main(config)

        self.assertEqual(constructor_args[0], constructor_args[1])
        self.assertFalse(constructor_args[1]["bounded"])

    def test_reference_consistency_student_is_identity_at_sigma_min(self):
        sigma_min = 0.002
        student = EDMConsistencyModel(
            latent_dim=4,
            hidden=8,
            depth=2,
            sigma_min=sigma_min,
            num_freq=4,
            max_freq=4.0,
        )
        z = torch.randn(3, 4)
        log_sigma = torch.full((3, 1), math.log(sigma_min))

        torch.testing.assert_close(student(z, log_sigma), z, atol=1e-6, rtol=0.0)

    def test_ayf_student_is_identity_on_zero_length_interval(self):
        student = EDMIntervalFlowMap(
            latent_dim=4,
            hidden=16,
            depth=2,
            num_freq=4,
            max_freq=4.0,
        )
        z = torch.randn(5, 4)
        time = torch.rand(5, 1)

        torch.testing.assert_close(student(z, time, time), z, atol=0.0, rtol=0.0)

    def test_ayf_emd_loss_is_finite_and_differentiable(self):
        teacher = EDMPrecondDenoiser(
            latent_dim=4,
            hidden=16,
            depth=2,
            num_freq=4,
            max_freq=4.0,
        )
        student = EDMIntervalFlowMap(
            latent_dim=4,
            hidden=16,
            depth=2,
            num_freq=4,
            max_freq=4.0,
        )
        loss = edm_ayf_emd_loss(
            student,
            teacher,
            lambda batch_size: torch.randn(batch_size, 4),
            batch_size=3,
            device="cpu",
            sigma_min=0.01,
            sigma_max=12.0,
            time_eps=1e-4,
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in student.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_ayf_emd_normalises_by_velocity_scale(self):
        """
        The residual must be scaled by the velocity magnitude, not driven to
        unit norm. With C(z, t, s) = z + (s - t) a and a constant slope a, the
        source-time JVP along (v, 1) is ``v - a``, so

            loss = ((v - a) / sqrt(1 + mean(v^2)))^2

        and dloss/da at a = 0 is -2v / (1 + mean(v^2)) = -4 / 5 for v = 2.

        A residual-norm normalisation (``r / (||r|| + c)``) would instead give
        -4 / 2.1 here, and would saturate to a constant for the large residuals
        the EDM velocity actually produces.
        """
        student = _ConstantSlopeStudent()
        z_noisy = torch.zeros(2, 1)
        source_time = torch.full((2, 1), 0.2)
        target_time = torch.full((2, 1), 0.8)

        with patch(
            "core.edm.losses._sample_edm_ayf_state",
            return_value=(z_noisy, source_time, target_time),
        ), patch(
            "core.edm.losses.edm_normalized_time_velocity",
            return_value=torch.full_like(z_noisy, 2.0),
        ):
            loss = edm_ayf_emd_loss(
                student,
                nn.Identity(),
                lambda batch_size: torch.zeros(batch_size, 1),
                batch_size=2,
                device="cpu",
            )

        loss.backward()
        expected_gradient = torch.tensor(-4.0 / 5.0)
        torch.testing.assert_close(student.value.grad, expected_gradient)

    def test_ayf_student_trains_on_its_own_sigma_floor(self):
        """
        The student parameterises normalized time over
        [edm_student_sigma_min, sigma_max]; extending it down to the teacher's
        sigma_min would spend most of t in [0, 1] where the PF-ODE is already
        the identity.
        """
        cfg = EDMConfig(device="cpu")
        cfg.edm_student_steps = 1
        cfg.log_every = 10**9
        recorded = {}

        def fake_loss(*args, **kwargs):
            recorded.update(kwargs)
            return torch.zeros((), requires_grad=True)

        student = EDMIntervalFlowMap(latent_dim=2, hidden=8, depth=2)
        with patch("core.edm.training.edm_ayf_emd_loss", side_effect=fake_loss):
            train_edm_student_ayf(
                student,
                EDMPrecondDenoiser(latent_dim=2, hidden=8, depth=2),
                lambda batch_size: torch.zeros(batch_size, 2),
                cfg,
            )

        self.assertEqual(recorded["sigma_min"], cfg.edm_student_sigma_min)
        self.assertNotEqual(cfg.edm_student_sigma_min, cfg.sigma_min)
        self.assertEqual(recorded["sigma_max"], cfg.sigma_max)

    def test_ayf_state_uses_logistic_normal_interval_schedule(self):
        source_fraction = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
        time_eps = 1e-4

        with patch(
            "core.ayf_rf.sampling.torch.randn",
            return_value=torch.zeros(5, 1),
        ), patch(
            "core.ayf_rf.sampling.torch.rand",
            return_value=source_fraction,
        ), patch(
            "core.edm.losses.torch.randn_like",
            return_value=torch.ones(5, 2),
        ), patch(
            "core.edm.losses.edm_sample_log_sigma",
            side_effect=AssertionError("AYF must not use EDM's log-normal sampler"),
        ):
            _, source_time, target_time = _sample_edm_ayf_state(
                lambda batch_size: torch.zeros(batch_size, 2),
                batch_size=5,
                device="cpu",
                sigma_min=0.002,
                sigma_max=2.0,
                time_eps=time_eps,
                interval_p_mean=0.0,
                interval_p_std=1.0,
            )

        expected_source = 0.5 * source_fraction
        expected_target = expected_source + 0.5
        torch.testing.assert_close(source_time, expected_source)
        torch.testing.assert_close(target_time, expected_target)

    def test_ayf_k_step_sampler_composes_deterministic_karras_intervals(self):
        student = _RecordingIntervalStudent()
        initial_noise = torch.ones(3, 2)

        with patch(
            "core.edm.sampling.torch.randn", return_value=initial_noise
        ) as initial:
            with patch("core.edm.sampling.torch.randn_like") as fresh_noise:
                output = sample_edm_ayf_latents_k_step(
                    student,
                    n=3,
                    latent_dim=2,
                    device="cpu",
                    k=4,
                    sigma_min=0.01,
                    sigma_max=12.0,
                    rho=7.0,
                    schedule="karras",
                )

        sigmas = edm_karras_sigmas(5, 0.01, 12.0, 7.0, "cpu")
        times = edm_log_sigma_to_time(sigmas.log(), 0.01, 12.0)
        self.assertEqual(len(student.intervals), 4)
        for index, (time_start, time_end) in enumerate(student.intervals):
            torch.testing.assert_close(time_start, times[index].expand(3, 1))
            torch.testing.assert_close(time_end, times[index + 1].expand(3, 1))
        torch.testing.assert_close(output, 12.0 * initial_noise)
        initial.assert_called_once()
        fresh_noise.assert_not_called()

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
            "core.edm.sampling.torch.randn_like", side_effect=fresh_noises
        ) as fresh_noise:
            output = sample_edm_student_latents_k_step(
                student,
                n=n,
                latent_dim=latent_dim,
                device="cpu",
                k=k,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
            )

        sigmas = edm_karras_sigmas(k + 1, sigma_min, sigma_max, rho, "cpu")
        scale_1 = torch.sqrt(sigmas[1].square() - sigma_min**2)
        scale_2 = torch.sqrt(sigmas[2].square() - sigma_min**2)

        self.assertEqual(len(student.inputs), k)
        self.assertEqual(fresh_noise.call_count, k - 1)
        torch.testing.assert_close(student.inputs[0], sigmas[0] * initial_noise)
        torch.testing.assert_close(student.inputs[1], scale_1 * fresh_noises[0])
        torch.testing.assert_close(student.inputs[2], scale_2 * fresh_noises[1])
        torch.testing.assert_close(output, torch.zeros_like(output))


if __name__ == "__main__":
    unittest.main()
