"""Noise and ODE utilities shared by both RF benchmark arms."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .models import RFTeacher

NoiseSampler = Callable[[int, int, str | torch.device], torch.Tensor]


def sample_uniform_noise(
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Sample ``U([-1, 1]^d)``."""
    return torch.empty(batch_size, latent_dim, device=device).uniform_(-1.0, 1.0)


def sample_gaussian_noise(
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Sample ``N(0, sigma^2 I)``."""
    return sigma * torch.randn(batch_size, latent_dim, device=device)


@torch.no_grad()
def integrate_teacher_interval(
    teacher: RFTeacher,
    z: torch.Tensor,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    steps: int = 32,
) -> torch.Tensor:
    """Euler-integrate the teacher ODE over a per-sample time interval."""
    z = z.clone()
    for i in range(steps):
        tau = t_start + (t_end - t_start) * (i / steps)
        dt = (t_end - t_start) / steps
        z = z + dt * teacher(z, tau)
    return z


@torch.no_grad()
def integrate_teacher_to_data(
    teacher: RFTeacher,
    z: torch.Tensor,
    t_start: torch.Tensor,
    steps: int = 64,
) -> torch.Tensor:
    """Euler-integrate from ``t_start`` to the data endpoint ``t=1``."""
    return integrate_teacher_interval(
        teacher, z, t_start, torch.ones_like(t_start), steps=steps
    )


# Kept as a concise alias for callers that only integrate to the endpoint.
integrate_teacher = integrate_teacher_to_data


@torch.no_grad()
def sample_teacher_latents(
    teacher: RFTeacher,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    noise_sampler: NoiseSampler = sample_uniform_noise,
    steps: int = 64,
) -> torch.Tensor:
    """Draw base noise and integrate a trained RF teacher to ``t=1``."""
    z0 = noise_sampler(n, latent_dim, device)
    t0 = torch.zeros(n, 1, device=device)
    return integrate_teacher_to_data(teacher, z0, t0, steps=steps)
