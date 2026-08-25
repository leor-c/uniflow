"""
Gaussian-prior AYF sampling utilities (return latent tensors, no decoding).

Identical to core.ayf_rf.sampling except the noise prior:
    z0 ~ N(0, sigma_data² I)   instead of Uniform[-1, 1]^d

    sample_noise                    – z0 ~ N(0, sigma_data² I)
    sample_ayf_time_pair        – re-exported from core.ayf_rf (unchanged)
    integrate_teacher_interval      – re-exported from core.ayf_rf (unchanged)
    integrate_teacher_to_data       – re-exported from core.ayf_rf (unchanged)
    sample_teacher_latents          – full ODE from Gaussian z0
    sample_student_latents_one_step – C(z0, 0, 1) with Gaussian z0
    sample_student_latents_k_step   – interval stepping with Gaussian z0
"""

from __future__ import annotations

import torch

from ..rf.models import RFTeacher
from ..ayf_rf.models import RFIntervalFlowMap
from ..ayf_rf.sampling import (
    sample_ayf_time_pair,
    integrate_teacher_interval,
    integrate_teacher_to_data,
)


def sample_noise(
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """z0 ~ N(0, sigma_data² I)  [B, D]."""
    return sigma_data * torch.randn(batch_size, latent_dim, device=device)


@torch.no_grad()
def sample_teacher_latents(
    teacher: RFTeacher,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    sigma_data: float = 0.5,
    steps: int = 64,
) -> torch.Tensor:
    """Full Euler ODE from z0 ~ N(0, sigma_data² I) to z1."""
    z = sample_noise(n, latent_dim, device, sigma_data)
    t_start = torch.zeros(n, 1, device=device)
    return integrate_teacher_to_data(teacher, z, t_start, steps=steps)


@torch.no_grad()
def sample_student_latents_one_step(
    student: RFIntervalFlowMap,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """One-step: C(z0, 0, 1) with z0 ~ N(0, sigma_data² I)."""
    z0 = sample_noise(n, latent_dim, device, sigma_data)
    t0 = torch.zeros(n, 1, device=device)
    s1 = torch.ones(n, 1, device=device)
    return student(z0, t0, s1)


@torch.no_grad()
def sample_student_latents_k_step(
    student: RFIntervalFlowMap,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    k: int = 4,
    sigma_data: float = 0.5,
    schedule: str = "uniform",
) -> torch.Tensor:
    """
    K-step AYF sampling with Gaussian noise prior:
        z_{t_{i+1}} = C_θ(z_{t_i}, t_i, t_{i+1})

    schedule: "uniform" | "quadratic" | "sqrt"
    """
    assert k >= 1
    z = sample_noise(n, latent_dim, device, sigma_data)

    base = torch.linspace(0.0, 1.0, k + 1, device=device)
    if schedule == "uniform":
        times = base
    elif schedule == "quadratic":
        times = base ** 2
    elif schedule == "sqrt":
        times = base.sqrt()
    else:
        raise ValueError(f"Unknown schedule: {schedule!r}")

    for i in range(k):
        t = torch.full((n, 1), float(times[i]), device=device)
        s = torch.full((n, 1), float(times[i + 1]), device=device)
        z = student(z, t, s)

    return z
