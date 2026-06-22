"""
RF-sCM sampling: noise prior, teacher ODE integration, and student/teacher
generation (return latent tensors; decoding is the caller's responsibility).
"""

from __future__ import annotations

import torch

from .models import RFTeacher, RFConsistencyStudent


# ---------------------------------------------------------------------------
# Noise prior
# ---------------------------------------------------------------------------

def sample_noise(batch_size: int, latent_dim: int, device: str | torch.device) -> torch.Tensor:
    """z0 ~ Uniform([-1, 1]^latent_dim)."""
    return torch.empty(batch_size, latent_dim, device=device).uniform_(-1.0, 1.0)


# ---------------------------------------------------------------------------
# ODE integration
# ---------------------------------------------------------------------------

@torch.no_grad()
def integrate_teacher(
    teacher: RFTeacher,
    z: torch.Tensor,
    t_start: torch.Tensor,
    steps: int = 32,
) -> torch.Tensor:
    """
    Euler-integrate dz/dt = v_phi(z, t) from t_start to t=1 (data).

    Args:
        teacher:  trained velocity network.
        z:        [B, latent_dim] starting points.
        t_start:  [B, 1] starting times (per sample, allows mixing).
        steps:    number of Euler steps.
    Returns:
        [B, latent_dim] latents clamped to [-1, 1].
    """
    z = z.clone()
    t_start = t_start.clone()
    for i in range(steps):
        tau = t_start + (1.0 - t_start) * (i / steps)
        dt = (1.0 - t_start) / steps
        z = z + dt * teacher(z, tau)
    return z.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_teacher_latents(
    teacher: RFTeacher,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    steps: int = 64,
) -> torch.Tensor:
    """Full ODE integration from uniform noise to data latent."""
    z0 = sample_noise(n, latent_dim, device)
    t0 = torch.zeros(n, 1, device=device)
    return integrate_teacher(teacher, z0, t0, steps=steps)


@torch.no_grad()
def sample_student_latents_one_step(
    student: RFConsistencyStudent,
    n: int,
    latent_dim: int,
    device: str | torch.device,
) -> torch.Tensor:
    """One-step consistency generation."""
    z0 = sample_noise(n, latent_dim, device)
    t0 = torch.zeros(n, 1, device=device)
    return student(z0, t0)


@torch.no_grad()
def sample_student_latents_k_step(
    student: RFConsistencyStudent,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    k: int = 4,
    schedule: str = "uniform",
) -> torch.Tensor:
    """
    k-step consistency sampling.

    At each step i:
      z1_hat = student(z_t, t_i)
    If i+1 < k, re-interpolate:
      z_{t_{i+1}} = (1 - t_{i+1}) * z0 + t_{i+1} * z1_hat

    schedule: "uniform" | "quadratic" | "sqrt" | explicit list of times in [0, 1).
    """
    assert k >= 1
    z0 = sample_noise(n, latent_dim, device)

    if isinstance(schedule, str):
        if k == 1:
            times = torch.tensor([0.0], device=device)
        elif schedule == "uniform":
            times = torch.linspace(0.0, 1.0, k + 1, device=device)[:-1]
        elif schedule == "quadratic":
            times = torch.linspace(0.0, 1.0, k + 1, device=device)[:-1] ** 2
        elif schedule == "sqrt":
            times = torch.linspace(0.0, 1.0, k + 1, device=device)[:-1].sqrt()
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")
    else:
        times = torch.as_tensor(schedule, device=device, dtype=torch.float32)
        assert len(times) == k

    z_t = z0
    z1_hat = None
    for i, t_val in enumerate(times):
        t = torch.full((n, 1), float(t_val), device=device)
        z1_hat = student(z_t, t)
        if i + 1 < len(times):
            t_next = float(times[i + 1])
            z_t = (1.0 - t_next) * z0 + t_next * z1_hat

    return z1_hat
