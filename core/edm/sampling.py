"""
EDM sampling (return latent tensors, no decoding):
    integrate_edm_teacher       – ODE from log_sigma_start to log(sigma_min)
    integrate_edm_teacher_interval – ODE between normalized EDM times
    sample_edm_teacher_latents  – full Karras ODE
    sample_edm_student_latents_one_step
    sample_edm_student_latents_k_step
    sample_edm_ayf_latents_one_step
    sample_edm_ayf_latents_k_step
"""

from __future__ import annotations

import math

import torch

from .models import EDMConsistencyModel, EDMIntervalFlowMap, EDMPrecondDenoiser
from .utils import (
    edm_dlog_sigma_dt,
    edm_karras_sigmas,
    edm_log_sigma_to_time,
    edm_time_to_log_sigma,
)


@torch.no_grad()
def integrate_edm_teacher(
    denoiser: EDMPrecondDenoiser,
    z: torch.Tensor,
    log_sigma_start: torch.Tensor,
    sigma_min: float,
    steps: int = 32,
) -> torch.Tensor:
    """
    Euler-integrate the EDM probability-flow ODE in log-sigma coordinates:
        dz/d(log_sigma) = z - D(z, sigma)

    Integrates from log_sigma_start down to log(sigma_min), then applies
    one final denoising step at sigma_min.
    """
    z = z.clone()
    log_sigma_end = torch.full_like(log_sigma_start, math.log(sigma_min))
    for i in range(steps):
        r = log_sigma_start + (log_sigma_end - log_sigma_start) * (i / steps)
        dr = (log_sigma_end - log_sigma_start) / steps
        den = denoiser(z, r)
        z = z + dr * (z - den)
    return denoiser(z, log_sigma_end)


def edm_normalized_time_velocity(
    denoiser: EDMPrecondDenoiser,
    z: torch.Tensor,
    time: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """EDM PF-ODE velocity in normalized noise-to-data time."""
    log_sigma = edm_time_to_log_sigma(time, sigma_min, sigma_max)
    return edm_dlog_sigma_dt(sigma_min, sigma_max) * (
        z - denoiser(z, log_sigma)
    )


@torch.no_grad()
def integrate_edm_teacher_interval(
    denoiser: EDMPrecondDenoiser,
    z: torch.Tensor,
    time_start: torch.Tensor,
    time_end: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    steps: int = 32,
) -> torch.Tensor:
    """Euler-integrate the EDM PF-ODE between normalized times."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    z = z.clone()
    for i in range(steps):
        time = time_start + (time_end - time_start) * (i / steps)
        dt = (time_end - time_start) / steps
        velocity = edm_normalized_time_velocity(
            denoiser, z, time, sigma_min, sigma_max
        )
        z = z + dt * velocity
    return z


@torch.no_grad()
def sample_edm_teacher_latents(
    denoiser: EDMPrecondDenoiser,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    sigma_min: float = 0.002,
    sigma_max: float = 2.0,
    rho: float = 7.0,
    steps: int = 64,
) -> torch.Tensor:
    """Full Karras ODE from sigma_max to sigma_min."""
    sigmas = edm_karras_sigmas(steps, sigma_min, sigma_max, rho, device)
    z = sigmas[0] * torch.randn(n, latent_dim, device=device)
    for i in range(steps - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        log_sigma = torch.full((n, 1), sigma.log().item(), device=device)
        den = denoiser(z, log_sigma)
        d = (z - den) / sigma
        z = z + (sigma_next - sigma) * d
    log_min = torch.full((n, 1), math.log(sigma_min), device=device)
    return denoiser(z, log_min)


@torch.no_grad()
def sample_edm_student_latents_one_step(
    student: EDMConsistencyModel,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    sigma_max: float = 2.0,
) -> torch.Tensor:
    """One-step consistency generation starting from sigma_max Gaussian noise."""
    z = sigma_max * torch.randn(n, latent_dim, device=device)
    log_sigma = torch.full((n, 1), math.log(sigma_max), device=device)
    return student(z, log_sigma)


@torch.no_grad()
def sample_edm_student_latents_k_step(
    student: EDMConsistencyModel,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    k: int = 4,
    sigma_min: float = 0.002,
    sigma_max: float = 2.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """
    Stochastic multi-step consistency sampling with ``k`` model evaluations.

    Each refinement draws fresh Gaussian noise with the variance required by
    the ``sigma_min`` boundary:

        z_clean = C(z_sigma, sigma)
        z_next  = z_clean + sqrt(sigma_next^2 - sigma_min^2) * eps
    """
    assert k >= 1
    sigmas = edm_karras_sigmas(k + 1, sigma_min, sigma_max, rho, device)
    z = sigmas[0] * torch.randn(n, latent_dim, device=device)
    z_clean = None
    for i, sigma in enumerate(sigmas[:-1]):
        log_sigma = torch.full((n, 1), sigma.log().item(), device=device)
        z_clean = student(z, log_sigma)
        if i + 1 < len(sigmas) - 1:
            sigma_next = sigmas[i + 1]
            noise_scale = torch.sqrt(
                (sigma_next.square() - sigma_min**2).clamp_min(0.0)
            )
            z = z_clean + noise_scale * torch.randn_like(z)
    return z_clean


@torch.no_grad()
def sample_edm_ayf_latents_one_step(
    student: EDMIntervalFlowMap,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    sigma_max: float = 2.0,
) -> torch.Tensor:
    """One EDM AYF interval from ``sigma_max`` to ``sigma_min``."""
    z = sigma_max * torch.randn(n, latent_dim, device=device)
    time_start = torch.zeros(n, 1, device=device)
    time_end = torch.ones(n, 1, device=device)
    return student(z, time_start, time_end)


@torch.no_grad()
def sample_edm_ayf_latents_k_step(
    student: EDMIntervalFlowMap,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    k: int = 4,
    sigma_min: float = 0.002,
    sigma_max: float = 2.0,
    rho: float = 7.0,
    schedule: str = "uniform",
) -> torch.Tensor:
    """
    Compose ``k`` deterministic AYF intervals.

    schedule: ``"uniform"`` spaces the nodes evenly in normalized time, which
    is what the RF arms' ``sample_student_latents_k_step`` does, so k-step
    numbers are comparable across arms. ``"karras"`` uses the EDM-native
    rho-schedule in sigma instead.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if schedule == "uniform":
        times = torch.linspace(0.0, 1.0, k + 1, device=device)
    elif schedule == "karras":
        sigmas = edm_karras_sigmas(k + 1, sigma_min, sigma_max, rho, device)
        times = edm_log_sigma_to_time(sigmas.log(), sigma_min, sigma_max)
    else:
        raise ValueError(f"Unknown schedule: {schedule!r}")
    z = sigma_max * torch.randn(n, latent_dim, device=device)
    for i in range(k):
        time_start = torch.full((n, 1), times[i].item(), device=device)
        time_end = torch.full((n, 1), times[i + 1].item(), device=device)
        z = student(z, time_start, time_end)
    return z
