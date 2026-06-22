"""
EDM sampling (return latent tensors, no decoding):
    integrate_edm_teacher       – ODE from log_sigma_start to log(sigma_min)
    sample_edm_teacher_latents  – full Karras ODE
    sample_edm_student_latents_one_step
    sample_edm_student_latents_k_step
"""

from __future__ import annotations

import math

import torch

from .models import EDMPrecondDenoiser
from .utils import edm_karras_sigmas


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
    student: EDMPrecondDenoiser,
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
    student: EDMPrecondDenoiser,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    k: int = 4,
    sigma_min: float = 0.002,
    sigma_max: float = 2.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """
    Multi-step EDM consistency sampling.

    Keeps the initial Gaussian direction eps fixed:
        z_clean = C(z_sigma, sigma)
        z_next  = z_clean + sigma_next * eps
    """
    assert k >= 1
    eps = torch.randn(n, latent_dim, device=device)
    sigmas = edm_karras_sigmas(k, sigma_min, sigma_max, rho, device)
    z = sigmas[0] * eps
    z_clean = None
    for i, sigma in enumerate(sigmas):
        log_sigma = torch.full((n, 1), sigma.log().item(), device=device)
        z_clean = student(z, log_sigma)
        if i + 1 < len(sigmas):
            z = z_clean + sigmas[i + 1] * eps
    return z_clean
