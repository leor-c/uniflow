"""
EDM losses:
    edm_teacher_loss
    edm_endpoint_distillation_loss
    edm_jvp_scm_loss
    edm_endpoint_eval_loss  (no_grad)

AYF distillation lives in ``core.ayf``, driven by ``core.edm.EDMFlowVelocity``.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.func import jvp

from .models import EDMConsistencyModel, EDMPrecondDenoiser
from .sampling import integrate_edm_teacher
from .utils import edm_sample_log_sigma


def edm_teacher_loss(
        denoiser: EDMPrecondDenoiser,
        latent_sampler: Callable[[int], torch.Tensor],
        batch_size: int,
        device: str | torch.device,
        sigma_data: float = 0.5,
        p_mean: float = -1.2,
        p_std: float = 1.2,
) -> torch.Tensor:
    """EDM weighted denoising score matching loss."""
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    log_sigma = edm_sample_log_sigma(batch_size, device, p_mean, p_std)
    sigma = log_sigma.exp()
    z_noisy = z1 + sigma * torch.randn_like(z1)
    pred = denoiser(z_noisy, log_sigma)
    weight = (sigma.pow(2) + sigma_data ** 2) / ((sigma * sigma_data) ** 2)
    per_sample = (pred - z1).pow(2).mean(dim=1, keepdim=True)
    return (weight * per_sample).mean()


def edm_endpoint_distillation_loss(
        student: EDMConsistencyModel,
        teacher: EDMPrecondDenoiser,
        latent_sampler: Callable[[int], torch.Tensor],
        batch_size: int,
        device: str | torch.device,
        sigma_min: float = 0.002,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_max: float = 2.0,
        ode_steps: int = 32,
) -> torch.Tensor:
    """Flow-map endpoint distillation loss for EDM student."""
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    log_sigma = edm_sample_log_sigma(batch_size, device, p_mean, p_std)
    sigma = log_sigma.exp()
    z_noisy = z1 + sigma * torch.randn_like(z1)
    with torch.no_grad():
        target = integrate_edm_teacher(teacher, z_noisy, log_sigma, sigma_min, ode_steps)
    pred = student(z_noisy, log_sigma)
    return F.mse_loss(pred, target)


@torch.no_grad()
def edm_endpoint_eval_loss(
        student: EDMConsistencyModel,
        teacher: EDMPrecondDenoiser,
        latent_sampler: Callable[[int], torch.Tensor],
        batch_size: int,
        device: str | torch.device,
        sigma_min: float = 0.002,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_max: float = 2.0,
        ode_steps: int = 32,
) -> torch.Tensor:
    """Eval-only EDM endpoint loss."""
    z1 = latent_sampler(batch_size)
    log_sigma = edm_sample_log_sigma(batch_size, device, p_mean, p_std)
    sigma = log_sigma.exp()
    z_noisy = z1 + sigma * torch.randn_like(z1)
    target = integrate_edm_teacher(teacher, z_noisy, log_sigma, sigma_min, ode_steps)
    pred = student(z_noisy, log_sigma)
    return F.mse_loss(pred, target)


def edm_jvp_scm_loss(
        student: EDMConsistencyModel,
        teacher: EDMPrecondDenoiser,
        latent_sampler: Callable[[int], torch.Tensor],
        batch_size: int,
        device: str | torch.device,
        sigma_min: float = 0.002,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_max: float = 2.0,
        jvp_dlog_ref: float = 1.0,
) -> torch.Tensor:
    """
    Pure JVP consistency loss in log-sigma coordinates.

    Along the EDM PF-ODE: dz/d(log_sigma) = z - D_teacher(z, sigma).
    The consistency map C should satisfy d/d(log_sigma) C(z, log_sigma) = 0.
    """
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    log_sigma = edm_sample_log_sigma(batch_size, device, p_mean, p_std)
    sigma = log_sigma.exp()
    z_noisy = z1 + sigma * torch.randn_like(z1)
    with torch.no_grad():
        den = teacher(z_noisy, log_sigma)
        tangent_z = z_noisy - den

    _, tangent = jvp(
        lambda z_in, ls_in: student(z_in, ls_in),
        primals=(z_noisy, log_sigma),
        tangents=(tangent_z, torch.ones_like(log_sigma)),
    )
    norm = torch.sqrt(1.0 + tangent_z.pow(2).mean(dim=1, keepdim=True)).detach()
    residual = tangent / norm
    return (jvp_dlog_ref ** 2) * residual.pow(2).mean()
