"""
EDM losses:
    edm_teacher_loss
    edm_ayf_emd_loss
    edm_ayf_lmd_loss
    edm_ayf_interval_eval_loss  (no_grad)
    edm_endpoint_distillation_loss
    edm_jvp_scm_loss
    edm_endpoint_eval_loss  (no_grad)
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.func import jvp

from .models import EDMConsistencyModel, EDMIntervalFlowMap, EDMPrecondDenoiser
from .sampling import (
    edm_normalized_time_velocity,
    integrate_edm_teacher,
    integrate_edm_teacher_interval,
)
from .utils import edm_sample_log_sigma, edm_time_to_log_sigma


def edm_teacher_loss(
    denoiser: EDMPrecondDenoiser,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    device: str | torch.device,
    sigma_data: float = 0.5,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    sigma_min: float = 0.002,
    sigma_max: float = 2.0,
) -> torch.Tensor:
    """EDM weighted denoising score matching loss."""
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    log_sigma = edm_sample_log_sigma(batch_size, device, p_mean, p_std, sigma_min, sigma_max)
    sigma = log_sigma.exp()
    z_noisy = z1 + sigma * torch.randn_like(z1)
    pred = denoiser(z_noisy, log_sigma)
    weight = (sigma.pow(2) + sigma_data ** 2) / ((sigma * sigma_data) ** 2)
    per_sample = (pred - z1).pow(2).mean(dim=1, keepdim=True)
    return (weight * per_sample).mean()


def _sample_edm_ayf_state(
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    device: str | torch.device,
    sigma_min: float,
    sigma_max: float,
    time_eps: float,
    interval_p_mean: float,
    interval_p_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        z_clean = latent_sampler(batch_size)
    # AYF Algorithm 1: sample a logistic-normal interval distance, then place
    # an interval of that length uniformly on the trajectory. Our normalized
    # EDM time runs noise -> data, opposite to the paper's convention.
    interval_logits = interval_p_mean + interval_p_std * torch.randn(
        batch_size, 1, device=device
    )
    distance = torch.sigmoid(interval_logits).clamp(time_eps, 1.0 - time_eps)
    time = (1.0 - distance) * torch.rand(batch_size, 1, device=device)
    target_time = time + distance
    log_sigma = edm_time_to_log_sigma(time, sigma_min, sigma_max)
    z_noisy = z_clean + log_sigma.exp() * torch.randn_like(z_clean)
    return z_noisy, time, target_time


def edm_ayf_emd_loss(
    student: EDMIntervalFlowMap,
    teacher: EDMPrecondDenoiser,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    device: str | torch.device,
    sigma_min: float = 0.002,
    sigma_max: float = 2.0,
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
    jvp_dt_ref: float = 1.0,
) -> torch.Tensor:
    """
    AYF EMD loss for the EDM probability-flow ODE.

    Source-time JVP:  d_t C(z,t,s) + J_z C(z,t,s) v_phi(z,t) = 0

    The residual is scaled by the local velocity magnitude, matching
    ``core.ayf`` and ``core.ayf_rf``. A residual-norm normalisation is not
    usable here: EDM's normalized-time velocity spans roughly three orders of
    magnitude between sigma_max and sigma_min, so dividing by ``||r|| + c``
    saturates for nearly every sample and discards that scale entirely.
    """
    z_noisy, time, target_time = _sample_edm_ayf_state(
        latent_sampler, batch_size, device, sigma_min, sigma_max, time_eps,
        interval_p_mean, interval_p_std,
    )

    with torch.no_grad():
        velocity = edm_normalized_time_velocity(
            teacher, z_noisy, time, sigma_min, sigma_max
        )

    _, tangent = jvp(
        lambda z_in, time_in: student(z_in, time_in, target_time),
        primals=(z_noisy, time),
        tangents=(velocity, torch.ones_like(time)),
    )

    norm = torch.sqrt(1.0 + velocity.pow(2).mean(dim=-1, keepdim=True)).detach()
    residual = tangent / norm
    return (jvp_dt_ref ** 2) * residual.pow(2).mean()


def edm_ayf_lmd_loss(
    student: EDMIntervalFlowMap,
    teacher: EDMPrecondDenoiser,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    device: str | torch.device,
    sigma_min: float = 0.002,
    sigma_max: float = 2.0,
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
    jvp_dt_ref: float = 1.0,
) -> torch.Tensor:
    """
    AYF LMD loss for the EDM probability-flow ODE.

    Target-time condition:  d_s C(z,t,s) = v_phi(C(z,t,s), s)

    As in ``edm_ayf_emd_loss``, the gradient flows through the JVP itself and
    the residual is normalised by the local velocity magnitude.
    """
    z_noisy, time, target_time = _sample_edm_ayf_state(
        latent_sampler, batch_size, device, sigma_min, sigma_max, time_eps,
        interval_p_mean, interval_p_std,
    )

    z_target, derivative = jvp(
        lambda target_in: student(z_noisy, time, target_in),
        primals=(target_time,),
        tangents=(torch.ones_like(target_time),),
    )

    with torch.no_grad():
        velocity = edm_normalized_time_velocity(
            teacher, z_target.detach(), target_time, sigma_min, sigma_max
        )

    norm = torch.sqrt(1.0 + velocity.pow(2).mean(dim=-1, keepdim=True)).detach()
    residual = (derivative - velocity) / norm
    return (jvp_dt_ref ** 2) * residual.pow(2).mean()


@torch.no_grad()
def edm_ayf_interval_eval_loss(
    student: EDMIntervalFlowMap,
    teacher: EDMPrecondDenoiser,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    device: str | torch.device,
    sigma_min: float = 0.002,
    sigma_max: float = 2.0,
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
    ode_steps: int = 32,
) -> torch.Tensor:
    """Evaluate an EDM AYF interval against the teacher PF-ODE."""
    z_noisy, time, target_time = _sample_edm_ayf_state(
        latent_sampler, batch_size, device, sigma_min, sigma_max, time_eps,
        interval_p_mean, interval_p_std,
    )
    target = integrate_edm_teacher_interval(
        teacher, z_noisy, time, target_time,
        sigma_min, sigma_max, steps=ode_steps,
    )
    return F.mse_loss(student(z_noisy, time, target_time), target)


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
    log_sigma = edm_sample_log_sigma(batch_size, device, p_mean, p_std, sigma_min, sigma_max)
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
    log_sigma = edm_sample_log_sigma(batch_size, device, p_mean, p_std, sigma_min, sigma_max)
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
    log_sigma = edm_sample_log_sigma(batch_size, device, p_mean, p_std, sigma_min, sigma_max)
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
