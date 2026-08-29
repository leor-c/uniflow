"""
AYF losses — identical to core.ayf_rf.losses except the noise prior:
    z0 ~ N(0, base_std² I)   instead of Uniform[-1, 1]^d

The teacher is the same RFTeacher (velocity field), trained on Gaussian paths.
All JVP objectives carry over unchanged.

    ayf_emd_loss            – source-time JVP with Gaussian noise
    ayf_lmd_loss            – target-time JVP with Gaussian noise
    ayf_endpoint_eval_loss  – (no_grad) C(z,t,1) vs teacher ODE
    ayf_interval_eval_loss  – (no_grad) C(z,t,s) vs teacher ODE
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.func import jvp

from ..rf.models import RFTeacher
from ..ayf_rf.models import RFIntervalFlowMap
from ..ayf_rf.losses import normalize_tangent
from ..ayf_rf.sampling import sample_ayf_time_pair
from ..ayf_rf.sampling import integrate_teacher_interval, integrate_teacher_to_data
from .sampling import sample_noise


def ayf_emd_loss(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    base_std: float = 1.0,
    jvp_dt_ref: float = 1.0,
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
) -> torch.Tensor:
    """
    AYF EMD loss with Gaussian noise ``z0 ~ N(0, base_std² I)``.

    Source-time JVP:  ∂_t C(z,t,s) + J_z C(z,t,s) v_φ(z,t) = 0
    """
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    z0 = sample_noise(batch_size, latent_dim, device, base_std)
    t, s = sample_ayf_time_pair(
        batch_size, device, eps=time_eps,
        p_mean=interval_p_mean, p_std=interval_p_std,
    )
    zt = (1.0 - t) * z0 + t * z1

    with torch.no_grad():
        v = teacher(zt, t)

    _, tangent = jvp(
        lambda z_in, t_in: student(z_in, t_in, s),
        primals=(zt, t),
        tangents=(v, torch.ones_like(t)),
    )

    residual = normalize_tangent(tangent)
    return (jvp_dt_ref ** 2) * residual.pow(2).mean()


def ayf_lmd_loss(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    base_std: float = 1.0,
    jvp_dt_ref: float = 1.0,
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
) -> torch.Tensor:
    """
    AYF LMD loss with Gaussian noise.

    Target-time condition:  ∂_s C(z,t,s) = v_φ(C(z,t,s), s)
    """
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    z0 = sample_noise(batch_size, latent_dim, device, base_std)
    t, s = sample_ayf_time_pair(
        batch_size, device, eps=time_eps,
        p_mean=interval_p_mean, p_std=interval_p_std,
    )
    zt = (1.0 - t) * z0 + t * z1

    z_s_hat, dC_ds = jvp(
        lambda s_in: student(zt, t, s_in),
        primals=(s,),
        tangents=(torch.ones_like(s),),
    )

    with torch.no_grad():
        v_target = teacher(z_s_hat, s)

    residual = normalize_tangent(dC_ds - v_target)
    return (jvp_dt_ref ** 2) * residual.pow(2).mean()


@torch.no_grad()
def ayf_endpoint_eval_loss(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    base_std: float = 1.0,
    time_eps: float = 1e-4,
    ode_steps: int = 32,
) -> torch.Tensor:
    """Eval-only: C(z_t, t, 1) vs teacher ODE target."""
    z1 = latent_sampler(batch_size)
    z0 = sample_noise(batch_size, latent_dim, device, base_std)
    t = torch.rand(batch_size, 1, device=device).clamp(time_eps, 1.0 - time_eps)
    s = torch.ones_like(t)
    zt = (1.0 - t) * z0 + t * z1
    target = integrate_teacher_to_data(teacher, zt, t, steps=ode_steps)
    pred = student(zt, t, s)
    return F.mse_loss(pred, target)


@torch.no_grad()
def ayf_interval_eval_loss(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    base_std: float = 1.0,
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
    ode_steps: int = 32,
) -> torch.Tensor:
    """Eval-only: C(z_t, t, s) vs teacher ODE interval target."""
    z1 = latent_sampler(batch_size)
    z0 = sample_noise(batch_size, latent_dim, device, base_std)
    t, s = sample_ayf_time_pair(
        batch_size, device, eps=time_eps,
        p_mean=interval_p_mean, p_std=interval_p_std,
    )
    zt = (1.0 - t) * z0 + t * z1
    target = integrate_teacher_interval(teacher, zt, t, s, steps=ode_steps)
    pred = student(zt, t, s)
    return F.mse_loss(pred, target)
