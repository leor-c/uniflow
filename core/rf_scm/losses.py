"""
RF-sCM losses (differentiable, return scalar unless noted):
    rf_teacher_loss
    jvp_scm_loss
    jvp_ct_loss          – teacher-free Consistency-Training analog of jvp_scm_loss
    endpoint_distillation_eval_loss  (no_grad, monitoring only)
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.func import jvp

from .models import RFTeacher, RFConsistencyStudent
from .sampling import sample_noise, integrate_teacher


def rf_teacher_loss(
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Rectified-flow regression loss for the teacher."""
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    z0 = sample_noise(batch_size, latent_dim, device)
    t = torch.rand(batch_size, 1, device=device)
    zt = (1.0 - t) * z0 + t * z1
    target_v = z1 - z0
    pred_v = teacher(zt, t)
    return F.mse_loss(pred_v, target_v)


@torch.no_grad()
def endpoint_distillation_eval_loss(
    student: RFConsistencyStudent,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    ode_steps: int = 32,
) -> torch.Tensor:
    """Eval-only endpoint distillation loss (no gradients computed)."""
    z1 = latent_sampler(batch_size)
    z0 = sample_noise(batch_size, latent_dim, device)
    t = torch.rand(batch_size, 1, device=device).clamp(0.0, 1.0 - 1e-4)
    zt = (1.0 - t) * z0 + t * z1
    target = integrate_teacher(teacher, zt, t, steps=ode_steps)
    pred = student(zt, t)
    return F.mse_loss(pred, target)


def _sample_zt(
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    jvp_clean_bias: bool,
    jvp_beta_alpha: float,
    jvp_beta_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared (z0, z1, z_t, t) sampling for the continuous-time JVP losses."""
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    z0 = sample_noise(batch_size, latent_dim, device)

    if jvp_clean_bias:
        t = torch.distributions.Beta(
            torch.tensor(jvp_beta_alpha, device=device),
            torch.tensor(jvp_beta_beta, device=device),
        ).sample((batch_size, 1))
    else:
        t = torch.rand(batch_size, 1, device=device)
    t = t.clamp(1e-4, 1.0 - 1e-4)

    zt = (1.0 - t) * z0 + t * z1
    return z0, z1, zt, t


def _jvp_consistency_loss(
    student: RFConsistencyStudent,
    zt: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    jvp_dt_ref: float,
    jvp_use_time_weight: bool,
    jvp_time_weight_floor: float,
) -> torch.Tensor:
    """
    Shared continuous-time consistency residual.

    Enforces d/dt C(z(t), t) = 0 along dz/dt = v:

        d/dt C(z, t) = ∂_t C(z, t) + J_z C(z, t) · v = 0

    Computed as a single forward-mode JVP of C w.r.t. (z, t) with tangent (v, 1).
    """
    _, tangent = jvp(
        lambda z_in, t_in: student(z_in, t_in),
        primals=(zt, t),
        tangents=(v, torch.ones_like(t)),
    )

    norm = torch.sqrt(1.0 + v.pow(2).mean(dim=-1, keepdim=True)).detach()
    residual = tangent / norm

    if jvp_use_time_weight:
        weight = 1.0 / (1.0 - t).clamp_min(jvp_time_weight_floor)
        return (weight * residual.pow(2)).mean()

    return (jvp_dt_ref ** 2) * residual.pow(2).mean()


def jvp_scm_loss(
    student: RFConsistencyStudent,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    jvp_dt_ref: float = 1.0,
    jvp_clean_bias: bool = False,
    jvp_beta_alpha: float = 2.0,
    jvp_beta_beta: float = 1.0,
    jvp_use_time_weight: bool = False,
    jvp_time_weight_floor: float = 0.05,
) -> torch.Tensor:
    """
    Continuous-time RF-sCM / JVP loss. The tangent direction v is the
    pretrained teacher's marginal velocity prediction at z_t. See
    _jvp_consistency_loss for the consistency condition being enforced.
    """
    _, _, zt, t = _sample_zt(
        latent_sampler, batch_size, latent_dim, device,
        jvp_clean_bias, jvp_beta_alpha, jvp_beta_beta,
    )
    with torch.no_grad():
        v = teacher(zt, t)
    return _jvp_consistency_loss(
        student, zt, t, v, jvp_dt_ref, jvp_use_time_weight, jvp_time_weight_floor
    )


def jvp_ct_loss(
    student: RFConsistencyStudent,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    jvp_dt_ref: float = 1.0,
    jvp_clean_bias: bool = False,
    jvp_beta_alpha: float = 2.0,
    jvp_beta_beta: float = 1.0,
    jvp_use_time_weight: bool = False,
    jvp_time_weight_floor: float = 0.05,
) -> torch.Tensor:
    """
    Consistency-Training (CT) analog of jvp_scm_loss — no teacher network.

    The tangent direction v is the per-sample straight-line velocity z1 - z0
    (a single-sample estimate of E[z1 - z0 | z_t]) rather than a pretrained
    teacher's marginal velocity prediction. This removes the need to train
    an RFTeacher before training the student. See _jvp_consistency_loss for
    the consistency condition being enforced.
    """
    z0, z1, zt, t = _sample_zt(
        latent_sampler, batch_size, latent_dim, device,
        jvp_clean_bias, jvp_beta_alpha, jvp_beta_beta,
    )
    v = z1 - z0
    return _jvp_consistency_loss(
        student, zt, t, v, jvp_dt_ref, jvp_use_time_weight, jvp_time_weight_floor
    )
