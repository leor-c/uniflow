"""
AYF-style interval flow map losses (differentiable, return scalar unless noted):
    ayf_emd_loss                   – teacher-based source-time JVP (EMD)
    ayf_emd_ct_loss                – teacher-free EMD analog, v = z1 - z0
    ayf_lmd_loss                   – teacher-based target-time JVP (LMD)
    ayf_pairwise_anchor_loss       – cheap chord-interval supervised anchor
    ayf_teacher_interval_anchor_loss – teacher ODE interval supervised anchor
    ayf_endpoint_eval_loss         – (no_grad) C(z,t,1) vs teacher ODE to t=1
    ayf_interval_eval_loss         – (no_grad) C(z,t,s) vs teacher ODE to s

The EMD loss enforces invariance of the learned interval map along the RF
teacher trajectory.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.func import jvp

from ..rf import RFTeacher, sample_uniform_noise
from .models import RFIntervalFlowMap
from .sampling import sample_ayf_time_pair, integrate_teacher_interval, integrate_teacher_to_data

# Align Your Flow tangent-normalization constant.
AYF_TANGENT_NORM_C = 0.1


def normalize_tangent(
    tangent: torch.Tensor,
    c: float = AYF_TANGENT_NORM_C,
) -> torch.Tensor:
    """AYF tangent normalization, shared by every arm in this repo.

        df/dt  ->  (df/dt) / (||df/dt|| + c),   c = 0.1

    The norm is taken per sample over the whole tangent vector and detached, so
    it rescales the residual without contributing a gradient of its own. The
    resulting squared loss grows like ``r^2 / c^2`` for a small tangent norm
    ``r`` and saturates at 1 for a large one, which is what bounds the
    influence of the rare huge tangents that destabilise map distillation.

    This replaces an earlier teacher-velocity normalization
    ``sqrt(1 + mean(v^2))``, which scaled the residual by the magnitude of the
    teacher velocity rather than by the tangent's own norm.
    """
    scale = tangent.detach().norm(dim=-1, keepdim=True)
    return tangent / (scale + c)


def ayf_emd_loss(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    jvp_dt_ref: float = 1.0,
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
) -> torch.Tensor:
    """
    Teacher-based Eulerian/source-time map distillation (EMD):

        || partial_t C(z,t,s) + J_z C(z,t,s) v_phi(z,t) ||^2

    The target time s is held fixed; this reduces to the endpoint
    RF-sCM/JVP loss when s = 1.
    """
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    z0 = sample_uniform_noise(batch_size, latent_dim, device)
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


def ayf_emd_ct_loss(
    student: RFIntervalFlowMap,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    jvp_dt_ref: float = 1.0,
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
) -> torch.Tensor:
    """
    Teacher-free AYF-style EMD objective (AYF-CT): uses the sampled pairwise
    RF velocity v = z1 - z0 instead of the teacher's prediction.

    Warning: with independent coupling, pairwise chords can cross, so this
    can be much harder to optimize than teacher-based EMD.
    """
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    z0 = sample_uniform_noise(batch_size, latent_dim, device)
    t, s = sample_ayf_time_pair(
        batch_size, device, eps=time_eps,
        p_mean=interval_p_mean, p_std=interval_p_std,
    )
    zt = (1.0 - t) * z0 + t * z1
    v = z1 - z0

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
    jvp_dt_ref: float = 1.0,
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
) -> torch.Tensor:
    """
    Teacher-based target-time/Lagrangian map distillation (LMD):

        || partial_s C(z,t,s) - v_phi(C(z,t,s), s) ||^2

    Asks the predicted target state to move according to the teacher
    velocity as the target time s changes.
    """
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    z0 = sample_uniform_noise(batch_size, latent_dim, device)
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


def ayf_pairwise_anchor_loss(
    student: RFIntervalFlowMap,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
) -> torch.Tensor:
    """
    Cheap pairwise interval anchor along the sampled RF chord:

        C(z_t,t,s) ≈ z_s = (1-s)z0 + s z1

    Not teacher ODE distillation — useful only as a sanity check or weak anchor.
    """
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    z0 = sample_uniform_noise(batch_size, latent_dim, device)
    t, s = sample_ayf_time_pair(
        batch_size, device, eps=time_eps,
        p_mean=interval_p_mean, p_std=interval_p_std,
    )
    zt = (1.0 - t) * z0 + t * z1
    zs = (1.0 - s) * z0 + s * z1
    pred = student(zt, t, s)
    return F.mse_loss(pred, zs)


def ayf_teacher_interval_anchor_loss(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
    ode_steps: int = 16,
) -> torch.Tensor:
    """
    Teacher ODE interval target: C(z_t,t,s) ≈ Phi_teacher(t -> s)(z_t).

    More expensive than EMD (integrates the teacher every batch); useful for
    debugging and as a strong anchor.
    """
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    z0 = sample_uniform_noise(batch_size, latent_dim, device)
    t, s = sample_ayf_time_pair(
        batch_size, device, eps=time_eps,
        p_mean=interval_p_mean, p_std=interval_p_std,
    )
    zt = (1.0 - t) * z0 + t * z1

    with torch.no_grad():
        target = integrate_teacher_interval(teacher, zt, t, s, steps=ode_steps)
    pred = student(zt, t, s)
    return F.mse_loss(pred, target)


@torch.no_grad()
def ayf_endpoint_eval_loss(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    time_eps: float = 1e-4,
    ode_steps: int = 32,
) -> torch.Tensor:
    """Eval-only: C(z_t,t,1) vs teacher ODE target Phi(t -> 1)(z_t)."""
    z1 = latent_sampler(batch_size)
    z0 = sample_uniform_noise(batch_size, latent_dim, device)
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
    time_eps: float = 1e-4,
    interval_p_mean: float = -0.6,
    interval_p_std: float = 1.6,
    ode_steps: int = 32,
) -> torch.Tensor:
    """Eval-only: C(z_t,t,s) vs teacher ODE target Phi(t -> s)(z_t)."""
    z1 = latent_sampler(batch_size)
    z0 = sample_uniform_noise(batch_size, latent_dim, device)
    t, s = sample_ayf_time_pair(
        batch_size, device, eps=time_eps,
        p_mean=interval_p_mean, p_std=interval_p_std,
    )
    zt = (1.0 - t) * z0 + t * z1
    target = integrate_teacher_interval(teacher, zt, t, s, steps=ode_steps)
    pred = student(zt, t, s)
    return F.mse_loss(pred, target)
