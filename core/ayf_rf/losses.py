"""
AYF-style interval flow map losses (differentiable, return scalar unless noted):
    ayf_emd_loss                   – teacher-based source-time JVP (EMD)
    ayf_emd_ct_loss                – teacher-free EMD analog, v = z1 - z0
    ayf_lmd_loss                   – target-time JVP (Boffi et al. LMD, not AYF-LMD)
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
    tangent_warmup: float = 1.0,
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
        # AYF Algorithm 1, lines 6-9. The tangent is built from the SLOPE net
        # under stopgrad'd weights, and the warmup r multiplies only the
        # (s-t)*dF/dt term -- so r=0 is pure flow matching (drive F onto the
        # teacher velocity) and r=1 is the full map-consistency tangent.
        slope_sg, dF_dt = jvp(
            lambda z_in, t_in: student.slope(z_in, t_in, s),
            primals=(zt, t),
            tangents=(v, torch.ones_like(t)),
        )
        g = (slope_sg - v) - tangent_warmup * (s - t) * dF_dt
        g = normalize_tangent(g)

    # Algorithm 1 line 10: the loss is on the slope F_theta, not the full map
    # C_theta = z + (s-t)F. Using C multiplies the gradient by (s-t), which
    # suppresses exactly the short intervals whose tangent carries the
    # well-constrained flow-matching signal.
    slope = student.slope(zt, t, s)
    residual = slope - slope_sg + g
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

    with torch.no_grad():
        # Tangent under stopgrad'd weights, as in ayf_emd_loss.
        _, tangent = jvp(
            lambda z_in, t_in: student(z_in, t_in, s),
            primals=(zt, t),
            tangents=(v, torch.ones_like(t)),
        )

    target_step = normalize_tangent(tangent)
    prediction = student(zt, t, s)
    # Paper target: f_theta-(x_t', t', s) ~= f + eps(s-t) * tangent, stopgrad'd.
    # The residual is therefore -tangent, so descent moves the map *along* the
    # teacher trajectory. Using +tangent gives the same loss value with the
    # gradient reversed, which drives the slope away from the teacher velocity.
    target = (prediction + target_step).detach()
    residual = prediction - target
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
    Target-time/Lagrangian map distillation.

        || partial_s C(z,t,s) - v_phi(C(z,t,s), s) ||^2

    NOTE: this is the LMD loss of Boffi et al., not AYF-LMD. Align Your Flow's
    Theorem 3.3 defines AYF-LMD with a stopgrad target,

        || f_theta(x_t,t,s) - ODE_{s'->s}(f_theta-(x_t,t,s')) ||^2,

    and states that "by removing the stop-gradient operation ... and allowing
    gradients to flow through both terms, we recover the LMD loss from Boffi et
    al." Here the JVP carries gradient and only the teacher velocity is
    detached, so this is the Boffi form with a partial stopgrad.

    That is a deliberate, documented gap rather than a bug. The paper reports
    the Boffi form "was already stable in practice and did not introduce
    training instabilities" and that "both versions performed similarly",
    recommending AYF-LMD for training *efficiency* (it avoids backpropagating
    through the JVP), not for quality.

    This also explains why EMD and LMD behaved so differently here before the
    EMD stopgrad was added: the paper says EMD needs the stopgrad and LMD does
    not, and this codebase originally lacked it in both.
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
