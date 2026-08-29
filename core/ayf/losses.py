"""
AYF losses — identical to core.ayf_rf.losses except the noise prior:
    z0 ~ N(0, base_std² I)   instead of Uniform[-1, 1]^d

The teacher is the same RFTeacher (velocity field), trained on Gaussian paths.
All JVP objectives carry over unchanged.

    ayf_emd_loss            – source-time JVP with Gaussian noise
    ayf_lmd_loss            – target-time JVP (Boffi et al. LMD, not AYF-LMD)
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
    tangent_warmup: float = 1.0,
) -> torch.Tensor:
    """
    AYF EMD loss with Gaussian noise ``z0 ~ N(0, base_std² I)``.

    Source-time JVP:  ∂_t C(z,t,s) + J_z C(z,t,s) v_φ(z,t) = 0

    AYF-EMD as published: the target is a stopgrad copy of the network.

    The paper defines ``theta- = stopgrad(theta)`` and minimises
    ``|| f_theta(x_t,t,s) - f_theta-(x_t',t',s) ||^2`` with ``t' = t+eps(s-t)``,
    so gradient flows only through the prediction and the tangent is a constant.
    Written in the continuous limit that is

        loss = || f_theta - sg(f_theta - g) ||^2,   g = normalised d f_theta-/dt

    which has the same value as ``||g||^2`` but the paper's gradient,
    ``grad_theta f_theta^T g``.

    Taking the JVP through the *live* network instead lets the model lower the
    loss by shrinking its own Jacobian rather than by satisfying the
    consistency condition -- the degeneracy the stopgrad exists to prevent.
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
