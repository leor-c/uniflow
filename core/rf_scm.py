"""
Rectified-Flow Self-Consistency Matching (RF-sCM) — generalized core.

Convention (t=0 → noise, t=1 → data):
    z0 ~ Uniform([-1,1]^d)
    z1 = encoder(x)        (data latent, bounded by tanh)
    z_t = (1 - t) * z0 + t * z1    t in [0, 1]
    teacher velocity target: v = z1 - z0

This is the OPPOSITE convention from core.interp (which has t=0 → data).
The code in this module is self-contained w.r.t. convention.

Public API
----------
Models:
    RFTeacher           – velocity field v(z, t)
    RFConsistencyStudent – consistency map C(z, t) with boundary C(z, 1) = z

Losses (differentiable, return scalar):
    rf_teacher_loss
    jvp_scm_loss
    jvp_ct_loss          – teacher-free Consistency-Training analog of jvp_scm_loss
    endpoint_distillation_eval_loss  (no_grad, monitoring only)

Training loops:
    train_teacher
    train_student_jvp
    train_student_ct     – teacher-free (CT-style) student training

Sampling (return latent tensors, no decoding):
    sample_noise            – uniform noise in [-1, 1]^d
    integrate_teacher       – ODE from t_start to 1
    sample_teacher_latents  – full generation from z0
    sample_student_latents_one_step
    sample_student_latents_k_step
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp

from .modules import FourierTimeEmbedding, MLP


# ---------------------------------------------------------------------------
# Noise prior
# ---------------------------------------------------------------------------

def sample_noise(batch_size: int, latent_dim: int, device: str | torch.device) -> torch.Tensor:
    """z0 ~ Uniform([-1, 1]^latent_dim)."""
    return torch.empty(batch_size, latent_dim, device=device).uniform_(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RFTeacher(nn.Module):
    """
    Latent-space RF velocity field v_phi(z, t).

    Regresses v = z1 - z0 along the straight-line path z_t = (1-t)z0 + t*z1.
    Input convention: t=0 is noise, t=1 is data.
    """

    def __init__(self, latent_dim: int, hidden: int = 128, depth: int = 4,
                 num_freq: int = 4, max_freq: float = 4.0):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(num_freq, max_freq)
        self.mlp = MLP(latent_dim + self.time_embed.dim, latent_dim, hidden=hidden, depth=depth)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        emb = self.time_embed(t)
        return self.mlp(torch.cat([z, emb], dim=-1))


class RFConsistencyStudent(nn.Module):
    """
    Latent-space RF consistency model.

    Parameterization (boundary-respecting):
        C(z, t) = t * z + (1 - t) * tanh(h_theta(z, t))

    This enforces:
        C(z, 1) = z          (data boundary: at t=1 the input IS the data latent)
        output stays bounded when z is bounded (tanh keeps the free part in (-1,1))
    """

    def __init__(self, latent_dim: int, hidden: int = 128, depth: int = 4,
                 num_freq: int = 4, max_freq: float = 4.0):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(num_freq, max_freq)
        self.mlp = MLP(latent_dim + self.time_embed.dim, latent_dim, hidden=hidden, depth=depth)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        emb = self.time_embed(t)
        h = self.mlp(torch.cat([z, emb], dim=-1))
        return t * z + (1.0 - t) * torch.tanh(h)


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
# Losses
# ---------------------------------------------------------------------------

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
    Continuous-time RF-sCM / JVP loss.

    Enforces d/dt C(z(t), t) = 0 along the teacher ODE dz/dt = v(z, t):

        d/dt C(z, t) = ∂_t C(z, t) + J_z C(z, t) · v(z, t) = 0

    Computed as a single forward-mode JVP of C w.r.t. (z, t) with tangent (v, 1).
    """
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
    with torch.no_grad():
        v = teacher(zt, t)

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

    Same continuous-time consistency condition as jvp_scm_loss,
        d/dt C(z_t, t) = ∂_t C + J_z C · v = 0,
    but the tangent direction v is the per-sample straight-line velocity
    z1 - z0 (a single-sample estimate of E[z1 - z0 | z_t]) rather than a
    pretrained teacher's marginal velocity prediction. This removes the need
    to train an RFTeacher before training the student.
    """
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
    v = z1 - z0

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


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_teacher(
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    """
    Train the RF teacher.

    cfg fields used: batch_size, latent_dim, device, teacher_lr,
                     teacher_steps, log_every.
    """
    teacher.train()
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.teacher_lr)
    for step in range(1, cfg.teacher_steps + 1):
        loss = rf_teacher_loss(
            teacher, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            print(f"[teacher] step {step:05d} | loss {loss.item():.6f}")
    teacher.eval()


def train_student_jvp(
    student: RFConsistencyStudent,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    """
    Train the consistency student with the pure JVP loss.

    cfg fields used: batch_size, latent_dim, device, student_lr,
                     student_steps, jvp_weight, jvp_dt_ref,
                     jvp_clean_bias, jvp_beta_alpha, jvp_beta_beta,
                     jvp_use_time_weight, jvp_time_weight_floor,
                     endpoint_ode_steps, log_every.
    """
    student.train()
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.student_lr)

    for step in range(1, cfg.student_steps + 1):
        loss_jvp = jvp_scm_loss(
            student=student,
            teacher=teacher,
            latent_sampler=latent_sampler,
            batch_size=cfg.batch_size,
            latent_dim=cfg.latent_dim,
            device=cfg.device,
            jvp_dt_ref=cfg.jvp_dt_ref,
            jvp_clean_bias=cfg.jvp_clean_bias,
            jvp_beta_alpha=cfg.jvp_beta_alpha,
            jvp_beta_beta=cfg.jvp_beta_beta,
            jvp_use_time_weight=cfg.jvp_use_time_weight,
            jvp_time_weight_floor=cfg.jvp_time_weight_floor,
        )
        loss = cfg.jvp_weight * loss_jvp

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite JVP loss at step {step}: {loss.item()}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0:
            eval_ep = endpoint_distillation_eval_loss(
                student, teacher, latent_sampler,
                cfg.batch_size, cfg.latent_dim, cfg.device,
                ode_steps=cfg.endpoint_ode_steps,
            ).item()
            print(
                f"[student jvp] step {step:05d} | "
                f"jvp {loss_jvp.item():.6f} | eval endpoint {eval_ep:.6f}"
            )

    student.eval()


def train_student_ct(
    student: RFConsistencyStudent,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
    teacher: RFTeacher | None = None,
) -> None:
    """
    Train the consistency student with the teacher-free CT-style JVP loss
    (jvp_ct_loss): the tangent direction is the per-sample velocity z1 - z0
    instead of a pretrained RFTeacher's prediction, so no teacher is needed
    for training.

    cfg fields used: batch_size, latent_dim, device, student_lr,
                     student_steps, jvp_weight, jvp_dt_ref,
                     jvp_clean_bias, jvp_beta_alpha, jvp_beta_beta,
                     jvp_use_time_weight, jvp_time_weight_floor,
                     endpoint_ode_steps, log_every.

    `teacher` is optional and used only to log an endpoint-distillation
    monitoring metric every cfg.log_every steps; it plays no role in the loss.
    """
    student.train()
    if teacher is not None:
        teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.student_lr)

    for step in range(1, cfg.student_steps + 1):
        loss_jvp = jvp_ct_loss(
            student=student,
            latent_sampler=latent_sampler,
            batch_size=cfg.batch_size,
            latent_dim=cfg.latent_dim,
            device=cfg.device,
            jvp_dt_ref=cfg.jvp_dt_ref,
            jvp_clean_bias=cfg.jvp_clean_bias,
            jvp_beta_alpha=cfg.jvp_beta_alpha,
            jvp_beta_beta=cfg.jvp_beta_beta,
            jvp_use_time_weight=cfg.jvp_use_time_weight,
            jvp_time_weight_floor=cfg.jvp_time_weight_floor,
        )
        loss = cfg.jvp_weight * loss_jvp

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite CT loss at step {step}: {loss.item()}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0:
            msg = f"[student ct] step {step:05d} | jvp {loss_jvp.item():.6f}"
            if teacher is not None:
                eval_ep = endpoint_distillation_eval_loss(
                    student, teacher, latent_sampler,
                    cfg.batch_size, cfg.latent_dim, cfg.device,
                    ode_steps=cfg.endpoint_ode_steps,
                ).item()
                msg += f" | eval endpoint {eval_ep:.6f}"
            print(msg)

    student.eval()


# ---------------------------------------------------------------------------
# Sampling (return latents; decoding is the caller's responsibility)
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