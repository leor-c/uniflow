"""
EDM (Elucidating the Design Space of Diffusion-Based Generative Models) baseline.

Uses the same bounded latent space (tanh AE) as RF-sCM, but:
  - Noise prior: Gaussian N(0, sigma^2 I) instead of Uniform[-1,1]^d
  - Time coordinate: log(sigma) (EDM convention)
  - Denoiser parameterization: EDM preconditioning (Karras et al. 2022)
  - Consistency loss: JVP in log-sigma coordinates

Convention (matching Karras et al.):
  Large sigma = more noise, small sigma = cleaner.
  ODE direction: sigma decreasing (noisy → clean).

Public API
----------
Models:
    EDMPrecondDenoiser  – preconditioned denoiser D(z_noisy, log_sigma) -> z_clean

Utilities:
    edm_sample_log_sigma    – log-normal sigma sampling for training
    edm_karras_sigmas       – Karras decreasing sigma schedule for sampling

Losses:
    edm_teacher_loss
    edm_endpoint_distillation_loss
    edm_jvp_scm_loss
    edm_endpoint_eval_loss  (no_grad)

Training:
    train_edm_teacher
    train_edm_student_jvp
    train_edm_student_hybrid

Sampling (return latents):
    integrate_edm_teacher       – ODE from log_sigma_start to log(sigma_min)
    sample_edm_teacher_latents  – full Karras ODE
    sample_edm_student_latents_one_step
    sample_edm_student_latents_k_step
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
# EDM utilities
# ---------------------------------------------------------------------------

def edm_sample_log_sigma(
    batch_size: int,
    device: str | torch.device,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    sigma_min: float = 0.002,
    sigma_max: float = 2.0,
) -> torch.Tensor:
    """
    Sample log(sigma) from the log-normal training distribution (Karras Table 1).
    Returns [B, 1] clamped to [log(sigma_min), log(sigma_max)].
    """
    log_sigma = p_mean + p_std * torch.randn(batch_size, 1, device=device)
    return log_sigma.clamp(math.log(sigma_min), math.log(sigma_max))


def edm_karras_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Karras rho-schedule: decreasing sigmas from sigma_max to sigma_min."""
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    return (max_inv + ramp * (min_inv - max_inv)) ** rho


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EDMPrecondDenoiser(nn.Module):
    """
    EDM-preconditioned denoiser D(z_noisy, log_sigma) → z_clean.

    Preconditioning (Karras et al. 2022, Table 1):
        c_skip  = sigma_data^2 / (sigma^2 + sigma_data^2)
        c_out   = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
        c_in    = 1 / sqrt(sigma^2 + sigma_data^2)
        c_noise = 0.25 * log_sigma      [input to time embedding]

    Boundary: as sigma → 0, D(z, sigma) → z.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 128,
        depth: int = 4,
        sigma_data: float = 0.5,
        num_freq: int = 4,
        max_freq: float = 4.0,
    ):
        super().__init__()
        self.sigma_data = float(sigma_data)
        self.noise_embed = FourierTimeEmbedding(num_freq, max_freq)
        self.mlp = MLP(
            latent_dim + self.noise_embed.dim, latent_dim, hidden=hidden, depth=depth
        )

    def forward(self, z: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        if log_sigma.ndim == 1:
            log_sigma = log_sigma[:, None]

        sigma = log_sigma.exp()
        sd = self.sigma_data
        s2 = sigma.pow(2)
        sd2 = sd ** 2

        c_skip = sd2 / (s2 + sd2)
        c_out = sigma * sd / torch.sqrt(s2 + sd2)
        c_in = 1.0 / torch.sqrt(s2 + sd2)
        c_noise = 0.25 * log_sigma

        emb = self.noise_embed(c_noise)
        h = self.mlp(torch.cat([c_in * z, emb], dim=-1))
        return c_skip * z + c_out * h


# ---------------------------------------------------------------------------
# ODE integration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

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


def edm_endpoint_distillation_loss(
    student: EDMPrecondDenoiser,
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
    student: EDMPrecondDenoiser,
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
    student: EDMPrecondDenoiser,
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


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_edm_teacher(
    denoiser: EDMPrecondDenoiser,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    """
    cfg fields: batch_size, device, edm_teacher_lr, edm_teacher_steps,
                sigma_data, edm_p_mean, edm_p_std, sigma_min, sigma_max, log_every.
    """
    denoiser.train()
    opt = torch.optim.AdamW(denoiser.parameters(), lr=cfg.edm_teacher_lr)
    for step in range(1, cfg.edm_teacher_steps + 1):
        loss = edm_teacher_loss(
            denoiser, latent_sampler, cfg.batch_size, cfg.device,
            sigma_data=cfg.sigma_data,
            p_mean=cfg.edm_p_mean, p_std=cfg.edm_p_std,
            sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            print(f"[edm teacher] step {step:05d} | loss {loss.item():.6f}")
    denoiser.eval()


def train_edm_student_jvp(
    student: EDMPrecondDenoiser,
    teacher: EDMPrecondDenoiser,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    """
    cfg fields: batch_size, device, edm_student_lr, edm_student_steps,
                edm_p_mean, edm_p_std, sigma_min, sigma_max, edm_jvp_dlog_ref,
                edm_endpoint_ode_steps, log_every.
    """
    student.train()
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.edm_student_lr)
    for step in range(1, cfg.edm_student_steps + 1):
        loss = edm_jvp_scm_loss(
            student, teacher, latent_sampler, cfg.batch_size, cfg.device,
            sigma_min=cfg.sigma_min, p_mean=cfg.edm_p_mean, p_std=cfg.edm_p_std,
            sigma_max=cfg.sigma_max, jvp_dlog_ref=cfg.edm_jvp_dlog_ref,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite EDM JVP loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            eval_ep = edm_endpoint_eval_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.device,
                sigma_min=cfg.sigma_min, p_mean=cfg.edm_p_mean, p_std=cfg.edm_p_std,
                sigma_max=cfg.sigma_max, ode_steps=cfg.edm_endpoint_ode_steps,
            ).item()
            print(
                f"[edm student jvp] step {step:05d} | "
                f"loss {loss.item():.6f} | eval endpoint {eval_ep:.6f}"
            )
    student.eval()


def train_edm_student_hybrid(
    student: EDMPrecondDenoiser,
    teacher: EDMPrecondDenoiser,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    """
    cfg fields: same as train_edm_student_jvp, plus edm_endpoint_every,
                edm_jvp_weight, edm_anchor_jvp_weight.
    """
    student.train()
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.edm_student_lr)
    for step in range(1, cfg.edm_student_steps + 1):
        do_ep = (step == 1) or (step % cfg.edm_endpoint_every == 0)

        if do_ep:
            loss_ep = edm_endpoint_distillation_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.device,
                sigma_min=cfg.sigma_min, p_mean=cfg.edm_p_mean, p_std=cfg.edm_p_std,
                sigma_max=cfg.sigma_max, ode_steps=cfg.edm_endpoint_ode_steps,
            )
            loss = loss_ep
            ep_val = loss_ep.item()
            jvp_val = float("nan")
            if cfg.edm_anchor_jvp_weight != 0.0:
                loss_jvp = edm_jvp_scm_loss(
                    student, teacher, latent_sampler, cfg.batch_size, cfg.device,
                    sigma_min=cfg.sigma_min, p_mean=cfg.edm_p_mean, p_std=cfg.edm_p_std,
                    sigma_max=cfg.sigma_max, jvp_dlog_ref=cfg.edm_jvp_dlog_ref,
                )
                loss = loss + cfg.edm_anchor_jvp_weight * loss_jvp
                jvp_val = loss_jvp.item()
            mode = "endpoint" if cfg.edm_anchor_jvp_weight == 0.0 else "endpoint+jvp"
        else:
            loss_jvp = edm_jvp_scm_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.device,
                sigma_min=cfg.sigma_min, p_mean=cfg.edm_p_mean, p_std=cfg.edm_p_std,
                sigma_max=cfg.sigma_max, jvp_dlog_ref=cfg.edm_jvp_dlog_ref,
            )
            loss = cfg.edm_jvp_weight * loss_jvp
            ep_val = float("nan")
            jvp_val = loss_jvp.item()
            mode = "jvp"

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite EDM student loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            eval_ep = edm_endpoint_eval_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.device,
                sigma_min=cfg.sigma_min, p_mean=cfg.edm_p_mean, p_std=cfg.edm_p_std,
                sigma_max=cfg.sigma_max, ode_steps=cfg.edm_endpoint_ode_steps,
            ).item()
            print(
                f"[edm student {mode}] step {step:05d} | "
                f"loss {loss.item():.6f} | endpoint {ep_val:.6f} | "
                f"jvp {jvp_val:.6f} | eval endpoint {eval_ep:.6f}"
            )
    student.eval()


# ---------------------------------------------------------------------------
# Sampling (return latents)
# ---------------------------------------------------------------------------

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