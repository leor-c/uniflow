"""
EDM training loops:
    train_edm_teacher
    train_edm_student_jvp
    train_edm_student_hybrid
"""

from __future__ import annotations

from typing import Callable

import torch

from .models import EDMPrecondDenoiser
from .losses import (
    edm_teacher_loss,
    edm_endpoint_distillation_loss,
    edm_jvp_scm_loss,
    edm_endpoint_eval_loss,
)


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
