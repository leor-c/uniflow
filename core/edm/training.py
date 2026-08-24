"""
EDM training loops:
    train_edm_teacher
    train_edm_student_ayf
    train_edm_student_jvp
    train_edm_student_hybrid
"""

from __future__ import annotations

from typing import Callable

import torch

from .models import EDMConsistencyModel, EDMIntervalFlowMap, EDMPrecondDenoiser
from .losses import (
    edm_ayf_emd_loss,
    edm_ayf_interval_eval_loss,
    edm_ayf_lmd_loss,
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


def train_edm_student_ayf(
    student: EDMIntervalFlowMap,
    teacher: EDMPrecondDenoiser,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    """Train an arbitrary-interval AYF map for an EDM teacher."""
    student.train()
    teacher.eval()
    # The student parameterises normalized time over [edm_student_sigma_min,
    # sigma_max], not the teacher's [sigma_min, sigma_max]. Below roughly
    # sigma_data / 10 the PF-ODE is already the identity map, so including that
    # range would spend a large fraction of t in [0, 1] transporting nothing.
    student_sigma_min = cfg.edm_student_sigma_min
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.edm_student_lr)
    for step in range(1, cfg.edm_student_steps + 1):
        loss = torch.zeros((), device=cfg.device)
        loss_emd = None
        loss_lmd = None
        if cfg.train_mode in {"emd", "emd+lmd"}:
            loss_emd = edm_ayf_emd_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.device,
                sigma_min=student_sigma_min, sigma_max=cfg.sigma_max,
                time_eps=cfg.edm_time_eps,
                interval_p_mean=cfg.edm_interval_p_mean,
                interval_p_std=cfg.edm_interval_p_std,
                jvp_dt_ref=cfg.edm_jvp_dt_ref,
            )
            loss = loss + loss_emd
        if cfg.train_mode in {"lmd", "emd+lmd"}:
            loss_lmd = edm_ayf_lmd_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.device,
                sigma_min=student_sigma_min, sigma_max=cfg.sigma_max,
                time_eps=cfg.edm_time_eps,
                interval_p_mean=cfg.edm_interval_p_mean,
                interval_p_std=cfg.edm_interval_p_std,
                jvp_dt_ref=cfg.edm_jvp_dt_ref,
            )
            weight = cfg.edm_lmd_weight if cfg.train_mode == "emd+lmd" else 1.0
            loss = loss + weight * loss_lmd
        if loss_emd is None and loss_lmd is None:
            raise ValueError(f"Unknown EDM AYF train mode: {cfg.train_mode!r}")
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite EDM AYF loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            interval = edm_ayf_interval_eval_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.device,
                sigma_min=student_sigma_min, sigma_max=cfg.sigma_max,
                time_eps=cfg.edm_time_eps,
                interval_p_mean=cfg.edm_interval_p_mean,
                interval_p_std=cfg.edm_interval_p_std,
                ode_steps=cfg.edm_eval_teacher_ode_steps,
            ).item()
            msg = (
                f"[edm ayf] step {step:05d} | loss {loss.item():.6f} | "
                f"eval interval {interval:.6f}"
            )
            if loss_emd is not None:
                msg += f" | emd {loss_emd.item():.6f}"
            if loss_lmd is not None:
                msg += f" | lmd {loss_lmd.item():.6f}"
            print(msg)
    student.eval()


def train_edm_student_jvp(
    student: EDMConsistencyModel,
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
    student: EDMConsistencyModel,
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
