"""
AYF training loops.

    train_teacher_gaussian  – RF teacher with z0 ~ N(0, sigma_data² I)
    train_student_ayf       – AYF interval student with Gaussian noise

cfg fields for teacher:  batch_size, latent_dim, device, teacher_lr,
                         teacher_steps, sigma_data, log_every.
cfg fields for student:  batch_size, latent_dim, device, student_lr,
                         student_steps, sigma_data, emd_weight, lmd_weight,
                         jvp_dt_ref, time_eps, eval_teacher_ode_steps, log_every.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from ..rf.models import RFTeacher
from ..ayf_rf.models import RFIntervalFlowMap
from .sampling import sample_noise
from .losses import (
    ayf_emd_loss,
    ayf_lmd_loss,
    ayf_endpoint_eval_loss,
    ayf_interval_eval_loss,
)


def train_teacher_gaussian(
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    """Train the RF teacher with Gaussian noise z0 ~ N(0, sigma_data² I)."""
    teacher.train()
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.teacher_lr)

    for step in range(1, cfg.teacher_steps + 1):
        with torch.no_grad():
            z1 = latent_sampler(cfg.batch_size)
        z0 = sample_noise(cfg.batch_size, cfg.latent_dim, cfg.device, cfg.sigma_data)
        t = torch.rand(cfg.batch_size, 1, device=cfg.device)
        zt = (1.0 - t) * z0 + t * z1
        pred_v = teacher(zt, t)
        loss = F.mse_loss(pred_v, z1 - z0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0:
            print(f"[AYF teacher] step {step:05d} | loss {loss.item():.6f}")

    teacher.eval()


def train_student_ayf(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    student.train()
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.student_lr)

    for step in range(1, cfg.student_steps + 1):
        loss_emd = ayf_emd_loss(
            student, teacher, latent_sampler,
            cfg.batch_size, cfg.latent_dim, cfg.device,
            sigma_data=cfg.sigma_data,
            jvp_dt_ref=cfg.jvp_dt_ref, time_eps=cfg.time_eps,
            interval_p_mean=cfg.interval_p_mean,
            interval_p_std=cfg.interval_p_std,
        )
        loss = cfg.emd_weight * loss_emd

        loss_lmd = None
        if cfg.lmd_weight > 0.0:
            loss_lmd = ayf_lmd_loss(
                student, teacher, latent_sampler,
                cfg.batch_size, cfg.latent_dim, cfg.device,
                sigma_data=cfg.sigma_data,
                jvp_dt_ref=cfg.jvp_dt_ref, time_eps=cfg.time_eps,
                interval_p_mean=cfg.interval_p_mean,
                interval_p_std=cfg.interval_p_std,
            )
            loss = loss + cfg.lmd_weight * loss_lmd

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite AYF loss at step {step}: {loss.item()}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0:
            ep = ayf_endpoint_eval_loss(
                student, teacher, latent_sampler,
                cfg.batch_size, cfg.latent_dim, cfg.device,
                sigma_data=cfg.sigma_data,
                time_eps=cfg.time_eps, ode_steps=cfg.eval_teacher_ode_steps,
            ).item()
            itv = ayf_interval_eval_loss(
                student, teacher, latent_sampler,
                cfg.batch_size, cfg.latent_dim, cfg.device,
                sigma_data=cfg.sigma_data,
                time_eps=cfg.time_eps,
                interval_p_mean=cfg.interval_p_mean,
                interval_p_std=cfg.interval_p_std, ode_steps=cfg.eval_teacher_ode_steps,
            ).item()

            msg = (
                f"[AYF] step {step:05d} | "
                f"loss {loss.item():.6f} | emd {loss_emd.item():.6f} | "
                f"eval endpoint {ep:.6f} | eval interval {itv:.6f}"
            )
            if loss_lmd is not None:
                msg += f" | lmd {loss_lmd.item():.6f}"
            print(msg)

    student.eval()
