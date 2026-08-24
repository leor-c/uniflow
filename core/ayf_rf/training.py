"""
AYF-style training loop:
    train_student_interval

The RF teacher is trained separately with ``core.rf.train_teacher``.
"""

from __future__ import annotations

from typing import Callable

import torch

from ..rf import RFTeacher
from .models import RFIntervalFlowMap
from .losses import (
    ayf_emd_loss,
    ayf_emd_ct_loss,
    ayf_lmd_loss,
    ayf_pairwise_anchor_loss,
    ayf_teacher_interval_anchor_loss,
    ayf_endpoint_eval_loss,
    ayf_interval_eval_loss,
)


def train_student_interval(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    """
    Train the AYF-style interval flow map.

    Default: teacher-based EMD only (cfg.train_mode == "emd").
    Optional terms (each enabled by a positive weight):
        LMD, pairwise chord anchor, teacher interval anchor.
    cfg.train_mode == "ct" replaces EMD with the teacher-free AYF-CT analog.

    cfg fields used: batch_size, latent_dim, device, student_lr, student_steps,
                     train_mode ("emd" | "ct"), emd_weight, lmd_weight,
                     pairwise_anchor_weight, teacher_interval_anchor_weight,
                     jvp_dt_ref, time_eps, interval_teacher_ode_steps,
                     eval_teacher_ode_steps, log_every.
    """
    student.train()
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.student_lr)

    for step in range(1, cfg.student_steps + 1):
        if cfg.train_mode == "emd":
            loss_emd = ayf_emd_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                jvp_dt_ref=cfg.jvp_dt_ref, time_eps=cfg.time_eps,
            )
        elif cfg.train_mode == "ct":
            loss_emd = ayf_emd_ct_loss(
                student, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                jvp_dt_ref=cfg.jvp_dt_ref, time_eps=cfg.time_eps,
            )
        else:
            raise ValueError(f"Unknown train_mode: {cfg.train_mode!r}")

        loss = cfg.emd_weight * loss_emd

        loss_lmd = None
        loss_pair = None
        loss_teacher_anchor = None

        if cfg.lmd_weight > 0.0:
            loss_lmd = ayf_lmd_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                jvp_dt_ref=cfg.jvp_dt_ref, time_eps=cfg.time_eps,
            )
            loss = loss + cfg.lmd_weight * loss_lmd

        if cfg.pairwise_anchor_weight > 0.0:
            loss_pair = ayf_pairwise_anchor_loss(
                student, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                time_eps=cfg.time_eps,
            )
            loss = loss + cfg.pairwise_anchor_weight * loss_pair

        if cfg.teacher_interval_anchor_weight > 0.0:
            loss_teacher_anchor = ayf_teacher_interval_anchor_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                time_eps=cfg.time_eps, ode_steps=cfg.interval_teacher_ode_steps,
            )
            loss = loss + cfg.teacher_interval_anchor_weight * loss_teacher_anchor

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite student loss at step {step}: {loss.item()}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0:
            ep = ayf_endpoint_eval_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                time_eps=cfg.time_eps, ode_steps=cfg.eval_teacher_ode_steps,
            ).item()
            itv = ayf_interval_eval_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                time_eps=cfg.time_eps, ode_steps=cfg.eval_teacher_ode_steps,
            ).item()

            msg = (
                f"[AYF {cfg.train_mode}] step {step:05d} | "
                f"loss {loss.item():.6f} | "
                f"emd/ct {loss_emd.item():.6f} | "
                f"eval endpoint {ep:.6f} | "
                f"eval interval {itv:.6f}"
            )
            if loss_lmd is not None:
                msg += f" | lmd {loss_lmd.item():.6f}"
            if loss_pair is not None:
                msg += f" | pair {loss_pair.item():.6f}"
            if loss_teacher_anchor is not None:
                msg += f" | teacher_anchor {loss_teacher_anchor.item():.6f}"
            print(msg)

    student.eval()
