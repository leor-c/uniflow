"""
RF-sCM training loops:
    train_teacher
    train_student_jvp
    train_student_ct     – teacher-free (CT-style) student training
"""

from __future__ import annotations

from typing import Callable

import torch

from .models import RFTeacher, RFConsistencyStudent
from .losses import jvp_scm_loss, jvp_ct_loss, rf_teacher_loss, endpoint_distillation_eval_loss


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
