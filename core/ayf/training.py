"""
AYF training loops.

    train_teacher_gaussian  – RF teacher with z0 ~ N(0, base_std² I)
    train_student_ayf       – AYF interval student with Gaussian noise

cfg fields for teacher:  batch_size, latent_dim, device, teacher_lr,
                         teacher_steps, base_std, log_every.
cfg fields for student:  batch_size, latent_dim, device, student_lr,
                         student_steps, base_std, train_mode, emd_weight, lmd_weight,
                         jvp_dt_ref, time_eps, eval_teacher_ode_steps, log_every.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from ..ayf_rf.training import resolve_train_mode
from ..rf.models import RFTeacher
from ..rf.training import train_teacher_reflow
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
    """Train the RF teacher with Gaussian noise ``z0 ~ N(0, base_std² I)``."""
    teacher.train()
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.teacher_lr)

    for step in range(1, cfg.teacher_steps + 1):
        with torch.no_grad():
            z1 = latent_sampler(cfg.batch_size)
        z0 = sample_noise(cfg.batch_size, cfg.latent_dim, cfg.device, cfg.base_std)
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
    # ReFlow uses this arm's own base, so the coupling matches what the student
    # will later be distilled against.
    train_teacher_reflow(
        teacher, cfg, label="AYF teacher",
        noise_sampler=lambda b, d, dev: sample_noise(b, d, dev, cfg.base_std),
    )


def train_student_ayf(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    student.train()
    teacher.eval()
    terms = resolve_train_mode(cfg)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.student_lr)

    warmup_steps = getattr(cfg, "tangent_warmup_steps", 0)
    warmup_max = getattr(cfg, "tangent_warmup_max", 0.99)

    for step in range(1, cfg.student_steps + 1):
        # AYF Algorithm 1 line 7: r = min(r_max, Iters / H). r scales only the
        # (s-t)dF/dt term, so training starts as flow matching and eases the
        # map-consistency term in. warmup_steps = 0 disables it (r = 1).
        r = (1.0 if warmup_steps <= 0
             else min(warmup_max, (step - 1) / float(warmup_steps)))
        loss = torch.zeros((), device=cfg.device)
        loss_emd = None
        loss_lmd = None

        if "emd" in terms:
            loss_emd = ayf_emd_loss(
                student, teacher, latent_sampler,
                cfg.batch_size, cfg.latent_dim, cfg.device,
                base_std=cfg.base_std,
                jvp_dt_ref=cfg.jvp_dt_ref, time_eps=cfg.time_eps,
                interval_p_mean=cfg.interval_p_mean,
                interval_p_std=cfg.interval_p_std,
                tangent_warmup=r,
            )
            loss = loss + cfg.emd_weight * loss_emd

        if "lmd" in terms:
            loss_lmd = ayf_lmd_loss(
                student, teacher, latent_sampler,
                cfg.batch_size, cfg.latent_dim, cfg.device,
                base_std=cfg.base_std,
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
                base_std=cfg.base_std,
                time_eps=cfg.time_eps, ode_steps=cfg.eval_teacher_ode_steps,
            ).item()
            itv = ayf_interval_eval_loss(
                student, teacher, latent_sampler,
                cfg.batch_size, cfg.latent_dim, cfg.device,
                base_std=cfg.base_std,
                time_eps=cfg.time_eps,
                interval_p_mean=cfg.interval_p_mean,
                interval_p_std=cfg.interval_p_std, ode_steps=cfg.eval_teacher_ode_steps,
            ).item()

            msg = (
                f"[AYF {cfg.train_mode}] step {step:05d} | "
                f"loss {loss.item():.6f} | "
                f"eval endpoint {ep:.6f} | eval interval {itv:.6f}"
            )
            if loss_emd is not None:
                msg += f" | emd {loss_emd.item():.6f}"
            if loss_lmd is not None:
                msg += f" | lmd {loss_lmd.item():.6f}"
            print(msg)

    student.eval()
