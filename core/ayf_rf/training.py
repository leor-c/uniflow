"""
AYF-style training loop:
    train_student_interval

Also hosts the shared ``train_mode`` grammar (``parse_train_mode`` /
``resolve_train_mode``) used by every arm, so "which AYF objective am I
training?" has one answer across ``core.ayf_rf``, ``core.ayf`` and the EDM arm.

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


def parse_train_mode(train_mode: str, *, allow_ct: bool = False) -> frozenset[str]:
    """Split an AYF ``train_mode`` string into the loss terms it enables.

    Terms are joined with ``+``:

        "emd"      teacher-based Eulerian map distillation (source-time JVP)
        "lmd"      Lagrangian map distillation (target-time JVP)
        "ct"       teacher-free EMD analog, where the arm supports it
        "emd+lmd"  both, mixed by ``emd_weight`` and ``lmd_weight``

    ``emd`` and ``ct`` both supply the source-time term, so they are mutually
    exclusive. The mode alone decides which terms are active and the weights
    only set the mix, so a zero weight can never silently switch a term off.

    Align Your Flow reports AYF-LMD as the more stable objective in their 2D
    toy experiments, and AYF-EMD as the better one on image data; this grammar
    is what makes that choice reachable per arm.
    """
    terms = frozenset(part.strip() for part in train_mode.split("+") if part.strip())
    valid = {"emd", "lmd"} | ({"ct"} if allow_ct else set())
    unknown = terms - valid
    if not terms or unknown:
        raise ValueError(
            f"Unknown AYF train_mode {train_mode!r}: expected '+'-joined terms "
            f"from {sorted(valid)}, e.g. 'emd', 'lmd' or 'emd+lmd'"
        )
    if {"emd", "ct"} <= terms:
        raise ValueError(
            f"train_mode {train_mode!r} combines 'emd' and 'ct'; both supply the "
            "source-time term, so pick one"
        )
    return terms


def resolve_train_mode(cfg, *, allow_ct: bool = False) -> frozenset[str]:
    """Parse ``cfg.train_mode`` and check the weights of its active terms."""
    terms = parse_train_mode(cfg.train_mode, allow_ct=allow_ct)
    if terms & {"emd", "ct"} and cfg.emd_weight <= 0.0:
        raise ValueError(
            f"train_mode {cfg.train_mode!r} needs a positive emd_weight, "
            f"got {cfg.emd_weight}"
        )
    if "lmd" in terms and cfg.lmd_weight <= 0.0:
        raise ValueError(
            f"train_mode {cfg.train_mode!r} needs a positive lmd_weight, "
            f"got {cfg.lmd_weight}"
        )
    return terms


def train_student_interval(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    """
    Train the AYF-style interval flow map.

    ``cfg.train_mode`` selects the objective; see ``parse_train_mode``. It is
    a '+'-joined subset of "emd" (teacher-based source-time), "ct" (the
    teacher-free EMD analog) and "lmd" (target-time), e.g. "emd", "lmd",
    "emd+lmd" or "ct+lmd". The auxiliary chord and teacher-interval anchors
    stay weight-gated and are outside the mode.

    cfg fields used: batch_size, latent_dim, device, student_lr, student_steps,
                     train_mode, emd_weight, lmd_weight,
                     pairwise_anchor_weight, teacher_interval_anchor_weight,
                     jvp_dt_ref, time_eps, interval_p_mean, interval_p_std,
                     interval_teacher_ode_steps,
                     eval_teacher_ode_steps, log_every.
    """
    student.train()
    teacher.eval()
    terms = resolve_train_mode(cfg, allow_ct=True)
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
        loss_pair = None
        loss_teacher_anchor = None

        if "emd" in terms:
            loss_emd = ayf_emd_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                jvp_dt_ref=cfg.jvp_dt_ref, time_eps=cfg.time_eps,
                interval_p_mean=cfg.interval_p_mean,
                interval_p_std=cfg.interval_p_std,
                tangent_warmup=r,
            )
        elif "ct" in terms:
            loss_emd = ayf_emd_ct_loss(
                student, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                jvp_dt_ref=cfg.jvp_dt_ref, time_eps=cfg.time_eps,
                interval_p_mean=cfg.interval_p_mean,
                interval_p_std=cfg.interval_p_std,
                tangent_warmup=r,
            )
        if loss_emd is not None:
            loss = loss + cfg.emd_weight * loss_emd

        if "lmd" in terms:
            loss_lmd = ayf_lmd_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                jvp_dt_ref=cfg.jvp_dt_ref, time_eps=cfg.time_eps,
                interval_p_mean=cfg.interval_p_mean,
                interval_p_std=cfg.interval_p_std,
            )
            loss = loss + cfg.lmd_weight * loss_lmd

        if cfg.pairwise_anchor_weight > 0.0:
            loss_pair = ayf_pairwise_anchor_loss(
                student, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                time_eps=cfg.time_eps,
                interval_p_mean=cfg.interval_p_mean,
                interval_p_std=cfg.interval_p_std,
            )
            loss = loss + cfg.pairwise_anchor_weight * loss_pair

        if cfg.teacher_interval_anchor_weight > 0.0:
            loss_teacher_anchor = ayf_teacher_interval_anchor_loss(
                student, teacher, latent_sampler, cfg.batch_size, cfg.latent_dim, cfg.device,
                time_eps=cfg.time_eps,
                interval_p_mean=cfg.interval_p_mean,
                interval_p_std=cfg.interval_p_std,
                ode_steps=cfg.interval_teacher_ode_steps,
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
                time_eps=cfg.time_eps,
                interval_p_mean=cfg.interval_p_mean,
                interval_p_std=cfg.interval_p_std, ode_steps=cfg.eval_teacher_ode_steps,
            ).item()

            msg = (
                f"[AYF {cfg.train_mode}] step {step:05d} | "
                f"loss {loss.item():.6f} | "
                f"eval endpoint {ep:.6f} | "
                f"eval interval {itv:.6f}"
            )
            if loss_emd is not None:
                msg += f" | emd/ct {loss_emd.item():.6f}"
            if loss_lmd is not None:
                msg += f" | lmd {loss_lmd.item():.6f}"
            if loss_pair is not None:
                msg += f" | pair {loss_pair.item():.6f}"
            if loss_teacher_anchor is not None:
                msg += f" | teacher_anchor {loss_teacher_anchor.item():.6f}"
            print(msg)

    student.eval()
