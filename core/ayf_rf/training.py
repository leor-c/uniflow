"""
AYF-style training loop:
    train_student_interval

Also hosts the shared ``train_mode`` grammar (``parse_train_mode`` /
``resolve_train_mode``) used by every arm, so "which AYF objective am I
training?" has one answer across ``core.ayf_rf``, ``core.ayf`` and the EDM arm.

The RF teacher is trained separately with ``core.rf.train_teacher``.
"""

from __future__ import annotations

from copy import copy
from typing import Callable

import torch
import torch.nn.functional as F

from ..lr_schedule import make_lr_scheduler

from ..rf import RFTeacher, sample_uniform_noise
from .models import RFIntervalFlowMap, VelocityView
from .sampling import sample_ayf_time_pair
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


def _train_student_interval_phase(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
    step_offset: int = 0,
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
    scheduler = make_lr_scheduler(opt, cfg, "student")

    warmup_steps = getattr(cfg, "tangent_warmup_steps", 0)
    warmup_max = getattr(cfg, "tangent_warmup_max", 0.99)

    for step in range(1, cfg.student_steps + 1):
        # AYF Algorithm 1 line 7: r = min(r_max, Iters / H). r scales only the
        # (s-t)dF/dt term, so training starts as flow matching and eases the
        # map-consistency term in. warmup_steps = 0 disables it (r = 1).
        r = (1.0 if warmup_steps <= 0
             else min(warmup_max, (step_offset + step - 1) / float(warmup_steps)))
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
        if scheduler is not None:
            scheduler.step()

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


def train_student_interval(
    student: RFIntervalFlowMap,
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
) -> None:
    """Train the interval map with an optional, separately counted refinement.

    AdamW is restarted for refinement, while tangent warmup continues from the
    initial phase. No teacher parameters are updated by either student phase.
    """
    refine_steps = getattr(cfg, "student_refine_steps", 0)
    refine_lr = getattr(cfg, "student_refine_lr", 1e-4)
    if refine_steps < 0 or (refine_steps and refine_lr <= 0):
        raise ValueError("Student refinement needs nonnegative steps and a positive LR")
    _train_student_interval_phase(student, teacher, latent_sampler, cfg)
    if refine_steps:
        print(f"\nRefining AYF student: {refine_steps} steps at LR {refine_lr:g}")
        refined = copy(cfg)
        refined.student_steps = refine_steps
        refined.student_lr = refine_lr
        refined.student_lr_drop_step = None
        _train_student_interval_phase(
            student, teacher, latent_sampler, refined, step_offset=cfg.student_steps
        )


def train_map_teacher(
    flow_map: RFIntervalFlowMap,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
    noise_sampler=None,
    label: str = "map teacher",
) -> VelocityView:
    """Train an interval flow map *as the RF teacher*, and return its velocity view.

    The RF velocity target ``z1 - z0`` is already the ``s = 1`` slice of the
    Euler slope (see :class:`core.ayf_rf.models.VelocityView`), so regressing
    the slope onto it produces an ordinary RF teacher that happens to be an
    ``RFIntervalFlowMap``. Teacher and student then share one class, and the
    student can be initialized with ``student.load_state_dict(
    flow_map.state_dict())`` -- every tensor copies, including the conditioning
    projection that a separate ``RFTeacher`` cannot supply.

    ``s`` is *sampled* from the student's own interval schedule rather than
    pinned to 1. The velocity target does not depend on ``s``, so every slice
    learns the same field and the student starts as "Euler step with the
    teacher's velocity" across the whole ``(t, s)`` domain. Pinning ``s = 1``
    instead leaves ``emb(s)`` constant and ``emb(s - t) = emb(1 - t)`` a clean
    function of ``t``, which lets the teacher route time information through the
    delta channel -- information that becomes wrong the moment the student
    varies ``s``. Measured on the 2D toy benchmark at 80k student steps, a
    student copied from an ``s``-pinned teacher reached -1.480 nats 1-step
    target NLL against -1.675 for a sampled-``s`` teacher, the former losing
    even to a randomly initialized student off the shipped ``RFTeacher``.
    """
    if noise_sampler is None:
        noise_sampler = sample_uniform_noise

    flow_map.train()
    opt = torch.optim.AdamW(flow_map.parameters(), lr=cfg.teacher_lr)
    scheduler = make_lr_scheduler(opt, cfg, "teacher")
    for step in range(1, cfg.teacher_steps + 1):
        with torch.no_grad():
            z1 = latent_sampler(cfg.batch_size)
        z0 = noise_sampler(cfg.batch_size, cfg.latent_dim, cfg.device)
        t = torch.rand(cfg.batch_size, 1, device=cfg.device)
        zt = (1.0 - t) * z0 + t * z1
        # Source time is uniform (plain RF); only the target time follows the
        # AYF interval schedule, so the slices the student will query are the
        # slices that get trained.
        _, s = sample_ayf_time_pair(
            cfg.batch_size, cfg.device, eps=cfg.time_eps,
            p_mean=cfg.interval_p_mean, p_std=cfg.interval_p_std,
        )
        loss = F.mse_loss(flow_map.slope(zt, t, s), z1 - z0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        if step % cfg.log_every == 0:
            print(f"[{label}] step {step:05d} | loss {loss.item():.6f}")

    flow_map.eval()
    return VelocityView(flow_map)
