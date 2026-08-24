"""Rectified-flow teacher training loop."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .losses import rf_teacher_loss
from .models import RFTeacher
from .sampling import NoiseSampler, sample_uniform_noise


def train_teacher(
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
    noise_sampler: NoiseSampler = sample_uniform_noise,
    label: str = "RF teacher",
) -> None:
    """Train an RF teacher using the base distribution supplied by the caller."""
    teacher.train()
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.teacher_lr)
    for step in range(1, cfg.teacher_steps + 1):
        loss = rf_teacher_loss(
            teacher,
            latent_sampler,
            cfg.batch_size,
            cfg.latent_dim,
            cfg.device,
            noise_sampler=noise_sampler,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            print(f"[{label}] step {step:05d} | loss {loss.item():.6f}")
    teacher.eval()
