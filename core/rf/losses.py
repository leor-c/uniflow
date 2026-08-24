"""Rectified-flow teacher objective."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from .models import RFTeacher
from .sampling import NoiseSampler, sample_uniform_noise


def rf_teacher_loss(
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    batch_size: int,
    latent_dim: int,
    device: str | torch.device,
    noise_sampler: NoiseSampler = sample_uniform_noise,
) -> torch.Tensor:
    """Regress the conditional RF velocity from independently paired endpoints."""
    with torch.no_grad():
        z1 = latent_sampler(batch_size)
    z0 = noise_sampler(batch_size, latent_dim, device)
    t = torch.rand(batch_size, 1, device=device)
    zt = (1.0 - t) * z0 + t * z1
    return F.mse_loss(teacher(zt, t), z1 - z0)
