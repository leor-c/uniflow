"""Rectified-flow velocity model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..modules import FiLMMLP, FourierTimeEmbedding


class RFTeacher(nn.Module):
    """Velocity field for ``z_t = (1 - t) z_0 + t z_1``."""

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 128,
        depth: int = 4,
        num_freq: int = 64,
        max_freq: float = 32.0,
    ):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(num_freq, max_freq, log_scale=False)
        self.mlp = FiLMMLP(
            latent_dim,
            self.time_embed.dim,
            latent_dim,
            hidden=hidden,
            depth=depth,
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.mlp(z, self.time_embed(t))
