"""Rectified-flow velocity model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..modules import FiLMMLP, FourierTimeEmbedding, RandomFourierFeatures


class RFTeacher(nn.Module):
    """Velocity field for ``z_t = (1 - t) z_0 + t z_1``."""

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 128,
        depth: int = 4,
        num_freq: int = 64,
        max_freq: float = 32.0,
        ff_scale: float | None = None,
        ff_size: int = 64,
        ff_input_scale: float = 1.0,
    ):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(num_freq, max_freq, log_scale=False)
        # Optional Fourier encoding of the state. Off by default; see
        # RandomFourierFeatures for why it matters at low latent_dim.
        self.state_features = (
            RandomFourierFeatures(latent_dim, ff_size, ff_scale, ff_input_scale)
            if ff_scale is not None else None
        )
        in_dim = latent_dim if self.state_features is None else self.state_features.dim
        self.mlp = FiLMMLP(
            in_dim,
            self.time_embed.dim,
            latent_dim,
            hidden=hidden,
            depth=depth,
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        h = z if self.state_features is None else self.state_features(z)
        return self.mlp(h, self.time_embed(t))
