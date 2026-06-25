"""
RF-sCM models: the teacher velocity field and the consistency student.

Convention (t=0 → noise, t=1 → data):
    z0 ~ Uniform([-1,1]^d)
    z1 = encoder(x)        (data latent, bounded by tanh)
    z_t = (1 - t) * z0 + t * z1    t in [0, 1]
    teacher velocity target: v = z1 - z0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..modules import FourierTimeEmbedding, FiLMMLP


class RFTeacher(nn.Module):
    """
    Latent-space RF velocity field v_phi(z, t).

    Regresses v = z1 - z0 along the straight-line path z_t = (1-t)z0 + t*z1.
    Input convention: t=0 is noise, t=1 is data.

    Time conditioning is FiLM-style (per-layer scale/shift from the time
    embedding) rather than concatenation.
    """

    def __init__(self, latent_dim: int, hidden: int = 128, depth: int = 4,
                 num_freq: int = 64, max_freq: float = 32.0):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(num_freq, max_freq, log_scale=False)
        self.mlp = FiLMMLP(latent_dim, self.time_embed.dim, latent_dim, hidden=hidden, depth=depth)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        emb = self.time_embed(t)
        return self.mlp(z, emb)


class RFConsistencyStudent(nn.Module):
    """
    Latent-space RF consistency model.

    Parameterization (boundary-respecting):
        C(z, t) = t * z + (1 - t) * tanh(h_theta(z, t))

    This enforces:
        C(z, 1) = z          (data boundary: at t=1 the input IS the data latent)
        output stays bounded when z is bounded (tanh keeps the free part in (-1,1))
    """

    def __init__(self, latent_dim: int, hidden: int = 128, depth: int = 4,
                 num_freq: int = 64, max_freq: float = 32.0):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(num_freq, max_freq)
        self.mlp = FiLMMLP(latent_dim, self.time_embed.dim, latent_dim, hidden=hidden, depth=depth)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        emb = self.time_embed(t)
        h = self.mlp(z, emb)
        return t * z + (1.0 - t) * torch.tanh(h)
