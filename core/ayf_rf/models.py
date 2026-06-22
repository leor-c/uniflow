"""
AYF-style interval flow map model.

Generalizes core.rf_scm.RFConsistencyStudent's endpoint-only map
C(z, t) -> z_1 to an arbitrary-interval map C(z, t, s) -> z_s. The RF
teacher (core.rf_scm.RFTeacher) is reused unchanged; this module only adds
the interval-aware student.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..modules import FourierTimeEmbedding, MLP


class RFIntervalFlowMap(nn.Module):
    """
    AYF-style bounded-latent interval flow map.

        C_theta(z, t, s) = z + (s - t) f_theta(z, t, s)

    where t is the source time and s is the target time.

    Identity boundary: C_theta(z, t, t) = z.
    Endpoint-consistency special case: C_theta(z, t, 1).

    Output head is unbounded; clamp externally during sampling if desired.
    """

    def __init__(self, latent_dim: int, hidden: int = 128, depth: int = 4,
                 num_freq: int = 4, max_freq: float = 4.0):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(num_freq, max_freq)
        time_dim = 3 * self.time_embed.dim  # emb(t), emb(s), emb(s - t)
        self.mlp = MLP(latent_dim + time_dim, latent_dim, hidden=hidden, depth=depth)

    def forward(self, z: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        if s.ndim == 1:
            s = s[:, None]

        delta = s - t
        emb_t = self.time_embed(t)
        emb_s = self.time_embed(s)
        emb_delta = self.time_embed(delta.clamp_min(0.0))

        f = self.mlp(torch.cat([z, emb_t, emb_s, emb_delta], dim=-1))
        return z + delta * f
