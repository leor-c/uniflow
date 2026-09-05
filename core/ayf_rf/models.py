"""
AYF-style interval flow map model.

Implements an arbitrary-interval map ``C(z, t, s) -> z_s`` for a trained RF
velocity field.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..dit import make_backbone
from ..modules import FourierTimeEmbedding, RandomFourierFeatures


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
                 num_freq: int = 64, max_freq: float = 32.0,
                 ff_scale: float | None = None, ff_size: int = 64,
                 ff_input_scale: float = 1.0,
                 arch: str = "mlp",
                 latent_shape: tuple[int, int, int] | None = None,
                 heads: int = 6, mlp_ratio: float = 4.0, patch: int = 1):
        super().__init__()
        # The AYF-EMD tangent differentiates this map in t, and ``d_t F_theta`` is
        # the dominant term in it: measured on a trained CIFAR-10 student the
        # tangent ``dF/dt`` was 99.8% ``d_t F``, the transport part ``J_z F . v``
        # only ~0.038. FourierTimeEmbedding's ``deriv_cap`` bounds each sin/cos
        # pair rather than the whole embedding, so the aggregate is
        # ``deriv_cap * sqrt(num_freq)`` -- 16 at these settings, against an
        # ``||emb||`` of 0.5. Dividing by sqrt(num_freq) restores the nominal
        # bound and is what keeps the map-consistency term from overwhelming the
        # flow-matching term once the tangent warmup ``r`` reaches full strength.
        # Without it the MNIST and CIFAR-10 students collapsed to the identity map
        # a few hundred steps after the warmup capped; see README.md.
        #
        # Scoped to the student on purpose: the teacher is only ever *evaluated*
        # in the AYF losses (under no_grad), never differentiated in t, so it
        # gains nothing from a softer embedding and measurably loses the time
        # resolution it needs -- on the 2D toy benchmark softening the teacher too
        # cost ~0.1-0.2 nats of teacher target NLL.
        self.time_embed = FourierTimeEmbedding(
            num_freq, max_freq, deriv_cap=2.0 / math.sqrt(num_freq)
        )
        time_dim = 3 * self.time_embed.dim  # emb(t), emb(s), emb(s - t)
        # The EMD/LMD tangent contains J_z C, so this encoding is differentiated
        # through. On unbounded latents the sinusoids alias in the tails and the
        # map degrades badly there; enable it on bounded latents, or alongside
        # encoder-side features.
        self.state_features = (
            RandomFourierFeatures(latent_dim, ff_size, ff_scale, ff_input_scale)
            if ff_scale is not None else None
        )
        in_dim = latent_dim if self.state_features is None else self.state_features.dim
        self.mlp = make_backbone(
            in_dim, time_dim, latent_dim, hidden, depth,
            arch=arch, latent_shape=latent_shape,
            heads=heads, mlp_ratio=mlp_ratio, patch=patch,
        )

    def slope(self, z: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Predict the interval slope ``F_theta(z, t, s)`` used by AYF."""
        if t.ndim == 1:
            t = t[:, None]
        if s.ndim == 1:
            s = s[:, None]

        delta = s - t
        emb_t = self.time_embed(t)
        emb_s = self.time_embed(s)
        emb_delta = self.time_embed(delta.clamp_min(0.0))

        cond = torch.cat([emb_t, emb_s, emb_delta], dim=-1)
        h = z if self.state_features is None else self.state_features(z)
        return self.mlp(h, cond)

    def forward(self, z: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        if s.ndim == 1:
            s = s[:, None]
        return z + (s - t) * self.slope(z, t, s)
