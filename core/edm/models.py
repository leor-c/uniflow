"""
EDM model: the preconditioned denoiser D(z_noisy, log_sigma) -> z_clean.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..modules import FourierTimeEmbedding, FiLMMLP


class EDMPrecondDenoiser(nn.Module):
    """
    EDM-preconditioned denoiser D(z_noisy, log_sigma) → z_clean.

    Preconditioning (Karras et al. 2022, Table 1):
        c_skip  = sigma_data^2 / (sigma^2 + sigma_data^2)
        c_out   = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
        c_in    = 1 / sqrt(sigma^2 + sigma_data^2)
        c_noise = 0.25 * log_sigma      [input to time embedding]

    Boundary: as sigma → 0, D(z, sigma) → z.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 128,
        depth: int = 4,
        sigma_data: float = 0.5,
        num_freq: int = 64,
        max_freq: float = 32.0,
    ):
        super().__init__()
        self.sigma_data = float(sigma_data)
        self.noise_embed = FourierTimeEmbedding(num_freq, max_freq)
        self.mlp = FiLMMLP(latent_dim, self.noise_embed.dim, latent_dim, hidden=hidden, depth=depth)

    def forward(self, z: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        if log_sigma.ndim == 1:
            log_sigma = log_sigma[:, None]

        sigma = log_sigma.exp()
        sd = self.sigma_data
        s2 = sigma.pow(2)
        sd2 = sd ** 2

        c_skip = sd2 / (s2 + sd2)
        c_out = sigma * sd / torch.sqrt(s2 + sd2)
        c_in = 1.0 / torch.sqrt(s2 + sd2)
        c_noise = 0.25 * log_sigma

        emb = self.noise_embed(c_noise)
        h = self.mlp(c_in * z, emb)
        return c_skip * z + c_out * h
