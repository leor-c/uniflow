"""EDM teacher and endpoint-consistency parameterizations."""

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
        c_skip, c_out, c_in = self._scalings(sigma)
        return c_skip * z + c_out * self.network_residual(c_in * z, log_sigma)

    def network_residual(
            self, scaled_input: torch.Tensor, log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Raw network output ``F`` for an already-preconditioned input ``c_in * z``.

        Exposed so :class:`core.edm.flow.EDMFlowVelocity` can rebuild the
        denoiser from algebraically stabilised coefficients instead of forming
        the raw ``z / t`` and ``sigma = (1 - t) / t``, which overflow as
        ``t -> 0``.
        """
        if log_sigma.ndim == 1:
            log_sigma = log_sigma[:, None]
        return self.mlp(scaled_input, self.noise_embed(0.25 * log_sigma))

    def _scalings(
            self, sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard EDM denoiser preconditioning."""
        sd = self.sigma_data
        denominator = sigma.square() + sd ** 2
        c_skip = sd ** 2 / denominator
        c_out = sigma * sd / torch.sqrt(denominator)
        c_in = 1.0 / torch.sqrt(denominator)
        return c_skip, c_out, c_in


class EDMConsistencyModel(EDMPrecondDenoiser):
    """Consistency map with the exact boundary ``C(x, sigma_min) = x``."""

    def __init__(
            self,
            latent_dim: int,
            hidden: int = 128,
            depth: int = 4,
            sigma_data: float = 0.5,
            sigma_min: float = 0.002,
            num_freq: int = 64,
            max_freq: float = 32.0,
    ):
        super().__init__(
            latent_dim=latent_dim,
            hidden=hidden,
            depth=depth,
            sigma_data=sigma_data,
            num_freq=num_freq,
            max_freq=max_freq,
        )
        self.sigma_min = float(sigma_min)

    def _scalings(
            self, sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sd = self.sigma_data
        delta = sigma - self.sigma_min
        c_skip = sd ** 2 / (delta.square() + sd ** 2)
        c_out = delta * sd / torch.sqrt(sigma.square() + sd ** 2)
        c_in = 1.0 / torch.sqrt(sigma.square() + sd ** 2)
        return c_skip, c_out, c_in


