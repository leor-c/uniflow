"""EDM noise-level and Karras-schedule utilities."""

from __future__ import annotations

import torch


def edm_sample_log_sigma(
    batch_size: int,
    device: str | torch.device,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    """Sample the untruncated EDM log-normal training distribution."""
    return p_mean + p_std * torch.randn(batch_size, 1, device=device)


def edm_karras_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Karras rho-schedule: decreasing sigmas from sigma_max to sigma_min."""
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    return (max_inv + ramp * (min_inv - max_inv)) ** rho


