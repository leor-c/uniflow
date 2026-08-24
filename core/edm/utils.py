"""EDM noise-level, normalized-time, and Karras-schedule utilities."""

from __future__ import annotations

import math

import torch


def edm_sample_log_sigma(
    batch_size: int,
    device: str | torch.device,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    sigma_min: float = 0.002,
    sigma_max: float = 2.0,
) -> torch.Tensor:
    """
    Sample log(sigma) from the log-normal training distribution (Karras Table 1).
    Returns [B, 1] clamped to [log(sigma_min), log(sigma_max)].
    """
    log_sigma = p_mean + p_std * torch.randn(batch_size, 1, device=device)
    return log_sigma.clamp(math.log(sigma_min), math.log(sigma_max))


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


def edm_time_to_log_sigma(
    time: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """Map normalized noise-to-data time ``[0, 1]`` to log-sigma."""
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError("Expected 0 < sigma_min < sigma_max")
    log_max = math.log(sigma_max)
    return log_max + time * (math.log(sigma_min) - log_max)


def edm_log_sigma_to_time(
    log_sigma: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """Map log-sigma to normalized noise-to-data time ``[0, 1]``."""
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError("Expected 0 < sigma_min < sigma_max")
    log_max = math.log(sigma_max)
    return (log_max - log_sigma) / (log_max - math.log(sigma_min))


def edm_dlog_sigma_dt(sigma_min: float, sigma_max: float) -> float:
    """Derivative of the linear log-sigma schedule used by EDM AYF."""
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError("Expected 0 < sigma_min < sigma_max")
    return math.log(sigma_min) - math.log(sigma_max)
