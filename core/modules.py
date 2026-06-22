"""
Shared neural-network building blocks.

Convention note: the rf_scm / EDM modules use t=0 → noise, t=1 → data,
which is the OPPOSITE of core.interp (t=0 → data, t=1 → noise).
These modules are shared across both conventions.
"""

import math

import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    """
    Low-frequency sinusoidal time embedding.

    Deliberately avoids the 2^k frequency schedule used in NeRF / transformers.
    The JVP / consistency losses differentiate through this embedding, so large
    frequencies create extremely stiff gradients w.r.t. t and destabilise training.
    """

    def __init__(self, num_frequencies: int = 4, max_freq: float = 4.0):
        super().__init__()
        freqs = torch.linspace(1.0, max_freq, num_frequencies).float()
        self.register_buffer("freqs", freqs)

    @property
    def dim(self) -> int:
        return 2 * int(self.freqs.numel())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        x = 2 * math.pi * t * self.freqs[None, :]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class MLP(nn.Module):
    """Plain MLP with SiLU activations."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)