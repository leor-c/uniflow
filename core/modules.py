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

    def __init__(self, num_frequencies: int = 4, max_freq: float = 4.0, log_scale: bool = False, deriv_cap: float = 2.0):
        super().__init__()
        freqs = torch.linspace(1.0, max_freq, num_frequencies).float()
        if log_scale:
            freqs = freqs.exp()

        omega = 2 * math.pi * freqs
        amp = torch.clamp(deriv_cap / omega, max=1.0)

        self.register_buffer("freqs", freqs)
        self.register_buffer("amp", amp)

    @property
    def dim(self) -> int:
        return 2 * int(self.freqs.numel())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        x = 2 * math.pi * t * self.freqs[None, :]

        sin = torch.sin(x) * self.amp[None, :]
        cos = torch.cos(x) * self.amp[None, :]
        return torch.cat([sin, cos], dim=-1)


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


class FiLMMLP(nn.Module):
    """
    MLP with FiLM-style per-layer conditioning (e.g. on a time embedding).

    Each hidden layer's activation h is modulated as:
        h = h * (1 + gamma_l(cond)) + beta_l(cond)
    where (gamma_l, beta_l) for every layer l come from a single linear
    projection of the conditioning vector, computed once per forward pass.

    Unlike concatenation-based conditioning (cond appended to x), FiLM lets
    the conditioning signal rescale/shift features at every depth instead of
    only at the input, which usually gives the time signal more influence in
    deeper layers without growing the input width.
    """

    def __init__(self, in_dim: int, cond_dim: int, out_dim: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.depth = depth
        self.in_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(depth - 1))
        self.film = nn.Linear(cond_dim, 2 * hidden * depth)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.film(cond).chunk(2 * self.depth, dim=-1)

        h = self.in_proj(x)
        h = self.act(h * (1 + gb[0]) + gb[1])
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = self.act(h * (1 + gb[2 * (i + 1)]) + gb[2 * (i + 1) + 1])
        return self.out_proj(h)