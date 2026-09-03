"""
Shared neural-network building blocks.

RF uses t=0 for noise and t=1 for data. EDM is parameterized by log-sigma.
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


# std(SiLU(x)) / std(x) for x ~ N(0, 1). PyTorch's default Linear/Conv2d init is
# kaiming_uniform with a=sqrt(5), calibrated for LeakyReLU(sqrt(5)), not SiLU.
# Combined with SiLU(x) = x*sigmoid(x) ~= x/2 for small x, that attenuates the
# forward signal at every layer and compounds with depth -- measured 25x from
# input to latent in the MNIST encoder, which drove tanh latents to |z|~0.01
# against the 0.5 that U([-1,1]) implies.
SILU_GAIN = 0.5595


def init_silu_(module: nn.Module, output_layers: tuple = ()) -> None:
    """Variance-preserving initialization for SiLU stacks.

    Sets weight std to ``1 / (SILU_GAIN * sqrt(fan_in))`` so each Linear/Conv2d
    followed by a SiLU preserves activation scale instead of shrinking it.

    This matters unevenly across the benchmark: the Gaussian and EDM arms
    rescale their latents post-hoc (``fit_latent_normalization``), so
    attenuation is silently corrected there, while the uniform arm's base is a
    fixed ``U([-1,1]^d)`` box with no such knob -- so the same bug lands
    entirely on it.
    """
    outputs = {id(layer) for layer in output_layers}
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            fan_in = nn.init._calculate_correct_fan(layer.weight, "fan_in")
            # Layers feeding a SiLU get the gain compensation; a final layer with
            # no activation after it would otherwise amplify by 1 / SILU_GAIN.
            gain = 1.0 if id(layer) in outputs else SILU_GAIN
            nn.init.normal_(layer.weight, 0.0, 1.0 / (gain * math.sqrt(fan_in)))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class RandomFourierFeatures(nn.Module):
    """Fixed random Fourier encoding of a coordinate input.

        gamma(u) = [u, sin(2*pi*(u/input_scale)B), cos(2*pi*(u/input_scale)B)]

    A plain MLP on a low-dimensional coordinate input has strong spectral bias
    and cannot represent sharp functions of it (Tancik et al., 2020). The 2D
    toy benchmark is exactly that regime: at ``latent_dim=2`` the rectified-flow
    velocity is near-discontinuous at the basin boundaries between modes, and
    without this encoding the teacher smooths across them and strands mass
    between the modes.

    ``B`` is fixed, not learned. The raw input is concatenated so the smooth,
    large-scale part of the target stays trivially representable.

    ``input_scale`` divides the input before projection, so one nominal
    ``scale`` means the same *effective* frequency regardless of how the arm
    scales its latents -- bounded latents sit near ``sqrt(1/3)`` while
    standardized ones sit near 1. Without it, a shared ``scale`` would favour
    whichever arm its magnitude happened to suit.
    """

    def __init__(
        self,
        in_dim: int,
        num_features: int = 64,
        scale: float = 1.0,
        input_scale: float = 1.0,
    ):
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be at least 1")
        if scale <= 0.0 or input_scale <= 0.0:
            raise ValueError("scale and input_scale must be strictly positive")
        self.in_dim = int(in_dim)
        # The projection u @ B sums in_dim terms, so its magnitude grows like
        # sqrt(in_dim). Without this factor a scale calibrated at one input
        # dimension aliases at higher ones -- the phases wrap, nearby inputs get
        # uncorrelated encodings and distant ones collide. Folding sqrt(in_dim)
        # into the normalization keeps the *effective* frequency, and hence the
        # kernel bandwidth, constant across dimensions.
        self.input_scale = float(input_scale) * math.sqrt(in_dim)
        self.register_buffer("B", torch.randn(in_dim, num_features) * scale)

    @property
    def dim(self) -> int:
        return self.in_dim + 2 * int(self.B.shape[1])

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        projection = 2 * math.pi * ((u / self.input_scale) @ self.B)
        return torch.cat([u, projection.sin(), projection.cos()], dim=-1)


FF_SITE_CODES = {"a": "ae", "t": "teacher", "s": "student"}
# Scales calibrated on the 2D toy benchmark; used only when a --ff spec turns on
# a site whose Config leaves it off (as MNIST does for all of them).
DEFAULT_FF_SCALES = {
    "ae": 0.5 * math.sqrt(2.0),
    "teacher": math.sqrt(2.0 / 3.0),
    "student": math.sqrt(2.0 / 3.0),
}


def parse_ff_sites(spec: str, *, allowed=("ae", "teacher", "student")) -> frozenset[str]:
    """Parse a Fourier-site spec into site names.

    Letters select sites: ``A`` encoder input, ``T`` teacher state, ``S``
    student state -- so ``ATS`` is all three, ``AT`` drops the student, ``S``
    is student-only. ``none`` / ``off`` / ``-`` disables every site, ``all`` is
    a synonym for every allowed site. Case-insensitive.

    Sites not named are switched OFF, so the spec fully determines the
    configuration rather than layering on top of whatever the defaults were.
    """
    text = spec.strip().lower()
    if text in {"none", "off", "-"}:
        return frozenset()
    if text == "all":
        return frozenset(allowed)
    sites = set()
    for char in text:
        site = FF_SITE_CODES.get(char)
        if site is None:
            raise ValueError(
                f"Unknown Fourier site {char!r} in {spec!r}: expected letters from "
                f"{sorted(FF_SITE_CODES)} (a=ae, t=teacher, s=student), "
                "or 'none'/'all'"
            )
        if site not in allowed:
            raise ValueError(
                f"Fourier site {site!r} is not available here; allowed: {sorted(allowed)}"
            )
        sites.add(site)
    if not sites:
        raise ValueError(f"Empty Fourier site spec {spec!r}; use 'none' to disable all")
    return frozenset(sites)


def apply_ff_spec(cfg, spec: str, *, allowed=("ae", "teacher", "student")) -> frozenset[str]:
    """Set ``cfg.<site>_ff_scale`` from a spec, switching unnamed sites off."""
    sites = parse_ff_sites(spec, allowed=allowed)
    for site in allowed:
        field = f"{site}_ff_scale"
        if site not in sites:
            setattr(cfg, field, None)
        elif getattr(cfg, field, None) is None:
            setattr(cfg, field, DEFAULT_FF_SCALES[site])
    return sites


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
        # init_silu_(self, output_layers=(self.net[-1],))

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
        init_silu_(self, output_layers=(self.out_proj,))
        # FiLM starts as the identity modulation (gamma = beta = 0), so the
        # conditioning ramps in during training rather than perturbing the
        # forward scale we just calibrated.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.film(cond).chunk(2 * self.depth, dim=-1)

        h = self.in_proj(x)
        h = self.act(h * (1 + gb[0]) + gb[1])
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = self.act(h * (1 + gb[2 * (i + 1)]) + gb[2 * (i + 1) + 1])
        return self.out_proj(h)
