"""
Diffusion Transformer (Peebles & Xie, 2023) as a drop-in replacement for
:class:`core.modules.FiLMMLP`.

Why a transformer here
----------------------
The MNIST arms run the flow in a *learned* latent space, and until now the
denoiser was an MLP on a flat latent vector. That is not the architecture any
real latent generative model uses, so the benchmark's architecture term was not
representative. The two realistic choices are a UNet and a DiT; at the 7x7
latent grid this AE produces, a UNet has essentially no room to downsample
(7 -> 4 -> 2, with odd-size padding bookkeeping the whole way) so its entire
multi-resolution inductive bias is wasted. A DiT with ``patch=1`` turns the same
grid into 49 tokens and lets every layer attend globally. It is also what
current latent generative models actually use.

Interface
---------
``DiT`` deliberately mirrors ``FiLMMLP``: ``forward(x, cond)`` with a flat
``(B, in_dim)`` state and a ``(B, cond_dim)`` conditioning vector, returning
``(B, out_dim)``. The spatial structure lives entirely inside this module -- it
reshapes the flat latent to ``(B, C, H, W)``, tokenizes, and flattens back on
the way out. Every loss, noise sampler, ODE integrator and metric in the
codebase therefore keeps treating latents as flat vectors of size ``C*H*W``,
and swapping architectures is a one-line change at each model class.

Two details are specific to this codebase
-----------------------------------------
1. **Attention is written out by hand rather than calling
   ``F.scaled_dot_product_attention``.** The AYF / consistency / EDM-student
   losses differentiate the network in *forward* mode via ``torch.func.jvp``
   (see ``core/ayf/losses.py``, ``core/ayf_rf/losses.py``,
   ``core/edm/losses.py``). SDPA dispatches to fused flash / mem-efficient
   kernels that have no forward-mode AD rule, so a JVP through them fails.
   The explicit ``softmax(QK^T / sqrt(d)) V`` has one, and at 49 tokens the
   fused kernels would buy nothing anyway.

2. **The time signal keeps using**
   :class:`core.modules.FourierTimeEmbedding`. Reference DiT feeds timesteps
   through the usual ``10000^(2i/d)`` sinusoidal schedule; that is precisely the
   high-frequency schedule ``FourierTimeEmbedding`` documents itself as avoiding,
   because the JVP losses differentiate through the embedding and large
   frequencies make the gradient w.r.t. ``t`` extremely stiff. The embedding
   stays the caller's responsibility -- this module only sees ``cond``.

Conditioning is adaLN-Zero, the same identity-at-initialization idea the
existing ``FiLMMLP`` already uses (it zero-inits its FiLM projection), so the
conditioning ramps in during training instead of perturbing a freshly
initialized forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Multi-head self-attention, written out for forward-mode AD.

    Deliberately does not call ``F.scaled_dot_product_attention``: the fused
    backends it selects have no ``torch.func.jvp`` rule, and the flow-map losses
    in this repo differentiate the network in forward mode.
    """

    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim)
        # -> (3, B, heads, tokens, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = attn.softmax(dim=-1) @ v
        out = out.transpose(1, 2).reshape(batch, tokens, width)
        return self.proj(out)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    Both residual branches are gated by ``alpha`` from a zero-initialized
    projection, so at initialization the block is exactly the identity.
    """

    def __init__(self, width: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # Affine parameters come from the conditioning vector instead, so the
        # norms themselves are parameter-free.
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(width, heads)
        hidden = int(mlp_ratio * width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, width),
        )
        self.modulation = nn.Linear(width, 6 * width)
        # adaLN-Zero: identity at init (shift = scale = 0 -> plain LayerNorm,
        # alpha = 0 -> both residual branches contribute nothing).
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # (B, 6*width) -> six (B, 1, width) tensors that broadcast over tokens.
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.modulation(cond)[:, None, :].chunk(6, dim=-1)
        )
        h = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn(h)
        h = self.norm2(x) * (1 + scale_m) + shift_m
        return x + gate_m * self.mlp(h)


class DiT(nn.Module):
    """DiT backbone with the :class:`core.modules.FiLMMLP` call signature.

    Parameters mirror ``FiLMMLP(in_dim, cond_dim, out_dim, hidden, depth)`` so
    the model classes can switch on a config flag. ``in_dim`` and ``out_dim``
    must both equal ``prod(latent_shape)``: the state is a flat latent vector
    that this module reshapes internally.

    ``hidden`` is the transformer width and ``depth`` the number of blocks, so
    the existing ``cfg.hidden`` / ``cfg.depth`` knobs keep their meaning even
    though the parameter counts they imply are of course different.
    """

    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        out_dim: int,
        latent_shape: tuple[int, int, int],
        hidden: int = 192,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        patch: int = 1,
    ):
        super().__init__()
        channels, height, width_px = latent_shape
        expected = channels * height * width_px
        if in_dim != expected or out_dim != expected:
            raise ValueError(
                f"DiT expects flat latents matching latent_shape {latent_shape} "
                f"(= {expected} dims), got in_dim={in_dim}, out_dim={out_dim}. "
                "Note that Fourier state features change in_dim and are not "
                "compatible with a spatial backbone."
            )
        if height % patch != 0 or width_px % patch != 0:
            raise ValueError(
                f"patch {patch} does not divide the latent grid {height}x{width_px}"
            )

        self.latent_shape = (channels, height, width_px)
        self.patch = patch
        self.grid = (height // patch, width_px // patch)
        self.num_tokens = self.grid[0] * self.grid[1]
        self.patch_dim = channels * patch * patch

        self.patch_embed = nn.Linear(self.patch_dim, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Conditioning vector (already a Fourier time embedding upstream) is
        # projected once and shared by every block's adaLN.
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.blocks = nn.ModuleList(
            DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth)
        )

        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Linear(hidden, 2 * hidden)
        self.final_proj = nn.Linear(hidden, self.patch_dim)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        # Standard DiT: the network outputs exactly zero at initialization.
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        self._init_transformer()

    def _init_transformer(self) -> None:
        """Xavier-uniform on the transformer's own Linears.

        ``core.modules.init_silu_`` is not used here: it compensates for SiLU
        attenuation compounding through an unnormalized stack, and every branch
        in this network is preceded by a LayerNorm, which fixes the activation
        scale directly. The zero-initialized modulation and output layers above
        are applied after this and must not be overwritten.
        """
        zeroed = {id(self.final_proj)}
        for block in self.blocks:
            zeroed.add(id(block.modulation))
        zeroed.add(id(self.final_modulation))
        for module in self.modules():
            if isinstance(module, nn.Linear) and id(module) not in zeroed:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def patchify(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, C*H*W)`` flat latent -> ``(B, tokens, patch_dim)``."""
        channels, height, width_px = self.latent_shape
        p = self.patch
        x = z.view(-1, channels, height, width_px)
        # (B, C, gh, p, gw, p) -> (B, gh, gw, C, p, p) -> (B, tokens, C*p*p)
        x = x.view(-1, channels, self.grid[0], p, self.grid[1], p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, self.num_tokens, self.patch_dim)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, tokens, patch_dim)`` -> ``(B, C*H*W)`` flat latent."""
        channels, height, width_px = self.latent_shape
        p = self.patch
        z = x.view(-1, self.grid[0], self.grid[1], channels, p, p)
        z = z.permute(0, 3, 1, 4, 2, 5).reshape(-1, channels * height * width_px)
        return z

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(self.patchify(x)) + self.pos_embed
        c = self.cond_proj(cond)
        for block in self.blocks:
            h = block(h, c)

        shift, scale = self.final_modulation(c)[:, None, :].chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale) + shift
        return self.unpatchify(self.final_proj(h))


def infer_latent_shape(latent_dim: int, channels: int, grid: int = 7) -> tuple[int, int, int]:
    """Validate that ``latent_dim`` is a ``channels x grid x grid`` map."""
    expected = channels * grid * grid
    if latent_dim != expected:
        raise ValueError(
            f"latent_dim {latent_dim} is not {channels}x{grid}x{grid} = {expected}; "
            "a spatial backbone needs the latent laid out as a feature map"
        )
    return (channels, grid, grid)


def make_backbone(
    in_dim: int,
    cond_dim: int,
    out_dim: int,
    hidden: int,
    depth: int,
    *,
    arch: str = "mlp",
    latent_shape: tuple[int, int, int] | None = None,
    heads: int = 6,
    mlp_ratio: float = 4.0,
    patch: int = 1,
) -> nn.Module:
    """Build the state/conditioning backbone shared by every model class.

    ``arch="mlp"`` returns the original :class:`core.modules.FiLMMLP`;
    ``arch="dit"`` returns a :class:`DiT` over ``latent_shape``. Both expose
    ``forward(x, cond)``, so the model classes differ only in this call.
    """
    from .modules import FiLMMLP

    if arch == "mlp":
        return FiLMMLP(in_dim, cond_dim, out_dim, hidden=hidden, depth=depth)
    if arch == "dit":
        if latent_shape is None:
            raise ValueError("arch='dit' requires latent_shape")
        return DiT(
            in_dim, cond_dim, out_dim, latent_shape,
            hidden=hidden, depth=depth, heads=heads,
            mlp_ratio=mlp_ratio, patch=patch,
        )
    raise ValueError(f"Unknown arch {arch!r}: expected 'mlp' or 'dit'")
