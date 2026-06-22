"""
Simple autoencoder that maps (unbounded) data to BOUNDED latents and back.

  Encoder E: R^{data_dim} -> (-1,1)^{latent_dim}   (final activation: tanh)
  Decoder D: (-1,1)^{latent_dim} -> R^{data_dim}

The flow (teacher + student) operates ENTIRELY in latent space, where the box
[-1,1]^{latent_dim} matches the uniform-noise prior. The data itself is NOT
tanh'd; boundedness comes only from the encoder's tanh, exactly as in the real
pipeline (tanh-bottleneck encoder upstream of the flow).

After training, E and D are FROZEN and wired (frozen) into the flow pipeline.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """R^{data_dim} -> (-1,1)^{latent_dim}, bounded by tanh. Plain residual blocks."""

    def __init__(self, data_dim, latent_dim, hidden=128, depth=2):
        super().__init__()
        self.in_proj = nn.Linear(data_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(depth)
        ])
        self.head = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        h = self.in_proj(x)
        for lin in self.blocks:
            h = h + lin(nn.functional.silu(h))
        return torch.tanh(self.head(h))  # latents in (-1,1)^d


class Decoder(nn.Module):
    """(-1,1)^{latent_dim} -> R^{data_dim} (unbounded output). Plain residual blocks."""

    def __init__(self, data_dim, latent_dim, hidden=128, depth=2):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(depth)
        ])
        self.head = nn.Linear(hidden, data_dim)

    def forward(self, z):
        h = self.in_proj(z)
        for lin in self.blocks:
            h = h + lin(nn.functional.silu(h))
        return self.head(h)


class AutoEncoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden=128, depth=2):
        super().__init__()
        self.encoder = Encoder(data_dim, latent_dim, hidden, depth)
        self.decoder = Decoder(data_dim, latent_dim, hidden, depth)
        self.latent_dim = latent_dim
        self.data_dim = data_dim

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_autoencoder(ae, data_sampler, iters, batch, lr, device,
                      latent_spread_weight=0.0, log_every=1000):
    """
    Train AE with reconstruction MSE. Optional `latent_spread_weight` adds a mild
    penalty encouraging the latent code to USE the box (avoid collapsing to a tiny
    region), which keeps the latent distribution non-degenerate w.r.t. the
    uniform prior. Default 0 (pure reconstruction).
    """
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    ae.train()
    for it in range(iters):
        x = data_sampler(batch)
        opt.zero_grad(set_to_none=True)
        x_hat, z = ae(x)
        recon = ((x_hat - x) ** 2).mean()
        loss = recon
        if latent_spread_weight > 0:
            # encourage per-dim latent variance toward that of Uniform(-1,1)=1/3,
            # a gentle anti-collapse term (NOT a hard constraint).
            var = z.var(dim=0)
            loss = loss + latent_spread_weight * ((var - 1.0 / 3.0) ** 2).mean()
        loss.backward()
        opt.step()
        if it % log_every == 0 or it == iters - 1:
            print(f"[ae] iter {it:5d}  recon {recon.item():.5f}")
    ae.eval()
    return ae


def freeze(module):
    """Freeze a module in-place: eval mode + no grad. Returns it for chaining."""
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


@torch.no_grad()
def latent_stats(encoder, data_sampler, n=4000, device=None):
    """Diagnostics: how the encoded data occupies the box."""
    x = data_sampler(n)
    z = encoder(x)
    return {
        "latent_dim": z.shape[1],
        "per_dim_std": z.std(dim=0).tolist(),
        "abs_max": z.abs().max().item(),
        "frac_saturated_0.99": (z.abs() > 0.99).float().mean().item(),
        "mean_abs": z.abs().mean().item(),
    }
