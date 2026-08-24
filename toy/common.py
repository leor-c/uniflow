"""Shared data, autoencoder, evaluation, and plotting helpers for toy runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules import MLP
from toy.datasets import get_sampler


def sample_8gaussians(
    batch_size: int,
    device: str | torch.device,
    radius: float = 0.75,
    std: float = 0.035,
) -> torch.Tensor:
    """Eight isotropic Gaussian modes arranged on a circle."""
    k = torch.randint(0, 8, (batch_size,), device=device)
    angles = 2 * math.pi * k.float() / 8.0
    centers = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1
    )
    return centers + std * torch.randn(batch_size, 2, device=device)


class ToyAE(nn.Module):
    """Two-dimensional autoencoder with an optional tanh latent boundary."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden: int = 128,
        depth: int = 2,
        bounded: bool = True,
    ):
        super().__init__()
        self.encoder_net = MLP(2, latent_dim, hidden=hidden, depth=depth)
        self.decoder_net = MLP(latent_dim, 2, hidden=hidden, depth=depth)
        self.bounded = bounded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_net(x)
        return torch.tanh(z) if self.bounded else z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def _latent_moment_regularizer(z: torch.Tensor) -> torch.Tensor:
    mean_loss = z.mean(dim=0).pow(2).mean()
    second_moment_loss = (z.pow(2).mean(dim=0) - 1.0 / 3.0).pow(2).mean()
    return mean_loss + second_moment_loss


def train_toy_ae(ae: ToyAE, data_sampler, cfg) -> None:
    ae.train()
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.ae_lr)
    for step in range(1, cfg.ae_steps + 1):
        x = data_sampler(cfg.batch_size)
        x_hat, z = ae(x)
        recon_loss = F.mse_loss(x_hat, x)
        reg_loss = _latent_moment_regularizer(z)
        loss = recon_loss + cfg.latent_reg_weight * reg_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            print(
                f"[AE] step {step:05d} | recon {recon_loss.item():.6f} | "
                f"regularizer {reg_loss.item():.6f}"
            )
    ae.eval()


def make_data_sampler(dataset: str, device: str | torch.device):
    if dataset == "8gaussians":
        return lambda n: sample_8gaussians(n, device)
    return get_sampler(dataset, device=device)


def make_latent_sampler(ae: ToyAE, data_sampler, device: str | torch.device):
    @torch.no_grad()
    def sampler(batch_size: int) -> torch.Tensor:
        return ae.encode(data_sampler(batch_size).to(device))

    return sampler


@torch.no_grad()
def sliced_wasserstein_distance(
    real: torch.Tensor,
    generated: torch.Tensor,
    projections: int = 256,
) -> float:
    """Mean one-dimensional W2 distance across fixed random projections."""
    n = min(len(real), len(generated))
    real = real[:n].float()
    generated = generated[:n].float()
    generator = torch.Generator(device=real.device).manual_seed(0)
    directions = torch.randn(
        real.shape[1], projections, generator=generator, device=real.device
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    real_proj = (real @ directions).sort(dim=0).values
    generated_proj = (generated @ directions).sort(dim=0).values
    return (real_proj - generated_proj).pow(2).mean().sqrt().item()


def save_benchmark_figure(
    panels: list[tuple[str, torch.Tensor]],
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.6 * len(panels), 4.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, (title, pts) in zip(axes, panels):
        pts = pts.detach().cpu()
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.45)
        ax.set_title(title, pad=10)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.3)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved figure -> {path}")


def save_metrics(metrics: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"Saved metrics -> {path}")
