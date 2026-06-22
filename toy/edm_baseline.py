"""
EDM baseline toy experiment — 2D datasets (8-Gaussians, two-moons, checkerboard).

Same data sampler, AE architecture, and AE training as the original standalone
edm_baseline.py so that results are reproducible. All EDM algorithm
logic is imported from core.edm.

Usage:
    python -m urfm.toy.edm_baseline_8gaussians
    python -m urfm.toy.edm_baseline_8gaussians --dataset two_moons
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from core import edm as E
from core.modules import MLP
from toy.datasets import get_sampler


# ---------------------------------------------------------------------------
# Data  (original params preserved exactly)
# ---------------------------------------------------------------------------

def sample_8gaussians(batch_size: int, device, radius: float = 0.75, std: float = 0.035):
    k = torch.randint(0, 8, (batch_size,), device=device)
    angles = 2 * math.pi * k.float() / 8.0
    centers = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1
    )
    return centers + std * torch.randn(batch_size, 2, device=device)


# ---------------------------------------------------------------------------
# Toy AE  (original architecture: plain MLP, tanh latent)
# ---------------------------------------------------------------------------

class ToyAE(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden: int = 128, depth: int = 2):
        super().__init__()
        self.encoder_net = MLP(2, latent_dim, hidden=hidden, depth=depth)
        self.decoder_net = MLP(latent_dim, 2, hidden=hidden, depth=depth)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder_net(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


def _latent_moment_regularizer(z: torch.Tensor) -> torch.Tensor:
    mean_loss = z.mean(dim=0).pow(2).mean()
    second_moment_loss = (z.pow(2).mean(dim=0) - 1.0 / 3.0).pow(2).mean()
    return mean_loss + second_moment_loss


def train_toy_ae(ae: ToyAE, data_sampler, cfg: Config) -> None:
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
                f"[ae] step {step:05d} | recon {recon_loss.item():.6f} | "
                f"reg {reg_loss.item():.6f}"
            )
    ae.eval()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    dataset: str = "8gaussians"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 2

    # AE
    latent_dim: int = 32
    hidden: int = 128
    ae_depth: int = 2
    ae_steps: int = 5000
    ae_lr: float = 1e-3
    latent_reg_weight: float = 1e-3

    # EDM teacher
    depth: int = 4
    edm_teacher_steps: int = 10000
    edm_teacher_lr: float = 1e-3

    # EDM noise
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 2.0
    edm_rho: float = 7.0
    edm_p_mean: float = -1.2
    edm_p_std: float = 1.2

    # EDM student
    edm_student_steps: int = 10000
    edm_student_lr: float = 1e-3
    edm_jvp_dlog_ref: float = 1.0
    train_edm_student_mode: str = "jvp"   # "jvp" | "hybrid"
    edm_endpoint_every: int = 32
    edm_jvp_weight: float = 1.0
    edm_anchor_jvp_weight: float = 0.1
    edm_endpoint_ode_steps: int = 32

    # Sampling / visualisation
    batch_size: int = 64
    edm_teacher_sample_steps: int = 64
    edm_student_sample_k: int = 8

    log_every: int = 500
    out: str = "edm_baseline_toy.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data_sampler(dataset: str, device):
    if dataset == "8gaussians":
        return lambda n: sample_8gaussians(n, device)
    return get_sampler(dataset, device=device)


def _make_latent_sampler(ae: ToyAE, data_sampler, device):
    @torch.no_grad()
    def sampler(batch_size: int) -> torch.Tensor:
        x = data_sampler(batch_size)
        return ae.encode(x.to(device))
    return sampler


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_plot(ae: ToyAE, teacher, student, data_sampler, cfg: Config):
    n = 5000
    data = data_sampler(n).cpu()

    x_real = data_sampler(n).to(cfg.device)
    x_hat = ae.decode(ae.encode(x_real)).cpu()

    teacher_z = E.sample_edm_teacher_latents(
        teacher, n, cfg.latent_dim, cfg.device,
        sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max,
        rho=cfg.edm_rho, steps=cfg.edm_teacher_sample_steps,
    )
    teacher_samples = ae.decode(teacher_z).cpu()

    s1_z = E.sample_edm_student_latents_one_step(
        student, n, cfg.latent_dim, cfg.device, sigma_max=cfg.sigma_max
    )
    s1 = ae.decode(s1_z).cpu()

    sk_z = E.sample_edm_student_latents_k_step(
        student, n, cfg.latent_dim, cfg.device,
        k=cfg.edm_student_sample_k,
        sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max, rho=cfg.edm_rho,
    )
    sk = ae.decode(sk_z).cpu()

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharex=True, sharey=True)
    panels = [
        ("Data", data),
        ("AE recon", x_hat),
        ("EDM teacher (ODE)", teacher_samples),
        ("EDM student 1-step", s1),
        (f"EDM student {cfg.edm_student_sample_k}-step", sk),
    ]
    for ax, (title, pts) in zip(axes, panels):
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.45)
        ax.set_title(title)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(cfg.out, dpi=160)
    print(f"Saved → {cfg.out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config | None = None):
    if cfg is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--dataset", default="8gaussians",
                        choices=["8gaussians", "two_moons", "checkerboard"])
        ap.add_argument("--latent_dim", type=int, default=32)
        ap.add_argument("--teacher_steps", type=int, default=10000)
        ap.add_argument("--student_steps", type=int, default=10000)
        ap.add_argument("--student_mode", default="jvp", choices=["jvp", "hybrid"])
        ap.add_argument("--seed", type=int, default=2)
        ap.add_argument("--out", default=None)
        args = ap.parse_args()
        cfg = Config(
            dataset=args.dataset,
            latent_dim=args.latent_dim,
            edm_teacher_steps=args.teacher_steps,
            edm_student_steps=args.student_steps,
            train_edm_student_mode=args.student_mode,
            seed=args.seed,
            out=args.out or f"edm_baseline_{args.dataset}.png",
        )

    print(cfg)
    torch.manual_seed(cfg.seed)

    data_sampler = _make_data_sampler(cfg.dataset, cfg.device)

    # AE (original architecture)
    ae = ToyAE(
        latent_dim=cfg.latent_dim,
        hidden=cfg.hidden,
        depth=cfg.ae_depth,
    ).to(cfg.device)
    print("\nTraining autoencoder...")
    train_toy_ae(ae, data_sampler, cfg)

    latent_sampler = _make_latent_sampler(ae, data_sampler, cfg.device)

    # EDM teacher
    teacher = E.EDMPrecondDenoiser(
        cfg.latent_dim, cfg.hidden, cfg.depth, cfg.sigma_data
    ).to(cfg.device)
    print("\nTraining EDM teacher...")
    E.train_edm_teacher(teacher, latent_sampler, cfg)

    # EDM student
    student = E.EDMPrecondDenoiser(
        cfg.latent_dim, cfg.hidden, cfg.depth, cfg.sigma_data
    ).to(cfg.device)
    print(f"\nTraining EDM student ({cfg.train_edm_student_mode})...")
    if cfg.train_edm_student_mode == "jvp":
        E.train_edm_student_jvp(student, teacher, latent_sampler, cfg)
    elif cfg.train_edm_student_mode == "hybrid":
        E.train_edm_student_hybrid(student, teacher, latent_sampler, cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.train_edm_student_mode!r}")

    print("\nPlotting...")
    make_plot(ae, teacher, student, data_sampler, cfg)


if __name__ == "__main__":
    main()