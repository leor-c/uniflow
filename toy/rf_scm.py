"""
RF-sCM toy experiment — 2D datasets (8-Gaussians, two-moons, checkerboard).

Data, AE architecture and AE training are kept exactly as in the original
standalone script so that results are reproducible. All RF-sCM algorithm
logic (teacher, student, losses, training loops, sampling) is imported from
core.rf_scm.

Usage:
    python -m urfm.toy.rf_scm                          # 8gaussians default
    python -m urfm.toy.rf_scm --dataset two_moons
    python -m urfm.toy.rf_scm --dataset checkerboard
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from core import rf_scm as R
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
# Toy AE  (original architecture preserved exactly: plain MLP, tanh latent)
# ---------------------------------------------------------------------------

class ToyAE(nn.Module):
    """
    2D → latent_dim bounded latent → 2D reconstruction.
    Matches the original rf_scm.py AE exactly (plain MLP, no LayerNorm).
    """

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
    """Penalise mean ≠ 0 and second-moment ≠ 1/3 (Uniform[-1,1] targets)."""
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

    # RF teacher
    depth: int = 4
    teacher_steps: int = 10000
    teacher_lr: float = 1e-3

    # RF-sCM student
    student_lr: float = 1e-3
    student_steps: int = 10000
    endpoint_ode_steps: int = 32
    jvp_weight: float = 1.0
    jvp_dt_ref: float = 1.0
    jvp_clean_bias: bool = False
    jvp_beta_alpha: float = 2.0
    jvp_beta_beta: float = 1.0
    jvp_use_time_weight: bool = False
    jvp_time_weight_floor: float = 0.05

    # Sampling / visualisation
    batch_size: int = 64
    teacher_sample_steps: int = 64
    student_sample_k: int = 8
    student_mode: str = "jvp"   # "jvp" | "ct"

    log_every: int = 500
    out: str = "rf_scm_toy.png"


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

    teacher_z = R.sample_teacher_latents(
        teacher, n, cfg.latent_dim, cfg.device, steps=cfg.teacher_sample_steps
    )
    teacher_samples = ae.decode(teacher_z).cpu()

    s1_z = R.sample_student_latents_one_step(student, n, cfg.latent_dim, cfg.device)
    s1 = ae.decode(s1_z).cpu()

    sk_z = R.sample_student_latents_k_step(
        student, n, cfg.latent_dim, cfg.device, k=cfg.student_sample_k
    )
    sk = ae.decode(sk_z).cpu()

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharex=True, sharey=True)
    panels = [
        ("Data", data),
        ("AE recon", x_hat),
        ("Teacher (ODE)", teacher_samples),
        ("Student 1-step", s1),
        (f"Student {cfg.student_sample_k}-step", sk),
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
        ap.add_argument("--student_mode", default="jvp", choices=["jvp", "ct"])
        ap.add_argument("--seed", type=int, default=2)
        ap.add_argument("--out", default=None)
        args = ap.parse_args()
        cfg = Config(
            dataset=args.dataset,
            latent_dim=args.latent_dim,
            teacher_steps=args.teacher_steps,
            student_steps=args.student_steps,
            student_mode=args.student_mode,
            seed=args.seed,
            out=args.out or f"rf_scm_{args.dataset}.png",
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

    # Teacher
    teacher = R.RFTeacher(cfg.latent_dim, cfg.hidden, cfg.depth).to(cfg.device)
    print("\nTraining RF teacher...")
    R.train_teacher(teacher, latent_sampler, cfg)

    # Student
    student = R.RFConsistencyStudent(cfg.latent_dim, cfg.hidden, cfg.depth).to(cfg.device)
    print(f"\nTraining RF-sCM student ({cfg.student_mode})...")
    if cfg.student_mode == "jvp":
        R.train_student_jvp(student, teacher, latent_sampler, cfg)
    elif cfg.student_mode == "ct":
        R.train_student_ct(student, latent_sampler, cfg, teacher=teacher)
    else:
        raise ValueError(f"Unknown student_mode: {cfg.student_mode!r}")

    print("\nPlotting...")
    make_plot(ae, teacher, student, data_sampler, cfg)


if __name__ == "__main__":
    main()