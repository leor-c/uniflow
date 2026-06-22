"""
AYF-style interval flow map toy experiment — 2D datasets (8-Gaussians,
two-moons, checkerboard).

Generalizes toy/rf_scm.py's endpoint consistency student C(z,t) -> z_1 to
the arbitrary-interval flow map C(z,t,s) -> z_s from core.ayf_rf. The AE and
RF teacher are unchanged from toy/rf_scm.py and reused directly; see
ayf_bounded_latent_rf_summary.md for the derivation.

Usage:
    python -m urfm.toy.ayf_rf                          # 8gaussians default
    python -m urfm.toy.ayf_rf --dataset two_moons
    python -m urfm.toy.ayf_rf --train_mode ct
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch
import matplotlib.pyplot as plt

from core import rf_scm as R
from core import ayf_rf as A
from toy.rf_scm import ToyAE, train_toy_ae, make_data_sampler, make_latent_sampler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    dataset: str = "8gaussians"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 1

    # AE
    latent_dim: int = 16
    hidden: int = 128
    ae_depth: int = 2
    ae_steps: int = 5000
    ae_lr: float = 1e-3
    latent_reg_weight: float = 1e-3

    # RF teacher
    depth: int = 4
    teacher_steps: int = 10000
    teacher_lr: float = 1e-3

    # AYF interval student
    student_lr: float = 1e-3
    student_steps: int = 10000

    train_mode: str = "emd"   # "emd" (teacher-based) | "ct" (teacher-free)
    emd_weight: float = 1.0
    lmd_weight: float = 0.0
    pairwise_anchor_weight: float = 0.0
    teacher_interval_anchor_weight: float = 0.0

    jvp_dt_ref: float = 1  # 0.02
    time_eps: float = 1e-4

    interval_teacher_ode_steps: int = 16
    eval_teacher_ode_steps: int = 32
    teacher_sample_steps: int = 64

    # Sampling / visualisation
    batch_size: int = 64
    sample_k: int = 8
    sample_schedule: str = "uniform"
    sample_clamp: bool = False

    log_every: int = 500
    out: str = "ayf_rf-{}.png"


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

    s1_z = A.sample_student_latents_one_step(student, n, cfg.latent_dim, cfg.device)
    s1 = ae.decode(s1_z).cpu()

    sk_z = A.sample_student_latents_k_step(
        student, n, cfg.latent_dim, cfg.device,
        k=cfg.sample_k, schedule=cfg.sample_schedule, clamp=cfg.sample_clamp,
    )
    sk = ae.decode(sk_z).cpu()

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharex=True, sharey=True)
    panels = [
        ("Data", data),
        ("AE recon", x_hat),
        ("Teacher (ODE)", teacher_samples),
        ("AYF student 1-step", s1),
        (f"AYF student {cfg.sample_k}-step", sk),
    ]
    for ax, (title, pts) in zip(axes, panels):
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.45)
        ax.set_title(title)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(cfg.out.format(cfg.dataset), dpi=160)
    print(f"Saved → {cfg.out.format(cfg.dataset)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config | None = None):
    if cfg is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--dataset", default="8gaussians",
                        choices=["8gaussians", "two_moons", "checkerboard"])
        ap.add_argument("--latent_dim", type=int, default=16)
        ap.add_argument("--teacher_steps", type=int, default=10000)
        ap.add_argument("--student_steps", type=int, default=10000)
        ap.add_argument("--train_mode", default="emd", choices=["emd", "ct"])
        ap.add_argument("--seed", type=int, default=1)
        ap.add_argument("--out", default=None)
        args = ap.parse_args()
        cfg = Config(
            dataset=args.dataset,
            train_mode=args.train_mode,
            seed=args.seed,
            out=args.out or f"ayf_rf_{args.dataset}.png",
        )
        if args.teacher_steps is not None:
            cfg.teacher_steps = args.teacher_steps
        if args.student_steps is not None:
            cfg.student_steps = args.student_steps

    print(cfg)
    torch.manual_seed(cfg.seed)

    data_sampler = make_data_sampler(cfg.dataset, cfg.device)

    # AE (shared with toy/rf_scm.py)
    ae = ToyAE(
        latent_dim=cfg.latent_dim,
        hidden=cfg.hidden,
        depth=cfg.ae_depth,
    ).to(cfg.device)
    print("\nTraining autoencoder...")
    train_toy_ae(ae, data_sampler, cfg)

    latent_sampler = make_latent_sampler(ae, data_sampler, cfg.device)

    # RF teacher (shared with toy/rf_scm.py)
    teacher = R.RFTeacher(cfg.latent_dim, cfg.hidden, cfg.depth).to(cfg.device)
    print("\nTraining RF teacher...")
    R.train_teacher(teacher, latent_sampler, cfg)

    # AYF interval student
    student = A.RFIntervalFlowMap(cfg.latent_dim, cfg.hidden, cfg.depth).to(cfg.device)
    print(f"\nTraining AYF interval flow map ({cfg.train_mode})...")
    A.train_student_interval(student, teacher, latent_sampler, cfg)

    print("\nPlotting...")
    make_plot(ae, teacher, student, data_sampler, cfg)


if __name__ == "__main__":
    main()
