"""
Uniform-prior, tanh-bounded RF + AYF toy benchmark.

The RF teacher and AYF student operate in the same compact latent cube.

Usage:
    python -m toy.bounded_rf
    python -m toy.bounded_rf --dataset two_moons
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch

from core import rf as R
from core import ayf_rf as A
from toy.common import (
    ToyAE,
    make_data_sampler,
    make_latent_sampler,
    save_benchmark_figure,
    save_metrics,
    sliced_wasserstein_distance,
    train_toy_ae,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    dataset: str = "8gaussians"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 1

    # AE
    latent_dim: int = 32
    hidden: int = 128
    ae_depth: int = 2
    ae_steps: int = 20000
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

    jvp_dt_ref: float = 1.0
    time_eps: float = 1e-4
    # Shared AYF Algorithm 1 interval schedule (same values in every arm).
    interval_p_mean: float = -0.6
    interval_p_std: float = 1.6

    interval_teacher_ode_steps: int = 16
    eval_teacher_ode_steps: int = 32
    teacher_sample_steps: int = 256

    # Sampling / visualisation
    batch_size: int = 64
    sample_k: int = 8
    sample_schedule: str = "uniform"
    sample_clamp: bool = False

    log_every: int = 500
    save_figure: bool = True
    out: str = "results/bounded_rf_{}.png"
    metrics_out: str = "results/bounded_rf_{}.json"


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

    panels = [
        ("Data", data),
        ("AE recon", x_hat),
        ("Bounded RF (ODE)", teacher_samples),
        ("Bounded RF + AYF (1-step)", s1),
        (f"Bounded RF + AYF ({cfg.sample_k}-step)", sk),
    ]
    if cfg.save_figure:
        save_benchmark_figure(panels, cfg.out.format(cfg.dataset))
    metrics = {
        "ae_reconstruction_swd": sliced_wasserstein_distance(data, x_hat),
        "teacher_swd": sliced_wasserstein_distance(data, teacher_samples),
        "student_1_step_swd": sliced_wasserstein_distance(data, s1),
        f"student_{cfg.sample_k}_step_swd": sliced_wasserstein_distance(data, sk),
    }
    save_metrics(metrics, cfg.metrics_out.format(cfg.dataset))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config | None = None):
    if cfg is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--dataset", default="8gaussians",
                        choices=["8gaussians", "two_moons", "checkerboard"])
        ap.add_argument("--latent_dim", type=int, default=32)
        ap.add_argument("--teacher_steps", type=int, default=None)
        ap.add_argument("--student_steps", type=int, default=None)
        ap.add_argument("--train_mode", default="emd", choices=["emd", "ct"])
        ap.add_argument("--seed", type=int, default=1)
        ap.add_argument("--out", default=None)
        args = ap.parse_args()
        cfg = Config(
            dataset=args.dataset,
            latent_dim=args.latent_dim,
            train_mode=args.train_mode,
            seed=args.seed,
            out=args.out or f"results/bounded_rf_{args.dataset}.png",
        )
        if args.teacher_steps is not None:
            cfg.teacher_steps = args.teacher_steps
        if args.student_steps is not None:
            cfg.student_steps = args.student_steps

    print(cfg)
    torch.manual_seed(cfg.seed)

    data_sampler = make_data_sampler(cfg.dataset, cfg.device)

    # Tanh-bounded AE
    ae = ToyAE(
        latent_dim=cfg.latent_dim,
        hidden=cfg.hidden,
        depth=cfg.ae_depth,
        bounded=True,
    ).to(cfg.device)
    print("\nTraining autoencoder...")
    train_toy_ae(ae, data_sampler, cfg)

    latent_sampler = make_latent_sampler(ae, data_sampler, cfg.device)

    # RF teacher with a uniform base distribution
    teacher = R.RFTeacher(cfg.latent_dim, cfg.hidden, cfg.depth).to(cfg.device)
    print("\nTraining RF teacher...")
    R.train_teacher(teacher, latent_sampler, cfg)

    # AYF interval student
    student = A.RFIntervalFlowMap(cfg.latent_dim, cfg.hidden, cfg.depth).to(cfg.device)
    print(f"\nTraining AYF interval flow map ({cfg.train_mode})...")
    A.train_student_interval(student, teacher, latent_sampler, cfg)

    print("\nPlotting...")
    return make_plot(ae, teacher, student, data_sampler, cfg)


if __name__ == "__main__":
    main()
