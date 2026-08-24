"""
Gaussian-prior RF + AYF toy benchmark.

Same architecture as ``toy.bounded_rf``. The principal difference is the
noise prior: z0 ~ N(0, sigma_data² I) instead of Uniform[-1, 1]^d.
The teacher is the same RFTeacher (velocity field), trained on Gaussian paths.

Usage:
    python -m toy.gaussian_rf
    python -m toy.gaussian_rf --dataset two_moons
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch

from core.rf import RFTeacher
from core import ayf as A
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

    # AE (same architecture as bounded_rf for fair comparison)
    latent_dim: int = 32
    hidden: int = 128
    ae_depth: int = 2
    ae_steps: int = 20000
    ae_lr: float = 1e-3
    latent_reg_weight: float = 1e-3

    # Gaussian noise scale
    sigma_data: float = 0.5

    # RF teacher (same architecture as bounded_rf)
    depth: int = 4
    teacher_steps: int = 10000
    teacher_lr: float = 1e-3

    # AYF student for the Gaussian RF teacher
    student_lr: float = 1e-3
    student_steps: int = 10000
    train_mode: str = "emd"    # "emd" | "emd+lmd"
    emd_weight: float = 1.0
    lmd_weight: float = 0.0
    jvp_dt_ref: float = 1.0
    time_eps: float = 1e-4

    eval_teacher_ode_steps: int = 32
    teacher_sample_steps: int = 256

    # Sampling / visualisation
    batch_size: int = 64
    sample_k: int = 8
    sample_schedule: str = "uniform"

    log_every: int = 500
    save_figure: bool = True
    out: str = "results/gaussian_rf_{}.png"
    metrics_out: str = "results/gaussian_rf_{}.json"


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_plot(ae: ToyAE, teacher, student, data_sampler, cfg: Config):
    n = 5000

    data = data_sampler(n).cpu()

    x_real = data_sampler(n).to(cfg.device)
    x_hat = ae.decode(ae.encode(x_real)).cpu()

    teacher_z = A.sample_teacher_latents(
        teacher, n, cfg.latent_dim, cfg.device,
        sigma_data=cfg.sigma_data, steps=cfg.teacher_sample_steps,
    )
    teacher_samples = ae.decode(teacher_z).cpu()

    s1_z = A.sample_student_latents_one_step(
        student, n, cfg.latent_dim, cfg.device, sigma_data=cfg.sigma_data,
    )
    s1 = ae.decode(s1_z).cpu()

    sk_z = A.sample_student_latents_k_step(
        student, n, cfg.latent_dim, cfg.device,
        k=cfg.sample_k, sigma_data=cfg.sigma_data, schedule=cfg.sample_schedule,
    )
    sk = ae.decode(sk_z).cpu()

    panels = [
        ("Data", data),
        ("AE recon", x_hat),
        ("Gaussian RF (ODE)", teacher_samples),
        ("Gaussian RF + AYF (1-step)", s1),
        (f"Gaussian RF + AYF ({cfg.sample_k}-step)", sk),
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
        ap.add_argument("--train_mode", default="emd", choices=["emd", "emd+lmd"])
        ap.add_argument("--sigma_data", type=float, default=0.5)
        ap.add_argument("--seed", type=int, default=1)
        ap.add_argument("--out", default=None)
        args = ap.parse_args()
        cfg = Config(
            dataset=args.dataset,
            latent_dim=args.latent_dim,
            sigma_data=args.sigma_data,
            train_mode=args.train_mode,
            seed=args.seed,
            out=args.out or f"results/gaussian_rf_{args.dataset}.png",
        )
        if args.teacher_steps is not None:
            cfg.teacher_steps = args.teacher_steps
        if args.student_steps is not None:
            cfg.student_steps = args.student_steps
        if args.train_mode == "emd+lmd":
            cfg.lmd_weight = 0.1

    print(cfg)
    torch.manual_seed(cfg.seed)

    data_sampler = make_data_sampler(cfg.dataset, cfg.device)

    # Unbounded AE
    ae = ToyAE(
        latent_dim=cfg.latent_dim,
        hidden=cfg.hidden,
        depth=cfg.ae_depth,
        bounded=False,
    ).to(cfg.device)
    print("\nTraining autoencoder...")
    train_toy_ae(ae, data_sampler, cfg)

    latent_sampler = make_latent_sampler(ae, data_sampler, cfg.device)

    # RF teacher with Gaussian noise
    teacher = RFTeacher(cfg.latent_dim, cfg.hidden, cfg.depth).to(cfg.device)
    print("\nTraining RF teacher (Gaussian noise)...")
    A.train_teacher_gaussian(teacher, latent_sampler, cfg)

    # AYF interval student (same RFIntervalFlowMap as ayf_rf)
    student = A.RFIntervalFlowMap(cfg.latent_dim, cfg.hidden, cfg.depth).to(cfg.device)
    print(f"\nTraining Gaussian RF + AYF student ({cfg.train_mode})...")
    A.train_student_ayf(student, teacher, latent_sampler, cfg)

    print("\nPlotting...")
    return make_plot(ae, teacher, student, data_sampler, cfg)


if __name__ == "__main__":
    main()
