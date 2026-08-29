"""
Gaussian-prior RF + AYF toy benchmark.

Same architecture as ``toy.uniform_rf``. The principal difference is the
noise prior: z0 ~ N(0, I) instead of Uniform[-1, 1]^d.
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
    evaluate_density_outputs,
    make_data_sampler,
    make_latent_sampler,
    save_benchmark_figure,
    save_metrics,
    save_evaluation_samples,
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

    # AE (same architecture as uniform_rf for fair comparison)
    latent_dim: int = 32
    hidden: int = 128
    ae_depth: int = 2
    ae_steps: int = 20000
    ae_lr: float = 1e-3

    # Standard Gaussian base distribution
    base_std: float = 1.0

    # RF teacher (same architecture as uniform_rf)
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
    # Shared AYF Algorithm 1 interval schedule (same values in every arm).
    interval_p_mean: float = -0.6
    interval_p_std: float = 1.6

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
        base_std=cfg.base_std, steps=cfg.teacher_sample_steps,
    )
    teacher_samples = ae.decode(teacher_z).cpu()

    s1_z = A.sample_student_latents_one_step(
        student, n, cfg.latent_dim, cfg.device, base_std=cfg.base_std,
    )
    s1 = ae.decode(s1_z).cpu()

    sk_z = A.sample_student_latents_k_step(
        student, n, cfg.latent_dim, cfg.device,
        k=cfg.sample_k, base_std=cfg.base_std, schedule=cfg.sample_schedule,
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
    outputs = {
        "ae_reconstruction": x_hat,
        "teacher": teacher_samples,
        "student_1_step": s1,
        f"student_{cfg.sample_k}_step": sk,
    }
    metrics_path = cfg.metrics_out.format(cfg.dataset)
    metrics = evaluate_density_outputs(cfg.dataset, data, outputs)
    save_evaluation_samples(data, outputs, metrics_path)
    save_metrics(metrics, metrics_path)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config | None = None):
    if cfg is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--dataset", default="8gaussians",
                        choices=["8gaussians", "two_moons", "checkerboard"])
        ap.add_argument("--latent-dim", type=int, default=32)
        ap.add_argument("--teacher-steps", type=int, default=None)
        ap.add_argument("--student-steps", type=int, default=None)
        ap.add_argument("--train-mode", default="emd", choices=["emd", "emd+lmd"])
        ap.add_argument("--base-std", type=float, default=1.0)
        ap.add_argument("--seed", type=int, default=1)
        ap.add_argument("--out", default=None)
        args = ap.parse_args()
        cfg = Config(
            dataset=args.dataset,
            latent_dim=args.latent_dim,
            base_std=args.base_std,
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
