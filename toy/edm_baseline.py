"""
EDM baseline toy experiment — 2D datasets (8-Gaussians, two-moons, checkerboard).

Uses the same toy data and unbounded autoencoder as the Gaussian RF arm.

Usage:
    python -m toy.edm_baseline
    python -m toy.edm_baseline --dataset two_moons
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch

from core import edm as E
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

    # EDM teacher
    depth: int = 4
    edm_teacher_steps: int = 10000
    edm_teacher_lr: float = 1e-3

    # EDM noise
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    # sigma_max must be large relative to sigma_data for N(0, sigma_max^2) to
    # stand in for p_{sigma_max}; at 2.0 (~3.5x) the student's one-shot map
    # starts measurably off-distribution.
    sigma_max: float = 5.0
    edm_rho: float = 7.0
    edm_p_mean: float = -1.2
    edm_p_std: float = 1.2

    # EDM AYF student
    edm_student_steps: int = 10000
    edm_student_lr: float = 1e-3
    train_mode: str = "emd"   # "emd" | "lmd" | "emd+lmd"
    edm_lmd_weight: float = 0.1
    edm_time_eps: float = 1e-4
    edm_interval_p_mean: float = -0.6
    edm_interval_p_std: float = 1.6
    edm_jvp_dt_ref: float = 1.0
    # Floor of the student's normalized-time parameterisation. The PF-ODE is
    # already the identity below ~sigma_data / 10, so carrying the teacher's
    # sigma_min down to 0.002 would spend ~47% of t in [0, 1] transporting
    # nothing (and 4 of 8 k-step intervals inside that dead zone).
    edm_student_sigma_min: float = 0.05
    edm_eval_teacher_ode_steps: int = 32

    # Sampling / visualisation
    batch_size: int = 64
    edm_teacher_sample_steps: int = 256
    edm_student_sample_k: int = 8

    log_every: int = 500
    save_figure: bool = True
    out: str = "results/edm_ayf_{}.png"
    metrics_out: str = "results/edm_ayf_{}.json"


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

    s1_z = E.sample_edm_ayf_latents_one_step(
        student, n, cfg.latent_dim, cfg.device, sigma_max=cfg.sigma_max
    )
    s1 = ae.decode(s1_z).cpu()

    sk_z = E.sample_edm_ayf_latents_k_step(
        student, n, cfg.latent_dim, cfg.device,
        k=cfg.edm_student_sample_k,
        sigma_min=cfg.edm_student_sigma_min, sigma_max=cfg.sigma_max,
        rho=cfg.edm_rho,
    )
    sk = ae.decode(sk_z).cpu()

    panels = [
        ("Data", data),
        ("AE recon", x_hat),
        ("EDM teacher (ODE)", teacher_samples),
        ("EDM + AYF (1-step)", s1),
        (f"EDM + AYF ({cfg.edm_student_sample_k}-step)", sk),
    ]
    if cfg.save_figure:
        save_benchmark_figure(panels, cfg.out.format(cfg.dataset))
    metrics = {
        "ae_reconstruction_swd": sliced_wasserstein_distance(data, x_hat),
        "teacher_swd": sliced_wasserstein_distance(data, teacher_samples),
        "student_1_step_swd": sliced_wasserstein_distance(data, s1),
        f"student_{cfg.edm_student_sample_k}_step_swd": sliced_wasserstein_distance(data, sk),
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
        ap.add_argument(
            "--student_mode",
            default="emd",
            choices=["emd", "lmd", "emd+lmd"],
        )
        ap.add_argument("--seed", type=int, default=1)
        ap.add_argument("--out", default=None)
        args = ap.parse_args()
        cfg = Config(
            dataset=args.dataset,
            latent_dim=args.latent_dim,
            train_mode=args.student_mode,
            seed=args.seed,
            out=args.out or f"results/edm_ayf_{args.dataset}.png",
        )
        if args.teacher_steps is not None:
            cfg.edm_teacher_steps = args.teacher_steps
        if args.student_steps is not None:
            cfg.edm_student_steps = args.student_steps
    print(cfg)
    torch.manual_seed(cfg.seed)

    data_sampler = make_data_sampler(cfg.dataset, cfg.device)

    # Match the Gaussian RF autoencoder: no final tanh.
    ae = ToyAE(
        latent_dim=cfg.latent_dim,
        hidden=cfg.hidden,
        depth=cfg.ae_depth,
        bounded=False,
    ).to(cfg.device)
    print("\nTraining autoencoder...")
    train_toy_ae(ae, data_sampler, cfg)

    latent_sampler = make_latent_sampler(ae, data_sampler, cfg.device)

    # EDM teacher
    teacher = E.EDMPrecondDenoiser(
        cfg.latent_dim, cfg.hidden, cfg.depth, cfg.sigma_data
    ).to(cfg.device)
    print("\nTraining EDM teacher...")
    E.train_edm_teacher(teacher, latent_sampler, cfg)

    # AYF interval student driven by the EDM probability-flow ODE.
    student = E.EDMIntervalFlowMap(
        cfg.latent_dim, cfg.hidden, cfg.depth
    ).to(cfg.device)
    print(f"\nTraining EDM + AYF student ({cfg.train_mode})...")
    E.train_edm_student_ayf(student, teacher, latent_sampler, cfg)

    print("\nPlotting...")
    return make_plot(ae, teacher, student, data_sampler, cfg)


if __name__ == "__main__":
    main()
