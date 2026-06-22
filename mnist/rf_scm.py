"""
RF-sCM experiment on MNIST.

Pipeline:
  1. Train convolutional AE (28×28 → latent_dim-d bounded latent).
  2. Pre-encode all training images.
  3. Train RF teacher in latent space.
  4. Train RF-sCM consistency student.
  5. Sample and visualise generated digits.

Usage:
    python -m urfm.mnist.rf_scm
    python -m urfm.mnist.rf_scm --latent_dim 128 --teacher_steps 30000
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import rf_scm as R
from core.autoencoder import freeze
from mnist.ae import (
    MNISTAutoEncoder,
    train_mnist_ae,
    make_mnist_data_loader,
    encode_dataset,
    make_mnist_latent_sampler,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 0
    data_root: str = "./data"

    # AE
    latent_dim: int = 64
    ae_steps: int = 20000
    ae_lr: float = 3e-4
    ae_batch_size: int = 256
    latent_spread_weight: float = 0 #1e-3

    # RF teacher
    hidden: int = 256
    depth: int = 4
    teacher_steps: int = 30000
    teacher_lr: float = 1e-3

    # RF-sCM student
    student_lr: float = 1e-3
    student_hybrid_steps: int = 20000
    student_endpoint_warmup_steps: int = 0
    endpoint_ode_steps: int = 32
    endpoint_every: int = 40
    jvp_weight: float = 1.0
    anchor_jvp_weight: float = 0.001
    jvp_dt_ref: float = 1.0
    jvp_clean_bias: bool = False
    jvp_beta_alpha: float = 2.0
    jvp_beta_beta: float = 1.0
    jvp_use_time_weight: bool = False
    jvp_time_weight_floor: float = 0.05

    # Sampling
    batch_size: int = 256
    teacher_sample_steps: int = 64
    student_sample_k: int = 8
    student_mode: str = "jvp"   # "jvp" | "hybrid" | "ct"

    log_every: int = 100
    out_dir: str = "runs/mnist_rf_scm"


def fast_cpu_config(**overrides) -> Config:
    """
    Reduced config for CPU-only runs (~15–25 min instead of 1–2 h).

    Key reductions:
      - latent_dim 64 → 32   (smaller MLP inputs; faster JVP)
      - hidden 256 → 128     (fewer parameters; ~4× faster per step)
      - ae_steps 20k → 5k   (CNN is the main bottleneck on CPU)
      - teacher 30k → 8k
      - student 20k → 8k
      - ae_batch_size 256 → 128 (less memory pressure per CNN step)
    Sample quality is lower but the pipeline runs end-to-end quickly.
    """
    cfg = Config(
        latent_dim=64,
        hidden=256,
        depth=4,
        ae_steps=2000,
        ae_lr=1e-3,
        ae_batch_size=32,
        teacher_steps=16000,
        student_hybrid_steps=16000,
        batch_size=64,
        teacher_sample_steps=256,
        log_every=100,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_image_grid(images: torch.Tensor, path: str, nrow: int = 10, title: str = ""):
    """Save a grid of (B, 1, H, W) images to `path`."""
    n = min(len(images), nrow * nrow)
    images = images[:n].cpu().clamp(0, 1)
    ncol = math.ceil(n / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 1.2, ncol * 1.2))
    axes = axes.flatten() if ncol > 1 else [axes] if nrow == 1 else axes.flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


@torch.no_grad()
def make_plots(ae: MNISTAutoEncoder, teacher, student, cfg: Config, loader):
    n = 100

    # AE reconstructions
    x_real, _ = next(iter(loader))
    x_real = x_real[:n].to(cfg.device)
    x_hat = ae.decode(ae.encode(x_real))
    save_image_grid(x_real, os.path.join(cfg.out_dir, "ae_input.png"), title="AE input")
    save_image_grid(x_hat, os.path.join(cfg.out_dir, "ae_recon.png"), title="AE reconstruction")

    # Teacher ODE samples
    tz = R.sample_teacher_latents(
        teacher, n, cfg.latent_dim, cfg.device, steps=cfg.teacher_sample_steps
    )
    save_image_grid(ae.decode(tz), os.path.join(cfg.out_dir, "teacher_samples.png"),
                    title="Teacher (ODE)")

    # Student 1-step
    s1z = R.sample_student_latents_one_step(student, n, cfg.latent_dim, cfg.device)
    save_image_grid(ae.decode(s1z), os.path.join(cfg.out_dir, "student_1step.png"),
                    title="Student 1-step")

    # Student k-step
    skz = R.sample_student_latents_k_step(
        student, n, cfg.latent_dim, cfg.device, k=cfg.student_sample_k
    )
    save_image_grid(ae.decode(skz),
                    os.path.join(cfg.out_dir, f"student_{cfg.student_sample_k}step.png"),
                    title=f"Student {cfg.student_sample_k}-step")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config | None = None):
    if cfg is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--fast", action="store_true",
                        help="Use reduced config for CPU-only runs (~15-25 min)")
        ap.add_argument("--latent_dim", type=int, default=None)
        ap.add_argument("--ae_steps", type=int, default=None)
        ap.add_argument("--teacher_steps", type=int, default=None)
        ap.add_argument("--student_steps", type=int, default=None)
        ap.add_argument("--student_mode", default="jvp", choices=["jvp", "hybrid", "ct"])
        ap.add_argument("--out_dir", default="runs/mnist_rf_scm")
        ap.add_argument("--data_root", default="./data")
        args = ap.parse_args()
        cfg = fast_cpu_config() if args.fast else Config()
        if args.latent_dim is not None:
            cfg.latent_dim = args.latent_dim
        if args.ae_steps is not None:
            cfg.ae_steps = args.ae_steps
        if args.teacher_steps is not None:
            cfg.teacher_steps = args.teacher_steps
        if args.student_steps is not None:
            cfg.student_hybrid_steps = args.student_steps
        cfg.student_mode = args.student_mode
        cfg.out_dir = args.out_dir
        cfg.data_root = args.data_root

    print(cfg)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # --- Data ---
    train_loader = make_mnist_data_loader(
        cfg.data_root, train=True, batch_size=cfg.ae_batch_size
    )

    # --- AE ---
    ae = MNISTAutoEncoder(latent_dim=cfg.latent_dim).to(cfg.device)
    print("\nTraining MNIST autoencoder...")
    train_mnist_ae(
        ae, train_loader,
        iters=cfg.ae_steps,
        lr=cfg.ae_lr,
        device=cfg.device,
        latent_spread_weight=cfg.latent_spread_weight,
        log_every=cfg.log_every,
    )
    torch.save(ae.state_dict(), os.path.join(cfg.out_dir, "ae.pt"))

    # --- Pre-encode training set ---
    print("\nEncoding training data...")
    all_latents = encode_dataset(ae, train_loader, cfg.device)
    print(f"  latents: {all_latents.shape}  "
          f"mean_abs={all_latents.abs().mean():.3f}  "
          f"max_abs={all_latents.abs().max():.3f}")
    freeze(ae)

    latent_sampler = make_mnist_latent_sampler(all_latents, cfg.device)

    # --- RF teacher ---
    teacher = R.RFTeacher(cfg.latent_dim, cfg.hidden, cfg.depth).to(cfg.device)
    print("\nTraining RF teacher in latent space...")
    R.train_teacher(teacher, latent_sampler, cfg)
    torch.save(teacher.state_dict(), os.path.join(cfg.out_dir, "teacher.pt"))

    # --- RF-sCM student ---
    student = R.RFConsistencyStudent(cfg.latent_dim, cfg.hidden, cfg.depth).to(cfg.device)
    print(f"\nTraining RF-sCM student ({cfg.student_mode})...")
    if cfg.student_mode == "jvp":
        R.train_student_jvp(student, teacher, latent_sampler, cfg)
    elif cfg.student_mode == "hybrid":
        R.train_student_hybrid(student, teacher, latent_sampler, cfg)
    elif cfg.student_mode == "ct":
        R.train_student_ct(student, latent_sampler, cfg, teacher=teacher)
    else:
        raise ValueError(f"Unknown student_mode: {cfg.student_mode!r}")
    torch.save(student.state_dict(), os.path.join(cfg.out_dir, "student.pt"))

    # --- Visualise ---
    print("\nGenerating sample plots...")
    make_plots(ae, teacher, student, cfg, train_loader)
    print(f"\nAll outputs in {cfg.out_dir}/")


if __name__ == "__main__":
    main()