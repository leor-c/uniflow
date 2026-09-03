"""
AYF-style interval flow map experiment on MNIST.

Applies the Uniform RF + AYF experiment to MNIST latents.

Pipeline:
  1. Train convolutional AE (28×28 → latent_dim-d bounded latent).
  2. Pre-encode all training images.
  3. Train RF teacher in latent space.
  4. Train AYF interval flow map student.
  5. Sample and visualise generated digits.

Usage:
    python -m mnist.ayf_rf
    python -m mnist.ayf_rf --latent_dim 128 --teacher_steps 30000
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

from core import rf as R
from core import ayf_rf as A
from core.autoencoder import freeze
from core.modules import apply_ff_spec
from core.seeding import seed_stream, stream_generator
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
    ae_steps: int = 60000
    ae_lr: float = 3e-4
    # AdamW's default weight_decay is 0.01, which was never a deliberate choice
    # here. Decay on the encoder shrinks its weights, which for the tanh head
    # pulls pre-activations toward 0 and the latents toward the origin while the
    # decoder rescales to compensate -- the reconstruction loss is indifferent
    # to that trade. Leading suspect for the bounded latents sitting at
    # mean|z| ~ 0.17 against the 0.5 that U([-1,1]) implies. Set 0 to disable.
    ae_weight_decay: float = 0.01
    ae_batch_size: int = 256
    latent_spread_weight: float = 0

    # RF teacher
    hidden: int = 256
    depth: int = 4
    teacher_steps: int = 80000
    teacher_lr: float = 1e-3

    # --- ReFlow (Liu et al. 2022, "Flow Straight and Fast") -------------------
    # After the teacher is fit, refit it on its own ODE coupling
    # ``(z0, ODE(z0))``. 1-RF trains on independent pairs, so paths cross and the
    # velocity is an average over crossings -- curved, and poorly served by one
    # Euler step. The coupling does not cross, so each round straightens the
    # field, which is exactly what 1-step sampling and AYF distillation need.
    # 0 = plain 1-RF (default), 1 = 2-RF, 2 = 3-RF.
    reflow_rounds: int = 0
    reflow_steps: int = 30000
    reflow_pairs: int = 200000
    # Accuracy of the coupling. Each round targets the previous ODE rather than
    # the data, so too few steps here bakes integration error into the teacher.
    reflow_ode_steps: int = 128

    # AYF interval student
    student_lr: float = 1e-3
    student_steps: int = 80000

    # '+'-joined AYF terms: "emd" | "ct" | "lmd", e.g. "emd+lmd".
    train_mode: str = "emd"
    emd_weight: float = 1.0
    lmd_weight: float = 1.0
    pairwise_anchor_weight: float = 0.0
    teacher_interval_anchor_weight: float = 0.0

    jvp_dt_ref: float = 0.02
    # Shared AYF Algorithm 1 interval schedule, same values as every toy arm.
    # train_student_interval has read these since the schedule was unified in
    # core; without them the student loop raises AttributeError on step 1.
    interval_p_mean: float = -0.6
    interval_p_std: float = 1.6

    # AYF Algorithm 1 line 7: tangent warmup r = min(r_max, Iters / H). r scales
    # only the (s-t)dF/dt term, so training begins as flow matching and eases
    # the map-consistency term in. The paper uses H = 10k with far longer runs;
    # set 0 to disable (r = 1 from step one, which is what we ran before).
    tangent_warmup_steps: int = 2000
    tangent_warmup_max: float = 0.99


    # --- Fourier features (off for MNIST) ------------------------------------
    # The spectral-bias argument these address concerns LOW-dimensional
    # coordinate inputs; the toy arms enable them at latent_dim=2, where a plain
    # MLP cannot represent the near-discontinuous velocity between modes. At
    # this arm's latent_dim they are untested, so they are reachable but off.
    # The toy scales were tuned at latent_dim=2 on 8-Gaussians and do not
    # transfer -- at latent_dim=8 they measurably hurt the teacher.
    ff_size: int = 64
    teacher_ff_scale: float | None = None
    student_ff_scale: float | None = None
    # Bounded latents satisfy E[z^2] = 1/3, so their scale is sqrt(1/3) --
    # matching toy/uniform_rf.py, so one nominal ff_scale means the same
    # effective frequency here as in the standardized arms.
    ff_input_scale: float = math.sqrt(1.0 / 3.0)
    time_eps: float = 1e-4

    interval_teacher_ode_steps: int = 16
    eval_teacher_ode_steps: int = 32

    # Sampling
    batch_size: int = 256
    teacher_sample_steps: int = 64
    sample_k: int = 8
    sample_schedule: str = "uniform"
    sample_clamp: bool = False

    log_every: int = 100
    out_dir: str = "results/mnist/uniform_rf"


def fast_cpu_config(**overrides) -> Config:
    """Reduced config for CPU-only runs (~15-25 min instead of 1-2 h)."""
    cfg = Config(
        latent_dim=64,
        hidden=256,
        depth=4,
        ae_steps=2000,
        ae_lr=1e-3,
        ae_batch_size=32,
        teacher_steps=16000,
        student_steps=16000,
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

    # AYF student 1-step
    s1z = A.sample_student_latents_one_step(student, n, cfg.latent_dim, cfg.device)
    save_image_grid(ae.decode(s1z), os.path.join(cfg.out_dir, "student_1step.png"),
                    title="AYF student 1-step")

    # AYF student k-step
    skz = A.sample_student_latents_k_step(
        student, n, cfg.latent_dim, cfg.device,
        k=cfg.sample_k, schedule=cfg.sample_schedule, clamp=cfg.sample_clamp,
    )
    save_image_grid(ae.decode(skz),
                    os.path.join(cfg.out_dir, f"student_{cfg.sample_k}step.png"),
                    title=f"AYF student {cfg.sample_k}-step")


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
        ap.add_argument(
            "--train_mode", default="emd",
            choices=["emd", "lmd", "emd+lmd", "ct", "ct+lmd"],
        )
        ap.add_argument("--lmd_weight", type=float, default=None)
        ap.add_argument(
            "--ff", default=None,
            help="Which Fourier sites to enable, as a letter string: T=teacher "
                 "state, S=student state (e.g. 'TS', 'T'). 'none' disables all. "
                 "Sites left out are switched OFF.",
        )
        ap.add_argument(
            "--ae_weight_decay", type=float, default=None,
            help="AdamW weight decay for the autoencoder (default 0.01, AdamW's "
                 "own default). Pass 0 to train the AE without decay.",
        )
        ap.add_argument("--out_dir", default="results/mnist/uniform_rf")
        ap.add_argument("--reflow_rounds", type=int, default=None,
                        help="ReFlow rounds: 0 = 1-RF (default), 1 = 2-RF.")
        ap.add_argument("--reflow_steps", type=int, default=None)
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
            cfg.student_steps = args.student_steps
        cfg.train_mode = args.train_mode
        if args.ff is not None:
            apply_ff_spec(cfg, args.ff, allowed=("teacher", "student"))
        if args.lmd_weight is not None:
            cfg.lmd_weight = args.lmd_weight
        if args.ae_weight_decay is not None:
            cfg.ae_weight_decay = args.ae_weight_decay
        cfg.out_dir = args.out_dir
        if args.reflow_rounds is not None:
            cfg.reflow_rounds = args.reflow_rounds
        if args.reflow_steps is not None:
            cfg.reflow_steps = args.reflow_steps
        cfg.data_root = args.data_root

    print(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # --- Data ---
    # One RNG stream per role, all derived from cfg.seed. A single global
    # manual_seed makes weight init, data order, training noise and the figure
    # draws share one stream, so arms with different architectures diverge on
    # their first parameter tensor and never see the same data again.
    train_loader = make_mnist_data_loader(
        cfg.data_root, train=True, batch_size=cfg.ae_batch_size,
        generator=stream_generator(cfg.seed, "ae-data"),
    )
    # Fixed dataset order: the encoded latent matrix and the figure inputs are
    # then the same rows and the same digits in every arm.
    ordered_loader = make_mnist_data_loader(
        cfg.data_root, train=True, batch_size=cfg.ae_batch_size, shuffle=False,
    )

    # --- AE ---
    seed_stream(cfg.seed, "ae-init")
    ae = MNISTAutoEncoder(latent_dim=cfg.latent_dim, bounded=True).to(cfg.device)
    print("\nTraining MNIST autoencoder...")
    seed_stream(cfg.seed, "ae-train")
    train_mnist_ae(
        ae, train_loader,
        iters=cfg.ae_steps,
        lr=cfg.ae_lr,
        device=cfg.device,
        latent_spread_weight=cfg.latent_spread_weight,
        weight_decay=cfg.ae_weight_decay,
        log_every=cfg.log_every,
    )
    torch.save(ae.state_dict(), os.path.join(cfg.out_dir, "ae.pt"))

    # --- Pre-encode training set ---
    print("\nEncoding training data...")
    all_latents = encode_dataset(ae, ordered_loader, cfg.device)
    print(f"  latents: {all_latents.shape}  "
          f"mean_abs={all_latents.abs().mean():.3f}  "
          f"max_abs={all_latents.abs().max():.3f}")
    freeze(ae)

    teacher_latents = make_mnist_latent_sampler(
        all_latents, cfg.device, generator=stream_generator(cfg.seed, "teacher-data")
    )
    student_latents = make_mnist_latent_sampler(
        all_latents, cfg.device, generator=stream_generator(cfg.seed, "student-data")
    )

    # --- RF teacher ---
    seed_stream(cfg.seed, "teacher-init")
    teacher = R.RFTeacher(
        cfg.latent_dim, cfg.hidden, cfg.depth,
        ff_scale=cfg.teacher_ff_scale, ff_size=cfg.ff_size,
        ff_input_scale=cfg.ff_input_scale,
    ).to(cfg.device)
    print("\nTraining RF teacher in latent space...")
    seed_stream(cfg.seed, "teacher-train")
    R.train_teacher(teacher, teacher_latents, cfg)
    torch.save(teacher.state_dict(), os.path.join(cfg.out_dir, "teacher.pt"))

    # --- AYF interval student ---
    seed_stream(cfg.seed, "student-init")
    student = A.RFIntervalFlowMap(
        cfg.latent_dim, cfg.hidden, cfg.depth,
        ff_scale=cfg.student_ff_scale, ff_size=cfg.ff_size,
        ff_input_scale=cfg.ff_input_scale,
    ).to(cfg.device)
    print(f"\nTraining AYF interval flow map ({cfg.train_mode})...")
    seed_stream(cfg.seed, "student-train")
    A.train_student_interval(student, teacher, student_latents, cfg)
    torch.save(student.state_dict(), os.path.join(cfg.out_dir, "student.pt"))

    # --- Visualise ---
    print("\nGenerating sample plots...")
    seed_stream(cfg.seed, "evaluation")
    make_plots(ae, teacher, student, cfg, ordered_loader)

    # These arms emit image grids rather than metrics, so nothing else would
    # record which configuration produced them. Write it alongside the grids.
    import dataclasses, json
    config_path = os.path.join(cfg.out_dir, "run_config.json")
    with open(config_path, "w") as handle:
        json.dump(dataclasses.asdict(cfg), handle, indent=2, sort_keys=True, default=str)
    print(f"Saved run config -> {config_path}")
    print(f"\nAll outputs in {cfg.out_dir}/")


if __name__ == "__main__":
    main()
