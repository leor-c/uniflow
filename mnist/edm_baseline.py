"""
EDM baseline experiment on MNIST.

Uses Gaussian perturbations and EDM preconditioning with the shared MNIST AE.

Usage:
    python -m mnist.edm_baseline
    python -m mnist.edm_baseline --latent_dim 128 --teacher_steps 30000
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

from core import ayf as A
from core import edm as E
from core.autoencoder import freeze
from core.modules import apply_ff_spec
from core.seeding import seed_stream, stream_generator
from mnist.ae import (
    LATENT_GRID,
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
    # The latent is a spatial feature map, not a flat vector. The encoder's
    # conv stack ends at 7x7 and a 1x1 conv head puts ``latent_channels``
    # feature planes there, so the generative model sees structure it can
    # actually attend over. ``latent_dim`` is just the flattened size and is
    # derived from the channel count in __post_init__: everything downstream
    # (noise samplers, ODE integrators, losses, metrics) keeps operating on
    # plain ``(B, latent_dim)`` tensors, and only the DiT backbone reshapes
    # back to ``(C, 7, 7)`` internally.
    latent_channels: int = 4
    latent_dim: int = 4 * LATENT_GRID * LATENT_GRID  # derived; see __post_init__
    ae_steps: int = 20000
    ae_lr: float = 1e-3
    # AdamW's default weight_decay is 0.01, which was never a deliberate choice
    # here. Decay on the encoder shrinks its weights, which for the tanh head
    # pulls pre-activations toward 0 and the latents toward the origin while the
    # decoder rescales to compensate -- the reconstruction loss is indifferent
    # to that trade. Leading suspect for the bounded latents sitting at
    # mean|z| ~ 0.17 against the 0.5 that U([-1,1]) implies. Set 0 to disable.
    ae_weight_decay: float = 0.01
    ae_batch_size: int = 256
    latent_spread_weight: float = 1e-3

    # EDM teacher
    hidden: int = 256
    depth: int = 4

    # --- Denoiser backbone ----------------------------------------------------
    # "dit" runs a Diffusion Transformer over the 7x7 latent grid (patch=1 ->
    # 49 tokens, global attention in every layer); "mlp" is the original flat
    # FiLM-MLP, kept as an ablation. ``hidden`` and ``depth`` above are the
    # transformer width and block count when arch="dit".
    #
    # A UNet was the other realistic option and was rejected: at a 7x7 grid it
    # can barely downsample (7 -> 4 -> 2), so its multi-resolution inductive
    # bias -- the entire reason to prefer one -- buys nothing here.
    arch: str = "dit"
    heads: int = 8
    mlp_ratio: float = 4.0
    patch: int = 1
    edm_teacher_steps: int = 30000
    edm_teacher_lr: float = 1e-3

    # EDM noise
    # Must match the latent scale the AE actually produces. This arm uses the
    # unbounded autoencoder, whose latents are rescaled to unit global scale by
    # fit_latent_normalization, so sigma_data is 1.0 -- the 0.5 that suited the
    # old bounded tanh latents mis-calibrates every EDM preconditioning term.
    sigma_data: float = 1.0
    # Exact 2x scaling of the previous sigma_data=0.5 configuration, so every
    # ratio to sigma_data is unchanged: sigma_max/sigma_data stays 10 (large
    # enough for N(0, sigma_max^2) to stand in for p_sigma_max) and sigma_max
    # stays 2.34 sd above the teacher's log-sigma mean, i.e. inside the range it
    # actually trains on. Losing that property is what broke the toy EDM arm.
    sigma_min: float = 0.004
    sigma_max: float = 10.0
    edm_rho: float = 7.0
    edm_p_mean: float = -1.2 + math.log(2.0)
    edm_p_std: float = 1.2

    # AYF student. The teacher is wrapped as a flow-matching velocity field,
    # so these are the shared AYF field names read by core.ayf.
    student_steps: int = 20000
    student_lr: float = 1e-3
    # '+'-joined AYF terms: "emd" | "lmd" | "emd+lmd".
    train_mode: str = "emd"
    emd_weight: float = 1.0
    lmd_weight: float = 1.0
    jvp_dt_ref: float = 1.0
    time_eps: float = 1e-4
    interval_p_mean: float = -0.6
    interval_p_std: float = 1.6

    # AYF Algorithm 1 line 7: tangent warmup r = min(r_max, Iters / H). r scales
    # only the (s-t)dF/dt term, so training begins as flow matching and eases
    # the map-consistency term in. The paper uses H = 10k with far longer runs;
    # set 0 to disable (r = 1 from step one, which is what we ran before).
    tangent_warmup_steps: int = 2000
    tangent_warmup_max: float = 0.99


    # --- Fourier features (off for MNIST) ------------------------------------
    # The spectral-bias argument these address is about LOW-dimensional
    # coordinate inputs; the toy arms enable them at latent_dim=2, where a plain
    # MLP cannot represent the near-discontinuous velocity between modes. At
    # this arm's latent_dim they are untested and unmotivated, so they are
    # exposed but disabled. The scales in toy/ were tuned at latent_dim=2 on
    # 8-Gaussians and should not be assumed to transfer.
    ff_size: int = 64
    teacher_ff_scale: float | None = None
    student_ff_scale: float | None = None
    ff_input_scale: float = 1.0

    eval_teacher_ode_steps: int = 32
    # Base distribution of the flow-matching interpolant.
    base_std: float = 1.0

    # Sampling
    batch_size: int = 256
    edm_teacher_sample_steps: int = 64
    edm_student_sample_k: int = 8

    log_every: int = 1000
    out_dir: str = "results/mnist/edm_ayf"

    def __post_init__(self):
        # Single source of truth is the channel count; the 7x7 grid is fixed by
        # the AE's downsampling, so latent_dim is never set independently.
        self.latent_dim = self.latent_channels * LATENT_GRID * LATENT_GRID

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.latent_channels, LATENT_GRID, LATENT_GRID)

    @property
    def backbone_kwargs(self) -> dict:
        """Backbone selection passed verbatim to every model class."""
        return dict(
            arch=self.arch,
            latent_shape=self.latent_shape,
            heads=self.heads,
            mlp_ratio=self.mlp_ratio,
            patch=self.patch,
        )


def fast_cpu_config(**overrides) -> Config:
    """Reduced config for CPU-only runs (~15–25 min)."""
    cfg = Config(
        latent_channels=4,
        hidden=256,
        depth=4,
        ae_steps=2000,
        ae_lr=1e-3,
        ae_batch_size=32,
        edm_teacher_steps=16000,
        student_steps=16000,
        batch_size=64,
        edm_teacher_sample_steps=256,
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
    n = min(len(images), nrow * nrow)
    images = images[:n].cpu().clamp(0, 1)
    ncol = math.ceil(n / nrow)
    axes_flat = plt.subplots(ncol, nrow, figsize=(nrow * 1.2, ncol * 1.2))[1].flatten()
    for i, ax in enumerate(axes_flat):
        if i < n:
            ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    if title:
        plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


@torch.no_grad()
def make_plots(ae: MNISTAutoEncoder, teacher, student, cfg: Config, loader):
    n = 100

    x_real, _ = next(iter(loader))
    x_real = x_real[:n].to(cfg.device)
    x_hat = ae.decode(ae.encode(x_real))
    save_image_grid(x_real, os.path.join(cfg.out_dir, "ae_input.png"), title="AE input")
    save_image_grid(x_hat, os.path.join(cfg.out_dir, "ae_recon.png"), title="AE reconstruction")

    tz = E.sample_edm_teacher_latents(
        teacher, n, cfg.latent_dim, cfg.device,
        sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max,
        rho=cfg.edm_rho, steps=cfg.edm_teacher_sample_steps,
    )
    save_image_grid(
        ae.decode(tz), os.path.join(cfg.out_dir, "teacher_samples.png"),
        title="EDM Teacher (ODE)",
    )

    s1z = A.sample_student_latents_one_step(
        student, n, cfg.latent_dim, cfg.device, base_std=cfg.base_std,
    )
    save_image_grid(
        ae.decode(s1z), os.path.join(cfg.out_dir, "student_1step.png"),
        title="EDM + AYF 1-step",
    )

    skz = A.sample_student_latents_k_step(
        student, n, cfg.latent_dim, cfg.device,
        k=cfg.edm_student_sample_k, base_std=cfg.base_std,
    )
    save_image_grid(
        ae.decode(skz),
        os.path.join(cfg.out_dir, f"student_{cfg.edm_student_sample_k}step.png"),
        title=f"EDM + AYF {cfg.edm_student_sample_k}-step",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config | None = None):
    if cfg is None:
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--fast", action="store_true",
            help="Use reduced config for CPU-only runs (~15-25 min)",
        )
        ap.add_argument("--latent_channels", type=int, default=None,
                        help="Latent feature planes on the 7x7 grid "
                             "(latent_dim = channels * 49).")
        ap.add_argument("--arch", default=None, choices=["dit", "mlp"],
                        help="Denoiser backbone (default: dit).")
        ap.add_argument("--ae_steps", type=int, default=None)
        ap.add_argument("--teacher_steps", type=int, default=None)
        ap.add_argument("--student_steps", type=int, default=None)
        ap.add_argument(
            "--student_mode",
            default="emd",
            choices=["emd", "lmd", "emd+lmd"],
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
        ap.add_argument("--out_dir", default="results/mnist/edm_ayf")
        ap.add_argument("--data_root", default="./data")
        args = ap.parse_args()
        cfg = fast_cpu_config() if args.fast else Config()
        if args.latent_channels is not None:
            cfg.latent_channels = args.latent_channels
            cfg.latent_dim = cfg.latent_channels * LATENT_GRID * LATENT_GRID
        if args.arch is not None:
            cfg.arch = args.arch
        if args.ae_steps is not None:
            cfg.ae_steps = args.ae_steps
        if args.teacher_steps is not None:
            cfg.edm_teacher_steps = args.teacher_steps
        if args.student_steps is not None:
            cfg.student_steps = args.student_steps
        cfg.train_mode = args.student_mode
        if args.ff is not None:
            apply_ff_spec(cfg, args.ff, allowed=("teacher", "student"))
        if args.lmd_weight is not None:
            cfg.lmd_weight = args.lmd_weight
        if args.ae_weight_decay is not None:
            cfg.ae_weight_decay = args.ae_weight_decay
        cfg.out_dir = args.out_dir
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
    ae = MNISTAutoEncoder(latent_dim=cfg.latent_dim, bounded=False,
                          latent_channels=cfg.latent_channels).to(cfg.device)
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
    print(
        f"  latents: {all_latents.shape}  "
        f"mean_abs={all_latents.abs().mean():.3f}  "
        f"max_abs={all_latents.abs().max():.3f}",
    )
    freeze(ae)

    teacher_latents = make_mnist_latent_sampler(
        all_latents, cfg.device, generator=stream_generator(cfg.seed, "teacher-data")
    )
    student_latents = make_mnist_latent_sampler(
        all_latents, cfg.device, generator=stream_generator(cfg.seed, "student-data")
    )

    # --- EDM teacher ---
    seed_stream(cfg.seed, "teacher-init")
    teacher = E.EDMPrecondDenoiser(
        cfg.latent_dim, cfg.hidden, cfg.depth, cfg.sigma_data,
        ff_scale=cfg.teacher_ff_scale, ff_size=cfg.ff_size,
        **cfg.backbone_kwargs,
    ).to(cfg.device)
    print("\nTraining EDM teacher in latent space...")
    seed_stream(cfg.seed, "teacher-train")
    E.train_edm_teacher(teacher, teacher_latents, cfg)
    torch.save(teacher.state_dict(), os.path.join(cfg.out_dir, "teacher.pt"))

    # --- EDM AYF student ---
    # The teacher is exposed as a flow-matching velocity field, so the shared
    # RF AYF student and objective drive it unchanged (see core.edm.flow).
    flow_teacher = E.EDMFlowVelocity(
        teacher, sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max,
    ).to(cfg.device)
    seed_stream(cfg.seed, "student-init")
    student = A.RFIntervalFlowMap(
        cfg.latent_dim, cfg.hidden, cfg.depth,
        ff_scale=cfg.student_ff_scale, ff_size=cfg.ff_size,
        ff_input_scale=cfg.ff_input_scale,
        **cfg.backbone_kwargs,
    ).to(cfg.device)
    print(f"\nTraining EDM + AYF student ({cfg.train_mode})...")
    seed_stream(cfg.seed, "student-train")
    A.train_student_ayf(student, flow_teacher, student_latents, cfg)
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
