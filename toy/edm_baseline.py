"""
EDM baseline toy experiment — 2D datasets (8-Gaussians, two-moons, checkerboard).

Uses the same toy data and unbounded autoencoder as the Gaussian RF arm.

Usage:
    python -m toy.edm_baseline
    python -m toy.edm_baseline --dataset two_moons
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch

from core import ayf as A
from core import edm as E
from core.modules import apply_ff_spec
from core.seeding import seed_stream
from toy.outputs import configure_outputs, save_run_config
from toy.lr_schedule import add_lr_arguments, apply_lr_overrides
from toy.common import (
    ToyAE,
    evaluate_density_outputs,
    make_latent_sampler,
    make_run_samplers,
    save_benchmark_figure,
    save_latent_figure,
    latent_figure_path,
    run_config_record,
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

    # AE
    # The 2D benchmark is dimension-preserving at latent_dim=2: the latent is
    # full-dimensional, so the bounded/uniform arm's supports genuinely
    # coincide. Larger values are an expansion, where a 2D manifold sits in a
    # much bigger space and the extra directions absorb student error.
    latent_dim: int = 2
    hidden: int = 256
    # Width of the autoencoder trunk, independent of the teacher/student
    # ``hidden`` below: the AE fits a 2-D reconstruction, the flow models fit a
    # velocity field over the latent, and there is no reason those need the same
    # capacity. Pairs with ``ae_depth``, which was already separate.
    ae_hidden: int = 128
    ae_depth: int = 2
    ae_steps: int = 40000
    ae_lr: float = 1e-3
    # AdamW's default weight_decay is 0.01, which was never a deliberate choice
    # here -- the AE is a fixed feature extractor, not a model we need to
    # regularize for generalization. Decay on the encoder shrinks its weights,
    # which for the tanh head pulls pre-activations toward 0 and the latents
    # toward the origin while the decoder simply rescales to compensate; the
    # reconstruction loss is indifferent to that trade, so nothing pushes back.
    # That is the leading suspect for MNIST bounded latents sitting at
    # mean|z| ~ 0.17 against the 0.5 that U([-1,1]) implies. Set 0 to disable.
    # Measured on this 2D toy: 0.01 vs 0 moves mean|z| by <0.005 (0.644 vs
    # 0.641, seed 1), far less than the seed spread, and the bounded latents
    # already sit at/above the 0.5 target -- so the knob is inert here and
    # the deeper MNIST encoder is where it could actually bite.
    ae_weight_decay: float = 0.01
    # EDM teacher
    depth: int = 4
    edm_teacher_steps: int = 80000
    edm_teacher_lr: float = 1e-3
    teacher_lr_drop_step: int | None = 40_000
    teacher_final_lr: float = 1e-4

    # EDM noise
    sigma_data: float = 1.0
    sigma_min: float = 0.004
    # Exact 2x scaling of the published sigma_data=0.5 EDM configuration for
    # unit-standardized latents.
    sigma_max: float = 160.0
    edm_rho: float = 7.0
    edm_p_mean: float = -1.2 + math.log(2.0)
    edm_p_std: float = 1.2

    # AYF student. The EDM teacher is wrapped as a flow-matching velocity
    # field, so this arm uses the same student, objective, interval schedule
    # and samplers as the two RF arms; only the teacher and base scale differ.
    student_steps: int = 80_000
    student_lr: float = 1e-3
    student_lr_drop_step: int | None = 40_000
    student_final_lr: float = 1e-4
    # '+'-joined AYF terms: "emd" | "lmd" | "emd+lmd". The weights only set the
    # mix of the terms the mode enables, so lmd_weight is inert without "lmd".
    train_mode: str = "lmd"
    emd_weight: float = 1.0
    lmd_weight: float = 1.0
    jvp_dt_ref: float = 1.0
    time_eps: float = 1e-4
    # Shared AYF Algorithm 1 interval schedule (same values in every arm).
    interval_p_mean: float = -0.6
    interval_p_std: float = 1.6

    # AYF Algorithm 1 line 7: tangent warmup r = min(r_max, Iters / H). r scales
    # only the (s-t)dF/dt term, so training begins as flow matching and eases
    # the map-consistency term in. The paper uses H = 10k with far longer runs;
    # set 0 to disable (r = 1 from step one, which is what we ran before).
    tangent_warmup_steps: int = 2000
    tangent_warmup_max: float = 0.99

    eval_teacher_ode_steps: int = 32
    # Base distribution of the flow-matching interpolant, z_0 ~ N(0, base_std^2 I).
    # Matched to sigma_data because the latents carry unit global scale.
    base_std: float = 1.0

    # --- Fourier features (off by default) -----------------------------------
    # Random Fourier encodings of the coordinate inputs. A plain MLP on a
    # low-dimensional input cannot represent the near-discontinuous velocity at
    # the basin boundaries between modes, which strands mass between them; this
    # matters at small latent_dim and is inert at large. Each site is
    # independent: encoder input, teacher state, student state.
    # Scales are divided by ``ff_input_scale`` below, so one nominal value means
    # the same effective frequency in every arm.
    ff_size: int = 64
    ae_ff_scale: float | None = 0.5 * math.sqrt(2.0)
    teacher_ff_scale: float | None = math.sqrt(2.0 / 3.0)
    student_ff_scale: float | None = math.sqrt(2.0 / 3.0)
    # Flow-matching latents are unit scale; the EDM teacher's features
    # sit on its preconditioned input, which is unit scale for every sigma.
    ff_input_scale: float = 1.0


    # Sampling / visualisation
    batch_size: int = 64
    edm_teacher_sample_steps: int = 256
    edm_student_sample_k: int = 8

    log_every: int = 500
    save_figure: bool = True
    out_dir: str | None = None
    out: str | None = None
    metrics_out: str | None = None


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_plot(ae: ToyAE, teacher, student, eval_sampler, cfg: Config):
    configure_outputs(cfg, "edm_ayf")
    n = 5000
    data = eval_sampler(n).cpu()

    x_real = eval_sampler(n).to(cfg.device)
    x_hat = ae.decode(ae.encode(x_real)).cpu()

    teacher_z = E.sample_edm_teacher_latents(
        teacher, n, cfg.latent_dim, cfg.device,
        sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max,
        rho=cfg.edm_rho, steps=cfg.edm_teacher_sample_steps,
    )
    teacher_samples = ae.decode(teacher_z).cpu()

    s1_z = A.sample_student_latents_one_step(
        student, n, cfg.latent_dim, cfg.device, base_std=cfg.base_std,
    )
    s1 = ae.decode(s1_z).cpu()

    sk_z = A.sample_student_latents_k_step(
        student, n, cfg.latent_dim, cfg.device,
        k=cfg.edm_student_sample_k, base_std=cfg.base_std,
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
    outputs = {
        "ae_reconstruction": x_hat,
        "teacher": teacher_samples,
        "student_1_step": s1,
        f"student_{cfg.edm_student_sample_k}_step": sk,
    }
    # At latent_dim == 2 the latent is a plane, so the same distributions can be
    # drawn before the decoder -- which separates "the flow model missed the
    # latent distribution" from "the decoder moved it", indistinguishable once
    # decoded. Silently skipped at any other dimension.
    if cfg.latent_dim == 2 and cfg.save_figure:
        save_latent_figure(
            [
                ("Encoded data", ae.encode(x_real)),
                ("Student base  N(0,I)", A.sample_noise(n, cfg.latent_dim, cfg.device, cfg.base_std)),
                ("EDM teacher (ODE)", teacher_z),
                ("AYF (1-step)", s1_z),
                (f"AYF ({cfg.edm_student_sample_k}-step)", sk_z),
            ],
            latent_figure_path(cfg.out.format(cfg.dataset)),
            box=None,
        )

    metrics_path = cfg.metrics_out.format(cfg.dataset)
    metrics = evaluate_density_outputs(cfg.dataset, data, outputs)
    metrics["run_config"] = run_config_record(cfg)
    save_evaluation_samples(data, outputs, metrics_path)
    save_metrics(metrics, metrics_path)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config | None = None):
    if cfg is None:
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--dataset", default="8gaussians",
            choices=["8gaussians", "two_moons", "checkerboard", "branching_trees"],
        )
        ap.add_argument("--latent-dim", type=int, default=None)
        ap.add_argument("--teacher-steps", type=int, default=None)
        ap.add_argument("--student-steps", type=int, default=None)
        # Named --train-mode to match toy.uniform_rf and toy.gaussian_rf;
        # --student-mode is kept as an alias for existing scripts.
        ap.add_argument(
            "--train-mode", "--student-mode", dest="train_mode",
            default="lmd", choices=["emd", "lmd", "emd+lmd"],
        )
        ap.add_argument("--lmd-weight", type=float, default=None)
        ap.add_argument(
            "--ae-ff-scale", type=float, default=None,
            help="Fourier feature frequency scale for the encoder input. "
                 "Omit to use the Config default (features ON); pass 0 to disable.",
        )
        ap.add_argument(
            "--teacher-ff-scale", type=float, default=None,
            help="Fourier feature frequency scale for the teacher state. "
                 "Omit to use the Config default (features ON); pass 0 to disable.",
        )
        ap.add_argument(
            "--student-ff-scale", type=float, default=None,
            help="Fourier feature frequency scale for the student state. "
                 "Omit to use the Config default (features ON); pass 0 to disable.",
        )
        ap.add_argument(
            "--ff", default=None,
            help="Which Fourier sites to enable, as a letter string: A=encoder "
                 "input, T=teacher state, S=student state (e.g. 'ATS', 'AT', "
                 "'S'). 'none' disables all. Sites left out are switched OFF. "
                 "Omit to keep the Config defaults; --*-ff-scale overrides the "
                 "frequency of a site this turns on.",
        )
        ap.add_argument(
            "--ae-weight-decay", type=float, default=None,
            help="AdamW weight decay for the autoencoder (default 0.01, AdamW's "
                 "own default). Pass 0 to train the AE without decay.",
        )
        ap.add_argument("--seed", type=int, default=1)
        ap.add_argument("--out-dir", default=None,
                        help="Run directory; default: results/toy/<dataset>/<baseline>/seed_<seed>.")
        ap.add_argument("--out", default=None)
        add_lr_arguments(ap)
        args = ap.parse_args()
        cfg = Config(
            dataset=args.dataset,
            train_mode=args.train_mode,
            seed=args.seed,
            out=args.out,
            out_dir=args.out_dir,
        )
        if args.lmd_weight is not None:
            cfg.lmd_weight = args.lmd_weight
        if args.teacher_steps is not None:
            cfg.edm_teacher_steps = args.teacher_steps
        if args.student_steps is not None:
            cfg.student_steps = args.student_steps
        if args.ff is not None:
            apply_ff_spec(cfg, args.ff)
        for site in ("ae", "teacher", "student"):
            value = getattr(args, f"{site}_ff_scale")
            if value is not None:
                # argparse None means "not supplied"; a non-positive value is the
                # explicit way to switch a site off, since Config uses None for that.
                setattr(cfg, f"{site}_ff_scale", value if value > 0.0 else None)

        if args.ae_weight_decay is not None:
            cfg.ae_weight_decay = args.ae_weight_decay
        if args.latent_dim is not None:
            cfg.latent_dim = args.latent_dim

        apply_lr_overrides(cfg, args, teacher_lr_field="edm_teacher_lr")

    configure_outputs(cfg, "edm_ayf")
    save_run_config(cfg)
    print(cfg)
    # One RNG stream per role, all derived from cfg.seed. A single global
    # manual_seed makes every consumer -- weight init, data, training noise,
    # evaluation draws -- share one stream, so arms with different
    # architectures diverge on their first parameter tensor and never see the
    # same data again. Named streams keep each role a function of the seed only.
    samplers = make_run_samplers(cfg)

    # Match the Gaussian RF autoencoder: no final tanh.
    seed_stream(cfg.seed, "ae-init")
    ae = ToyAE(
        latent_dim=cfg.latent_dim,
        hidden=cfg.ae_hidden,
        depth=cfg.ae_depth,
        bounded=False,
        ff_scale=cfg.ae_ff_scale,
        ff_size=cfg.ff_size,
    ).to(cfg.device)
    print("\nTraining autoencoder...")
    seed_stream(cfg.seed, "ae-train")
    train_toy_ae(ae, samplers.ae, cfg, samplers.calibration)

    teacher_latents = make_latent_sampler(ae, samplers.teacher, cfg.device)
    student_latents = make_latent_sampler(ae, samplers.student, cfg.device)

    # EDM teacher
    seed_stream(cfg.seed, "teacher-init")
    teacher = E.EDMPrecondDenoiser(
        cfg.latent_dim, cfg.hidden, cfg.depth, cfg.sigma_data,
        ff_scale=cfg.teacher_ff_scale, ff_size=cfg.ff_size,
    ).to(cfg.device)
    print("\nTraining EDM teacher...")
    seed_stream(cfg.seed, "teacher-train")
    E.train_edm_teacher(teacher, teacher_latents, cfg)

    # Align Your Flow distils diffusion teachers inside the flow-matching
    # framework, so the EDM denoiser is exposed as a velocity field and the
    # shared RF AYF student is trained against it unchanged.
    flow_teacher = E.EDMFlowVelocity(
        teacher, sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max,
    ).to(cfg.device)
    seed_stream(cfg.seed, "student-init")
    student = A.RFIntervalFlowMap(
        cfg.latent_dim, cfg.hidden, cfg.depth,
        ff_scale=cfg.student_ff_scale, ff_size=cfg.ff_size,
        ff_input_scale=cfg.ff_input_scale,
    ).to(cfg.device)
    print(f"\nTraining EDM + AYF student ({cfg.train_mode})...")
    seed_stream(cfg.seed, "student-train")
    A.train_student_ayf(student, flow_teacher, student_latents, cfg)

    print("\nPlotting...")
    seed_stream(cfg.seed, "evaluation")
    return make_plot(ae, teacher, student, samplers.evaluation, cfg)


if __name__ == "__main__":
    main()
