"""
Uniform RF + AYF toy benchmark.

The RF teacher and AYF student operate in the same compact latent cube.

Usage:
    python -m toy.uniform_rf
    python -m toy.uniform_rf --dataset two_moons
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from core import rf as R
from core import ayf_rf as A
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
    # potentially full-dimensional. Tanh puts it inside the base cube, but
    # does not guarantee that the encoder covers that cube. Larger values are
    # an expansion, where a 2D manifold sits in a
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
    # Off by default: at latent_dim=2 the moment penalty is unnecessary (it
    # changes 1-step NLL by less than the seed spread). It is NOT optional at
    # large latent_dim -- without it the encoder collapses coordinates and the
    # teacher itself fails, so raise it if you increase latent_dim.
    latent_reg_weight: float = 0.0

    # --- Fourier features ----------------------------------------------------
    # Random Fourier encodings of the coordinate inputs. A plain MLP on a
    # low-dimensional input cannot represent the near-discontinuous velocity at
    # the basin boundaries between modes, which strands mass between them; this
    # matters at small latent_dim and is inert at large. Each site is
    # independent: encoder input, teacher state, student state.
    # Scales are divided by ``ff_input_scale`` below, so one nominal value means
    # the same effective frequency in every arm.
    ff_size: int = 64
    # Kept identical to the other arms: the encoder input is the same 2D data
    # everywhere, so an arm-specific scale would make the comparison uncontrolled.
    ae_ff_scale: float | None = 0.5 * math.sqrt(2.0)
    teacher_ff_scale: float | None = math.sqrt(2.0 / 3.0)
    student_ff_scale: float | None = math.sqrt(2.0 / 3.0)
    # The uniform BASE has E[z^2] = 1/3. Bounded encoded data need not share it.
    ff_input_scale: float = math.sqrt(1.0 / 3.0)


    # RF teacher
    depth: int = 4
    teacher_steps: int = 80_000
    teacher_lr: float = 1e-3
    teacher_lr_drop_step: int | None = 40_000
    teacher_final_lr: float = 1e-4
    # Optional low-LR finishing phase, counted in addition to teacher_steps.
    teacher_refine_steps: int = 0
    teacher_refine_lr: float = 1e-4

    # --- ReFlow (Liu et al. 2022, "Flow Straight and Fast") -------------------
    # After the teacher is fit, refit it on its own ODE coupling
    # ``(z0, ODE(z0))``. 1-RF trains on independent pairs, so paths cross and the
    # velocity is an average over crossings -- curved, and poorly served by one
    # Euler step. The coupling does not cross, so each round straightens the
    # field, which is exactly what 1-step sampling and AYF distillation need.
    # 0 = plain 1-RF (default), 1 = 2-RF, 2 = 3-RF.
    reflow_rounds: int = 0
    reflow_steps: int = 20000
    reflow_pairs: int = 200_000
    # Accuracy of the coupling. Each round targets the previous ODE rather than
    # the data, so too few steps here bakes integration error into the teacher.
    reflow_ode_steps: int = 256

    # --- Shared-architecture teacher --------------------------------------
    # Train the RF teacher as an RFIntervalFlowMap (velocity regression on its
    # slope at sampled s) instead of an RFTeacher, then initialize the student
    # with a full weight copy of it. Teacher and student differ only in the
    # width of the conditioning projection otherwise, so a separate RFTeacher
    # is the only reason the student has to start from scratch.
    #
    # Measured on 8-Gaussians, LMD, 3 seeds: at the shipped 80k student steps
    # this is worth ~0.13 nats 1-step / ~0.18 nats 8-step target NLL, almost
    # all of it from the map teacher rather than the copy. At a 10k budget the
    # split reverses and the copy carries it -- copy-initialized students reach
    # -1.52 nats 8-step against -0.82 for scratch, within 0.09 of a scratch
    # student trained 8x longer. It also cuts seed spread ~3-8x.
    #
    # Off by default: it changes every teacher and student number this arm
    # records, so switching it on is a deliberate re-baselining, not a default.
    shared_teacher: bool = False

    # AYF interval student
    student_lr: float = 1e-3
    student_lr_drop_step: int | None = 40_000
    student_final_lr: float = 1e-4
    student_steps: int = 80_000
    student_refine_steps: int = 0
    student_refine_lr: float = 1e-4

    # '+'-joined AYF terms: "emd" (teacher-based) | "ct" (teacher-free) | "lmd".
    # e.g. "emd", "lmd", "emd+lmd", "ct+lmd". The weights only set the mix of
    # the terms the mode enables, so lmd_weight is inert unless "lmd" is in it.
    train_mode: str = "lmd"
    emd_weight: float = 1.0
    lmd_weight: float = 1.0
    pairwise_anchor_weight: float = 0.0
    teacher_interval_anchor_weight: float = 0.0

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


    interval_teacher_ode_steps: int = 16
    eval_teacher_ode_steps: int = 32
    teacher_sample_steps: int = 256

    # Sampling / visualisation
    batch_size: int = 64
    sample_k: int = 8
    sample_schedule: str = "uniform"
    sample_clamp: bool = True

    log_every: int = 500
    save_figure: bool = True
    out_dir: str | None = None
    out: str | None = None
    metrics_out: str | None = None
    checkpoint_out: str | None = None


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_plot(ae: ToyAE, teacher, student, eval_sampler, cfg: Config):
    configure_outputs(cfg, "uniform_rf")
    n = 5000
    data = eval_sampler(n).cpu()

    x_real = eval_sampler(n).to(cfg.device)
    x_hat = ae.decode(ae.encode(x_real)).cpu()

    teacher_z = R.sample_teacher_latents(
        teacher, n, cfg.latent_dim, cfg.device, steps=cfg.teacher_sample_steps
    )
    if cfg.sample_clamp:
        teacher_z = teacher_z.clamp(-1, 1)
    teacher_samples = ae.decode(teacher_z).cpu()

    s1_z = A.sample_student_latents_one_step(student, n, cfg.latent_dim, cfg.device)
    if cfg.sample_clamp:
        s1_z = s1_z.clamp(-1, 1)
    s1 = ae.decode(s1_z).cpu()

    sk_z = A.sample_student_latents_k_step(
        student, n, cfg.latent_dim, cfg.device,
        k=cfg.sample_k, schedule=cfg.sample_schedule, clamp=cfg.sample_clamp,
    )
    sk = ae.decode(sk_z).cpu()

    panels = [
        ("Data", data),
        ("AE recon", x_hat),
        ("Uniform RF (ODE)", teacher_samples),
        ("Uniform RF + AYF (1-step)", s1),
        (f"Uniform RF + AYF ({cfg.sample_k}-step)", sk),
    ]
    if cfg.save_figure:
        save_benchmark_figure(panels, cfg.out.format(cfg.dataset))
    outputs = {
        "ae_reconstruction": x_hat,
        "teacher": teacher_samples,
        "student_1_step": s1,
        f"student_{cfg.sample_k}_step": sk,
    }
    # At latent_dim == 2 the latent is a plane, so the same distributions can be
    # drawn before the decoder -- which separates "the flow model missed the
    # latent distribution" from "the decoder moved it", indistinguishable once
    # decoded. Silently skipped at any other dimension.
    if cfg.latent_dim == 2 and cfg.save_figure:
        save_latent_figure(
            [
                ("Encoded data", ae.encode(x_real)),
                ("Base  U([-1,1]^2)", R.sample_uniform_noise(n, cfg.latent_dim, cfg.device)),
                ("Uniform RF (ODE)", teacher_z),
                ("AYF (1-step)", s1_z),
                (f"AYF ({cfg.sample_k}-step)", sk_z),
            ],
            latent_figure_path(cfg.out.format(cfg.dataset)),
            box=1.0,
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
        ap.add_argument("--dataset", default="8gaussians",
                        choices=["8gaussians", "two_moons", "checkerboard"])
        ap.add_argument("--latent-dim", type=int, default=None)
        ap.add_argument("--teacher-steps", type=int, default=None)
        ap.add_argument("--student-steps", type=int, default=None)
        for role in ("teacher", "student"):
            ap.add_argument(f"--{role}-refine-steps", type=int, default=0,
                            help="Additional updates at the refinement learning rate.")
            ap.add_argument(f"--{role}-refine-lr", type=float, default=1e-4)
        ap.add_argument("--teacher-sample-steps", type=int, default=256)
        ap.add_argument("--no-sample-clamp", action="store_true",
                        help="Disable latent clipping for all three generated panels.")
        ap.add_argument("--checkpoint-out", default=None,
                        help="Save AE, teacher, student state dicts and full run config.")
        ap.add_argument("--metrics-out", default=None)
        ap.add_argument("--log-every", type=int, default=500)
        ap.add_argument(
            "--train-mode", default="lmd",
            choices=["emd", "lmd", "emd+lmd", "ct", "ct+lmd"],
        )
        ap.add_argument("--lmd-weight", type=float, default=None)
        ap.add_argument("--shared-teacher", action="store_true",
                        help="Train the RF teacher as an interval flow map and "
                             "initialize the student from its weights.")
        ap.add_argument("--ae-latent-reg-weight", type=float, default=None)
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
        ap.add_argument(
            "--reflow-rounds", type=int, default=None,
            help="ReFlow rounds after the teacher fit: 0 = plain 1-RF (default), "
                 "1 = 2-RF, 2 = 3-RF. Straightens the velocity field.",
        )
        ap.add_argument("--reflow-steps", type=int, default=None,
                        help="Training steps per ReFlow round.")
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
            metrics_out=args.metrics_out,
            teacher_refine_steps=args.teacher_refine_steps,
            teacher_refine_lr=args.teacher_refine_lr,
            student_refine_steps=args.student_refine_steps,
            student_refine_lr=args.student_refine_lr,
            teacher_sample_steps=args.teacher_sample_steps,
            sample_clamp=not args.no_sample_clamp,
            checkpoint_out=args.checkpoint_out,
            log_every=args.log_every,
        )
        if args.teacher_steps is not None:
            cfg.teacher_steps = args.teacher_steps
        if args.lmd_weight is not None:
            cfg.lmd_weight = args.lmd_weight
        cfg.shared_teacher = args.shared_teacher
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

        if args.ae_latent_reg_weight is not None:
            cfg.latent_reg_weight = args.ae_latent_reg_weight
        if args.ae_weight_decay is not None:
            cfg.ae_weight_decay = args.ae_weight_decay
        if args.reflow_rounds is not None:
            cfg.reflow_rounds = args.reflow_rounds
        if args.reflow_steps is not None:
            cfg.reflow_steps = args.reflow_steps
        if args.latent_dim is not None:
            cfg.latent_dim = args.latent_dim

        apply_lr_overrides(cfg, args, teacher_lr_field="teacher_lr")

    configure_outputs(cfg, "uniform_rf")
    if cfg.teacher_sample_steps < 1 or cfg.log_every < 1:
        raise ValueError("teacher_sample_steps and log_every must be positive")
    for role in ("teacher", "student"):
        steps = getattr(cfg, f"{role}_refine_steps")
        lr = getattr(cfg, f"{role}_refine_lr")
        if steps < 0 or (steps and lr <= 0):
            raise ValueError(f"{role} refinement needs nonnegative steps and a positive LR")
    checkpoint_path = (
        Path(cfg.checkpoint_out.format(cfg.dataset)) if cfg.checkpoint_out else None
    )
    if checkpoint_path is not None:
        samples_path = Path(cfg.metrics_out.format(cfg.dataset)).with_suffix(".pt")
        if checkpoint_path.resolve() == samples_path.resolve():
            raise ValueError("Model checkpoint must differ from evaluation-sample .pt path")
    save_run_config(cfg)
    print(cfg)
    # One RNG stream per role, all derived from cfg.seed. A single global
    # manual_seed makes every consumer -- weight init, data, training noise,
    # evaluation draws -- share one stream, so arms with different
    # architectures diverge on their first parameter tensor and never see the
    # same data again. Named streams keep each role a function of the seed only.
    samplers = make_run_samplers(cfg)

    # Tanh-bounded AE
    seed_stream(cfg.seed, "ae-init")
    ae = ToyAE(
        latent_dim=cfg.latent_dim,
        hidden=cfg.ae_hidden,
        depth=cfg.ae_depth,
        bounded=True,
        ff_scale=cfg.ae_ff_scale,
        ff_size=cfg.ff_size,
    ).to(cfg.device)
    print("\nTraining autoencoder...")
    seed_stream(cfg.seed, "ae-train")
    train_toy_ae(ae, samplers.ae, cfg)

    teacher_latents = make_latent_sampler(ae, samplers.teacher, cfg.device)
    student_latents = make_latent_sampler(ae, samplers.student, cfg.device)

    # RF teacher with a uniform base distribution. Under cfg.shared_teacher the
    # teacher is itself an interval flow map, exposed to everything downstream
    # through VelocityView, which has the RFTeacher call signature.
    seed_stream(cfg.seed, "teacher-init")
    if cfg.shared_teacher:
        teacher_map = A.RFIntervalFlowMap(
            cfg.latent_dim, cfg.hidden, cfg.depth,
            ff_scale=cfg.student_ff_scale, ff_size=cfg.ff_size,
            ff_input_scale=cfg.ff_input_scale,
        ).to(cfg.device)
        print("\nTraining RF teacher (shared interval-map architecture)...")
        seed_stream(cfg.seed, "teacher-train")
        teacher = A.train_map_teacher(teacher_map, teacher_latents, cfg)
    else:
        teacher_map = None
        teacher = R.RFTeacher(
            cfg.latent_dim, cfg.hidden, cfg.depth,
            ff_scale=cfg.teacher_ff_scale, ff_size=cfg.ff_size,
            ff_input_scale=cfg.ff_input_scale,
        ).to(cfg.device)
        print("\nTraining RF teacher...")
        seed_stream(cfg.seed, "teacher-train")
        R.train_teacher(teacher, teacher_latents, cfg)

    # AYF interval student
    seed_stream(cfg.seed, "student-init")
    student = A.RFIntervalFlowMap(
        cfg.latent_dim, cfg.hidden, cfg.depth,
        ff_scale=cfg.student_ff_scale, ff_size=cfg.ff_size,
        ff_input_scale=cfg.ff_input_scale,
    ).to(cfg.device)
    if teacher_map is not None:
        # Same class, same arguments: every tensor copies, so the student opens
        # as an Euler step with the teacher's velocity on the whole (t, s)
        # domain rather than at random.
        student.load_state_dict(teacher_map.state_dict())
        print("  student initialized from the teacher's weights")
    print(f"\nTraining AYF interval flow map ({cfg.train_mode})...")
    seed_stream(cfg.seed, "student-train")
    A.train_student_interval(student, teacher, student_latents, cfg)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "run_config": asdict(cfg),
            "ae": ae.state_dict(),
            # Under shared_teacher the teacher is a VelocityView wrapper, whose
            # state_dict is prefixed; save the map itself so the checkpoint
            # loads back into the class that produced it.
            "teacher": (teacher_map or teacher).state_dict(),
            "student": student.state_dict(),
        }, checkpoint_path)
        print(f"Saved model checkpoint -> {checkpoint_path}")

    print("\nPlotting...")
    seed_stream(cfg.seed, "evaluation")
    return make_plot(ae, teacher, student, samplers.evaluation, cfg)


if __name__ == "__main__":
    main()
