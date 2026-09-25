"""
Run the three CIFAR-10 benchmark arms: Uniform RF + AYF, Gaussian RF + AYF,
and EDM + AYF.

Each arm uses its own default Config. The --latent-channels, --ae-steps,
--teacher-steps, --student-steps and --train-mode flags are applied on top of
those defaults only when explicitly passed; any flag left unset keeps each
arm's own default for that field.

Unlike ``toy.run_all``, this script does not aggregate
metrics, write summary tables, or support --resume. The image arms produce
grids rather than density metrics, so there is nothing to aggregate: comparison
here is visual, by looking at the grids in each arm's output directory. Adding a
quantitative image metric (FID or similar) is what a summary table would
require -- and CIFAR-10 is the dataset where that would actually pay off, since
unconditional FID on it is widely published.

Runs are sequential by default. The CIFAR-10 arms are heavier than the MNIST
ones -- a wider convolutional autoencoder over 3-channel inputs, 50k encoded
latents per arm, and 64 DiT tokens instead of 49 -- so parallel seeds can
exhaust GPU memory; --workers opts in explicitly.

Usage:
    python -m cifar10.run_all
    python -m cifar10.run_all --train-mode lmd
    python -m cifar10.run_all --seeds 0 1 2 --ae-steps 5000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
from pathlib import Path

import torch

from core.modules import apply_ff_spec
from cifar10 import edm_baseline, gaussian_rf, uniform_rf
from cifar10.ae import LATENT_GRID


PIPELINES = ("uniform_rf", "gaussian_rf", "edm_ayf")
PIPELINE_LABELS = {
    "uniform_rf": "Uniform RF + AYF",
    "gaussian_rf": "Gaussian RF + AYF",
    "edm_ayf": "EDM + AYF",
}
# result key -> (module, teacher-steps field, student-steps field, default out dir)
PIPELINE_MODULES = {
    "uniform_rf": (uniform_rf, "teacher_steps", "student_steps", "results/cifar10/uniform_rf"),
    "gaussian_rf": (gaussian_rf, "teacher_steps", "student_steps", "results/cifar10/gaussian_rf"),
    "edm_ayf": (edm_baseline, "edm_teacher_steps", "student_steps", "results/cifar10/edm_ayf"),
}


def _apply_overrides(cfg, args, seed: int, *, teacher_field: str, student_field: str):
    cfg.seed = seed
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.latent_channels is not None:
        cfg.latent_channels = args.latent_channels
        cfg.latent_dim = cfg.latent_channels * LATENT_GRID * LATENT_GRID
    if args.arch is not None:
        cfg.arch = args.arch
    if args.patch is not None:
        cfg.patch = args.patch
    if args.train_mode is not None:
        cfg.train_mode = args.train_mode
    if args.ff is not None:
        apply_ff_spec(cfg, args.ff, allowed=("teacher", "student"))
    for site in ("teacher", "student"):
        value = getattr(args, f"{site}_ff_scale")
        if value is not None:
            setattr(cfg, f"{site}_ff_scale", value if value > 0.0 else None)
    # The EDM arm trains a denoiser, not an RF velocity, so it has no reflow
    # field -- skip it rather than inventing one.
    for field in ("reflow_rounds", "reflow_steps"):
        value = getattr(args, field)
        if value is not None and hasattr(cfg, field):
            setattr(cfg, field, value)
    if args.ae_weight_decay is not None:
        cfg.ae_weight_decay = args.ae_weight_decay
    if args.ae_lpips_weight is not None:
        cfg.ae_lpips_weight = args.ae_lpips_weight
    if args.ae_steps is not None:
        cfg.ae_steps = args.ae_steps
    if args.teacher_steps is not None:
        setattr(cfg, teacher_field, args.teacher_steps)
    if args.student_steps is not None:
        setattr(cfg, student_field, args.student_steps)
    return cfg


def _output_dir(pipeline: str, seed: int, args, *, multi_seed: bool) -> str:
    base = Path(args.out_root) if args.out_root else Path(PIPELINE_MODULES[pipeline][3])
    if args.out_root:
        base = base / f"cifar10_{pipeline}"
    return str(base / f"seed-{seed}" if multi_seed else base)


def _run_seed(args, seed: int, *, multi_seed: bool):
    """Run all three arms for one seed. Returns {pipeline: output directory}."""
    outputs = {}
    for pipeline in PIPELINES:
        module, teacher_field, student_field, _ = PIPELINE_MODULES[pipeline]
        print("\n" + "=" * 60)
        print(f"Running seed {seed}: cifar10.{module.__name__.rsplit('.', 1)[-1]}")
        print("=" * 60)
        cfg = _apply_overrides(
            module.Config(), args, seed,
            teacher_field=teacher_field, student_field=student_field,
        )
        cfg.out_dir = _output_dir(pipeline, seed, args, multi_seed=multi_seed)
        module.main(cfg)
        outputs[pipeline] = cfg.out_dir
    return outputs


def _run_seed_worker(args_dict: dict, seed: int, multi_seed: bool):
    """Spawn-safe wrapper for one independent seed run."""
    args = argparse.Namespace(**args_dict)
    return seed, _run_seed(args, seed, multi_seed=multi_seed)


def _worker_count(cuda_available: bool, pending_seeds: int, requested: int | None) -> int:
    if not cuda_available or pending_seeds <= 1 or requested is None:
        return 1
    return min(requested, pending_seeds)


def main():
    ap = argparse.ArgumentParser()
    seed_group = ap.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None)
    seed_group.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument(
        "--train-mode", default=None, choices=["emd", "lmd", "emd+lmd"],
        help="AYF objective applied to every arm, same grammar as the "
             "individual arms. Omit to keep each arm's Config default. "
             "('ct' is uniform-only; use python -m cifar10.uniform_rf for it.)",
    )
    for _site in ("teacher", "student"):
        ap.add_argument(
            f"--{_site}-ff-scale", type=float, default=None,
            help=f"Fourier scale for the {_site} state, applied to every arm. "
                 "Omit for the Config default (off for CIFAR-10); 0 disables.",
        )
    ap.add_argument("--latent-channels", type=int, default=None,
                    help="Latent feature planes on the 8x8 grid "
                         "(latent_dim = channels * 64).")
    ap.add_argument("--arch", default=None, choices=["dit", "mlp"],
                    help="Denoiser backbone for every arm (default: dit).")
    ap.add_argument("--patch", type=int, default=None,
                    help="DiT patch size over the 8x8 grid, for every arm: "
                         "1 (64 tokens, default) or 2 (16 tokens, cheaper).")
    ap.add_argument(
        "--ff", default=None,
        help="Which Fourier sites to enable in EVERY arm, as a letter string: "
             "T=teacher state, S=student state (e.g. 'TS', 'T'). 'none' "
             "disables all. Sites left out are switched OFF.",
    )
    ap.add_argument(
        "--ae-weight-decay", type=float, default=None,
        help="AdamW weight decay for the autoencoder in EVERY arm (default 0.01, "
             "AdamW's own default). Pass 0 to train the AEs without decay.",
    )
    ap.add_argument(
        "--reflow-rounds", type=int, default=None,
        help="ReFlow rounds for the RF arms: 0 = plain 1-RF (default), 1 = 2-RF, "
             "2 = 3-RF. Ignored by the EDM arm, which has no RF velocity.",
    )
    ap.add_argument("--reflow-steps", type=int, default=None,
                    help="Training steps per ReFlow round.")
    ap.add_argument("--ae-steps", type=int, default=None)
    ap.add_argument(
        "--ae-lpips-weight", type=float, default=None,
        help="VGG LPIPS weight in AE reconstruction for every arm "
             "(default 0.1; 0 disables).",
    )
    ap.add_argument("--teacher-steps", type=int, default=None)
    ap.add_argument("--student-steps", type=int, default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument(
        "--out-root", default=None,
        help="Parent directory for all arms. Default: each arm's own results/cifar10/ path.",
    )
    ap.add_argument(
        "--workers", type=int, default=None,
        help="Run this many seeds concurrently. Sequential by default: the "
             "CIFAR-10 arms are memory-heavy and parallel seeds can exhaust the GPU.",
    )
    args = ap.parse_args()
    if args.workers is not None and args.workers < 1:
        ap.error("--workers must be at least 1")

    if args.seeds is not None:
        seeds, multi_seed = args.seeds, True
    elif args.seed is not None:
        seeds, multi_seed = [args.seed], False
    else:
        seeds, multi_seed = [0], False
    if len(set(seeds)) != len(seeds):
        ap.error("--seeds must not contain duplicates")

    workers = _worker_count(torch.cuda.is_available(), len(seeds), args.workers)
    all_outputs = {}
    if workers > 1:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context,
        ) as pool:
            futures = [
                pool.submit(_run_seed_worker, vars(args), seed, multi_seed)
                for seed in seeds
            ]
            for future in concurrent.futures.as_completed(futures):
                seed, outputs = future.result()
                all_outputs[seed] = outputs
    else:
        for seed in seeds:
            all_outputs[seed] = _run_seed(args, seed, multi_seed=multi_seed)

    print("\n" + "=" * 60)
    print("All CIFAR-10 arms complete. Compare the image grids in:")
    print("=" * 60)
    for seed in sorted(all_outputs):
        for pipeline in PIPELINES:
            print(f"  seed {seed}  {PIPELINE_LABELS[pipeline]:20s} {all_outputs[seed][pipeline]}")


if __name__ == "__main__":
    main()
