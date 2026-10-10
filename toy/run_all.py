"""
Run the three release benchmark arms: Uniform RF + AYF, Gaussian RF + AYF,
and EDM + AYF. With --teacher-init, the RF students start from their
teachers, while EDM remains a scratch-initialized reference.

Each experiment uses its own default Config. The --dataset, --seed/--seeds,
--latent_dim, --ae_steps, --student_steps, and --teacher_steps flags are
applied on top of those defaults only when explicitly passed; any flag left
unset keeps each experiment's own default for that field.

Usage:
    python -m toy.run_all
    python -m toy.run_all --dataset two_moons --seed 0 --teacher_steps 2000 --student_steps 2000
    python -m toy.run_all --seeds 1 2 3 4 5 --resume
    python -m toy.run_all --plot-all-seeds
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from core.modules import apply_ff_spec
from toy import edm_baseline, gaussian_rf, uniform_rf
from toy.outputs import RESULTS_ROOT, run_directory
from toy.lr_schedule import add_lr_arguments, apply_lr_overrides
from toy.common import (
    BENCHMARK_PROTOCOL,
    evaluation_protocol,
    DENSITY_SAMPLES,
    TARGET_NOISE_SCALES,
    TARGET_EVALUATION_BANDWIDTHS,
    save_metrics,
    run_config_record,
)


PIPELINES = ("uniform_rf", "gaussian_rf", "edm_ayf")
PIPELINE_LABELS = {
    "uniform_rf": "Uniform RF + AYF",
    "gaussian_rf": "Gaussian RF + AYF",
    "edm_ayf": "EDM + AYF",
}
# result key -> (module, teacher-steps field, student-steps field)
PIPELINE_MODULES = {
    "uniform_rf": (uniform_rf, "teacher_steps", "student_steps"),
    "gaussian_rf": (gaussian_rf, "teacher_steps", "student_steps"),
    "edm_ayf": (edm_baseline, "edm_teacher_steps", "student_steps"),
}
OUTPUT_PREFIXES = (
    "ae_reconstruction",
    "teacher",
    "student_1_step",
    "student_8_step",
)
METRIC_SUFFIXES = (
    "target_nll",
    "coverage_nll",
    "off_support_rate",
)
METRIC_LABELS = {
    "target_nll": "Fidelity: target NLL ↓",
    "coverage_nll": "Coverage: KDE NLL ↓",
    "off_support_rate": "Off support (>3σ) ↓",
}
REQUIRED_METRIC_KEYS = {
    f"{output}_{metric}"
    for output in OUTPUT_PREFIXES
    for metric in METRIC_SUFFIXES
}


def _apply_overrides(cfg, args, seed: int, *, teacher_field: str, student_field: str):
    if args.dataset is not None:
        cfg.dataset = args.dataset
    cfg.seed = seed
    if args.latent_dim is not None:
        cfg.latent_dim = args.latent_dim
    if args.train_mode is not None:
        cfg.train_mode = args.train_mode
    if args.ff is not None:
        apply_ff_spec(cfg, args.ff)
    for site in ("ae", "teacher", "student"):
        value = getattr(args, f"{site}_ff_scale")
        if value is not None:
            # argparse None means "not supplied"; a non-positive value is the
            # explicit way to switch a site off, since Config uses None for that.
            setattr(cfg, f"{site}_ff_scale", value if value > 0.0 else None)
    # The EDM arm trains a denoiser, not an RF velocity, so it has no reflow
    # field -- skip it rather than inventing one.
    for field in ("reflow_rounds", "reflow_steps"):
        value = getattr(args, field)
        if value is not None and hasattr(cfg, field):
            setattr(cfg, field, value)
    if args.ae_weight_decay is not None:
        cfg.ae_weight_decay = args.ae_weight_decay
    if args.ae_steps is not None:
        cfg.ae_steps = args.ae_steps
    if args.teacher_steps is not None:
        setattr(cfg, teacher_field, args.teacher_steps)
    if args.student_steps is not None:
        setattr(cfg, student_field, args.student_steps)
    apply_lr_overrides(cfg, args, teacher_lr_field=teacher_field.replace("_steps", "_lr"))
    # EDM has a denoising teacher, so only the two RF arms can copy teacher
    # weights into the student. The EDM arm remains a scratch reference.
    if hasattr(cfg, "shared_teacher"):
        cfg.shared_teacher = getattr(args, "shared_teacher", False)
    return cfg


def _seed_metrics_path(pipeline: str, dataset: str, seed: int, args=None) -> Path:
    if getattr(args, "shared_teacher", False) and pipeline != "edm_ayf":
        return (RESULTS_ROOT / dataset / pipeline / "experiments" /
                "shared_teacher_40k_40k" / f"seed_{seed}" / "metrics.json")
    return run_directory(dataset, pipeline, seed) / "metrics.json"


def _seed_metrics_pattern(pipeline: str, seed: int, args=None) -> str:
    return str(_seed_metrics_path(pipeline, "{}", seed, args))


def _configure_multi_seed_outputs(cfg, pipeline: str, seed: int, args) -> None:
    metrics_path = _seed_metrics_path(pipeline, cfg.dataset, seed, args)
    cfg.metrics_out = _seed_metrics_pattern(pipeline, seed, args)
    cfg.save_figure = args.plot_all_seeds or seed == args.plot_seed
    cfg.out_dir = str(metrics_path.parent)
    cfg.out = str(metrics_path.parent / "samples.png")


def _has_current_metric_schema(metrics: dict, dataset: str | None = None) -> bool:
    return all(
        REQUIRED_METRIC_KEYS.issubset(pipeline_metrics)
        and pipeline_metrics.get("benchmark_protocol") == BENCHMARK_PROTOCOL
        and pipeline_metrics.get("evaluation_protocol") == evaluation_protocol(dataset)
        for pipeline_metrics in metrics.values()
    )


def _load_pipeline_metrics(pipeline: str, dataset: str, seed: int, args=None):
    path = _seed_metrics_path(pipeline, dataset, seed, args)
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text())
    if (not REQUIRED_METRIC_KEYS.issubset(metrics)
            or metrics.get("benchmark_protocol") != BENCHMARK_PROTOCOL
            or metrics.get("evaluation_protocol") != evaluation_protocol(dataset)):
        return None
    module, teacher_field, student_field = PIPELINE_MODULES[pipeline]
    cfg = module.Config(dataset=dataset, seed=seed)
    if args is not None:
        _apply_overrides(cfg, args, seed, teacher_field=teacher_field,
                         student_field=student_field)
    expected = run_config_record(cfg)
    saved = metrics.get("run_config", {})
    if any(key not in saved or saved[key] != value
           for key, value in expected.items() if key != "log_every"):
        return None
    return metrics


def _load_seed_metrics(dataset: str, seed: int, args=None):
    metrics = {
        pipeline: _load_pipeline_metrics(pipeline, dataset, seed, args)
        for pipeline in PIPELINES
    }
    return metrics if all(value is not None for value in metrics.values()) else None


def _run_seed(args, seed: int, *, multi_seed: bool):
    metrics = {}
    for pipeline in PIPELINES:
        module, teacher_field, student_field = PIPELINE_MODULES[pipeline]
        print("\n" + "=" * 60)
        print(f"Running seed {seed}: toy.{module.__name__.rsplit('.', 1)[-1]}")
        print("=" * 60)
        cfg = _apply_overrides(
            module.Config(), args, seed,
            teacher_field=teacher_field, student_field=student_field,
        )
        if multi_seed or args.plot_all_seeds or args.shared_teacher:
            _configure_multi_seed_outputs(cfg, pipeline, seed, args)
        if args.resume or (args.shared_teacher and pipeline == "edm_ayf"):
            saved = _load_pipeline_metrics(pipeline, cfg.dataset, seed, args)
            if saved is not None:
                print(f"Reusing {pipeline} metrics for seed {seed}")
                metrics[pipeline] = saved
                continue
            if args.shared_teacher and pipeline == "edm_ayf":
                raise FileNotFoundError(
                    f"EDM scratch reference missing or configuration mismatch: "
                    f"{_seed_metrics_path(pipeline, cfg.dataset, seed, args)}"
                )
        metrics[pipeline] = module.main(cfg)
    return metrics


def _aggregate(all_metrics: dict[int, dict], dataset: str | None = None,
               teacher_init: bool = False) -> dict:
    seeds = sorted(all_metrics)
    aggregate = {}
    for pipeline in PIPELINES:
        metric_names = sorted(
            key
            for key in all_metrics[seeds[0]][pipeline]
            if key in REQUIRED_METRIC_KEYS
        )
        aggregate[pipeline] = {}
        for metric_name in metric_names:
            values = [all_metrics[seed][pipeline][metric_name] for seed in seeds]
            aggregate[pipeline][metric_name] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": {str(seed): value for seed, value in zip(seeds, values)},
            }
    return {
        "num_seeds": len(seeds),
        "seeds": seeds,
        "teacher_init": teacher_init,
        "metric": "two-sided density fidelity and coverage",
        "metric_samples": DENSITY_SAMPLES,
        "metric_sampling": "uniform without replacement",
        "target_density": (
            "arc-length tree measure with evaluation-only Gaussian smoothing"
            if dataset == "branching_trees"
            else "known analytic mixture (arc mixture by midpoint quadrature)"
        ),
        "coverage_density": "generated-sample isotropic Gaussian KDE",
        "target_noise_scales": TARGET_NOISE_SCALES,
        "evaluation_bandwidths": TARGET_EVALUATION_BANDWIDTHS,
        "off_support_threshold": (
            "three evaluation bandwidths"
            if dataset == "branching_trees" else "three target noise scales"
        ),
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "evaluation_protocol": evaluation_protocol(dataset),
        "std_definition": "sample standard deviation (n - 1 denominator)",
        "pipelines": aggregate,
    }


def _report_output_prefixes(summary: dict) -> list[str]:
    available = summary["pipelines"][PIPELINES[0]]
    student_prefixes = sorted(
        {
            key.removesuffix(f"_{metric}")
            for key in available
            for metric in METRIC_SUFFIXES
            if key.startswith("student_") and key.endswith(f"_{metric}")
        },
        key=lambda prefix: int(prefix.split("_")[1]),
    )
    return ["teacher", *student_prefixes]


def _output_label(output_prefix: str) -> str:
    if output_prefix == "teacher":
        return "Teacher"
    if output_prefix == "ae_reconstruction":
        return "AE reconstruction"
    steps = output_prefix.split("_")[1]
    return f"Student, {steps} step{'s' if steps != '1' else ''}"


def _format_metric(metric: dict, suffix: str) -> str:
    if suffix == "off_support_rate":
        return f"{100.0 * metric['mean']:.1f}% ± {100.0 * metric['std']:.1f}%"
    return f"{metric['mean']:.3f} ± {metric['std']:.3f}"


def _render_markdown_summary(summary: dict, dataset: str) -> str:
    output_prefixes = _report_output_prefixes(summary)
    seeds = ", ".join(str(seed) for seed in summary["seeds"])
    dataset_title = dataset.replace("8gaussians", "8-Gaussians").replace("_", " ")

    density_description = (
        "For branching trees, target NLL uses Gaussian smoothing only for evaluation; "
        "samples are drawn exactly on the branches. Off-support uses three evaluation "
        f"bandwidths ({3.0 * TARGET_EVALUATION_BANDWIDTHS[dataset]:g}) as its distance threshold."
        if dataset == "branching_trees"
        else "Generated-sample NLL uses the known target density. Off-support is the "
        "fraction farther than three target noise scales "
        f"({3.0 * TARGET_NOISE_SCALES[dataset]:g}) from clean support."
    )

    lines = [
        f"# {dataset_title} benchmark results",
        "",
        f"Aggregated over **{summary['num_seeds']} seeds** ({seeds}). Values are mean ± sample",
        f"standard deviation on {DENSITY_SAMPLES:,} uniformly sampled points.",
        density_description,
        "Coverage is held-out target NLL under a generated-sample KDE.",
        "All metrics are lower-is-better and should be interpreted together.",
        "",
    ]
    if summary.get("teacher_init"):
        lines.extend([
            "Uniform RF and Gaussian RF students start from their respective",
            "teachers. EDM remains a scratch-initialized reference because its",
            "denoising teacher weights cannot be copied into the current student.",
            "The teacher-init variant also changes the RF teacher architecture,",
            "so this is not an isolated test of weight initialization.",
            "",
        ])

    for output_prefix in output_prefixes:
        lines.extend(
            [
                f"## {_output_label(output_prefix)}",
                "",
                "| Pipeline | "
                + " | ".join(METRIC_LABELS[suffix] for suffix in METRIC_SUFFIXES)
                + " |",
                "|---|---:|---:|---:|",
            ]
        )
        for pipeline in PIPELINES:
            cells = [
                _format_metric(
                    summary["pipelines"][pipeline][f"{output_prefix}_{suffix}"],
                    suffix,
                )
                for suffix in METRIC_SUFFIXES
            ]
            lines.append(f"| {PIPELINE_LABELS[pipeline]} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(
        [
            "## AE reconstruction reference",
            "",
            "The AE row provides a near-data calibration point for finite-sample and density-estimation effects.",
            "",
            "| Pipeline | "
            + " | ".join(METRIC_LABELS[suffix] for suffix in METRIC_SUFFIXES)
            + " |",
            "|---|---:|---:|---:|",
        ]
    )
    for pipeline in PIPELINES:
        cells = [
            _format_metric(
                summary["pipelines"][pipeline][f"ae_reconstruction_{suffix}"],
                suffix,
            )
            for suffix in METRIC_SUFFIXES
        ]
        lines.append(f"| {PIPELINE_LABELS[pipeline]} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Aggregate comparison",
            "",
            "![Aggregate target-NLL comparison](summary.png)",
            "",

            "## Observations",
            "",
        ]
    )
    for output_prefix in output_prefixes:
        winners = []
        for suffix in METRIC_SUFFIXES:
            metric_key = f"{output_prefix}_{suffix}"
            best_pipeline = min(
                PIPELINES,
                key=lambda pipeline: summary["pipelines"][pipeline][metric_key]["mean"],
            )
            winners.append(
                f"{METRIC_LABELS[suffix].replace(' ↓', '')}: {PIPELINE_LABELS[best_pipeline]}"
            )
        lines.append(f"- {_output_label(output_prefix)} — " + "; ".join(winners) + ".")

    lines.extend(
        [
            "- These five-seed results are descriptive; overlapping standard deviations "
            "should not be interpreted as statistically significant differences.",
            "",
            "Raw aggregate and per-seed values: [`summary.json`](summary.json).",
            "",
        ]
    )
    return "\n".join(lines)


def _save_markdown_summary(summary: dict, dataset: str, output_dir: Path | None = None) -> Path:
    path = (output_dir or RESULTS_ROOT / dataset) / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_markdown_summary(summary, dataset))
    print(f"Saved Markdown report -> {path}")
    return path


def _save_aggregate_plot(summary: dict, dataset: str, output_dir: Path | None = None) -> Path:
    output_prefixes = _report_output_prefixes(summary)
    group_positions = list(range(len(output_prefixes)))
    width = 0.24
    metric_suffix = "target_nll"
    colors = {
        "uniform_rf": "#4C78A8",
        "gaussian_rf": "#F58518",
        "edm_ayf": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for pipeline_index, pipeline in enumerate(PIPELINES):
        offset = (pipeline_index - (len(PIPELINES) - 1) / 2) * width
        positions = [position + offset for position in group_positions]
        metric_keys = [f"{output}_{metric_suffix}" for output in output_prefixes]
        means = [
            summary["pipelines"][pipeline][key]["mean"]
            for key in metric_keys
        ]
        stds = [
            summary["pipelines"][pipeline][key]["std"]
            for key in metric_keys
        ]
        ax.bar(
            positions,
            means,
            width,
            yerr=stds,
            capsize=4,
            color=colors[pipeline],
            edgecolor="white",
            linewidth=0.8,
            label=PIPELINE_LABELS[pipeline],
            zorder=2,
        )
        for position, metric_key in zip(positions, metric_keys):
            values = list(
                summary["pipelines"][pipeline][metric_key]["values"].values()
            )
            spread = width * 0.42
            jitters = [0.0] if len(values) == 1 else [
                -spread / 2 + spread * index / (len(values) - 1)
                for index in range(len(values))
            ]
            ax.scatter(
                [position + jitter for jitter in jitters],
                values,
                s=16,
                color="#202020",
                alpha=0.72,
                linewidths=0,
                zorder=3,
            )

    dataset_title = dataset.replace("8gaussians", "8-Gaussians").replace("_", " ")
    ax.set_title(f"{dataset_title}: sample fidelity")
    ax.set_ylabel("Target NLL of generated samples ↓")
    ax.set_xticks(group_positions, [_output_label(key) for key in output_prefixes])
    ax.grid(axis="y", alpha=0.25, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    path = (output_dir or RESULTS_ROOT / dataset) / "summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved aggregate plot -> {path}")
    return path


def _run_seed_worker(args_dict: dict, seed: int, multi_seed: bool):
    """Spawn-safe wrapper for one independent seed run."""
    args = argparse.Namespace(**args_dict)
    return seed, _run_seed(args, seed, multi_seed=multi_seed)


def _worker_count(cuda_available: bool, pending_seeds: int, requested: int | None) -> int:
    if not cuda_available or pending_seeds <= 1:
        return 1
    return min(requested or pending_seeds, pending_seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    choices=["8gaussians", "two_moons", "checkerboard", "branching_trees"])
    seed_group = ap.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None)
    seed_group.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--plot-seed", type=int, default=1,
                    help="Only this seed writes qualitative figures in multi-seed mode.")
    ap.add_argument("--plot-all-seeds", action="store_true",
                    help="Write one suffixed qualitative figure per pipeline and seed.")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse complete per-seed metric files when available.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Concurrent seed processes on CUDA (default: all seeds).")
    ap.add_argument("--latent-dim", type=int, default=None,
                    help="Shared autoencoder latent dimension for all pipelines.")
    ap.add_argument(
        "--train-mode", default=None, choices=["emd", "lmd", "emd+lmd"],
        help="AYF objective applied to every arm, same grammar as the "
             "individual arms. Omit to keep each arm's Config default. "
             "('ct' is uniform-only; use python -m toy.uniform_rf for it.)",
    )
    for _site, _where in (("ae", "the encoder input"),
                          ("teacher", "the teacher state"),
                          ("student", "the student state")):
        ap.add_argument(
            f"--{_site}-ff-scale", type=float, default=None,
            help=f"Fourier feature frequency scale for {_where}, applied to "
                 "every arm. Omit to use each arm's Config default (features "
                 "ON); pass 0 to disable that site.",
        )
    ap.add_argument(
        "--ff", default=None,
        help="Which Fourier sites to enable in EVERY arm, as a letter string: "
             "A=encoder input, T=teacher state, S=student state (e.g. 'ATS', "
             "'AT', 'S'). 'none' disables all. Sites left out are switched OFF. "
             "Omit to keep each arm's default. This is the knob for a "
             "controlled Fourier ablation across the benchmark.",
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
    ap.add_argument("--teacher-steps", type=int, default=None)
    ap.add_argument("--student-steps", type=int, default=None)
    ap.add_argument("--teacher-init", dest="shared_teacher", action="store_true",
                    help="Initialize RF students from their teachers; use the existing "
                         "scratch EDM run as reference. Save results separately.")
    ap.add_argument("--shared-teacher", dest="shared_teacher", action="store_true",
                    help=argparse.SUPPRESS)
    add_lr_arguments(ap)
    args = ap.parse_args()
    if args.plot_all_seeds and args.resume:
        ap.error(
            "--plot-all-seeds cannot be combined with --resume because "
            "checkpoints are not saved"
        )
    if args.workers is not None and args.workers < 1:
        ap.error("--workers must be at least 1")

    dataset = args.dataset or "8gaussians"
    if args.seeds is not None:
        seeds = args.seeds
        multi_seed = True
    elif args.seed is not None:
        seeds = [args.seed]
        multi_seed = False
    else:
        seeds = [1, 2, 3, 4, 5]
        multi_seed = True
    if len(set(seeds)) != len(seeds):
        ap.error("--seeds must not contain duplicates")

    all_metrics = {}
    pending_seeds = []
    for seed in seeds:
        metrics = _load_seed_metrics(dataset, seed, args) if multi_seed and args.resume else None
        if metrics is not None:
            print(f"Reusing complete metrics for seed {seed}")
            all_metrics[seed] = metrics
        else:
            pending_seeds.append(seed)

    workers = _worker_count(
        torch.cuda.is_available(), len(pending_seeds), args.workers
    )
    if workers > 1:
        print(
            f"Running {len(pending_seeds)} seeds across {workers} parallel "
            "CUDA worker processes"
        )
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    _run_seed_worker, vars(args), seed, multi_seed
                ): seed
                for seed in pending_seeds
            }
            for future in concurrent.futures.as_completed(futures):
                seed, metrics = future.result()
                all_metrics[seed] = metrics
                print(f"Completed seed {seed}")
    else:
        if len(pending_seeds) > 1 and not torch.cuda.is_available():
            print("CUDA is unavailable; running seeds sequentially")
        for seed in pending_seeds:
            all_metrics[seed] = _run_seed(args, seed, multi_seed=multi_seed)

    aggregate = _aggregate(all_metrics, dataset, teacher_init=args.shared_teacher)
    summary = aggregate if multi_seed else all_metrics[seeds[0]]
    output_dir = (RESULTS_ROOT / dataset / "experiments" / "shared_teacher_40k_40k"
                  if args.shared_teacher else RESULTS_ROOT / dataset)
    save_metrics(summary, str(output_dir / "summary.json"))
    _save_aggregate_plot(aggregate, dataset, output_dir)
    _save_markdown_summary(aggregate, dataset, output_dir)


if __name__ == "__main__":
    main()
