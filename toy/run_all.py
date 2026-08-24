"""
Run the three release benchmark arms: bounded RF + AYF, Gaussian RF + AYF,
and EDM + AYF.

Each experiment uses its own default Config. The --dataset, --seed/--seeds,
--ae_steps, --student_steps, and --teacher_steps flags are applied on top
of those defaults only when explicitly passed; any flag left unset keeps
each experiment's own default for that field.

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

from toy import bounded_rf, edm_baseline, gaussian_rf
from toy.common import save_metrics


PIPELINES = ("bounded_rf", "gaussian_rf", "edm_ayf")
PIPELINE_LABELS = {
    "bounded_rf": "Bounded RF + AYF",
    "gaussian_rf": "Gaussian RF + AYF",
    "edm_ayf": "EDM + AYF",
}


def _apply_overrides(cfg, args, seed: int, *, teacher_field: str, student_field: str):
    if args.dataset is not None:
        cfg.dataset = args.dataset
    cfg.seed = seed
    if args.ae_steps is not None:
        cfg.ae_steps = args.ae_steps
    if args.teacher_steps is not None:
        setattr(cfg, teacher_field, args.teacher_steps)
    if args.student_steps is not None:
        setattr(cfg, student_field, args.student_steps)
    return cfg


def _seed_metrics_path(pipeline: str, dataset: str, seed: int) -> Path:
    return Path("results/seeds") / f"{pipeline}_{dataset}_seed{seed}.json"


def _seed_metrics_pattern(pipeline: str, seed: int) -> str:
    return str(Path("results/seeds") / f"{pipeline}_{{}}_seed{seed}.json")


def _configure_multi_seed_outputs(cfg, pipeline: str, seed: int, args) -> None:
    cfg.metrics_out = _seed_metrics_pattern(pipeline, seed)
    cfg.save_figure = args.plot_all_seeds or seed == args.plot_seed
    if cfg.save_figure:
        cfg.out = str(Path("results") / f"{pipeline}_{{}}-seed-{seed}.png")


def _load_seed_metrics(dataset: str, seed: int):
    paths = {
        pipeline: _seed_metrics_path(pipeline, dataset, seed)
        for pipeline in PIPELINES
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    return {
        pipeline: json.loads(path.read_text())
        for pipeline, path in paths.items()
    }


def _run_seed(args, seed: int, *, multi_seed: bool):
    dataset = args.dataset or "8gaussians"
    print("\n" + "=" * 60)
    print(f"Running seed {seed}: toy.bounded_rf")
    print("=" * 60)
    cfg = _apply_overrides(
        bounded_rf.Config(), args, seed,
        teacher_field="teacher_steps", student_field="student_steps",
    )
    if multi_seed or args.plot_all_seeds:
        _configure_multi_seed_outputs(cfg, "bounded_rf", seed, args)
    bounded_metrics = bounded_rf.main(cfg)

    print("\n" + "=" * 60)
    print(f"Running seed {seed}: toy.gaussian_rf")
    print("=" * 60)
    cfg = _apply_overrides(
        gaussian_rf.Config(), args, seed,
        teacher_field="teacher_steps", student_field="student_steps",
    )
    if multi_seed or args.plot_all_seeds:
        _configure_multi_seed_outputs(cfg, "gaussian_rf", seed, args)
    gaussian_metrics = gaussian_rf.main(cfg)

    print("\n" + "=" * 60)
    print(f"Running seed {seed}: toy.edm_baseline")
    print("=" * 60)
    cfg = _apply_overrides(
        edm_baseline.Config(), args, seed,
        teacher_field="edm_teacher_steps", student_field="edm_student_steps",
    )
    if multi_seed or args.plot_all_seeds:
        _configure_multi_seed_outputs(cfg, "edm_ayf", seed, args)
    edm_metrics = edm_baseline.main(cfg)

    return {
        "bounded_rf": bounded_metrics,
        "gaussian_rf": gaussian_metrics,
        "edm_ayf": edm_metrics,
    }


def _aggregate(all_metrics: dict[int, dict]) -> dict:
    seeds = sorted(all_metrics)
    aggregate = {}
    for pipeline in PIPELINES:
        metric_names = sorted(all_metrics[seeds[0]][pipeline])
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
        "std_definition": "sample standard deviation (n - 1 denominator)",
        "pipelines": aggregate,
    }


def _report_metric_keys(summary: dict) -> list[str]:
    available = summary["pipelines"][PIPELINES[0]]
    student_keys = sorted(
        (key for key in available if key.startswith("student_") and key.endswith("_step_swd")),
        key=lambda key: int(key.split("_")[1]),
    )
    return ["teacher_swd", *student_keys]


def _metric_label(metric_key: str) -> str:
    if metric_key == "teacher_swd":
        return "Teacher"
    steps = metric_key.split("_")[1]
    return f"Student, {steps} step{'s' if steps != '1' else ''}"


def _observation_metric_label(metric_key: str) -> str:
    if metric_key == "teacher_swd":
        return "teacher"
    steps = metric_key.split("_")[1]
    return f"{steps}-step student"


def _render_markdown_summary(summary: dict, dataset: str) -> str:
    metric_keys = _report_metric_keys(summary)
    seeds = ", ".join(str(seed) for seed in summary["seeds"])
    dataset_title = dataset.replace("8gaussians", "8-Gaussians").replace("_", " ")

    lines = [
        f"# {dataset_title} benchmark results",
        "",
        f"Aggregated over **{summary['num_seeds']} seeds** ({seeds}). Values are mean ± sample",
        "standard deviation of sliced Wasserstein distance (SWD); lower is better.",
        "",
        "| Pipeline | " + " | ".join(_metric_label(key) for key in metric_keys) + " |",
        "|---|" + "---:|" * len(metric_keys),
    ]
    for pipeline in PIPELINES:
        cells = []
        for metric_key in metric_keys:
            metric = summary["pipelines"][pipeline][metric_key]
            cells.append(f"{metric['mean']:.3f} ± {metric['std']:.3f}")
        lines.append(f"| {PIPELINE_LABELS[pipeline]} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Aggregate comparison",
            "",
            f"![Aggregate SWD comparison](summary_{dataset}.png)",
            "",
            "Bars show means, error bars show ± one sample standard deviation, and points show individual seeds.",
            "",
            "## Observations",
            "",
        ]
    )
    for metric_key in metric_keys:
        best_pipeline = min(
            PIPELINES,
            key=lambda pipeline: summary["pipelines"][pipeline][metric_key]["mean"],
        )
        best = summary["pipelines"][best_pipeline][metric_key]["mean"]
        lines.append(
            f"- Lowest observed mean {_observation_metric_label(metric_key)} SWD: "
            f"{PIPELINE_LABELS[best_pipeline]} ({best:.3f})."
        )

    one_step_key = "student_1_step_swd"
    later_steps = [key for key in metric_keys if key.startswith("student_") and key != one_step_key]
    if one_step_key in metric_keys and later_steps:
        final_key = later_steps[-1]
        final_steps = final_key.split("_")[1]
        for pipeline in PIPELINES:
            one_step = summary["pipelines"][pipeline][one_step_key]["mean"]
            final_step = summary["pipelines"][pipeline][final_key]["mean"]
            direction = "improves" if final_step < one_step else "worsens"
            lines.append(
                f"- For {PIPELINE_LABELS[pipeline]}, {final_steps}-step sampling {direction} "
                f"the mean from {one_step:.3f} to {final_step:.3f}."
            )

    lines.extend(
        [
            "- These five-seed results are descriptive; overlapping standard deviations "
            "should not be interpreted as statistically significant differences.",
            "",
            f"Raw aggregate and per-seed values: [`summary_{dataset}.json`](summary_{dataset}.json).",
            "",
        ]
    )
    return "\n".join(lines)


def _save_markdown_summary(summary: dict, dataset: str) -> Path:
    path = Path("results") / f"summary_{dataset}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_markdown_summary(summary, dataset))
    print(f"Saved Markdown report -> {path}")
    return path


def _save_aggregate_plot(summary: dict, dataset: str) -> Path:
    metric_keys = _report_metric_keys(summary)
    group_positions = list(range(len(metric_keys)))
    width = 0.24
    colors = {
        "bounded_rf": "#4C78A8",
        "gaussian_rf": "#F58518",
        "edm_ayf": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for pipeline_index, pipeline in enumerate(PIPELINES):
        offset = (pipeline_index - (len(PIPELINES) - 1) / 2) * width
        positions = [position + offset for position in group_positions]
        means = [
            summary["pipelines"][pipeline][metric_key]["mean"]
            for metric_key in metric_keys
        ]
        stds = [
            summary["pipelines"][pipeline][metric_key]["std"]
            for metric_key in metric_keys
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

        for metric_position, metric_key in zip(positions, metric_keys):
            values = list(summary["pipelines"][pipeline][metric_key]["values"].values())
            if len(values) == 1:
                jitters = [0.0]
            else:
                spread = width * 0.42
                jitters = [
                    -spread / 2 + spread * index / (len(values) - 1)
                    for index in range(len(values))
                ]
            ax.scatter(
                [metric_position + jitter for jitter in jitters],
                values,
                s=16,
                color="#202020",
                alpha=0.72,
                linewidths=0,
                zorder=3,
            )

    dataset_title = dataset.replace("8gaussians", "8-Gaussians").replace("_", " ")
    ax.set_title(
        f"{dataset_title}: aggregate SWD across {summary['num_seeds']} seeds\n"
        "bars = mean, error bars = sample standard deviation, points = seeds"
    )
    ax.set_ylabel("Sliced Wasserstein distance (lower is better)")
    ax.set_xticks(group_positions, [_metric_label(key) for key in metric_keys])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    path = Path("results") / f"summary_{dataset}.png"
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
                    choices=["8gaussians", "two_moons", "checkerboard"])
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
    ap.add_argument("--ae_steps", type=int, default=None)
    ap.add_argument("--teacher_steps", type=int, default=None)
    ap.add_argument("--student_steps", type=int, default=None)
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
        metrics = _load_seed_metrics(dataset, seed) if multi_seed and args.resume else None
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

    aggregate = _aggregate(all_metrics)
    summary = aggregate if multi_seed else all_metrics[seeds[0]]
    save_metrics(summary, f"results/summary_{dataset}.json")
    _save_aggregate_plot(aggregate, dataset)
    _save_markdown_summary(aggregate, dataset)


if __name__ == "__main__":
    main()
