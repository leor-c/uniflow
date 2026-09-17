"""Output layout shared by individual toy runs and the multi-seed runner."""
from dataclasses import asdict
import json
from pathlib import Path


RESULTS_ROOT = Path("results/toy")


def run_directory(dataset: str, baseline: str, seed: int) -> Path:
    return RESULTS_ROOT / dataset / baseline / f"seed_{seed}"


def configure_outputs(cfg, baseline: str) -> None:
    """Resolve default paths without overriding explicitly supplied filenames."""
    explicit_figure = bool(cfg.out)
    directory = Path(cfg.out_dir) if getattr(cfg, "out_dir", None) else (
        Path(cfg.out.format(cfg.dataset)).parent if explicit_figure
        else run_directory(cfg.dataset, baseline, cfg.seed)
    )
    if not cfg.out:
        cfg.out = str(directory / "samples.png")
    if not cfg.metrics_out:
        cfg.metrics_out = (
            str(Path(cfg.out).with_suffix(".json")) if explicit_figure
            else str(directory / "metrics.json")
        )
    cfg.out_dir = str(directory)


def save_run_config(cfg) -> None:
    metrics = Path(cfg.metrics_out.format(cfg.dataset))
    name = "run_config.json" if metrics.name == "metrics.json" else f"{metrics.stem}_config.json"
    path = metrics.with_name(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n")
