import unittest
from argparse import Namespace

from toy.common import BENCHMARK_PROTOCOL, DENSITY_PROTOCOL
from toy.run_all import (
    METRIC_SUFFIXES,
    OUTPUT_PREFIXES,
    _aggregate,
    _apply_overrides,
    _configure_multi_seed_outputs,
    _has_current_metric_schema,
    _render_markdown_summary,
    _worker_count,
)


class _Config:
    dataset = "8gaussians"
    seed = 1
    latent_dim = 32
    ae_steps = 20_000
    teacher_steps = 10_000
    student_steps = 10_000
    metrics_out = ""
    out = ""
    save_figure = False


def _pipeline_metrics(value: float, *, protocol: bool = False) -> dict:
    metrics = {
        f"{output}_{suffix}": value
        for output in OUTPUT_PREFIXES
        for suffix in METRIC_SUFFIXES
    }
    if protocol:
        metrics["benchmark_protocol"] = BENCHMARK_PROTOCOL
        metrics["evaluation_protocol"] = DENSITY_PROTOCOL
    return metrics


def _args(**overrides):
    """A run_all argparse Namespace with every flag defaulted to "not supplied"."""
    fields = dict(
        dataset=None, latent_dim=None, ae_steps=None, teacher_steps=None,
        student_steps=None, train_mode=None, ff=None, ae_weight_decay=None,
        reflow_rounds=None, reflow_steps=None,
        teacher_lr=None, teacher_final_lr=None, teacher_lr_drop_step=None,
        student_lr=None, student_final_lr=None, student_lr_drop_step=None,
        ae_ff_scale=None, teacher_ff_scale=None, student_ff_scale=None,
        shared_teacher=False,
    )
    unknown = set(overrides) - set(fields)
    assert not unknown, f"unknown run_all flag(s): {sorted(unknown)}"
    fields.update(overrides)
    return Namespace(**fields)


class RunAllAggregationTests(unittest.TestCase):
    def test_learning_rate_overrides_reach_every_arm(self):
        from toy.run_all import PIPELINE_MODULES
        for module, teacher_field, student_field in PIPELINE_MODULES.values():
            cfg = _apply_overrides(
                module.Config(device="cpu"),
                _args(teacher_lr=2e-3, teacher_final_lr=2e-4,
                      teacher_lr_drop_step=20, student_lr_drop_step=-1),
                seed=1, teacher_field=teacher_field, student_field=student_field,
            )
            self.assertEqual(getattr(cfg, teacher_field.replace("_steps", "_lr")), 2e-3)
            self.assertEqual(cfg.teacher_final_lr, 2e-4)
            self.assertEqual(cfg.teacher_lr_drop_step, 20)
            self.assertIsNone(cfg.student_lr_drop_step)

    def test_resume_requires_matching_schedule_and_training_budget(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from toy.common import run_config_record
        from toy.run_all import PIPELINE_MODULES, _load_seed_metrics

        with tempfile.TemporaryDirectory() as tmp:
            paths = {name: Path(tmp) / f"{name}.json" for name in PIPELINE_MODULES}
            for name, (module, _, _) in PIPELINE_MODULES.items():
                metrics = _pipeline_metrics(0.1, protocol=True)
                metrics["run_config"] = run_config_record(module.Config(device="cpu"))
                paths[name].write_text(json.dumps(metrics))
            with patch("toy.run_all._seed_metrics_path", side_effect=lambda name, *a: paths[name]):
                self.assertIsNotNone(_load_seed_metrics("8gaussians", 1, _args()))
                self.assertIsNone(_load_seed_metrics("8gaussians", 1, _args(teacher_final_lr=5e-5)))
                self.assertIsNone(_load_seed_metrics("8gaussians", 1, _args(student_steps=20000)))
                self.assertIsNone(_load_seed_metrics("8gaussians", 1, _args(teacher_lr_drop_step=-1)))

    def test_shared_teacher_applies_only_to_rf_and_keeps_edm_reference(self):
        from toy.run_all import PIPELINE_MODULES, _seed_metrics_path
        args = _args(shared_teacher=True)
        for pipeline, (module, teacher_field, student_field) in PIPELINE_MODULES.items():
            cfg = _apply_overrides(
                module.Config(device="cpu"), args, seed=2,
                teacher_field=teacher_field, student_field=student_field,
            )
            if pipeline == "edm_ayf":
                self.assertFalse(hasattr(cfg, "shared_teacher"))
                self.assertEqual(
                    str(_seed_metrics_path(pipeline, "branching_trees", 2, args)),
                    "results/toy/branching_trees/edm_ayf/seed_2/metrics.json",
                )
            else:
                self.assertTrue(cfg.shared_teacher)
                self.assertEqual(
                    str(_seed_metrics_path(pipeline, "branching_trees", 2, args)),
                    f"results/toy/branching_trees/{pipeline}/experiments/"
                    "shared_teacher_40k_40k/seed_2/metrics.json",
                )

    def test_shared_overrides_include_latent_dimension(self):
        cfg = _Config()
        args = _args(latent_dim=8)

        result = _apply_overrides(
            cfg,
            args,
            seed=3,
            teacher_field="teacher_steps",
            student_field="student_steps",
        )

        self.assertIs(result, cfg)
        self.assertEqual(cfg.latent_dim, 8)
        self.assertEqual(cfg.seed, 3)

    def test_train_mode_override_reaches_every_arm(self):
        """--train-mode must apply to all arms, and default to each Config's own."""
        from toy.run_all import PIPELINE_MODULES, PIPELINES

        def resolve(mode):
            args = _args(train_mode=mode)
            out = {}
            for pipeline in PIPELINES:
                module, teacher_field, student_field = PIPELINE_MODULES[pipeline]
                cfg = _apply_overrides(
                    module.Config(device="cpu"), args, seed=1,
                    teacher_field=teacher_field, student_field=student_field,
                )
                out[pipeline] = cfg.train_mode
            return out

        self.assertEqual(set(resolve("lmd").values()), {"lmd"})
        self.assertEqual(set(resolve("emd+lmd").values()), {"emd+lmd"})
        # omitted -> each arm keeps its own default rather than being forced
        self.assertEqual(resolve(None), {p: "lmd" for p in PIPELINES})

    def test_fourier_override_reaches_every_arm_and_zero_disables(self):
        """run_all must be able to ablate the Fourier sites across all arms."""
        from toy.run_all import PIPELINE_MODULES, PIPELINES

        def resolve(value):
            args = _args(ae_ff_scale=value, teacher_ff_scale=value,
                         student_ff_scale=value)
            out = {}
            for pipeline in PIPELINES:
                module, teacher_field, student_field = PIPELINE_MODULES[pipeline][:3]
                cfg = _apply_overrides(
                    module.Config(device="cpu"), args, seed=1,
                    teacher_field=teacher_field, student_field=student_field,
                )
                out[pipeline] = (cfg.ae_ff_scale, cfg.teacher_ff_scale, cfg.student_ff_scale)
            return out

        # 0 disables every site on every arm
        self.assertEqual(set(resolve(0.0).values()), {(None, None, None)})
        # omitted keeps each arm's default, which is features ON
        for scales in resolve(None).values():
            self.assertTrue(all(s is not None for s in scales))

    def test_ff_spec_selects_sites_across_every_arm(self):
        """--ff ATS/AT/S must turn exactly the named sites on and the rest off."""
        from toy.run_all import PIPELINE_MODULES, PIPELINES

        def resolve(spec):
            args = _args(ff=spec)
            out = {}
            for pipeline in PIPELINES:
                module, teacher_field, student_field = PIPELINE_MODULES[pipeline][:3]
                cfg = _apply_overrides(
                    module.Config(device="cpu"), args, seed=1,
                    teacher_field=teacher_field, student_field=student_field,
                )
                out[pipeline] = tuple(
                    getattr(cfg, f"{site}_ff_scale") is not None
                    for site in ("ae", "teacher", "student")
                )
            return out

        for spec, expected in [
            ("ATS", (True, True, True)),
            ("AT", (True, True, False)),
            ("S", (False, False, True)),
            ("ats", (True, True, True)),
            ("none", (False, False, False)),
        ]:
            with self.subTest(spec=spec):
                self.assertEqual(set(resolve(spec).values()), {expected})

    def test_ff_spec_and_explicit_scale_compose(self):
        """--ff picks the sites; an explicit --*-ff-scale still sets the value."""
        from toy.run_all import PIPELINE_MODULES

        module, teacher_field, student_field = PIPELINE_MODULES["uniform_rf"][:3]
        cfg = _apply_overrides(
            module.Config(device="cpu"), _args(ff="AT", teacher_ff_scale=2.5),
            seed=1, teacher_field=teacher_field, student_field=student_field,
        )
        self.assertEqual(cfg.teacher_ff_scale, 2.5)
        self.assertIsNotNone(cfg.ae_ff_scale)
        self.assertIsNone(cfg.student_ff_scale)

    def test_ae_weight_decay_override_reaches_every_arm(self):
        """--ae-weight-decay must apply to all arms; 0 is a real value, not 'unset'."""
        from toy.run_all import PIPELINE_MODULES, PIPELINES

        def resolve(value):
            args = _args(ae_weight_decay=value)
            out = {}
            for pipeline in PIPELINES:
                module, teacher_field, student_field = PIPELINE_MODULES[pipeline][:3]
                cfg = _apply_overrides(
                    module.Config(device="cpu"), args, seed=1,
                    teacher_field=teacher_field, student_field=student_field,
                )
                out[pipeline] = cfg.ae_weight_decay
            return out

        self.assertEqual(set(resolve(0.0).values()), {0.0})
        self.assertEqual(set(resolve(0.05).values()), {0.05})
        # omitted -> each arm keeps its default, which is AdamW's own 0.01
        self.assertEqual(set(resolve(None).values()), {0.01})

    def test_ff_spec_rejects_unknown_letters(self):
        from core.modules import parse_ff_sites

        for bad in ("AX", "", "A S", "aa t"):
            with self.subTest(spec=bad):
                with self.assertRaises(ValueError):
                    parse_ff_sites(bad)
        # the AE site does not exist in the CIFAR-10 arms
        with self.assertRaises(ValueError):
            parse_ff_sites("ATS", allowed=("teacher", "student"))

    def test_every_pipeline_has_a_module_and_label(self):
        from toy.run_all import PIPELINE_LABELS, PIPELINE_MODULES, PIPELINES

        self.assertEqual(set(PIPELINE_MODULES), set(PIPELINES))
        self.assertEqual(set(PIPELINE_LABELS), set(PIPELINES))

    def test_cuda_parallelizes_all_pending_seeds_by_default(self):
        self.assertEqual(_worker_count(True, pending_seeds=5, requested=None), 5)

    def test_worker_limit_and_cpu_fallback(self):
        self.assertEqual(_worker_count(True, pending_seeds=5, requested=2), 2)
        self.assertEqual(_worker_count(False, pending_seeds=5, requested=5), 1)

    def test_plot_all_seeds_uses_run_directory(self):
        cfg = _Config()
        args = Namespace(plot_all_seeds=True, plot_seed=1)

        _configure_multi_seed_outputs(cfg, "edm_ayf", 3, args)

        self.assertTrue(cfg.save_figure)
        self.assertEqual(cfg.out, "results/toy/8gaussians/edm_ayf/seed_3/samples.png")
        self.assertEqual(cfg.metrics_out, "results/toy/{}/edm_ayf/seed_3/metrics.json")

    def test_unplotted_seed_still_has_its_own_run_directory(self):
        cfg = _Config()
        cfg.out = "results/edm_ayf_{}.png"
        args = Namespace(plot_all_seeds=False, plot_seed=1)

        _configure_multi_seed_outputs(cfg, "edm_ayf", 2, args)

        self.assertFalse(cfg.save_figure)
        self.assertEqual(cfg.out, "results/toy/8gaussians/edm_ayf/seed_2/samples.png")

    def test_selected_plot_seed_gets_stable_run_directory(self):
        cfg = _Config()
        cfg.out = "results/edm_ayf_{}.png"
        args = Namespace(plot_all_seeds=False, plot_seed=1)

        _configure_multi_seed_outputs(cfg, "edm_ayf", 1, args)

        self.assertTrue(cfg.save_figure)
        self.assertEqual(cfg.out, "results/toy/8gaussians/edm_ayf/seed_1/samples.png")

    def test_aggregate_reports_seed_values_mean_and_sample_std(self):
        all_metrics = {
            seed: {
                pipeline: _pipeline_metrics(float(2 * seed - 1))
                for pipeline in ("uniform_rf", "gaussian_rf", "edm_ayf")
            }
            for seed in (1, 2)
        }

        result = _aggregate(all_metrics)

        self.assertEqual(result["num_seeds"], 2)
        self.assertEqual(result["seeds"], [1, 2])
        teacher = result["pipelines"]["edm_ayf"]["teacher_target_nll"]
        self.assertEqual(teacher["values"], {"1": 1.0, "2": 3.0})
        self.assertEqual(teacher["mean"], 2.0)
        self.assertAlmostEqual(teacher["std"], 2**0.5)
        self.assertEqual(result["metric"], "two-sided density fidelity and coverage")
        self.assertEqual(result["metric_samples"], 2048)
        self.assertEqual(result["off_support_threshold"], "three target noise scales")

    def test_markdown_summary_contains_density_tables(self):
        all_metrics = {
            seed: {
                pipeline: _pipeline_metrics(float(seed))
                for pipeline in ("uniform_rf", "gaussian_rf", "edm_ayf")
            }
            for seed in (1, 2)
        }

        report = _render_markdown_summary(_aggregate(all_metrics), "two_moons")

        self.assertIn("# two moons benchmark results", report)
        self.assertIn("## Teacher", report)
        self.assertIn("## Student, 1 step", report)
        self.assertIn("Fidelity: target NLL ↓", report)
        self.assertIn("Coverage: KDE NLL ↓", report)
        self.assertIn("Off support (>3σ) ↓", report)
        self.assertIn("| Uniform RF + AYF | 1.500 ± 0.707", report)
        self.assertIn("150.0% ± 70.7%", report)
        self.assertIn("Aggregate target-NLL comparison", report)
        self.assertIn("summary.json", report)

        tree_summary = _aggregate(all_metrics, "branching_trees")
        tree_report = _render_markdown_summary(tree_summary, "branching_trees")
        self.assertEqual(
            tree_summary["off_support_threshold"], "three evaluation bandwidths"
        )
        self.assertIn("smoothed-line", tree_summary["evaluation_protocol"])
        self.assertIn("samples are drawn exactly on the branches", tree_report)

    def test_current_schema_rejects_mmd_metrics(self):
        old = {
            pipeline: {
                "evaluation_protocol": "biased-multiscale-rbf-mmd-uniform-without-replacement-v1",
                "ae_reconstruction_mmd": 0.1,
                "teacher_mmd": 0.2,
                "student_1_step_mmd": 0.3,
                "student_8_step_mmd": 0.4,
            }
            for pipeline in ("uniform_rf", "gaussian_rf", "edm_ayf")
        }

        self.assertFalse(_has_current_metric_schema(old))

    def test_current_schema_rejects_missing_coverage_metric(self):
        incomplete = {
            pipeline: _pipeline_metrics(0.1, protocol=True)
            for pipeline in ("uniform_rf", "gaussian_rf", "edm_ayf")
        }
        del incomplete["gaussian_rf"]["teacher_coverage_nll"]

        self.assertFalse(_has_current_metric_schema(incomplete))

    def test_current_schema_rejects_previous_benchmark_protocol(self):
        previous = {
            pipeline: _pipeline_metrics(0.1, protocol=True)
            for pipeline in ("uniform_rf", "gaussian_rf", "edm_ayf")
        }
        previous["gaussian_rf"]["benchmark_protocol"] = "previous-baseline"

        self.assertFalse(_has_current_metric_schema(previous))

    def test_current_schema_accepts_density_metrics(self):
        current = {
            pipeline: _pipeline_metrics(0.1, protocol=True)
            for pipeline in ("uniform_rf", "gaussian_rf", "edm_ayf")
        }

        self.assertTrue(_has_current_metric_schema(current))


if __name__ == "__main__":
    unittest.main()
