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


class RunAllAggregationTests(unittest.TestCase):
    def test_shared_overrides_include_latent_dimension(self):
        cfg = _Config()
        args = Namespace(
            dataset=None,
            latent_dim=8,
            ae_steps=None,
            teacher_steps=None,
            student_steps=None,
            train_mode=None,
        )

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
            args = Namespace(dataset=None, latent_dim=None, ae_steps=None,
                             teacher_steps=None, student_steps=None, train_mode=mode)
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
        self.assertEqual(resolve(None), {p: "emd" for p in PIPELINES})

    def test_every_pipeline_has_a_module_and_label(self):
        from toy.run_all import PIPELINE_LABELS, PIPELINE_MODULES, PIPELINES

        self.assertEqual(set(PIPELINE_MODULES), set(PIPELINES))
        self.assertEqual(set(PIPELINE_LABELS), set(PIPELINES))

    def test_cuda_parallelizes_all_pending_seeds_by_default(self):
        self.assertEqual(_worker_count(True, pending_seeds=5, requested=None), 5)

    def test_worker_limit_and_cpu_fallback(self):
        self.assertEqual(_worker_count(True, pending_seeds=5, requested=2), 2)
        self.assertEqual(_worker_count(False, pending_seeds=5, requested=5), 1)

    def test_plot_all_seeds_adds_seed_suffix(self):
        cfg = _Config()
        args = Namespace(plot_all_seeds=True, plot_seed=1)

        _configure_multi_seed_outputs(cfg, "edm_ayf", 3, args)

        self.assertTrue(cfg.save_figure)
        self.assertEqual(cfg.out, "results/edm_ayf_{}-seed-3.png")
        self.assertEqual(cfg.metrics_out, "results/seeds/edm_ayf_{}_seed3.json")

    def test_single_plot_seed_keeps_release_figure_name(self):
        cfg = _Config()
        cfg.out = "results/edm_ayf_{}.png"
        args = Namespace(plot_all_seeds=False, plot_seed=1)

        _configure_multi_seed_outputs(cfg, "edm_ayf", 2, args)

        self.assertFalse(cfg.save_figure)
        self.assertEqual(cfg.out, "results/edm_ayf_{}.png")

    def test_selected_plot_seed_gets_stable_seed_suffix(self):
        cfg = _Config()
        cfg.out = "results/edm_ayf_{}.png"
        args = Namespace(plot_all_seeds=False, plot_seed=1)

        _configure_multi_seed_outputs(cfg, "edm_ayf", 1, args)

        self.assertTrue(cfg.save_figure)
        self.assertEqual(cfg.out, "results/edm_ayf_{}-seed-1.png")

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
        self.assertIn("summary_two_moons.json", report)

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
