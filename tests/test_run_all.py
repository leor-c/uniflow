import unittest
from argparse import Namespace

from toy.run_all import (
    _aggregate,
    _configure_multi_seed_outputs,
    _render_markdown_summary,
    _worker_count,
)


class _Config:
    metrics_out = ""
    out = ""
    save_figure = False


class RunAllAggregationTests(unittest.TestCase):
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
            1: {
                pipeline: {"teacher_swd": 1.0, "student_1_step_swd": 2.0}
                for pipeline in ("bounded_rf", "gaussian_rf", "edm_ayf")
            },
            2: {
                pipeline: {"teacher_swd": 3.0, "student_1_step_swd": 4.0}
                for pipeline in ("bounded_rf", "gaussian_rf", "edm_ayf")
            },
        }

        result = _aggregate(all_metrics)

        self.assertEqual(result["num_seeds"], 2)
        self.assertEqual(result["seeds"], [1, 2])
        teacher = result["pipelines"]["edm_ayf"]["teacher_swd"]
        self.assertEqual(teacher["values"], {"1": 1.0, "2": 3.0})
        self.assertEqual(teacher["mean"], 2.0)
        self.assertAlmostEqual(teacher["std"], 2**0.5)

    def test_markdown_summary_contains_aggregate_table(self):
        all_metrics = {
            seed: {
                pipeline: {
                    "teacher_swd": float(seed),
                    "student_1_step_swd": float(seed + 1),
                    "student_8_step_swd": float(seed + 2),
                }
                for pipeline in ("bounded_rf", "gaussian_rf", "edm_ayf")
            }
            for seed in (1, 2)
        }

        report = _render_markdown_summary(_aggregate(all_metrics), "8gaussians")

        self.assertIn("# 8-Gaussians benchmark results", report)
        self.assertIn("| Pipeline | Teacher | Student, 1 step | Student, 8 steps |", report)
        self.assertIn("| Bounded RF + AYF | 1.500 ± 0.707", report)
        self.assertIn("| EDM + AYF | 1.500 ± 0.707", report)
        self.assertIn("summary_8gaussians.json", report)


if __name__ == "__main__":
    unittest.main()
