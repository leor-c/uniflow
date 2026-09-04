import unittest

import torch

from toy.common import (
    _uniform_subsample_rows,
    clean_support_distance,
    kde_coverage_negative_log_likelihood,
    off_support_rate,
    target_negative_log_likelihood,
)


class DensityMetricTests(unittest.TestCase):
    def test_target_nll_penalizes_points_away_from_eight_gaussian_modes(self):
        on_mode = torch.tensor([[0.75, 0.0]]).repeat(32, 1)
        between_modes = torch.tensor([[0.5, 0.2]]).repeat(32, 1)

        self.assertLess(
            target_negative_log_likelihood("8gaussians", on_mode),
            target_negative_log_likelihood("8gaussians", between_modes),
        )

    def test_target_nll_penalizes_two_moons_gap(self):
        on_arc = torch.tensor([[-0.25, 0.25]]).repeat(32, 1)
        in_gap = torch.tensor([[0.0, -0.05]]).repeat(32, 1)

        self.assertLess(
            target_negative_log_likelihood("two_moons", on_arc),
            target_negative_log_likelihood("two_moons", in_gap),
        )

    def test_coverage_nll_penalizes_translated_generated_samples(self):
        generator = torch.Generator().manual_seed(3)
        real = 0.035 * torch.randn(256, 2, generator=generator)
        matching = 0.035 * torch.randn(256, 2, generator=generator)

        matched_nll = kde_coverage_negative_log_likelihood(
            "8gaussians", real, matching
        )
        translated_nll = kde_coverage_negative_log_likelihood(
            "8gaussians", real, matching + 0.5
        )

        self.assertLess(matched_nll, translated_nll)

    def test_two_moons_support_distance_handles_arcs_and_wrong_sides(self):
        points = torch.tensor(
            [
                [-0.25, 0.25],  # top of first arc
                [0.25, -0.5],   # bottom of second arc
                [-0.25, -0.75], # radial continuation on wrong side
            ]
        )

        distances = clean_support_distance("two_moons", points)

        torch.testing.assert_close(distances[:2], torch.zeros(2, dtype=torch.float64))
        self.assertGreater(distances[2].item(), 0.4)

    def test_off_support_rate_uses_three_noise_scales(self):
        points = torch.tensor(
            [
                [-0.25, 0.25],
                [-0.25, 0.35],
                [-0.25, 0.45],
            ]
        )

        self.assertAlmostEqual(off_support_rate("two_moons", points), 1.0 / 3.0)

    def test_ordered_modes_are_uniformly_subsampled(self):
        ordered_modes = torch.cat(
            [torch.zeros(2500, 1), torch.ones(2500, 1)], dim=0
        )

        subset = _uniform_subsample_rows(ordered_modes, 2048, seed=0)

        second_mode_fraction = subset.mean().item()
        self.assertGreater(second_mode_fraction, 0.45)
        self.assertLess(second_mode_fraction, 0.55)
        torch.testing.assert_close(
            subset,
            _uniform_subsample_rows(ordered_modes, 2048, seed=0),
        )

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            target_negative_log_likelihood("two_moons", torch.empty(0, 2))


if __name__ == "__main__":
    unittest.main()
