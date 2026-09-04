import os
import unittest
from types import SimpleNamespace

import torch

from toy.common import ToyAE, train_toy_ae


class ToyAutoencoderTests(unittest.TestCase):
    def test_unbounded_latent_normalization_preserves_reconstruction(self):
        torch.manual_seed(0)
        ae = ToyAE(latent_dim=4, hidden=16, depth=2, bounded=False)
        data = torch.randn(512, 2)

        reconstruction_before, _ = ae(data)
        ae.fit_latent_normalization(data)
        reconstruction_after, latents = ae(data)

        torch.testing.assert_close(
            latents.mean(dim=0), torch.zeros(4), atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            latents.std(correction=0), torch.tensor(1.0), atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            reconstruction_after, reconstruction_before, atol=1e-6, rtol=1e-6
        )

    def test_latent_normalization_is_isotropic(self):
        """
        The scale must be a single scalar, so the transform is a similarity and
        the latent manifold reaches the generative model undistorted.
        Per-coordinate whitening would rescale each axis separately, amplifying
        the lowest-variance encoder directions -- which the multi-step teacher
        tolerates but the one-step AYF student does not.
        """
        torch.manual_seed(0)
        ae = ToyAE(latent_dim=8, hidden=16, depth=2, bounded=False)
        data = torch.randn(512, 2)

        raw = ae.encoder_net(data).detach()
        # A deliberately anisotropic encoder output: one near-collapsed axis.
        self.assertGreater(
            raw.std(dim=0).max().item() / raw.std(dim=0).min().item(), 1.0
        )

        ae.fit_latent_normalization(data)
        torch.testing.assert_close(
            ae.latent_std, ae.latent_std[0].expand(8), atol=0.0, rtol=0.0
        )

        latents = ae.encode(data).detach()
        before = torch.pdist(raw)
        after = torch.pdist(latents)
        ratio = after / before
        torch.testing.assert_close(
            ratio, ratio[0].expand_as(ratio), atol=1e-4, rtol=0.0
        )

    def test_latent_normalization_rejects_a_collapsed_latent_space(self):
        ae = ToyAE(latent_dim=4, hidden=16, depth=2, bounded=False)
        with torch.no_grad():
            for parameter in ae.encoder_net.parameters():
                parameter.zero_()
        with self.assertRaises(RuntimeError):
            ae.fit_latent_normalization(torch.randn(64, 2))

    def test_training_uses_reconstruction_only_configuration(self):
        torch.manual_seed(0)
        ae = ToyAE(latent_dim=4, hidden=16, depth=2, bounded=False)
        cfg = SimpleNamespace(
            ae_lr=1e-3,
            ae_weight_decay=0.01,
            ae_steps=2,
            batch_size=16,
            log_every=100,
        )

        train_toy_ae(ae, lambda n: torch.randn(n, 2), cfg)

        self.assertFalse(ae.training)
        self.assertTrue(torch.isfinite(ae.latent_mean).all())
        self.assertTrue(torch.isfinite(ae.latent_std).all())

    def test_bounded_training_uses_uniform_moment_regularizer(self):
        torch.manual_seed(0)
        baseline = ToyAE(latent_dim=4, hidden=16, depth=2, bounded=True)
        regularized = ToyAE(latent_dim=4, hidden=16, depth=2, bounded=True)
        regularized.load_state_dict(baseline.state_dict())
        data = torch.randn(16, 2)
        common = {
            "ae_lr": 1e-3,
            "ae_weight_decay": 0.01,
            "ae_steps": 1,
            "batch_size": len(data),
            "log_every": 100,
        }

        train_toy_ae(
            baseline,
            lambda _: data,
            SimpleNamespace(**common, latent_reg_weight=0.0),
        )
        train_toy_ae(
            regularized,
            lambda _: data,
            SimpleNamespace(**common, latent_reg_weight=1e-3),
        )

        encoder_difference = sum(
            (left - right).abs().sum().item()
            for left, right in zip(
                baseline.encoder_net.parameters(),
                regularized.encoder_net.parameters(),
            )
        )
        self.assertGreater(encoder_difference, 0.0)


if __name__ == "__main__":
    unittest.main()


class LatentFigureTests(unittest.TestCase):
    """The latent-space figure is the only view of the pre-decoder distribution.

    Decoded panels cannot distinguish "the flow model missed the latent
    distribution" from "the decoder moved it", so this figure is what makes the
    d=2 diagnosis possible at all.
    """

    def test_path_is_a_sibling_of_the_benchmark_figure(self):
        from toy.common import latent_figure_path

        self.assertEqual(latent_figure_path("results/uniform_rf_8gaussians.png"),
                         "results/uniform_rf_8gaussians_latent.png")
        # must not collide with the figure it accompanies
        self.assertNotEqual(latent_figure_path("a/b.png"), "a/b.png")

    def test_writes_a_figure_and_survives_outliers(self):
        import tempfile
        from toy.common import save_latent_figure

        pts = torch.randn(256, 2)
        pts[0] = torch.tensor([1e4, 1e4])          # a single far outlier
        with tempfile.TemporaryDirectory() as tmp:
            out = f"{tmp}/latent.png"
            save_latent_figure([("a", pts), ("b", torch.randn(256, 2))], out, box=1.0)
            self.assertTrue(os.path.exists(out) and os.path.getsize(out) > 0)

    def test_every_toy_arm_plots_latents_only_at_dimension_two(self):
        import ast
        import pathlib as _pathlib

        for name in ("uniform_rf", "gaussian_rf", "edm_baseline"):
            source = _pathlib.Path(f"toy/{name}.py").read_text()
            tree = ast.parse(source)
            calls = [n for n in ast.walk(tree)
                     if isinstance(n, ast.Call)
                     and getattr(n.func, "id", None) == "save_latent_figure"]
            with self.subTest(arm=name):
                self.assertEqual(len(calls), 1, f"{name}: expected one latent figure call")
                guards = [ast.unparse(n.test) for n in ast.walk(tree)
                          if isinstance(n, ast.If)
                          and any(isinstance(c, ast.Call)
                                  and getattr(c.func, "id", None) == "save_latent_figure"
                                  for c in ast.walk(n))]
                self.assertTrue(
                    any("cfg.latent_dim == 2" in g for g in guards),
                    f"{name}: latent figure must be guarded by latent_dim == 2, got {guards}",
                )
