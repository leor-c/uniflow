"""Tests for the DiT backbone and the spatial MNIST latent.

The properties that matter here are the ones the rest of the codebase silently
depends on:

* the DiT is a drop-in for ``FiLMMLP`` -- flat ``(B, D)`` in, flat ``(B, D)``
  out -- so no loss, sampler or metric needs to know about the latent grid;
* patchify/unpatchify round-trip exactly, or the spatial layout the encoder
  produces is not the layout the transformer attends over;
* ``torch.func.jvp`` works through it, because the AYF / consistency / EDM
  student losses all differentiate the network in forward mode. This is the
  reason attention is written out instead of calling
  ``F.scaled_dot_product_attention``, whose fused kernels have no forward-mode
  rule.
"""

import unittest

import torch
import torch.nn as nn
from torch.func import jvp

from core.dit import DiT, make_backbone
from core.edm.models import EDMConsistencyModel, EDMPrecondDenoiser
from core.ayf_rf.models import RFIntervalFlowMap
from core.rf.models import RFTeacher
from mnist.ae import LATENT_GRID, MNISTAutoEncoder

SHAPE = (4, LATENT_GRID, LATENT_GRID)
DIM = SHAPE[0] * SHAPE[1] * SHAPE[2]
SMALL = dict(arch="dit", latent_shape=SHAPE, hidden=48, depth=2, heads=4)


class DiTBackboneTests(unittest.TestCase):
    def _dit(self, **kw):
        opts = dict(hidden=48, depth=2, heads=4)
        opts.update(kw)
        return DiT(DIM, 16, DIM, SHAPE, **opts)

    def test_flat_in_flat_out(self):
        """The FiLMMLP call signature is preserved exactly."""
        net = self._dit()
        z, cond = torch.randn(5, DIM), torch.randn(5, 16)
        self.assertEqual(net(z, cond).shape, (5, DIM))

    def test_patchify_roundtrip_is_identity(self):
        for patch in (1, 7):
            with self.subTest(patch=patch):
                net = self._dit(patch=patch)
                z = torch.randn(3, DIM)
                self.assertTrue(torch.equal(net.unpatchify(net.patchify(z)), z))

    def test_patchify_preserves_the_encoder_layout(self):
        """Token c of patch (i, j) must be latent channel c at grid cell (i, j)."""
        net = self._dit(patch=1)
        grid = torch.arange(DIM, dtype=torch.float32).view(1, *SHAPE)
        tokens = net.patchify(grid.reshape(1, DIM))
        self.assertEqual(tokens.shape, (1, LATENT_GRID * LATENT_GRID, SHAPE[0]))
        for i in range(LATENT_GRID):
            for j in range(LATENT_GRID):
                token = tokens[0, i * LATENT_GRID + j]
                self.assertTrue(torch.equal(token, grid[0, :, i, j]))

    def test_token_count_follows_patch_size(self):
        self.assertEqual(self._dit(patch=1).num_tokens, 49)
        self.assertEqual(self._dit(patch=7).num_tokens, 1)

    def test_adaln_zero_starts_at_zero_output(self):
        """Zero-initialized output layer: the net predicts 0 before training."""
        net = self._dit()
        out = net(torch.randn(4, DIM), torch.randn(4, 16))
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_conditioning_actually_reaches_the_output(self):
        """After the zero-init is broken, cond must change the prediction."""
        net = self._dit()
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.05 * torch.randn_like(p))
        z = torch.randn(4, DIM)
        a = net(z, torch.zeros(4, 16))
        b = net(z, torch.ones(4, 16))
        self.assertGreater((a - b).abs().max().item(), 1e-6)

    def test_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            DiT(DIM + 1, 16, DIM, SHAPE, hidden=48, depth=2, heads=4)
        with self.assertRaises(ValueError):
            DiT(DIM, 16, DIM, SHAPE, hidden=48, depth=2, heads=4, patch=2)

    def test_make_backbone_dispatch(self):
        mlp = make_backbone(DIM, 16, DIM, 32, 2, arch="mlp")
        self.assertNotIsInstance(mlp, DiT)
        self.assertIsInstance(make_backbone(DIM, 16, DIM, 48, 2, arch="dit",
                                            latent_shape=SHAPE, heads=4), DiT)
        with self.assertRaises(ValueError):
            make_backbone(DIM, 16, DIM, 48, 2, arch="unet")
        with self.assertRaises(ValueError):
            make_backbone(DIM, 16, DIM, 48, 2, arch="dit")  # no latent_shape


class ForwardModeADTests(unittest.TestCase):
    """``torch.func.jvp`` must traverse every model class using a DiT."""

    def test_jvp_through_teacher_matches_finite_differences(self):
        model = RFTeacher(DIM, **SMALL)
        with torch.no_grad():  # break the zero-init so the tangent is non-trivial
            for p in model.parameters():
                p.add_(0.05 * torch.randn_like(p))
        z, t = torch.randn(3, DIM), torch.rand(3, 1) * 0.8 + 0.1

        _, tangent = jvp(lambda tt: model(z, tt), (t,), (torch.ones_like(t),))
        self.assertGreater(tangent.abs().mean().item(), 0.0)

        with torch.no_grad():
            eps = 1e-4
            finite_difference = (model(z, t + eps) - model(z, t - eps)) / (2 * eps)
        self.assertLess((tangent - finite_difference).abs().max().item(), 1e-2)

    def test_jvp_through_interval_flow_map(self):
        model = RFIntervalFlowMap(DIM, **SMALL)
        z, t, s = torch.randn(3, DIM), torch.rand(3, 1) * 0.5, torch.rand(3, 1)
        out, tangent = jvp(lambda ss: model(z, t, ss), (s,), (torch.ones_like(s),))
        self.assertEqual(out.shape, (3, DIM))
        self.assertEqual(tangent.shape, (3, DIM))

    def test_jvp_through_edm_denoiser(self):
        model = EDMPrecondDenoiser(DIM, sigma_data=0.5, **SMALL)
        z, log_sigma = torch.randn(3, DIM), torch.randn(3, 1) * 0.5
        _, tangent = jvp(lambda l: model(z, l), (log_sigma,),
                         (torch.ones_like(log_sigma),))
        self.assertEqual(tangent.shape, (3, DIM))

    def test_boundary_conditions_hold_with_a_dit(self):
        """Parameterization boundaries are architecture-independent."""
        student = RFIntervalFlowMap(DIM, **SMALL)
        z, t = torch.randn(3, DIM), torch.rand(3, 1)
        # C(z, t, t) = z
        self.assertTrue(torch.allclose(student(z, t, t), z, atol=1e-6))

        sigma_min = 0.002
        consistency = EDMConsistencyModel(DIM, sigma_data=0.5,
                                          sigma_min=sigma_min, **SMALL)
        log_sigma_min = torch.full((3, 1), sigma_min).log()
        self.assertTrue(torch.allclose(consistency(z, log_sigma_min), z, atol=1e-5))

    def test_fourier_state_features_are_rejected_by_the_dit(self):
        """RFF changes in_dim and destroys the grid, so it must not silently pass."""
        with self.assertRaises(ValueError):
            RFTeacher(DIM, ff_scale=1.0, ff_size=8, **SMALL)


class SpatialLatentAETests(unittest.TestCase):
    def test_spatial_latent_is_flat_and_correctly_sized(self):
        ae = MNISTAutoEncoder(latent_dim=DIM, bounded=True, latent_channels=4)
        z = ae.encode(torch.randn(3, 1, 28, 28))
        self.assertEqual(z.shape, (3, DIM))
        self.assertEqual(ae.latent_shape, SHAPE)
        self.assertEqual(ae.decode(z).shape, (3, 1, 28, 28))

    def test_tanh_bounds_every_latent_pixel(self):
        """The uniform arm's base is U([-1,1]) per dim, so bounds are elementwise."""
        ae = MNISTAutoEncoder(latent_dim=DIM, bounded=True, latent_channels=4)
        z = ae.encode(torch.randn(16, 1, 28, 28) * 5.0)
        self.assertLessEqual(z.abs().max().item(), 1.0)

    def test_flat_latent_path_still_works(self):
        ae = MNISTAutoEncoder(latent_dim=64, bounded=True, latent_channels=None)
        z = ae.encode(torch.randn(3, 1, 28, 28))
        self.assertEqual(z.shape, (3, 64))
        self.assertIsNone(ae.latent_shape)

    def test_unbounded_normalization_is_a_global_scalar(self):
        ae = MNISTAutoEncoder(latent_dim=DIM, bounded=False, latent_channels=4)
        ae.fit_latent_normalization(torch.randn(64, 1, 28, 28))
        self.assertEqual(ae.latent_mean.shape, (DIM,))
        # One scalar broadcast across dims, so the transform stays a similarity.
        self.assertEqual(ae.latent_std.unique().numel(), 1)

    def test_mismatched_latent_dim_is_rejected(self):
        with self.assertRaises(ValueError):
            MNISTAutoEncoder(latent_dim=100, bounded=True, latent_channels=4)

    def test_encoder_head_is_convolutional_when_spatial(self):
        """A Linear head would discard the grid before the DiT ever sees it."""
        spatial = MNISTAutoEncoder(latent_dim=DIM, bounded=True, latent_channels=4)
        self.assertIsInstance(spatial.encoder.head, nn.Conv2d)
        flat = MNISTAutoEncoder(latent_dim=64, bounded=True)
        self.assertIsInstance(flat.encoder.head, nn.Linear)


class ArmConfigTests(unittest.TestCase):
    """Every MNIST arm must agree on the latent layout and default to the DiT."""

    def _arms(self):
        from mnist import edm_baseline, gaussian_rf, uniform_rf
        return {"uniform_rf": uniform_rf, "gaussian_rf": gaussian_rf,
                "edm_baseline": edm_baseline}

    def test_defaults_are_a_4x7x7_dit(self):
        for name, module in self._arms().items():
            with self.subTest(arm=name):
                cfg = module.Config()
                self.assertEqual(cfg.arch, "dit")
                self.assertEqual(cfg.latent_channels, 4)
                self.assertEqual(cfg.latent_dim, DIM)
                self.assertEqual(cfg.latent_shape, SHAPE)

    def test_latent_dim_is_derived_not_set(self):
        for name, module in self._arms().items():
            with self.subTest(arm=name):
                cfg = module.Config(latent_channels=8)
                self.assertEqual(cfg.latent_dim, 8 * 49)
                # latent_dim is never an independent knob
                self.assertEqual(module.Config(latent_dim=1234).latent_dim, DIM)

    def test_backbone_kwargs_build_a_working_model(self):
        for name, module in self._arms().items():
            with self.subTest(arm=name):
                cfg = module.Config()
                cfg.hidden, cfg.depth, cfg.heads = 48, 2, 4
                model = RFTeacher(cfg.latent_dim, cfg.hidden, cfg.depth,
                                  **cfg.backbone_kwargs)
                out = model(torch.randn(2, cfg.latent_dim), torch.rand(2, 1))
                self.assertEqual(out.shape, (2, cfg.latent_dim))

    def test_fast_config_keeps_the_spatial_latent(self):
        for name, module in self._arms().items():
            with self.subTest(arm=name):
                self.assertEqual(module.fast_cpu_config().latent_dim, DIM)


if __name__ == "__main__":
    unittest.main()
