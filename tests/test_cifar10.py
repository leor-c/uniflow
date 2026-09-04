"""Tests for the CIFAR-10 arm, mirroring the MNIST coverage in test_dit.py.

The CIFAR-10 package is a mirror of ``mnist/``, so the properties worth testing
are the same ones plus the places the mirror could silently drift:

* the spatial latent is flat on the way out and ``C x 8 x 8`` on the way in, so
  no loss, sampler or metric needs to know about the grid;
* all three arms agree on that layout and default to the DiT;
* ``torch.func.jvp`` works through a model built from an arm's own
  ``backbone_kwargs``, because the AYF / EDM student losses differentiate the
  network in forward mode;
* the RGB figure path handles ``(B, 3, H, W)``, which is the one place the arms
  differ from MNIST's single channel.
"""

import unittest

import torch
import torch.nn as nn
from torch.func import jvp

from core.ayf_rf.models import RFIntervalFlowMap
from core.edm.models import EDMPrecondDenoiser
from core.rf.models import RFTeacher
from cifar10.ae import LATENT_GRID, IMAGE_CHANNELS, CIFAR10AutoEncoder

SHAPE = (8, LATENT_GRID, LATENT_GRID)
DIM = SHAPE[0] * SHAPE[1] * SHAPE[2]
SMALL = dict(arch="dit", latent_shape=SHAPE, hidden=48, depth=2, heads=4)
IMAGE = (IMAGE_CHANNELS, 32, 32)


class SpatialLatentAETests(unittest.TestCase):
    def test_spatial_latent_is_flat_and_correctly_sized(self):
        ae = CIFAR10AutoEncoder(latent_dim=DIM, bounded=True, latent_channels=8)
        z = ae.encode(torch.randn(3, *IMAGE))
        self.assertEqual(z.shape, (3, DIM))
        self.assertEqual(ae.latent_shape, SHAPE)
        self.assertEqual(ae.decode(z).shape, (3, *IMAGE))

    def test_grid_is_eight_so_patch_two_is_legal(self):
        """The 8x8 grid is the reason CIFAR-10 can tokenize at patch=2 and MNIST cannot."""
        self.assertEqual(LATENT_GRID, 8)
        self.assertEqual(LATENT_GRID % 2, 0)

    def test_tanh_bounds_every_latent_pixel(self):
        """The uniform arm's base is U([-1,1]) per dim, so bounds are elementwise."""
        ae = CIFAR10AutoEncoder(latent_dim=DIM, bounded=True, latent_channels=8)
        z = ae.encode(torch.randn(16, *IMAGE) * 5.0)
        self.assertLessEqual(z.abs().max().item(), 1.0)

    def test_flat_latent_path_still_works(self):
        ae = CIFAR10AutoEncoder(latent_dim=64, bounded=True, latent_channels=None)
        z = ae.encode(torch.randn(3, *IMAGE))
        self.assertEqual(z.shape, (3, 64))
        self.assertIsNone(ae.latent_shape)

    def test_unbounded_normalization_is_a_global_scalar(self):
        ae = CIFAR10AutoEncoder(latent_dim=DIM, bounded=False, latent_channels=8)
        ae.fit_latent_normalization(torch.randn(64, *IMAGE))
        self.assertEqual(ae.latent_mean.shape, (DIM,))
        # One scalar broadcast across dims, so the transform stays a similarity.
        self.assertEqual(ae.latent_std.unique().numel(), 1)

    def test_mismatched_latent_dim_is_rejected(self):
        with self.assertRaises(ValueError):
            CIFAR10AutoEncoder(latent_dim=100, bounded=True, latent_channels=8)

    def test_encoder_head_is_convolutional_when_spatial(self):
        """A Linear head would discard the grid before the DiT ever sees it."""
        spatial = CIFAR10AutoEncoder(latent_dim=DIM, bounded=True, latent_channels=8)
        self.assertIsInstance(spatial.encoder.head, nn.Conv2d)
        flat = CIFAR10AutoEncoder(latent_dim=64, bounded=True)
        self.assertIsInstance(flat.encoder.head, nn.Linear)


class ArmConfigTests(unittest.TestCase):
    """Every CIFAR-10 arm must agree on the latent layout and default to the DiT."""

    def _arms(self):
        from cifar10 import edm_baseline, gaussian_rf, uniform_rf
        return {"uniform_rf": uniform_rf, "gaussian_rf": gaussian_rf,
                "edm_baseline": edm_baseline}

    def test_defaults_are_an_8x8x8_dit(self):
        for name, module in self._arms().items():
            with self.subTest(arm=name):
                cfg = module.Config()
                self.assertEqual(cfg.arch, "dit")
                self.assertEqual(cfg.latent_channels, 8)
                self.assertEqual(cfg.latent_dim, DIM)
                self.assertEqual(cfg.latent_shape, SHAPE)

    def test_latent_dim_is_derived_not_set(self):
        for name, module in self._arms().items():
            with self.subTest(arm=name):
                cfg = module.Config(latent_channels=4)
                self.assertEqual(cfg.latent_dim, 4 * 64)
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

    def test_patch_two_tokenization_is_supported(self):
        """16 tokens instead of 64 -- the cheap option the even grid unlocks."""
        for name, module in self._arms().items():
            with self.subTest(arm=name):
                cfg = module.Config()
                cfg.hidden, cfg.depth, cfg.heads, cfg.patch = 48, 2, 4, 2
                model = RFTeacher(cfg.latent_dim, cfg.hidden, cfg.depth,
                                  **cfg.backbone_kwargs)
                self.assertEqual(model.mlp.num_tokens, 16)
                out = model(torch.randn(2, cfg.latent_dim), torch.rand(2, 1))
                self.assertEqual(out.shape, (2, cfg.latent_dim))

    def test_fast_config_keeps_the_spatial_latent(self):
        for name, module in self._arms().items():
            with self.subTest(arm=name):
                self.assertEqual(module.fast_cpu_config().latent_dim, DIM)

    def test_out_dirs_do_not_collide_with_mnist(self):
        """Both packages write image grids; a shared path would overwrite them."""
        from mnist import edm_baseline as mnist_edm
        from mnist import gaussian_rf as mnist_gaussian
        from mnist import uniform_rf as mnist_uniform

        mnist_dirs = {m.Config().out_dir for m in
                      (mnist_uniform, mnist_gaussian, mnist_edm)}
        cifar_dirs = {m.Config().out_dir for m in self._arms().values()}
        self.assertEqual(len(cifar_dirs), 3)
        self.assertFalse(mnist_dirs & cifar_dirs)


class ForwardModeADTests(unittest.TestCase):
    """The AYF and EDM student losses differentiate the network via torch.func.jvp."""

    def test_jvp_through_an_arm_backbone(self):
        from cifar10 import uniform_rf

        cfg = uniform_rf.Config()
        cfg.hidden, cfg.depth, cfg.heads = 48, 2, 4
        model = RFIntervalFlowMap(cfg.latent_dim, cfg.hidden, cfg.depth,
                                  **cfg.backbone_kwargs)
        z = torch.randn(2, cfg.latent_dim)
        t = torch.rand(2, 1)
        s = torch.rand(2, 1)
        primal, tangent = jvp(
            lambda tt: model(z, tt, s),
            (t,), (torch.ones_like(t),),
        )
        self.assertEqual(primal.shape, (2, cfg.latent_dim))
        self.assertEqual(tangent.shape, (2, cfg.latent_dim))
        self.assertTrue(torch.isfinite(tangent).all())

    def test_edm_denoiser_builds_from_arm_kwargs(self):
        from cifar10 import edm_baseline

        cfg = edm_baseline.Config()
        cfg.hidden, cfg.depth, cfg.heads = 48, 2, 4
        model = EDMPrecondDenoiser(cfg.latent_dim, cfg.hidden, cfg.depth,
                                   cfg.sigma_data, **cfg.backbone_kwargs)
        out = model(torch.randn(2, cfg.latent_dim), torch.randn(2, 1))
        self.assertEqual(out.shape, (2, cfg.latent_dim))


class RGBFigureTests(unittest.TestCase):
    """The one place the arms genuinely differ from MNIST: 3-channel figures."""

    def test_every_arm_saves_an_rgb_grid(self):
        import tempfile
        from pathlib import Path
        from cifar10 import edm_baseline, gaussian_rf, uniform_rf

        images = torch.rand(12, *IMAGE)
        with tempfile.TemporaryDirectory() as tmp:
            for module in (uniform_rf, gaussian_rf, edm_baseline):
                name = module.__name__.rsplit(".", 1)[-1]
                path = Path(tmp) / f"{name}.png"
                with self.subTest(arm=name):
                    module.save_image_grid(images, str(path), nrow=4, title=name)
                    self.assertTrue(path.exists())
                    self.assertGreater(path.stat().st_size, 0)


class SeedingTests(unittest.TestCase):
    def test_arms_use_named_streams(self):
        from cifar10 import edm_baseline, gaussian_rf, uniform_rf

        for module in (uniform_rf, gaussian_rf, edm_baseline):
            names = module.main.__code__.co_names
            with self.subTest(arm=module.__name__):
                self.assertIn("stream_generator", names)
                self.assertIn("seed_stream", names)
                self.assertNotIn("manual_seed", names)

    def test_latent_sampler_takes_a_generator(self):
        from core.seeding import stream_generator
        from cifar10.ae import make_cifar10_latent_sampler

        latents = torch.arange(100.0).unsqueeze(1)
        teacher = make_cifar10_latent_sampler(
            latents, "cpu", generator=stream_generator(0, "teacher-data")
        )
        student = make_cifar10_latent_sampler(
            latents, "cpu", generator=stream_generator(0, "student-data")
        )
        again = make_cifar10_latent_sampler(
            latents, "cpu", generator=stream_generator(0, "teacher-data")
        )
        batch = teacher(32)
        self.assertFalse(torch.equal(batch, student(32)))
        self.assertTrue(torch.equal(batch, again(32)))


class RunAllTests(unittest.TestCase):
    def test_pipeline_table_covers_every_arm(self):
        from cifar10.run_all import PIPELINES, PIPELINE_LABELS, PIPELINE_MODULES

        self.assertEqual(set(PIPELINES), set(PIPELINE_MODULES))
        self.assertEqual(set(PIPELINES), set(PIPELINE_LABELS))
        for pipeline in PIPELINES:
            module, teacher_field, student_field, out_dir = PIPELINE_MODULES[pipeline]
            with self.subTest(pipeline=pipeline):
                cfg = module.Config()
                # The steps fields must exist, or --teacher-steps silently
                # creates a new attribute nobody reads.
                self.assertTrue(hasattr(cfg, teacher_field))
                self.assertTrue(hasattr(cfg, student_field))
                self.assertTrue(out_dir.startswith("results/cifar10/"))

    def test_overrides_reach_every_arm(self):
        from argparse import Namespace
        from cifar10.run_all import PIPELINES, PIPELINE_MODULES, _apply_overrides

        args = Namespace(
            data_root=None, latent_channels=4, arch="mlp", patch=None,
            train_mode="lmd", ff=None, teacher_ff_scale=None,
            student_ff_scale=None, reflow_rounds=1, reflow_steps=None,
            ae_weight_decay=0.0, ae_steps=7, teacher_steps=11, student_steps=13,
        )
        for pipeline in PIPELINES:
            module, teacher_field, student_field, _ = PIPELINE_MODULES[pipeline]
            with self.subTest(pipeline=pipeline):
                cfg = _apply_overrides(
                    module.Config(), args, seed=3,
                    teacher_field=teacher_field, student_field=student_field,
                )
                self.assertEqual(cfg.seed, 3)
                self.assertEqual(cfg.latent_channels, 4)
                self.assertEqual(cfg.latent_dim, 4 * 64)
                self.assertEqual(cfg.arch, "mlp")
                self.assertEqual(cfg.train_mode, "lmd")
                self.assertEqual(cfg.ae_weight_decay, 0.0)
                self.assertEqual(cfg.ae_steps, 7)
                self.assertEqual(getattr(cfg, teacher_field), 11)
                self.assertEqual(getattr(cfg, student_field), 13)

    def test_reflow_override_skips_the_edm_arm(self):
        """The EDM arm has no RF velocity, so it must not grow a reflow field."""
        from argparse import Namespace
        from cifar10 import edm_baseline
        from cifar10.run_all import _apply_overrides

        args = Namespace(
            data_root=None, latent_channels=None, arch=None, patch=None,
            train_mode=None, ff=None, teacher_ff_scale=None,
            student_ff_scale=None, reflow_rounds=2, reflow_steps=99,
            ae_weight_decay=None, ae_steps=None, teacher_steps=None,
            student_steps=None,
        )
        cfg = _apply_overrides(
            edm_baseline.Config(), args, seed=0,
            teacher_field="edm_teacher_steps", student_field="student_steps",
        )
        self.assertFalse(hasattr(cfg, "reflow_rounds"))

    def test_multi_seed_output_dirs_are_distinct(self):
        from argparse import Namespace
        from cifar10.run_all import PIPELINES, _output_dir

        args = Namespace(out_root=None)
        dirs = {_output_dir(p, s, args, multi_seed=True)
                for p in PIPELINES for s in (0, 1)}
        self.assertEqual(len(dirs), len(PIPELINES) * 2)


if __name__ == "__main__":
    unittest.main()
