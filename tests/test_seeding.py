"""RNG streams: a run must be reproducible *and* comparable across arms.

The failure this guards against is subtle. With one global ``manual_seed`` per
run, every consumer -- weight init, the data sampler, the training noise, the
evaluation draws -- shares a single stream, so what each of them sees depends
on how many numbers everything before it consumed. Arms with different
architectures diverge on their first parameter tensor, and the same seed then
trains and evaluates them on different data.
"""

import unittest

import torch

from core.seeding import derive_seed, seed_stream, stream_generator
from toy import edm_baseline, gaussian_rf, uniform_rf
from toy.common import make_run_samplers


TOY_ARMS = (uniform_rf, gaussian_rf, edm_baseline)


def _cpu_config(module, seed=1):
    cfg = module.Config()
    cfg.device = "cpu"
    cfg.seed = seed
    return cfg


class DerivedSeedTests(unittest.TestCase):
    def test_sub_seeds_are_deterministic(self):
        self.assertEqual(derive_seed(3, "teacher-data"), derive_seed(3, "teacher-data"))

    def test_sub_seeds_differ_by_stream_and_by_seed(self):
        self.assertNotEqual(derive_seed(3, "teacher-data"), derive_seed(3, "student-data"))
        self.assertNotEqual(derive_seed(3, "teacher-data"), derive_seed(4, "teacher-data"))

    def test_sub_seeds_are_valid_torch_seeds(self):
        for stream in ("ae-init", "teacher-train", "evaluation-data"):
            value = derive_seed(0, stream)
            self.assertGreaterEqual(value, 0)
            self.assertLess(value, 2**63)
            torch.manual_seed(value)


class ToyStreamTests(unittest.TestCase):
    def test_every_phase_sees_the_same_data_in_every_arm(self):
        """The point of the split: one seed, one dataset, three arms."""
        samplers = [make_run_samplers(_cpu_config(module)) for module in TOY_ARMS]
        for phase in ("ae", "teacher", "student", "evaluation", "calibration"):
            reference = getattr(samplers[0], phase)(64)
            for module, other in zip(TOY_ARMS[1:], samplers[1:]):
                self.assertTrue(
                    torch.equal(reference, getattr(other, phase)(64)),
                    f"{phase} data differs between toy.uniform_rf and {module.__name__}",
                )

    def test_phases_draw_from_independent_streams(self):
        samplers = make_run_samplers(_cpu_config(uniform_rf))
        draws = [getattr(samplers, phase)(64) for phase in
                 ("ae", "teacher", "student", "evaluation", "calibration")]
        for index, first in enumerate(draws):
            for second in draws[index + 1:]:
                self.assertFalse(torch.equal(first, second))

    def test_data_streams_ignore_the_global_rng(self):
        """A model built mid-run must not shift the data the next phase sees."""
        samplers = make_run_samplers(_cpu_config(uniform_rf))
        torch.manual_seed(0)
        expected = samplers.teacher(64)

        samplers = make_run_samplers(_cpu_config(uniform_rf))
        torch.manual_seed(12345)
        torch.randn(1000)
        torch.testing.assert_close(samplers.teacher(64), expected, atol=0.0, rtol=0.0)

    def test_streams_still_change_with_the_seed(self):
        first = make_run_samplers(_cpu_config(uniform_rf, seed=1)).teacher(64)
        second = make_run_samplers(_cpu_config(uniform_rf, seed=2)).teacher(64)
        self.assertFalse(torch.equal(first, second))

    def test_arms_sharing_an_architecture_share_an_initialisation(self):
        """Same seed, same module: the AYF student starts from the same weights."""
        states = []
        for module in TOY_ARMS:
            cfg = _cpu_config(module)
            seed_stream(cfg.seed, "student-init")
            states.append(
                edm_baseline.A.RFIntervalFlowMap(
                    cfg.latent_dim, cfg.hidden, cfg.depth,
                    ff_scale=cfg.student_ff_scale, ff_size=cfg.ff_size,
                    ff_input_scale=1.0,
                ).state_dict()
            )
        reference = states[0]
        for state in states[1:]:
            for key, value in reference.items():
                self.assertTrue(torch.equal(value, state[key]))


class ArmSeedingTests(unittest.TestCase):
    """No arm may fall back to one global seed for the whole run."""

    def test_toy_arms_use_named_streams(self):
        for module in TOY_ARMS:
            names = module.main.__code__.co_names
            with self.subTest(arm=module.__name__):
                self.assertIn("make_run_samplers", names)
                self.assertIn("seed_stream", names)
                self.assertNotIn("manual_seed", names)

    def test_mnist_arms_use_named_streams(self):
        from mnist import edm_baseline as mnist_edm
        from mnist import gaussian_rf as mnist_gaussian
        from mnist import uniform_rf as mnist_uniform

        for module in (mnist_uniform, mnist_gaussian, mnist_edm):
            names = module.main.__code__.co_names
            with self.subTest(arm=module.__name__):
                self.assertIn("stream_generator", names)
                self.assertIn("seed_stream", names)
                self.assertNotIn("manual_seed", names)


class MNISTStreamTests(unittest.TestCase):
    def test_latent_sampler_takes_a_generator(self):
        from mnist.ae import make_mnist_latent_sampler

        latents = torch.arange(100.0).unsqueeze(1)
        teacher = make_mnist_latent_sampler(
            latents, "cpu", generator=stream_generator(0, "teacher-data")
        )
        student = make_mnist_latent_sampler(
            latents, "cpu", generator=stream_generator(0, "student-data")
        )
        again = make_mnist_latent_sampler(
            latents, "cpu", generator=stream_generator(0, "teacher-data")
        )
        batch = teacher(32)
        self.assertFalse(torch.equal(batch, student(32)))
        self.assertTrue(torch.equal(batch, again(32)))


if __name__ == "__main__":
    unittest.main()
