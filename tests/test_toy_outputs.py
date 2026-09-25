import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from toy import uniform_rf, gaussian_rf, edm_baseline
from toy.outputs import configure_outputs, save_run_config
from toy.migrate_results import legacy_destination, apply_migration


class OutputTests(unittest.TestCase):
    def test_every_arm_separates_datasets_and_seeds(self):
        for module, baseline in [(uniform_rf, 'uniform_rf'), (gaussian_rf, 'gaussian_rf'),
                                 (edm_baseline, 'edm_ayf')]:
            cfg = module.Config(dataset='two_moons', seed=3, device='cpu')
            configure_outputs(cfg, baseline)
            root = f'results/toy/two_moons/{baseline}/seed_3'
            self.assertEqual(cfg.out, f'{root}/samples.png')
            self.assertEqual(cfg.metrics_out, f'{root}/metrics.json')
            configure_outputs(cfg, baseline)
            self.assertEqual(cfg.metrics_out, f'{root}/metrics.json')

    def test_custom_figure_gets_sibling_metrics(self):
        cfg = uniform_rf.Config(out='custom/ablation.png')
        configure_outputs(cfg, 'uniform_rf')
        self.assertEqual(cfg.out, 'custom/ablation.png')
        self.assertEqual(cfg.metrics_out, 'custom/ablation.json')

    def test_explicit_paths_are_preserved(self):
        cfg = gaussian_rf.Config(out_dir='custom/run', out='elsewhere/plot.png',
                                 metrics_out='metrics/custom.json')
        configure_outputs(cfg, 'gaussian_rf')
        self.assertEqual(cfg.out, 'elsewhere/plot.png')
        self.assertEqual(cfg.metrics_out, 'metrics/custom.json')

    def test_custom_run_directory_and_full_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = uniform_rf.Config(out_dir=tmp, teacher_refine_steps=40)
            configure_outputs(cfg, 'uniform_rf')
            save_run_config(cfg)
            self.assertEqual(cfg.metrics_out, f'{tmp}/metrics.json')
            record = json.loads((Path(tmp) / 'run_config.json').read_text())
            self.assertEqual(record['teacher_refine_steps'], 40)
            self.assertEqual(record['out'], f'{tmp}/samples.png')

    def test_resume_uses_structured_seed_metrics(self):
        from toy.run_all import _seed_metrics_path
        self.assertEqual(str(_seed_metrics_path('uniform_rf', '8gaussians', 2)),
                         'results/toy/8gaussians/uniform_rf/seed_2/metrics.json')


class MigrationTests(unittest.TestCase):
    def test_seeded_figures_and_metrics_share_directory(self):
        image = legacy_destination(Path('results/uniform_rf_two_moons-seed-1.png'))
        metrics = legacy_destination(Path('results/seeds/uniform_rf_two_moons_seed1.json'))
        self.assertEqual(image.parent, metrics.parent)
        self.assertEqual(image.name, 'samples.png')
        self.assertEqual(metrics.name, 'metrics.json')
        self.assertIn('legacy_default', str(legacy_destination(Path('uniform_rf_two_moons.png'))))

    def test_migration_refuses_collisions_before_moving_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / 'one', root / 'two'
            first.write_text('first')
            second.write_text('second')
            existing = root / 'exists'
            existing.write_text('keep')
            with self.assertRaises(FileExistsError):
                apply_migration({first: root / 'new', second: existing})
            self.assertTrue(first.exists())
            self.assertEqual(existing.read_text(), 'keep')

    def test_migration_records_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'old' / 'artifact.pt'
            source.parent.mkdir()
            source.write_bytes(b'checkpoint\x00\x01')
            target = root / 'new' / 'artifact.pt'
            with patch('toy.migrate_results.RESULTS_ROOT', root / 'new'):
                apply_migration({source: target})
            self.assertEqual(target.read_bytes(), b'checkpoint\x00\x01')
            self.assertFalse(source.exists())
            self.assertEqual(len(json.loads((root / 'new/migration_manifest.json').read_text())), 1)
