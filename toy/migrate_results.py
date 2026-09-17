"""Move legacy toy artifacts into the structured layout without overwriting files.

Preview with ``python -m toy.migrate_results``; apply with ``--apply``.
MNIST/CIFAR artifacts are not moved. Every move is recorded in a manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from toy.outputs import RESULTS_ROOT

PATTERN = re.compile(r"^(uniform_rf|gaussian_rf|edm_ayf|edm)_(8gaussians|two_moons|checkerboard)(.*)$")
DIAGNOSIS = RESULTS_ROOT / '8gaussians/uniform_rf/experiments/diagnosis_seed_1'


def legacy_destination(path: Path, archived: bool = False) -> Path | None:
    summary = re.fullmatch(r'summary_(8gaussians|two_moons|checkerboard)', path.stem)
    if summary:
        directory = RESULTS_ROOT / summary[1]
        if archived:
            directory /= 'archive/pre_init_fix'
        return directory / f'summary{path.suffix}'
    match = PATTERN.fullmatch(path.stem)
    if not match:
        return None
    baseline, dataset, tail = match.groups()
    baseline = 'edm_reference' if baseline == 'edm' else baseline
    directory = RESULTS_ROOT / dataset / baseline
    if archived:
        directory /= 'archive/pre_init_fix'
    seed = re.fullmatch(r'(?:-seed-|_seed)(\d+)(_latent)?', tail)
    if seed:
        directory /= f'seed_{seed[1]}'
        latent = bool(seed[2])
    elif tail in ('', '_latent'):
        directory /= 'legacy_default'
        latent = tail == '_latent'
    elif tail in ('_40k', '_latent_40k'):
        directory /= 'experiments/legacy_40k'
        latent = tail == '_latent_40k'
    else:
        return None
    name = ('samples_latent' if latent else 'samples') if path.suffix == '.png' else 'metrics'
    return directory / f'{name}{path.suffix}'


def migration_plan() -> dict[Path, Path]:
    moves = {}
    for source, archived in [(Path('results'), False), (Path('results/seeds'), False),
                             (Path('results/_pre_init_fix'), True),
                             (Path('results/_pre_init_fix/seeds'), True)]:
        for path in source.glob('*'):
            if path.is_file() and (target := legacy_destination(path, archived)):
                moves[path] = target
    source = Path('results/uniform_rf_diagnosis')
    for path in source.rglob('*'):
        if path.is_file():
            moves[path] = DIAGNOSIS / path.relative_to(source)
    for path in Path('ff_figures').glob('*8gaussians.png'):
        moves[path] = RESULTS_ROOT / '8gaussians/comparisons/fourier' / path.name
    path = Path('edm_baseline_8gaussians.png')
    if path.exists():
        moves[path] = RESULTS_ROOT / '8gaussians/edm_reference/legacy_root/samples.png'
    return moves


def apply_migration(moves: dict[Path, Path]) -> None:
    if len(set(moves.values())) != len(moves):
        raise FileExistsError('Multiple legacy files resolve to the same destination')
    for target in moves.values():
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
    manifest_path = RESULTS_ROOT / 'migration_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for source, target in moves.items():
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        manifest.append({'source': str(source), 'destination': str(target), 'sha256': digest})
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    # Remove only empty source directories, leaving unrelated archives intact.
    parents = {parent for source in moves for parent in source.parents if parent != Path('.')}
    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    moves = migration_plan()
    for source, target in moves.items():
        print(f'{source} -> {target}')
    print(f'{len(moves)} files; {"applying" if args.apply else "preview only"}')
    if args.apply:
        apply_migration(moves)


if __name__ == '__main__':
    main()
