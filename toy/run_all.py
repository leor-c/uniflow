"""
Run all three toy experiments back to back: AYF interval flow map, RF-sCM,
and the EDM baseline.

Each experiment uses its own default Config. The --seed, --ae_steps,
--student_steps, and --teacher_steps flags are applied on top of those
defaults only when explicitly passed; any flag left unset keeps each
experiment's own default for that field.

Usage:
    python -m urfm.toy.run_all
    python -m urfm.toy.run_all --seed 0 --teacher_steps 2000 --student_steps 2000
"""

from __future__ import annotations

import argparse

from toy import ayf_rf, rf_scm, edm_baseline


def _apply_overrides(cfg, args, *, teacher_field: str, student_field: str):
    if args.seed is not None:
        cfg.seed = args.seed
    if args.ae_steps is not None:
        cfg.ae_steps = args.ae_steps
    if args.teacher_steps is not None:
        setattr(cfg, teacher_field, args.teacher_steps)
    if args.student_steps is not None:
        setattr(cfg, student_field, args.student_steps)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ae_steps", type=int, default=None)
    ap.add_argument("--teacher_steps", type=int, default=None)
    ap.add_argument("--student_steps", type=int, default=None)
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("Running toy.ayf_rf")
    print("=" * 60)
    cfg = _apply_overrides(
        ayf_rf.Config(), args, teacher_field="teacher_steps", student_field="student_steps"
    )
    ayf_rf.main(cfg)

    print("\n" + "=" * 60)
    print("Running toy.rf_scm")
    print("=" * 60)
    cfg = _apply_overrides(
        rf_scm.Config(), args, teacher_field="teacher_steps", student_field="student_steps"
    )
    rf_scm.main(cfg)

    print("\n" + "=" * 60)
    print("Running toy.edm_baseline")
    print("=" * 60)
    cfg = _apply_overrides(
        edm_baseline.Config(), args,
        teacher_field="edm_teacher_steps", student_field="edm_student_steps",
    )
    edm_baseline.main(cfg)


if __name__ == "__main__":
    main()
