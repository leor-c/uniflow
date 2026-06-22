"""
Phase 0 consistency driver: train teacher, then distill via continuous-time sCM.

Supports three consistency variants for ablation:
  - "continuous" (default): continuous-time sCM, the JVP tangent residual form.
    Best results; no discretization, no teacher integration for target.
  - "scm": discrete sCM, accurate t->s teacher integration.
  - "cd": discrete CD, single teacher Euler step to adjacent grid point.

Run:
    python -m urfm.toy.train_toy_consistency --dataset 8gaussians \
        --teacher_iters 3000 --distill_iters 3000 --method continuous --out runs/cscm
"""

import argparse
import json
import os
import torch

from core import velocity as V
from core import consistency as C
from core.interp import sample_uniform_noise
from toy.datasets import get_sampler
from toy.train_toy import train_teacher, energy_distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="8gaussians",
                    choices=["8gaussians", "two_moons", "checkerboard"])
    ap.add_argument("--teacher_iters", type=int, default=16000)
    ap.add_argument("--distill_iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--teacher_lr", type=float, default=2e-3)
    ap.add_argument("--student_lr", type=float, default=1e-3)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--method", default="continuous",
                    choices=["continuous", "scm", "cd"],
                    help="Consistency variant: continuous (sCM tangent), "
                         "scm (discrete, accurate t->s), cd (discrete, single step)")
    # continuous-sCM options
    ap.add_argument("--tangent_norm", action="store_true", default=True)
    ap.add_argument("--no_tangent_norm", dest="tangent_norm", action="store_false")
    ap.add_argument("--norm_c", type=float, default=0.1)
    ap.add_argument("--time_weight", action="store_true", default=True)
    ap.add_argument("--no_time_weight", dest="time_weight", action="store_false")
    # discrete sCM/CD options
    ap.add_argument("--n_times", type=int, default=40, help="Grid resolution for discrete CD")
    ap.add_argument("--dt_max", type=float, default=0.3, help="Max t-s gap for discrete sCM")
    ap.add_argument("--substeps", type=int, default=8, help="Teacher sub-steps for t->s integration")
    ap.add_argument("--start_from_noise", action="store_true", default=False,
                    help="Get x_t by integrating from noise (more faithful, pricier)")
    # shared
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--jac_lambda", type=float, default=0.0)
    ap.add_argument("--n_power_iters", type=int, default=1)
    ap.add_argument("--teacher_steps_gen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/consistency")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    data_dim = 2
    data_sampler = get_sampler(args.dataset, device=device)

    print(f"=== Consistency ({args.method}): {args.dataset} on {device} ===")

    # 1. teacher
    teacher = train_teacher(data_sampler, data_dim, args.teacher_iters,
                            args.batch, args.teacher_lr, device,
                            jac_lambda=args.jac_lambda,
                            n_power_iters=args.n_power_iters,
                            weight_decay=args.weight_decay)

    # 2. consistency student
    model = C.ConsistencyModel(data_dim, ema_decay=args.ema_decay).to(device)

    if args.method == "continuous":
        C.train_continuous_scm(
            model, teacher, data_sampler, args.distill_iters, args.batch,
            args.student_lr, device, tangent_norm=args.tangent_norm,
            norm_c=args.norm_c, time_weight=args.time_weight,
            start_from_noise=args.start_from_noise, substeps=args.substeps,
            weight_decay=args.weight_decay)
    elif args.method == "scm":
        C.train_scm(
            model, teacher, data_sampler, args.distill_iters, args.batch,
            args.student_lr, device, dt_max=args.dt_max, substeps=args.substeps,
            start_from_noise=args.start_from_noise, weight_decay=args.weight_decay)
    elif args.method == "cd":
        C.train_cd(
            model, teacher, data_sampler, args.distill_iters, args.batch,
            args.student_lr, device, n_times=args.n_times,
            weight_decay=args.weight_decay, endpoint_sup_weight=0.0)

    # 3. evaluate
    real = data_sampler(4000)
    teacher_samples = V.generate(teacher, 4000, data_dim,
                                 steps=args.teacher_steps_gen, device=device)
    z = sample_uniform_noise((4000, data_dim), device=device)
    with torch.no_grad():
        student_samples = model(z)
    ed_teacher = energy_distance(teacher_samples, real)
    ed_student = energy_distance(student_samples, real)
    print(f"[eval] energy-distance: teacher={ed_teacher:.5f}  "
          f"student(1-step)={ed_student:.5f}")

    metrics = {
        "dataset": args.dataset,
        "method": args.method,
        "energy_distance_teacher": ed_teacher,
        "energy_distance_student": ed_student,
        "student_std": student_samples.std(0).tolist(),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {os.path.join(args.out, 'metrics.json')}")

    if not args.no_plot:
        try:
            _plot(args.out, real, teacher_samples, student_samples, args.method)
            print(f"[saved] plots in {args.out}")
        except Exception as e:
            print(f"[plot] skipped ({e})")


def _plot(out, real, x_teacher, x_student, method):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    real = real.cpu().numpy()
    xt = x_teacher.cpu().numpy()
    xs = x_student.cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    lim = max(abs(real).max(), abs(xt).max(), abs(xs).max()) * 1.1
    for a, d, t in [(ax[0], real, "real"),
                    (ax[1], xt, "teacher multi-step"),
                    (ax[2], xs, f"student 1-step ({method})")]:
        a.scatter(d[:, 0], d[:, 1], s=3, alpha=0.4)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_aspect("equal"); a.set_title(t)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "consistency.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
