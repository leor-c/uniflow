"""
Phase 0 consistency driver (latent-space variant): autoencoder-in-the-loop.

Pipeline:
  1. Train an autoencoder on RAW (unbounded) 2D data. Encoder ends in tanh, so
     latents live in (-1,1)^latent_dim. Data is NOT tanh'd.
  2. FREEZE encoder + decoder.
  3. Train the uniform-noise RF teacher IN LATENT SPACE: its "data" is E(x),
     its noise is Uniform([-1,1]^latent_dim).
  4. Distill a consistency student in latent space.
  5. Decode latent samples back to R^2; evaluate/plot in data space.

Run:
    python -m urfm.toy.train_toy_consistency_ae --dataset 8gaussians \
        --latent_dim 8 --ae_iters 4000 --teacher_iters 3000 \
        --distill_iters 3000 --method continuous --out runs/cscm_ae8
"""

import argparse
import json
import os
import torch

from core import velocity as V
from core import consistency as C
from core import autoencoder as AE
from core.interp import sample_uniform_noise
from toy.train_toy import energy_distance
from toy.train_toy_ae import raw_data_sampler, latent_sampler, train_teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="8gaussians",
                    choices=["8gaussians", "two_moons"])
    ap.add_argument("--latent_dim", type=int, default=8)
    ap.add_argument("--ae_iters", type=int, default=4000)
    ap.add_argument("--teacher_iters", type=int, default=8000)
    ap.add_argument("--distill_iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ae_lr", type=float, default=1e-3)
    ap.add_argument("--teacher_lr", type=float, default=3e-4)
    ap.add_argument("--student_lr", type=float, default=3e-4)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--latent_spread_weight", type=float, default=0.0)
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
                    help="Get z_t by integrating from noise (more faithful, pricier)")
    # shared
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--jac_lambda", type=float, default=0.0)
    ap.add_argument("--n_power_iters", type=int, default=1)
    ap.add_argument("--teacher_steps_gen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/consistency_ae")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    raw = raw_data_sampler(args.dataset, device)

    print(f"=== Consistency AE ({args.method}): {args.dataset}, "
          f"d={args.latent_dim}, {device} ===")

    # 1-2. train + freeze autoencoder
    ae = AE.AutoEncoder(2, args.latent_dim).to(device)
    AE.train_autoencoder(ae, raw, args.ae_iters, args.batch, args.ae_lr, device,
                         latent_spread_weight=args.latent_spread_weight)
    AE.freeze(ae.encoder)
    AE.freeze(ae.decoder)
    stats = AE.latent_stats(ae.encoder, raw, device=device)
    print(f"[latent] dim={stats['latent_dim']} mean_abs={stats['mean_abs']:.3f} "
          f"sat0.99={stats['frac_saturated_0.99']:.3f} per_dim_std="
          f"{[round(s, 2) for s in stats['per_dim_std']]}")
    with torch.no_grad():
        x = raw(2000)
        x_hat, _ = ae(x)
        recon = ((x_hat - x) ** 2).mean().item()
    print(f"[ae] reconstruction MSE = {recon:.5f}")

    z_data = latent_sampler(ae.encoder, raw)

    # 3. teacher in latent space
    teacher = train_teacher(args.latent_dim, z_data, args.teacher_iters,
                            args.batch, args.teacher_lr, device,
                            jac_lambda=args.jac_lambda,
                            n_power_iters=args.n_power_iters,
                            weight_decay=args.weight_decay)

    # 4. consistency student in latent space
    model = C.ConsistencyModel(args.latent_dim, ema_decay=args.ema_decay).to(device)

    if args.method == "continuous":
        C.train_continuous_scm(
            model, teacher, z_data, args.distill_iters, args.batch,
            args.student_lr, device, tangent_norm=args.tangent_norm,
            norm_c=args.norm_c, time_weight=args.time_weight,
            start_from_noise=args.start_from_noise, substeps=args.substeps,
            weight_decay=args.weight_decay)
    elif args.method == "scm":
        C.train_scm(
            model, teacher, z_data, args.distill_iters, args.batch,
            args.student_lr, device, dt_max=args.dt_max, substeps=args.substeps,
            start_from_noise=args.start_from_noise, weight_decay=args.weight_decay)
    elif args.method == "cd":
        C.train_cd(
            model, teacher, z_data, args.distill_iters, args.batch,
            args.student_lr, device, n_times=args.n_times,
            weight_decay=args.weight_decay, endpoint_sup_weight=0.0)

    # 5. decode + evaluate in data space
    real = raw(4000)
    z_teacher = V.generate(teacher, 4000, args.latent_dim,
                           steps=args.teacher_steps_gen, device=device)
    z_noise = sample_uniform_noise((4000, args.latent_dim), device=device)
    with torch.no_grad():
        z_student = model(z_noise)
        x_teacher = ae.decoder(z_teacher)
        x_student = ae.decoder(z_student)
    ed_teacher = energy_distance(x_teacher, real)
    ed_student = energy_distance(x_student, real)
    print(f"[eval] decoded energy-distance: teacher={ed_teacher:.5f}  "
          f"student(1-step)={ed_student:.5f}")

    metrics = {
        "dataset": args.dataset,
        "latent_dim": args.latent_dim,
        "method": args.method,
        "recon_mse": recon,
        "latent_stats": stats,
        "ed_teacher_decoded": ed_teacher,
        "ed_student_decoded": ed_student,
        "student_latent_std": z_student.std(0).tolist(),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {os.path.join(args.out, 'metrics.json')}")

    if not args.no_plot:
        try:
            _plot(args.out, real, x_teacher, x_student, args.method)
            print(f"[saved] plots in {args.out}")
        except Exception as e:
            print(f"[plot] skipped ({e})")


def _plot(out, real, x_teacher, x_student, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = real.cpu().numpy()
    xt = x_teacher.cpu().numpy()
    xs = x_student.cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    lim = max(abs(real).max(), abs(xt).max(), abs(xs).max()) * 1.1
    for a, d, t in [(ax[0], real, "real"),
                    (ax[1], xt, "teacher->decode"),
                    (ax[2], xs, f"student 1-step->decode ({method})")]:
        a.scatter(d[:, 0], d[:, 1], s=3, alpha=0.4)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        a.set_title(t)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "consistency_ae.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
