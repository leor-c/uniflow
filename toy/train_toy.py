"""
Phase 0 driver: the full toy pipeline on a 2D dataset.

Steps:
  1. Train the uniform-noise RF teacher.
  2. Profile L(t) via the power-iteration stiffness probe (theory validation:
     boundary anchoring L(0)=L(1)=1, finite interior bump, Lambda estimate).
  3. Distill a one-step velocity-matching student (two-timescale, EMA critic).
  4. Evaluate: sample quality (energy-distance to data), and produce plots.

Run:
    python -m urfm.toy.train_toy --dataset 8gaussians --teacher_iters 4000 \
        --distill_iters 3000 --out runs/toy_8g
All defaults are CPU-friendly (small nets, 2D data).
"""

import argparse
import json
import os
import torch

from core import velocity as V
from core import distill as D
from core import stiffness as S
from core.interp import interpolate, sample_uniform_noise
from toy.datasets import get_sampler


def energy_distance(x, y):
    """
    Unbiased-ish energy distance between samples x (N,d) and y (M,d).
    Lower = closer distributions. A simple scalar quality metric for 2D.
    """
    def pdist_mean(a, b):
        # mean pairwise Euclidean distance
        d = torch.cdist(a, b)
        return d.mean()
    return (2 * pdist_mean(x, y) - pdist_mean(x, x) - pdist_mean(y, y)).item()


def train_teacher(data_sampler, data_dim, iters, batch, lr, device, log_every=500,
                  jac_lambda=0.0, n_power_iters=1, weight_decay=0.0):
    net = V.MLPVelocity(data_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()
    for it in range(iters):
        x0 = data_sampler(batch)
        opt.zero_grad(set_to_none=True)
        if jac_lambda > 0:
            loss, rf_val, jac_val = V.rf_loss_regularized(
                net, x0, jac_lambda=jac_lambda, n_power_iters=n_power_iters)
            loss.backward()
            opt.step()
            if it % log_every == 0 or it == iters - 1:
                print(f"[teacher] iter {it:5d}  rf_loss {rf_val.item():.5f}  "
                      f"jac_penalty {jac_val.item():.3f}  "
                      f"total {loss.item():.5f}")
        else:
            loss = V.rf_loss(net, x0)
            loss.backward()
            opt.step()
            if it % log_every == 0 or it == iters - 1:
                print(f"[teacher] iter {it:5d}  loss {loss.item():.5f}")
    net.eval()
    return net


def profile_stiffness(teacher, data_sampler, data_dim, device, n_probe=64,
                      n_t=21, n_iters=12):
    """Profile L(t) probing at points sampled from p_t (real data + uniform noise)."""
    def sample_points_fn(t):
        x0 = data_sampler(n_probe)
        eps = sample_uniform_noise((n_probe, data_dim), device=device)
        t_vec = torch.full((n_probe,), float(t), device=device)
        return interpolate(x0, eps, t_vec)

    t_grid = [i / (n_t - 1) for i in range(n_t)]
    profile, lam = S.profile_L(teacher, sample_points_fn, t_grid, n_iters=n_iters)
    return profile, lam


def distill_student(teacher, data_dim, iters, batch, g_lr, c_lr, critic_steps,
                    device, log_every=500, critic_init_teacher=False,
                    weighting="none"):
    gen = D.OneStepGenerator(data_dim).to(device)
    critic = V.MLPVelocity(data_dim).to(device)
    # Critic init: empirically, RANDOM init gives cleaner results than warm-
    # starting from the teacher. Teacher-init makes v_S ~ v_T at the start, so
    # the velocity-difference signal is ~0 and the critic must first un-learn the
    # teacher distribution before tracking the student. Random init avoids that.
    if critic_init_teacher:
        critic.load_state_dict(teacher.state_dict())
    g_opt = torch.optim.Adam(gen.parameters(), lr=g_lr, betas=(0.0, 0.999))
    c_opt = torch.optim.Adam(critic.parameters(), lr=c_lr, betas=(0.0, 0.999))
    gen.train(); critic.train()
    for it in range(iters):
        g_loss, c_loss = D.distill_step(
            gen, teacher, critic, g_opt, c_opt, batch, data_dim, device,
            critic_steps=critic_steps, weighting=weighting,
        )
        if it % log_every == 0 or it == iters - 1:
            print(f"[distill] iter {it:5d}  g_loss {g_loss:.5f}  c_loss {c_loss:.5f}")
    gen.eval()
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="8gaussians", choices=["8gaussians", "two_moons", "checkerboard"])
    ap.add_argument("--teacher_iters", type=int, default=4000)
    ap.add_argument("--distill_iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--teacher_lr", type=float, default=2e-3)
    ap.add_argument("--g_lr", type=float, default=1e-4)
    ap.add_argument("--c_lr", type=float, default=2e-4)
    ap.add_argument("--critic_steps", type=int, default=5)
    ap.add_argument("--teacher_steps_gen", type=int, default=100, help="ODE steps for teacher multi-step eval")
    ap.add_argument("--jac_lambda", type=float, default=0.0, help="Jacobian spectral regularization weight for teacher (0=off)")
    ap.add_argument("--n_power_iters", type=int, default=1, help="Power iteration steps for Jacobian penalty")
    ap.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay for teacher (cheap implicit Jacobian regularization)")
    ap.add_argument("--signal_weighting", default="none", choices=["none", "t", "1-t"], help="t-weighting of the velocity-difference distillation signal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/toy")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    data_dim = 2
    data_sampler = get_sampler(args.dataset, device=device)

    print(f"=== Phase 0: {args.dataset} on {device} ===")

    # 1. teacher
    teacher = train_teacher(data_sampler, data_dim, args.teacher_iters,
                            args.batch, args.teacher_lr, device,
                            jac_lambda=args.jac_lambda,
                            n_power_iters=args.n_power_iters,
                            weight_decay=args.weight_decay)

    # 2. stiffness profile (theory validation)
    profile, lam = profile_stiffness(teacher, data_sampler, data_dim, device)
    print(f"[stiffness] L(0)={profile[0]['max']:.4f}  L(1)={profile[-1]['max']:.4f}  "
          f"L_peak={max(p['max'] for p in profile):.4f}  Lambda(int L dt)={lam:.4f}")

    # 3. distill
    student = distill_student(teacher, data_dim, args.distill_iters, args.batch,
                              args.g_lr, args.c_lr, args.critic_steps, device,
                              weighting=args.signal_weighting)

    # 4. evaluate
    real = data_sampler(4000)
    teacher_samples = V.generate(teacher, 4000, data_dim,
                                 steps=args.teacher_steps_gen, device=device)
    z = sample_uniform_noise((4000, data_dim), device=device)
    with torch.no_grad():
        student_samples = student(z)
    ed_teacher = energy_distance(teacher_samples, real)
    ed_student = energy_distance(student_samples, real)
    print(f"[eval] energy-distance to data:  teacher(multi-step)={ed_teacher:.5f}  "
          f"student(1-step)={ed_student:.5f}")

    # save metrics
    metrics = {
        "dataset": args.dataset,
        "stiffness_profile": profile,
        "Lambda": lam,
        "L0": profile[0]["max"],
        "L1": profile[-1]["max"],
        "L_peak": max(p["max"] for p in profile),
        "energy_distance_teacher": ed_teacher,
        "energy_distance_student": ed_student,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {os.path.join(args.out, 'metrics.json')}")

    # plots
    if not args.no_plot:
        try:
            make_plots(args.out, real, teacher_samples, student_samples, profile)
            print(f"[saved] plots in {args.out}")
        except Exception as e:
            print(f"[plot] skipped ({e})")


def make_plots(out, real, teacher_samples, student_samples, profile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = real.cpu().numpy()
    ts_ = teacher_samples.cpu().numpy()
    ss_ = student_samples.cpu().numpy()
    t_grid = [p["t"] for p in profile]
    L_max = [p["max"] for p in profile]
    L_mean = [p["mean"] for p in profile]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, data, title in [
        (axes[0], real, "real data"),
        (axes[1], ts_, "teacher (multi-step)"),
        (axes[2], ss_, "student (1-step)"),
    ]:
        ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.4)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal"); ax.set_title(title)

    axes[3].plot(t_grid, L_max, "o-", label="L(t) max", color="crimson")
    axes[3].plot(t_grid, L_mean, "s--", label="L(t) mean", color="steelblue", alpha=0.7)
    axes[3].axhline(1.0, ls=":", color="gray", label="L=1 (boundary)")
    axes[3].set_xlabel("t"); axes[3].set_ylabel("stiffness L(t)")
    axes[3].set_title("stiffness profile"); axes[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "phase0.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
