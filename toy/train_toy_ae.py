"""
Phase 0 (latent-space variant): autoencoder-in-the-loop.

Pipeline:
  1. Train an autoencoder on RAW (unbounded) 2D data. Encoder ends in tanh, so
     latents live in (-1,1)^latent_dim. Data is NOT tanh'd.
  2. FREEZE encoder + decoder.
  3. Train the uniform-noise RF teacher IN LATENT SPACE: its "data" is E(x),
     its noise is Uniform([-1,1]^latent_dim).
  4. Profile stiffness L(t) in latent space.
  5. Distill the one-step student in latent space.
  6. Decode latent samples back to R^2; evaluate/plot in data space.

Run:
  python -m urfm.toy.train_toy_ae --dataset 8gaussians --latent_dim 8 \
      --ae_iters 4000 --teacher_iters 3000 --distill_iters 3000 --out runs/ae8
"""

import argparse
import json
import os
import math
import torch

from core import velocity as V
from core import distill as D
from core import stiffness as S
from core import autoencoder as AE
from core.interp import interpolate, sample_uniform_noise
from toy.train_toy import energy_distance


def raw_data_sampler(name, device):
    """UNBOUNDED 2D data (no tanh, no clamp). Spread out in R^2."""
    def eight_gaussians(n):
        r = 4.0
        c = torch.tensor([[r * math.cos(2 * math.pi * k / 8),
                           r * math.sin(2 * math.pi * k / 8)] for k in range(8)],
                         device=device)
        i = torch.randint(0, 8, (n,), device=device)
        return c[i] + 0.3 * torch.randn(n, 2, device=device)

    def two_moons(n):
        n1 = n // 2; n2 = n - n1
        t1 = torch.rand(n1, device=device) * math.pi
        x1 = torch.stack([torch.cos(t1), torch.sin(t1)], dim=1)
        t2 = torch.rand(n2, device=device) * math.pi
        x2 = torch.stack([1 - torch.cos(t2), 1 - torch.sin(t2) - 0.5], dim=1)
        x = torch.cat([x1, x2], 0) + 0.05 * torch.randn(n, 2, device=device)
        return x * 3.0

    return {"8gaussians": eight_gaussians, "two_moons": two_moons}[name]


def latent_sampler(encoder, raw_sampler):
    """Wrap: sample raw data, encode to latents (the flow's 'data')."""
    def sample(n):
        with torch.no_grad():
            return encoder(raw_sampler(n))
    return sample


def train_teacher(net_dim, data_sampler, iters, batch, lr, device, log_every=1000,
                  jac_lambda=0.0, n_power_iters=1, weight_decay=0.0):
    net = V.MLPVelocity(net_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()
    for it in range(iters):
        z0 = data_sampler(batch)
        opt.zero_grad(set_to_none=True)
        if jac_lambda > 0:
            loss, rf_val, jac_val = V.rf_loss_regularized(
                net, z0, jac_lambda=jac_lambda, n_power_iters=n_power_iters)
            loss.backward(); opt.step()
            if it % log_every == 0 or it == iters - 1:
                print(f"[teacher] iter {it:5d}  rf_loss {rf_val.item():.5f}  "
                      f"jac_penalty {jac_val.item():.3f}  total {loss.item():.5f}")
        else:
            loss = V.rf_loss(net, z0)
            loss.backward(); opt.step()
            if it % log_every == 0 or it == iters - 1:
                print(f"[teacher] iter {it:5d}  loss {loss.item():.5f}")
    net.eval(); return net


def distill_student(teacher, dim, iters, batch, g_lr, c_lr, critic_steps, device,
                    log_every=1000, weighting="none"):
    gen = D.OneStepGenerator(dim).to(device)
    critic = V.MLPVelocity(dim).to(device)  # random init (empirically better)
    g_opt = torch.optim.Adam(gen.parameters(), lr=g_lr, betas=(0.0, 0.999))
    c_opt = torch.optim.Adam(critic.parameters(), lr=c_lr, betas=(0.0, 0.999))
    gen.train(); critic.train()
    for it in range(iters):
        g_loss, c_loss = D.distill_step(gen, teacher, critic, g_opt, c_opt,
                                        batch, dim, device, critic_steps=critic_steps,
                                        weighting=weighting)
        if it % log_every == 0 or it == iters - 1:
            print(f"[distill] iter {it:5d}  g_loss {g_loss:.5f}  c_loss {c_loss:.5f}")
    gen.eval(); return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="8gaussians", choices=["8gaussians", "two_moons"])
    ap.add_argument("--latent_dim", type=int, default=8)
    ap.add_argument("--ae_iters", type=int, default=4000)
    ap.add_argument("--teacher_iters", type=int, default=3000)
    ap.add_argument("--distill_iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ae_lr", type=float, default=1e-3)
    ap.add_argument("--teacher_lr", type=float, default=2e-3)
    ap.add_argument("--g_lr", type=float, default=5e-4)
    ap.add_argument("--c_lr", type=float, default=2e-4)
    ap.add_argument("--critic_steps", type=int, default=5)
    ap.add_argument("--latent_spread_weight", type=float, default=0.0)
    ap.add_argument("--jac_lambda", type=float, default=0.0, help="Jacobian spectral regularization weight for teacher (0=off)")
    ap.add_argument("--n_power_iters", type=int, default=1, help="Power iteration steps for Jacobian penalty")
    ap.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay for teacher")
    ap.add_argument("--signal_weighting", default="none", choices=["none", "t", "1-t"], help="t-weighting of the velocity-difference distillation signal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ae")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    raw = raw_data_sampler(args.dataset, device)

    print(f"=== Phase 0 (AE latent): {args.dataset}, d={args.latent_dim}, {device} ===")

    # 1-2. train + freeze autoencoder
    ae = AE.AutoEncoder(2, args.latent_dim).to(device)
    AE.train_autoencoder(ae, raw, args.ae_iters, args.batch, args.ae_lr, device,
                         latent_spread_weight=args.latent_spread_weight)
    AE.freeze(ae.encoder); AE.freeze(ae.decoder)
    stats = AE.latent_stats(ae.encoder, raw, device=device)
    print(f"[latent] dim={stats['latent_dim']} mean_abs={stats['mean_abs']:.3f} "
          f"sat0.99={stats['frac_saturated_0.99']:.3f} per_dim_std="
          f"{[round(s,2) for s in stats['per_dim_std']]}")
    # reconstruction quality
    with torch.no_grad():
        x = raw(2000); x_hat, _ = ae(x)
        recon = ((x_hat - x) ** 2).mean().item()
    print(f"[ae] reconstruction MSE = {recon:.5f}")

    z_data = latent_sampler(ae.encoder, raw)

    # 3. teacher in latent space
    teacher = train_teacher(args.latent_dim, z_data, args.teacher_iters,
                            args.batch, args.teacher_lr, device,
                            jac_lambda=args.jac_lambda,
                            n_power_iters=args.n_power_iters,
                            weight_decay=args.weight_decay)

    # 4. stiffness in latent space
    def spf(t):
        z0 = z_data(64)
        eps = sample_uniform_noise((64, args.latent_dim), device=device)
        return interpolate(z0, eps, torch.full((64,), float(t), device=device))
    profile, lam = S.profile_L(teacher, spf, [i / 20 for i in range(21)], n_iters=30)
    print(f"[stiffness] L(0)={profile[0]['max']:.3f} L(1)={profile[-1]['max']:.3f} "
          f"L_peak={max(p['max'] for p in profile):.3f} Lambda={lam:.3f}")

    # 5. distill student in latent space
    student = distill_student(teacher, args.latent_dim, args.distill_iters,
                              args.batch, args.g_lr, args.c_lr, args.critic_steps, device,
                              weighting=args.signal_weighting)

    # 6. decode + evaluate in data space
    real = raw(4000)
    z_teacher = V.generate(teacher, 4000, args.latent_dim, steps=100, device=device)
    z_student = student(sample_uniform_noise((4000, args.latent_dim), device=device))
    with torch.no_grad():
        x_teacher = ae.decoder(z_teacher)
        x_student = ae.decoder(z_student)
    # z-space quality (flow only) and x-space quality (end-to-end)
    ed_t_x = energy_distance(x_teacher, real)
    ed_s_x = energy_distance(x_student, real)
    print(f"[eval] decoded energy-distance: teacher={ed_t_x:.5f}  student(1-step)={ed_s_x:.5f}")

    metrics = {
        "dataset": args.dataset, "latent_dim": args.latent_dim,
        "recon_mse": recon, "latent_stats": stats,
        "stiffness_profile": profile, "Lambda": lam,
        "L0": profile[0]["max"], "L1": profile[-1]["max"],
        "L_peak": max(p["max"] for p in profile),
        "ed_teacher_decoded": ed_t_x, "ed_student_decoded": ed_s_x,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {os.path.join(args.out, 'metrics.json')}")

    if not args.no_plot:
        try:
            _plot(args.out, real, x_teacher, x_student, profile)
            print(f"[saved] plots in {args.out}")
        except Exception as e:
            print(f"[plot] skipped ({e})")


def _plot(out, real, x_teacher, x_student, profile):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    real = real.cpu().numpy(); xt = x_teacher.cpu().numpy(); xs = x_student.cpu().numpy()
    tg = [p["t"] for p in profile]; Lm = [p["max"] for p in profile]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    lim = max(abs(real).max(), abs(xt).max(), abs(xs).max()) * 1.1
    for a, d, t in [(ax[0], real, "real (decoded space)"),
                    (ax[1], xt, "teacher->decode"),
                    (ax[2], xs, "student 1-step->decode")]:
        a.scatter(d[:, 0], d[:, 1], s=3, alpha=0.4)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal"); a.set_title(t)
    ax[3].plot(tg, Lm, "o-", color="crimson"); ax[3].axhline(1, ls=":", color="gray")
    ax[3].set_xlabel("t"); ax[3].set_ylabel("L(t)"); ax[3].set_title("latent stiffness")
    fig.tight_layout(); fig.savefig(os.path.join(out, "phase0_ae.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
