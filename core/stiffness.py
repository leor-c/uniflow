"""
Stiffness probe for the velocity field.

Theory (Section 5): the Lipschitz constant of the velocity field controls the
W2 distillation bound via  Lambda = int_0^1 L(t) dt,  where
    L(t) = sup_x || grad_x w(x, t) ||_op   (top singular value of input-Jacobian).

Key predictions to verify empirically:
  - L(0) = 1 and L(1) = 1 exactly (boundary anchoring), unconditionally.
  - L(t) is a finite "bump" in the interior; Lambda < infinity.

We estimate the top singular value of the Jacobian J = d w / d x at a point via
power iteration using only JVPs and VJPs (no explicit Jacobian materialized):
    repeat:  u = J v / ||J v||;   v = J^T u / ||J^T u||
    sigma_max ~ || J v ||.
Cost per iteration: ~one JVP + one VJP; independent of dimension d.

We report L(t) as the MAX over a sample of points x (an empirical lower bound on
the sup), at each of a grid of t values.
"""

import torch
from torch.func import jvp, vjp


def top_singular_value(fn, x, n_iters=30, tol=1e-7):
    """
    Estimate the top singular value of the Jacobian of fn at x (single input).

    Args:
        fn: callable mapping x (d,) -> (d,). Should be differentiable.
        x:  (d,) input point (requires no grad; we use functional transforms).
        n_iters: power-iteration steps.
    Returns:
        scalar tensor: estimated top singular value.
    """
    x = x.detach()
    v = torch.randn_like(x)
    v = v / (v.norm() + 1e-12)
    sigma_prev = torch.tensor(0.0, device=x.device)

    for _ in range(n_iters):
        # u = J v   (forward-mode)
        _, u = jvp(fn, (x,), (v,))
        u_norm = u.norm() + 1e-12
        u = u / u_norm
        # v = J^T u  (reverse-mode)
        _, vjp_fn = vjp(fn, x)
        v = vjp_fn(u)[0]
        v_norm = v.norm() + 1e-12
        v = v / v_norm
        sigma = u_norm  # ||J v_old|| estimate of sigma_max
        if (sigma - sigma_prev).abs() < tol:
            sigma_prev = sigma
            break
        sigma_prev = sigma

    # final estimate
    _, u = jvp(fn, (x,), (v,))
    return u.norm()


def estimate_L_at_t(net, t_scalar, sample_points, n_iters=30):
    """
    Estimate L(t) = max over sample_points of ||grad_x w(x,t)||_op.

    Args:
        net: velocity network, net(x (B,d), t (B,)) -> (B,d).
        t_scalar: float time in [0,1].
        sample_points: (N, d) points at which to probe (e.g. samples from p_t).
        n_iters: power-iteration steps.
    Returns:
        dict with 'max', 'mean', 'median' of the per-point top singular values.
    """
    device = sample_points.device
    svals = []
    for i in range(sample_points.shape[0]):
        xi = sample_points[i]

        def fn(x):
            # net expects batched input; wrap single point
            t = torch.full((1,), t_scalar, device=device, dtype=x.dtype)
            return net(x.unsqueeze(0), t).squeeze(0)

        s = top_singular_value(fn, xi, n_iters=n_iters)
        svals.append(s.detach())
    svals = torch.stack(svals)
    return {
        "t": t_scalar,
        "max": svals.max().item(),
        "mean": svals.mean().item(),
        "median": svals.median().item(),
    }


def profile_L(net, sample_points_fn, t_grid, n_iters=30):
    """
    Compute L(t) across a grid of t values.

    Args:
        net: velocity network.
        sample_points_fn: callable(t_scalar) -> (N, d) probe points at level t.
            Typically samples x_t = (1-t) x0 + t eps from real data + uniform noise,
            so we probe where p_t actually has mass.
        t_grid: iterable of float t values.
        n_iters: power-iteration steps.
    Returns:
        list of dicts (one per t) and the trapezoidal estimate of Lambda.
    """
    profile = []
    for t in t_grid:
        pts = sample_points_fn(t)
        rec = estimate_L_at_t(net, t, pts, n_iters=n_iters)
        profile.append(rec)

    # trapezoidal integral of the 'max' curve over [0,1]
    ts = [r["t"] for r in profile]
    Ls = [r["max"] for r in profile]
    lam = 0.0
    for i in range(1, len(ts)):
        lam += 0.5 * (Ls[i] + Ls[i - 1]) * (ts[i] - ts[i - 1])
    return profile, lam
