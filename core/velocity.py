"""
Velocity field network + rectified-flow teacher training + ODE generation.

The teacher is trained with the plain rectified-flow regression loss:
    L = E_{x0, eps, t} || v_theta(x_t, t) - (x0 - eps) ||^2
No noise schedule, no preconditioning, no score. The network output IS the
velocity (toward-data convention, matching core.interp.velocity_target).

Generation integrates the probability-flow ODE from t=1 (noise) to t=0 (data).
With our convention dx_t/dt = eps - x0 = -w, so as t decreases we step
    x <- x + w(x, t) * dt_step   (dt_step > 0, moving toward data)
which is forward Euler in decreasing t. Heun (2nd order) is also provided.
"""

import torch
import torch.nn as nn

from .interp import (
    interpolate,
    velocity_target,
    sample_uniform_noise,
    sample_t,
    _broadcast_t,
)


def timestep_embedding(t, dim, max_period=10000.0):
    """Sinusoidal embedding of a (B,) time vector -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=t.device))
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLPVelocity(nn.Module):
    """
    Simple MLP velocity field for low-dim toy data (e.g. 2D).
    Input:  x (B, d), t (B,)  ->  velocity (B, d).
    Plain residual blocks with SiLU — no LayerNorm.
    """

    def __init__(self, data_dim, hidden=128, depth=2, t_emb_dim=16):
        super().__init__()
        self.data_dim = data_dim
        self.t_emb_dim = t_emb_dim
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.in_proj = nn.Linear(data_dim + hidden, hidden)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(depth)
        ])
        self.head = nn.Linear(hidden, data_dim)

    def forward(self, x, t):
        te = timestep_embedding(t, self.t_emb_dim)
        te = self.t_proj(te)
        h = self.in_proj(torch.cat([x, te], dim=-1))
        for lin in self.blocks:
            h = h + lin(nn.functional.silu(h))
        return self.head(h)


def rf_loss(net, x0):
    """
    Rectified-flow teacher loss on a batch of data x0 (B, d), x0 in [-1,1]^d.
    Returns scalar loss.
    """
    B = x0.shape[0]
    device = x0.device
    eps = sample_uniform_noise(x0.shape, device=device, dtype=x0.dtype)
    t = sample_t(B, device=device, dtype=x0.dtype)
    x_t = interpolate(x0, eps, t)
    target = velocity_target(x0, eps)
    pred = net(x_t, t)
    return ((pred - target) ** 2).mean()


def _batch_jvp(net, x_t, t, v):
    """
    Batched Jacobian-vector product: J(x_t) @ v for each sample in the batch.
    Uses forward-mode AD via torch.func.jvp.

    Args:
        net: velocity network, (B, d), (B,) -> (B, d).
        x_t: (B, d) input points.
        t:   (B,) time values.
        v:   (B, d) tangent vectors.
    Returns:
        Jv: (B, d) — one JVP per sample.
    """
    # jvp needs a function of the differentiated input only; t is held fixed.
    # We use torch.func.vmap + jvp for a clean batched version, but fall back
    # to a simple loop-free approach: since net is applied batch-wise, we can
    # compute the JVP by differentiating the batched forward pass w.r.t. x_t
    # with v as the tangent. This is exact and efficient.
    from torch.func import jvp as func_jvp

    def fn(x):
        return net(x, t)

    _, Jv = func_jvp(fn, (x_t,), (v,))
    return Jv


def jacobian_spectral_penalty(net, x_t, t, n_power_iters=1):
    """
    Estimate the mean top singular value of grad_x w(x_t, t) across a batch,
    using power iteration with JVPs. This is the spectral-normalization trick
    (Miyato et al.) applied to the input-Jacobian of the velocity field.

    Cost: n_power_iters forward passes through the net per call (via JVP).
    With n_power_iters=1, this is ~1 extra forward pass — cheap enough to run
    every training step.

    The penalty is differentiable w.r.t. net parameters (the JVP is computed
    through the net's forward pass with grad enabled), so it provides gradients
    that push the net toward a smoother velocity field.

    Args:
        net: velocity network.
        x_t: (B, d) input points (should have requires_grad=False; the JVP
             differentiates through net's parameters, not through x_t).
        t:   (B,) time values.
        n_power_iters: power iteration steps. 1 is standard for training
             (the estimate improves over training steps as the random vector
             tracks the top singular vector across updates). More is more
             accurate per step but more expensive.
    Returns:
        scalar: mean estimated top singular value squared, across the batch.
             Squared because (a) it's what the bound uses (L^2 enters the
             Gronwall), and (b) it gives cleaner gradients than the sqrt.
    """
    B, d = x_t.shape
    # initialize random direction (or could maintain a running vector across
    # training steps for better tracking — left as an enhancement)
    v = torch.randn_like(x_t)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-12)

    for _ in range(n_power_iters):
        # u = J v  (forward-mode JVP, differentiable through net)
        Jv = _batch_jvp(net, x_t, t, v)
        # top singular value estimate per sample: ||J v||
        sigma = Jv.norm(dim=1, keepdim=True) + 1e-12
        # update direction for next iteration (detach to avoid double-backprop
        # through the iteration, which is unnecessary for the penalty)
        v = (Jv / sigma).detach()

    # final forward pass to get differentiable sigma estimate
    Jv = _batch_jvp(net, x_t, t, v)
    sigma_sq = (Jv ** 2).sum(dim=1)  # ||J v||^2 per sample
    return sigma_sq.mean()


def rf_loss_regularized(net, x0, jac_lambda=0.01, n_power_iters=1):
    """
    Rectified-flow loss + Jacobian spectral regularization.

    L_total = L_RF + jac_lambda * E[ ||grad_x w(x_t, t)||_op^2 ]

    The Jacobian penalty discourages unnecessarily large spatial derivatives
    in the velocity field, pushing the network toward a smoother function that
    still fits the regression targets. This directly reduces the measured
    stiffness Lambda (and hence tightens the W2 distillation bound) without
    sacrificing — and potentially improving — sample quality (smoother fields
    integrate more accurately with fewer ODE steps).

    The penalty is estimated via 1-step power iteration (one JVP per training
    step), the same trick as spectral normalization in GANs.

    Args:
        net:    velocity network.
        x0:    (B, d) data batch.
        jac_lambda: regularization weight. 0 = plain RF loss.
        n_power_iters: power iteration steps for the penalty estimate.
    Returns:
        total_loss, rf_loss_value, jac_penalty_value  (all scalars).
    """
    B = x0.shape[0]
    device = x0.device
    eps = sample_uniform_noise(x0.shape, device=device, dtype=x0.dtype)
    t = sample_t(B, device=device, dtype=x0.dtype)
    x_t = interpolate(x0, eps, t)
    target = velocity_target(x0, eps)
    pred = net(x_t, t)
    l_rf = ((pred - target) ** 2).mean()

    if jac_lambda > 0:
        # x_t is detached for the penalty: we want gradients through the net's
        # parameters (how the Jacobian depends on theta), not through x_t.
        l_jac = jacobian_spectral_penalty(net, x_t.detach(), t.detach(),
                                          n_power_iters=n_power_iters)
        total = l_rf + jac_lambda * l_jac
    else:
        l_jac = torch.zeros(1, device=device)
        total = l_rf

    return total, l_rf.detach(), l_jac.detach()


@torch.no_grad()
def generate(net, n, data_dim, steps=100, method="heun", device=None,
             return_traj=False):
    """
    Sample by integrating the ODE from t=1 (uniform noise) to t=0 (data).

    dx/dt = eps - x0 = -w(x,t).  Discretize t: 1 = t_0 > t_1 > ... > t_S = 0.
    Forward Euler in decreasing t:  x_{k+1} = x_k + w(x_k, t_k) * (t_k - t_{k+1}).
    """
    net_was_training = net.training
    net.eval()
    x = sample_uniform_noise((n, data_dim), device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    traj = [x.clone()] if return_traj else None

    for k in range(steps):
        t_k = ts[k].expand(n)
        t_next = ts[k + 1]
        dt = (ts[k] - t_next).item()  # positive
        w_k = net(x, t_k)
        if method == "euler":
            x = x + w_k * dt
        elif method == "heun":
            x_pred = x + w_k * dt
            if t_next.item() > 0:
                w_next = net(x_pred, t_next.expand(n))
                x = x + 0.5 * (w_k + w_next) * dt
            else:
                x = x_pred
        else:
            raise ValueError(f"unknown method {method}")
        if return_traj:
            traj.append(x.clone())

    if net_was_training:
        net.train()
    return (x, traj) if return_traj else x
