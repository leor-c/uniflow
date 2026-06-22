"""
Consistency Distillation (CD) on the uniform-noise / bounded-latent substrate.

Lesson from the failed teacher-free attempt (UCT): parameterizing the consistency
function as f = x_t + t*w with a MARGINAL velocity field forces the mean map
(f(eps,1) = eps + E[x0-eps|eps] = mu), collapsing one-step generation to the data
mean. Linear-interpolation paths from INDEPENDENT noise cross each other, so the
consistency target at high noise is ambiguous and the network learns the average.

CD fixes this with TWO changes:
  (1) FREE-FORM boundary parameterization:
          f_theta(x, t) = (1 - t) * x + t * F_theta(x, t)
      Boundary: f(x,0) = x (exact). At t=1: f(eps,1) = F_theta(eps,1) is a FREE
      map (the one-step generator), NOT tied to the marginal velocity. This is
      what lets it pick a specific endpoint instead of the mean.
  (2) TEACHER-STEP target: the lower-noise neighbor is obtained by ONE reverse
      ODE step of the (non-crossing) teacher field, not by the raw linear-
      interpolation neighbor. Only ONE teacher evaluation per iter.

One-step sampling:  x0_hat = clamp(f_theta(eps, t=1)) = clamp(F_theta(eps, 1)).
"""

import copy
import torch
import torch.nn as nn

from .interp import sample_uniform_noise, interpolate, _broadcast_t
from .velocity import MLPVelocity


class ConsistencyModel(nn.Module):
    """
    f_theta(x,t) = (1-t)*x + t*F_theta(x,t), with an EMA target net for the CD
    target. Boundary f(x,0)=x exact; at t=1, f(x,1)=F_theta(x,1) is free.
    """

    def __init__(self, data_dim, hidden=128, depth=2, t_emb_dim=16, ema_decay=0.999):
        super().__init__()
        self.F = MLPVelocity(data_dim, hidden=hidden, depth=depth, t_emb_dim=t_emb_dim)
        self.target = copy.deepcopy(self.F).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.ema_decay = ema_decay
        self.data_dim = data_dim

    def f(self, x, t, use_target=False):
        net = self.target if use_target else self.F
        tb = _broadcast_t(t, x)
        return (1 - tb) * x + tb * net(x, t)

    @torch.no_grad()
    def update_target(self):
        for tp, p in zip(self.target.parameters(), self.F.parameters()):
            tp.mul_(self.ema_decay).add_(p, alpha=1 - self.ema_decay)

    def forward(self, z, clamp=True):
        """One-step sample: x0_hat = f(z, t=1) = F(z, 1)."""
        t1 = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        x0_hat = self.f(z, t1)
        return x0_hat.clamp(-1, 1) if clamp else x0_hat


def teacher_velocity_step(teacher, x_t, t, t_prev):
    """
    One reverse Euler step of the teacher PF-ODE from t to t_prev < t.
    dx/dt = -u^T  =>  x_{t_prev} = x_t + u^T(x_t,t)*(t - t_prev).
    """
    with torch.no_grad():
        u = teacher(x_t, t)
        dt = _broadcast_t(t - t_prev, x_t)
        return x_t + u * dt


def cd_loss(model, teacher, x0, n_times=40, huber_delta=0.0):
    """
    CD loss. Forward-noise to x_t (closed form), one teacher step to x_{t_prev},
    match f_online(x_t,t) to stop-grad f_target(x_{t_prev}, t_prev).
    """
    B = x0.shape[0]
    device = x0.device
    eps = sample_uniform_noise(x0.shape, device=device, dtype=x0.dtype)

    i = torch.randint(1, n_times, (B,), device=device)
    t = i.to(x0.dtype) / n_times
    t_prev = (i - 1).to(x0.dtype) / n_times

    x_t = interpolate(x0, eps, t)
    x_tprev = teacher_velocity_step(teacher, x_t, t, t_prev)

    f_online = model.f(x_t, t, use_target=False)
    with torch.no_grad():
        f_target = model.f(x_tprev, t_prev, use_target=True)

    diff = f_online - f_target
    if huber_delta > 0:
        d2 = (diff ** 2).sum(dim=1)
        loss = (torch.sqrt(d2 + huber_delta ** 2) - huber_delta).mean()
    else:
        loss = (diff ** 2).mean()
    return loss


def integrate_teacher_between(teacher, x_t, t, s, substeps=8):
    """
    Integrate the teacher PF-ODE accurately from level t down to level s < t,
    starting at x_t. Returns x_s on the (non-crossing) teacher trajectory.
    Uses `substeps` Euler sub-steps for accuracy over the interval.
    """
    x = x_t
    with torch.no_grad():
        for j in range(substeps):
            tj = t + (s - t) * (j / substeps)        # from t down toward s
            tj1 = t + (s - t) * ((j + 1) / substeps)
            dt = _broadcast_t(tj - tj1, x)            # positive
            x = x + teacher(x, tj) * dt
    return x


def scm_loss(model, teacher, x0, dt_max=0.3, substeps=8, huber_delta=0.0,
             start_from_noise=False):
    """
    sCM-style consistency-distillation loss.

      - sample eps ~ Uniform, two times s < t,
      - obtain x_t (forward-noise from data, or integrate from pure noise),
      - integrate the teacher ACCURATELY from t down to s -> x_s,
      - match f_online(x_t, t) to stop-grad f_target(x_s, s).

    Args:
        dt_max: max interval (t - s).
        substeps: teacher sub-steps for the t->s integration (accuracy).
        start_from_noise: if True, x_t is obtained by integrating the teacher from
            t=1 down to t (both x_t, x_s genuinely on one trajectory); else x_t is
            forward-noised from data (cheaper).
    """
    B = x0.shape[0]
    device = x0.device
    eps = sample_uniform_noise(x0.shape, device=device, dtype=x0.dtype)

    t = torch.rand(B, device=device, dtype=x0.dtype).clamp(min=1e-3)
    gap = torch.rand(B, device=device, dtype=x0.dtype) * dt_max
    s = (t - gap).clamp(min=0.0)

    if start_from_noise:
        # x_t by integrating teacher from 1 -> t (more faithful, pricier)
        x1 = eps
        x_t = integrate_teacher_between(teacher, x1, torch.ones_like(t), t, substeps=substeps)
    else:
        # x_t by forward-noising data (cheap)
        x_t = interpolate(x0, eps, t)

    # accurate teacher integration t -> s
    x_s = integrate_teacher_between(teacher, x_t, t, s, substeps=substeps)

    f_online = model.f(x_t, t, use_target=False)
    with torch.no_grad():
        f_target = model.f(x_s, s, use_target=True)

    diff = f_online - f_target
    if huber_delta > 0:
        d2 = (diff ** 2).sum(dim=1)
        loss = (torch.sqrt(d2 + huber_delta ** 2) - huber_delta).mean()
    else:
        loss = (diff ** 2).mean()
    return loss


def train_scm(model, teacher, data_sampler, iters, batch, lr, device,
              dt_max=0.3, substeps=8, huber_delta=0.0, weight_decay=0.0,
              start_from_noise=False, log_every=500):
    """Train a ConsistencyModel by sCM-style CD (accurate t->s integration)."""
    opt = torch.optim.AdamW(model.F.parameters(), lr=lr, weight_decay=weight_decay)
    model.F.train(); teacher.eval()
    for it in range(iters):
        x0 = data_sampler(batch)
        opt.zero_grad(set_to_none=True)
        loss = scm_loss(model, teacher, x0, dt_max=dt_max, substeps=substeps,
                        huber_delta=huber_delta, start_from_noise=start_from_noise)
        loss.backward()
        opt.step()
        model.update_target()
        if it % log_every == 0 or it == iters - 1:
            print(f"[scm] iter {it:5d}  loss {loss.item():.6f}")
    model.F.eval()
    return model


def continuous_scm_loss(model, teacher, x0, tangent_norm=True, norm_c=0.1,
                        time_weight=True, start_from_noise=False, substeps=8):
    """
    Continuous-time sCM loss (the Delta -> 0 limit of the discrete CM loss).

    The consistency condition "f constant along the PF-ODE trajectory" becomes,
    in the limit, the vanishing of the total derivative of f along the flow:

        df/dt = partial_t f + (grad_x f) . (dx/dt)
              = partial_t f - (grad_x f) . u^T(x_t, t)    [dx/dt = -u^T]
              = 0.

    We compute df/dt as a single JVP of f w.r.t. inputs (x, t) with tangent
    (-u^T, +1), and minimize its squared norm. The free-form boundary param
    f = (1-t)x + t*F (with f(x,0)=x) excludes the trivial f=const solution.

    No s, no Delta, no teacher integration for the target — the teacher enters
    only through u^T at the single point x_t (one teacher eval). This removes the
    discretization bias that under-disperses the discrete version.

    Stabilization (sCM-style, both optional):
      - tangent_norm: divide the residual by (||df/dt|| + norm_c), detached, so the
        loss focuses on direction rather than magnitude (training a derivative is
        otherwise unstable).
      - time_weight: weight by t (the boundary end t->0 is already pinned by the
        parameterization; emphasize larger t where the map is being learned).
    """
    from torch.func import jvp

    B = x0.shape[0]
    device = x0.device
    eps = sample_uniform_noise(x0.shape, device=device, dtype=x0.dtype)
    t = torch.rand(B, device=device, dtype=x0.dtype).clamp(min=1e-3, max=1.0)

    if start_from_noise:
        x_t = integrate_teacher_between(teacher, eps, torch.ones_like(t), t, substeps=substeps)
    else:
        x_t = interpolate(x0, eps, t)

    # tangent in x is -u^T (flow direction), tangent in t is +1
    with torch.no_grad():
        u_T = teacher(x_t, t)
    tx = -u_T
    tt = torch.ones_like(t)

    # df/dt along the flow via JVP through f(x,t) = (1-t)x + t F(x,t)
    def fn(x, tau):
        return model.f(x, tau, use_target=False)

    f_val, dfdt = jvp(fn, (x_t, t), (tx, tt))

    residual = dfdt  # want this -> 0
    if tangent_norm:
        scale = (residual.detach().norm(dim=1, keepdim=True) + norm_c)
        residual = residual / scale

    loss_per = (residual ** 2).sum(dim=1)
    if time_weight:
        loss_per = loss_per * t
    return loss_per.mean()


def train_continuous_scm(model, teacher, data_sampler, iters, batch, lr, device,
                         tangent_norm=True, norm_c=0.1, time_weight=True,
                         start_from_noise=False, substeps=8, weight_decay=0.0,
                         endpoint_sup_weight=0.0, endpoint_sup_steps=30,
                         log_every=500):
    """
    Train a ConsistencyModel by continuous-time sCM.

    endpoint_sup_weight > 0 optionally adds direct endpoint supervision (regress
    f(eps,1) onto the full teacher-integration endpoint), for comparison/ablation
    against pure continuous sCM (endpoint_sup_weight=0).
    """
    opt = torch.optim.AdamW(model.F.parameters(), lr=lr, weight_decay=weight_decay)
    model.F.train(); teacher.eval()
    for it in range(iters):
        x0 = data_sampler(batch)
        opt.zero_grad(set_to_none=True)
        loss = continuous_scm_loss(model, teacher, x0, tangent_norm=tangent_norm,
                                   norm_c=norm_c, time_weight=time_weight,
                                   start_from_noise=start_from_noise, substeps=substeps)
        if endpoint_sup_weight > 0:
            z = sample_uniform_noise((batch, model.data_dim), device=device)
            with torch.no_grad():
                tgt = integrate_teacher(teacher, z, steps=endpoint_sup_steps)
            f1 = model.f(z, torch.ones(batch, device=device))
            loss = loss + endpoint_sup_weight * ((f1 - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        model.update_target()
        if it % log_every == 0 or it == iters - 1:
            print(f"[cscm] iter {it:5d}  loss {loss.item():.6f}")
    model.F.eval()
    return model


def integrate_teacher(teacher, z, steps=30):
    """Full reverse integration of the teacher PF-ODE from noise z to data."""
    x = z.clone()
    ts = torch.linspace(1.0, 0.0, steps + 1, device=z.device)
    for k in range(steps):
        t = ts[k].expand(x.shape[0])
        with torch.no_grad():
            x = x + teacher(x, t) * (ts[k] - ts[k + 1]).item()
    return x


def train_cd(model, teacher, data_sampler, iters, batch, lr, device,
             n_times=40, huber_delta=0.0, weight_decay=0.0, log_every=500,
             endpoint_sup_weight=1.0, endpoint_sup_steps=30, endpoint_sup_every=2):
    """
    Train a ConsistencyModel by CD against a frozen teacher.

    IMPORTANT empirical finding: pure CD bootstrap (consistency loss alone) does
    NOT reliably propagate the trajectory endpoint on this substrate — it satisfies
    local consistency (tiny loss) while leaving the global map wrong (dropped modes).
    Adding DIRECT endpoint supervision — regress f(eps,1) onto the full teacher-
    integration endpoint — fixes it (8/8 modes, matched spread). With endpoint
    supervision on, this is effectively trajectory/CD-with-targets regression; the
    consistency term becomes auxiliary. Set endpoint_sup_weight=0 for pure CD.
    """
    opt = torch.optim.AdamW(model.F.parameters(), lr=lr, weight_decay=weight_decay)
    model.F.train(); teacher.eval()
    for it in range(iters):
        x0 = data_sampler(batch)
        opt.zero_grad(set_to_none=True)
        loss = cd_loss(model, teacher, x0, n_times=n_times, huber_delta=huber_delta)
        if endpoint_sup_weight > 0 and it % endpoint_sup_every == 0:
            z = sample_uniform_noise((batch, model.data_dim), device=device)
            with torch.no_grad():
                tgt = integrate_teacher(teacher, z, steps=endpoint_sup_steps)
            f1 = model.f(z, torch.ones(batch, device=device))
            loss = loss + endpoint_sup_weight * ((f1 - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        model.update_target()
        if it % log_every == 0 or it == iters - 1:
            print(f"[cd] iter {it:5d}  loss {loss.item():.6f}")
    model.F.eval()
    return model
