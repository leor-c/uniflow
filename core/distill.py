"""
Velocity-matching distillation (DMD2-style, score-free).

Setup:
  - Frozen teacher velocity net  v_T  (rectified-flow teacher, core.velocity).
  - Student generator  G_phi: z (uniform noise) -> x0_hat  (one step).
  - Fake-velocity critic  v_S_psi: tracks the velocity field of the STUDENT's
    output distribution, trained with the same RF regression loss but on
    student samples. Updated on a faster timescale than the generator
    (two-timescale rule).

Distribution-matching gradient (the Theorem-1 quantity):
  Corrupt a student sample to level t:  x_t = (1-t) x0_hat + t eps.
  The gradient signal pushing x0_hat is proportional to the velocity difference
      (v_S(x_t, t) - v_T(x_t, t)),
  back-propagated into phi. This is the score-free analogue of the DMD2 KL
  gradient (which used score differences); here we use velocity differences.

We implement the standard DMD-style "stop-gradient through the difference, treat
it as the target direction" trick so the generator loss is a simple MSE whose
gradient equals the (detached) velocity difference.
"""

import copy
import torch
import torch.nn as nn

from .interp import interpolate, velocity_target, sample_uniform_noise, sample_t, _broadcast_t


class OneStepGenerator(nn.Module):
    """
    Velocity-parameterized one-step generator.

    The student is now a VELOCITY network u^S_phi(z, t) — the SAME type of object
    as the teacher (it points along the data-minus-noise direction). The clean
    sample is reconstructed by taking a single full Euler step from the noise:
        x0_hat = clamp( z + u^S_phi(z, t=1), -1, 1 ).
    This keeps the student type-consistent with the teacher (both velocities),
    so the distribution-matching signal compares velocity-to-velocity rather than
    mixing a clean-data point against a velocity.

    Architecturally identical to MLPVelocity (it IS a velocity net). The reusable
    velocity backbone is wrapped so .forward(z) returns the reconstructed x0_hat,
    and .velocity(z, t) exposes the raw velocity if needed (few-step integration).
    """

    def __init__(self, data_dim, hidden=128, depth=2, t_emb_dim=128, clamp=True):
        super().__init__()
        from .velocity import MLPVelocity
        self.net = MLPVelocity(data_dim, hidden=hidden, depth=depth, t_emb_dim=t_emb_dim)
        self.data_dim = data_dim
        self.clamp = clamp

    def velocity(self, z, t):
        """Raw student velocity u^S(z, t)."""
        return self.net(z, t)

    def forward(self, z):
        """One full Euler step from noise (t=1) to data (t=0): x0_hat = z + u^S(z,1)."""
        t1 = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        x0_hat = z + self.net(z, t1)
        if self.clamp:
            x0_hat = x0_hat.clamp(-1.0, 1.0)
        return x0_hat


def critic_loss(critic, x0_hat_detached):
    """
    RF regression loss for the fake critic on (detached) student samples.
    Trains v_S to be the velocity field of the student's current output dist.
    """
    B = x0_hat_detached.shape[0]
    device = x0_hat_detached.device
    eps = sample_uniform_noise(x0_hat_detached.shape, device=device,
                               dtype=x0_hat_detached.dtype)
    t = sample_t(B, device=device, dtype=x0_hat_detached.dtype)
    x_t = interpolate(x0_hat_detached, eps, t)
    target = velocity_target(x0_hat_detached, eps)
    pred = critic(x_t, t)
    return ((pred - target) ** 2).mean()


def generator_loss(generator, teacher, critic, z, t_sample_fn=None, weighting="none"):
    """
    Distribution-matching loss for the generator.

    Builds the velocity-difference target (detached) and returns an MSE whose
    gradient w.r.t. phi equals  E[ (v_S - v_T) . d x_t / d phi ].

    Args:
        z: (B, d) uniform noise input to the generator.
        t_sample_fn: optional callable(B, device, dtype)->t; defaults to U(0,1).
    """
    device = z.device
    x0_hat = generator(z)                       # (B, d), differentiable in phi
    B = x0_hat.shape[0]
    eps = sample_uniform_noise(x0_hat.shape, device=device, dtype=x0_hat.dtype)
    if t_sample_fn is None:
        t = sample_t(B, device=device, dtype=x0_hat.dtype)
    else:
        t = t_sample_fn(B, device, x0_hat.dtype)
    x_t = interpolate(x0_hat, eps, t)           # differentiable in phi

    with torch.no_grad():
        v_T = teacher(x_t, t)
        v_S = critic(x_t, t)
        # Our distribution-matching signal (findings §4): the velocity difference.
        # v_S tracks the student's distribution, v_T the teacher's; both point
        # toward data. To move the student's distribution toward the teacher's,
        # gradient DESCENT should move x0_hat by (v_T - v_S), i.e. the surrogate
        # gradient on x0_hat is (v_S - v_T).
        grad = (v_S - v_T)
        # Optional t-weighting of the signal. The uniform-noise theory does not
        # pin this down, so it is a design choice to sweep:
        #   "none"  -> bare (v_S - v_T)
        #   "t"     -> t * (v_S - v_T)        [denoiser-difference reading: f_S - f_T = t(v_S - v_T)]
        #   "1-t"   -> (1-t) * (v_S - v_T)    [KL/chain-rule-through-x_t reading]
        if weighting != "none":
            if weighting == "t":
                wt = t
            elif weighting == "1-t":
                wt = 1.0 - t
            else:
                raise ValueError(f"unknown weighting {weighting!r}")
            grad = grad * _broadcast_t(wt, grad)

    # MSE surrogate whose gradient w.r.t. x0_hat equals `grad`:
    #   L = 0.5 * || x0_hat - stop_grad(x0_hat - grad) ||^2.
    # Gradient flows into x0_hat, then through the student's velocity
    # construction x0_hat = clamp(z + u^S(z,1)) into phi. Bare (v_S - v_T) signal
    # (no t-weighting) per the velocity-matching objective; the t/(1-t) weighting
    # is a design choice the uniform-noise theory does not pin down.
    target = (x0_hat - grad).detach()
    loss = 0.5 * ((x0_hat - target) ** 2).mean()
    return loss, x0_hat.detach()


class EMA:
    """Exponential moving average of model parameters (optional, for generator)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)


def distill_step(generator, teacher, critic, g_opt, c_opt, batch_size,
                 data_dim, device, critic_steps=5, gen_ema=None, weighting="none"):
    """
    One two-timescale distillation step:
      - update critic `critic_steps` times on fresh student samples,
      - then update generator once.
    `weighting` selects the t-weighting of the velocity-difference signal
    ("none" | "t" | "1-t"); see generator_loss.
    Returns (gen_loss, critic_loss) floats.
    """
    # --- critic updates (faster timescale) ---
    c_loss_val = 0.0
    for _ in range(critic_steps):
        z = sample_uniform_noise((batch_size, data_dim), device=device)
        with torch.no_grad():
            x0_hat = generator(z)
        c_opt.zero_grad(set_to_none=True)
        c_loss = critic_loss(critic, x0_hat)
        c_loss.backward()
        c_opt.step()
        c_loss_val = c_loss.item()

    # --- generator update (slower timescale) ---
    z = sample_uniform_noise((batch_size, data_dim), device=device)
    g_opt.zero_grad(set_to_none=True)
    g_loss, _ = generator_loss(generator, teacher, critic, z, weighting=weighting)
    g_loss.backward()
    g_opt.step()
    if gen_ema is not None:
        gen_ema.update(generator)

    return g_loss.item(), c_loss_val
