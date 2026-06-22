"""
Uniform-noise rectified-flow interpolation on the bounded box [-1, 1]^d.

Conventions (fixed once, used everywhere):
  - Data x0 ~ p_data, supported on [-1, 1]^d (e.g. via a tanh latent).
  - Noise eps ~ Uniform([-1, 1]^d).
  - Linear interpolation:   x_t = (1 - t) * x0 + t * eps,   t in [0, 1].
        t = 0 -> data,  t = 1 -> pure noise.
  - Velocity target (what the network regresses):
        w(x_t, t) = E[x0 - eps | x_t]          ("toward data" convention)
    The per-sample regression target is simply (x0 - eps), constant along
    each interpolation segment.
  - Generation integrates  dx/dt = -w(x, t)  from t=1 down to t=0
    (i.e. dx = w dt as t decreases). See velocity.py:generate.

Because x_t is a convex combination of two points in [-1,1]^d, x_t stays in
[-1,1]^d for all t. This boundedness is relied on throughout the theory.
"""

import torch


def sample_uniform_noise(shape, device=None, dtype=torch.float32):
    """eps ~ Uniform([-1, 1]^d)."""
    return torch.rand(shape, device=device, dtype=dtype) * 2.0 - 1.0


def interpolate(x0, eps, t):
    """
    x_t = (1 - t) * x0 + t * eps.

    Args:
        x0:  (B, ...) data samples.
        eps: (B, ...) uniform noise, same shape as x0.
        t:   (B,) or (B, 1, ...) broadcastable time in [0, 1].
    Returns:
        x_t: same shape as x0.
    """
    t = _broadcast_t(t, x0)
    return (1.0 - t) * x0 + t * eps


def velocity_target(x0, eps):
    """
    The rectified-flow velocity target w = x0 - eps ("toward data").

    Constant along the segment, independent of t. The marginal velocity the
    network learns is w(x_t, t) = E[x0 - eps | x_t].
    """
    return x0 - eps


def sample_t(batch_size, device=None, dtype=torch.float32, eps_margin=0.0):
    """
    Sample t ~ Uniform(eps_margin, 1 - eps_margin).

    eps_margin > 0 optionally avoids the exact endpoints. The theory says the
    endpoints are well-behaved (L(0)=L(1)=1), so margin can be 0; a tiny margin
    can help numerical stability of the denoiser identity f = x_t + t*w near t=0.
    """
    lo, hi = eps_margin, 1.0 - eps_margin
    return torch.rand(batch_size, device=device, dtype=dtype) * (hi - lo) + lo


def denoiser_from_velocity(x_t, w, t):
    """
    Recover the (derived) denoiser  f = E[x0 | x_t] = x_t + t * w.

    This is the exact algebraic identity from the theory (Section 5):
        x_t = (1-t) f + t g,  w = f - g  =>  f = x_t + t w.
    Useful for diagnostics and for the stiffness identity, NOT a trained object.
    """
    t = _broadcast_t(t, x_t)
    return x_t + t * w


def _broadcast_t(t, ref):
    """Reshape t of shape (B,) to broadcast against ref of shape (B, ...)."""
    if t.dim() == ref.dim():
        return t
    if t.dim() == 1:
        return t.view(t.shape[0], *([1] * (ref.dim() - 1)))
    raise ValueError(f"Cannot broadcast t of shape {t.shape} against {ref.shape}")
