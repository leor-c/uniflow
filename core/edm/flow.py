"""Expose a trained EDM denoiser as a flow-matching velocity field.

Align Your Flow distils diffusion teachers inside the linear flow-matching
framework rather than in EDM's sigma space, moving between the two with the
change of variables ``sigma_t = t / (1 - t)`` (their convention: ``t = 0``
data, ``t = 1`` noise). Every arm in this repo runs the opposite way round --
``t = 0`` noise, ``t = 1`` data, ``z_t = (1 - t) z_0 + t z_1`` -- so the same
relation reads

    sigma(t) = (1 - t) / t,   x_hat(t) = z_t / t,
    v(z_t, t) = (D(x_hat, sigma) - z_t) / (1 - t).

Wrapping the EDM teacher this way lets the shared RF AYF student, objective,
interval schedule and samplers drive it without modification, so the three
benchmark arms differ only in their teacher and base distribution.

It also removes the preconditioned unit-scale state ``y = z / sqrt(sigma^2 +
sigma_data^2)`` this arm used to distil through. In that coordinate the true
velocity collapses to ``O((sigma_data / sigma)^2)`` at the noise end while the
teacher's denoiser error does not, so at ``sigma_max = 160`` the distillation
target was roughly twenty times smaller than the error on it. The
flow-matching velocity stays ``O(1)`` across the whole interval instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .models import EDMPrecondDenoiser


class EDMFlowVelocity(nn.Module):
    """Present an :class:`EDMPrecondDenoiser` with the ``RFTeacher`` interface.

    Calling the module returns the flow-matching velocity ``v(z, t)`` for the
    linear interpolant, with ``t = 0`` noise and ``t = 1`` data.

    The velocity is evaluated in an algebraically stabilised form. Writing
    ``u = 1 - t`` and ``S^2 = u^2 + sigma_data^2 t^2`` (so ``S = t * sqrt(sigma^2
    + sigma_data^2)``), substituting EDM's preconditioning into
    ``v = (D - z) / u`` and cancelling the common factor of ``u`` gives

        v(z, t) = z * (sigma_data^2 * t - u) / S^2  +  sigma_data * F(z / S) / S

    where ``F`` is the raw network output. Both endpoints are finite: at
    ``t = 0`` this is ``-z + sigma_data * F`` and at ``t = 1`` it is ``z + F``,
    matching the exact limits ``v(z, 0) = -z_0`` and ``v(z, 1) = z_1`` for a
    teacher whose residual vanishes at extreme noise levels. Only the noise
    level used for conditioning is clamped, to ``[sigma_min, sigma_max]``; the
    ``t``-dependent coefficients stay exact so the limits are preserved.
    """

    def __init__(
            self,
            denoiser: EDMPrecondDenoiser,
            sigma_min: float = 0.002,
            sigma_max: float = 80.0,
    ):
        super().__init__()
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError("Expected 0 < sigma_min < sigma_max")
        self.denoiser = denoiser
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    @property
    def sigma_data(self) -> float:
        return self.denoiser.sigma_data

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]

        sigma_data = self.denoiser.sigma_data
        u = 1.0 - t
        squared_scale = u.square() + (sigma_data ** 2) * t.square()
        scale = squared_scale.sqrt()

        # sigma = u / t diverges at t = 0 and vanishes at t = 1; it is used only
        # to condition the network, so clamping it keeps the embedding finite
        # without perturbing the coefficients that carry the endpoint limits.
        sigma = torch.clamp(u / t, self.sigma_min, self.sigma_max)
        residual = self.denoiser.network_residual(z / scale, sigma.log())

        return (
            z * ((sigma_data ** 2) * t - u) / squared_scale
            + sigma_data * residual / scale
        )
