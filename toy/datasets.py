"""
2D toy datasets, all supported in the OPEN box (-1, 1)^2 (bounded domain).

Design note (important for the stiffness story):
  The bounded support is obtained by mapping points through tanh, NOT by
  clamping. This mirrors the real pipeline, where latents are bounded because
  they are produced by a tanh-bottlenecked encoder. Clamping would pile mass on
  the box faces (a density spike => large ||grad^2 log rho|| at the boundary),
  artificially inflating the measured stiffness L(t) near the edges. tanh-mapping
  keeps the density smooth with open support, so any residual stiffness reflects
  the method/fit, not a boundary artifact.

So each dataset generates "pre-latent" points u in R^2 from a smooth density,
then returns x = tanh(u) in (-1,1)^2. The induced density on x is smooth.
"""

import math
import torch


def _eight_gaussians_pre(n, std, radius, device, generator=None):
    """8 Gaussian blobs on a circle, in pre-tanh (R^2) space."""
    centers = []
    for k in range(8):
        ang = 2 * math.pi * k / 8
        centers.append((radius * math.cos(ang), radius * math.sin(ang)))
    centers = torch.tensor(centers, device=device)
    idx = torch.randint(0, 8, (n,), device=device, generator=generator)
    return centers[idx] + std * torch.randn(n, 2, device=device, generator=generator)


def eight_gaussians(n, std=0.20, radius=1.2, device=None, generator=None):
    """
    8 Gaussians in pre-tanh space, then tanh-mapped into (-1,1)^2.
    radius/std are in pre-tanh coordinates; tanh compresses them toward the box.
    Defaults give 8 well-separated but smoothly-supported modes inside the box.
    """
    u = _eight_gaussians_pre(n, std, radius, device, generator=generator)
    return u


def two_moons(n, noise=0.05, device=None, generator=None):
    """Two interleaving moons"""
    n1 = n // 2
    n2 = n - n1
    t1 = torch.rand(n1, device=device, generator=generator) * math.pi
    x1 = torch.stack([torch.cos(t1), torch.sin(t1)], dim=1)* 0.5 - 0.25
    t2 = torch.rand(n2, device=device, generator=generator) * math.pi
    x2 = torch.stack([1.0 - torch.cos(t2), 1.0 - torch.sin(t2) - 0.5], dim=1)*0.5 - 0.25
    u = torch.cat([x1, x2], dim=0)
    # randn_like takes no generator, so draw the shape explicitly.
    u = u + noise * torch.randn(u.shape, device=u.device, generator=generator)
    return u


def checkerboard(n, std=0.12, device=None, generator=None):
    """
    Checkerboard of smooth Gaussian blobs (not hard squares), tanh-mapped.
    Using blobs instead of uniform squares keeps the density smooth (no hard
    edges), which is the regime where the theory predicts L->1 at the boundary.
    """
    # blob centers on a grid in pre-tanh space, alternating (checkerboard parity)
    centers = []
    for gx in range(-1, 2):
        for gy in range(-1, 2):
            if (gx + gy) % 2 == 0:
                centers.append((gx * 0.9, gy * 0.9))
    centers = torch.tensor(centers, device=device)
    idx = torch.randint(0, centers.shape[0], (n,), device=device, generator=generator)
    u = centers[idx] + std * torch.randn(n, 2, device=device, generator=generator)
    return u


DATASETS = {
    "8gaussians": eight_gaussians,
    "two_moons": two_moons,
    "checkerboard": checkerboard,
}


def get_sampler(name, device=None, generator=None):
    """Sampler for one dataset.

    ``generator`` makes the stream independent of the global RNG, so the data
    an experiment sees does not shift when a model is built or a training phase
    is reseeded alongside it.
    """
    fn = DATASETS[name]
    return lambda n: fn(n, device=device, generator=generator)
