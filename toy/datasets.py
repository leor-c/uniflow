"""2D toy data samplers for the autoencoder benchmark."""

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
    Eight Gaussian modes in the plane. This sampler is not the benchmark
    8-Gaussians path, which uses toy.common.sample_8gaussians.
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
    Checkerboard of smooth Gaussian blobs rather than hard squares.
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


def branching_tree_segments():
    """Two mirrored binary trees, as (start, end) line segments in data space."""
    segments = []
    for side in (-1, 1):
        start = (side * 0.02, 0.0)
        end = (side * 0.25, 0.0)
        segments.append((start, end))
        tips = [(end, 0.0)]
        for level in range(4):
            children = []
            length = 0.29 * 0.75**level
            for origin, angle in tips:
                for turn in (-0.65, 0.65):
                    direction = angle + turn
                    child = (
                        origin[0] + side * length * math.cos(direction),
                        origin[1] + 1.2 * length * math.sin(direction),
                    )
                    segments.append((origin, child))
                    children.append((child, direction))
            tips = children
    return torch.tensor(segments, dtype=torch.float64)


def branching_tree_quadrature(points=2048):
    """Equally spaced arc-length midpoints for the tree's target density."""
    segments = branching_tree_segments()
    starts, ends = segments[:, 0], segments[:, 1]
    lengths = (ends - starts).norm(dim=1)
    cumulative = lengths.cumsum(dim=0)
    positions = (torch.arange(points, dtype=torch.float64) + 0.5) * (
        cumulative[-1] / points
    )
    index = torch.searchsorted(cumulative, positions)
    previous = torch.cat([cumulative.new_zeros(1), cumulative[:-1]])
    fraction = ((positions - previous[index]) / lengths[index]).unsqueeze(1)
    return starts[index] + fraction * (ends[index] - starts[index])


def branching_trees(n, device=None, generator=None):
    """Sample exactly on two branching trees, with no added noise."""
    segments = branching_tree_segments().to(device=device, dtype=torch.float32)
    starts, ends = segments[:, 0], segments[:, 1]
    lengths = (ends - starts).norm(dim=1)
    branch = torch.multinomial(lengths, n, replacement=True, generator=generator)
    position = torch.rand(n, 1, device=device, generator=generator)
    points = starts[branch] + position * (ends[branch] - starts[branch])
    return points


DATASETS = {
    "8gaussians": eight_gaussians,
    "two_moons": two_moons,
    "checkerboard": checkerboard,
    "branching_trees": branching_trees,
}


def get_sampler(name, device=None, generator=None):
    """Sampler for one dataset.

    ``generator`` makes the stream independent of the global RNG, so the data
    an experiment sees does not shift when a model is built or a training phase
    is reseeded alongside it.
    """
    fn = DATASETS[name]
    return lambda n: fn(n, device=device, generator=generator)
