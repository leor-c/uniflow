"""Shared data, autoencoder, evaluation, and plotting helpers for toy runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules import MLP
from toy.datasets import get_sampler

DENSITY_SAMPLES = 2048
DENSITY_PROTOCOL = "exact-target-nll-kde-coverage-three-sigma-support-v1"
BENCHMARK_PROTOCOL = "global-scalar-latent-scale-edm-flow-matching-ayf-v5"
LATENT_NORMALIZATION_SAMPLES = 4096
TARGET_NOISE_SCALES = {
    "8gaussians": 0.035,
    "two_moons": 0.05,
    "checkerboard": 0.12,
}
TWO_MOONS_QUADRATURE_POINTS = 2048


def sample_8gaussians(
    batch_size: int,
    device: str | torch.device,
    radius: float = 0.75,
    std: float = 0.035,
) -> torch.Tensor:
    """Eight isotropic Gaussian modes arranged on a circle."""
    k = torch.randint(0, 8, (batch_size,), device=device)
    angles = 2 * math.pi * k.float() / 8.0
    centers = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1
    )
    return centers + std * torch.randn(batch_size, 2, device=device)


class ToyAE(nn.Module):
    """Two-dimensional autoencoder with bounded or standardized latents.

    The unbounded variant is trained as a plain reconstruction autoencoder. After
    training, ``fit_latent_normalization`` records a fixed affine transform that
    centres the latents and rescales them by a single global scalar. ``decode``
    applies the exact inverse, so normalization does not alter autoencoder
    reconstructions.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden: int = 128,
        depth: int = 2,
        bounded: bool = True,
    ):
        super().__init__()
        self.encoder_net = MLP(2, latent_dim, hidden=hidden, depth=depth)
        self.decoder_net = MLP(latent_dim, 2, hidden=hidden, depth=depth)
        self.bounded = bounded
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_net(x)
        if self.bounded:
            return torch.tanh(z)
        return (z - self.latent_mean) / self.latent_std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if not self.bounded:
            z = z * self.latent_std + self.latent_mean
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def fit_latent_normalization(self, x: torch.Tensor) -> None:
        """Fit a centring shift and one global scale for an unbounded encoder.

        The scale is a single scalar, not a per-coordinate vector, so the
        transform is a similarity (translation plus isotropic scaling) and the
        latent manifold reaches the generative model undistorted. This follows
        latent-diffusion practice, which rescales the whole latent by one
        scalar factor. Per-coordinate whitening would instead amplify the
        lowest-variance encoder directions -- for a 2D toy manifold in 32
        latent dimensions those carry almost no signal, and blowing them up to
        unit variance leaves a target the multi-step teacher still fits but the
        one-step student does not.
        """
        if self.bounded:
            raise ValueError("Bounded latents are not Gaussian-standardized")
        raw_latents = self.encoder_net(x)
        latent_mean = raw_latents.mean(dim=0)
        # Measure the scale after centring, so the normalized latents really do
        # have unit overall standard deviation and match the base distribution.
        latent_std = (raw_latents - latent_mean).std(correction=0)
        if latent_std <= 1e-6:
            raise RuntimeError("Cannot standardize a collapsed latent space")
        self.latent_mean.copy_(latent_mean)
        self.latent_std.fill_(latent_std.item())


def _uniform_latent_moment_regularizer(z: torch.Tensor) -> torch.Tensor:
    """Match bounded latent marginals to the first two uniform moments."""
    mean_loss = z.mean(dim=0).pow(2).mean()
    second_moment_loss = (z.pow(2).mean(dim=0) - 1.0 / 3.0).pow(2).mean()
    return mean_loss + second_moment_loss


def train_toy_ae(ae: ToyAE, data_sampler, cfg) -> None:
    ae.train()
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.ae_lr)
    for step in range(1, cfg.ae_steps + 1):
        x = data_sampler(cfg.batch_size)
        x_hat, z = ae(x)
        recon_loss = F.mse_loss(x_hat, x)
        if ae.bounded:
            reg_loss = _uniform_latent_moment_regularizer(z)
            loss = recon_loss + cfg.latent_reg_weight * reg_loss
        else:
            reg_loss = None
            loss = recon_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            message = f"[AE] step {step:05d} | recon {recon_loss.item():.6f}"
            if reg_loss is not None:
                message += f" | uniform moment regularizer {reg_loss.item():.6f}"
            print(message)
    ae.eval()
    if not ae.bounded:
        calibration_data = data_sampler(LATENT_NORMALIZATION_SAMPLES)
        ae.fit_latent_normalization(calibration_data)


def make_data_sampler(dataset: str, device: str | torch.device):
    if dataset == "8gaussians":
        return lambda n: sample_8gaussians(n, device)
    return get_sampler(dataset, device=device)


def make_latent_sampler(ae: ToyAE, data_sampler, device: str | torch.device):
    @torch.no_grad()
    def sampler(batch_size: int) -> torch.Tensor:
        return ae.encode(data_sampler(batch_size).to(device))

    return sampler


@torch.no_grad()
def _uniform_subsample_rows(
    points: torch.Tensor,
    samples: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Select a reproducible uniform subset without assuming random row order."""
    if len(points) <= samples:
        return points
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(len(points), generator=generator)[:samples]
    return points[indices]


@torch.no_grad()
def _validate_sample_matrix(points: torch.Tensor, metric: str) -> None:
    if len(points) == 0:
        raise ValueError(f"{metric} requires at least one sample")
    if points.ndim != 2:
        raise ValueError(f"{metric} input must be a two-dimensional sample matrix")


def _fixed_target_components(dataset: str) -> tuple[torch.Tensor, float]:
    if dataset == "8gaussians":
        angles = torch.arange(8, dtype=torch.float64) * (2.0 * math.pi / 8.0)
        centers = 0.75 * torch.stack([angles.cos(), angles.sin()], dim=1)
    elif dataset == "checkerboard":
        centers = torch.tensor(
            [
                [-0.9, -0.9],
                [-0.9, 0.9],
                [0.0, 0.0],
                [0.9, -0.9],
                [0.9, 0.9],
            ],
            dtype=torch.float64,
        )
    elif dataset == "two_moons":
        t = (
            torch.arange(TWO_MOONS_QUADRATURE_POINTS, dtype=torch.float64) + 0.5
        ) * (math.pi / TWO_MOONS_QUADRATURE_POINTS)
        first = 0.5 * torch.stack([t.cos(), t.sin()], dim=1) - 0.25
        second = torch.stack(
            [0.25 - 0.5 * t.cos(), -0.5 * t.sin()],
            dim=1,
        )
        centers = torch.cat([first, second], dim=0)
    else:
        raise ValueError(f"Unknown toy dataset: {dataset}")
    return centers, TARGET_NOISE_SCALES[dataset]


def _gaussian_mixture_log_density(
    points: torch.Tensor,
    centers: torch.Tensor,
    bandwidth: float,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Log density of an equally weighted isotropic Gaussian mixture."""
    _validate_sample_matrix(points, "density evaluation")
    if centers.ndim != 2 or points.shape[1] != centers.shape[1]:
        raise ValueError("Density points and mixture centers must have matching dimensions")
    if bandwidth <= 0:
        raise ValueError("Density bandwidth must be strictly positive")

    points = points.detach().cpu().double()
    centers = centers.detach().cpu().double()
    log_normalizer = (
        math.log(len(centers))
        + points.shape[1] * math.log(bandwidth)
        + 0.5 * points.shape[1] * math.log(2.0 * math.pi)
    )
    values = []
    for chunk in points.split(chunk_size):
        squared_distance = torch.cdist(chunk, centers).square()
        values.append(
            torch.logsumexp(-squared_distance / (2.0 * bandwidth**2), dim=1)
            - log_normalizer
        )
    return torch.cat(values)


@torch.no_grad()
def target_negative_log_likelihood(
    dataset: str,
    generated: torch.Tensor,
    samples: int = DENSITY_SAMPLES,
) -> float:
    """Generated-sample NLL under the known toy target density (fidelity)."""
    _validate_sample_matrix(generated, "target NLL")
    generated = _uniform_subsample_rows(
        generated.detach().cpu(), min(len(generated), samples), seed=0
    )
    centers, noise = _fixed_target_components(dataset)
    return -_gaussian_mixture_log_density(generated, centers, noise).mean().item()


@torch.no_grad()
def kde_coverage_negative_log_likelihood(
    dataset: str,
    real: torch.Tensor,
    generated: torch.Tensor,
    samples: int = DENSITY_SAMPLES,
) -> float:
    """Held-out target NLL under a fixed-bandwidth generated-sample KDE."""
    _validate_sample_matrix(real, "coverage NLL")
    _validate_sample_matrix(generated, "coverage NLL")
    if real.shape[1] != generated.shape[1]:
        raise ValueError("Coverage NLL inputs must have matching dimensions")
    n = min(len(real), len(generated), samples)
    real = _uniform_subsample_rows(real.detach().cpu(), n, seed=0)
    generated = _uniform_subsample_rows(generated.detach().cpu(), n, seed=0)
    log_density = _gaussian_mixture_log_density(
        real,
        generated,
        TARGET_NOISE_SCALES[dataset],
    )
    return -log_density.mean().item()


def _two_moons_support_distance(points: torch.Tensor) -> torch.Tensor:
    radius = 0.5
    first_center = points.new_tensor([-0.25, -0.25])
    second_center = points.new_tensor([0.25, 0.0])

    def semicircle_distance(center: torch.Tensor, *, upper: bool) -> torch.Tensor:
        offset = points - center
        on_arc_side = offset[:, 1] >= 0 if upper else offset[:, 1] <= 0
        radial = (offset.norm(dim=1) - radius).abs()
        left = center + points.new_tensor([-radius, 0.0])
        right = center + points.new_tensor([radius, 0.0])
        endpoint = torch.minimum(
            (points - left).norm(dim=1),
            (points - right).norm(dim=1),
        )
        return torch.where(on_arc_side, radial, endpoint)

    return torch.minimum(
        semicircle_distance(first_center, upper=True),
        semicircle_distance(second_center, upper=False),
    )


@torch.no_grad()
def clean_support_distance(dataset: str, points: torch.Tensor) -> torch.Tensor:
    """Distance to the noiseless modes or arcs underlying a toy dataset."""
    _validate_sample_matrix(points, "support distance")
    points = points.detach().cpu().double()
    if dataset == "two_moons":
        return _two_moons_support_distance(points)
    centers, _ = _fixed_target_components(dataset)
    return torch.cdist(points, centers).min(dim=1).values


@torch.no_grad()
def off_support_rate(
    dataset: str,
    generated: torch.Tensor,
    samples: int = DENSITY_SAMPLES,
) -> float:
    """Fraction farther than three target noise scales from clean support."""
    _validate_sample_matrix(generated, "off-support rate")
    generated = _uniform_subsample_rows(
        generated.detach().cpu(), min(len(generated), samples), seed=0
    )
    threshold = 3.0 * TARGET_NOISE_SCALES[dataset]
    return (clean_support_distance(dataset, generated) > threshold).double().mean().item()


@torch.no_grad()
def evaluate_density_outputs(
    dataset: str,
    real: torch.Tensor,
    outputs: dict[str, torch.Tensor],
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "evaluation_protocol": DENSITY_PROTOCOL,
    }
    for output_name, generated in outputs.items():
        metrics[f"{output_name}_target_nll"] = target_negative_log_likelihood(
            dataset, generated
        )
        metrics[f"{output_name}_coverage_nll"] = (
            kde_coverage_negative_log_likelihood(dataset, real, generated)
        )
        metrics[f"{output_name}_off_support_rate"] = off_support_rate(
            dataset, generated
        )
    return metrics


def save_evaluation_samples(
    real: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    metrics_output_path: str,
) -> None:
    """Save ignored evaluation tensors so metrics can be changed without retraining."""
    path = Path(metrics_output_path).with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "real": real.detach().cpu(),
            "outputs": {name: value.detach().cpu() for name, value in outputs.items()},
        },
        path,
    )
    print(f"Saved evaluation samples -> {path}")


def save_benchmark_figure(
    panels: list[tuple[str, torch.Tensor]],
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.6 * len(panels), 4.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, (title, pts) in zip(axes, panels):
        pts = pts.detach().cpu()
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.45)
        ax.set_title(title, pad=10)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.3)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved figure -> {path}")


def save_metrics(metrics: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"Saved metrics -> {path}")
