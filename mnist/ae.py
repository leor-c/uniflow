"""
Convolutional autoencoder for MNIST (28×28 grayscale → bounded latent space).

Architecture:
    Encoder: Conv(1→32, 3×3, s=2) → Conv(32→64, 3×3, s=2) → Conv(64→128, 3×3) → Linear → tanh
    Decoder: Linear → Upsample(2×) → Conv(128→64, 3×3) → Upsample(2×) → Conv(64→32, 3×3)
             → Conv(32→1, 3×3) → Sigmoid

Decoder uses nearest-neighbour upsample + conv instead of ConvTranspose2d to
avoid checkerboard artefacts (Odena et al., 2016).

Latent vectors are bounded in (-1, 1)^latent_dim by the encoder tanh, compatible
with the Uniform[-1,1]^d noise prior used by the Uniform RF arm.

Public API
----------
    MNISTAutoEncoder            – full AE
    train_mnist_ae              – training loop (MSE reconstruction loss)
    encode_dataset              – pre-encode entire dataset → latent tensor
    make_mnist_latent_sampler   – fast random sampler over pre-encoded latents
    make_mnist_data_loader      – torchvision DataLoader for MNIST
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class MNISTEncoder(nn.Module):
    """
    (B, 1, 28, 28) → latent_dim-d vector in (-1, 1)^d.

    Spatial map (stride-2 conv for downsampling, kernel 3 throughout):
        28×28 → Conv(s=2) → 14×14 → Conv(s=2) → 7×7 → Conv(s=1) → 7×7
        → Flatten → Linear → tanh
    """

    def __init__(self, latent_dim: int = 64, bounded: bool = True):
        super().__init__()
        self.bounded = bounded
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(16,   32,  kernel_size=3, stride=2, padding=1),  # 28→14
            nn.SiLU(),
            nn.Conv2d(32,  64,  kernel_size=3, stride=2, padding=1),  # 14→7
            nn.SiLU(),
            nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1),  # 7→7
            nn.SiLU(),
        )
        self.head = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw encoder output; the tanh boundary applies only to bounded latents."""
        h = self.net(x).flatten(1)
        z = self.head(h)
        return torch.tanh(z) if self.bounded else z


class MNISTDecoder(nn.Module):
    """
    latent_dim-d vector → (B, 1, 28, 28) pixel image in (0, 1).

    Spatial map (nearest upsample + conv, no ConvTranspose):
        latent → Linear → (128, 7, 7)
        → Upsample(2×) → Conv(3×3) → (64, 14, 14)
        → Upsample(2×) → Conv(3×3) → (32, 28, 28)
        → Conv(3×3) → (1, 28, 28) → Sigmoid
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),          # 7→14
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),          # 14→28
            nn.Conv2d(64,  32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(32,   16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).view(-1, 128, 7, 7)
        return self.net(h)


class MNISTAutoEncoder(nn.Module):
    """Full MNIST AE, mirroring ``toy.common.ToyAE``.

    ``bounded=True`` gives tanh latents in ``(-1, 1)^d`` for the uniform arm.
    ``bounded=False`` trains a plain reconstruction autoencoder whose latents are
    then centred and divided by a single global scalar
    (``fit_latent_normalization``), so the transform is a similarity and the
    latent geometry reaches the generative model undistorted -- the Gaussian RF
    and EDM arms both use that form.
    """

    def __init__(self, latent_dim: int = 64, bounded: bool = True):
        super().__init__()
        self.encoder = MNISTEncoder(latent_dim, bounded=bounded)
        self.decoder = MNISTDecoder(latent_dim)
        self.latent_dim = latent_dim
        self.bounded = bounded
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

    def raw_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output before standardization (already tanh'd if bounded)."""
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.bounded:
            return z
        return (z - self.latent_mean) / self.latent_std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if not self.bounded:
            z = z * self.latent_std + self.latent_mean
        return self.decoder(z)

    @torch.no_grad()
    def fit_latent_normalization(self, x: torch.Tensor) -> None:
        """Centre, then rescale by one global scalar. See ToyAE for the rationale."""
        if self.bounded:
            raise ValueError("Bounded latents are not Gaussian-standardized")
        raw = self.raw_latent(x)
        mean = raw.mean(dim=0)
        std = (raw - mean).std(correction=0)
        if std <= 1e-6:
            raise RuntimeError("Cannot standardize a collapsed latent space")
        self.latent_mean.copy_(mean)
        self.latent_std.fill_(std.item())

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mnist_ae(
    ae: MNISTAutoEncoder,
    loader,
    iters: int,
    lr: float,
    device: str | torch.device,
    latent_spread_weight: float = 1e-3,
    log_every: int = 500,
) -> None:
    """
    Train the MNIST AE.

    Reconstruction loss: mean squared error on pixels.

    Regularisation: mild per-dim second-moment penalty encouraging the latent
distribution to use the full [-1,1]^d box (mean→0, variance→1/3).
    """
    ae.to(device)
    ae.train()
    opt = torch.optim.AdamW(ae.parameters(), lr=lr)
    data_iter = _infinite(loader)

    for step in range(1, iters + 1):
        x, _ = next(data_iter)
        x = x.to(device)

        opt.zero_grad(set_to_none=True)
        x_hat, z = ae(x)

        recon = F.mse_loss(x_hat, x)
        loss = recon

        if ae.bounded and latent_spread_weight > 0:
            mean_pen = z.mean(dim=0).pow(2).mean()
            var_pen  = (z.var(dim=0) - 1.0 / 3.0).pow(2).mean()
            loss = loss + latent_spread_weight * (mean_pen + var_pen)

        loss.backward()
        opt.step()

        if step % log_every == 0 or step == iters:
            print(f"[mnist ae] step {step:05d} | mse {recon.item():.4f}")

    ae.eval()
    if not ae.bounded:
        with torch.no_grad():
            calibration, _ = next(_infinite(loader))
            ae.fit_latent_normalization(calibration.to(device))
        print(f"[mnist ae] latent global scale {ae.latent_std[0].item():.4f}")


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def make_mnist_data_loader(
    root: str = "./data",
    train: bool = True,
    batch_size: int = 256,
    num_workers: int = 2,
):
    """Return a torchvision DataLoader for MNIST (downloads if needed)."""
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError as e:
        raise ImportError(
            "torchvision is required for MNIST experiments. "
            "Install with: pip install torchvision"
        ) from e

    dataset = torchvision.datasets.MNIST(
        root=root, train=train, download=True,
        transform=T.ToTensor(),
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


@torch.no_grad()
def encode_dataset(
    ae: MNISTAutoEncoder,
    loader,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Pre-encode all images in `loader` to latents stored on CPU.

    Returns [N, latent_dim]. Sampling from this during flow training avoids
    repeated CNN forward passes and decouples the flow from the data pipeline.
    """
    ae.eval()
    all_z = []
    for x, _ in loader:
        all_z.append(ae.encode(x.to(device)).cpu())
    return torch.cat(all_z, dim=0)


def make_mnist_latent_sampler(
    all_latents: torch.Tensor,
    device: str | torch.device,
):
    """Returns callable(batch_size) → [B, latent_dim] on device."""
    def sampler(batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, len(all_latents), (batch_size,))
        return all_latents[idx].to(device)
    return sampler


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _infinite(loader):
    while True:
        yield from loader
