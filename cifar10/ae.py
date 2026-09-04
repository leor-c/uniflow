"""
Convolutional autoencoder for CIFAR-10 (32×32 RGB → bounded latent space).

Mirror of :mod:`mnist.ae`. Same architecture family, same latent contract, same
public API; the three things that actually differ from MNIST are called out
below, because everything else in this file is deliberately identical.

Architecture:
    Encoder: Conv(3→32, 1×1) → Conv(32→64, 3×3, s=2) → Conv(64→128, 3×3, s=2)
             → Conv(128→256, 3×3) → head → tanh
    Decoder: head → Upsample(2×) → Conv(256→128, 3×3) → Upsample(2×)
             → Conv(128→64, 3×3) → Conv(64→32, 3×3) → Conv(32→3, 1×1)

Decoder uses nearest-neighbour upsample + conv instead of ConvTranspose2d to
avoid checkerboard artefacts (Odena et al., 2016).

What differs from the MNIST AE, and why
---------------------------------------
1. **The grid is 8×8, not 7×7.** The same two stride-2 convs take 32 → 16 → 8.
   Unlike MNIST's 7, 8 is even, so ``patch=2`` is also a legal DiT tokenization
   here (16 tokens instead of 64) if the 64-token default proves too slow.
2. **Channel widths are doubled** (32/64/128/256 against MNIST's 16/32/64/128).
   CIFAR-10 is 3-channel natural imagery with far more high-frequency content
   than near-binary digits, and the MNIST widths bottleneck reconstruction
   badly enough to dominate the sample grids.
3. **The default latent is 8 planes, not 4** (``latent_dim`` 512 vs 196). A
   CIFAR image is 3072 dims to MNIST's 784, so 4 planes would mean 12×
   compression against MNIST's 4×. At ``patch=1`` the channel count does not
   change the DiT's token count -- only the patch-embedding input width -- so
   this is close to free for the generative model and buys real reconstruction
   quality.

Latent layout
-------------
``latent_channels=None`` gives the original flat latent: the 256×8×8 feature map
is collapsed by a single Linear into a ``latent_dim`` vector.

``latent_channels=C`` instead keeps the 8×8 grid and uses a 1×1 conv head, so
the latent is a ``C×8×8`` feature map -- the layout a DiT or UNet denoiser needs
in order to have any spatial structure to model at all. This is the default for
the CIFAR-10 arms (C=8, i.e. latent_dim=512); see :mod:`core.dit`.

The map is *flattened on the way out* of the encoder, so every consumer of the
latent (noise samplers, ODE integrators, losses, metrics) keeps working with
plain ``(B, latent_dim)`` tensors and only the backbone reshapes back to
``(C, 8, 8)``.

tanh is elementwise either way, so latents are bounded in (-1, 1)^latent_dim and
stay compatible with the Uniform[-1,1]^d prior used by the Uniform RF arm.

Public API
----------
    CIFAR10AutoEncoder            – full AE
    train_cifar10_ae              – training loop (MSE reconstruction loss)
    encode_dataset                – pre-encode entire dataset → latent tensor
    make_cifar10_latent_sampler   – fast random sampler over pre-encoded latents
    make_cifar10_data_loader      – torchvision DataLoader for CIFAR-10
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules import init_silu_


# The encoder downsamples 32 -> 16 -> 8, so a spatial latent lives on an 8x8
# grid. Fixed by the conv stack above, not a free parameter.
LATENT_GRID = 8

# CIFAR-10 is RGB; MNIST's single channel is the only place the pixel-space
# shape leaks into the arms (the loaders and the figure code).
IMAGE_CHANNELS = 3


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class CIFAR10Encoder(nn.Module):
    """
    (B, 3, 32, 32) → latent_dim-d vector in (-1, 1)^d.

    Spatial map (stride-2 conv for downsampling, kernel 3 throughout):
        32×32 → Conv(s=2) → 16×16 → Conv(s=2) → 8×8 → Conv(s=1) → 8×8
        → Flatten → Linear → tanh
    """

    def __init__(
        self,
        latent_dim: int = 512,
        bounded: bool = True,
        latent_channels: int | None = None,
    ):
        super().__init__()
        self.bounded = bounded
        self.latent_channels = latent_channels
        self.net = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(32,  64,  kernel_size=3, stride=2, padding=1),  # 32→16
            nn.SiLU(),
            nn.Conv2d(64,  128, kernel_size=3, stride=2, padding=1),  # 16→8
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 8→8
            nn.SiLU(),
        )
        if latent_channels is None:
            # Flat latent: collapse the whole feature map through one Linear.
            self.head = nn.Linear(256 * LATENT_GRID * LATENT_GRID, latent_dim)
        else:
            # Spatial latent: a 1x1 conv keeps the 8x8 grid, so the generative
            # model downstream sees a feature map rather than an unstructured
            # vector. Flattened on the way out so every consumer of the latent
            # (noise samplers, ODE integrators, losses, metrics) keeps working
            # on plain (B, latent_dim) tensors.
            if latent_dim != latent_channels * LATENT_GRID * LATENT_GRID:
                raise ValueError(
                    f"latent_dim {latent_dim} does not match "
                    f"{latent_channels}x{LATENT_GRID}x{LATENT_GRID}"
                )
            self.head = nn.Conv2d(256, latent_channels, kernel_size=1)
        init_silu_(self, output_layers=(self.head,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw encoder output; the tanh boundary applies only to bounded latents."""
        h = self.net(x)
        if self.latent_channels is None:
            z = self.head(h.flatten(1))
        else:
            z = self.head(h).flatten(1)
        # tanh is elementwise, so a spatial latent is bounded per latent pixel
        # and the uniform arm's base stays i.i.d. U([-1, 1]) over all d dims.
        return torch.tanh(z) if self.bounded else z


class CIFAR10Decoder(nn.Module):
    """
    latent_dim-d vector → (B, 3, 32, 32) pixel image.

    Spatial map (nearest upsample + conv, no ConvTranspose):
        latent → Linear → (256, 8, 8)
        → Upsample(2×) → Conv(3×3) → (128, 16, 16)
        → Upsample(2×) → Conv(3×3) → (64, 32, 32)
        → Conv(3×3) → (32, 32, 32) → Conv(1×1) → (3, 32, 32)
    """

    def __init__(self, latent_dim: int = 512, latent_channels: int | None = None):
        super().__init__()
        self.latent_channels = latent_channels
        if latent_channels is None:
            self.proj = nn.Linear(latent_dim, 256 * LATENT_GRID * LATENT_GRID)
        else:
            # Mirror of the encoder's 1x1 conv head.
            self.proj = nn.Conv2d(latent_channels, 256, kernel_size=1)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),          # 8→16
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),          # 16→32
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(64,  32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, IMAGE_CHANNELS, kernel_size=1, stride=1, padding=0),
            # Output is left unbounded, as in mnist.ae: the reconstruction loss
            # is MSE against pixels in [0, 1] and the figure code clamps.
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.latent_channels is None:
            h = self.proj(z).view(-1, 256, LATENT_GRID, LATENT_GRID)
        else:
            grid = z.view(-1, self.latent_channels, LATENT_GRID, LATENT_GRID)
            h = self.proj(grid)
        return self.net(h)


class CIFAR10AutoEncoder(nn.Module):
    """Full CIFAR-10 AE, mirroring ``mnist.ae.MNISTAutoEncoder``.

    ``bounded=True`` gives tanh latents in ``(-1, 1)^d`` for the uniform arm.
    ``bounded=False`` trains a plain reconstruction autoencoder whose latents are
    then centred and divided by a single global scalar
    (``fit_latent_normalization``), so the transform is a similarity and the
    latent geometry reaches the generative model undistorted -- the Gaussian RF
    and EDM arms both use that form.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        bounded: bool = True,
        latent_channels: int | None = None,
    ):
        super().__init__()
        self.encoder = CIFAR10Encoder(latent_dim, bounded=bounded,
                                      latent_channels=latent_channels)
        self.decoder = CIFAR10Decoder(latent_dim, latent_channels=latent_channels)
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.bounded = bounded
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

    @property
    def latent_shape(self) -> tuple[int, int, int] | None:
        """``(C, H, W)`` for a spatial latent, ``None`` for a flat one."""
        if self.latent_channels is None:
            return None
        return (self.latent_channels, LATENT_GRID, LATENT_GRID)

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

def train_cifar10_ae(
    ae: CIFAR10AutoEncoder,
    loader,
    iters: int,
    lr: float,
    device: str | torch.device,
    latent_spread_weight: float = 1e-3,
    weight_decay: float = 0.01,
    log_every: int = 500,
) -> None:
    """
    Train the CIFAR-10 AE.

    Reconstruction loss: mean squared error on pixels.

    Regularisation: mild per-dim second-moment penalty encouraging the latent
distribution to use the full [-1,1]^d box (mean→0, variance→1/3).
    """
    ae.to(device)
    ae.train()
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=weight_decay)
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
            print(f"[cifar10 ae] step {step:05d} | mse {recon.item():.4f}")

    ae.eval()
    if not ae.bounded:
        with torch.no_grad():
            calibration, _ = next(_infinite(loader))
            ae.fit_latent_normalization(calibration.to(device))
        print(f"[cifar10 ae] latent global scale {ae.latent_std[0].item():.4f}")


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def make_cifar10_data_loader(
    root: str = "./data",
    train: bool = True,
    batch_size: int = 256,
    num_workers: int = 2,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
):
    """Return a torchvision DataLoader for CIFAR-10 (downloads if needed).

    Pixels stay in ``[0, 1]`` rather than the ``[-1, 1]`` many CIFAR diffusion
    codebases use: the flow never sees pixels here, only AE latents, and
    matching :mod:`mnist.ae` keeps the reconstruction MSE scale and the figure
    clamp identical across the two datasets.

    ``generator`` drives the shuffle, so the batch order is a function of that
    stream alone rather than of whatever else has drawn from the global RNG.
    ``shuffle=False`` gives the fixed dataset order, which is what the encoding
    pass and the figure inputs want: the same images in every arm and seed.
    """
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError as e:
        raise ImportError(
            "torchvision is required for CIFAR-10 experiments. "
            "Install with: pip install torchvision"
        ) from e

    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train, download=True,
        transform=T.ToTensor(),
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


@torch.no_grad()
def encode_dataset(
    ae: CIFAR10AutoEncoder,
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


def make_cifar10_latent_sampler(
    all_latents: torch.Tensor,
    device: str | torch.device,
    generator: torch.Generator | None = None,
):
    """Returns callable(batch_size) → [B, latent_dim] on device.

    Give the teacher and the student separate generators so neither phase's
    batches depend on how many the other consumed.
    """
    def sampler(batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, len(all_latents), (batch_size,), generator=generator)
        return all_latents[idx].to(device)
    return sampler


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _infinite(loader):
    while True:
        yield from loader
