# latent_rf_consistency_8gaussians.py

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -------------------------
# Data
# -------------------------

def sample_8gaussians(batch_size, device, radius=0.75, std=0.035):
    k = torch.randint(0, 8, (batch_size,), device=device)
    angles = 2 * math.pi * k.float() / 8.0
    centers = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)],
        dim=-1,
    )
    return centers + std * torch.randn(batch_size, 2, device=device)


def sample_uniform(batch_size, dim, device):
    return torch.empty(batch_size, dim, device=device).uniform_(-1.0, 1.0)


# -------------------------
# Basic modules
# -------------------------

class FourierTimeEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        x = 2 * math.pi * t * self.freqs[None, :]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, depth=3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Autoencoder
# -------------------------

class AutoEncoder(nn.Module):
    """
    2D -> 16D bounded latent -> 2D.

    Encoder output is bounded with tanh, so z is in [-1, 1]^latent_dim.
    """
    def __init__(self, latent_dim=16, hidden=128):
        super().__init__()
        self.encoder_net = MLP(2, latent_dim, hidden=hidden, depth=3)
        self.decoder_net = MLP(latent_dim, 2, hidden=hidden, depth=3)

    def encode(self, x):
        return torch.tanh(self.encoder_net(x))

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def latent_moment_regularizer(z):
    """
    Optional weak regularizer to avoid all latents collapsing near zero.

    Uniform[-1,1] has mean 0 and variance 1/3.
    This does not make z uniform, but it encourages using the latent range.
    """
    mean_loss = z.mean(dim=0).pow(2).mean()
    second_moment_loss = (z.pow(2).mean(dim=0) - 1.0 / 3.0).pow(2).mean()
    return mean_loss + second_moment_loss


def train_autoencoder(ae, cfg):
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.ae_lr)

    for step in range(1, cfg.ae_steps + 1):
        x = sample_8gaussians(cfg.batch_size, cfg.device)
        x_hat, z = ae(x)

        recon_loss = F.mse_loss(x_hat, x)
        reg_loss = latent_moment_regularizer(z)

        loss = recon_loss + cfg.latent_reg_weight * reg_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0:
            print(
                f"[ae] step {step:05d} | "
                f"recon {recon_loss.item():.6f} | "
                f"reg {reg_loss.item():.6f}"
            )

    ae.eval()


# -------------------------
# RF teacher and consistency student
# -------------------------

class RFTeacher(nn.Module):
    """
    Latent-space velocity field v_phi(z,t).
    """
    def __init__(self, latent_dim=16, hidden=128):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(16)
        self.mlp = MLP(latent_dim + 32, latent_dim, hidden=hidden, depth=3)

    def forward(self, z, t):
        if t.ndim == 1:
            t = t[:, None]
        emb = self.time_embed(t)
        return self.mlp(torch.cat([z, emb], dim=-1))


class RFConsistencyStudent(nn.Module):
    """
    Latent-space consistency model.

    C_theta(z,t) = t z + (1 - t) tanh(h_theta(z,t)).

    This enforces:
        C_theta(z,1) = z

    and also keeps C_theta(z,t) in [-1,1]^latent_dim when z is in [-1,1]^latent_dim.
    """
    def __init__(self, latent_dim=16, hidden=128):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(16)
        self.mlp = MLP(latent_dim + 32, latent_dim, hidden=hidden, depth=3)

    def forward(self, z, t):
        if t.ndim == 1:
            t = t[:, None]
        emb = self.time_embed(t)
        h = self.mlp(torch.cat([z, emb], dim=-1))
        pred_data_latent = torch.tanh(h)
        return t * z + (1.0 - t) * pred_data_latent


@torch.no_grad()
def sample_data_latents(ae, batch_size, cfg):
    x = sample_8gaussians(batch_size, cfg.device)
    z1 = ae.encode(x)
    return z1


def rf_teacher_loss(ae, teacher, cfg):
    """
    Train RF teacher in latent space.

    z1 = encoder(x)
    u ~ Uniform([-1,1]^latent_dim)
    zt = (1 - t)u + t z1
    target velocity = z1 - u
    """
    with torch.no_grad():
        z1 = sample_data_latents(ae, cfg.batch_size, cfg)

    u = sample_uniform(cfg.batch_size, cfg.latent_dim, cfg.device)
    t = torch.rand(cfg.batch_size, 1, device=cfg.device)

    zt = (1.0 - t) * u + t * z1
    target_v = z1 - u

    pred_v = teacher(zt, t)
    return F.mse_loss(pred_v, target_v)


def train_teacher(ae, teacher, cfg):
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.teacher_lr)

    for step in range(1, cfg.teacher_steps + 1):
        loss = rf_teacher_loss(ae, teacher, cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0:
            print(f"[teacher] step {step:05d} | loss {loss.item():.6f}")

    teacher.eval()


# -------------------------
# Teacher ODE integration
# -------------------------

@torch.no_grad()
def integrate_teacher_to_data(teacher, z, t_start, steps=32):
    """
    Integrate dz/dt = v_phi(z,t) from each sample's t_start to t=1.

    z:       [B, latent_dim]
    t_start: [B, 1]
    """
    z = z.clone()
    t_start = t_start.clone()

    for i in range(steps):
        tau = t_start + (1.0 - t_start) * (i / steps)
        dt = (1.0 - t_start) / steps
        z = z + dt * teacher(z, tau)

    return z.clamp(-1.0, 1.0)


def jvp_scm_loss(ae, student, teacher, cfg):
    """
    Continuous-time RF-sCM/JVP loss in latent space:

        || d/dt C_theta(z(t), t) ||^2

    where dz/dt = v_phi(z,t).
    """
    with torch.no_grad():
        z1 = sample_data_latents(ae, cfg.batch_size, cfg)

    u = sample_uniform(cfg.batch_size, cfg.latent_dim, cfg.device)

    # Usually better to avoid exact endpoints.
    # You can try Beta(2,1) later, but uniform is cleaner for debugging.
    t = torch.rand(cfg.batch_size, 1, device=cfg.device)
    t = t.clamp(1e-4, 1.0 - 1e-4)

    zt = (1.0 - t) * u + t * z1

    with torch.no_grad():
        v = teacher(zt, t)

    def C_fn(z_in, t_in):
        return student(z_in, t_in)

    _, tangent = jvp(
        C_fn,
        primals=(zt, t),
        tangents=(v, torch.ones_like(t)),
    )

    # Mild tangent normalization.
    # This prevents large velocities from dominating the loss.
    norm = torch.sqrt(1.0 + v.pow(2).mean(dim=-1, keepdim=True)).detach()

    return ((tangent / norm) ** 2).mean()


def endpoint_distillation_loss(ae, student, teacher, cfg, ode_steps=32):
    """
    Strong consistency distillation target:

        C_theta(z_t,t) ≈ ODESolve_teacher(z_t, t -> 1)

    All in latent space.
    """
    with torch.no_grad():
        z1 = sample_data_latents(ae, cfg.batch_size, cfg)

    u = sample_uniform(cfg.batch_size, cfg.latent_dim, cfg.device)

    t = torch.rand(cfg.batch_size, 1, device=cfg.device).clamp(0.0, 0.999)
    zt = (1.0 - t) * u + t * z1

    with torch.no_grad():
        target = integrate_teacher_to_data(
            teacher=teacher,
            z=zt,
            t_start=t,
            steps=ode_steps,
        )

    pred = student(zt, t)
    return F.mse_loss(pred, target)


def train_student_hybrid(ae, student, teacher, cfg):
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.student_lr)

    for step in range(1, cfg.student_steps + 1):
        loss_jvp = jvp_scm_loss(
            ae=ae,
            student=student,
            teacher=teacher,
            cfg=cfg,
        )

        if step % cfg.endpoint_every == 0:
            loss_endpoint = endpoint_distillation_loss(
                ae=ae,
                student=student,
                teacher=teacher,
                cfg=cfg,
                ode_steps=cfg.endpoint_ode_steps,
            )

            loss = loss_endpoint + cfg.jvp_weight_on_endpoint_steps * loss_jvp

            loss_endpoint_value = loss_endpoint.item()
            loss_jvp_value = loss_jvp.item()
            mode = "endpoint+jvp"
        else:
            loss = cfg.jvp_weight * loss_jvp

            loss_endpoint_value = float("nan")
            loss_jvp_value = loss_jvp.item()
            mode = "jvp"

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0:
            print(
                f"[student {mode}] step {step:05d} | "
                f"loss {loss.item():.6f} | "
                f"endpoint {loss_endpoint_value:.6f} | "
                f"jvp {loss_jvp_value:.6f}"
            )

    student.eval()


# -------------------------
# Sampling
# -------------------------

@torch.no_grad()
def sample_teacher_decoded(ae, teacher, n, cfg, steps=64):
    u = sample_uniform(n, cfg.latent_dim, cfg.device)
    t0 = torch.zeros(n, 1, device=cfg.device)
    z = integrate_teacher_to_data(teacher, u, t0, steps=steps)
    return ae.decode(z)


@torch.no_grad()
def sample_student_one_step_decoded(ae, student, n, cfg):
    u = sample_uniform(n, cfg.latent_dim, cfg.device)
    t0 = torch.zeros(n, 1, device=cfg.device)
    z = student(u, t0)
    return ae.decode(z)


@torch.no_grad()
def sample_student_two_step_decoded(ae, student, n, cfg, tau=0.5):
    u = sample_uniform(n, cfg.latent_dim, cfg.device)

    t0 = torch.zeros(n, 1, device=cfg.device)
    z1_hat = student(u, t0)

    tt = torch.full((n, 1), tau, device=cfg.device)
    z_tau = (1.0 - tt) * u + tt * z1_hat

    z1_hat_2 = student(z_tau, tt)
    return ae.decode(z1_hat_2)


@torch.no_grad()
def sample_ae_recon(ae, n, cfg):
    x = sample_8gaussians(n, cfg.device)
    x_hat, _ = ae(x)
    return x, x_hat


# -------------------------
# Plotting
# -------------------------

@torch.no_grad()
def make_plot(ae, teacher, student, cfg, path="latent_rf_consistency_8gaussians.png"):
    n = 5000

    data = sample_8gaussians(n, cfg.device).cpu()

    x_recon_src, x_recon = sample_ae_recon(ae, n, cfg)
    x_recon = x_recon.cpu()

    teacher_samples = sample_teacher_decoded(
        ae, teacher, n, cfg, steps=cfg.teacher_sample_steps
    ).cpu()

    student_1 = sample_student_one_step_decoded(ae, student, n, cfg).cpu()

    student_2 = sample_student_two_step_decoded(
        ae, student, n, cfg, tau=0.5
    ).cpu()

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharex=True, sharey=True)

    plots = [
        ("Data", data),
        ("AE reconstruction", x_recon),
        ("Decoded RF teacher", teacher_samples),
        ("Decoded student 1 step", student_1),
        ("Decoded student 2 step", student_2),
    ]

    for ax, (title, pts) in zip(axes, plots):
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.45)
        ax.set_title(title)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    print(f"Saved plot to {path}")


# -------------------------
# Main config
# -------------------------

# @dataclass
# class Config:
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#
#     latent_dim: int = 16
#     hidden: int = 256
#     batch_size: int = 64
#
#     ae_steps: int = 3000
#     teacher_steps: int = 32000
#     student_steps: int = 5000
#
#     ae_lr: float = 2e-3
#     teacher_lr: float = 1e-3
#     student_lr: float = 1e-3
#
#     latent_reg_weight: float = 1e-3
#
#     endpoint_ode_steps: int = 64
#     teacher_sample_steps: int = 128
#
#     log_every: int = 500


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim: int = 16
    hidden: int = 128
    batch_size: int = 64

    ae_steps: int = 3000
    teacher_steps: int = 6000
    student_steps: int = 8000

    ae_lr: float = 2e-3
    teacher_lr: float = 2e-3

    # JVP training is more fragile, so use a slightly smaller LR.
    student_lr: float = 1e-3

    latent_reg_weight: float = 1e-3

    endpoint_ode_steps: int = 32
    teacher_sample_steps: int = 64

    # Hybrid schedule.
    endpoint_every: int = 8

    # On ordinary steps, optimize mostly/purely JVP.
    jvp_weight: float = 1.0

    # On anchor steps, endpoint should dominate.
    jvp_weight_on_endpoint_steps: float = 0.1

    log_every: int = 500


def main():
    cfg = Config()
    print(cfg)

    ae = AutoEncoder(latent_dim=cfg.latent_dim, hidden=cfg.hidden).to(cfg.device)
    teacher = RFTeacher(latent_dim=cfg.latent_dim, hidden=cfg.hidden).to(cfg.device)
    student = RFConsistencyStudent(latent_dim=cfg.latent_dim, hidden=cfg.hidden).to(cfg.device)

    print("\nTraining autoencoder...")
    train_autoencoder(ae, cfg)

    print("\nTraining latent RF teacher...")
    train_teacher(ae, teacher, cfg)

    print("\nTraining latent consistency student...")
    train_student_hybrid(ae, student, teacher, cfg)

    print("\nPlotting...")
    make_plot(ae, teacher, student, cfg)


if __name__ == "__main__":
    main()