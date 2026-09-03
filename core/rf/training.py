"""Rectified-flow teacher training loop."""

from __future__ import annotations

from collections.abc import Callable

import torch

import torch.nn.functional as F

from .losses import rf_teacher_loss
from .models import RFTeacher
from .sampling import NoiseSampler, sample_uniform_noise, integrate_teacher_to_data


def train_teacher(
    teacher: RFTeacher,
    latent_sampler: Callable[[int], torch.Tensor],
    cfg,
    noise_sampler: NoiseSampler = sample_uniform_noise,
    label: str = "RF teacher",
) -> None:
    """Train an RF teacher using the base distribution supplied by the caller."""
    teacher.train()
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.teacher_lr)
    for step in range(1, cfg.teacher_steps + 1):
        loss = rf_teacher_loss(
            teacher,
            latent_sampler,
            cfg.batch_size,
            cfg.latent_dim,
            cfg.device,
            noise_sampler=noise_sampler,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            print(f"[{label}] step {step:05d} | loss {loss.item():.6f}")
    teacher.eval()
    train_teacher_reflow(teacher, cfg, noise_sampler=noise_sampler, label=label)


# ---------------------------------------------------------------------------
# ReFlow (Liu et al. 2022, "Flow Straight and Fast")
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_reflow_pairs(
    teacher: RFTeacher,
    num_pairs: int,
    latent_dim: int,
    device: str | torch.device,
    noise_sampler: NoiseSampler,
    ode_steps: int = 128,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the deterministic coupling ``(z0, ODE(z0))`` from the current teacher.

    1-RF trains on *independent* pairs ``(z0, z1)``, so the marginal paths cross
    and the learned velocity is an average over crossings -- curved, and badly
    approximated by one Euler step. Integrating the ODE gives a coupling that by
    construction does not cross, and refitting on it yields a straighter field.

    Generated under ``no_grad`` with the teacher frozen: these pairs are data for
    the next round, not something to backpropagate through.
    """
    was_training = teacher.training
    teacher.eval()
    sources, targets = [], []
    for start in range(0, num_pairs, batch_size):
        size = min(batch_size, num_pairs - start)
        z0 = noise_sampler(size, latent_dim, device)
        t0 = torch.zeros(size, 1, device=device)
        sources.append(z0)
        targets.append(integrate_teacher_to_data(teacher, z0, t0, steps=ode_steps))
    teacher.train(was_training)
    return torch.cat(sources), torch.cat(targets)


def train_teacher_reflow(
    teacher: RFTeacher,
    cfg,
    noise_sampler: NoiseSampler = sample_uniform_noise,
    label: str = "RF teacher",
) -> None:
    """Refit ``teacher`` on its own ODE coupling, ``cfg.reflow_rounds`` times.

    ``reflow_rounds = 0`` leaves the teacher untouched (plain 1-RF, the default);
    1 gives 2-RF, 2 gives 3-RF, and so on. Weights carry over between rounds
    rather than being reinitialized -- the field only needs straightening, not
    relearning, which is what makes ReFlow cheap relative to the first fit.

    Caveat worth stating: each round fits the pushforward of the base under the
    *previous* ODE, not the data. That equals the data distribution only insofar
    as the previous round was accurate, so fidelity errors accumulate across
    rounds while straightness improves. ``reflow_ode_steps`` controls how much
    of that error each round introduces.
    """
    rounds = int(getattr(cfg, "reflow_rounds", 0))
    if rounds <= 0:
        return

    steps = int(getattr(cfg, "reflow_steps", cfg.teacher_steps))
    num_pairs = int(getattr(cfg, "reflow_pairs", 100_000))
    ode_steps = int(getattr(cfg, "reflow_ode_steps", 128))

    for round_index in range(1, rounds + 1):
        tag = f"{label} reflow {round_index}/{rounds}"
        print(f"\n[{tag}] generating {num_pairs} ODE pairs ({ode_steps} steps)...")
        z0_all, z1_all = generate_reflow_pairs(
            teacher, num_pairs, cfg.latent_dim, cfg.device,
            noise_sampler, ode_steps=ode_steps,
        )
        # Straightness of the coupling: 1.0 means the ODE already moved every
        # point in a straight line, so a further round would buy nothing.
        displacement = (z1_all - z0_all).norm(dim=-1).mean().item()
        print(f"[{tag}] mean |z1 - z0| = {displacement:.4f}")

        teacher.train()
        opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.teacher_lr)
        for step in range(1, steps + 1):
            index = torch.randint(0, z0_all.shape[0], (cfg.batch_size,), device=cfg.device)
            z0, z1 = z0_all[index], z1_all[index]
            t = torch.rand(cfg.batch_size, 1, device=cfg.device)
            zt = (1.0 - t) * z0 + t * z1
            loss = F.mse_loss(teacher(zt, t), z1 - z0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % cfg.log_every == 0:
                print(f"[{tag}] step {step:05d} | loss {loss.item():.6f}")

        del z0_all, z1_all
    teacher.eval()
