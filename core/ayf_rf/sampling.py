"""
AYF-style sampling (return latent tensors, no decoding):
    sample_ayf_time_pair        – AYF Algorithm 1 interval schedule (shared)
    integrate_teacher_interval  – teacher ODE from t_start to an arbitrary t_end
    integrate_teacher_to_data   – teacher ODE from t_start to t=1
    sample_student_latents_one_step
    sample_student_latents_k_step  – true interval stepping z_{t_i} -> z_{t_{i+1}}

Teacher ODE integration is shared with ``core.rf``.
"""

from __future__ import annotations

import torch

from ..rf import RFTeacher, sample_uniform_noise
from .models import RFIntervalFlowMap


def sample_ayf_time_pair(
    batch_size: int,
    device: str | torch.device,
    eps: float = 1e-4,
    p_mean: float = -0.6,
    p_std: float = 1.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    AYF Algorithm 1 interval schedule, shared by every arm in this repo.

    Draw a logistic-normal interval length, then place an interval of that
    length uniformly along the trajectory:

        d = sigmoid(N(p_mean, p_std^2)),  t ~ U(0, 1 - d),  s = t + d

    Returns ``0 <= t < s <= 1`` as ``[B, 1]`` tensors. This is the single
    definition used by ``core.ayf``, ``core.ayf_rf`` and ``core.edm`` so that
    the three arms are trained on the same distribution of intervals.
    """
    logits = p_mean + p_std * torch.randn(batch_size, 1, device=device)
    distance = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    t = (1.0 - distance) * torch.rand(batch_size, 1, device=device)
    return t, t + distance


@torch.no_grad()
def integrate_teacher_interval(
    teacher: RFTeacher,
    z: torch.Tensor,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    steps: int = 32,
) -> torch.Tensor:
    """Euler-integrate dz/dt = v_phi(z, t) from t_start to t_end (per-sample)."""
    z = z.clone()
    t_start = t_start.clone()
    t_end = t_end.clone()
    for i in range(steps):
        tau = t_start + (t_end - t_start) * (i / steps)
        dt = (t_end - t_start) / steps
        z = z + dt * teacher(z, tau)
    return z


@torch.no_grad()
def integrate_teacher_to_data(
    teacher: RFTeacher,
    z: torch.Tensor,
    t_start: torch.Tensor,
    steps: int = 64,
) -> torch.Tensor:
    """Euler-integrate dz/dt = v_phi(z, t) from t_start to t=1 (data)."""
    t_end = torch.ones_like(t_start)
    return integrate_teacher_interval(teacher, z, t_start, t_end, steps=steps)


@torch.no_grad()
def sample_student_latents_one_step(
    student: RFIntervalFlowMap,
    n: int,
    latent_dim: int,
    device: str | torch.device,
) -> torch.Tensor:
    """One-step generation: C(z0, 0, 1)."""
    z0 = sample_uniform_noise(n, latent_dim, device)
    t0 = torch.zeros(n, 1, device=device)
    s1 = torch.ones(n, 1, device=device)
    return student(z0, t0, s1)


@torch.no_grad()
def sample_student_latents_k_step(
    student: RFIntervalFlowMap,
    n: int,
    latent_dim: int,
    device: str | torch.device,
    k: int = 4,
    schedule: str = "uniform",
    clamp: bool = False,
) -> torch.Tensor:
    """
    AYF-style k-step sampling: true intermediate stepping
        z_{t_{i+1}} = C_theta(z_{t_i}, t_i, t_{i+1}),
    unlike endpoint-consistency sampling, which repeatedly predicts z_1 and
    re-interpolates with the original noise.

    schedule: "uniform" | "quadratic" | "sqrt" | explicit list of k+1 times in [0, 1].
    """
    assert k >= 1
    z = sample_uniform_noise(n, latent_dim, device)

    if isinstance(schedule, str):
        base = torch.linspace(0.0, 1.0, k + 1, device=device)
        if schedule == "uniform":
            times = base
        elif schedule == "quadratic":
            times = base ** 2
        elif schedule == "sqrt":
            times = base.sqrt()
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")
    else:
        times = torch.as_tensor(schedule, device=device, dtype=torch.float32)
        assert len(times) == k + 1

    for i in range(k):
        t = torch.full((n, 1), float(times[i]), device=device)
        s = torch.full((n, 1), float(times[i + 1]), device=device)
        z = student(z, t, s)
        if clamp:
            z = z.clamp(-1.0, 1.0)

    return z
