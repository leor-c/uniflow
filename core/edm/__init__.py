"""
EDM (Elucidating the Design Space of Diffusion-Based Generative Models) baseline.

Uses the same bounded latent space (tanh AE) as RF-sCM, but:
  - Noise prior: Gaussian N(0, sigma^2 I) instead of Uniform[-1,1]^d
  - Time coordinate: log(sigma) (EDM convention)
  - Denoiser parameterization: EDM preconditioning (Karras et al. 2022)
  - Consistency loss: JVP in log-sigma coordinates

Convention (matching Karras et al.):
  Large sigma = more noise, small sigma = cleaner.
  ODE direction: sigma decreasing (noisy → clean).

Public API
----------
Models (models.py):
    EDMPrecondDenoiser  – preconditioned denoiser D(z_noisy, log_sigma) -> z_clean

Utilities (utils.py):
    edm_sample_log_sigma    – log-normal sigma sampling for training
    edm_karras_sigmas       – Karras decreasing sigma schedule for sampling

Losses (losses.py):
    edm_teacher_loss
    edm_endpoint_distillation_loss
    edm_jvp_scm_loss
    edm_endpoint_eval_loss  (no_grad)

Training (training.py):
    train_edm_teacher
    train_edm_student_jvp
    train_edm_student_hybrid

Sampling (sampling.py; return latents):
    integrate_edm_teacher       – ODE from log_sigma_start to log(sigma_min)
    sample_edm_teacher_latents  – full Karras ODE
    sample_edm_student_latents_one_step
    sample_edm_student_latents_k_step
"""

from .models import EDMPrecondDenoiser
from .utils import edm_sample_log_sigma, edm_karras_sigmas
from .sampling import (
    integrate_edm_teacher,
    sample_edm_teacher_latents,
    sample_edm_student_latents_one_step,
    sample_edm_student_latents_k_step,
)
from .losses import (
    edm_teacher_loss,
    edm_endpoint_distillation_loss,
    edm_jvp_scm_loss,
    edm_endpoint_eval_loss,
)
from .training import train_edm_teacher, train_edm_student_jvp, train_edm_student_hybrid

__all__ = [
    "EDMPrecondDenoiser",
    "edm_sample_log_sigma",
    "edm_karras_sigmas",
    "integrate_edm_teacher",
    "sample_edm_teacher_latents",
    "sample_edm_student_latents_one_step",
    "sample_edm_student_latents_k_step",
    "edm_teacher_loss",
    "edm_endpoint_distillation_loss",
    "edm_jvp_scm_loss",
    "edm_endpoint_eval_loss",
    "train_edm_teacher",
    "train_edm_student_jvp",
    "train_edm_student_hybrid",
]
