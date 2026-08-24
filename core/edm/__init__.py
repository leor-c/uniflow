"""
EDM (Elucidating the Design Space of Diffusion-Based Generative Models) baseline.

The release toy arm uses an unbounded latent space matched to Gaussian RF.
The EDM implementation itself provides:
  - Noise prior: Gaussian N(0, sigma^2 I) instead of Uniform[-1,1]^d
  - Teacher noise coordinate: log(sigma) (EDM convention)
  - AYF time coordinate: normalized [0, 1], sigma_max → sigma_min
  - Denoiser parameterization: EDM preconditioning (Karras et al. 2022)
  - AYF distillation: arbitrary intervals along the EDM probability-flow ODE

Convention (matching Karras et al.):
  Large sigma = more noise, small sigma = cleaner.
  ODE direction: sigma decreasing (noisy → clean).

Public API
----------
Models (models.py):
    EDMPrecondDenoiser  – preconditioned denoiser D(z_noisy, log_sigma) -> z_clean
    EDMIntervalFlowMap  – AYF map between arbitrary normalized EDM times
    EDMConsistencyModel – endpoint map with exact boundary at sigma_min

Utilities (utils.py):
    edm_sample_log_sigma    – log-normal sigma sampling for training
    edm_karras_sigmas       – Karras decreasing sigma schedule for sampling
    edm_time_to_log_sigma   – normalized EDM time to log-sigma
    edm_log_sigma_to_time   – log-sigma to normalized EDM time
    edm_dlog_sigma_dt       – constant schedule derivative

Losses (losses.py):
    edm_teacher_loss
    edm_ayf_emd_loss
    edm_ayf_lmd_loss
    edm_ayf_interval_eval_loss  (no_grad)
    edm_endpoint_distillation_loss
    edm_jvp_scm_loss
    edm_endpoint_eval_loss  (no_grad)

Training (training.py):
    train_edm_teacher
    train_edm_student_ayf
    train_edm_student_jvp
    train_edm_student_hybrid

Sampling (sampling.py; return latents):
    integrate_edm_teacher       – ODE from log_sigma_start to log(sigma_min)
    integrate_edm_teacher_interval – ODE between normalized EDM times
    sample_edm_teacher_latents  – full Karras ODE
    sample_edm_ayf_latents_one_step
    sample_edm_ayf_latents_k_step
    sample_edm_student_latents_one_step
    sample_edm_student_latents_k_step
"""

from .models import EDMConsistencyModel, EDMIntervalFlowMap, EDMPrecondDenoiser
from .utils import (
    edm_dlog_sigma_dt,
    edm_karras_sigmas,
    edm_log_sigma_to_time,
    edm_sample_log_sigma,
    edm_time_to_log_sigma,
)
from .sampling import (
    edm_normalized_time_velocity,
    integrate_edm_teacher,
    integrate_edm_teacher_interval,
    sample_edm_ayf_latents_k_step,
    sample_edm_ayf_latents_one_step,
    sample_edm_teacher_latents,
    sample_edm_student_latents_one_step,
    sample_edm_student_latents_k_step,
)
from .losses import (
    edm_ayf_emd_loss,
    edm_ayf_interval_eval_loss,
    edm_ayf_lmd_loss,
    edm_teacher_loss,
    edm_endpoint_distillation_loss,
    edm_jvp_scm_loss,
    edm_endpoint_eval_loss,
)
from .training import (
    train_edm_student_ayf,
    train_edm_student_hybrid,
    train_edm_student_jvp,
    train_edm_teacher,
)

__all__ = [
    "EDMPrecondDenoiser",
    "EDMIntervalFlowMap",
    "EDMConsistencyModel",
    "edm_sample_log_sigma",
    "edm_karras_sigmas",
    "edm_time_to_log_sigma",
    "edm_log_sigma_to_time",
    "edm_dlog_sigma_dt",
    "edm_normalized_time_velocity",
    "integrate_edm_teacher",
    "integrate_edm_teacher_interval",
    "sample_edm_ayf_latents_one_step",
    "sample_edm_ayf_latents_k_step",
    "sample_edm_teacher_latents",
    "sample_edm_student_latents_one_step",
    "sample_edm_student_latents_k_step",
    "edm_teacher_loss",
    "edm_ayf_emd_loss",
    "edm_ayf_lmd_loss",
    "edm_ayf_interval_eval_loss",
    "edm_endpoint_distillation_loss",
    "edm_jvp_scm_loss",
    "edm_endpoint_eval_loss",
    "train_edm_teacher",
    "train_edm_student_ayf",
    "train_edm_student_jvp",
    "train_edm_student_hybrid",
]
