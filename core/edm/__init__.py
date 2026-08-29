"""
EDM (Elucidating the Design Space of Diffusion-Based Generative Models) baseline.

The release toy arm uses an unbounded latent space matched to Gaussian RF.
The EDM implementation itself provides:
  - Teacher noise coordinate: log(sigma) (EDM convention)
  - Denoiser parameterization: EDM preconditioning (Karras et al. 2022)
  - Untruncated log-normal training noise, Karras rho-schedule sampling
  - A flow-matching view of the trained denoiser for AYF distillation

Convention (matching Karras et al.):
  Large sigma = more noise, small sigma = cleaner.
  ODE direction: sigma decreasing (noisy -> clean).

AYF distillation
----------------
Align Your Flow works in the linear flow-matching framework rather than in
sigma space, so this package does not carry its own interval-flow student.
``EDMFlowVelocity`` exposes a trained denoiser as a flow-matching velocity
field ``v(z, t)`` with ``t = 0`` noise and ``t = 1`` data; the shared student,
objective, interval schedule and samplers in ``core.ayf`` then apply unchanged.

Public API
----------
Models (models.py):
    EDMPrecondDenoiser  - preconditioned denoiser D(z_noisy, log_sigma) -> z_clean
    EDMConsistencyModel - endpoint map with exact boundary at sigma_min

Flow-matching view (flow.py):
    EDMFlowVelocity     - trained denoiser as a flow-matching velocity field

Utilities (utils.py):
    edm_sample_log_sigma    - log-normal sigma sampling for training
    edm_karras_sigmas       - Karras decreasing sigma schedule for sampling

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
    integrate_edm_teacher       - ODE from log_sigma_start to log(sigma_min)
    sample_edm_teacher_latents  - full Karras ODE
    sample_edm_student_latents_one_step
    sample_edm_student_latents_k_step
"""

from .models import EDMConsistencyModel, EDMPrecondDenoiser
from .flow import EDMFlowVelocity
from .utils import edm_karras_sigmas, edm_sample_log_sigma
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
from .training import (
    train_edm_student_hybrid,
    train_edm_student_jvp,
    train_edm_teacher,
)

__all__ = [
    "EDMPrecondDenoiser",
    "EDMConsistencyModel",
    "EDMFlowVelocity",
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
