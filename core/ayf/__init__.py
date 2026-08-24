"""
AYF (Align Your Flow) with Gaussian noise prior z0 ~ N(0, sigma_data² I).

Same as core.ayf_rf but replaces the Uniform[-1,1] noise prior with
a Gaussian N(0, sigma_data² I), matching the original AYF paper setup.
The teacher is still an RF velocity field (RFTeacher), trained on the
Gaussian-noise RF paths.  The student (RFIntervalFlowMap) and all JVP
objectives are unchanged.

RF time convention: t=0 → noise, t=1 → data.
Path: z_t = (1-t)·z0 + t·z1,  z0 ~ N(0, sigma_data² I).

Public API
----------
Models (from core.ayf_rf):
    RFIntervalFlowMap

Sampling (sampling.py):
    sample_noise                    – z0 ~ N(0, sigma_data² I)
    sample_time_pair_forward        – re-exported from core.ayf_rf
    integrate_teacher_interval      – re-exported from core.ayf_rf
    integrate_teacher_to_data       – re-exported from core.ayf_rf
    sample_teacher_latents
    sample_student_latents_one_step
    sample_student_latents_k_step

Losses (losses.py):
    ayf_emd_loss
    ayf_lmd_loss
    ayf_endpoint_eval_loss   (no_grad)
    ayf_interval_eval_loss   (no_grad)

Training (training.py):
    train_teacher_gaussian
    train_student_ayf
"""

from ..ayf_rf.models import RFIntervalFlowMap
from .sampling import (
    sample_noise,
    sample_time_pair_forward,
    integrate_teacher_interval,
    integrate_teacher_to_data,
    sample_teacher_latents,
    sample_student_latents_one_step,
    sample_student_latents_k_step,
)
from .losses import (
    ayf_emd_loss,
    ayf_lmd_loss,
    ayf_endpoint_eval_loss,
    ayf_interval_eval_loss,
)
from .training import train_teacher_gaussian, train_student_ayf

__all__ = [
    "RFIntervalFlowMap",
    "sample_noise",
    "sample_time_pair_forward",
    "integrate_teacher_interval",
    "integrate_teacher_to_data",
    "sample_teacher_latents",
    "sample_student_latents_one_step",
    "sample_student_latents_k_step",
    "ayf_emd_loss",
    "ayf_lmd_loss",
    "ayf_endpoint_eval_loss",
    "ayf_interval_eval_loss",
    "train_teacher_gaussian",
    "train_student_ayf",
]