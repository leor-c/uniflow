"""
AYF (Align Your Flow) style bounded-latent interval flow maps for RF-sCM.

Generalizes the endpoint consistency model C(z, t) -> z_1 (core.rf_scm) to
an arbitrary-interval flow map C(z, t, s) -> z_s, where t is the source time
and s is the target time. The endpoint model is the special case s = 1. See
ayf_bounded_latent_rf_summary.md for the full derivation.

This package builds directly on core.rf_scm: the RF teacher (RFTeacher),
noise prior (sample_noise), and teacher training loop (train_teacher) are
reused unchanged — only the interval-aware student, its losses, sampling,
and training loop are new.

Convention (t=0 → noise, t=1 → data), same as core.rf_scm:
    z0 ~ Uniform([-1,1]^d)
    z1 = encoder(x)
    z_t = (1 - t) * z0 + t * z1
    C_theta(z, t, s) = z + (s - t) * f_theta(z, t, s)   (identity boundary C(z,t,t) = z)

Public API
----------
Models (models.py):
    RFIntervalFlowMap  – interval consistency map C(z, t, s) -> z_s

Losses (losses.py; differentiable, return scalar unless noted):
    ayf_emd_loss                     – teacher-based source-time JVP (EMD)
    ayf_emd_ct_loss                  – teacher-free EMD analog, v = z1 - z0
    ayf_lmd_loss                     – teacher-based target-time JVP (LMD)
    ayf_pairwise_anchor_loss         – cheap chord-interval supervised anchor
    ayf_teacher_interval_anchor_loss – teacher ODE interval supervised anchor
    ayf_endpoint_eval_loss  (no_grad)
    ayf_interval_eval_loss  (no_grad)

Training (training.py):
    train_student_interval

Sampling (sampling.py; return latent tensors, no decoding):
    sample_time_pair_forward    – source/target time pair 0 <= t < s <= 1
    integrate_teacher_interval  – teacher ODE from t_start to arbitrary t_end
    integrate_teacher_to_data   – teacher ODE from t_start to t=1
    sample_student_latents_one_step
    sample_student_latents_k_step  – true interval stepping z_{t_i} -> z_{t_{i+1}}
"""

from .models import RFIntervalFlowMap
from .sampling import (
    sample_time_pair_forward,
    integrate_teacher_interval,
    integrate_teacher_to_data,
    sample_student_latents_one_step,
    sample_student_latents_k_step,
)
from .losses import (
    ayf_emd_loss,
    ayf_emd_ct_loss,
    ayf_lmd_loss,
    ayf_pairwise_anchor_loss,
    ayf_teacher_interval_anchor_loss,
    ayf_endpoint_eval_loss,
    ayf_interval_eval_loss,
)
from .training import train_student_interval

__all__ = [
    "RFIntervalFlowMap",
    "sample_time_pair_forward",
    "integrate_teacher_interval",
    "integrate_teacher_to_data",
    "sample_student_latents_one_step",
    "sample_student_latents_k_step",
    "ayf_emd_loss",
    "ayf_emd_ct_loss",
    "ayf_lmd_loss",
    "ayf_pairwise_anchor_loss",
    "ayf_teacher_interval_anchor_loss",
    "ayf_endpoint_eval_loss",
    "ayf_interval_eval_loss",
    "train_student_interval",
]
