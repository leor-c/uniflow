"""
Rectified-Flow Self-Consistency Matching (RF-sCM) — generalized core.

Convention (t=0 → noise, t=1 → data):
    z0 ~ Uniform([-1,1]^d)
    z1 = encoder(x)        (data latent, bounded by tanh)
    z_t = (1 - t) * z0 + t * z1    t in [0, 1]
    teacher velocity target: v = z1 - z0

Public API
----------
Models (models.py):
    RFTeacher           – velocity field v(z, t)
    RFConsistencyStudent – consistency map C(z, t) with boundary C(z, 1) = z

Losses (losses.py; differentiable, return scalar):
    rf_teacher_loss
    jvp_scm_loss
    jvp_ct_loss          – teacher-free Consistency-Training analog of jvp_scm_loss
    endpoint_distillation_eval_loss  (no_grad, monitoring only)

Training loops (training.py):
    train_teacher
    train_student_jvp
    train_student_ct     – teacher-free (CT-style) student training

Sampling (sampling.py; return latent tensors, no decoding):
    sample_noise            – uniform noise in [-1, 1]^d
    integrate_teacher       – ODE from t_start to 1
    sample_teacher_latents  – full generation from z0
    sample_student_latents_one_step
    sample_student_latents_k_step
"""

from .models import RFTeacher, RFConsistencyStudent
from .sampling import (
    sample_noise,
    integrate_teacher,
    sample_teacher_latents,
    sample_student_latents_one_step,
    sample_student_latents_k_step,
)
from .losses import (
    rf_teacher_loss,
    jvp_scm_loss,
    jvp_ct_loss,
    endpoint_distillation_eval_loss,
)
from .training import train_teacher, train_student_jvp, train_student_ct

__all__ = [
    "RFTeacher",
    "RFConsistencyStudent",
    "sample_noise",
    "integrate_teacher",
    "sample_teacher_latents",
    "sample_student_latents_one_step",
    "sample_student_latents_k_step",
    "rf_teacher_loss",
    "jvp_scm_loss",
    "jvp_ct_loss",
    "endpoint_distillation_eval_loss",
    "train_teacher",
    "train_student_jvp",
    "train_student_ct",
]
