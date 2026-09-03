"""Shared rectified-flow teacher components.

The experiment-specific choice of base distribution lives in the caller:
Gaussian noise for the standard RF arm and uniform noise for the bounded arm.
"""

from .models import RFTeacher
from .sampling import (
    integrate_teacher,
    integrate_teacher_interval,
    integrate_teacher_to_data,
    sample_gaussian_noise,
    sample_teacher_latents,
    sample_uniform_noise,
)
from .losses import rf_teacher_loss
from .training import train_teacher, train_teacher_reflow, generate_reflow_pairs

__all__ = [
    "RFTeacher",
    "integrate_teacher",
    "integrate_teacher_interval",
    "integrate_teacher_to_data",
    "sample_gaussian_noise",
    "sample_teacher_latents",
    "sample_uniform_noise",
    "rf_teacher_loss",
    "train_teacher",
    "train_teacher_reflow",
    "generate_reflow_pairs",
]
