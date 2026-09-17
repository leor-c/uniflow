"""Optional fixed-step learning-rate drop, shared by all toy teacher/student loops."""
import math

import torch


def make_lr_scheduler(optimizer, cfg, role):
    """Drop the LR after N completed updates, preserving optimizer state.

    Configurations without ``<role>_lr_drop_step`` retain their constant LR;
    this keeps the image experiments' training policy unchanged.
    """
    drop = getattr(cfg, f"{role}_lr_drop_step", None)
    if drop is None:
        return None
    final_lr = getattr(cfg, f"{role}_final_lr")
    if not isinstance(drop, int) or drop < 0:
        raise ValueError("LR drop step must be a nonnegative integer or None")
    if not math.isfinite(final_lr) or final_lr <= 0:
        raise ValueError("Final learning rate must be finite and positive")
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[drop], gamma=final_lr / optimizer.param_groups[0]['lr']
    )
