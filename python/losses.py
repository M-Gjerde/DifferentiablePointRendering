from __future__ import annotations

import numpy as np


def compute_l2_grad(rendered: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Maple loss per element:
        J = 1/2 * (rendered_i - target_i)^2

    Gradient per element:
        dJ/d(rendered_i) = (rendered_i - target_i)

    Note: This returns the per-element gradient of J (with normalization by N).
    """
    if rendered.shape != target.shape:
        raise RuntimeError(
            f"Shape mismatch: rendered {rendered.shape}, target {target.shape}"
        )

    diff = rendered - target
    grad = diff / diff.size
    return grad.astype(np.float32)



def compute_l2_loss(rendered: np.ndarray, target: np.ndarray) -> float:
    """Return the mean per-element half squared RGB error."""
    if rendered.shape != target.shape:
        raise RuntimeError(f"Shape mismatch: rendered {rendered.shape}, target {target.shape}")
    diff = rendered.astype(np.float64) - target.astype(np.float64)
    return float(0.5 * np.mean(diff ** 2))
