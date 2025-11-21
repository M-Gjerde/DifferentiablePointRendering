from __future__ import annotations

import numpy as np


def compute_l2_loss(rendered: np.ndarray, target: np.ndarray) -> float:
    """
    Simple L2 loss between rendered and target RGB images.

    Both inputs must be (H,W,3) float32.
    """
    if rendered.shape != target.shape:
        raise RuntimeError(
            f"Shape mismatch: rendered {rendered.shape}, target {target.shape}"
        )
    diff = rendered - target
    return float(np.mean(diff * diff))


def compute_l2_loss_and_grad(
    rendered: np.ndarray,
    target: np.ndarray,
    return_loss_image: bool = False,
):
    """
    L2 loss and gradient w.r.t. rendered image.

        C = mean((rendered - target)^2)
        dC/d(rendered) = 2 * (rendered - target) / (H * W * 3)

    If return_loss_image=True:
        Also returns an (H,W,3) per-pixel loss image: (rendered - target)^2
    """

    if rendered.shape != target.shape:
        raise RuntimeError(
            f"Shape mismatch: rendered {rendered.shape}, target {target.shape}"
        )

    diff = rendered - target
    loss_image = diff * diff
    loss = float(np.mean(loss_image))

    num_elements = diff.size
    grad_image = (2.0 / float(num_elements)) * diff

    if return_loss_image:
        return loss, grad_image, loss_image

    return loss, grad_image
