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


def compute_parameter_mse(current_params: dict[str, np.ndarray],
                          initial_params: dict[str, np.ndarray]) -> float:
    """
    Compute a single scalar MSE over all points and all parameters
    (position, tangent_u, tangent_v, scale, color), comparing the
    current values to the initial ones.

    If the number of points changes (densification), we only compare
    over the first min(N_initial, N_current) points for each tensor.
    """
    total_sq = 0.0
    total_count = 0

    for key in ("position", "tangent_u", "tangent_v", "scale", "color"):
        cur = current_params[key]
        init = initial_params[key]

        n = min(cur.shape[0], init.shape[0])
        if n == 0:
            continue

        diff = cur[:n] - init[:n]
        diff_flat = diff.reshape(-1)

        total_sq += float(np.dot(diff_flat, diff_flat))
        total_count += diff_flat.size

    if total_count == 0:
        return 0.0
    return total_sq / total_count

