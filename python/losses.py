from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


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

    for key in ("position", "tangent_u", "tangent_v", "scale", "color", "opacity"):
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



def _create_gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g2d = g[:, None] * g[None, :]
    window = g2d.view(1, 1, window_size, window_size)
    window = window.repeat(channels, 1, 1, 1)  # groups = channels
    return window


def _ssim_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channels: int,
) -> torch.Tensor:
    """
    img1, img2: (N, C, H, W) in [0, 1]
    window: (C, 1, window_size, window_size)
    """
    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=channels)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channels) - mu1_mu2

    # Standard SSIM constants for images in [0, 1]
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-8)

    return ssim_map.mean()


def compute_l2_ssim_loss_and_grad(
    current_rgb: np.ndarray,
    target_rgb: np.ndarray,
    ssim_weight: float = 0.2,
    window_size: int = 5,
    sigma: float = 1.5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Combined L2 + SSIM loss, with gradient w.r.t. the rendered image.

    Args:
        current_rgb: (H, W, 3), float32, typically [0, 1] or HDR.
        target_rgb:  (H, W, 3), same shape.
        ssim_weight: mixing parameter (similar spirit to 3DGS). 0 -> pure L2, 1 -> pure SSIM.
        window_size, sigma: SSIM window parameters.

    Returns:
        loss_value: scalar float (combined L2 + SSIM loss)
        grad_image: (H, W, 3) numpy array with dLoss/dI
        loss_image: (H, W) numpy array (here: per-pixel L2 map for visualization)
    """
    assert current_rgb.shape == target_rgb.shape, "current_rgb and target_rgb must have the same shape"
    H, W, C = current_rgb.shape
    assert C == 3, "SSIM helper currently assumes 3 channels"

    device = torch.device("cpu")  # safe & simple; can be changed to "cuda" if desired

    # (1) Wrap images as torch tensors
    pred = torch.tensor(current_rgb, dtype=torch.float32, device=device)
    tgt = torch.tensor(target_rgb, dtype=torch.float32, device=device)

    # Normalize to [0, 1] for SSIM stability (approximate; keep L2 on original scale)
    # Prevent division by zero if max == min.
    pred_min, pred_max = float(pred.min()), float(pred.max())
    tgt_min, tgt_max = float(tgt.min()), float(tgt.max())
    eps = 1e-6

    pred_norm = (pred - pred_min) / (pred_max - pred_min + eps)
    tgt_norm = (tgt - tgt_min) / (tgt_max - tgt_min + eps)

    pred_norm = pred_norm.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    tgt_norm = tgt_norm.permute(2, 0, 1).unsqueeze(0)

    pred_norm = pred_norm.clone().detach().requires_grad_(True)

    # (2) L2 term (on original, un-normalized scale, averaged)
    mse = torch.mean((pred - tgt) ** 2)

    # (3) SSIM term on normalized images
    window = _create_gaussian_window(window_size, sigma, channels=3).to(device)
    ssim_val = _ssim_torch(pred_norm, tgt_norm, window, window_size, channels=3)
    ssim_loss = 1.0 - ssim_val  # DSSIM-like

    # (4) Combine like 3DGS-style: L = (1 - w) * L2 + w * DSSIM
    loss = (1.0 - ssim_weight) * mse + ssim_weight * ssim_loss

    loss.backward()

    # (5) Gradient w.r.t. the original image (H, W, 3) via chain rule:
    # dL/dI = dL/dI_norm * dI_norm/dI, but we approximated by normalizing only for SSIM.
    # A simple and stable approach: reuse dL/d(pred) from mse + ssim via pred_norm path.
    # To keep consistent with your pipeline, we backprop through pred_norm scaling.
    grad_pred_norm = pred_norm.grad.detach()  # (1, 3, H, W)

    # Map gradient in normalized space back to original image approx:
    # I_norm = (I - min) / (max - min + eps)
    # => dI_norm/dI ≈ 1 / (max - min + eps)
    scale = 1.0 / (pred_max - pred_min + eps)
    grad_image = grad_pred_norm * scale
    grad_image = grad_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

    # (6) Per-pixel L2 map (for debug / visualization)
    loss_image = np.mean((current_rgb - target_rgb) ** 2, axis=-1).astype(np.float32)

    return float(loss.item()), grad_image, loss_image

