from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_l2_loss(rendered: np.ndarray, target: np.ndarray) -> float:
    """
    L2 loss (1/2 * mean squared error) between rendered and target RGB images.

    rendered, target: (H, W, 3) float32 arrays.
    """
    if rendered.shape != target.shape:
        raise RuntimeError(
            f"Shape mismatch: rendered {rendered.shape}, target {target.shape}"
        )

    diff = rendered - target                       # residuals r_i = p_i - t_i
    loss = 0.5 * np.mean(diff * diff)              # 1/2 * mean(r^2)
    return float(loss)


def compute_l2_grad(rendered: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Gradient of the L2 loss w.r.t. rendered image.

        dL/d(rendered_i) = (rendered_i - target_i) / N

    Returns:
        grad: same shape as rendered.
    """
    if rendered.shape != target.shape:
        raise RuntimeError(
            f"Shape mismatch: rendered {rendered.shape}, target {target.shape}"
        )

    diff = rendered - target
    #grad = diff / diff.size   # We do this in the renderer
    grad = diff
    return grad.astype(np.float32)


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
        current_rgb: (H, W, 3), float32, arbitrary range (e.g. [0,1] or HDR).
        target_rgb:  (H, W, 3), same shape.
        ssim_weight: mixing parameter; 0 -> pure L2, 1 -> pure SSIM.
        window_size, sigma: SSIM window parameters.

    Returns:
        loss_value: scalar float (combined L2 + SSIM loss)
        grad_image: (H, W, 3) numpy array with dLoss/dI
        loss_image: (H, W) numpy array, per-pixel L2 map (for visualization)
    """
    assert current_rgb.shape == target_rgb.shape, "current_rgb and target_rgb must have the same shape"
    H, W, C = current_rgb.shape
    assert C == 3, "SSIM helper currently assumes 3 channels"

    device = torch.device("cpu")  # or "cuda" if you like

    # Wrap as a leaf tensor that we will differentiate w.r.t.
    pred = torch.tensor(current_rgb, dtype=torch.float32, device=device, requires_grad=True)
    tgt = torch.tensor(target_rgb, dtype=torch.float32, device=device)

    # L2 term on original scale
    mse = torch.mean((pred - tgt) ** 2)

    # Normalization for SSIM (do not backprop through min/max -> treat as constants)
    with torch.no_grad():
        pred_min = pred.min().item()
        pred_max = pred.max().item()
        tgt_min = tgt.min().item()
        tgt_max = tgt.max().item()

    eps = 1e-6
    pred_norm = (pred - pred_min) / (pred_max - pred_min + eps)
    tgt_norm = (tgt - tgt_min) / (tgt_max - tgt_min + eps)

    # To NCHW
    pred_norm = pred_norm.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    tgt_norm = tgt_norm.permute(2, 0, 1).unsqueeze(0)

    # SSIM term
    window = _create_gaussian_window(window_size, sigma, channels=3).to(device)
    ssim_val = _ssim_torch(pred_norm, tgt_norm, window, window_size, channels=3)
    ssim_loss = 1.0 - ssim_val  # DSSIM-style

    # Combined objective
    loss = (1.0 - ssim_weight) * mse + ssim_weight * ssim_loss

    # Backprop to the original image
    loss.backward()
    grad_image = pred.grad.detach().cpu().numpy()  # (H, W, 3)

    # Per-pixel L2 map for debug
    loss_image = np.mean((current_rgb - target_rgb) ** 2, axis=-1).astype(np.float32)

    return float(loss.item()), grad_image, loss_image
