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


def compute_l2_loss_and_grad(
    rendered: np.ndarray,
    target: np.ndarray,
    return_loss_image: bool = False,
):
    """
    Discrete approximation using mean over all elements (pixels * channels):

        C = mean( 1/2 * (rendered - target)^2 )
        dC/d(rendered) = (rendered - target) / N

    If return_loss_image=True:
        loss_image is per-element: 1/2 * (rendered - target)^2
    """
    if rendered.shape != target.shape:
        raise RuntimeError(
            f"Shape mismatch: rendered {rendered.shape}, target {target.shape}"
        )

    diff = rendered - target

    loss_image = 0.5 * diff * diff
    loss = float(np.mean(loss_image))

    num_elements = diff.size
    grad_image = diff / float(num_elements)

    if return_loss_image:
        return loss, grad_image.astype(np.float32), loss_image.astype(np.float32)

    return loss, grad_image.astype(np.float32)



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
    ssim_map = numerator / torch.clamp_min(denominator, 1.0e-12)

    return ssim_map.mean()

def compute_l2_ssim_loss_and_grad(
    current_rgb: np.ndarray,
    target_rgb: np.ndarray,
    ssim_weight: float = 0.2,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Combined L2 + SSIM loss, with gradient w.r.t. the rendered image.

    Args:
        current_rgb: (H, W, 3), float32 in the optimization color space.
        target_rgb:  (H, W, 3), same shape.
        ssim_weight: mixing parameter; 0 -> half-MSE, 1 -> DSSIM.
        window_size, sigma: SSIM window parameters.

    Returns:
        loss_value: scalar float (combined L2 + SSIM loss)
        grad_image: (H, W, 3) numpy array with dLoss/dI
        loss_image: (H, W) numpy array, per-pixel L2 map (for visualization)
    """
    if current_rgb.shape != target_rgb.shape:
        raise RuntimeError("current_rgb and target_rgb must have the same shape")
    if current_rgb.ndim != 3 or current_rgb.shape[2] != 3:
        raise RuntimeError("SSIM helper expects HxWx3 RGB images")
    if not 0.0 <= float(ssim_weight) <= 1.0:
        raise ValueError("ssim_weight must be in [0, 1]")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    device = torch.device("cpu")  # or "cuda" if you like

    # Wrap as a leaf tensor that we will differentiate w.r.t.
    pred = torch.tensor(current_rgb, dtype=torch.float32, device=device, requires_grad=True)
    tgt = torch.tensor(target_rgb, dtype=torch.float32, device=device)

    # Preserve the renderer's historical half-MSE convention. SSIM uses the same
    # image values directly; per-image min/max normalization would make the target
    # and its gradient depend on unrelated extrema.
    l2 = 0.5 * torch.mean((pred - tgt) ** 2)

    # To NCHW
    pred_nchw = pred.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    tgt_nchw = tgt.permute(2, 0, 1).unsqueeze(0)

    # SSIM term
    window = _create_gaussian_window(window_size, sigma, channels=3).to(device)
    ssim_val = _ssim_torch(pred_nchw, tgt_nchw, window, window_size, channels=3)
    ssim_loss = 1.0 - ssim_val  # DSSIM-style

    # Combined objective
    loss = (1.0 - ssim_weight) * l2 + ssim_weight * ssim_loss

    # Backprop to the original image
    loss.backward()
    grad_image = pred.grad.detach().cpu().numpy()  # (H, W, 3)

    # Per-pixel half-MSE map for debug.
    loss_image = (0.5 * np.mean((current_rgb - target_rgb) ** 2, axis=-1)).astype(np.float32)

    return float(loss.item()), grad_image, loss_image


def compute_l2_ssim_metrics(
    rendered: np.ndarray,
    target: np.ndarray,
    ssim_weight: float = 0.2,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tuple[float, float, float]:
    """Return ``(combined, half_mse, dssim)`` using the training definition."""
    if rendered.shape != target.shape:
        raise RuntimeError(f"Shape mismatch: rendered {rendered.shape}, target {target.shape}")
    if rendered.ndim != 3 or rendered.shape[2] != 3:
        raise RuntimeError("SSIM helper expects HxWx3 RGB images")
    if not 0.0 <= float(ssim_weight) <= 1.0:
        raise ValueError("ssim_weight must be in [0, 1]")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    rendered_tensor = torch.as_tensor(rendered, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    target_tensor = torch.as_tensor(target, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    l2 = 0.5 * torch.mean((rendered_tensor - target_tensor) ** 2)
    window = _create_gaussian_window(window_size, sigma, channels=3)
    ssim_value = _ssim_torch(
        rendered_tensor,
        target_tensor,
        window,
        window_size,
        channels=3,
    )
    dssim = 1.0 - ssim_value
    combined = (1.0 - ssim_weight) * l2 + ssim_weight * dssim
    return float(combined.item()), float(l2.item()), float(dssim.item())
