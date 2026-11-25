from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def resolve_rgb_asinh_scale(
    images: Iterable[np.ndarray], objective: str, explicit_scale: float | None = None,
) -> float:
    """Calibrate the HDR loss from positive linear training luminance. Zero selects linear L2."""
    if objective == "linear":
        return 0.0
    if objective != "asinh":
        raise ValueError(f"Unsupported RGB objective {objective!r}")
    if explicit_scale is not None:
        if not np.isfinite(explicit_scale) or explicit_scale <= 0.0:
            raise ValueError("The RGB asinh scale must be finite and positive")
        return float(explicit_scale)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    positive = []
    for image in images:
        luminance = image[..., :3] @ weights
        values = luminance[np.isfinite(luminance) & (luminance > 0)]
        if values.size:
            positive.append(values)
    # An all-black capture supplies no positive radiance from which to estimate a scale.
    return float(np.quantile(np.concatenate(positive), 0.75, overwrite_input=True)) if positive else 0.0


def _rgb_residual(rendered: np.ndarray, target: np.ndarray, scale: float) -> np.ndarray:
    if rendered.shape != target.shape:
        raise RuntimeError(f"Shape mismatch: rendered {rendered.shape}, target {target.shape}")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("The RGB asinh scale must be finite and nonnegative")
    if scale == 0.0:
        return rendered - target
    return scale * (np.arcsinh(rendered / scale) - np.arcsinh(target / scale))


def compute_l2_grad(rendered: np.ndarray, target: np.ndarray, rgb_asinh_scale: float = 0.0) -> np.ndarray:
    """Derivative of the mean per-channel half squared error with respect to linear RGB."""
    prediction = np.asarray(rendered, dtype=np.float64)
    diff = _rgb_residual(prediction, np.asarray(target, dtype=np.float64), rgb_asinh_scale)
    if rgb_asinh_scale:
        diff *= rgb_asinh_scale / np.hypot(prediction, rgb_asinh_scale)
    return (diff / diff.size).astype(np.float32)


def compute_l2_loss(rendered: np.ndarray, target: np.ndarray, rgb_asinh_scale: float = 0.0) -> float:
    """Mean per-channel half squared error in linear or asinh-transformed RGB."""
    diff = _rgb_residual(np.asarray(rendered, dtype=np.float64), np.asarray(target, dtype=np.float64), rgb_asinh_scale)
    return float(0.5 * np.mean(diff ** 2))
