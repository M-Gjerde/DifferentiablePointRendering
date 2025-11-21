from __future__ import annotations

import os
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np


def save_gradient_sign_png_py(
    file_path: Path,
    rgba32f: np.ndarray,  # (H,W,4) float32
    adjoint_spp: float = 32.0,
    abs_quantile: float = 0.99,
    flip_y: bool = True,
) -> bool:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(rgba32f, dtype=np.float32, order="C")
    if img.ndim != 3 or img.shape[2] < 3:
        return False

    rgb = img[..., :3] / float(max(adjoint_spp, 1e-8))
    scalar = np.mean(rgb, axis=2)
    scalar[~np.isfinite(scalar)] = 0.0

    finite_abs = np.abs(scalar[np.isfinite(scalar)])
    if finite_abs.size:
        q = np.clip(abs_quantile, 0.0, 1.0)
        scale_abs = np.quantile(finite_abs, q) if q < 1.0 else finite_abs.max()
        if not (np.isfinite(scale_abs) and scale_abs > 0.0):
            scale_abs = 1.0
    else:
        scale_abs = 1.0
    norm = np.clip(scalar / scale_abs, -1.0, 1.0)

    cmap = matplotlib.colormaps["seismic"]
    t = 0.5 * (norm + 1.0)
    rgba = cmap(t, bytes=True)
    out = rgba[..., :3]

    if flip_y:
        out = np.flipud(out)
    iio.imwrite(str(file_path), out)
    return True


def load_target_image(path: Path) -> np.ndarray:
    """
    Load a target RGB image as float32 array (H,W,3).
    """
    img = iio.imread(path.as_posix())
    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]

    if img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError(f"Target image must be HxWx3, got shape {img.shape}")

    if img.max() > 1.0 + 1e-4:
        img = img / 255.0

    return np.ascontiguousarray(img)


def save_positions_numpy(path: Path, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(positions, dtype=np.float32, order="C"))


def save_render(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(rgb, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    iio.imwrite(path.as_posix(), img_u8)


def save_loss_image(
    output_dir: Path,
    loss_image: np.ndarray,
    iteration: int,
) -> None:
    loss_image_vis = np.clip(
        loss_image / np.percentile(loss_image, 99.0), 0.0, 1.0
    )
    loss_image_u8 = (loss_image_vis * 255).astype(np.uint8)
    os.makedirs(output_dir / "loss", exist_ok=True)
    iio.imwrite(
        (output_dir / "loss" / f"loss_image_iter_{iteration:04d}.png").as_posix(),
        loss_image_u8,
    )
