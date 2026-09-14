"""Geometry map export conventions shared by the DPR and 2DGS orbit tools.

The companion repository carries an identical copy so each tool is standalone.
Depth is positive camera-space Z; normals are unit vectors in world space.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image


def depth_display_range(document: dict, override=None) -> tuple[float, float]:
    """Use one range for every frame and both renderers, derived from the poses."""
    if override is not None:
        near, far = map(float, override)
    else:
        extent = float(document["model_extent"])
        distances = [np.linalg.norm(np.asarray(f["position"]) - f["target"])
                     for f in document["frames"]]
        near = max(float(document["intrinsics"].get("near_clip", 0.01)),
                   min(distances) - extent)
        far = max(distances) + extent
    if not np.isfinite([near, far]).all() or not 0 <= near < far:
        raise ValueError("Depth display range must be finite with 0 <= NEAR < FAR")
    return float(near), float(far)


def geometry_maps(depth, normals, *, alpha=None):
    """Accept HxW depth and HxWx3/4 normals; mask invalid/background pixels."""
    depth = np.asarray(depth, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or normals.ndim != 3 or normals.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected depth/normal shapes: {depth.shape}, {normals.shape}")
    if normals.shape[:2] != depth.shape:
        raise ValueError("Depth and normal map dimensions do not match")
    valid = np.isfinite(depth) & (depth > 0)
    if alpha is not None:
        alpha = np.asarray(alpha)
        if alpha.shape != depth.shape:
            raise ValueError("Alpha and depth map dimensions do not match")
        valid &= np.isfinite(alpha) & (alpha > 0.5)
    normals = normals[..., :3]
    lengths = np.linalg.norm(normals, axis=-1)
    normal_valid = valid & np.isfinite(normals).all(axis=-1) & (lengths > 1e-8)
    unit_normals = np.zeros_like(normals)
    np.divide(normals, lengths[..., None], out=unit_normals,
              where=normal_valid[..., None])
    return np.where(valid, depth, 0), unit_normals, valid, normal_valid


def save_geometry_frame(output_dir: Path, index: int, depth, normals,
                        depth_range, *, alpha=None, save_raw=False) -> dict[str, np.ndarray]:
    depth, normals, valid, normal_valid = geometry_maps(depth, normals, alpha=alpha)
    near, far = depth_range
    # Viridis maps near to purple and far to yellow, with a black background. The fixed range
    # avoids flickering and makes the two renderers' depth videos comparable.
    scaled_depth = np.clip((depth - near) / (far - near), 0, 1)
    depth_rgb = colormaps["viridis"](scaled_depth, bytes=True)[..., :3].copy()
    depth_rgb[~valid] = 0
    normal_rgb = np.round((np.clip(normals, -1, 1) + 1) * 127.5).astype(np.uint8)
    normal_rgb[~normal_valid] = 0
    images = {"depth": depth_rgb, "normal": normal_rgb}
    for name, pixels in images.items():
        directory = output_dir / f"{name}_frames"
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(directory / f"frame_{index:05d}.png")
    if save_raw:
        directory = output_dir / "geometry_maps"
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"frame_{index:05d}.npz", depth=depth,
                            normal=normals, valid_depth=valid, valid_normal=normal_valid)
    return images


def geometry_video_path(rgb_video_path: Path, name: str) -> Path:
    return rgb_video_path.with_name(f"{rgb_video_path.stem}_{name}{rgb_video_path.suffix}")


def write_geometry_metadata(output_dir: Path, depth_range, *, save_raw: bool) -> None:
    (output_dir / "geometry_maps.json").write_text(json.dumps({
        "depth": "median camera-space Z in scene units; invalid pixels are zero",
        "depth_display_range": list(depth_range),
        "depth_colors": "viridis: near purple, far yellow, background black",
        "normal": "world-space unit surface normal; RGB = (normal + 1) / 2",
        "normal_background": "black",
        "raw_maps": "geometry_maps/frame_XXXXX.npz" if save_raw else None,
    }, indent=2) + "\n", encoding="utf-8")
