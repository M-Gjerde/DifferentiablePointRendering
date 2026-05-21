from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from collections import deque

import numpy as np
import torch

import pale
from config import OptimizationConfig, RendererSettingsConfig
from io_utils import (
    load_target_image,
    save_positions_numpy,
    save_render,
    save_gradient_sign_png_py,
    save_loss_image,
    save_gaussians_to_ply,
)
from losses import (
    compute_l2_loss,
    compute_l2_grad,
    compute_parameter_mse,
)
from optimizers import (
    create_masked_optimizer,
)
from density_control import (
    make_under_reconstruction_evsplits,
    make_under_reconstruction_clones,
    compute_prune_indices_by_degenerate_scale,
    project_gradient_to_surfel_tangent_plane_np,
    compute_prune_indices_by_opacity,
    compute_scale_grow_shrink_pressure_np,
    add_densification_stats_np
)
from render_hooks import (
    remove_points,
    fetch_parameters,
    apply_point_parameters,
    verify_tangents_inplace,
    verify_scales_inplace,
    verify_albedos_inplace,
    verify_opacities_inplace,
    verify_beta_inplace,
    verify_positions_inplace,
    add_new_points,
    rebuild_bvh,
    get_training_camera_names, get_all_camera_names,
)
from debug_init_utils import add_debug_noise_to_initial_parameters

import sys
import select


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_forward_rgba(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    return np.asarray(
        forward_out[camera_name]["image"],
        dtype=np.float32,
        order="C",
    )


def get_forward_rgb(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    return get_forward_rgba(forward_out, camera_name)[..., :3]


def _infer_hw_from_forward(forward_out: Dict[str, dict], camera_name: str) -> tuple[int, int]:
    image = np.asarray(forward_out[camera_name]["image"], dtype=np.float32, order="C")
    return int(image.shape[0]), int(image.shape[1])


def get_forward_depth_distortion(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "depth_distortion" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w), dtype=np.float32)

    depth = np.asarray(
        camera_out["depth_distortion"],
        dtype=np.float32,
        order="C",
    )
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def get_forward_visible_normal(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "visible_normal" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w, 4), dtype=np.float32)

    normal = np.asarray(
        camera_out["visible_normal"],
        dtype=np.float32,
        order="C",
    )
    return np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)


def get_forward_normal_from_depth(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "normal_from_depth" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w, 4), dtype=np.float32)

    normal = np.asarray(
        camera_out["normal_from_depth"],
        dtype=np.float32,
        order="C",
    )
    return np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)


def save_normal_map_snapshot(
        output_path_png: Path,
        normal_rgba: np.ndarray,
        save_npy: bool = False,
) -> None:
    normal_rgba = np.asarray(normal_rgba, dtype=np.float32, order="C")
    normal_rgb = np.clip(normal_rgba[..., :3], -1.0, 1.0)

    if normal_rgba.shape[-1] >= 4:
        valid = normal_rgba[..., 3] > 0.0
    else:
        valid = np.ones(normal_rgb.shape[:2], dtype=bool)

    vis = 0.5 * (normal_rgb + 1.0)
    vis[~valid] = 0.0

    save_render(output_path_png, vis)

    if save_npy:
        np.save(output_path_png.with_suffix(".npy"), normal_rgba)


def make_trainable_surfel_mask_from_powers(
        powers: torch.Tensor,
        eps: float = 0.0,
) -> torch.Tensor:
    """
    Returns True for surfels that are allowed to receive gradient updates.

    Surfels with non-zero power are treated as light/emissive surfels and frozen.
    """
    with torch.no_grad():
        power_values = powers.detach()

        if power_values.ndim == 1:
            emissive_mask = torch.abs(power_values) > eps
        else:
            emissive_mask = torch.any(torch.abs(power_values) > eps, dim=1)

        return ~emissive_mask


def zero_frozen_surfel_gradients_np(
        trainable_mask: torch.Tensor,
        grad_position_np: np.ndarray,
        grad_tangent_u_np: np.ndarray,
        grad_tangent_v_np: np.ndarray,
        grad_scales_np: np.ndarray,
        grad_albedos_np: np.ndarray,
        grad_opacities_np: np.ndarray,
        grad_betas_np: np.ndarray,
) -> None:
    """
    In-place zeroing of gradients for frozen surfels.

    Frozen surfels are typically emissive surfels where power != 0.
    """
    trainable_mask_np = trainable_mask.detach().cpu().numpy().astype(bool)
    frozen_mask_np = ~trainable_mask_np

    grad_position_np[frozen_mask_np] = 0.0
    grad_tangent_u_np[frozen_mask_np] = 0.0
    grad_tangent_v_np[frozen_mask_np] = 0.0
    grad_scales_np[frozen_mask_np] = 0.0
    grad_albedos_np[frozen_mask_np] = 0.0
    grad_opacities_np[frozen_mask_np] = 0.0
    grad_betas_np[frozen_mask_np] = 0.0


def make_mean_reduction_adjoint_image(
        image_2d: np.ndarray,
        loss_weight: float,
) -> np.ndarray:
    """
    If loss_dist = loss_weight * mean(image_2d),
    then d loss_dist / d image_2d[p] = loss_weight / N for every pixel p.
    """
    pixel_count = max(image_2d.size, 1)
    return np.full(
        image_2d.shape,
        fill_value=loss_weight / float(pixel_count),
        dtype=np.float32,
    )


def sum_gradient_dicts(*gradient_dicts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Sum any number of renderer-produced gradient dictionaries parameter-wise.
    All dicts must contain the same keys and shapes.
    """
    active = [g for g in gradient_dicts if g is not None and len(g) > 0]
    if not active:
        return {}

    result: Dict[str, np.ndarray] = {
        key: np.asarray(value, dtype=np.float32, order="C").copy()
        for key, value in active[0].items()
    }

    for gradient_dict in active[1:]:
        if set(gradient_dict.keys()) != set(result.keys()):
            raise RuntimeError(
                f"Gradient key mismatch: {set(result.keys())} vs {set(gradient_dict.keys())}"
            )

        for key in result.keys():
            g = np.asarray(gradient_dict[key], dtype=np.float32, order="C")
            if result[key].shape != g.shape:
                raise RuntimeError(
                    f"Gradient shape mismatch for '{key}': "
                    f"{result[key].shape} vs {g.shape}"
                )
            result[key] += g

    return result


def poll_hotkey() -> Optional[str]:
    """
    Non-blocking check for a single-line keyboard input.

    Returns:
        's' if the user typed 's' + Enter      -> manual snapshot (render + points)
        'g' if the user typed 'g' + Enter      -> gradient dump
        None otherwise.

    Only works when stdin is a TTY (interactive terminal).
    """
    if not sys.stdin.isatty():
        return None

    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None

    line = sys.stdin.readline().strip().lower()
    if line in ("s", "g"):
        return line
    return None

def position_gradient_stats_or_zero(
        gradient_dict: Dict[str, np.ndarray],
) -> tuple[float, float]:
    if not gradient_dict or "position" not in gradient_dict:
        return 0.0, 0.0

    grad_position = np.asarray(
        gradient_dict["position"],
        dtype=np.float32,
        order="C",
    )

    return rms_point(grad_position), max_point_norm(grad_position)

def save_gradients_snapshot(
        output_dir: Path,
        iteration: int,
        grad_position_np: np.ndarray,
        grad_tangent_u_np: np.ndarray,
        grad_tangent_v_np: np.ndarray,
        grad_scales_np: np.ndarray,
        grad_albedos_np: np.ndarray,
        grad_opacities_np: np.ndarray,
        grad_betas_np: np.ndarray,
) -> None:
    """
    Save point-wise gradients to disk:

    - CSV: one row per point, easy to open in Excel:
        point_index,
        grad_pos_x, grad_pos_y, grad_pos_z,
        grad_tan_u_x, grad_tan_u_y, grad_tan_u_z,
        grad_tan_v_x, grad_tan_v_y, grad_tan_v_z,
        grad_scale_u, grad_scale_v,
        grad_albedo_r, grad_albedo_g, grad_albedo_b,
        grad_opacity, grad_beta

    - NPZ: raw arrays for programmatic inspection.
    """
    gradients_dir = output_dir / "gradients"
    gradients_dir.mkdir(parents=True, exist_ok=True)

    num_points = grad_position_np.shape[0]

    # Make sure 1D quantities are truly 1D
    grad_opacity_flat = grad_opacities_np.reshape(num_points)
    grad_beta_flat = grad_betas_np.reshape(num_points)

    # 1) CSV for quick inspection
    csv_path = gradients_dir / f"gradients_iter_{iteration:04d}.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        header = [
            "point_index",
            # position
            "grad_pos_x", "grad_pos_y", "grad_pos_z",
            # tangent_u
            "grad_tan_u_x", "grad_tan_u_y", "grad_tan_u_z",
            # tangent_v
            "grad_tan_v_x", "grad_tan_v_y", "grad_tan_v_z",
            # scales (s_u, s_v)
            "grad_scale_u", "grad_scale_v",
            # albedos (R,G,B)
            "grad_albedo_r", "grad_albedo_g", "grad_albedo_b",
            # scalar params
            "grad_opacity",
            "grad_beta",
        ]
        writer.writerow(header)

        for idx in range(num_points):
            gx, gy, gz = grad_position_np[idx]
            gux, guy, guz = grad_tangent_u_np[idx]
            gvx, gvy, gvz = grad_tangent_v_np[idx]
            gsu, gsv = grad_scales_np[idx]
            gcr, gcg, gcb = grad_albedos_np[idx]
            gop = grad_opacity_flat[idx]
            gb = grad_beta_flat[idx]

            row = [
                idx,
                gx, gy, gz,
                gux, guy, guz,
                gvx, gvy, gvz,
                gsu, gsv,
                gcr, gcg, gcb,
                gop,
                gb,
            ]
            writer.writerow(row)

    # 2) NPZ with full arrays (for Python / NumPy)
    npz_path = gradients_dir / f"gradients_iter_{iteration:04d}.npz"
    np.savez_compressed(
        npz_path,
        grad_position=grad_position_np,
        grad_tangent_u=grad_tangent_u_np,
        grad_tangent_v=grad_tangent_v_np,
        grad_scales=grad_scales_np,
        grad_albedos=grad_albedos_np,
        grad_opacities=grad_opacities_np,
        grad_betas=grad_betas_np,
    )

    print(
        f"[Iter {iteration:04d}] Hotkey 'g' pressed -> "
        f"saved gradients to:\n  {csv_path}\n  {npz_path}"
    )


from pathlib import Path

import numpy as np
import matplotlib


def save_depth_distortion_snapshot(
        output_path_png: Path,
        depth_distortion: np.ndarray,
        quantile: float = 0.99,
        save_npy: bool = False,
        cmap: str = "inferno",  # or "seismic"
) -> None:
    """
    Save a scalar depth-distortion map as:
      - PNG visualization (colormap applied, quantile-normalized)
      - optional raw .npy next to it
    """
    depth = np.asarray(depth_distortion, dtype=np.float32, order="C")
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.maximum(depth, 0.0)

    if save_npy:
        np.save(output_path_png.with_suffix(".npy"), depth)

    if depth.size == 0:
        vis = np.zeros((1, 1, 3), dtype=np.float32)
    else:
        vmax = float(np.quantile(depth, quantile))
        if not np.isfinite(vmax) or vmax <= 1e-12:
            vmax = 1.0

        vis_scalar = np.clip(depth / vmax, 0.0, 1.0)

        cmap_fn = matplotlib.colormaps[cmap]
        vis = cmap_fn(vis_scalar)[..., :3].astype(np.float32, copy=False)

    save_render(output_path_png, vis)

    if save_npy:
        np.save(output_path_png.with_suffix(".npy"), depth)


def save_checkpoint_snapshot(
        output_dir: Path,
        iteration: int,
        camera_ids: List[str],
        current_images: Dict[str, np.ndarray],
        positions: torch.Tensor,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
        camera_name: str,
) -> None:
    """
    Save an iteration checkpoint:
      - output_dir/checkpoints/iter_XXXX/points.ply
      - output_dir/checkpoints/iter_XXXX/render_<camera>.png
      - output_dir/checkpoints/iter_XXXX/depth_distortion_<camera>.png
      - output_dir/checkpoints/iter_XXXX/depth_distortion_<camera>.npy
      - output_dir/checkpoints/iter_XXXX/render_final.png
      - output_dir/checkpoints/iter_XXXX/depth_distortion_final.png
    """
    checkpoint_dir = output_dir / "checkpoints" / f"iter_{iteration:04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save points
    ply_path = checkpoint_dir / "points.ply"
    save_gaussians_to_ply(
        ply_path,
        positions,
        tangent_u,
        tangent_v,
        scales,
        albedos,
        opacities,
        betas,
        powers,
        shape_default=0.0,
    )

    # Save renders + depth distortion (all cameras)
    for cam_name in camera_ids:
        image_numpy = get_forward_rgb(current_images, cam_name)
        save_render(checkpoint_dir / f"render_{cam_name}.png", image_numpy)

        depth_numpy = get_forward_depth_distortion(current_images, cam_name)
        save_depth_distortion_snapshot(
            checkpoint_dir / f"depth_distortion_{cam_name}.png",
            depth_numpy,
            quantile=0.99,
            save_npy=False,
        )
        median_depth_numpy = get_forward_median_depth(current_images, cam_name)
        save_median_depth_snapshot(
            checkpoint_dir / f"median_depth_{cam_name}.png",
            median_depth_numpy,
            quantile=0.99,
            save_npy=False,
        )

    # Convenience: chosen/main camera
    main_img = get_forward_rgb(current_images, camera_name)
    save_render(checkpoint_dir / "render_final.png", main_img)

    main_depth = get_forward_depth_distortion(current_images, camera_name)
    save_depth_distortion_snapshot(
        checkpoint_dir / "depth_distortion_final.png",
        main_depth,
        quantile=0.99,
        save_npy=False,
    )

    main_median_depth = get_forward_median_depth(current_images, camera_name)
    save_median_depth_snapshot(
        checkpoint_dir / "median_depth_final.png",
        main_median_depth,
        quantile=0.99,
        save_npy=False,
    )

    print(f"[Iter {iteration:04d}] Saved checkpoint: {checkpoint_dir}")


def rms_point(x):
    n = max(x.shape[0], 1)
    return float(np.linalg.norm(x) / np.sqrt(n))


def rms_scalar(x):
    return float(np.linalg.norm(x) / np.sqrt(max(x.size, 1)))


def get_forward_median_depth(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "median_depth" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w), dtype=np.float32)

    depth = np.asarray(
        camera_out["median_depth"],
        dtype=np.float32,
        order="C",
    )
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def save_median_depth_snapshot(
        output_path_png: Path,
        median_depth: np.ndarray,
        quantile: float = 0.99,
        save_npy: bool = False,
        cmap: str = "viridis",
) -> None:
    depth = np.asarray(median_depth, dtype=np.float32, order="C")
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    if save_npy:
        np.save(output_path_png.with_suffix(".npy"), depth)

    valid = depth > 0.0

    if depth.size == 0 or not np.any(valid):
        vis = np.zeros((max(depth.shape[0], 1), max(depth.shape[1], 1), 3), dtype=np.float32)
    else:
        valid_depth = depth[valid]

        vmin = float(np.min(valid_depth))
        vmax = float(np.quantile(valid_depth, quantile))

        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin + 1e-12:
            vmax = vmin + 1.0

        vis_scalar = np.zeros_like(depth, dtype=np.float32)
        vis_scalar[valid] = np.clip((depth[valid] - vmin) / (vmax - vmin), 0.0, 1.0)

        cmap_fn = matplotlib.colormaps[cmap]
        vis = cmap_fn(vis_scalar)[..., :3].astype(np.float32, copy=False)
        vis[~valid] = 0.0

    save_render(output_path_png, vis)


def save_manual_snapshot(
        renderer: Pale.Renderer,
        output_dir: Path,
        iteration: int,
        positions: torch.Tensor,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
        camera_ids: List[str],
) -> None:
    """
    Save the current state using the same filenames as the final output.
    These will be overwritten later, both by future manual saves and at the end.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    final_images = renderer.render_forward()

    for camera_name in camera_ids:
        img_np = get_forward_rgb(final_images, camera_name)
        save_render(Path(output_dir / f"render_final_{camera_name}.png"), img_np)

        depth_np = get_forward_depth_distortion(final_images, camera_name)
        save_depth_distortion_snapshot(
            Path(output_dir / f"depth_distortion_final_{camera_name}.png"),
            depth_np,
            quantile=0.99,
            save_npy=False,
        )

        median_depth_np = get_forward_median_depth(final_images, camera_name)
        save_median_depth_snapshot(
            Path(output_dir / f"median_depth_final_{camera_name}.png"),
            median_depth_np,
            quantile=0.99,
            save_npy=False,
        )

    # Save full parameter set as PLY
    ply_path = output_dir / "points_final.ply"
    save_gaussians_to_ply(
        ply_path,
        positions,
        tangent_u,
        tangent_v,
        scales,
        albedos,
        opacities,
        betas,
        powers,
        shape_default=0.0,
    )
    print(
        f"[Iter {iteration:04d}] Hotkey 's' pressed -> "
        f"saved render_final_<camera>.png, depth_distortion_final_<camera>.png, and points_final.ply"
    )


def clear_output_dir(output_dir: Path) -> None:
    """
    Remove all files in the given directory (and direct subdirectories),
    then ensure the directory exists.
    """
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                for sub_item in item.iterdir():
                    if sub_item.is_file():
                        sub_item.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def max_point_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    if x.ndim == 1:
        return float(np.max(np.abs(x)))
    return float(np.max(np.linalg.norm(x, axis=1))) if x.shape[0] > 0 else 0.0


def rms_any(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.linalg.norm(x.ravel()) / np.sqrt(x.size))


def gradient_stats_from_dict(gradient_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for key, value in gradient_dict.items():
        g = np.asarray(value, dtype=np.float32, order="C")
        stats[key] = {
            "rms": rms_any(g),
            "max": max_point_norm(g),
        }
    return stats


def format_gradient_stats(tag: str, stats: Dict[str, Dict[str, float]]) -> str:
    def s(name: str) -> str:
        if name not in stats:
            return f"{name}=NA"
        return f"{name}_rms={stats[name]['rms']:.2e}, {name}_max={stats[name]['max']:.2e}"

    return (
        f"{tag}: "
        f"{s('position')}, "
        f"{s('tangent_u')}, "
        f"{s('tangent_v')}, "
        f"{s('scale')}, "
        f"{s('albedo')}, "
        f"{s('opacity')}, "
        f"{s('beta')}"
    )


def compute_opacity_target_regularizer_and_gradients(
        opacities: torch.Tensor,
        trainable_surfel_mask: torch.Tensor,
        opacity_target: float,
        opacity_weight: float,
        use_opacity_loss: bool,
) -> tuple[float, np.ndarray]:
    """
    Quadratic opacity target regularizer.

    L = opacity_weight * mean((opacity - opacity_target)^2)

    Only trainable surfels are included in the mean and receive gradients.
    Frozen emissive surfels receive zero gradient.

    Returns:
        loss_value,
        grad_opacity_np with same shape as opacities
    """
    opacity_np = opacities.detach().cpu().numpy().astype(np.float32, copy=False)
    grad_opacity_np = np.zeros_like(opacity_np, dtype=np.float32)

    if not use_opacity_loss or opacity_weight == 0.0:
        return 0.0, grad_opacity_np

    trainable_mask_np = trainable_surfel_mask.detach().cpu().numpy().astype(bool).reshape(-1)

    opacity_flat = opacity_np.reshape(-1)
    grad_opacity_flat = grad_opacity_np.reshape(-1)

    if opacity_flat.shape[0] != trainable_mask_np.shape[0]:
        raise RuntimeError(
            "Opacity regularizer shape mismatch: "
            f"opacities has {opacity_flat.shape[0]} scalar values, "
            f"but trainable_surfel_mask has {trainable_mask_np.shape[0]} entries."
        )

    if not np.any(trainable_mask_np):
        return 0.0, grad_opacity_np

    active_opacities = opacity_flat[trainable_mask_np]
    opacity_error = active_opacities - float(opacity_target)

    active_count = max(int(opacity_error.size), 1)

    loss_value = float(
        float(opacity_weight) * np.mean(opacity_error * opacity_error)
    )

    grad_active = (
                          2.0 * float(opacity_weight) / float(active_count)
                  ) * opacity_error

    grad_opacity_flat[trainable_mask_np] = grad_active.astype(np.float32, copy=False)

    return loss_value, grad_opacity_np


def compute_normal_consistency_loss_and_adjoints(
        visible_normal_rgba: np.ndarray,
        depth_normal_rgba: np.ndarray,
        weight: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    vis_rgba = np.asarray(visible_normal_rgba, dtype=np.float32, order="C")
    dep_rgba = np.asarray(depth_normal_rgba, dtype=np.float32, order="C")

    vis = vis_rgba[..., :3]
    dep = dep_rgba[..., :3]

    vis_valid = vis_rgba[..., 3] > 0.0
    dep_valid = dep_rgba[..., 3] > 0.0

    vis_finite = np.isfinite(vis).all(axis=-1)
    dep_finite = np.isfinite(dep).all(axis=-1)

    mask = vis_valid & dep_valid & vis_finite & dep_finite
    valid_count = max(int(mask.sum()), 1)

    scale = float(weight) / float(valid_count)

    dot_nd = np.sum(vis * dep, axis=-1)
    loss_map = np.zeros_like(dot_nd, dtype=np.float32)
    loss_map[mask] = scale * (1.0 - dot_nd[mask])
    loss = float(loss_map.sum())

    dL_dvis = np.zeros_like(vis, dtype=np.float32)
    dL_ddep = np.zeros_like(dep, dtype=np.float32)

    dL_dvis[mask] = -scale * dep[mask]
    dL_ddep[mask] = -scale * vis[mask]

    return loss, dL_dvis, dL_ddep


def summarize_depth_distortion_maps(
        depth_maps: Dict[str, np.ndarray],
        adjoint_maps: Dict[str, np.ndarray],
) -> str:
    if not depth_maps:
        return "depth_distortion: no cameras"

    means = []
    maxs = []
    p99s = []
    adjoint_means = []
    adjoint_maxs = []

    for camera_name, depth in depth_maps.items():
        d = np.asarray(depth, dtype=np.float32)
        a = np.asarray(adjoint_maps[camera_name], dtype=np.float32)

        means.append(float(np.mean(d)))
        maxs.append(float(np.max(d)))
        p99s.append(float(np.quantile(d, 0.99)))

        adjoint_means.append(float(np.mean(a)))
        adjoint_maxs.append(float(np.max(np.abs(a))))

    return (
        "depth_distortion_maps: "
        f"mean(avg)={np.mean(means):.3e}, "
        f"p99(avg)={np.mean(p99s):.3e}, "
        f"max(global)={np.max(maxs):.3e}, "
        f"adj_mean(avg)={np.mean(adjoint_means):.3e}, "
        f"adj_max(global)={np.max(adjoint_maxs):.3e}"
    )


def refetch_parameters_as_torch(
        renderer: pale.Renderer,
        device: torch.device,
) -> Tuple[torch.nn.Parameter, ...]:
    """
    Fetch parameters from the renderer, convert them to torch.nn.Parameter tensors.
    Returns (positions, tangent_u, tangent_v, scales, albedos, opacities).
    """
    updated = fetch_parameters(renderer)

    positions = torch.nn.Parameter(
        torch.tensor(updated["position"], device=device, dtype=torch.float32)
    )
    tangent_u = torch.nn.Parameter(
        torch.tensor(updated["tangent_u"], device=device, dtype=torch.float32)
    )
    tangent_v = torch.nn.Parameter(
        torch.tensor(updated["tangent_v"], device=device, dtype=torch.float32)
    )
    scales = torch.nn.Parameter(
        torch.tensor(updated["scale"], device=device, dtype=torch.float32)
    )
    albedos = torch.nn.Parameter(
        torch.tensor(updated["albedo"], device=device, dtype=torch.float32)
    )
    opacities = torch.nn.Parameter(
        torch.tensor(updated["opacity"], device=device, dtype=torch.float32)
    )
    betas = torch.nn.Parameter(
        torch.tensor(updated["beta"], device=device, dtype=torch.float32)
    )
    powers = torch.nn.Parameter(
        torch.tensor(updated["power"], device=device, dtype=torch.float32)
    )

    return positions, tangent_u, tangent_v, scales, albedos, opacities, betas, powers


def verify_parameters_inplane(
        positions: torch.Tensor,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        trainable_surfel_mask: Optional[torch.Tensor] = None,
) -> None:
    """
    Enforce parameter constraints in-place.

    For now, trainable_surfel_mask is only applied to beta verification.
    Frozen emissive surfels are therefore not beta-clamped.
    """
    verify_tangents_inplace(tangent_u, tangent_v)
    verify_scales_inplace(scales)
    verify_positions_inplace(positions)
    verify_albedos_inplace(albedos)
    verify_opacities_inplace(opacities)

    verify_beta_inplace(
        betas,
        trainable_surfel_mask=trainable_surfel_mask,
    )


def assign_numpy_gradients_to_tensors(
        device: torch.device,
        positions: torch.nn.Parameter,
        tangent_u: torch.nn.Parameter,
        tangent_v: torch.nn.Parameter,
        scales: torch.nn.Parameter,
        albedos: torch.nn.Parameter,
        opacities: torch.nn.Parameter,
        betas: torch.nn.Parameter,
        grad_position_np: np.ndarray,
        grad_tangent_u_np: np.ndarray,
        grad_tangent_v_np: np.ndarray,
        grad_scales_np: np.ndarray,
        grad_albedos_np: np.ndarray,
        grad_opacities_np: np.ndarray,
        grad_betas_np: np.ndarray,
) -> None:
    """
    Copy numpy gradient arrays into the .grad fields of the given torch Parameters.
    """
    positions.grad = torch.tensor(grad_position_np, device=device, dtype=torch.float32)
    tangent_u.grad = torch.tensor(grad_tangent_u_np, device=device, dtype=torch.float32)
    tangent_v.grad = torch.tensor(grad_tangent_v_np, device=device, dtype=torch.float32)
    scales.grad = torch.tensor(grad_scales_np, device=device, dtype=torch.float32)
    albedos.grad = torch.tensor(grad_albedos_np, device=device, dtype=torch.float32)
    opacities.grad = torch.tensor(grad_opacities_np, device=device, dtype=torch.float32)
    betas.grad = torch.tensor(grad_betas_np, device=device, dtype=torch.float32)


def compute_density_importance(
        grad_position_np: np.ndarray,
        grad_scales_np: np.ndarray,
        grad_opacities_np: np.ndarray,
) -> torch.Tensor:
    """
    Construct a scalar importance per point from position, scale, and opacity gradients.
    """
    grad_pos_norm = np.linalg.norm(grad_position_np, axis=1)
    grad_scale_norm = np.linalg.norm(grad_scales_np, axis=1)
    grad_opacity_abs = np.abs(grad_opacities_np).reshape(-1)

    importance_np = (
            grad_pos_norm
            + 0.3 * grad_scale_norm
            + 0.1 * grad_opacity_abs
    ).astype(np.float32)

    return torch.from_numpy(importance_np)


def run_optimization(
        renderer: Pale.Renderer,
        config: OptimizationConfig,
        renderer_settings: RendererSettingsConfig,
) -> None:
    def sum_gradient_dicts_any(*gradient_dicts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        nonempty = [g for g in gradient_dicts if g]
        if not nonempty:
            return {}

        reference_keys = list(nonempty[0].keys())
        result: Dict[str, np.ndarray] = {}

        for key in reference_keys:
            accumulated = None
            reference_shape = None

            for gradient_dict in nonempty:
                if key not in gradient_dict:
                    raise RuntimeError(f"Missing gradient key '{key}' in one of the gradient dictionaries.")

                grad = np.asarray(gradient_dict[key], dtype=np.float32, order="C")

                if reference_shape is None:
                    reference_shape = grad.shape
                    accumulated = np.array(grad, dtype=np.float32, copy=True, order="C")
                else:
                    if grad.shape != reference_shape:
                        raise RuntimeError(
                            f"Gradient shape mismatch for '{key}': "
                            f"expected {reference_shape}, got {grad.shape}"
                        )
                    accumulated += grad

            result[key] = accumulated

        return result

    # ------------------------------------------------------------------
    # 1a. Load target images per camera
    # ------------------------------------------------------------------
    target_path = Path(config.dataset_path)
    target_images: Dict[str, np.ndarray] = {}

    if not target_path.is_dir():
        raise RuntimeError(
            f"Target path '{target_path}' must be a directory when multiple cameras are used."
        )

    print(f"Loading target images from directory: {target_path}")
    training_camera_ids = get_training_camera_names(renderer)
    all_camera_ids = get_all_camera_names(renderer)

    for camera_name in training_camera_ids:
        image_path = target_path / "images" / f"{camera_name}.png"
        if not image_path.is_file():
            raise RuntimeError(
                f"Missing target image for camera '{camera_name}': {image_path}"
            )
        target_images[camera_name] = load_target_image(image_path)
        print(
            f"  Camera '{camera_name}': loaded target {image_path} "
            f"with shape {target_images[camera_name].shape}"
        )

    depth_distortion_weight = float(getattr(config, "depth_distort_weight", 0.0))
    normal_consistency_weight = float(getattr(config, "normal_consistency_weight", 0.0))

    opacity_loss_weight = float(getattr(config, "opacity_loss_weight", 0.0))
    opacity_target = config.opacity_target

    use_depth_distortion = depth_distortion_weight != 0.0
    use_normal_consistency = normal_consistency_weight != 0.0
    use_opacity_loss = opacity_loss_weight != 0.0

    print(
        "Loss terms: "
        f"depth_distortion={use_depth_distortion} weight={depth_distortion_weight:.3e}, "
        f"normal_consistency={use_normal_consistency} weight={normal_consistency_weight:.3e}, "
        f"opacity_loss_weight={opacity_loss_weight:.3e}, "
        f"opacity_target={opacity_target:.3f}"
    )

    # ------------------------------------------------------------------
    # Fetch initial parameters from renderer
    # ------------------------------------------------------------------
    initial_params = fetch_parameters(renderer)
    initial_positions_np = initial_params["position"]
    initial_tangent_u_np = initial_params["tangent_u"]
    initial_tangent_v_np = initial_params["tangent_v"]
    initial_scale_np = initial_params["scale"]
    initial_albedo_np = initial_params["albedo"]
    initial_opacity_np = initial_params["opacity"]
    initial_beta_np = initial_params["beta"]
    initial_power_np = initial_params["power"]

    num_points_initial = initial_positions_np.shape[0]
    print(f"Fetched {num_points_initial} initial points from PLY.")

    initial_params_reference: Dict[str, np.ndarray] = {
        "position": initial_positions_np.copy(),
        "tangent_u": initial_tangent_u_np.copy(),
        "tangent_v": initial_tangent_v_np.copy(),
        "scale": initial_scale_np.copy(),
        "albedo": initial_albedo_np.copy(),
        "opacity": initial_opacity_np.copy(),
        "beta": initial_beta_np.copy(),
        "power": initial_power_np.copy(),
    }

    apply_noise = False
    if apply_noise:
        (
            initial_positions_np,
            initial_tangent_u_np,
            initial_tangent_v_np,
            initial_scale_np,
            initial_albedo_np,
            initial_opacity_np,
            initial_beta_np,
        ) = add_debug_noise_to_initial_parameters(
            initial_positions_np,
            initial_tangent_u_np,
            initial_tangent_v_np,
            initial_scale_np,
            initial_albedo_np,
            initial_opacity_np,
            initial_beta_np,
        )
        print("Initial parameters perturbed by debug Gaussian noise.")

    device = torch.device(config.device)

    positions = torch.nn.Parameter(
        torch.tensor(initial_positions_np, device=device, dtype=torch.float32)
    )
    tangent_u = torch.nn.Parameter(
        torch.tensor(initial_tangent_u_np, device=device, dtype=torch.float32)
    )
    tangent_v = torch.nn.Parameter(
        torch.tensor(initial_tangent_v_np, device=device, dtype=torch.float32)
    )
    scales = torch.nn.Parameter(
        torch.tensor(initial_scale_np, device=device, dtype=torch.float32)
    )
    albedos = torch.nn.Parameter(
        torch.tensor(initial_albedo_np, device=device, dtype=torch.float32)
    )
    opacities = torch.nn.Parameter(
        torch.tensor(initial_opacity_np, device=device, dtype=torch.float32)
    )
    betas = torch.nn.Parameter(
        torch.tensor(initial_beta_np, device=device, dtype=torch.float32)
    )
    powers = torch.nn.Parameter(
        torch.tensor(initial_power_np, device=device, dtype=torch.float32)
    )

    # ------------------------------------------------------------------
    # 2. Initial reparameterization and sync with renderer
    # ------------------------------------------------------------------

    trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)

    frozen_surfel_count = int((~trainable_surfel_mask).sum().item())
    print(
        f"Frozen emissive surfels: {frozen_surfel_count} / "
        f"{int(trainable_surfel_mask.numel())}"
    )
    verify_parameters_inplane(positions, tangent_u, tangent_v, scales, albedos, opacities, betas,
                              trainable_surfel_mask=trainable_surfel_mask,
                              )

    apply_point_parameters(
        renderer, positions, tangent_u, tangent_v, scales, albedos, opacities, betas, powers
    )
    rebuild_bvh(renderer)

    (
        positions,
        tangent_u,
        tangent_v,
        scales,
        albedos,
        opacities,
        betas,
        powers,
    ) = refetch_parameters_as_torch(renderer, device)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = create_masked_optimizer(
        config,
        positions,
        tangent_u,
        tangent_v,
        scales,
        albedos,
        opacities,
        betas,
        powers,
    )

    # ------------------------------------------------------------------
    # 3. Initial loss and output dir setup (multi-camera)
    # ------------------------------------------------------------------
    initial_images = renderer.render_forward()

    initial_rgb_loss = 0.0
    initial_depth_distortion_loss_raw = 0.0
    initial_normal_loss_raw = 0.0

    initial_points_path = config.output_dir / "initial_points.ply"
    save_gaussians_to_ply(
        initial_points_path,
        positions,
        tangent_u,
        tangent_v,
        scales,
        albedos,
        opacities,
        betas,
        powers,
        shape_default=0.0,
    )
    print(f"Initial parameters written to PLY: {initial_points_path}")

    for camera_name in all_camera_ids:
        img_np = get_forward_rgb(initial_images, camera_name)

        camera_base_dir = config.output_dir / camera_name
        camera_base_dir.mkdir(parents=True, exist_ok=True)

        save_render(
            config.output_dir / f"render_initial_{camera_name}.png",
            img_np,
        )

        if camera_name not in target_images:
            print(
                f"Warning: no target image found for camera '{camera_name}', "
                f"skipping target save and loss."
            )
            continue

        tgt_np = target_images[camera_name]
        initial_rgb_loss += float(compute_l2_loss(img_np, tgt_np))

        save_render(
            config.output_dir / f"render_target_{camera_name}.png",
            tgt_np,
        )

        if use_depth_distortion:
            dist_np = get_forward_depth_distortion(initial_images, camera_name)
            initial_depth_distortion_loss_raw += float(dist_np.mean())

        if use_normal_consistency:
            visible_normal = get_forward_visible_normal(initial_images, camera_name)
            normal_from_depth = get_forward_normal_from_depth(initial_images, camera_name)
            raw_normal_loss_value, _, _ = compute_normal_consistency_loss_and_adjoints(
                visible_normal,
                normal_from_depth,
                1.0,
            )
            initial_normal_loss_raw += raw_normal_loss_value

    initial_depth_distortion_loss_weighted = (
            depth_distortion_weight * initial_depth_distortion_loss_raw
    )
    initial_normal_loss_weighted = (
            normal_consistency_weight * initial_normal_loss_raw
    )
    initial_opacity_regularizer_loss, _ = (
        compute_opacity_target_regularizer_and_gradients(
            opacities=opacities,
            trainable_surfel_mask=trainable_surfel_mask,
            opacity_target=opacity_target,
            opacity_weight=opacity_loss_weight,
            use_opacity_loss=use_opacity_loss,
        )
    )

    initial_total_loss = (
            initial_rgb_loss +
            initial_depth_distortion_loss_weighted +
            initial_normal_loss_weighted +
            initial_opacity_regularizer_loss
    )

    print(f"Initial RGB loss                       : {initial_rgb_loss:.6e}")
    print(f"Initial depth distortion loss (raw)    : {initial_depth_distortion_loss_raw:.6e}")
    print(f"Initial depth distortion loss (weighted): {initial_depth_distortion_loss_weighted:.6e}")
    print(f"Initial normal consistency loss (raw)  : {initial_normal_loss_raw:.6e}")
    print(f"Initial normal consistency loss (weighted): {initial_normal_loss_weighted:.6e}")
    print(f"Initial opacity regularizer loss        : {initial_opacity_regularizer_loss:.6e}")
    print(f"Initial total loss                     : {initial_total_loss:.6e}")

    # ------------------------------------------------------------------
    # 4. Density control / scheduling hyperparameters
    # ------------------------------------------------------------------
    densify_position_grad_accum_np = np.zeros(
        (positions.shape[0], 1),
        dtype=np.float32,
    )
    densify_position_grad_denom_np = np.zeros(
        (positions.shape[0], 1),
        dtype=np.float32,
    )
    densify_position_grad_vector_accum_np = np.zeros(
        tuple(positions.shape),
        dtype=np.float32,
    )
    point_birth_iteration_np = np.zeros(
        (positions.shape[0],),
        dtype=np.int64,
    )
    densify_scale_grow_accum_np = np.zeros(
        (positions.shape[0], 1),
        dtype=np.float32,
    )
    densify_scale_shrink_accum_np = np.zeros(
        (positions.shape[0], 1),
        dtype=np.float32,
    )
    densify_scale_pressure_denom_np = np.zeros(
        (positions.shape[0], 1),
        dtype=np.float32,
    )
    densify_split_min_shrink_fraction = 0.25
    densify_split_max_grow_to_shrink = 3.0
    densify_split_min_shrink_abs = 0.0
    densify_split_min_radius = 0.031 # Dont split with radius less than 0.001

    iteration = 0

    densification_interval = config.densification_interval
    prune_interval = config.prune_interval

    densify_after = (
        config.densify_after
        if config.densify_after >= 0
        else densification_interval
    )
    prune_after = (
        config.prune_after
        if config.prune_after >= 0
        else prune_interval
    )

    if config.densify_until_iteration >= 0:
        densify_until_iteration = config.densify_until_iteration
    else:
        densify_until_iteration = int(config.densify_until_fraction * config.iterations)

    opacity_prune_threshold = config.opacity_prune_threshold
    max_prune_fraction = config.max_prune_fraction

    reset_opacity_interval = config.reset_opacity_interval
    reset_opacity_value = config.reset_opacity_value

    densification_verbose = config.densification_verbose
    densification_grad_quantile = config.densification_grad_quantile
    densification_grad_abs_min = config.densification_grad_abs_min

    densify_bsdf_floor = config.densify_bsdf_floor
    densify_bsdf_gamma = config.densify_bsdf_gamma

    evsplit_preserve_integrated_opacity = config.evsplit_preserve_integrated_opacity

    rebuild_bvh_interval = config.rebuild_bvh_interval

    metrics_csv_path = config.output_dir / "metrics.csv"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "iteration",
                "camera_name",
                "loss_rgb_sum",
                "loss_depth_distortion_raw_sum",
                "loss_depth_distortion_weighted_sum",
                "loss_normal_consistency_raw_sum",
                "loss_normal_consistency_weighted_sum",
                "loss_opacity_regularizer",
                "loss_total_sum",
                "parameter_mse",
                "num_points",
                "iteration_time_sec",
                "total_time",
                "grad_position_renderer_rms",
                "grad_position_renderer_max",
                "grad_position_depth_distortion_rms",
                "grad_position_depth_distortion_max",
                "grad_position_normal_consistency_rms",
                "grad_position_normal_consistency_max",
                "grad_position_total_rms",
                "grad_position_total_max",
                "grad_opacity_total_rms",
                "grad_opacity_total_max",
                "grad_opacity_regularizer_rms",
                "grad_opacity_regularizer_max",
            ]
        )

        total_start_time = time.perf_counter()

        try:
            for iteration in range(1, config.iterations + 1):
                iteration_start = time.perf_counter()

                # --------------------------------------------------------------
                # 5. Forward pass and image-space loss (multi-camera)
                # --------------------------------------------------------------
                forward_out = renderer.render_forward()

                total_rgb_loss_value = 0.0
                total_depth_distortion_loss_raw = 0.0
                total_depth_distortion_loss_weighted = 0.0
                total_normal_loss_raw = 0.0
                total_normal_loss_weighted = 0.0
                total_loss_value = 0.0

                visible_normal_adjoints: Dict[str, np.ndarray] = {}
                depth_normal_adjoints: Dict[str, np.ndarray] = {}

                loss_grad_images: Dict[str, np.ndarray] = {}
                depth_distortion_grad_images: Dict[str, np.ndarray] = {}
                loss_images: Dict[str, np.ndarray] = {}
                depth_distortion_maps_for_logging: Dict[str, np.ndarray] = {}
                visible_normal_maps_for_logging: Dict[str, np.ndarray] = {}
                depth_normal_maps_for_logging: Dict[str, np.ndarray] = {}

                for camera_name in training_camera_ids:
                    current_rgb_np = get_forward_rgb(forward_out, camera_name)
                    target_rgb_np = target_images[camera_name]

                    rgb_grad = compute_l2_grad(current_rgb_np, target_rgb_np)
                    rgb_loss_value = float(compute_l2_loss(current_rgb_np, target_rgb_np))

                    total_rgb_loss_value += rgb_loss_value
                    total_loss_value += rgb_loss_value

                    loss_grad_images[camera_name] = rgb_grad
                    loss_images[camera_name] = rgb_grad

                    if use_depth_distortion:
                        current_depth_distortion_np = get_forward_depth_distortion(forward_out, camera_name)
                        depth_distortion_maps_for_logging[camera_name] = current_depth_distortion_np

                        depth_distortion_loss_raw = float(current_depth_distortion_np.mean())
                        depth_distortion_loss_weighted = (
                                depth_distortion_weight * depth_distortion_loss_raw
                        )

                        total_depth_distortion_loss_raw += depth_distortion_loss_raw
                        total_depth_distortion_loss_weighted += depth_distortion_loss_weighted
                        total_loss_value += depth_distortion_loss_weighted

                        depth_distortion_grad_images[camera_name] = make_mean_reduction_adjoint_image(
                            current_depth_distortion_np,
                            depth_distortion_weight,
                        )

                    if use_normal_consistency:
                        visible_normal = get_forward_visible_normal(forward_out, camera_name)
                        normal_from_depth = get_forward_normal_from_depth(forward_out, camera_name)

                        visible_normal_maps_for_logging[camera_name] = visible_normal
                        depth_normal_maps_for_logging[camera_name] = normal_from_depth

                        raw_normal_loss_value, dvis_raw, ddepth_raw = (
                            compute_normal_consistency_loss_and_adjoints(
                                visible_normal,
                                normal_from_depth,
                                1.0,
                            )
                        )

                        weighted_normal_loss_value = (
                                normal_consistency_weight * raw_normal_loss_value
                        )

                        total_normal_loss_raw += raw_normal_loss_value
                        total_normal_loss_weighted += weighted_normal_loss_value
                        total_loss_value += weighted_normal_loss_value

                        visible_normal_adjoints[camera_name] = (
                                normal_consistency_weight * dvis_raw
                        ).astype(np.float32, copy=False)
                        depth_normal_adjoints[camera_name] = (
                                normal_consistency_weight * ddepth_raw
                        ).astype(np.float32, copy=False)

                # ------------------------------------------------------------------
                # Backward passes
                # ------------------------------------------------------------------
                photo_gradients, adjoint_images = renderer.render_backward(loss_grad_images)

                distortion_gradients: Dict[str, np.ndarray] = {}
                normal_gradients: Dict[str, np.ndarray] = {}

                if use_depth_distortion and len(depth_distortion_grad_images) > 0:
                    distortion_gradients = renderer.render_depth_distortion_backward(
                        depth_distortion_grad_images
                    )

                if use_normal_consistency and len(visible_normal_adjoints) > 0:
                    normal_gradients = renderer.render_normal_consistency_backward(
                        visible_normal_adjoints,
                        depth_normal_adjoints,
                    )

                photo_gradient_stats = gradient_stats_from_dict(photo_gradients)
                distortion_gradient_stats = (
                    gradient_stats_from_dict(distortion_gradients)
                    if distortion_gradients else {}
                )
                normal_gradient_stats = (
                    gradient_stats_from_dict(normal_gradients)
                    if normal_gradients else {}
                )

                # ------------------------------------------------------------------
                # Blend the final update vector in Python
                # ------------------------------------------------------------------
                total_gradients = sum_gradient_dicts_any(
                    photo_gradients,
                    distortion_gradients,
                    normal_gradients,
                )

                grad_position_np = np.asarray(total_gradients["position"], dtype=np.float32, order="C")
                grad_tangent_u_np = np.asarray(total_gradients["tangent_u"], dtype=np.float32, order="C")
                grad_tangent_v_np = np.asarray(total_gradients["tangent_v"], dtype=np.float32, order="C")
                grad_scales_np = np.asarray(total_gradients["scale"], dtype=np.float32, order="C")
                grad_albedos_np = np.asarray(total_gradients["albedo"], dtype=np.float32, order="C")
                grad_opacities_np = np.asarray(total_gradients["opacity"], dtype=np.float32, order="C")
                grad_betas_np = np.asarray(total_gradients["beta"], dtype=np.float32, order="C")

                opacity_regularizer_loss, grad_opacity_regularizer_np = (
                    compute_opacity_target_regularizer_and_gradients(
                        opacities=opacities,
                        trainable_surfel_mask=trainable_surfel_mask,
                        opacity_target=opacity_target,
                        opacity_weight=opacity_loss_weight,
                        use_opacity_loss=use_opacity_loss,
                    )
                )
                opacity_regularizer_gradient_stats = gradient_stats_from_dict(
                    {"opacity": grad_opacity_regularizer_np}
                )
                grad_opacities_np += grad_opacity_regularizer_np
                total_loss_value += opacity_regularizer_loss

                grad_opacity_regularizer_rms = rms_any(grad_opacity_regularizer_np)
                grad_opacity_regularizer_max = max_point_norm(grad_opacity_regularizer_np)

                grad_opacity_total_rms = rms_any(grad_opacities_np)
                grad_opacity_total_max = max_point_norm(grad_opacities_np)
                grad_position_renderer_rms, grad_position_renderer_max = (
                    position_gradient_stats_or_zero(photo_gradients)
                )

                grad_position_depth_distortion_rms, grad_position_depth_distortion_max = (
                    position_gradient_stats_or_zero(distortion_gradients)
                )

                grad_position_normal_consistency_rms, grad_position_normal_consistency_max = (
                    position_gradient_stats_or_zero(normal_gradients)
                )

                grad_position_total_rms = rms_point(grad_position_np)
                grad_position_total_max = max_point_norm(grad_position_np)


                current_positions_shape = tuple(positions.shape)
                if grad_position_np.shape != current_positions_shape:
                    raise RuntimeError(
                        f"Gradient shape mismatch for position: expected {current_positions_shape}, "
                        f"got {grad_position_np.shape}"
                    )

                density_grad_position_np_raw = np.asarray(
                    total_gradients["position"],
                    dtype=np.float32,
                    order="C",
                )

                density_grad_position_np = project_gradient_to_surfel_tangent_plane_np(
                    grad_position_np=density_grad_position_np_raw,
                    tangent_u=tangent_u,
                    tangent_v=tangent_v,
                )

                with torch.no_grad():
                    albedo_np = albedos.detach().cpu().numpy().astype(np.float32)
                linear_rgb_bsdf_scale_np = np.mean(albedo_np, axis=1)
                bsdf_normalizer_np = np.maximum(
                    linear_rgb_bsdf_scale_np,
                    densify_bsdf_floor,
                ) ** densify_bsdf_gamma

                with torch.no_grad():
                    scales_np = scales.detach().cpu().numpy().astype(np.float32)
                area_np = (
                        np.maximum(scales_np[:, 0], 1.0e-12)
                        * np.maximum(scales_np[:, 1], 1.0e-12)
                )

                radius_np = np.sqrt(area_np)

                # Use the same scale at which surfels become eligible for splitting.
                # This prevents tiny surfels from getting artificially huge scores.
                score_min_radius = float(densify_split_min_radius)
                score_min_area = score_min_radius * score_min_radius

                densify_score_area_power = float(
                    getattr(config, "densify_score_area_power", 1.0)
                )

                area_normalizer_np = np.maximum(area_np, score_min_area) ** densify_score_area_power

                density_grad_position_np_for_score = (
                        density_grad_position_np
                        / bsdf_normalizer_np[:, None]
                       # / area_normalizer_np[:, None]
                )

                density_grad_scale_np_raw = np.asarray(
                    total_gradients["scale"],
                    dtype=np.float32,
                    order="C",
                )
                with torch.no_grad():
                    scales_np_for_density = scales.detach().cpu().numpy().astype(np.float32)
                scale_grow_pressure_np, scale_shrink_pressure_np, _ = (
                    compute_scale_grow_shrink_pressure_np(
                        scales_np=scales_np_for_density,
                        grad_scales_np=density_grad_scale_np_raw,
                        normalizer_np=bsdf_normalizer_np,
                    )
                )


                if distortion_gradients:
                    distortion_grad_position_np = np.asarray(
                        distortion_gradients["position"],
                        dtype=np.float32,
                        order="C",
                    )

                    distortion_grad_position_np = project_gradient_to_surfel_tangent_plane_np(
                        grad_position_np=distortion_grad_position_np,
                        tangent_u=tangent_u,
                        tangent_v=tangent_v,
                    )

                    distortion_grad_position_np = (
                            distortion_grad_position_np
                            / bsdf_normalizer_np[:, None]
                    )

                    distortion_score_np = np.linalg.norm(distortion_grad_position_np, axis=1)
                    distortion_score_np = np.nan_to_num(
                        distortion_score_np,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                else:
                    distortion_score_np = np.zeros((positions.shape[0],), dtype=np.float32)


                add_densification_stats_np(
                    grad_position_np=density_grad_position_np_for_score,
                    trainable_surfel_mask=trainable_surfel_mask,
                    accum_np=densify_position_grad_accum_np,
                    denom_np=densify_position_grad_denom_np,
                    update_only_nonzero=True,
                )

                # Accumulate vector direction too.
                with torch.no_grad():
                    trainable_np_for_density = (
                        trainable_surfel_mask
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(bool)
                        .reshape(-1)
                    )

                    grad_norm_for_density_np = np.linalg.norm(
                        density_grad_position_np_for_score,
                        axis=1,
                    )

                    update_density_vector_mask_np = (
                            trainable_np_for_density
                            & np.isfinite(grad_norm_for_density_np)
                            & (grad_norm_for_density_np > 0.0)
                    )

                    densify_position_grad_vector_accum_np[update_density_vector_mask_np] += (
                        density_grad_position_np_for_score[update_density_vector_mask_np]
                    )
                    scale_pressure_np = scale_grow_pressure_np + scale_shrink_pressure_np

                    update_scale_pressure_mask_np = (
                            trainable_np_for_density
                            & np.isfinite(scale_pressure_np)
                            & (scale_pressure_np > 0.0)
                    )

                    densify_scale_grow_accum_np[update_scale_pressure_mask_np, 0] += (
                        scale_grow_pressure_np[update_scale_pressure_mask_np]
                    )

                    densify_scale_shrink_accum_np[update_scale_pressure_mask_np, 0] += (
                        scale_shrink_pressure_np[update_scale_pressure_mask_np]
                    )

                    densify_scale_pressure_denom_np[update_scale_pressure_mask_np, 0] += 1.0

                # --------------------------------------------------------------
                # 7. Optimizer step
                # --------------------------------------------------------------
                optimizer.zero_grad(set_to_none=True)

                zero_frozen_surfel_gradients_np(
                    trainable_surfel_mask,
                    grad_position_np,
                    grad_tangent_u_np,
                    grad_tangent_v_np,
                    grad_scales_np,
                    grad_albedos_np,
                    grad_opacities_np,
                    grad_betas_np,
                )

                assign_numpy_gradients_to_tensors(
                    device,
                    positions,
                    tangent_u,
                    tangent_v,
                    scales,
                    albedos,
                    opacities,
                    betas,
                    grad_position_np,
                    grad_tangent_u_np,
                    grad_tangent_v_np,
                    grad_scales_np,
                    grad_albedos_np,
                    grad_opacities_np,
                    grad_betas_np,
                )

                optimizer.step()

                if reset_opacity_interval > 0 and iteration % reset_opacity_interval == 0:
                    with torch.no_grad():
                        opacities[trainable_surfel_mask] = float(reset_opacity_value)
                        print(f"[Iter {iteration:04d}] "
                          f"Resetting all opacities to {reset_opacity_value}")
                # --------------------------------------------------------------
                # 8. Reparameterization, sync, BVH
                # --------------------------------------------------------------
                verify_parameters_inplane(
                    positions, tangent_u, tangent_v, scales, albedos, opacities, betas,
                    trainable_surfel_mask=trainable_surfel_mask,

                )

                apply_point_parameters(
                    renderer,
                    positions,
                    tangent_u,
                    tangent_v,
                    scales,
                    albedos,
                    opacities,
                    betas,
                    powers
                )

                if iteration % rebuild_bvh_interval == 0:
                    rebuild_bvh(renderer)
                # --------------------------------------------------------------
                # 9. Densification + pruning
                # --------------------------------------------------------------
                densification_result: Optional[Dict[str, np.ndarray]] = None
                indices_to_remove_list: List[int] = []

                if (
                        densify_after <= iteration <= densify_until_iteration
                        and iteration % densification_interval == 0
                ):
                    with torch.no_grad():
                        valid_denom_np = densify_position_grad_denom_np.reshape(-1) > 0.0
                        valid_scale_pressure_np = densify_scale_pressure_denom_np.reshape(-1) > 0.0

                        avg_grow_pressure_np = np.zeros(
                            (positions.shape[0],),
                            dtype=np.float32,
                        )
                        avg_shrink_pressure_np = np.zeros(
                            (positions.shape[0],),
                            dtype=np.float32,
                        )

                        avg_grow_pressure_np[valid_scale_pressure_np] = (
                                densify_scale_grow_accum_np.reshape(-1)[valid_scale_pressure_np]
                                /
                                densify_scale_pressure_denom_np.reshape(-1)[valid_scale_pressure_np]
                        )

                        avg_shrink_pressure_np[valid_scale_pressure_np] = (
                                densify_scale_shrink_accum_np.reshape(-1)[valid_scale_pressure_np]
                                /
                                densify_scale_pressure_denom_np.reshape(-1)[valid_scale_pressure_np]
                        )

                        scale_total_pressure_np = avg_grow_pressure_np + avg_shrink_pressure_np

                        scale_shrink_fraction_np = np.zeros_like(avg_shrink_pressure_np)
                        nonzero_scale_pressure_np = scale_total_pressure_np > 1.0e-20
                        scale_shrink_fraction_np[nonzero_scale_pressure_np] = (
                                avg_shrink_pressure_np[nonzero_scale_pressure_np]
                                /
                                scale_total_pressure_np[nonzero_scale_pressure_np]
                        )

                        # Plane suppression / curvature allowance:
                        #
                        # - Flat undercovered plane:
                        #       grow_pressure large, shrink_pressure small -> no split.
                        #
                        # - Curved or spherical patch represented by too large a tangent surfel:
                        #       shrink_pressure nontrivial -> split allowed.
                        split_scale_gate_np = (
                                valid_scale_pressure_np
                                & (avg_shrink_pressure_np >= densify_split_min_shrink_abs)
                                & (scale_shrink_fraction_np >= densify_split_min_shrink_fraction)
                                & (
                                        avg_grow_pressure_np
                                        <= densify_split_max_grow_to_shrink
                                        * np.maximum(avg_shrink_pressure_np, 1.0e-20)
                                )
                        )

                        avg_density_grad_norm_np = np.zeros(
                            (positions.shape[0],),
                            dtype=np.float32,
                        )

                        avg_density_grad_norm_np[valid_denom_np] = (
                                densify_position_grad_accum_np.reshape(-1)[valid_denom_np]
                                /
                                densify_position_grad_denom_np.reshape(-1)[valid_denom_np]
                        )

                        avg_density_grad_vector_np = np.zeros(
                            tuple(positions.shape),
                            dtype=np.float32,
                        )

                        avg_density_grad_vector_np[valid_denom_np] = (
                                densify_position_grad_vector_accum_np[valid_denom_np]
                                /
                                densify_position_grad_denom_np.reshape(-1, 1)[valid_denom_np]
                        )
                        grad_pos_norm_np = np.nan_to_num(
                            avg_density_grad_norm_np,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )

                        depth_distortion_split_suppression = 1e3

                        distortion_scale = np.quantile(
                            distortion_score_np[np.isfinite(distortion_score_np)],
                            0.90,
                        ) if np.any(np.isfinite(distortion_score_np)) else 1.0

                        if not np.isfinite(distortion_scale) or distortion_scale <= 1.0e-12:
                            distortion_scale = 1.0

                        distortion_score_normalized_np = distortion_score_np / distortion_scale

                        #grad_pos_norm_np = (
                        #        grad_pos_norm_np
                        #        / (1.0 + depth_distortion_split_suppression * distortion_score_normalized_np)
                        #)

                        trainable_np = trainable_surfel_mask.detach().cpu().numpy().astype(bool)

                        point_age_np = iteration - point_birth_iteration_np
                        finite_grad = np.isfinite(grad_pos_norm_np)
                        large_enough_to_split_np = radius_np >= densify_split_min_radius

                        candidate_mask_np = (
                                valid_denom_np
                                & finite_grad
                                & trainable_np
                                #& split_scale_gate_np
                                #& large_enough_to_split_np
                                & (grad_pos_norm_np >= densification_grad_abs_min)
                        )
                        n_new_from_densification = 0
                        densify_reason = "not_attempted"

                        finite_count = int(np.count_nonzero(finite_grad))
                        trainable_count = int(np.count_nonzero(trainable_np))
                        above_abs_count = int(np.count_nonzero(grad_pos_norm_np >= densification_grad_abs_min))
                        candidate_count = int(np.count_nonzero(candidate_mask_np))
                        valid_denom_count = int(np.count_nonzero(valid_denom_np))

                        valid_scale_pressure_count = int(np.count_nonzero(valid_scale_pressure_np))
                        #split_scale_gate_count = int(np.count_nonzero(split_scale_gate_np))

                        if valid_scale_pressure_count > 0:
                            grow_mean = float(np.mean(avg_grow_pressure_np[valid_scale_pressure_np]))
                            shrink_mean = float(np.mean(avg_shrink_pressure_np[valid_scale_pressure_np]))
                            shrink_frac_mean = float(np.mean(scale_shrink_fraction_np[valid_scale_pressure_np]))
                        else:
                            grow_mean = 0.0
                            shrink_mean = 0.0
                            shrink_frac_mean = 0.0

                        grad_threshold = float("nan")
                        grad_quantile_threshold = float("nan")

                        if grad_pos_norm_np.size > 0:
                            finite_signal_np = grad_pos_norm_np[np.isfinite(grad_pos_norm_np)]
                        else:
                            finite_signal_np = np.zeros((0,), dtype=np.float32)

                        if finite_signal_np.size > 0:
                            signal_min = float(np.min(finite_signal_np))
                            signal_p50 = float(np.quantile(finite_signal_np, 0.50))
                            signal_p90 = float(np.quantile(finite_signal_np, 0.90))
                            signal_p95 = float(np.quantile(finite_signal_np, 0.95))
                            signal_p98 = float(np.quantile(finite_signal_np, 0.98))
                            signal_max = float(np.max(finite_signal_np))
                        else:
                            signal_min = signal_p50 = signal_p90 = signal_p95 = signal_p98 = signal_max = 0.0

                        if not np.any(valid_denom_np):
                            densify_reason = "no_density_samples"
                        elif candidate_count == 0:
                            densify_reason = "no_candidates_after_grad_trainable"
                        else:
                            active_grad = grad_pos_norm_np[candidate_mask_np]

                            grad_quantile_threshold = float(
                                np.quantile(active_grad, densification_grad_quantile)
                            )

                            grad_threshold = max(
                                float(densification_grad_abs_min),
                                grad_quantile_threshold,
                            )

                            densify_mask_torch = torch.as_tensor(
                                candidate_mask_np,
                                device=positions.device,
                                dtype=torch.bool,
                            )

                            densification_result = make_under_reconstruction_evsplits(
                                positions=positions,
                                tangent_u=tangent_u,
                                tangent_v=tangent_v,
                                scales=scales,
                                albedos=albedos,
                                opacities=opacities,
                                betas=betas,
                                powers=powers,
                                grad_position_np=avg_density_grad_vector_np,
                                selection_score_np=grad_pos_norm_np,
                                trainable_surfel_mask=densify_mask_torch,
                                grad_threshold=grad_threshold,
                                min_scale=config.evsplit_min_scale,
                                preserve_integrated_opacity=evsplit_preserve_integrated_opacity,
                            )
                            #densification_result = make_under_reconstruction_clones(
                            #    positions=positions,
                            #    tangent_u=tangent_u,
                            #    tangent_v=tangent_v,
                            #    scales=scales,
                            #    albedos=albedos,
                            #    opacities=opacities,
                            #    betas=betas,
                            #    powers=powers,
                            #    grad_position_np=avg_density_grad_vector_np,
                            #    selection_score_np=grad_pos_norm_np,
                            #    trainable_surfel_mask=densify_mask_torch,
                            #    grad_threshold=grad_threshold
                            #)

                            if densification_result is not None:
                                if densification_result.get("replace_source", False):
                                    src = densification_result.get("source_index", None)
                                    if src is not None:
                                        indices_to_remove_list.extend(int(i) for i in np.asarray(src, dtype=np.int64))

                                new_block = densification_result.get("new", None)
                                if new_block is not None:
                                    n_new_from_densification = int(new_block["position"].shape[0])
                                    densify_reason = "evsplit_added"
                                else:
                                    densify_reason = "evsplit_result_without_new_block"
                            else:
                                densify_reason = "selected_candidates_but_evsplit_filter_rejected_all"

                        if densification_verbose:
                            print(
                                f"[Iter {iteration:04d}] Densification check | "
                                f"reason={densify_reason}, "
                                f"added={n_new_from_densification}, "
                                f"pts={positions.shape[0]}, "
                                f"valid_denom={valid_denom_count}, "
                                f"finite={finite_count}, "
                                f"trainable={trainable_count}, "
                                f"above_abs={above_abs_count}, "
                                f"candidates={candidate_count}, "
                                f"signal_min={signal_min:.3e}, "
                                f"signal_p50={signal_p50:.3e}, "
                                f"signal_p90={signal_p90:.3e}, "
                                f"signal_p95={signal_p95:.3e}, "
                                f"signal_p98={signal_p98:.3e}, "
                                f"signal_max={signal_max:.3e}, "
                                f"grad_q_thr={grad_quantile_threshold:.3e}, "
                                f"grad_thr={grad_threshold:.3e}, "
                                f"abs_thr={densification_grad_abs_min:.3e}, "
                                f"rgb={total_rgb_loss_value:.3e}, "
                                f"valid_scale={valid_scale_pressure_count}, "
                                #f"split_scale_gate={split_scale_gate_count}, "
                                f"grow_mean={grow_mean:.3e}, "
                                f"shrink_mean={shrink_mean:.3e}, "
                                f"shrink_frac_mean={shrink_frac_mean:.3f}, "
                            )
                        elif n_new_from_densification > 0:
                            print(
                                f"[Iter {iteration:04d}] Clone densification: "
                                f"adding {n_new_from_densification} surfels | "
                                f"grad_thr={grad_threshold:.3e}, "
                                f"abs_thr={densification_grad_abs_min:.3e}, "
                                f"pts={positions.shape[0]}"
                            )

                scale_prune_indices = np.zeros((0,), dtype=np.int64)
                opacity_prune_indices = np.zeros((0,), dtype=np.int64)

                if iteration >= prune_after and iteration % prune_interval == 0:
                    # ----------------------------------------------------------
                    # Prune geometrically degenerate surfels.
                    #
                    # A surfel is degenerate if either:
                    #   scale_u <= 1e-5
                    #   scale_v <= 1e-5
                    #
                    # trainable_surfel_mask protects emissive/light surfels.
                    # ----------------------------------------------------------
                    scale_prune_indices = compute_prune_indices_by_degenerate_scale(
                        scales,
                        min_scale=config.scale_prune_min_scale,
                        trainable_mask=trainable_surfel_mask,
                        min_points_to_keep=config.min_points_to_keep_after_scale_prune,
                    )

                    if scale_prune_indices.size > 0:
                        indices_to_remove_list.extend(int(i) for i in scale_prune_indices)

                    # ----------------------------------------------------------
                    # Prune low-opacity surfels.
                    # ----------------------------------------------------------
                    opacity_prune_indices = compute_prune_indices_by_opacity(
                        opacities,
                        min_opacity=opacity_prune_threshold,
                        use_quantile=False,
                        max_fraction_to_prune=max_prune_fraction,
                    )

                    if opacity_prune_indices.size > 0:
                        indices_to_remove_list.extend(int(i) for i in opacity_prune_indices)

                if indices_to_remove_list or densification_result is not None:
                    if indices_to_remove_list:
                        scale_prune_set = set(int(i) for i in scale_prune_indices)
                        opacity_prune_set = set(int(i) for i in opacity_prune_indices)

                        scale_only_set = scale_prune_set - opacity_prune_set
                        opacity_only_set = opacity_prune_set - scale_prune_set
                        overlap_set = scale_prune_set & opacity_prune_set

                        indices_to_remove = np.unique(
                            np.asarray(indices_to_remove_list, dtype=np.int64)
                        )

                        print(
                            f"[Iter {iteration:04d}] Pruning {indices_to_remove.size} unique surfels | "
                            f"scale={len(scale_prune_set)}, "
                            f"opacity={len(opacity_prune_set)}, "
                            f"both={len(overlap_set)}, "
                            f"scale_only={len(scale_only_set)}, "
                            f"opacity_only={len(opacity_only_set)}"
                        )

                        remove_points(renderer, indices_to_remove)

                        # Keep age array consistent with pruning.
                        keep_mask_np = np.ones(point_birth_iteration_np.shape[0], dtype=bool)
                        keep_mask_np[indices_to_remove] = False
                        point_birth_iteration_np = point_birth_iteration_np[keep_mask_np]
                        densify_position_grad_accum_np = densify_position_grad_accum_np[keep_mask_np]
                        densify_position_grad_denom_np = densify_position_grad_denom_np[keep_mask_np]
                        densify_position_grad_vector_accum_np = densify_position_grad_vector_accum_np[keep_mask_np]
                        densify_scale_grow_accum_np = densify_scale_grow_accum_np[keep_mask_np]
                        densify_scale_shrink_accum_np = densify_scale_shrink_accum_np[keep_mask_np]
                        densify_scale_pressure_denom_np = densify_scale_pressure_denom_np[keep_mask_np]


                    if densification_result is not None:
                        new_block = densification_result.get("new", None)
                        if new_block is not None:
                            n_new = int(new_block["position"].shape[0])

                            add_new_points(renderer, densification_result)
                            # New points are born now; do not allow immediate re-cloning.
                            point_birth_iteration_np = np.concatenate(
                                [point_birth_iteration_np, np.full((n_new,), iteration, dtype=np.int64), ], axis=0, )
                            densify_position_grad_accum_np = np.concatenate(
                                [densify_position_grad_accum_np, np.zeros((n_new, 1), dtype=np.float32), ], axis=0, )
                            densify_position_grad_denom_np = np.concatenate(
                                [densify_position_grad_denom_np, np.zeros((n_new, 1), dtype=np.float32), ], axis=0,
                            )
                            densify_position_grad_vector_accum_np = np.concatenate(
                                [densify_position_grad_vector_accum_np, np.zeros((n_new, 3), dtype=np.float32), ],
                                axis=0,
                            )
                            densify_scale_grow_accum_np = np.concatenate(
                                [
                                    densify_scale_grow_accum_np,
                                    np.zeros((n_new, 1), dtype=np.float32),
                                ],
                                axis=0,
                            )

                            densify_scale_shrink_accum_np = np.concatenate(
                                [
                                    densify_scale_shrink_accum_np,
                                    np.zeros((n_new, 1), dtype=np.float32),
                                ],
                                axis=0,
                            )

                            densify_scale_pressure_denom_np = np.concatenate(
                                [
                                    densify_scale_pressure_denom_np,
                                    np.zeros((n_new, 1), dtype=np.float32),
                                ],
                                axis=0,
                            )

                    rebuild_bvh(renderer)

                    (
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        albedos,
                        opacities,
                        betas,
                        powers,
                    ) = refetch_parameters_as_torch(renderer, device)

                    if point_birth_iteration_np.shape[0] != positions.shape[0]:
                        raise RuntimeError(
                            "point_birth_iteration_np length mismatch after topology change: "
                            f"{point_birth_iteration_np.shape[0]} vs {positions.shape[0]}"
                        )

                    trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)

                    verify_parameters_inplane(
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        albedos,
                        opacities,
                        betas,
                        trainable_surfel_mask=trainable_surfel_mask,
                    )

                    apply_point_parameters(
                        renderer,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        albedos,
                        opacities,
                        betas,
                        powers,
                    )

                    rebuild_bvh(renderer)

                    optimizer = create_masked_optimizer(
                        config,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        albedos,
                        opacities,
                        betas,
                        powers,
                    )

                    trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)

                    frozen_surfel_count = int((~trainable_surfel_mask).sum().item())
                    print(
                        f"Frozen emissive surfels: {frozen_surfel_count} / "
                        f"{int(trainable_surfel_mask.numel())}"
                    )

                    # Reset density-control statistics after every densification attempt,
                    # even if no clone/prune happened.
                if (densify_after <= iteration <= densify_until_iteration
                        and iteration % densification_interval == 0):
                    densify_position_grad_accum_np[:] = 0.0
                    densify_position_grad_denom_np[:] = 0.0
                    densify_position_grad_vector_accum_np[:] = 0.0
                    densify_scale_grow_accum_np[:] = 0.0
                    densify_scale_shrink_accum_np[:] = 0.0
                    densify_scale_pressure_denom_np[:] = 0.0
                # --------------------------------------------------------------
                # 10. Snapshots
                # --------------------------------------------------------------
                if iteration % config.save_interval == 0 or iteration == config.iterations:
                    for camera_name in all_camera_ids:
                        camera_base_dir = config.output_dir / camera_name
                        camera_render_dir = camera_base_dir / "render"
                        camera_grad_dir = camera_base_dir / "grad"
                        camera_depth_dir = camera_base_dir / "depth_distortion"
                        camera_visible_normal_dir = camera_base_dir / "visible_normal"
                        camera_depth_normal_dir = camera_base_dir / "normal_from_depth"
                        camera_median_depth_dir = camera_base_dir / "median_depth"
                        camera_render_dir.mkdir(parents=True, exist_ok=True)
                        camera_grad_dir.mkdir(parents=True, exist_ok=True)

                        image_numpy = get_forward_rgb(forward_out, camera_name)
                        render_path = camera_render_dir / f"{iteration:04d}_render.png"
                        save_render(render_path, image_numpy)

                        if use_depth_distortion:
                            camera_depth_dir.mkdir(parents=True, exist_ok=True)
                            depth_distortion_numpy = get_forward_depth_distortion(forward_out, camera_name)
                            depth_distortion_path = camera_depth_dir / f"{iteration:04d}_depth_distortion.png"
                            save_depth_distortion_snapshot(
                                depth_distortion_path,
                                depth_distortion_numpy,
                                quantile=0.99,
                                save_npy=False,
                            )

                        if use_normal_consistency:
                            camera_median_depth_dir.mkdir(parents=True, exist_ok=True)
                            camera_visible_normal_dir.mkdir(parents=True, exist_ok=True)
                            camera_depth_normal_dir.mkdir(parents=True, exist_ok=True)

                            median_depth_numpy = get_forward_median_depth(forward_out, camera_name)
                            visible_normal_numpy = get_forward_visible_normal(forward_out, camera_name)
                            normal_from_depth_numpy = get_forward_normal_from_depth(forward_out, camera_name)

                            save_median_depth_snapshot(
                                camera_median_depth_dir / f"{iteration:04d}_median_depth.png",
                                median_depth_numpy,
                                quantile=0.99,
                                save_npy=False,
                            )

                            save_normal_map_snapshot(
                                camera_visible_normal_dir / f"{iteration:04d}_visible_normal.png",
                                visible_normal_numpy,
                                save_npy=False,
                            )

                            save_normal_map_snapshot(
                                camera_depth_normal_dir / f"{iteration:04d}_normal_from_depth.png",
                                normal_from_depth_numpy,
                                save_npy=False,
                            )
                        adjoint_source_images = adjoint_images.get("adjoint_source")
                        if adjoint_source_images is not None and camera_name in adjoint_source_images:
                            grad_img_np = np.asarray(
                                adjoint_source_images[camera_name],
                                dtype=np.float32,
                                order="C",
                            )
                            grad_img_np = np.nan_to_num(
                                grad_img_np,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )

                            grad_path = camera_grad_dir / f"{iteration:04d}_grad_099.png"
                            save_gradient_sign_png_py(
                                grad_path,
                                grad_img_np,
                                adjoint_spp=renderer_settings.adjoint_passes,
                                abs_quantile=0.999,
                                flip_y=False,
                            )

                # --------------------------------------------------------------
                # 11. Metrics and logging
                # --------------------------------------------------------------
                num_points = positions.shape[0]
                iteration_end = time.perf_counter()
                iteration_time = iteration_end - iteration_start
                total_time = iteration_end - total_start_time

                current_params_np = {
                    "position": positions.detach().cpu().numpy(),
                    "tangent_u": tangent_u.detach().cpu().numpy(),
                    "tangent_v": tangent_v.detach().cpu().numpy(),
                    "scale": scales.detach().cpu().numpy(),
                    "albedo": albedos.detach().cpu().numpy(),
                    "opacity": opacities.detach().cpu().numpy(),
                    "beta": betas.detach().cpu().numpy(),
                    "power": powers.detach().cpu().numpy(),
                }
                parameter_mse = compute_parameter_mse(
                    current_params_np,
                    initial_params_reference,
                )

                csv_writer.writerow(
                    [
                        iteration,
                        "ALL_CAMERAS",
                        total_rgb_loss_value,
                        total_depth_distortion_loss_raw,
                        total_depth_distortion_loss_weighted,
                        total_normal_loss_raw,
                        total_normal_loss_weighted,
                        opacity_regularizer_loss,
                        total_loss_value,
                        parameter_mse,
                        num_points,
                        iteration_time,
                        total_time,
                        grad_position_renderer_rms,
                        grad_position_renderer_max,
                        grad_position_depth_distortion_rms,
                        grad_position_depth_distortion_max,
                        grad_position_normal_consistency_rms,
                        grad_position_normal_consistency_max,
                        grad_position_total_rms,
                        grad_position_total_max,
                        grad_opacity_total_rms,
                        grad_opacity_total_max,
                        grad_opacity_regularizer_rms,
                        grad_opacity_regularizer_max,
                    ]
                )
                csv_file.flush()

                if iteration % config.log_interval == 0 or iteration == 1:
                    grad_pos_rms = rms_point(grad_position_np)
                    grad_tanu_rms = rms_point(grad_tangent_u_np)
                    grad_tanv_rms = rms_point(grad_tangent_v_np)
                    grad_scale_rms = rms_point(grad_scales_np)
                    grad_albedo_rms = rms_point(grad_albedos_np)
                    grad_opacity_rms = rms_point(grad_opacities_np)
                    grad_beta_rms = rms_point(grad_betas_np)

                    grad_pos_max = max_point_norm(grad_position_np)
                    grad_tanu_max = max_point_norm(grad_tangent_u_np)
                    grad_tanv_max = max_point_norm(grad_tangent_v_np)
                    grad_scale_max = max_point_norm(grad_scales_np)
                    grad_albedo_max = max_point_norm(grad_albedos_np)
                    grad_opacity_max = max_point_norm(grad_opacities_np)
                    grad_beta_max = max_point_norm(grad_betas_np)

                    print(
                        f"[Iter {iteration:04d}/{config.iterations}] "
                        f"RGB={total_rgb_loss_value:.3e}, "
                        f"DdistRaw={total_depth_distortion_loss_raw:.3e}, "
                        f"DdistW={total_depth_distortion_loss_weighted:.3e}, "
                        f"NconsRaw={total_normal_loss_raw:.3e}, "
                        f"NconsW={total_normal_loss_weighted:.3e}, "
                        f"OpacityReg={opacity_regularizer_loss:.3e}, "
                        f"Total={total_loss_value:.3e}, "
                        f"t={iteration_time:.3f} s, "
                        f"pos_rms={grad_pos_rms:.2e}, "
                        f"tu_rms={grad_tanu_rms:.2e}, "
                        f"tv_rms={grad_tanv_rms:.2e}, "
                        f"su,sv_rms={grad_scale_rms:.2e}, "
                        f"rho_rms={grad_albedo_rms:.2e}, "
                        f"eta_rms={grad_opacity_rms:.2e}, "
                        f"beta_rms={grad_beta_rms:.2e}, "
                        f"pos_max={grad_pos_max:.2e}, "
                        f"tu_max={grad_tanu_max:.2e}, "
                        f"tv_max={grad_tanv_max:.2e}, "
                        f"su,sv_max={grad_scale_max:.2e}, "
                        f"rho_max={grad_albedo_max:.2e}, "
                        f"eta_max={grad_opacity_max:.2e}, "
                        f"beta_max={grad_beta_max:.2e}, "
                        f"pts={num_points}, "
                        f"t_total={total_time:.1f} s, "
                        f"it/s={1.0 / iteration_time:.2f}"
                    )

                    print(format_gradient_stats("render_grads", photo_gradient_stats))

                    if distortion_gradients:
                        print(format_gradient_stats("depth_distort_reg ", distortion_gradient_stats))

                    if normal_gradients:
                        print(format_gradient_stats("normal_reg ", normal_gradient_stats))

                    if use_depth_distortion and depth_distortion_maps_for_logging:
                        print(
                            summarize_depth_distortion_maps(
                                depth_distortion_maps_for_logging,
                                depth_distortion_grad_images,
                            )
                        )

                    if use_opacity_loss:
                        print(format_gradient_stats("opacity_reg", opacity_regularizer_gradient_stats))

                    hotkey = poll_hotkey()
                    if hotkey == "s":
                        save_manual_snapshot(
                            renderer,
                            config.output_dir,
                            iteration,
                            positions,
                            tangent_u,
                            tangent_v,
                            scales,
                            albedos,
                            opacities,
                            betas,
                            powers,
                            training_camera_ids,
                        )
                    elif hotkey == "g":
                        save_gradients_snapshot(
                            config.output_dir,
                            iteration,
                            grad_position_np,
                            grad_tangent_u_np,
                            grad_tangent_v_np,
                            grad_scales_np,
                            grad_albedos_np,
                            grad_opacities_np,
                            grad_betas_np,
                        )

        except KeyboardInterrupt:
            elapsed = time.perf_counter() - total_start_time
            print(
                f"\nCtrl+C detected at iteration {iteration:04d}. "
                f"Total elapsed time: {elapsed:.1f} s. "
                "Stopping optimization loop and saving current result..."
            )

    # ------------------------------------------------------------------
    # 12. Final render and export (multi-camera)
    # ------------------------------------------------------------------
    apply_point_parameters(
        renderer,
        positions,
        tangent_u,
        tangent_v,
        scales,
        albedos,
        opacities,
        betas,
        powers
    )

    final_images = renderer.render_forward()

    final_rgb_loss = 0.0
    final_depth_distortion_loss_raw = 0.0
    final_depth_distortion_loss_weighted = 0.0
    final_normal_loss_raw = 0.0
    final_normal_loss_weighted = 0.0
    final_total_loss = 0.0

    for camera_name in training_camera_ids:
        img_np = get_forward_rgb(final_images, camera_name)
        tgt_np = target_images[camera_name]

        rgb_loss_cam = float(compute_l2_loss(img_np, tgt_np))
        final_rgb_loss += rgb_loss_cam
        final_total_loss += rgb_loss_cam

        save_render(Path(config.output_dir / f"render_final_{camera_name}.png"), img_np)

        if use_depth_distortion:
            dist_np = get_forward_depth_distortion(final_images, camera_name)
            dist_loss_cam_raw = float(dist_np.mean())
            dist_loss_cam_weighted = depth_distortion_weight * dist_loss_cam_raw

            final_depth_distortion_loss_raw += dist_loss_cam_raw
            final_depth_distortion_loss_weighted += dist_loss_cam_weighted
            final_total_loss += dist_loss_cam_weighted

            save_depth_distortion_snapshot(
                Path(config.output_dir / f"depth_distortion_final_{camera_name}.png"),
                dist_np,
                quantile=0.99,
                save_npy=False,
            )

        if use_normal_consistency:
            median_depth = get_forward_median_depth(final_images, camera_name)
            visible_normal = get_forward_visible_normal(final_images, camera_name)
            normal_from_depth = get_forward_normal_from_depth(final_images, camera_name)

            normal_loss_cam_raw, _, _ = compute_normal_consistency_loss_and_adjoints(
                visible_normal,
                normal_from_depth,
                1.0,
            )
            normal_loss_cam_weighted = normal_consistency_weight * normal_loss_cam_raw

            final_normal_loss_raw += normal_loss_cam_raw
            final_normal_loss_weighted += normal_loss_cam_weighted
            final_total_loss += normal_loss_cam_weighted

            save_median_depth_snapshot(
                Path(config.output_dir / f"median_depth_final_{camera_name}.png"),
                median_depth,
                save_npy=False,
            )
            save_normal_map_snapshot(
                Path(config.output_dir / f"visible_normal_final_{camera_name}.png"),
                visible_normal,
                save_npy=False,
            )
            save_normal_map_snapshot(
                Path(config.output_dir / f"normal_from_depth_final_{camera_name}.png"),
                normal_from_depth,
                save_npy=False,
            )

    final_opacity_regularizer_loss, _ = (
        compute_opacity_target_regularizer_and_gradients(
            opacities=opacities,
            trainable_surfel_mask=trainable_surfel_mask,
            opacity_target=opacity_target,
            opacity_weight=opacity_loss_weight,
            use_opacity_loss=use_opacity_loss,
        )
    )

    final_total_loss += final_opacity_regularizer_loss

    print(f"Initial RGB loss                        : {initial_rgb_loss:.6e}")
    print(f"Initial depth distortion loss (raw)     : {initial_depth_distortion_loss_raw:.6e}")
    print(f"Initial depth distortion loss (weighted): {initial_depth_distortion_loss_weighted:.6e}")
    print(f"Initial normal consistency loss (raw)   : {initial_normal_loss_raw:.6e}")
    print(f"Initial normal consistency loss (weighted): {initial_normal_loss_weighted:.6e}")
    print(f"Initial opacity regularizer loss         : {initial_opacity_regularizer_loss:.6e}")
    print(f"Initial total loss                      : {initial_total_loss:.6e}")

    print(f"Final RGB loss                          : {final_rgb_loss:.6e}")
    print(f"Final depth distortion loss (raw)       : {final_depth_distortion_loss_raw:.6e}")
    print(f"Final depth distortion loss (weighted)  : {final_depth_distortion_loss_weighted:.6e}")
    print(f"Final normal consistency loss (raw)     : {final_normal_loss_raw:.6e}")
    print(f"Final normal consistency loss (weighted): {final_normal_loss_weighted:.6e}")
    print(f"Final opacity regularizer loss           : {final_opacity_regularizer_loss:.6e}")
    print(f"Final total loss                        : {final_total_loss:.6e}")

    ply_path = config.output_dir / "points_final.ply"
    save_gaussians_to_ply(
        ply_path,
        positions,
        tangent_u,
        tangent_v,
        scales,
        albedos,
        opacities,
        betas,
        powers,
        shape_default=0.0,
    )
    print(f"Final parameters written to PLY: {ply_path}")

    print("\nOptimization completed.")
    print(f"Outputs saved in: {config.output_dir.resolve()}")
    total_elapsed = time.perf_counter() - total_start_time
    print(f"Total optimization wall time: {total_elapsed:.1f} s")
