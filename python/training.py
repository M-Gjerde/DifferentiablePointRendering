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
    compute_prune_indices_by_opacity,
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

def get_forward_depth_distortion(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    depth = np.asarray(
        forward_out[camera_name]["depth_distortion"],
        dtype=np.float32,
        order="C",
    )
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


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


def sum_gradient_dicts(
    photo_gradients: Dict[str, np.ndarray],
    regularizer_gradients: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Sum two renderer-produced gradient dictionaries parameter-wise.
    Assumes both dicts contain the same keys and shapes.
    """
    result: Dict[str, np.ndarray] = {}
    for key in photo_gradients.keys():
        g_photo = np.asarray(photo_gradients[key], dtype=np.float32, order="C")
        g_reg = np.asarray(regularizer_gradients[key], dtype=np.float32, order="C")
        if g_photo.shape != g_reg.shape:
            raise RuntimeError(
                f"Gradient shape mismatch for '{key}': "
                f"photo {g_photo.shape} vs reg {g_reg.shape}"
            )
        result[key] = g_photo + g_reg
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

def save_depth_distortion_snapshot(
        output_path_png: Path,
        depth_distortion: np.ndarray,
        quantile: float = 0.99,
        save_npy: bool = True,
) -> None:
    """
    Save a scalar depth-distortion map as:
      - PNG visualization (grayscale, quantile-normalized)
      - optional raw .npy next to it
    """
    depth = np.asarray(depth_distortion, dtype=np.float32, order="C")
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.maximum(depth, 0.0)

    if depth.size == 0:
        vis = np.zeros((1, 1, 3), dtype=np.float32)
    else:
        vmax = float(np.quantile(depth, quantile))
        if not np.isfinite(vmax) or vmax <= 1e-12:
            vmax = 1.0
        vis_scalar = np.clip(depth / vmax, 0.0, 1.0)
        vis = np.repeat(vis_scalar[..., None], 3, axis=2).astype(np.float32, copy=False)

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
            save_npy=True,
        )

    # Convenience: chosen/main camera
    main_img = get_forward_rgb(current_images, camera_name)
    save_render(checkpoint_dir / "render_final.png", main_img)

    main_depth = get_forward_depth_distortion(current_images, camera_name)
    save_depth_distortion_snapshot(
        checkpoint_dir / "depth_distortion_final.png",
        main_depth,
        quantile=0.99,
        save_npy=True,
    )

    print(f"[Iter {iteration:04d}] Saved checkpoint: {checkpoint_dir}")


def rms_point(x):
    n = max(x.shape[0], 1)
    return float(np.linalg.norm(x) / np.sqrt(n))


def rms_scalar(x):
    return float(np.linalg.norm(x) / np.sqrt(max(x.size, 1)))

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
            save_npy=True,
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
) -> None:
    """
    Enforce parameter constraints in-place:
    - orthonormal tangents
    - valid scales, albedos, and opacities
    """
    verify_tangents_inplace(tangent_u, tangent_v)
    verify_scales_inplace(scales)
    verify_positions_inplace(positions)
    verify_albedos_inplace(albedos)
    verify_opacities_inplace(opacities)
    verify_beta_inplace(betas)


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


# ---------------------------------------------------------------------------
# Main optimization
# ---------------------------------------------------------------------------

def run_optimization(
        renderer: Pale.Renderer,
        config: OptimizationConfig,
        renderer_settings: RendererSettingsConfig,
) -> None:
    # ------------------------------------------------------------------
    # 1a. Load target images per camera
    # ------------------------------------------------------------------
    target_path = Path(config.dataset_path)
    target_images: Dict[str, np.ndarray] = {}

    isColmap = True
    # Multi-camera mode: interpret dataset path as a directory
    isDirectory = target_path.is_dir()
    if not target_path.is_dir():
        raise RuntimeError(
            f"Target path '{target_path}' must be a directory when multiple cameras are used."
        )

    if isColmap:
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

    depth_distortion_weight = config.depth_distort_weight
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

    # Immutable reference for parameter MSE
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

    # (Optionally apply debug noise as before...)
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

    # Create initial trainable tensors from numpy (unchanged)
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
    verify_parameters_inplane(positions, tangent_u, tangent_v, scales, albedos, opacities, betas)

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

    initial_loss = 0.0
    # clear_output_dir(config.output_dir)

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
            print(f"Warning: no target image found for camera '{camera_name}', skipping target save and loss.")
            continue

        tgt_np = target_images[camera_name]

        loss_cam = compute_l2_loss(img_np, tgt_np)
        initial_loss += loss_cam

        save_render(
            config.output_dir / f"render_target_{camera_name}.png",
            tgt_np,
        )

    print(f"Initial loss (L2, summed over cameras): {initial_loss:.6e}")

    # ------------------------------------------------------------------
    # 4. Density control / scheduling hyperparameters
    # ------------------------------------------------------------------
    iteration = 0

    densification_interval = 1e100
    prune_interval = 1e100
    burnin_iterations = 1

    reset_opacity_interval = int(1e10)
    densification_grad_threshold = 1e-9

    opacity_prune_threshold = 0.4
    max_prune_fraction = 0.3
    rebuild_bvh_interval = 1

    metrics_csv_path = config.output_dir / "metrics.csv"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "iteration",
                "camera_name",
                "loss_rgb_sum",
                "loss_depth_distortion_sum",
                "loss_total_sum",
                "parameter_mse",
                "num_points",
                "iteration_time_sec",
                "total_time",
            ]
        )

        # camera_name = camera_ids[0]
        numCameras = len(training_camera_ids)
        total_start_time = time.perf_counter()

        try:
            for iteration in range(1, config.iterations + 1):
                iteration_start = time.perf_counter()

                # --------------------------------------------------------------
                # 5. Forward pass and image-space loss (multi-camera)
                # --------------------------------------------------------------
                # dict[name -> HxWx3]
                forward_out = renderer.render_forward()

                total_rgb_loss_value = 0.0
                total_depth_distortion_value = 0.0
                total_loss_value = 0.0

                loss_grad_images: Dict[str, np.ndarray] = {}
                depth_distortion_grad_images: Dict[str, np.ndarray] = {}
                loss_images: Dict[str, np.ndarray] = {}
                depth_distortion_maps_for_logging: Dict[str, np.ndarray] = {}

                for camera_name in training_camera_ids:
                    current_rgb_np = get_forward_rgb(forward_out, camera_name)
                    current_depth_distortion_np = get_forward_depth_distortion(forward_out, camera_name)
                    target_rgb_np = target_images[camera_name]
                    depth_distortion_maps_for_logging[camera_name] = current_depth_distortion_np

                    # Photometric loss and image-space adjoint
                    rgb_grad = compute_l2_grad(current_rgb_np, target_rgb_np)
                    rgb_loss_value = float(compute_l2_loss(current_rgb_np, target_rgb_np))

                    # Depth distortion loss from renderer-produced map
                    depth_distortion_loss_value = float(current_depth_distortion_np.mean())

                    total_rgb_loss_value += rgb_loss_value
                    total_depth_distortion_value += depth_distortion_loss_value
                    total_loss_value += rgb_loss_value + depth_distortion_weight * depth_distortion_loss_value

                    loss_grad_images[camera_name] = rgb_grad

                    # This is dL / d(distortionBuffer[pixel]) for:
                    # L_dist = depth_distortion_weight * mean(distortionBuffer)
                    depth_distortion_grad_images[camera_name] = make_mean_reduction_adjoint_image(
                        current_depth_distortion_np,
                        depth_distortion_weight,
                    )

                    # For visualization only
                    loss_images[camera_name] = rgb_grad

                # ------------------------------------------------------------------
                # Photometric backward: transport adjoint gradient
                # ------------------------------------------------------------------
                photo_gradients, adjoint_images = renderer.render_backward(loss_grad_images)

                # ------------------------------------------------------------------
                # Regularizer backward: explicit depth-distortion gradient
                # ------------------------------------------------------------------
                distortion_gradients = renderer.render_depth_distortion_backward(
                    depth_distortion_grad_images
                )

                photo_gradient_stats = gradient_stats_from_dict(photo_gradients)
                distortion_gradient_stats = gradient_stats_from_dict(distortion_gradients)

                # ------------------------------------------------------------------
                # Blend the final update vector in Python
                # ------------------------------------------------------------------
                total_gradients = sum_gradient_dicts(photo_gradients, distortion_gradients)

                # Extract numpy gradients
                grad_position_np = np.asarray(total_gradients["position"], dtype=np.float32, order="C")
                grad_tangent_u_np = np.asarray(total_gradients["tangent_u"], dtype=np.float32, order="C")
                grad_tangent_v_np = np.asarray(total_gradients["tangent_v"], dtype=np.float32, order="C")
                grad_scales_np = np.asarray(total_gradients["scale"], dtype=np.float32, order="C")
                grad_albedos_np = np.asarray(total_gradients["albedo"], dtype=np.float32, order="C")
                grad_opacities_np = np.asarray(total_gradients["opacity"], dtype=np.float32, order="C")
                grad_betas_np = np.asarray(total_gradients["beta"], dtype=np.float32, order="C")

                # Sanity check shapes
                current_positions_shape = tuple(positions.shape)
                if grad_position_np.shape != current_positions_shape:
                    raise RuntimeError(
                        f"Gradient shape mismatch for position: expected {current_positions_shape}, "
                        f"got {grad_position_np.shape}"
                    )

                # --------------------------------------------------------------
                # 7. Optimizer step
                # --------------------------------------------------------------
                optimizer.zero_grad(set_to_none=True)

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

                # Reset opacities on schedule (unchanged)
                if iteration % reset_opacity_interval == 0:
                    with torch.no_grad():
                        opacities[:] = 0.1
                    print(f"[Iter {iteration:04d}] Resetting all opacities to 0.1")

                # --------------------------------------------------------------
                # 8. Reparameterization, sync, BVH (unchanged logic)
                # --------------------------------------------------------------
                verify_parameters_inplane(positions, tangent_u, tangent_v, scales, albedos, opacities, betas)

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
                    (
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        albedos,
                        opacities,
                        betas,
                        powers
                    ) = refetch_parameters_as_torch(renderer, device)

                    optimizer = create_masked_optimizer(
                        config,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        albedos,
                        opacities,
                        betas,
                        powers
                    )

                # --------------------------------------------------------------
                # 9. Densification + pruning (unchanged)
                # --------------------------------------------------------------
                densification_result: Optional[Dict[str, np.ndarray]] = None
                indices_to_remove_list: List[int] = []

                if iteration >= burnin_iterations and iteration % prune_interval == 0:
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
                        indices_to_remove = np.unique(
                            np.asarray(indices_to_remove_list, dtype=np.int64)
                        )
                        remove_points(renderer, indices_to_remove)
                        rebuild_bvh(renderer)

                    (
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        albedos,
                        opacities,
                        betas,
                        powers
                    ) = refetch_parameters_as_torch(renderer, device)

                    verify_parameters_inplane(positions, tangent_u, tangent_v, scales, albedos, opacities, betas)
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
                        powers
                    )

                # --------------------------------------------------------------
                # 10. Snapshots (per-camera images)
                # --------------------------------------------------------------
                if iteration % config.save_interval == 0 or iteration == config.iterations:
                    for camera_name in all_camera_ids:
                        camera_base_dir = config.output_dir / camera_name
                        camera_render_dir = camera_base_dir / "render"
                        camera_grad_dir = camera_base_dir / "grad"
                        camera_depth_dir = camera_base_dir / "depth_distortion"

                        camera_render_dir.mkdir(parents=True, exist_ok=True)
                        camera_grad_dir.mkdir(parents=True, exist_ok=True)
                        camera_depth_dir.mkdir(parents=True, exist_ok=True)

                        image_numpy = get_forward_rgb(forward_out, camera_name)
                        render_path = camera_render_dir / f"{iteration:04d}_render.png"
                        save_render(render_path, image_numpy)

                        depth_distortion_numpy = get_forward_depth_distortion(forward_out, camera_name)
                        depth_distortion_path = camera_depth_dir / f"{iteration:04d}_depth_distortion.png"
                        save_depth_distortion_snapshot(
                            depth_distortion_path,
                            depth_distortion_numpy,
                            quantile=0.99,
                            save_npy=True,
                        )

                        adjointSourceImages = adjoint_images.get("adjoint_source")
                        if adjointSourceImages is not None and camera_name in adjointSourceImages:
                            grad_img_np = np.asarray(
                                adjointSourceImages[camera_name],
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

                            main_loss_image = loss_images[camera_name]
                            main_camera_loss_root = config.output_dir / camera_name
                            save_loss_image(main_camera_loss_root, main_loss_image, iteration)
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
                        total_depth_distortion_value,
                        total_loss_value,
                        parameter_mse,
                        num_points,
                        iteration_time,
                        total_time,
                    ]
                )
                csv_file.flush()

                if iteration % config.log_interval == 0 or iteration == 1:
                    def rms_point(x: np.ndarray) -> float:
                        n = max(x.shape[0], 1)
                        return float(np.linalg.norm(x) / np.sqrt(n))

                    def max_point_norm(x: np.ndarray) -> float:
                        if x.ndim == 1:
                            return float(np.max(np.abs(x))) if x.size > 0 else 0.0
                        return float(np.max(np.linalg.norm(x, axis=1))) if x.shape[0] > 0 else 0.0

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
                        f"Ddist={total_depth_distortion_value:.3e}, "
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

                    #print(format_gradient_stats("photo_grads", photo_gradient_stats))
                    #print(format_gradient_stats("dist_grads ", distortion_gradient_stats))
                    #print(
                    #    summarize_depth_distortion_maps(
                    #        depth_distortion_maps_for_logging,
                    #        depth_distortion_grad_images,
                    #    )
                    #)

                    # Hotkey snapshot: use main camera for the image
                    # --------------------------------------------------------------
                    # 12. Hotkeys (snapshot + gradient dump)
                    # --------------------------------------------------------------
                    hotkey = poll_hotkey()
                    if hotkey == "s":
                        # Save current state (render + PLY) using main camera
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
                        # Save gradients for all points
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
    final_depth_distortion_loss = 0.0
    final_total_loss = 0.0

    for camera_name in training_camera_ids:
        img_np = get_forward_rgb(final_images, camera_name)
        dist_np = get_forward_depth_distortion(final_images, camera_name)
        tgt_np = target_images[camera_name]

        rgb_loss_cam = float(compute_l2_loss(img_np, tgt_np))
        dist_loss_cam = float(dist_np.mean())

        final_rgb_loss += rgb_loss_cam
        final_depth_distortion_loss += dist_loss_cam
        final_total_loss += rgb_loss_cam + depth_distortion_weight * dist_loss_cam

        save_render(Path(config.output_dir / f"render_final_{camera_name}.png"), img_np)
        save_depth_distortion_snapshot(
            Path(config.output_dir / f"depth_distortion_final_{camera_name}.png"),
            dist_np,
            quantile=0.99,
            save_npy=True,
        )
    print(f"Initial RGB loss   (sum over cameras): {initial_loss:.6e}")
    print(f"Final RGB loss     (sum over cameras): {final_rgb_loss:.6e}")
    print(f"Final depth dist.  (sum over cameras): {final_depth_distortion_loss:.6e}")
    print(f"Final total loss   (sum over cameras): {final_total_loss:.6e}")

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
    print(f"Initial loss (sum over cameras): {initial_loss:.6e}")
    print(f"Final loss   (sum over cameras): {final_total_loss:.6e}")
    print(f"Final rgb loss   (sum over cameras): {final_rgb_loss:.6e}")
    print(f"Final depth loss   (sum over cameras): {final_depth_distortion_loss:.6e}")
    print(f"Outputs saved in: {config.output_dir.resolve()}")
    total_elapsed = time.perf_counter() - total_start_time
    print(f"Total optimization wall time: {total_elapsed:.1f} s")
