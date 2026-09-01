from __future__ import annotations

import copy
import csv
import math
import select
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

from config import OptimizationConfig, RendererSettingsConfig
from density_control import (
    compute_prune_indices_by_degenerate_area,
    compute_prune_indices_by_opacity,
    make_under_reconstruction_clones,
    normalize_quaternions_torch,
    quaternion_to_tangent_frame_torch
)
from io_utils import (
    SUPPORTED_TARGET_IMAGE_SUFFIXES,
    linear_to_srgb,
    load_target_image,
    save_gaussians_to_ply,
    save_gradient_sign_png_py,
    save_render,
)
from losses import (
    compute_l2_grad,
    compute_l2_ssim_loss_and_grad,
    compute_l2_ssim_metrics,
)
from optimizers import create_masked_optimizer
from render_hooks import *

PARAMETER_NAMES = (
    "position",
    "rotation",
    "scale",
    "albedo",
    "opacity",
    "beta",
    "power",
)

LOSS_VALUE_KEYS = (
    "total_rgb_loss_value",
    "total_rgb_l2_loss_value",
    "total_rgb_dssim_loss_value",
    "total_depth_distortion_loss_raw",
    "total_depth_distortion_loss_weighted",
    "total_normal_loss_raw",
    "total_normal_loss_weighted",
    "total_opacity_prior_loss_raw",
    "total_opacity_prior_loss_weighted",
    "total_intra_slab_depth_loss_raw",
    "total_intra_slab_depth_loss_weighted",
    "total_curvature_scale_loss_raw",
    "total_curvature_scale_loss_weighted",
    "total_loss_value",
)


def make_zero_loss_values() -> Dict[str, float]:
    return {loss_key: 0.0 for loss_key in LOSS_VALUE_KEYS}


def make_averaged_loss_state_from_camera_cache(
        latest_loss_values_by_camera: Dict[str, Dict[str, float]],
        expected_camera_ids: List[str],
) -> Dict[str, Any]:
    available_camera_ids = [
        camera_name
        for camera_name in expected_camera_ids
        if camera_name in latest_loss_values_by_camera
    ]

    if not available_camera_ids:
        raise RuntimeError("Cannot compute averaged loss: no camera losses have been recorded.")

    averaged_loss_state: Dict[str, Any] = {}

    for loss_key in LOSS_VALUE_KEYS:
        averaged_loss_state[loss_key] = float(np.mean([
            latest_loss_values_by_camera[camera_name][loss_key]
            for camera_name in available_camera_ids
        ]))

    averaged_loss_state["loss_metric_camera_count"] = len(available_camera_ids)
    averaged_loss_state["loss_metric_expected_camera_count"] = len(expected_camera_ids)
    averaged_loss_state["loss_metric_is_complete"] = (
            len(available_camera_ids) == len(expected_camera_ids)
    )

    return averaged_loss_state


def select_active_training_camera_ids(
        training_camera_ids: list[str],
        iteration: int,
        config: OptimizationConfig,
) -> list[str]:
    if not training_camera_ids:
        raise RuntimeError("No training cameras available.")

    if not config.one_camera_per_iteration:
        return training_camera_ids

    if config.camera_sampling_mode == "round_robin":
        camera_index = (iteration - 1) % len(training_camera_ids)
        return [training_camera_ids[camera_index]]

    if config.camera_sampling_mode == "random":
        rng = np.random.default_rng(config.camera_sampling_seed + iteration)
        camera_index = int(rng.integers(0, len(training_camera_ids)))
        return [training_camera_ids[camera_index]]

    raise RuntimeError(f"Unknown camera_sampling_mode: {config.camera_sampling_mode}")


def as_config_float(value: Any) -> float:
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError(f"Expected scalar or single-item tuple, got {value}")
        value = value[0]
    return float(value)


def make_named_parameter_dict(
        positions: torch.nn.Parameter,
        rotation_delta: torch.nn.Parameter,
        scales: torch.nn.Parameter,
        albedos: torch.nn.Parameter,
        opacities: torch.nn.Parameter,
        betas: torch.nn.Parameter,
        powers: torch.nn.Parameter,
) -> Dict[str, torch.nn.Parameter]:
    return {
        "position": positions,
        "rotation": rotation_delta,
        "scale": scales,
        "albedo": albedos,
        "opacity": opacities,
        "beta": betas,
        "power": powers,
    }


def repair_nonfinite_gradient_array_inplace(
        name: str,
        array: np.ndarray,
        iteration: int,
        zero_entire_row: bool = True,
) -> int:
    if array is None:
        return 0

    gradient_array = np.asarray(array)
    finite_mask = np.isfinite(gradient_array)
    if bool(np.all(finite_mask)):
        return 0

    bad_mask = ~finite_mask
    bad_count = int(np.count_nonzero(bad_mask))
    first_bad_flat_index = int(np.flatnonzero(bad_mask)[0])
    first_bad_index = np.unravel_index(first_bad_flat_index, gradient_array.shape)
    first_bad_value = gradient_array[first_bad_index]

    if zero_entire_row and gradient_array.ndim >= 2:
        bad_rows = np.any(bad_mask.reshape(gradient_array.shape[0], -1), axis=1)
        repaired_count = int(np.count_nonzero(bad_rows))
        gradient_array[bad_rows] = 0.0
        repair_text = f"zeroed_rows={repaired_count}"
    else:
        gradient_array[bad_mask] = 0.0
        repair_text = f"zeroed_values={bad_count}"

    print(
        f"[Iter {iteration:04d}] Repaired non-finite gradient in {name}: "
        f"bad_count={bad_count}, first_bad_index={first_bad_index}, "
        f"first_bad_value={first_bad_value}, {repair_text}"
    )
    return bad_count


def repair_nonfinite_gradient_dict_inplace(
        tag: str,
        gradient_dict: Dict[str, np.ndarray],
        iteration: int,
) -> int:
    total_bad_count = 0
    for gradient_name, gradient_array in gradient_dict.items():
        total_bad_count += repair_nonfinite_gradient_array_inplace(
            name=f"{tag}[{gradient_name}]",
            array=gradient_array,
            iteration=iteration,
            zero_entire_row=True,
        )
    return total_bad_count


def apply_densification_source_updates_inplace(
        densification_result: Optional[Dict[str, Any]],
        positions: torch.nn.Parameter,
        rotations: torch.Tensor,
        scales: torch.nn.Parameter,
        albedos: torch.nn.Parameter,
        opacities: torch.nn.Parameter,
        betas: torch.nn.Parameter,
        powers: torch.nn.Parameter,
) -> None:
    if densification_result is None:
        return
    update = densification_result.get("update_source", None)
    if update is None:
        return
    device = positions.device
    idx = torch.as_tensor(update["index"], device=device, dtype=torch.long).reshape(-1)
    if idx.numel() == 0:
        return
    with torch.no_grad():
        if "position" in update:
            v = torch.as_tensor(update["position"], device=device, dtype=positions.dtype)
            positions.data[idx] = v.view_as(positions.data[idx])
        if "rotation" in update:
            v = torch.as_tensor(update["rotation"], device=device, dtype=rotations.dtype)
            rotations.data[idx] = normalize_quaternions_torch(v.view_as(rotations.data[idx]))
        if "scale" in update:
            v = torch.as_tensor(update["scale"], device=device, dtype=scales.dtype)
            scales.data[idx] = v.view_as(scales.data[idx])
        if "albedo" in update:
            v = torch.as_tensor(update["albedo"], device=device, dtype=albedos.dtype)
            albedos.data[idx] = v.view_as(albedos.data[idx])
        if "opacity" in update:
            v = torch.as_tensor(update["opacity"], device=device, dtype=opacities.dtype)
            opacities.data[idx] = v.view_as(opacities.data[idx])
        if "beta" in update:
            v = torch.as_tensor(update["beta"], device=device, dtype=betas.dtype)
            betas.data[idx] = v.view_as(betas.data[idx])
        if "power" in update:
            v = torch.as_tensor(update["power"], device=device, dtype=powers.dtype)
            powers.data[idx] = v.view_as(powers.data[idx])


def rebuild_optimizer_preserving_state(
        config: OptimizationConfig,
        old_optimizer: torch.optim.Optimizer,
        old_params: Dict[str, torch.nn.Parameter],
        new_params: Dict[str, torch.nn.Parameter],
        keep_mask_np: np.ndarray,
        source_index_for_new_np: Optional[np.ndarray] = None,
        copy_source_state_to_new: bool = True,
) -> torch.optim.Optimizer:
    new_optimizer = create_masked_optimizer(
        config,
        new_params["position"],
        new_params["rotation"],
        new_params["scale"],
        new_params["albedo"],
        new_params["opacity"],
        new_params["beta"],
        new_params["power"],
    )

    keep_mask_np = np.asarray(keep_mask_np, dtype=bool).reshape(-1)
    old_n = int(keep_mask_np.shape[0])
    keep_idx_np = np.nonzero(keep_mask_np)[0].astype(np.int64)
    kept_n = int(keep_idx_np.shape[0])

    new_n = int(new_params["position"].shape[0])
    n_new = new_n - kept_n
    if n_new < 0:
        raise RuntimeError(f"Invalid optimizer migration sizes: new_n={new_n}, kept_n={kept_n}")

    if source_index_for_new_np is not None:
        source_index_for_new_np = np.asarray(source_index_for_new_np, dtype=np.int64).reshape(-1)
        if source_index_for_new_np.shape[0] != n_new:
            source_index_for_new_np = None

    for name in PARAMETER_NAMES:
        old_p = old_params[name]
        new_p = new_params[name]
        old_state = old_optimizer.state.get(old_p, None)
        if not old_state:
            continue

        new_state = new_optimizer.state[new_p]
        for key, value in old_state.items():
            if not torch.is_tensor(value):
                new_state[key] = copy.deepcopy(value)
                continue

            if value.ndim >= 1 and value.shape[0] == old_n:
                out = torch.zeros_like(new_p.data)

                keep_idx_t = torch.as_tensor(keep_idx_np, device=value.device, dtype=torch.long)
                kept_value = value.index_select(0, keep_idx_t)
                out[:kept_n] = kept_value.to(device=out.device, dtype=out.dtype)

                if copy_source_state_to_new and source_index_for_new_np is not None and n_new > 0:
                    src_idx_t = torch.as_tensor(source_index_for_new_np, device=value.device, dtype=torch.long)
                    new_value = value.index_select(0, src_idx_t)
                    out[kept_n:kept_n + n_new] = new_value.to(device=out.device, dtype=out.dtype)

                new_state[key] = out
            else:
                new_state[key] = value.detach().clone().to(new_p.device)

    return new_optimizer


DEVICE_ADAM_STATE_ARRAY_KEYS = (
    "position_m",
    "position_v",
    "rotation_m",
    "rotation_v",
    "scale_m",
    "scale_v",
    "albedo_m",
    "albedo_v",
    "opacity_m",
    "opacity_v",
    "beta_m",
    "beta_v",
)


def migrate_device_adam_state_snapshot(
        snapshot: Optional[Dict[str, Any]],
        keep_mask_np: np.ndarray,
        new_point_count: int,
        source_index_for_new_np: Optional[np.ndarray] = None,
        copy_source_state_to_new: bool = False,
) -> Optional[Dict[str, Any]]:
    if not snapshot:
        return None

    old_n = int(snapshot.get("point_count", 0))
    if old_n <= 0:
        return None

    keep_mask_np = np.asarray(keep_mask_np, dtype=bool).reshape(-1)
    if keep_mask_np.shape[0] != old_n:
        raise RuntimeError(
            "Device Adam migration keep-mask size mismatch: "
            f"{keep_mask_np.shape[0]} vs {old_n}"
        )

    keep_idx_np = np.nonzero(keep_mask_np)[0].astype(np.int64)
    kept_n = int(keep_idx_np.shape[0])
    new_n = int(new_point_count)
    n_new = new_n - kept_n
    if n_new < 0:
        raise RuntimeError(f"Invalid device Adam migration sizes: new_n={new_n}, kept_n={kept_n}")

    if source_index_for_new_np is not None:
        source_index_for_new_np = np.asarray(source_index_for_new_np, dtype=np.int64).reshape(-1)
        if source_index_for_new_np.shape[0] != n_new:
            source_index_for_new_np = None

    migrated: Dict[str, Any] = {
        "point_count": new_n,
        "step": int(snapshot.get("step", 0)),
    }

    for key in DEVICE_ADAM_STATE_ARRAY_KEYS:
        if key not in snapshot:
            raise RuntimeError(f"Device Adam snapshot is missing '{key}'")

        values = np.asarray(snapshot[key], dtype=np.float32, order="C")
        if values.ndim < 1 or values.shape[0] != old_n:
            raise RuntimeError(
                f"Device Adam snapshot '{key}' has incompatible shape: "
                f"{values.shape}, expected first dimension {old_n}"
            )

        out = np.zeros((new_n,) + tuple(values.shape[1:]), dtype=np.float32)
        out[:kept_n] = values[keep_idx_np]

        if copy_source_state_to_new and source_index_for_new_np is not None and n_new > 0:
            out[kept_n:kept_n + n_new] = values[source_index_for_new_np]

        migrated[key] = out

    return migrated


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


def save_depth_distortion_snapshot(
        output_path_png: Path,
        depth_distortion: np.ndarray,
        quantile: float = 0.99,
        save_npy: bool = False,
        cmap: str = "inferno",
) -> None:
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
        vis = matplotlib.colormaps[cmap](vis_scalar)[..., :3].astype(np.float32, copy=False)

    save_render(output_path_png, vis)


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

        vis = matplotlib.colormaps[cmap](vis_scalar)[..., :3].astype(np.float32, copy=False)
        vis[~valid] = 0.0

    save_render(output_path_png, vis)


def make_trainable_surfel_mask_from_powers(
        powers: torch.Tensor,
        eps: float = 0.0,
) -> torch.Tensor:
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
        grad_rotation_np: np.ndarray,
        grad_scales_np: np.ndarray,
        grad_albedos_np: np.ndarray,
        grad_opacities_np: np.ndarray,
        grad_betas_np: np.ndarray,
) -> None:
    trainable_mask_np = trainable_mask.detach().cpu().numpy().astype(bool)
    frozen_mask_np = ~trainable_mask_np

    grad_position_np[frozen_mask_np] = 0.0
    grad_rotation_np[frozen_mask_np] = 0.0
    grad_scales_np[frozen_mask_np] = 0.0
    grad_albedos_np[frozen_mask_np] = 0.0
    grad_opacities_np[frozen_mask_np] = 0.0
    grad_betas_np[frozen_mask_np] = 0.0


def make_mean_reduction_adjoint_image(
        image_2d: np.ndarray,
        loss_weight: float,
) -> np.ndarray:
    pixel_count = max(image_2d.size, 1)
    return np.full(image_2d.shape, loss_weight / float(pixel_count), dtype=np.float32)


def sum_gradient_dicts(*gradient_dicts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    active = [gradient_dict for gradient_dict in gradient_dicts if gradient_dict]
    if not active:
        return {}

    result: Dict[str, np.ndarray] = {
        key: np.asarray(value, dtype=np.float32, order="C").copy()
        for key, value in active[0].items()
    }

    for gradient_dict in active[1:]:
        if set(gradient_dict.keys()) != set(result.keys()):
            raise RuntimeError(f"Gradient key mismatch: {set(result.keys())} vs {set(gradient_dict.keys())}")

        for key, value in gradient_dict.items():
            gradient_array = np.asarray(value, dtype=np.float32, order="C")
            if result[key].shape != gradient_array.shape:
                raise RuntimeError(
                    f"Gradient shape mismatch for '{key}': "
                    f"{result[key].shape} vs {gradient_array.shape}"
                )
            result[key] += gradient_array

    return result


def poll_hotkey() -> Optional[str]:
    if not sys.stdin.isatty():
        return None

    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None

    line = sys.stdin.readline().strip().lower()
    if line in ("s", "g"):
        return line
    return None


def gradient_l2_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.linalg.norm(x.ravel()))


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


def rms_point(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.linalg.norm(x) / np.sqrt(max(x.shape[0], 1)))


def position_gradient_norm_stats_or_zero(
        gradient_dict: Dict[str, np.ndarray],
) -> tuple[float, float]:
    if not gradient_dict or "position" not in gradient_dict:
        return 0.0, 0.0

    grad_position = np.asarray(gradient_dict["position"], dtype=np.float32, order="C")
    return gradient_l2_norm(grad_position), max_point_norm(grad_position)


def gradient_stats_from_dict(gradient_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}

    for key, value in gradient_dict.items():
        gradient_array = np.asarray(value, dtype=np.float32, order="C")
        stats[key] = {
            "rms": rms_any(gradient_array),
            "max": max_point_norm(gradient_array),
        }

    return stats


def format_gradient_stats(tag: str, stats: Dict[str, Dict[str, float]]) -> str:
    def format_stat(name: str) -> str:
        if name not in stats:
            return f"{name}=NA"
        return f"{name}_rms={stats[name]['rms']:.2e}, {name}_max={stats[name]['max']:.2e}"

    return (
        f"{tag}: "
        f"{format_stat('position')}, "
        f"{format_stat('rotation')}, "
        f"{format_stat('scale')}, "
        f"{format_stat('albedo')}, "
        f"{format_stat('opacity')}, "
        f"{format_stat('beta')}"
    )


def zero_optimizer_state_for_parameter(optimizer, parameter: torch.Tensor) -> None:
    optimizer_state = optimizer.state.get(parameter, None)
    if not optimizer_state:
        return

    for state_value in optimizer_state.values():
        if torch.is_tensor(state_value):
            state_value.zero_()


def scale_gradient_dict(gradient_dict: Dict[str, np.ndarray], scale: float) -> Dict[str, np.ndarray]:
    return {name: gradient * scale for name, gradient in gradient_dict.items()}


def scheduled_regularizer_weight(
        base_weight: float,
        iteration: int,
        start_iteration: int,
) -> float:
    if base_weight == 0.0:
        return 0.0

    if start_iteration <= 0:
        return float(base_weight)

    return float(base_weight) if int(iteration) >= int(start_iteration) else 0.0


def scheduled_densification_grad_abs_min(
        initial_threshold: float,
        final_threshold: float,
        iteration: int,
        start_iteration: int,
        end_iteration: int,
        decay_power: float = 2.0,
) -> float:
    initial_threshold = float(initial_threshold)
    final_threshold = float(final_threshold)
    start_iteration = int(start_iteration)
    end_iteration = int(end_iteration)

    if end_iteration <= start_iteration:
        return final_threshold if int(iteration) >= start_iteration else initial_threshold

    if int(iteration) <= start_iteration:
        return initial_threshold

    if int(iteration) >= end_iteration:
        return final_threshold

    t = float(int(iteration) - start_iteration) / float(end_iteration - start_iteration)

    return final_threshold + (initial_threshold - final_threshold) * (1.0 - t) ** decay_power

def densification_scene_extent_for_positions(
        config: OptimizationConfig,
        positions,
        trainable_surfel_mask: Optional[torch.Tensor] = None,
) -> float:
    scene_extent = float(getattr(config, "densification_scene_extent", 0.0))
    if scene_extent > 0.0:
        return scene_extent

    positions_np = positions.detach().cpu().numpy()
    if trainable_surfel_mask is not None and positions_np.size > 0:
        trainable_np = trainable_surfel_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        if trainable_np.shape[0] != positions_np.shape[0]:
            raise RuntimeError(
                "Trainable surfel mask length mismatch for densification scene extent: "
                f"{trainable_np.shape[0]} vs {positions_np.shape[0]}"
            )
        trainable_positions_np = positions_np[trainable_np]
        if trainable_positions_np.size > 0:
            positions_np = trainable_positions_np

    if positions_np.size > 0:
        scene_extent = float(np.max(np.ptp(positions_np, axis=0)))
    else:
        scene_extent = 1.0

    return max(scene_extent, 1.0e-6)


def exact_clone_scale_threshold_for_positions(
        config: OptimizationConfig,
        positions,
        trainable_surfel_mask: Optional[torch.Tensor] = None,
) -> float:
    exact_clone_percent_dense = float(
        getattr(config, "densification_exact_clone_percent_dense", 0.0)
    )
    if exact_clone_percent_dense <= 0.0:
        return 0.0

    return exact_clone_percent_dense * densification_scene_extent_for_positions(
        config=config,
        positions=positions,
        trainable_surfel_mask=trainable_surfel_mask,
    )


def minimum_splittable_scale_for_config(config: OptimizationConfig) -> float:
    split_scale_factor = max(
        float(getattr(config, "densification_split_scale_factor", math.sqrt(2.0))),
        1.0,
    )
    min_clone_scale = float(getattr(config, "densification_scale_min", 8.0e-3))
    return min_clone_scale * split_scale_factor * (1.0 + 1.0e-4)


def make_loss_breakdown(loss_state: Dict[str, Any]) -> Dict[str, float]:
    rgb_loss = float(loss_state["total_rgb_loss_value"])
    depth_weighted = float(loss_state["total_depth_distortion_loss_weighted"])
    normal_weighted = float(loss_state["total_normal_loss_weighted"])
    opacity_weighted = float(loss_state["total_opacity_prior_loss_weighted"])
    intra_slab_weighted = float(loss_state["total_intra_slab_depth_loss_weighted"])
    curvature_scale_weighted = float(loss_state["total_curvature_scale_loss_weighted"])

    after_depth = rgb_loss + depth_weighted
    after_normal = after_depth + normal_weighted
    after_opacity = after_normal + opacity_weighted
    after_intra_slab = after_opacity + intra_slab_weighted
    after_curvature_scale = after_intra_slab + curvature_scale_weighted
    regularizer_total = (
        depth_weighted + normal_weighted + opacity_weighted +
        intra_slab_weighted + curvature_scale_weighted
    )

    return {
        "before_regularizers": rgb_loss,
        "after_depth_distortion": after_depth,
        "after_normal_consistency": after_normal,
        "after_opacity_prior": after_opacity,
        "after_intra_slab_depth": after_intra_slab,
        "after_curvature_scale": after_curvature_scale,
        "regularizer_total": regularizer_total,
        "total": float(loss_state["total_loss_value"]),
    }


def format_loss_breakdown(loss_state: Dict[str, Any]) -> str:
    rgb_loss = float(loss_state["total_rgb_loss_value"])
    depth_weighted = float(loss_state["total_depth_distortion_loss_weighted"])
    normal_weighted = float(loss_state["total_normal_loss_weighted"])
    opacity_weighted = float(loss_state["total_opacity_prior_loss_weighted"])
    intra_slab_weighted = float(loss_state["total_intra_slab_depth_loss_weighted"])
    curvature_scale_weighted = float(loss_state["total_curvature_scale_loss_weighted"])
    total_loss = float(loss_state["total_loss_value"])

    after_depth = rgb_loss + depth_weighted
    after_normal = after_depth + normal_weighted
    after_opacity = after_normal + opacity_weighted
    after_intra_slab = after_opacity + intra_slab_weighted
    after_curvature_scale = after_intra_slab + curvature_scale_weighted
    regularizer_total = (
        depth_weighted + normal_weighted + opacity_weighted +
        intra_slab_weighted + curvature_scale_weighted
    )
    loss_camera_count = int(loss_state.get("loss_metric_camera_count", 1))
    loss_camera_expected_count = int(loss_state.get("loss_metric_expected_camera_count", 1))

    return (
        f"Mean loss stack [{loss_camera_count}/{loss_camera_expected_count} cameras]:\n"

        "Loss stack:\n"
        f"  {'RGB only':<28} {rgb_loss:>12.3e}\n"
        f"  {'+ depth distortion':<28} {after_depth:>12.3e}  "
        f"(+{depth_weighted:.3e})\n"
        f"  {'+ normal consistency':<28} {after_normal:>12.3e}  "
        f"(+{normal_weighted:.3e})\n"
        f"  {'+ opacity prior':<28} {after_opacity:>12.3e}  "
        f"(+{opacity_weighted:.3e})\n"
        f"  {'+ intra-slab depth':<28} {after_intra_slab:>12.3e}  "
        f"(+{intra_slab_weighted:.3e})\n"
        f"  {'+ curvature scale':<28} {after_curvature_scale:>12.3e}  "
        f"(+{curvature_scale_weighted:.3e})\n"
        f"  {'regularizer total':<28} {regularizer_total:>12.3e}\n"
        f"  {'total':<28} {total_loss:>12.3e}"
    )


def gradient_norm_for_key(
        gradient_dict: Dict[str, np.ndarray],
        key: str,
) -> float:
    if not gradient_dict or key not in gradient_dict:
        return 0.0
    return gradient_l2_norm(np.asarray(gradient_dict[key], dtype=np.float32, order="C"))


def format_training_iteration_log(
        iteration: int,
        total_iterations: int,
        iteration_time: float,
        iteration_rate: float,
        total_time: float,
        num_points: int,
        loss_state: Dict[str, Any],
        lr_position: float,
        global_lr_scale: float,
        position_lr_scale: float,
        active_densification_interval: int,
        active_prune_interval: int,
        active_densification_grad_abs_min: float,
        active_depth_distortion_weight: float,
        active_normal_consistency_weight: float,
        active_opacity_prior_weight: float,
        exact_clone_scale_threshold: float,
        minimum_splittable_scale: float,
        grad_pos_rms: float,
        grad_rotation_rms: float,
        grad_scale_rms: float,
        grad_albedo_rms: float,
        grad_opacity_rms: float,
        grad_beta_rms: float,
        grad_pos_max: float,
        grad_rotation_max: float,
        grad_scale_max: float,
        grad_albedo_max: float,
        grad_opacity_max: float,
        grad_beta_max: float,
) -> str:
    loss_camera_count = int(loss_state.get("loss_metric_camera_count", 1))
    loss_camera_expected_count = int(loss_state.get("loss_metric_expected_camera_count", 1))

    return (
        f"\n[Iter {iteration:04d}/{total_iterations}] "
        f"time={iteration_time:.3f}s total={total_time:.1f}s "
        f"it/s={iteration_rate:.2f} pts={num_points} "
        f"adaptive_lr_pos={lr_position:.3e} "
        f"lr_global_scale={global_lr_scale:.3e} "
        f"lr_pos_scale={position_lr_scale:.3e} "
        f"densify_interval={active_densification_interval} "
        f"prune_interval={active_prune_interval} "
        f"densify_thr={active_densification_grad_abs_min:.3e} "
        f"depth_active_w={active_depth_distortion_weight:.3e} "
        f"normal_active_w={active_normal_consistency_weight:.3e} "
        f"clone_only_max_scale={exact_clone_scale_threshold:.3e} "
        f"split_min_scale={minimum_splittable_scale:.3e}\n"
        f"  losses_mean[{loss_camera_count}/{loss_camera_expected_count} cameras]:"
        f" rgb={loss_state['total_rgb_loss_value']:.3e}"
        f" rgb_l2={loss_state['total_rgb_l2_loss_value']:.3e}"
        f" dssim={loss_state['total_rgb_dssim_loss_value']:.3e}"
        f" depth_raw={loss_state['total_depth_distortion_loss_raw']:.3e}"
        f" depth_w={loss_state['total_depth_distortion_loss_weighted']:.3e}"
        f" normal_raw={loss_state['total_normal_loss_raw']:.3e}"
        f" normal_w={loss_state['total_normal_loss_weighted']:.3e}"
        f" opacity_raw={loss_state['total_opacity_prior_loss_raw']:.3e}"
        f" opacity_w={loss_state['total_opacity_prior_loss_weighted']:.3e}"
        f" intra_slab_raw={loss_state['total_intra_slab_depth_loss_raw']:.3e}"
        f" intra_slab_w={loss_state['total_intra_slab_depth_loss_weighted']:.3e}"
        f" curvature_scale_raw={loss_state['total_curvature_scale_loss_raw']:.3e}"
        f" curvature_scale_w={loss_state['total_curvature_scale_loss_weighted']:.3e}"
        f" opacity_active_w={active_opacity_prior_weight:.3e}"
        f" total={loss_state['total_loss_value']:.3e}\n"
        f"  grad_rms:"
        f" pos={grad_pos_rms:.2e}"
        f" rot={grad_rotation_rms:.2e}"
        f" scale={grad_scale_rms:.2e}"
        f" albedo={grad_albedo_rms:.2e}"
        f" opacity={grad_opacity_rms:.2e}"
        f" beta={grad_beta_rms:.2e}\n"
        f"  grad_max:"
        f" pos={grad_pos_max:.2e}"
        f" rot={grad_rotation_max:.2e}"
        f" scale={grad_scale_max:.2e}"
        f" albedo={grad_albedo_max:.2e}"
        f" opacity={grad_opacity_max:.2e}"
        f" beta={grad_beta_max:.2e}"
    )


def format_gradient_source_balance(
        loss_gradients: Dict[str, np.ndarray],
        depth_regularizer_gradients: Dict[str, np.ndarray],
        normal_regularizer_gradients: Dict[str, np.ndarray],
        opacity_prior_gradients: Dict[str, np.ndarray],
        intra_slab_depth_gradients: Dict[str, np.ndarray],
        curvature_scale_gradients: Dict[str, np.ndarray],
        surface_regularizer_gradients: Dict[str, np.ndarray],
        total_gradients: Dict[str, np.ndarray],
) -> str:
    keys = [
        ("position", "pos"),
        ("rotation", "rot"),
        ("scale", "scale"),
        ("albedo", "albedo"),
        ("opacity", "opacity"),
        ("beta", "beta"),
    ]

    lines = [
        "Gradient source balance:",
        "  "
        f"{'param':<8}"
        f"{'loss_grad':>11}"
        f"{'reg_grad':>11}"
        f"{'total_grad':>11}"
        f"{'loss%':>8}"
        f"{'depth%':>8}"
        f"{'normal%':>9}"
        f"{'opacity%':>10}"
        f"{'intra%':>9}"
        f"{'curv%':>8}"
        f"   {'source norms'}",
    ]

    for key, label in keys:
        loss_norm = gradient_norm_for_key(loss_gradients, key)

        depth_norm = gradient_norm_for_key(depth_regularizer_gradients, key)
        normal_norm = gradient_norm_for_key(normal_regularizer_gradients, key)
        opacity_norm = gradient_norm_for_key(opacity_prior_gradients, key)
        intra_slab_norm = gradient_norm_for_key(intra_slab_depth_gradients, key)
        curvature_scale_norm = gradient_norm_for_key(curvature_scale_gradients, key)

        surface_regularizer_norm = gradient_norm_for_key(
            surface_regularizer_gradients,
            key,
        )
        total_norm = gradient_norm_for_key(total_gradients, key)

        source_norm_denom = (
                loss_norm
                + depth_norm
                + normal_norm
                + opacity_norm
                + intra_slab_norm
                + curvature_scale_norm
        )

        loss_percent = (
            100.0 * loss_norm / source_norm_denom
            if source_norm_denom > 1.0e-20
            else 0.0
        )
        depth_percent = (
            100.0 * depth_norm / source_norm_denom
            if source_norm_denom > 1.0e-20
            else 0.0
        )
        normal_percent = (
            100.0 * normal_norm / source_norm_denom
            if source_norm_denom > 1.0e-20
            else 0.0
        )
        opacity_percent = (
            100.0 * opacity_norm / source_norm_denom
            if source_norm_denom > 1.0e-20
            else 0.0
        )
        intra_slab_percent = (
            100.0 * intra_slab_norm / source_norm_denom
            if source_norm_denom > 1.0e-20
            else 0.0
        )
        curvature_scale_percent = (
            100.0 * curvature_scale_norm / source_norm_denom
            if source_norm_denom > 1.0e-20
            else 0.0
        )

        lines.append(
            "  "
            f"{label:<8}"
            f"{loss_norm:>11.2e}"
            f"{surface_regularizer_norm:>11.2e}"
            f"{total_norm:>11.2e}"
            f"{loss_percent:>7.1f}%"
            f"{depth_percent:>7.1f}%"
            f"{normal_percent:>8.1f}%"
            f"{opacity_percent:>9.1f}%"
            f"{intra_slab_percent:>8.1f}%"
            f"{curvature_scale_percent:>7.1f}%"
            f"   "
            f"depth={depth_norm:.2e}, "
            f"normal={normal_norm:.2e}, "
            f"opacity={opacity_norm:.2e}, "
            f"intra={intra_slab_norm:.2e}, "
            f"curvature={curvature_scale_norm:.2e}"
        )

    return "\n".join(lines)


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
        depth_np = np.asarray(depth, dtype=np.float32)
        adjoint_np = np.asarray(adjoint_maps[camera_name], dtype=np.float32)

        means.append(float(np.mean(depth_np)))
        maxs.append(float(np.max(depth_np)))
        p99s.append(float(np.quantile(depth_np, 0.99)))
        adjoint_means.append(float(np.mean(adjoint_np)))
        adjoint_maxs.append(float(np.max(np.abs(adjoint_np))))

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
    updated = fetch_parameters(renderer)
    positions = torch.nn.Parameter(torch.tensor(updated["position"], device=device, dtype=torch.float32))
    rotations = torch.nn.Parameter(torch.tensor(updated["rotation"], device=device, dtype=torch.float32),
                                   requires_grad=False)
    scales = torch.nn.Parameter(torch.tensor(updated["scale"], device=device, dtype=torch.float32))
    albedos = torch.nn.Parameter(torch.tensor(updated["albedo"], device=device, dtype=torch.float32))
    opacities = torch.nn.Parameter(torch.tensor(updated["opacity"], device=device, dtype=torch.float32))
    betas = torch.nn.Parameter(torch.tensor(updated["beta"], device=device, dtype=torch.float32))
    powers = torch.nn.Parameter(torch.tensor(updated["power"], device=device, dtype=torch.float32))
    verify_rotations_inplace(rotations)
    return positions, rotations, scales, albedos, opacities, betas, powers


def verify_rotations_inplace(rotations: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        before_norm = torch.linalg.norm(rotations.data, dim=1)
        rotations.data.copy_(normalize_quaternions_torch(rotations.data))
        after_norm = torch.linalg.norm(rotations.data, dim=1)
        return {
            "before_min_norm": float(before_norm.min().item()),
            "before_max_norm": float(before_norm.max().item()),
            "after_min_norm": float(after_norm.min().item()),
            "after_max_norm": float(after_norm.max().item()),
        }


def verify_parameters_inplane(
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        trainable_surfel_mask: Optional[torch.Tensor] = None,
) -> None:
    verify_rotations_inplace(rotations)
    verify_scales_inplace(scales)
    verify_positions_inplace(positions)
    verify_albedos_inplace(albedos)
    verify_opacities_inplace(opacities)
    verify_beta_inplace(betas, trainable_surfel_mask=trainable_surfel_mask)


def assign_numpy_gradients_to_tensors(
        device: torch.device,
        positions: torch.nn.Parameter,
        rotation_delta: torch.nn.Parameter,
        scales: torch.nn.Parameter,
        albedos: torch.nn.Parameter,
        opacities: torch.nn.Parameter,
        betas: torch.nn.Parameter,
        grad_position_np: np.ndarray,
        grad_rotation_np: np.ndarray,
        grad_scales_np: np.ndarray,
        grad_albedos_np: np.ndarray,
        grad_opacities_np: np.ndarray,
        grad_betas_np: np.ndarray,
) -> None:
    positions.grad = torch.tensor(grad_position_np, device=device, dtype=torch.float32)
    rotation_delta.grad = torch.tensor(grad_rotation_np, device=device, dtype=torch.float32)
    scales.grad = torch.tensor(grad_scales_np, device=device, dtype=torch.float32)
    albedos.grad = torch.tensor(grad_albedos_np, device=device, dtype=torch.float32)
    opacities.grad = torch.tensor(grad_opacities_np, device=device, dtype=torch.float32)
    betas.grad = torch.tensor(grad_betas_np, device=device, dtype=torch.float32)


def save_gradients_snapshot(
        output_dir: Path,
        iteration: int,
        grad_position_np: np.ndarray,
        grad_rotation_np: np.ndarray,
        grad_scales_np: np.ndarray,
        grad_albedos_np: np.ndarray,
        grad_opacities_np: np.ndarray,
        grad_betas_np: np.ndarray,
) -> None:
    gradients_dir = output_dir / "gradients"
    gradients_dir.mkdir(parents=True, exist_ok=True)

    num_points = grad_position_np.shape[0]
    grad_opacity_flat = grad_opacities_np.reshape(num_points)
    grad_beta_flat = grad_betas_np.reshape(num_points)

    csv_path = gradients_dir / f"gradients_iter_{iteration:04d}.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "point_index",
            "grad_pos_x", "grad_pos_y", "grad_pos_z",
            "grad_rotation_x", "grad_rotation_y", "grad_rotation_z",
            "grad_scale_u", "grad_scale_v",
            "grad_albedo_r", "grad_albedo_g", "grad_albedo_b",
            "grad_opacity", "grad_beta",
        ])
        for idx in range(num_points):
            gx, gy, gz = grad_position_np[idx]
            grx, gry, grz = grad_rotation_np[idx]
            gsu, gsv = grad_scales_np[idx]
            gcr, gcg, gcb = grad_albedos_np[idx]
            writer.writerow(
                [idx, gx, gy, gz, grx, gry, grz, gsu, gsv, gcr, gcg, gcb, grad_opacity_flat[idx], grad_beta_flat[idx]])

    npz_path = gradients_dir / f"gradients_iter_{iteration:04d}.npz"
    np.savez_compressed(
        npz_path,
        grad_position=grad_position_np,
        grad_rotation=grad_rotation_np,
        grad_scales=grad_scales_np,
        grad_albedos=grad_albedos_np,
        grad_opacities=grad_opacities_np,
        grad_betas=grad_betas_np,
    )

    print(f"[Iter {iteration:04d}] Hotkey 'g' pressed -> saved gradients to:\n  {csv_path}\n  {npz_path}")


def save_manual_snapshot(
        renderer: pale.Renderer,
        output_dir: Path,
        iteration: int,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
        camera_ids: List[str],
        densification_origins: np.ndarray | None = None,
        primitive_ages: np.ndarray | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_images = renderer.render_forward()

    for camera_name in camera_ids:
        img_np = get_forward_rgb(final_images, camera_name)
        save_render(output_dir / f"render_final_{camera_name}.png", img_np)

        depth_np = get_forward_depth_distortion(final_images, camera_name)
        save_depth_distortion_snapshot(
            output_dir / f"depth_distortion_final_{camera_name}.png",
            depth_np,
            quantile=0.99,
            save_npy=False,
        )

        median_depth_np = get_forward_median_depth(final_images, camera_name)
        save_median_depth_snapshot(
            output_dir / f"median_depth_final_{camera_name}.png",
            median_depth_np,
            quantile=0.99,
            save_npy=False,
        )

        visible_normal_np = get_forward_visible_normal(final_images, camera_name)
        save_normal_map_snapshot(
            output_dir / f"visible_normal_final_{camera_name}.png",
            visible_normal_np,
            save_npy=False,
        )

        normal_from_depth_np = get_forward_normal_from_depth(final_images, camera_name)
        save_normal_map_snapshot(
            output_dir / f"normal_from_depth_final_{camera_name}.png",
            normal_from_depth_np,
            save_npy=False,
        )

    ply_path = output_dir / "points_final.ply"
    save_gaussians_to_ply(
        ply_path,
        positions,
        rotations,
        scales,
        albedos,
        opacities,
        betas,
        powers,
        shape_default=0.0,
        densification_origins=densification_origins,
        primitive_ages=primitive_ages,
    )

    print(
        f"[Iter {iteration:04d}] Hotkey 's' pressed -> saved "
        f"render_final_<camera>.png, depth_distortion_final_<camera>.png, "
        f"median_depth_final_<camera>.png, visible_normal_final_<camera>.png, "
        f"normal_from_depth_final_<camera>.png, and points_final.ply"
    )
    return ply_path


def save_iteration_point_cloud_snapshot(
        output_dir: Path,
        iteration: int,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
        densification_origins: np.ndarray | None = None,
        primitive_ages: np.ndarray | None = None,
) -> Path:
    points_dir = output_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)
    ply_path = points_dir / f"iter_{iteration:05d}_points.ply"
    save_gaussians_to_ply(
        ply_path,
        positions,
        rotations,
        scales,
        albedos,
        opacities,
        betas,
        powers,
        shape_default=0.0,
        densification_origins=densification_origins,
        primitive_ages=primitive_ages,
    )
    #print(f"[Iter {iteration:04d}] Saved point cloud snapshot: {ply_path}")
    return ply_path


def compute_snapshot_adjoint_images(
        renderer: pale.Renderer,
        forward_out: Dict[str, dict],
        target_images: Dict[str, np.ndarray],
        camera_ids: List[str],
        ssim_weight: float = 0.0,
        ssim_window_size: int = 11,
        ssim_sigma: float = 1.5,
) -> Dict[str, Any]:
    loss_grad_images: Dict[str, np.ndarray] = {}

    for camera_name in camera_ids:
        if camera_name not in target_images:
            continue

        current_rgb_np = get_forward_linear_rgb(forward_out, camera_name)
        target_rgb_np = target_images[camera_name]
        if ssim_weight > 0.0:
            _, loss_grad_images[camera_name], _ = compute_l2_ssim_loss_and_grad(
                current_rgb_np,
                target_rgb_np,
                ssim_weight=ssim_weight,
                window_size=ssim_window_size,
                sigma=ssim_sigma,
            )
        else:
            loss_grad_images[camera_name] = compute_l2_grad(current_rgb_np, target_rgb_np)

    if not loss_grad_images:
        return {}

    _, adjoint_images = renderer.render_backward(loss_grad_images)
    return adjoint_images


def load_target_images(
        renderer: pale.Renderer,
        dataset_path: Path,
        target_color_space: str = "auto",
) -> tuple[Dict[str, np.ndarray], List[str], List[str]]:
    target_path = Path(dataset_path)
    if not target_path.is_dir():
        raise RuntimeError(f"Target path '{target_path}' must be a directory when multiple cameras are used.")

    training_camera_ids = get_training_camera_names(renderer)
    all_camera_ids = get_all_camera_names(renderer)
    target_images: Dict[str, np.ndarray] = {}
    images_path = target_path / "images"
    if not images_path.is_dir():
        raise RuntimeError(f"Target dataset is missing images directory: {images_path}")

    print(f"Loading target images from directory: {target_path}")
    for camera_name in training_camera_ids:
        candidates = sorted(
            path for path in images_path.iterdir()
            if path.is_file()
            and path.stem == camera_name
            and path.suffix.lower() in SUPPORTED_TARGET_IMAGE_SUFFIXES
        )
        if not candidates:
            suffixes = ", ".join(SUPPORTED_TARGET_IMAGE_SUFFIXES)
            raise RuntimeError(
                f"Missing target image for camera '{camera_name}' in {images_path}; "
                f"supported extensions: {suffixes}"
            )
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple target images found for camera '{camera_name}': "
                + ", ".join(str(path) for path in candidates)
            )
        image_path = candidates[0]

        print(f"  Camera '{camera_name}': loading target {image_path}")
        target_images[camera_name] = load_target_image(
            image_path,
            color_space=target_color_space,
        )
        print(
            f"    loaded linear training target with shape {target_images[camera_name].shape}"
        )

    return target_images, training_camera_ids, all_camera_ids


def create_torch_parameters_from_initial(
        initial_params: Dict[str, np.ndarray],
        device: torch.device,
) -> Tuple[torch.nn.Parameter, ...]:
    positions = torch.nn.Parameter(torch.tensor(initial_params["position"], device=device, dtype=torch.float32))
    rotations = torch.nn.Parameter(torch.tensor(initial_params["rotation"], device=device, dtype=torch.float32),
                                   requires_grad=False)
    scales = torch.nn.Parameter(torch.tensor(initial_params["scale"], device=device, dtype=torch.float32))
    albedos = torch.nn.Parameter(torch.tensor(initial_params["albedo"], device=device, dtype=torch.float32))
    opacities = torch.nn.Parameter(torch.tensor(initial_params["opacity"], device=device, dtype=torch.float32))
    betas = torch.nn.Parameter(torch.tensor(initial_params["beta"], device=device, dtype=torch.float32))
    powers = torch.nn.Parameter(torch.tensor(initial_params["power"], device=device, dtype=torch.float32))
    verify_rotations_inplace(rotations)
    return positions, rotations, scales, albedos, opacities, betas, powers


def make_initial_params_reference(initial_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "position": initial_params["position"].copy(),
        "rotation": initial_params["rotation"].copy(),
        "scale": initial_params["scale"].copy(),
        "albedo": initial_params["albedo"].copy(),
        "opacity": initial_params["opacity"].copy(),
        "beta": initial_params["beta"].copy(),
        "power": initial_params["power"].copy(),
    }


def current_params_as_numpy(
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
) -> Dict[str, np.ndarray]:
    return {
        "position": positions.detach().cpu().numpy(),
        "rotation": rotations.detach().cpu().numpy(),
        "scale": scales.detach().cpu().numpy(),
        "albedo": albedos.detach().cpu().numpy(),
        "opacity": opacities.detach().cpu().numpy(),
        "beta": betas.detach().cpu().numpy(),
        "power": powers.detach().cpu().numpy(),
    }


def compute_initial_losses_and_save_outputs(
        output_dir: Path,
        initial_images: Dict[str, dict],
        target_images: Dict[str, np.ndarray],
        all_camera_ids: List[str],
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
        ssim_weight: float,
        ssim_window_size: int,
        ssim_sigma: float,
        depth_distortion_weight: float,
        normal_consistency_weight: float,
        opacity_prior_weight: float,
        intra_slab_depth_weight: float,
        curvature_scale_weight: float,
        use_depth_distortion: bool,
        use_normal_consistency: bool,
        use_opacity_prior: bool,
        use_intra_slab_depth: bool,
        use_curvature_scale: bool,
) -> tuple[float, ...]:
    initial_points_path = output_dir / "initial_points.ply"
    save_gaussians_to_ply(
        initial_points_path,
        positions,
        rotations,
        scales,
        albedos,
        opacities,
        betas,
        powers,
        shape_default=0.0,
    )
    print(f"Initial parameters written to PLY: {initial_points_path}")

    initial_rgb_loss = 0.0
    initial_depth_distortion_loss_raw = 0.0
    initial_normal_loss_raw = 0.0
    initial_opacity_prior_loss_raw = 0.0
    initial_intra_slab_depth_loss_raw = 0.0
    initial_curvature_scale_loss_raw = 0.0

    for camera_name in all_camera_ids:
        img_np = get_forward_rgb(initial_images, camera_name)
        img_linear_np = get_forward_linear_rgb(initial_images, camera_name)
        (output_dir / camera_name).mkdir(parents=True, exist_ok=True)
        save_render(output_dir / f"render_initial_{camera_name}.png", img_np)

        if camera_name not in target_images:
            print(f"Warning: no target image found for camera '{camera_name}', skipping target save and loss.")
            continue

        tgt_np = target_images[camera_name]
        rgb_loss_value, _, _ = compute_l2_ssim_metrics(
            img_linear_np,
            tgt_np,
            ssim_weight=ssim_weight,
            window_size=ssim_window_size,
            sigma=ssim_sigma,
        )
        initial_rgb_loss += rgb_loss_value
        save_render(
            output_dir / f"render_target_{camera_name}.png",
            linear_to_srgb(tgt_np),
        )

        if use_depth_distortion:
            initial_depth_distortion_loss_raw += float(get_forward_depth_distortion(initial_images, camera_name).mean())

        if use_normal_consistency:
            visible_normal = get_forward_visible_normal(initial_images, camera_name)
            normal_from_depth = get_forward_normal_from_depth(initial_images, camera_name)
            raw_normal_loss_value, _, _ = compute_normal_consistency_loss_and_adjoints(
                visible_normal,
                normal_from_depth,
                1.0,
            )
            initial_normal_loss_raw += raw_normal_loss_value

        if use_opacity_prior:
            initial_opacity_prior_loss_raw += float(get_forward_opacity_prior(initial_images, camera_name).mean())

        if use_intra_slab_depth:
            loss_map = get_forward_intra_slab_depth(initial_images, camera_name)
            active_count = max(
                1,
                int(get_forward_intra_slab_depth_active_slab_count(
                    initial_images, camera_name
                ).sum(dtype=np.uint64)),
            )
            initial_intra_slab_depth_loss_raw += float(loss_map.sum() / active_count)

        if use_curvature_scale:
            loss_map = get_forward_curvature_scale(initial_images, camera_name)
            active_count = max(
                1,
                int(get_forward_curvature_scale_active_slab_count(
                    initial_images, camera_name
                ).sum(dtype=np.uint64)),
            )
            initial_curvature_scale_loss_raw += float(loss_map.sum() / active_count)

    initial_depth_distortion_loss_weighted = depth_distortion_weight * initial_depth_distortion_loss_raw
    initial_normal_loss_weighted = normal_consistency_weight * initial_normal_loss_raw
    initial_opacity_prior_loss_weighted = opacity_prior_weight * initial_opacity_prior_loss_raw
    initial_intra_slab_depth_loss_weighted = (
        intra_slab_depth_weight * initial_intra_slab_depth_loss_raw
    )
    initial_curvature_scale_loss_weighted = (
        curvature_scale_weight * initial_curvature_scale_loss_raw
    )

    initial_total_loss = (
            initial_rgb_loss
            + initial_depth_distortion_loss_weighted
            + initial_normal_loss_weighted
            + initial_opacity_prior_loss_weighted
            + initial_intra_slab_depth_loss_weighted
            + initial_curvature_scale_loss_weighted
    )

    return (
        initial_rgb_loss,
        initial_depth_distortion_loss_raw,
        initial_depth_distortion_loss_weighted,
        initial_normal_loss_raw,
        initial_normal_loss_weighted,
        initial_opacity_prior_loss_raw,
        initial_opacity_prior_loss_weighted,
        initial_intra_slab_depth_loss_raw,
        initial_intra_slab_depth_loss_weighted,
        initial_curvature_scale_loss_raw,
        initial_curvature_scale_loss_weighted,
        initial_total_loss,
    )


def print_loss_summary(
        prefix: str,
        rgb_loss: float,
        depth_distortion_loss_raw: float,
        depth_distortion_loss_weighted: float,
        normal_loss_raw: float,
        normal_loss_weighted: float,
        opacity_prior_loss_raw: float,
        opacity_prior_loss_weighted: float,
        intra_slab_depth_loss_raw: float,
        intra_slab_depth_loss_weighted: float,
        curvature_scale_loss_raw: float,
        curvature_scale_loss_weighted: float,
        total_loss: float,
) -> None:
    print(f"{prefix} RGB loss                               : {rgb_loss:.6e}")
    print(f"{prefix} depth distortion loss (raw)            : {depth_distortion_loss_raw:.6e}")
    print(f"{prefix} depth distortion loss (weighted)       : {depth_distortion_loss_weighted:.6e}")
    print(f"{prefix} normal consistency loss (raw)          : {normal_loss_raw:.6e}")
    print(f"{prefix} normal consistency loss (weighted)     : {normal_loss_weighted:.6e}")
    print(f"{prefix} opacity prior loss (raw)               : {opacity_prior_loss_raw:.6e}")
    print(f"{prefix} opacity prior loss (weighted)          : {opacity_prior_loss_weighted:.6e}")
    print(f"{prefix} intra-slab depth loss (raw)            : {intra_slab_depth_loss_raw:.6e}")
    print(f"{prefix} intra-slab depth loss (weighted)       : {intra_slab_depth_loss_weighted:.6e}")
    print(f"{prefix} curvature scale loss (raw)             : {curvature_scale_loss_raw:.6e}")
    print(f"{prefix} curvature scale loss (weighted)        : {curvature_scale_loss_weighted:.6e}")
    print(f"{prefix} total loss                             : {total_loss:.6e}")


def compute_surface_regularizer_losses_and_adjoints(
        forward_out: Dict[str, dict],
        training_camera_ids: List[str],
        depth_distortion_weight: float,
        normal_consistency_weight: float,
        opacity_prior_weight: float,
        intra_slab_depth_weight: float,
        curvature_scale_weight: float,
        use_depth_distortion: bool,
        use_normal_consistency: bool,
        use_opacity_prior: bool,
        use_intra_slab_depth: bool,
        use_curvature_scale: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = make_zero_loss_values()
    result.update({
        "depth_distortion_grad_images": {},
        "visible_normal_adjoints": {},
        "depth_normal_adjoints": {},
        "depth_distortion_maps_for_logging": {},
        "intra_slab_depth_grad_images": {},
        "curvature_scale_grad_images": {},
        "intra_slab_depth_maps_for_logging": {},
        "curvature_scale_maps_for_logging": {},
        "per_camera_loss_values": {},
    })

    for camera_name in training_camera_ids:
        camera_loss_values = make_zero_loss_values()

        if use_depth_distortion:
            current_depth_distortion_np = get_forward_depth_distortion(forward_out, camera_name)
            depth_distortion_loss_raw = float(current_depth_distortion_np.mean())
            depth_distortion_loss_weighted = depth_distortion_weight * depth_distortion_loss_raw

            camera_loss_values["total_depth_distortion_loss_raw"] = depth_distortion_loss_raw
            camera_loss_values["total_depth_distortion_loss_weighted"] = depth_distortion_loss_weighted
            camera_loss_values["total_loss_value"] += depth_distortion_loss_weighted

            result["depth_distortion_maps_for_logging"][camera_name] = current_depth_distortion_np
            result["depth_distortion_grad_images"][camera_name] = make_mean_reduction_adjoint_image(
                current_depth_distortion_np,
                depth_distortion_weight,
            )

        if use_normal_consistency:
            visible_normal = get_forward_visible_normal(forward_out, camera_name)
            normal_from_depth = get_forward_normal_from_depth(forward_out, camera_name)

            normal_loss_raw, visible_normal_adjoint, depth_normal_adjoint = (
                compute_normal_consistency_loss_and_adjoints(
                    visible_normal,
                    normal_from_depth,
                    1.0,
                )
            )

            normal_loss_weighted = normal_consistency_weight * normal_loss_raw

            camera_loss_values["total_normal_loss_raw"] = normal_loss_raw
            camera_loss_values["total_normal_loss_weighted"] = normal_loss_weighted
            camera_loss_values["total_loss_value"] += normal_loss_weighted

            result["visible_normal_adjoints"][camera_name] = (
                    normal_consistency_weight * visible_normal_adjoint
            ).astype(np.float32, copy=False)

            result["depth_normal_adjoints"][camera_name] = (
                    normal_consistency_weight * depth_normal_adjoint
            ).astype(np.float32, copy=False)

        if use_opacity_prior:
            opacity_prior_np = get_forward_opacity_prior(forward_out, camera_name)
            opacity_prior_loss_raw = float(opacity_prior_np.mean())
            opacity_prior_loss_weighted = opacity_prior_weight * opacity_prior_loss_raw

            camera_loss_values["total_opacity_prior_loss_raw"] = opacity_prior_loss_raw
            camera_loss_values["total_opacity_prior_loss_weighted"] = opacity_prior_loss_weighted
            camera_loss_values["total_loss_value"] += opacity_prior_loss_weighted

        if use_intra_slab_depth:
            intra_slab_depth_np = get_forward_intra_slab_depth(forward_out, camera_name)
            active_slab_count_np = get_forward_intra_slab_depth_active_slab_count(
                forward_out, camera_name
            )
            active_slab_count = max(1, int(active_slab_count_np.sum(dtype=np.uint64)))
            intra_slab_depth_loss_raw = float(intra_slab_depth_np.sum() / active_slab_count)
            intra_slab_depth_loss_weighted = (
                intra_slab_depth_weight * intra_slab_depth_loss_raw
            )
            camera_loss_values["total_intra_slab_depth_loss_raw"] = intra_slab_depth_loss_raw
            camera_loss_values["total_intra_slab_depth_loss_weighted"] = (
                intra_slab_depth_loss_weighted
            )
            camera_loss_values["total_loss_value"] += intra_slab_depth_loss_weighted
            result["intra_slab_depth_maps_for_logging"][camera_name] = intra_slab_depth_np
            result["intra_slab_depth_grad_images"][camera_name] = np.where(
                active_slab_count_np > 0,
                intra_slab_depth_weight / float(active_slab_count),
                0.0,
            ).astype(np.float32, copy=False)

        if use_curvature_scale:
            curvature_scale_np = get_forward_curvature_scale(forward_out, camera_name)
            active_slab_count_np = get_forward_curvature_scale_active_slab_count(
                forward_out, camera_name
            )
            active_slab_count = max(1, int(active_slab_count_np.sum(dtype=np.uint64)))
            curvature_scale_loss_raw = float(curvature_scale_np.sum() / active_slab_count)
            curvature_scale_loss_weighted = curvature_scale_weight * curvature_scale_loss_raw
            camera_loss_values["total_curvature_scale_loss_raw"] = curvature_scale_loss_raw
            camera_loss_values["total_curvature_scale_loss_weighted"] = curvature_scale_loss_weighted
            camera_loss_values["total_loss_value"] += curvature_scale_loss_weighted
            result["curvature_scale_maps_for_logging"][camera_name] = curvature_scale_np
            result["curvature_scale_grad_images"][camera_name] = np.where(
                active_slab_count_np > 0,
                curvature_scale_weight / float(active_slab_count),
                0.0,
            ).astype(np.float32, copy=False)

        for loss_key in LOSS_VALUE_KEYS:
            result[loss_key] += camera_loss_values[loss_key]

        result["per_camera_loss_values"][camera_name] = camera_loss_values

    return result


def extract_total_gradient_arrays(
        total_gradients: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grad_position_np = np.asarray(total_gradients["position"], dtype=np.float32, order="C")
    grad_rotation_np = np.asarray(total_gradients["rotation"], dtype=np.float32, order="C")
    grad_scales_np = np.asarray(total_gradients["scale"], dtype=np.float32, order="C")
    grad_albedos_np = np.asarray(total_gradients["albedo"], dtype=np.float32, order="C")
    grad_opacities_np = np.asarray(total_gradients["opacity"], dtype=np.float32, order="C")
    grad_betas_np = np.asarray(total_gradients["beta"], dtype=np.float32, order="C")
    return (grad_position_np, grad_rotation_np, grad_scales_np, grad_albedos_np, grad_opacities_np, grad_betas_np)


def update_densification_statistics(
        iteration: int,
        densification_interval: int,
        densification_cycle_start_iteration: int,
        densification_stats_skip_iterations: int,
        densify_position_grad_accum_np: np.ndarray,
        densify_position_grad_denom_np: np.ndarray,
        densify_position_grad_vector_accum_np: np.ndarray,
        rotations: torch.Tensor,
        albedos: torch.Tensor,
        trainable_surfel_mask: torch.Tensor,
        densify_bsdf_floor: float,
        densify_bsdf_gamma: float,
        densify_position_grad_per_camera_np: np.ndarray,
        densify_position_grad_per_camera_count_np: np.ndarray,
) -> None:
    if densification_interval <= 0:
        return

    densification_phase = max(0, int(iteration) - int(densification_cycle_start_iteration))
    should_accumulate = (
            densification_interval <= 1
            or (
                    densification_phase >= densification_stats_skip_iterations
                    and densification_phase != 0
            )
    )
    if not should_accumulate:
        return

    if densify_position_grad_per_camera_np is None:
        raise RuntimeError(
            "update_densification_statistics requires renderer gradient stats: "
            "densify_position_grad_per_camera_np is None. "
            "Expected adjoint_images['gradient_stats']['clone_signal_per_camera']."
        )

    if densify_position_grad_per_camera_count_np is None:
        raise RuntimeError(
            "update_densification_statistics requires renderer gradient stats: "
            "densify_position_grad_per_camera_count_np is None. "
            "Expected adjoint_images['gradient_stats']['clone_signal_record_count_per_camera']."
        )

    per_camera_grad_np = np.asarray(
        densify_position_grad_per_camera_np,
        dtype=np.float32,
        order="C",
    )

    per_camera_count_np = np.asarray(
        densify_position_grad_per_camera_count_np,
        dtype=np.uint32,
        order="C",
    )

    if per_camera_grad_np.ndim != 3 or per_camera_grad_np.shape[2] != 3:
        raise RuntimeError(
            "densify_position_grad_per_camera_np must have shape (N, C, 3), "
            f"got {per_camera_grad_np.shape}"
        )

    point_count = per_camera_grad_np.shape[0]
    camera_count = per_camera_grad_np.shape[1]

    if per_camera_count_np.shape != (point_count, camera_count):
        raise RuntimeError(
            "densify_position_grad_per_camera_count_np must have shape "
            f"{(point_count, camera_count)}, got {per_camera_count_np.shape}"
        )

    if densify_position_grad_accum_np.shape[0] != point_count:
        raise RuntimeError(
            "Densification accumulator point-count mismatch: "
            f"accum={densify_position_grad_accum_np.shape[0]}, renderer_stats={point_count}"
        )

    with torch.no_grad():
        tangent_u, tangent_v, tangent_w = quaternion_to_tangent_frame_torch(rotations.detach())
        tangent_u_np = tangent_u.detach().cpu().numpy().astype(np.float32)
        tangent_v_np = tangent_v.detach().cpu().numpy().astype(np.float32)
        tangent_w_np = tangent_w.detach().cpu().numpy().astype(np.float32)
        albedo_np = albedos.detach().cpu().numpy().astype(np.float32)
        trainable_np = trainable_surfel_mask.detach().cpu().numpy().astype(bool).reshape(-1)

    if tangent_u_np.shape != (point_count, 3):
        raise RuntimeError(f"tangent_u shape mismatch: expected {(point_count, 3)}, got {tangent_u_np.shape}")
    if tangent_v_np.shape != (point_count, 3):
        raise RuntimeError(f"tangent_v shape mismatch: expected {(point_count, 3)}, got {tangent_v_np.shape}")
    if tangent_w_np.shape != (point_count, 3):
        raise RuntimeError(f"tangent_w shape mismatch: expected {(point_count, 3)}, got {tangent_w_np.shape}")
    if trainable_np.shape[0] != point_count:
        raise RuntimeError(f"trainable mask length mismatch: expected {point_count}, got {trainable_np.shape[0]}")

    tangent_u_norm_np = np.linalg.norm(tangent_u_np, axis=1, keepdims=True)
    tangent_v_norm_np = np.linalg.norm(tangent_v_np, axis=1, keepdims=True)
    tangent_w_norm_np = np.linalg.norm(tangent_w_np, axis=1, keepdims=True)

    tangent_u_unit_np = tangent_u_np / np.maximum(tangent_u_norm_np, 1.0e-8)
    tangent_v_unit_np = tangent_v_np / np.maximum(tangent_v_norm_np, 1.0e-8)
    tangent_w_unit_np = tangent_w_np / np.maximum(tangent_w_norm_np, 1.0e-8)

    dot_u_np = np.sum(per_camera_grad_np * tangent_u_unit_np[:, None, :], axis=2, keepdims=True)
    dot_v_np = np.sum(per_camera_grad_np * tangent_v_unit_np[:, None, :], axis=2, keepdims=True)
    dot_w_np = np.sum(per_camera_grad_np * tangent_w_unit_np[:, None, :], axis=2)

    per_camera_local_tangent_grad_np = np.concatenate(
        [
            dot_u_np,
            dot_v_np,
            np.zeros_like(dot_u_np),
        ],
        axis=2,
    )

    visible_camera_mask_np = per_camera_count_np > 0
    active_camera_count_np = visible_camera_mask_np.sum(axis=1, keepdims=True).astype(np.float32)

    per_camera_tangent_grad_norm_np = np.sqrt(
        np.square(dot_u_np[:, :, 0]) + np.square(dot_v_np[:, :, 0])
    ).astype(np.float32)
    per_camera_normal_grad_abs_np = np.abs(dot_w_np).astype(np.float32)
    per_camera_local_grad_norm_np = np.sqrt(
        np.square(per_camera_tangent_grad_norm_np) + np.square(per_camera_normal_grad_abs_np)
    ).astype(np.float32)
    normal_direction_downweight_np = (
            per_camera_tangent_grad_norm_np
            / np.maximum(per_camera_local_grad_norm_np, 1.0e-12)
    ).astype(np.float32)
    per_camera_tangent_grad_norm_np = np.nan_to_num(
        per_camera_tangent_grad_norm_np,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    normal_direction_downweight_np = np.nan_to_num(
        normal_direction_downweight_np,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    visible_mask_float_np = visible_camera_mask_np.astype(np.float32)
    visible_downweight_np = visible_mask_float_np * normal_direction_downweight_np

    safe_active_camera_count_np = np.maximum(active_camera_count_np, 1.0)

    # Scalar score:
    #     mean_visible(||project_tangent(g_camera)|| * tangent_fraction(g_camera))
    # where tangent_fraction suppresses clone pressure from mostly-normal motion.
    densify_position_signal_np = (
                                         per_camera_tangent_grad_norm_np * visible_downweight_np
                                 ).sum(axis=1, keepdims=True) / safe_active_camera_count_np

    # Signed vector direction:
    #     mean_visible((dot_u, dot_v, 0) * tangent_fraction(g_camera))
    # Accumulating local tangent coordinates keeps the direction stable if the surfel rotates.
    density_grad_position_vector_np = (
                                              per_camera_local_tangent_grad_np * visible_downweight_np[:, :, None]
                                      ).sum(axis=1) / safe_active_camera_count_np

    density_grad_position_vector_np[active_camera_count_np[:, 0] == 0.0] = 0.0
    densify_position_signal_np[active_camera_count_np[:, 0] == 0.0] = 0.0

    linear_rgb_bsdf_scale_np = np.mean(albedo_np, axis=1)
    bsdf_normalizer_np = (np.maximum(linear_rgb_bsdf_scale_np, densify_bsdf_floor) ** densify_bsdf_gamma).astype(
        np.float32)
    densify_position_signal_np = densify_position_signal_np / bsdf_normalizer_np[:, None]
    density_grad_position_vector_np = density_grad_position_vector_np / bsdf_normalizer_np[:, None]

    densify_position_signal_np = np.nan_to_num(
        densify_position_signal_np,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    density_grad_position_vector_np = np.nan_to_num(
        density_grad_position_vector_np,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    update_density_scalar_mask_np = (
            trainable_np
            & np.isfinite(densify_position_signal_np[:, 0])
            & (densify_position_signal_np[:, 0] > 0.0)
    )

    densify_position_grad_accum_np[update_density_scalar_mask_np, 0] += (
        densify_position_signal_np[update_density_scalar_mask_np, 0]
    )
    densify_position_grad_denom_np[update_density_scalar_mask_np, 0] += 1.0

    vector_norm_np = np.linalg.norm(density_grad_position_vector_np, axis=1)

    update_density_vector_mask_np = (
            trainable_np
            & np.isfinite(vector_norm_np)
            & (vector_norm_np > 0.0)
    )

    densify_position_grad_vector_accum_np[update_density_vector_mask_np] += (
        density_grad_position_vector_np[update_density_vector_mask_np]
    )


CURVATURE_DENSIFICATION_STAT_KEYS = (
    "violation_sum",
    "violation_count",
    "direction_tensor_uu",
    "direction_tensor_uv",
    "direction_tensor_vv",
)


def make_curvature_densification_accumulators(point_count: int) -> Dict[str, np.ndarray]:
    return {
        "violation_sum": np.zeros((point_count,), dtype=np.float32),
        "violation_count": np.zeros((point_count,), dtype=np.uint64),
        "direction_tensor_uu": np.zeros((point_count,), dtype=np.float32),
        "direction_tensor_uv": np.zeros((point_count,), dtype=np.float32),
        "direction_tensor_vv": np.zeros((point_count,), dtype=np.float32),
    }


def update_curvature_densification_statistics(
        iteration: int,
        densification_interval: int,
        densification_cycle_start_iteration: int,
        densification_stats_skip_iterations: int,
        renderer_stats: Dict[str, np.ndarray],
        accumulators: Dict[str, np.ndarray],
) -> None:
    """Accumulate scalar curvature, but retain only the latest direction tensor.

    The scalar violation is basis-independent and remains useful across a full
    densification cycle. Tensor components live in the surfel's current local
    tangent frame, so summing them across optimizer iterations would mix frames
    whenever the surfel rotates.
    """
    if densification_interval <= 0:
        return

    densification_phase = max(0, int(iteration) - int(densification_cycle_start_iteration))
    should_accumulate = (
            densification_interval <= 1
            or (
                    densification_phase >= densification_stats_skip_iterations
                    and densification_phase != 0
            )
    )
    if not should_accumulate:
        return

    missing_keys = [
        key for key in CURVATURE_DENSIFICATION_STAT_KEYS
        if key not in renderer_stats or key not in accumulators
    ]
    if missing_keys:
        raise RuntimeError(
            "Curvature densification statistics are missing keys: "
            + ", ".join(missing_keys)
        )

    violation_sum = np.asarray(renderer_stats["violation_sum"], dtype=np.float32).reshape(-1)
    violation_count = np.asarray(renderer_stats["violation_count"], dtype=np.uint64).reshape(-1)
    tensor_uu = np.asarray(renderer_stats["direction_tensor_uu"], dtype=np.float32).reshape(-1)
    tensor_uv = np.asarray(renderer_stats["direction_tensor_uv"], dtype=np.float32).reshape(-1)
    tensor_vv = np.asarray(renderer_stats["direction_tensor_vv"], dtype=np.float32).reshape(-1)

    point_count = accumulators["violation_sum"].shape[0]
    for key, values in (
            ("violation_sum", violation_sum),
            ("violation_count", violation_count),
            ("direction_tensor_uu", tensor_uu),
            ("direction_tensor_uv", tensor_uv),
            ("direction_tensor_vv", tensor_vv),
    ):
        if values.shape != (point_count,):
            raise RuntimeError(
                f"Curvature densification {key} shape mismatch: "
                f"expected {(point_count,)}, got {values.shape}"
            )

    valid_observation = (
            (violation_count > 0)
            & np.isfinite(violation_sum)
            & (violation_sum >= 0.0)
    )
    accumulators["violation_sum"][valid_observation] += violation_sum[valid_observation]
    accumulators["violation_count"][valid_observation] += violation_count[valid_observation]

    # K_uu/K_uv/K_vv are expressed in the *current* local tangent frame. Keep
    # this renderer iteration as a snapshot rather than accumulating components
    # from older frames. Clearing unobserved entries also prevents stale axes
    # from directing a split when a surfel has no current curvature observation.
    for key, values in (
            ("direction_tensor_uu", tensor_uu),
            ("direction_tensor_uv", tensor_uv),
            ("direction_tensor_vv", tensor_vv),
    ):
        accumulators[key].fill(0.0)
        accumulators[key][valid_observation] = np.nan_to_num(
            values[valid_observation], nan=0.0, posinf=0.0, neginf=0.0
        )


def maybe_make_densification_result(
        iteration: int,
        config: OptimizationConfig,
        positions: torch.nn.Parameter,
        rotations: torch.Tensor,
        scales: torch.nn.Parameter,
        albedos: torch.nn.Parameter,
        opacities: torch.nn.Parameter,
        betas: torch.nn.Parameter,
        powers: torch.nn.Parameter,
        trainable_surfel_mask: torch.Tensor,
        densify_position_grad_accum_np: np.ndarray,
        densify_position_grad_denom_np: np.ndarray,
        densify_position_grad_vector_accum_np: np.ndarray,
        densify_after: int,
        densification_interval: int,
        densification_verbose: bool,
        densification_grad_quantile: float,
        densification_grad_abs_min: float,
        densify_curvature_stats_accum: Optional[Dict[str, np.ndarray]] = None,
        force_densification: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    if densification_interval <= 0:
        return None

    if iteration < densify_after:
        return None

    if (
            not force_densification
            and iteration % densification_interval != 0
    ):
        return None

    with torch.no_grad():
        valid_denom_np = densify_position_grad_denom_np.reshape(-1) > 0.0
        avg_density_grad_norm_np = np.zeros((positions.shape[0],), dtype=np.float32)
        avg_density_grad_vector_local_np = np.zeros(tuple(positions.shape), dtype=np.float32)

        avg_density_grad_norm_np[valid_denom_np] = (
                densify_position_grad_accum_np.reshape(-1)[valid_denom_np]
                / densify_position_grad_denom_np.reshape(-1)[valid_denom_np]
        )

        avg_density_grad_vector_local_np[valid_denom_np] = (
                densify_position_grad_vector_accum_np[valid_denom_np]
                / densify_position_grad_denom_np.reshape(-1, 1)[valid_denom_np]
        )
        avg_density_grad_vector_local_np = np.nan_to_num(
            avg_density_grad_vector_local_np,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        tangent_u, tangent_v, _ = quaternion_to_tangent_frame_torch(rotations.detach())
        tangent_u_np = tangent_u.detach().cpu().numpy().astype(np.float32)
        tangent_v_np = tangent_v.detach().cpu().numpy().astype(np.float32)
        avg_density_grad_vector_np = (
                avg_density_grad_vector_local_np[:, 0:1] * tangent_u_np
                + avg_density_grad_vector_local_np[:, 1:2] * tangent_v_np
        ).astype(np.float32)
        avg_density_grad_vector_np = np.nan_to_num(
            avg_density_grad_vector_np,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        grad_pos_norm_np = np.nan_to_num(
            avg_density_grad_norm_np,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        trainable_np = trainable_surfel_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        finite_grad = np.isfinite(grad_pos_norm_np)
        position_abs_candidate_mask_np = (
                valid_denom_np
                & finite_grad
                & trainable_np
                & (grad_pos_norm_np >= densification_grad_abs_min)
        )

        curvature_violation_threshold = float(
            getattr(config, "curvature_violation_threshold", 0.0)
        )
        curvature_enabled = (
                np.isfinite(curvature_violation_threshold)
                and curvature_violation_threshold > 0.0
                and densify_curvature_stats_accum is not None
        )
        curvature_violation_mean_np = np.zeros((positions.shape[0],), dtype=np.float32)
        curvature_tensor_uu_np = np.zeros_like(curvature_violation_mean_np)
        curvature_tensor_uv_np = np.zeros_like(curvature_violation_mean_np)
        curvature_tensor_vv_np = np.zeros_like(curvature_violation_mean_np)
        valid_curvature_observation_np = np.zeros((positions.shape[0],), dtype=bool)

        if curvature_enabled:
            missing_keys = [
                key for key in CURVATURE_DENSIFICATION_STAT_KEYS
                if key not in densify_curvature_stats_accum
            ]
            if missing_keys:
                raise RuntimeError(
                    "Curvature densification accumulators are missing keys: "
                    + ", ".join(missing_keys)
                )

            curvature_sum_np = np.asarray(
                densify_curvature_stats_accum["violation_sum"], dtype=np.float32
            ).reshape(-1)
            curvature_count_np = np.asarray(
                densify_curvature_stats_accum["violation_count"], dtype=np.uint64
            ).reshape(-1)
            curvature_tensor_uu_np = np.asarray(
                densify_curvature_stats_accum["direction_tensor_uu"], dtype=np.float32
            ).reshape(-1)
            curvature_tensor_uv_np = np.asarray(
                densify_curvature_stats_accum["direction_tensor_uv"], dtype=np.float32
            ).reshape(-1)
            curvature_tensor_vv_np = np.asarray(
                densify_curvature_stats_accum["direction_tensor_vv"], dtype=np.float32
            ).reshape(-1)

            curvature_arrays = (
                curvature_sum_np,
                curvature_count_np,
                curvature_tensor_uu_np,
                curvature_tensor_uv_np,
                curvature_tensor_vv_np,
            )
            if any(array.shape != (positions.shape[0],) for array in curvature_arrays):
                raise RuntimeError(
                    "Curvature densification accumulator length does not match the point count"
                )

            valid_curvature_observation_np = (
                    (curvature_count_np > 0)
                    & np.isfinite(curvature_sum_np)
                    & (curvature_sum_np >= 0.0)
            )
            curvature_violation_mean_np[valid_curvature_observation_np] = (
                    curvature_sum_np[valid_curvature_observation_np]
                    / curvature_count_np[valid_curvature_observation_np].astype(np.float32)
            )
            curvature_violation_mean_np = np.nan_to_num(
                curvature_violation_mean_np, nan=0.0, posinf=0.0, neginf=0.0
            )
            curvature_tensor_uu_np = np.nan_to_num(
                curvature_tensor_uu_np, nan=0.0, posinf=0.0, neginf=0.0
            )
            curvature_tensor_uv_np = np.nan_to_num(
                curvature_tensor_uv_np, nan=0.0, posinf=0.0, neginf=0.0
            )
            curvature_tensor_vv_np = np.nan_to_num(
                curvature_tensor_vv_np, nan=0.0, posinf=0.0, neginf=0.0
            )

        densification_result = None
        densify_reason = "not_attempted"
        n_new_from_densification = 0

        finite_count = int(np.count_nonzero(finite_grad))
        trainable_count = int(np.count_nonzero(trainable_np))
        above_abs_count = int(np.count_nonzero(grad_pos_norm_np >= densification_grad_abs_min))
        position_abs_candidate_count = int(np.count_nonzero(position_abs_candidate_mask_np))
        valid_denom_count = int(np.count_nonzero(valid_denom_np))
        valid_curvature_count = int(np.count_nonzero(valid_curvature_observation_np))

        grad_threshold = float("nan")
        grad_quantile_threshold = float("nan")
        scene_extent = densification_scene_extent_for_positions(
            config=config,
            positions=positions,
            trainable_surfel_mask=trainable_surfel_mask,
        )
        exact_clone_scale_threshold = exact_clone_scale_threshold_for_positions(
            config=config,
            positions=positions,
            trainable_surfel_mask=trainable_surfel_mask,
        )
        split_offset_scale = float(getattr(config, "densification_split_offset_scale", 0.3))

        finite_signal_np = (
            grad_pos_norm_np[np.isfinite(grad_pos_norm_np)]
            if grad_pos_norm_np.size > 0
            else np.zeros((0,), dtype=np.float32)
        )

        if finite_signal_np.size > 0:
            signal_min = float(np.min(finite_signal_np))
            signal_p50 = float(np.quantile(finite_signal_np, 0.50))
            signal_p90 = float(np.quantile(finite_signal_np, 0.90))
            signal_p95 = float(np.quantile(finite_signal_np, 0.95))
            signal_p98 = float(np.quantile(finite_signal_np, 0.98))
            signal_max = float(np.max(finite_signal_np))
        else:
            signal_min = signal_p50 = signal_p90 = signal_p95 = signal_p98 = signal_max = 0.0

        if position_abs_candidate_count > 0:
            active_grad = grad_pos_norm_np[position_abs_candidate_mask_np]
            grad_quantile_threshold = float(np.quantile(active_grad, densification_grad_quantile))
            grad_threshold = max(
                float(densification_grad_abs_min),
                grad_quantile_threshold,
                float(np.finfo(np.float32).tiny),
            )
        else:
            grad_threshold = max(
                float(densification_grad_abs_min),
                float(np.finfo(np.float32).tiny),
            )

        position_candidate_mask_np = (
                valid_denom_np
                & finite_grad
                & trainable_np
                & (grad_pos_norm_np >= grad_threshold)
        )
        curvature_candidate_mask_np = (
                valid_curvature_observation_np
                & trainable_np
                & (curvature_violation_mean_np >= curvature_violation_threshold)
        ) if curvature_enabled else np.zeros_like(trainable_np)
        combined_candidate_mask_np = position_candidate_mask_np | curvature_candidate_mask_np

        position_candidate_count = int(np.count_nonzero(position_candidate_mask_np))
        curvature_candidate_count = int(np.count_nonzero(curvature_candidate_mask_np))
        candidate_count = int(np.count_nonzero(combined_candidate_mask_np))

        if candidate_count == 0:
            if not np.any(valid_denom_np) and not np.any(valid_curvature_observation_np):
                densify_reason = "no_density_samples"
            else:
                densify_reason = "no_candidates_after_position_or_curvature"
        else:

            densification_result = make_under_reconstruction_clones(
                positions=positions,
                rotations=rotations,
                scales=scales,
                albedos=albedos,
                opacities=opacities,
                betas=betas,
                powers=powers,
                grad_position_np=avg_density_grad_vector_np,
                selection_score_np=grad_pos_norm_np,
                trainable_surfel_mask=trainable_surfel_mask,
                grad_threshold=grad_threshold,
                max_clone_fraction=float(getattr(config, "densification_max_new_fraction", 1.0)),
                clone_offset_scale=split_offset_scale,
                clone_scale_factor=float(getattr(config, "densification_split_scale_factor", math.sqrt(2.0))),
                min_clone_scale=float(getattr(config, "densification_scale_min", 8.0e-3)),
                exact_clone_scale_threshold=exact_clone_scale_threshold,
                curvature_violation_np=curvature_violation_mean_np,
                curvature_direction_uu_np=curvature_tensor_uu_np,
                curvature_direction_uv_np=curvature_tensor_uv_np,
                curvature_direction_vv_np=curvature_tensor_vv_np,
                curvature_violation_threshold=curvature_violation_threshold,
            )

            if densification_result is not None:
                new_block = densification_result.get("new", None)
                if new_block is not None:
                    n_new_from_densification = int(new_block["position"].shape[0])
                    clone_count = int(densification_result.get("clone_count", 0))
                    split_count = int(densification_result.get("split_count", 0))
                    position_split_count = int(
                        densification_result.get("position_split_count", 0)
                    )
                    curvature_split_count = int(
                        densification_result.get("curvature_split_count", 0)
                    )
                    densify_reason = (
                        f"densified_clone={clone_count}_split={split_count}"
                        f"_position={position_split_count}"
                        f"_curvature={curvature_split_count}"
                    )
                else:
                    densify_reason = "densification_result_without_new_block"
            else:
                densify_reason = "selected_candidates_but_densification_rejected_all"

        if densification_verbose:
            print(
                f"[Iter {iteration:04d}] Densification check | "
                f"reason={densify_reason}, "
                f"added={n_new_from_densification}, "
                f"pts={positions.shape[0]}, "
                f"valid_denom={valid_denom_count}, "
                f"valid_curvature={valid_curvature_count}, "
                f"finite={finite_count}, "
                f"trainable={trainable_count}, "
                f"above_abs={above_abs_count}, "
                f"position_candidates={position_candidate_count}, "
                f"curvature_candidates={curvature_candidate_count}, "
                f"combined_candidates={candidate_count}, "
                f"signal_min={signal_min:.3e}, "
                f"signal_p50={signal_p50:.3e}, "
                f"signal_p90={signal_p90:.3e}, "
                f"signal_p95={signal_p95:.3e}, "
                f"signal_p98={signal_p98:.3e}, "
                f"signal_max={signal_max:.3e}, "
                f"grad_q_thr={grad_quantile_threshold:.3e}, "
                f"grad_thr={grad_threshold:.3e}, "
                f"abs_thr={densification_grad_abs_min:.3e}, "
                f"curvature_thr={curvature_violation_threshold:.3e}, "
                f"scene_extent={scene_extent:.3e}, "
                f"exact_clone_scale_thr={exact_clone_scale_threshold:.3e}, "
                f"split_offset={split_offset_scale:.3e}"
            )
        elif n_new_from_densification > 0:
            if config.densification_verbose:
                clone_count = int(densification_result.get("clone_count", 0)) if densification_result is not None else 0
                split_count = int(densification_result.get("split_count", 0)) if densification_result is not None else 0
                position_split_count = int(
                    densification_result.get("position_split_count", 0)
                ) if densification_result is not None else 0
                curvature_split_count = int(
                    densification_result.get("curvature_split_count", 0)
                ) if densification_result is not None else 0
                print(
                    f"[Iter {iteration:04d}] Tangent split densification: "
                    f"adding {n_new_from_densification} surfels "
                    f"(clone={clone_count}, split={split_count}, "
                    f"position={position_split_count}, "
                    f"curvature={curvature_split_count}) | "
                    f"grad_thr={grad_threshold:.3e}, "
                    f"curvature_thr={curvature_violation_threshold:.3e}, "
                    f"exact_clone_scale_thr={exact_clone_scale_threshold:.3e}, "
                    f"split_offset={split_offset_scale:.3e}, "
                    f"abs_thr={densification_grad_abs_min:.3e}, "
                    f"pts={positions.shape[0]}"
                )

        return densification_result


def maybe_make_prune_indices(
        iteration: int,
        config: OptimizationConfig,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        trainable_surfel_mask: torch.Tensor,
        prune_after: int,
        prune_interval: int,
        reset_opacity_interval: int,
        opacity_prune_threshold: float,
        max_prune_fraction: float,
) -> tuple[np.ndarray, np.ndarray, List[int]]:
    scale_prune_indices = np.zeros((0,), dtype=np.int64)
    opacity_prune_indices = np.zeros((0,), dtype=np.int64)
    indices_to_remove_list: List[int] = []

    is_reset_iteration = (
            reset_opacity_interval > 0
            and iteration % reset_opacity_interval == 0
    )

    if prune_interval <= 0:
        return scale_prune_indices, opacity_prune_indices, indices_to_remove_list

    if not (iteration >= prune_after and iteration % prune_interval == 0 and not is_reset_iteration):
        return scale_prune_indices, opacity_prune_indices, indices_to_remove_list

    area_prune_indices = compute_prune_indices_by_degenerate_area(
        scales,
        min_area=config.min_surfel_area,
        trainable_mask=trainable_surfel_mask,
    )

    scale_prune_indices = area_prune_indices
    if scale_prune_indices.size > 0:
        indices_to_remove_list.extend(int(i) for i in scale_prune_indices)

    opacity_prune_indices = compute_prune_indices_by_opacity(
        opacities,
        min_opacity=opacity_prune_threshold,
        use_quantile=False,
        max_fraction_to_prune=max_prune_fraction,
    )
    if opacity_prune_indices.size > 0:
        indices_to_remove_list.extend(int(i) for i in opacity_prune_indices)

    return scale_prune_indices, opacity_prune_indices, indices_to_remove_list


def save_iteration_outputs(
        output_dir: Path,
        iteration: int,
        save_interval: int,
        final_iteration: int,
        all_camera_ids: List[str],
        forward_out: Dict[str, dict],
        adjoint_images: Dict[str, Any],
        renderer_settings: RendererSettingsConfig,
        save_rgb: bool = True,
        save_median_depth: bool = True,
        save_depth_distortion: bool = False,
        save_visible_normal: bool = False,
        save_normal_from_depth: bool = False,
        save_grad: bool = False,
        force: bool = False,
) -> None:
    save_interval = int(save_interval)

    if save_interval <= 0 and not force:
        return

    if (
            not force
            and iteration % save_interval != 0
            and iteration != final_iteration
    ):
        return

    for camera_name in all_camera_ids:
        camera_base_dir = output_dir / camera_name

        if save_rgb:
            camera_render_dir = camera_base_dir / "render"
            camera_render_dir.mkdir(parents=True, exist_ok=True)

            save_render(
                camera_render_dir / f"{iteration:04d}_render.png",
                get_forward_rgb(forward_out, camera_name),
            )

        if save_median_depth:
            camera_median_depth_dir = camera_base_dir / "median_depth"
            camera_median_depth_dir.mkdir(parents=True, exist_ok=True)

            save_median_depth_snapshot(
                camera_median_depth_dir / f"{iteration:04d}_median_depth.png",
                get_forward_median_depth(forward_out, camera_name),
                quantile=0.99,
                save_npy=False,
            )

        if save_depth_distortion:
            camera_depth_dir = camera_base_dir / "depth_distortion"
            camera_depth_dir.mkdir(parents=True, exist_ok=True)

            save_depth_distortion_snapshot(
                camera_depth_dir / f"{iteration:04d}_depth_distortion.png",
                get_forward_depth_distortion(forward_out, camera_name),
                quantile=0.99,
                save_npy=False,
            )

        if save_visible_normal:
            camera_visible_normal_dir = camera_base_dir / "visible_normal"
            camera_visible_normal_dir.mkdir(parents=True, exist_ok=True)

            save_normal_map_snapshot(
                camera_visible_normal_dir / f"{iteration:04d}_visible_normal.png",
                get_forward_visible_normal(forward_out, camera_name),
                save_npy=False,
            )

        if save_normal_from_depth:
            camera_depth_normal_dir = camera_base_dir / "normal_from_depth"
            camera_depth_normal_dir.mkdir(parents=True, exist_ok=True)

            save_normal_map_snapshot(
                camera_depth_normal_dir / f"{iteration:04d}_normal_from_depth.png",
                get_forward_normal_from_depth(forward_out, camera_name),
                save_npy=False,
            )

        if save_grad:
            adjoint_source_images = adjoint_images.get("adjoint_source")

            if adjoint_source_images is not None and camera_name in adjoint_source_images:
                camera_grad_dir = camera_base_dir / "grad"
                camera_grad_dir.mkdir(parents=True, exist_ok=True)

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

                save_gradient_sign_png_py(
                    camera_grad_dir / f"{iteration:04d}_grad_099.png",
                    grad_img_np,
                    adjoint_spp=renderer_settings.adjoint_passes,
                    abs_quantile=0.999,
                    flip_y=False,
                )


def write_metrics_header(csv_writer: csv.writer) -> None:
    csv_writer.writerow(
        [
            "iteration",
            "active_camera_name",
            "active_camera_count",
            "loss_average_camera_count",
            "loss_average_expected_camera_count",
            "loss_average_is_complete",
            "loss_rgb_mean",
            "loss_rgb_l2_mean",
            "loss_rgb_dssim_mean",
            "loss_depth_distortion_raw_mean",
            "loss_depth_distortion_weighted_mean",
            "loss_normal_consistency_raw_mean",
            "loss_normal_consistency_weighted_mean",
            "loss_opacity_prior_raw_mean",
            "loss_opacity_prior_weighted_mean",
            "loss_intra_slab_depth_raw_mean",
            "loss_intra_slab_depth_weighted_mean",
            "loss_curvature_scale_raw_mean",
            "loss_curvature_scale_weighted_mean",
            "loss_total_mean",
            "num_points",
            "densification_new_points",
            "densification_clone_points",
            "densification_split_points",
            "densification_position_split_points",
            "densification_curvature_split_points",
            "densification_clone_points_total",
            "densification_split_points_total",
            "densification_position_split_points_total",
            "densification_curvature_split_points_total",
            "densification_clone_points_active",
            "densification_split_points_active",
            "densification_position_split_points_active",
            "densification_curvature_split_points_active",
            "prune_scale_area_points",
            "prune_inactive_transport_points",
            "iteration_time_sec",
            "total_time_sec",
            "grad_position_renderer_norm",
            "grad_position_renderer_max",
            "grad_position_surface_regularizer_norm",
            "grad_position_surface_regularizer_max",
            "grad_position_total_norm",
            "grad_position_total_max",
            "grad_opacity_total_norm",
            "grad_opacity_total_max",
        ]
    )
