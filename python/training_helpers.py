from __future__ import annotations

import copy
import csv
import math
import select
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pale
import torch

import density_control as density
import io_utils
import losses
import optimizers
import render_hooks as render
from config import OptimizationConfig, RendererSettingsConfig
from terminal_output import print_status

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
    "total_alpha_loss_weighted",
    "total_depth_distortion_loss_raw",
    "total_depth_distortion_loss_weighted",
    "total_normal_loss_raw",
    "total_normal_loss_weighted",
    "total_intra_slab_depth_loss_raw",
    "total_intra_slab_depth_loss_weighted",
    "total_loss_value",
)


def make_zero_loss_values() -> dict[str, float]:
    return {loss_key: 0.0 for loss_key in LOSS_VALUE_KEYS}


def make_averaged_loss_state_from_camera_cache(
        latest_loss_values_by_camera: dict[str, dict[str, float]],
        expected_camera_ids: list[str],
) -> dict[str, Any]:
    available_camera_ids = [
        camera_name
        for camera_name in expected_camera_ids
        if camera_name in latest_loss_values_by_camera
    ]

    if not available_camera_ids:
        raise RuntimeError("Cannot compute averaged loss: no camera losses have been recorded.")

    averaged_loss_state: dict[str, Any] = {}

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


@lru_cache(maxsize=1)
def _camera_permutation(camera_count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(camera_count)


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
        # A fresh permutation per full camera loop samples without replacement,
        # as in 3DGS. Deriving it from the cycle also preserves checkpoint resume.
        cycle_index, cycle_offset = divmod(iteration - 1, len(training_camera_ids))
        order = _camera_permutation(len(training_camera_ids), config.camera_sampling_seed + cycle_index)
        camera_index = int(order[cycle_offset])
        return [training_camera_ids[camera_index]]

    raise RuntimeError(f"Unknown camera_sampling_mode: {config.camera_sampling_mode}")


def transport_prune_indices(scores, camera_samples, camera_count, min_score, trainable_mask):
    """Compare interval support in camera-pass units, excluding under-observed surfels.

    Per-surfel exposure handles points appended partway through the window.
    A zero threshold disables transport pruning.
    """
    scores = np.asarray(scores, dtype=np.float64)
    camera_samples = np.asarray(camera_samples)
    if min_score <= 0 or camera_count <= 0:
        return np.zeros(0, dtype=np.int64)
    mean_pass_score = scores * camera_count / np.maximum(camera_samples, 1)
    return np.flatnonzero(
        np.asarray(trainable_mask, dtype=bool)
        & (camera_samples >= camera_count)
        & (mean_pass_score < min_score)
    ).astype(np.int64)


def topology_updates_allowed(iteration: int, final_iteration: int, freeze_last_iterations: int) -> bool:
    """Keep the last N global iterations free of densification and all pruning."""
    return iteration <= final_iteration - freeze_last_iterations


def make_named_parameter_dict(
        positions: torch.nn.Parameter,
        rotation_delta: torch.nn.Parameter,
        scales: torch.nn.Parameter,
        albedos: torch.nn.Parameter,
        opacities: torch.nn.Parameter,
        betas: torch.nn.Parameter,
        powers: torch.nn.Parameter,
) -> dict[str, torch.nn.Parameter]:
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

    print_status(
        f"[Iter {iteration:04d}] Repaired non-finite gradient in {name}: "
        f"bad_count={bad_count}, first_bad_index={first_bad_index}, "
        f"first_bad_value={first_bad_value}, {repair_text}"
    )
    return bad_count


def repair_nonfinite_gradient_dict_inplace(
        tag: str,
        gradient_dict: dict[str, np.ndarray],
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
        densification_result: dict[str, Any] | None,
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
    parameters = make_named_parameter_dict(positions, rotations, scales, albedos, opacities, betas, powers)
    with torch.no_grad():
        for name, parameter in parameters.items():
            if name not in update:
                continue
            values = torch.as_tensor(update[name], device=parameter.device, dtype=parameter.dtype)
            values = values.reshape((idx.numel(), *parameter.shape[1:]))
            parameter[idx] = density.normalize_quaternions_torch(values) if name == "rotation" else values


def rebuild_optimizer_preserving_state(
        config: OptimizationConfig,
        old_optimizer: torch.optim.Optimizer,
        old_params: dict[str, torch.nn.Parameter],
        new_params: dict[str, torch.nn.Parameter],
        keep_mask_np: np.ndarray,
        source_index_for_new_np: np.ndarray | None = None,
        copy_source_state_to_new: bool = True,
) -> torch.optim.Optimizer:
    new_optimizer = optimizers.create_masked_optimizer(
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
                out = torch.zeros(
                    (new_n,) + tuple(value.shape[1:]),
                    device=new_p.device,
                    dtype=value.dtype,
                )

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
        snapshot: dict[str, Any] | None,
        keep_mask_np: np.ndarray,
        new_point_count: int,
        source_index_for_new_np: np.ndarray | None = None,
        copy_source_state_to_new: bool = False,
) -> dict[str, Any] | None:
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

    migrated: dict[str, Any] = {
        "point_count": new_n,
        "step": int(snapshot.get("step", 0)),
        # Legacy device snapshots predate these fields and use pure log-space moments.
        "use_log_scale": bool(snapshot.get("use_log_scale", True)),
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
    io_utils.save_render(output_path_png, vis)

    if save_npy:
        np.save(output_path_png.with_suffix(".npy"), normal_rgba)


def renders_output_dir(output_dir: Path) -> Path:
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    return renders_dir


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

    io_utils.save_render(output_path_png, vis)


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

    io_utils.save_render(output_path_png, vis)


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


def sum_gradient_dicts(*gradient_dicts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    active = [gradient_dict for gradient_dict in gradient_dicts if gradient_dict]
    if not active:
        return {}

    result: dict[str, np.ndarray] = {
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


def poll_hotkey() -> str | None:
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


def position_gradient_norm_stats_or_zero(
        gradient_dict: dict[str, np.ndarray],
) -> tuple[float, float]:
    if not gradient_dict or "position" not in gradient_dict:
        return 0.0, 0.0

    grad_position = np.asarray(gradient_dict["position"], dtype=np.float32, order="C")
    return gradient_l2_norm(grad_position), max_point_norm(grad_position)


def scale_gradient_dict(gradient_dict: dict[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
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
        decay_power: float = 1.5, # 1 = linear
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

    # With decay_power=2.0, quadratic decay is fast early and slows toward the end:
    # Progress through [start_iteration, end_iteration]:  0%    25%     50%    75%    100%
    # Remaining fraction of (initial - final):          100%   56.25%  25%    6.25%   0%
    # These percentages describe the gap, not the full threshold (unless final=0).
    return final_threshold + (initial_threshold - final_threshold) * (1.0 - t) ** decay_power


def cloning_max_scale_for_config(config: OptimizationConfig, scene_extent: float | None) -> float | None:
    if not config.densification_use_cloning:
        return None
    fraction = float(config.densification_clone_scale_fraction)
    if not math.isfinite(fraction) or fraction <= 0:
        raise ValueError("densification_clone_scale_fraction must be finite and positive")
    if scene_extent is None or not math.isfinite(scene_extent) or scene_extent <= 0:
        raise ValueError("Cloning requires a finite positive training-camera scene extent")
    boundary = fraction * scene_extent
    if not math.isfinite(boundary) or boundary <= 0:
        raise ValueError("Cloning size boundary must be finite and positive")
    return boundary


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

    # W carries accumulated opacity. Treat it as a detached confidence factor:
    # differentiate only the two normal vectors, not this multiplier.
    opacity = vis_rgba[..., 3]
    dot_nd = np.sum(vis * dep, axis=-1)
    loss_map = np.zeros_like(dot_nd, dtype=np.float32)
    loss_map[mask] = scale * (1.0 - opacity[mask] * dot_nd[mask])
    loss = float(loss_map.sum())

    dL_dvis = np.zeros_like(vis, dtype=np.float32)
    dL_ddep = np.zeros_like(dep, dtype=np.float32)
    dL_dvis[mask] = -scale * opacity[mask, None] * dep[mask]
    dL_ddep[mask] = -scale * opacity[mask, None] * vis[mask]

    return loss, dL_dvis, dL_ddep


def refetch_parameters_as_torch(
        renderer: pale.Renderer,
        device: torch.device,
) -> tuple[torch.nn.Parameter, ...]:
    updated = render.fetch_parameters(renderer)
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
        rotations.data.copy_(density.normalize_quaternions_torch(rotations.data))
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
        trainable_surfel_mask: torch.Tensor | None = None,
) -> None:
    verify_rotations_inplace(rotations)
    render.verify_scales_inplace(scales)
    render.verify_positions_inplace(positions)
    render.verify_albedos_inplace(albedos)
    render.verify_opacities_inplace(opacities)
    render.verify_beta_inplace(betas, trainable_surfel_mask=trainable_surfel_mask)


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

    print_status(f"[Iter {iteration:04d}] Hotkey 'g' pressed -> saved gradients to:\n  {csv_path}\n  {npz_path}")


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
        camera_ids: list[str],
        densification_origins: np.ndarray | None = None,
        primitive_ages: np.ndarray | None = None,
        densification_position_signals: np.ndarray | None = None,
        densification_position_sample_counts: np.ndarray | None = None,
        densification_position_threshold: float | np.ndarray | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = renders_output_dir(output_dir)
    final_images = renderer.render_forward()

    for camera_name in camera_ids:
        img_np = render.get_forward_rgb(final_images, camera_name)
        io_utils.save_render(renders_dir / f"render_final_{camera_name}.png", img_np)
        io_utils.save_linear_raw_exr(
            renders_dir / f"linear_raw_final_{camera_name}.exr",
            render.get_forward_linear_rgb(final_images, camera_name),
        )

        depth_np = render.get_forward_depth_distortion(final_images, camera_name)
        save_depth_distortion_snapshot(
            renders_dir / f"depth_distortion_final_{camera_name}.png",
            depth_np,
            quantile=0.99,
            save_npy=False,
        )

        median_depth_np = render.get_forward_median_depth(final_images, camera_name)
        save_median_depth_snapshot(
            renders_dir / f"median_depth_final_{camera_name}.png",
            median_depth_np,
            quantile=0.99,
            save_npy=False,
        )

        visible_normal_np = render.get_forward_visible_normal(final_images, camera_name)
        save_normal_map_snapshot(
            renders_dir / f"visible_normal_final_{camera_name}.png",
            visible_normal_np,
            save_npy=False,
        )

        normal_from_depth_np = render.get_forward_normal_from_depth(final_images, camera_name)
        save_normal_map_snapshot(
            renders_dir / f"normal_from_depth_final_{camera_name}.png",
            normal_from_depth_np,
            save_npy=False,
        )

    ply_path = output_dir / "points_final.ply"
    io_utils.save_gaussians_to_ply(
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
        densification_position_signals=densification_position_signals,
        densification_position_sample_counts=densification_position_sample_counts,
        densification_position_threshold=densification_position_threshold,
    )

    print_status(
        f"[Iter {iteration:04d}] Hotkey 's' pressed -> saved "
        f"render_final_<camera>.png, linear_raw_final_<camera>.exr, depth_distortion_final_<camera>.png, "
        f"median_depth_final_<camera>.png, visible_normal_final_<camera>.png, "
        f"normal_from_depth_final_<camera>.png, and points_final.ply"
    )
    return ply_path


def should_save_point_cloud_snapshot(
        iteration: int,
        save_interval: int,
        next_densification_iteration: int | None,
        before_densification: bool,
        first_iteration: bool = False,
) -> bool:
    if save_interval <= 0:
        return False
    if first_iteration:
        return True
    if before_densification:
        return (
            next_densification_iteration is not None
            and iteration + 1 == next_densification_iteration
        )
    return iteration % save_interval == 0


def should_save_image_snapshot(
        iteration: int,
        save_interval: int,
        final_iteration: int,
        next_densification_iteration: int | None,
        first_iteration: bool = False,
) -> bool:
    if first_iteration:
        return True
    if save_interval <= 0:
        return False
    return (
        iteration == final_iteration
        or iteration % save_interval == 0
    )


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
        densification_position_signals: np.ndarray | None = None,
        densification_position_sample_counts: np.ndarray | None = None,
        densification_position_threshold: float | np.ndarray | None = None,
) -> Path:
    points_dir = output_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)
    ply_path = points_dir / f"iter_{iteration:05d}_points.ply"
    io_utils.save_gaussians_to_ply(
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
        densification_position_signals=densification_position_signals,
        densification_position_sample_counts=densification_position_sample_counts,
        densification_position_threshold=densification_position_threshold,
    )
    return ply_path


def compute_snapshot_adjoint_images(
        renderer: pale.Renderer,
        forward_out: dict[str, dict],
        target_images: dict[str, np.ndarray],
        camera_ids: list[str],
) -> dict[str, Any]:
    loss_grad_images: dict[str, np.ndarray] = {}

    for camera_name in camera_ids:
        if camera_name not in target_images:
            continue

        current_rgb_np = render.get_forward_linear_rgb(forward_out, camera_name)
        target_rgb_np = target_images[camera_name]
        loss_grad_images[camera_name] = losses.compute_l2_grad(current_rgb_np, target_rgb_np)

    if not loss_grad_images:
        return {}

    _, adjoint_images = renderer.render_backward(loss_grad_images)
    return adjoint_images


def load_target_images(
        renderer: pale.Renderer,
        dataset_path: Path,
        target_color_space: str = "auto",
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    target_path = Path(dataset_path)
    if not target_path.is_dir():
        raise RuntimeError(f"Target path '{target_path}' must be a directory when multiple cameras are used.")

    training_camera_ids = render.get_training_camera_names(renderer)
    all_camera_ids = render.get_all_camera_names(renderer)
    target_images: dict[str, np.ndarray] = {}
    images_path = target_path / "images"
    if not images_path.is_dir():
        raise RuntimeError(f"Target dataset is missing images directory: {images_path}")

    print_status(
        f"Loading {len(training_camera_ids)} target images from directory: {target_path} "
        f"({target_color_space} -> linear sRGB training)"
    )
    for camera_name in training_camera_ids:
        candidates = sorted(
            path for path in images_path.iterdir()
            if path.is_file()
            and path.stem == camera_name
            and path.suffix.lower() in io_utils.SUPPORTED_TARGET_IMAGE_SUFFIXES
        )
        if not candidates:
            suffixes = ", ".join(io_utils.SUPPORTED_TARGET_IMAGE_SUFFIXES)
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

        target_images[camera_name] = io_utils.load_target_image(
            image_path,
            color_space=target_color_space,
            log_color_interpretation=False,
        )

    image_shapes = {image.shape for image in target_images.values()}
    shape_summary = (
        str(next(iter(image_shapes)))
        if len(image_shapes) == 1
        else f"{len(image_shapes)} distinct shapes"
    )
    print_status(f"Loaded {len(target_images)} target images ({shape_summary}).")

    return target_images, training_camera_ids, all_camera_ids


def create_torch_parameters_from_initial(
        initial_params: dict[str, np.ndarray],
        device: torch.device,
) -> tuple[torch.nn.Parameter, ...]:
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


def compute_initial_losses_and_save_outputs(
        output_dir: Path,
        initial_images: dict[str, dict],
        target_images: dict[str, np.ndarray],
        all_camera_ids: list[str],
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
        depth_distortion_weight: float,
        normal_consistency_weight: float,
        intra_slab_depth_weight: float,
        use_depth_distortion: bool,
        use_normal_consistency: bool,
        use_intra_slab_depth: bool,
        save_linear_raw: bool = False,
        alpha_loss_weighted: float = 0.0,
) -> tuple[float, ...]:
    initial_points_path = output_dir / "initial_points.ply"
    io_utils.save_gaussians_to_ply(
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
    print_status(f"Initial parameters written to PLY: {initial_points_path}")

    initial_rgb_loss = 0.0
    initial_depth_distortion_loss_raw = 0.0
    initial_normal_loss_raw = 0.0
    initial_intra_slab_depth_loss_raw = 0.0

    renders_dir = renders_output_dir(output_dir)
    for camera_name in all_camera_ids:
        img_np = render.get_forward_rgb(initial_images, camera_name)
        img_linear_np = render.get_forward_linear_rgb(initial_images, camera_name)
        (renders_dir / camera_name).mkdir(parents=True, exist_ok=True)
        io_utils.save_render(renders_dir / f"render_initial_{camera_name}.png", img_np)
        if save_linear_raw:
            io_utils.save_linear_raw_exr(renders_dir / f"linear_raw_initial_{camera_name}.exr", img_linear_np)

        if camera_name not in target_images:
            print_status(f"Warning: no target image found for camera '{camera_name}', skipping target save and loss.")
            continue

        tgt_np = target_images[camera_name]
        rgb_loss_value = losses.compute_l2_loss(
            img_linear_np,
            tgt_np,
        )
        initial_rgb_loss += rgb_loss_value
        io_utils.save_render(
            renders_dir / f"render_target_{camera_name}.png",
            io_utils.linear_to_srgb(tgt_np),
        )

        if use_depth_distortion:
            initial_depth_distortion_loss_raw += float(render.get_forward_depth_distortion(initial_images, camera_name).mean())

        if use_normal_consistency:
            visible_normal = render.get_forward_visible_normal(initial_images, camera_name)
            normal_from_depth = render.get_forward_normal_from_depth(initial_images, camera_name)
            raw_normal_loss_value, _, _ = compute_normal_consistency_loss_and_adjoints(
                visible_normal,
                normal_from_depth,
                1.0,
            )
            initial_normal_loss_raw += raw_normal_loss_value


        if use_intra_slab_depth:
            loss_map = render.get_forward_intra_slab_depth(initial_images, camera_name)
            active_count = max(
                1,
                int(render.get_forward_intra_slab_depth_active_slab_count(
                    initial_images, camera_name
                ).sum(dtype=np.uint64)),
            )
            initial_intra_slab_depth_loss_raw += float(loss_map.sum() / active_count)


    initial_depth_distortion_loss_weighted = depth_distortion_weight * initial_depth_distortion_loss_raw
    initial_normal_loss_weighted = normal_consistency_weight * initial_normal_loss_raw
    initial_intra_slab_depth_loss_weighted = (
        intra_slab_depth_weight * initial_intra_slab_depth_loss_raw
    )

    initial_total_loss = (
            initial_rgb_loss
            + alpha_loss_weighted
            + initial_depth_distortion_loss_weighted
            + initial_normal_loss_weighted
            + initial_intra_slab_depth_loss_weighted
    )

    return (
        initial_rgb_loss,
        initial_depth_distortion_loss_raw,
        initial_depth_distortion_loss_weighted,
        initial_normal_loss_raw,
        initial_normal_loss_weighted,
        initial_intra_slab_depth_loss_raw,
        initial_intra_slab_depth_loss_weighted,
        initial_total_loss,
        alpha_loss_weighted,
    )


def print_loss_summary(
        prefix: str,
        rgb_loss: float,
        depth_distortion_loss_raw: float,
        depth_distortion_loss_weighted: float,
        normal_loss_raw: float,
        normal_loss_weighted: float,
        intra_slab_depth_loss_raw: float,
        intra_slab_depth_loss_weighted: float,
        total_loss: float,
        alpha_loss_weighted: float = 0.0,
) -> None:
    print_status(f"{prefix} RGB loss                               : {rgb_loss:.6e}")
    print_status(f"{prefix} alpha loss (weighted)                  : {alpha_loss_weighted:.6e}")
    print_status(f"{prefix} depth distortion loss (raw)            : {depth_distortion_loss_raw:.6e}")
    print_status(f"{prefix} depth distortion loss (weighted)       : {depth_distortion_loss_weighted:.6e}")
    print_status(f"{prefix} normal consistency loss (raw)          : {normal_loss_raw:.6e}")
    print_status(f"{prefix} normal consistency loss (weighted)     : {normal_loss_weighted:.6e}")
    print_status(f"{prefix} intra-slab depth loss (raw)            : {intra_slab_depth_loss_raw:.6e}")
    print_status(f"{prefix} intra-slab depth loss (weighted)       : {intra_slab_depth_loss_weighted:.6e}")
    print_status(f"{prefix} total loss                             : {total_loss:.6e}")


def compute_surface_regularizer_losses_and_adjoints(
        forward_out: dict[str, dict],
        training_camera_ids: list[str],
        depth_distortion_weight: float,
        normal_consistency_weight: float,
        intra_slab_depth_weight: float,
        use_depth_distortion: bool,
        use_normal_consistency: bool,
        use_intra_slab_depth: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = make_zero_loss_values()
    result.update({
        "depth_distortion_grad_images": {},
        "visible_normal_adjoints": {},
        "depth_normal_adjoints": {},
        "depth_distortion_maps_for_logging": {},
        "intra_slab_depth_grad_images": {},
        "intra_slab_depth_maps_for_logging": {},
        "per_camera_loss_values": {},
    })

    for camera_name in training_camera_ids:
        camera_loss_values = make_zero_loss_values()

        if use_depth_distortion:
            current_depth_distortion_np = render.get_forward_depth_distortion(forward_out, camera_name)
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
            visible_normal = render.get_forward_visible_normal(forward_out, camera_name)
            normal_from_depth = render.get_forward_normal_from_depth(forward_out, camera_name)

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


        if use_intra_slab_depth:
            intra_slab_depth_np = render.get_forward_intra_slab_depth(forward_out, camera_name)
            active_slab_count_np = render.get_forward_intra_slab_depth_active_slab_count(
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


        for loss_key in LOSS_VALUE_KEYS:
            result[loss_key] += camera_loss_values[loss_key]

        result["per_camera_loss_values"][camera_name] = camera_loss_values

    return result


def extract_total_gradient_arrays(
        total_gradients: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.asarray(total_gradients[name], dtype=np.float32, order="C")
                 for name in PARAMETER_NAMES[:-1])


def should_accumulate_densification_statistics(
        iteration: int,
        densification_interval: int,
        densification_cycle_start_iteration: int,
        densification_stats_skip_iterations: int,
) -> bool:
    """The same window gate is used by the host and device reducers."""
    phase = max(0, int(iteration) - int(densification_cycle_start_iteration))
    return densification_interval > 0 and (
        densification_interval <= 1
        or (phase >= densification_stats_skip_iterations and phase != 0)
    )


def download_densification_statistics(
        renderer: pale.Renderer,
        point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download a window snapshot; reading does not reset device accumulation."""
    state = renderer.get_densification_stats()
    arrays = []
    for key, components in (("position_sum", 1), ("position_count", 1),
                            ("direction_sum", 3)):
        values = np.asarray(state[key], dtype=np.float32)
        if values.size != point_count * components:
            raise RuntimeError(f"Densification {key} has shape {values.shape} for {point_count} points")
        arrays.append(values.reshape(point_count, components))
    return tuple(arrays)


def upload_densification_statistics(
        renderer: pale.Renderer,
        position_sum: np.ndarray,
        position_count: np.ndarray,
        direction_sum: np.ndarray,
) -> None:
    """Restore canonical row order after a topology change within a window."""
    renderer.upload_densification_stats({
        "position_sum": np.ascontiguousarray(position_sum.reshape(-1), dtype=np.float32),
        "position_count": np.ascontiguousarray(position_count.reshape(-1), dtype=np.float32),
        "direction_sum": np.ascontiguousarray(direction_sum, dtype=np.float32),
    })


def update_densification_statistics(
        iteration: int,
        densification_interval: int,
        densification_cycle_start_iteration: int,
        densification_stats_skip_iterations: int,
        densify_position_grad_accum_np: np.ndarray,
        densify_position_grad_denom_np: np.ndarray,
        densify_position_grad_vector_accum_np: np.ndarray,
        rotations: torch.Tensor,
        trainable_surfel_mask: torch.Tensor,
        densify_position_grad_per_camera_np: np.ndarray,
        densify_position_grad_per_camera_count_np: np.ndarray,
        densification_downweight_normal_gradients: bool = False,
        densification_image_space: bool = False,
        densify_image_grad_per_camera_np: np.ndarray | None = None,
) -> None:
    if not should_accumulate_densification_statistics(
            iteration, densification_interval, densification_cycle_start_iteration,
            densification_stats_skip_iterations):
        return
    gradients = np.asarray(densify_position_grad_per_camera_np, dtype=np.float32)
    counts = np.asarray(densify_position_grad_per_camera_count_np, dtype=np.uint32)
    if gradients.ndim != 3 or gradients.shape[2] != 3:
        raise RuntimeError("Densification gradients must have shape (N, C, 3)")
    point_count, camera_count, _ = gradients.shape
    if counts.shape != (point_count, camera_count):
        raise RuntimeError("Densification camera counts must have shape (N, C)")
    if (densify_position_grad_accum_np.shape != (point_count, 1)
            or densify_position_grad_denom_np.shape != (point_count, 1)
            or densify_position_grad_vector_accum_np.shape != (point_count, 3)):
        raise RuntimeError("Densification accumulators must match the current point count")
    trainable = trainable_surfel_mask.detach().cpu().numpy().astype(bool).reshape(-1)
    if trainable.shape != (point_count,):
        raise RuntimeError("Densification mask must match the current point count")

    visible = counts > 0
    denominator = np.maximum(visible.sum(axis=1), 1).astype(np.float32)
    weights = visible.astype(np.float32)
    # Image-space scores already have their metric. Only construct/download the
    # tangent frame for world-space scoring or explicit normal downweighting.
    if not densification_image_space or densification_downweight_normal_gradients:
        tu, tv, normal = density.quaternion_to_tangent_frame_torch(rotations.detach())
        gu = np.einsum("nci,ni->nc", gradients, tu.cpu().numpy())
        gv = np.einsum("nci,ni->nc", gradients, tv.cpu().numpy())
        tangent_norm = np.nan_to_num(np.sqrt(gu*gu + gv*gv), nan=0., posinf=0., neginf=0.)
        if densification_downweight_normal_gradients:
            gn = np.einsum("nci,ni->nc", gradients, normal.cpu().numpy())
            total_norm = np.sqrt(tangent_norm*tangent_norm + gn*gn)
            weights *= np.nan_to_num(tangent_norm / np.maximum(total_norm, 1.e-12),
                                     nan=0., posinf=0., neginf=0.)
        score_norm = tangent_norm
    if densification_image_space:
        image_gradients = np.asarray(densify_image_grad_per_camera_np, dtype=np.float32)
        if image_gradients.shape != gradients.shape:
            raise RuntimeError("Image-space gradient statistics must have shape (N, C, 3)")
        score_norm = np.nan_to_num(np.linalg.norm(image_gradients[..., :2], axis=2),
                                  nan=0., posinf=0., neginf=0.)
    signal = np.nan_to_num((score_norm * weights).sum(axis=1) / denominator,
                           nan=0., posinf=0., neginf=0.)
    selected = trainable & (signal > 0)
    densify_position_grad_accum_np[selected, 0] += signal[selected]
    densify_position_grad_denom_np[selected, 0] += 1

    # Retain signed world-space sums for diagnostics, independent of split placement.
    direction = np.nan_to_num(np.einsum("nci,nc->ni", gradients, weights) / denominator[:, None],
                              nan=0., posinf=0., neginf=0.)
    direction_norm = np.linalg.norm(direction, axis=1)
    selected = trainable & np.isfinite(direction_norm) & (direction_norm > 0)
    densify_position_grad_vector_accum_np[selected] += direction[selected]


def position_densification_snapshot_statistics(
        densify_position_grad_accum_np: np.ndarray,
        densify_position_grad_denom_np: np.ndarray,
        trainable_surfel_mask: torch.Tensor,
        densification_grad_abs_min: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return window-averaged signals, sample counts, and absolute thresholds."""
    accum = np.asarray(densify_position_grad_accum_np, dtype=np.float32).reshape(-1)
    denom = np.asarray(densify_position_grad_denom_np, dtype=np.float32).reshape(-1)
    if accum.shape != denom.shape or accum.size != trainable_surfel_mask.numel():
        raise RuntimeError(
            "Position densification snapshot arrays must have matching point counts: "
            f"accum={accum.shape}, denom={denom.shape}, trainable={trainable_surfel_mask.numel()}"
        )

    valid = np.isfinite(denom) & (denom > 0.0)
    signal = np.zeros(accum.shape, dtype=np.float32)
    signal[valid] = accum[valid] / denom[valid]
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    threshold = max(float(densification_grad_abs_min), float(np.finfo(np.float32).tiny))

    sample_counts = np.clip(
        np.rint(np.nan_to_num(denom, nan=0.0, posinf=0.0, neginf=0.0)),
        0.0,
        float(np.iinfo(np.uint32).max),
    ).astype(np.uint32)
    # Compare and persist float32 thresholds so a PLY round-trip cannot move a
    # primitive across a selection boundary through scalar rounding.
    threshold = float(np.float32(threshold))
    return signal, sample_counts, threshold


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
        densify_after: int,
        densification_interval: int,
        densification_verbose: bool,
        densification_grad_abs_min: float,
        force_densification: bool = False,
        densification_scene_extent: float | None = None,
        scheduled_prune_indices: list[int] | None = None,
) -> dict | None:
    if densification_interval <= 0 or iteration < densify_after:
        return None
    if not force_densification and (
            iteration <= densification_interval or (iteration - 1) % densification_interval != 0):
        return None

    clone_max_scale = cloning_max_scale_for_config(config, densification_scene_extent)
    prune_after = config.prune_after if config.prune_after >= 0 else config.prune_interval
    if scheduled_prune_indices is not None:
        # Use the complete decision (area, opacity, and transport) made on the
        # original population. Do not mutate the mask used for optimization.
        trainable_surfel_mask = trainable_surfel_mask.detach().clone()
        trainable_surfel_mask[torch.as_tensor(
            scheduled_prune_indices, device=trainable_surfel_mask.device, dtype=torch.long,
        )] = False
    elif config.prune_interval > 0 and iteration >= prune_after and iteration % config.prune_interval == 0:
        # Do not recreate parents that this same topology update will prune.
        trainable_surfel_mask = trainable_surfel_mask.detach().clone()
        if clone_max_scale is not None:
            area_indices = density.compute_prune_indices_by_degenerate_area(
                scales, min_area=config.min_surfel_area, trainable_mask=trainable_surfel_mask,
            )
            trainable_surfel_mask[torch.as_tensor(area_indices, device=positions.device)] = False
        trainable_surfel_mask &= ~compute_opacity_prune_mask(
            opacities, trainable_surfel_mask, config.min_surfel_opacity,
        )

    signals, counts, threshold = position_densification_snapshot_statistics(
        densify_position_grad_accum_np, densify_position_grad_denom_np,
        trainable_surfel_mask, densification_grad_abs_min,
    )
    candidates = (counts > 0) & (signals >= threshold)
    candidate_count = np.count_nonzero(candidates & trainable_surfel_mask.detach().cpu().numpy().reshape(-1))
    result = None
    if candidate_count:
        result = density.make_under_reconstruction_clones(
            positions=positions, rotations=rotations, scales=scales, albedos=albedos,
            opacities=opacities, betas=betas, powers=powers,
            selection_score_np=signals / threshold, trainable_surfel_mask=trainable_surfel_mask,
            grad_threshold=1.0, max_clone_fraction=config.densification_max_new_fraction,
            clone_scale_factor=config.densification_split_scale_factor,
            min_clone_scale=config.densification_scale_min, clone_max_scale=clone_max_scale,
        )
    if densification_verbose:
        # Percentiles and device-to-host counts are diagnostics, not selection work.
        quantiles = np.quantile(signals, [0, .5, .9, .95, .98, 1]) if signals.size else np.zeros(6)
        summary = ", ".join(f"signal_{label}={value:.3e}" for label, value in
                            zip(("min", "p50", "p90", "p95", "p98", "max"), quantiles))
        clones = result["clone_count"] if result else 0
        splits = result["split_count"] if result else 0
        print_status(
            f"[Iter {iteration:04d}] Densification check | pts={len(positions)}, "
            f"position_candidates={candidate_count}, added={clones + splits}, "
            f"clone={clones}, split={splits}, grad_thr={threshold:.3e}, "
            f"{summary}, split_sampling=beta_footprint"
        )
    return result


@torch.no_grad()
def compute_opacity_prune_mask(opacities: torch.Tensor, trainable_surfel_mask: torch.Tensor, threshold: float) -> torch.Tensor:
    """Select trainable surfels with stored opacity strictly below the threshold."""
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Opacity prune threshold must be finite and in [0, 1], got {threshold}")
    if opacities.ndim != 1 and not (opacities.ndim == 2 and opacities.shape[1] == 1):
        raise ValueError(f"Expected opacities with shape (N,) or (N, 1), got {tuple(opacities.shape)}")
    opacity_values = opacities.detach().reshape(-1)
    trainable_mask = trainable_surfel_mask.detach().to(device=opacity_values.device, dtype=torch.bool).reshape(-1)
    if trainable_mask.shape != opacity_values.shape:
        raise ValueError(f"Opacity/mask shape mismatch: {tuple(opacity_values.shape)} vs {tuple(trainable_mask.shape)}")
    if threshold == 0.0:
        return torch.zeros_like(trainable_mask)
    if bool(torch.any(trainable_mask & ~torch.isfinite(opacity_values)).item()):
        raise ValueError("Opacity pruning requires finite opacities for trainable surfels")
    return trainable_mask & (opacity_values < threshold)


def maybe_make_prune_indices(
        iteration: int,
        config: OptimizationConfig,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        trainable_surfel_mask: torch.Tensor,
        prune_after: int,
        prune_interval: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return area indices, opacity indices, and their deduplicated union."""
    if prune_interval <= 0 or iteration < prune_after or iteration % prune_interval != 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), []
    if scales.shape[0] != opacities.shape[0]:
        raise ValueError(f"Scale/opacity point-count mismatch: {scales.shape[0]} vs {opacities.shape[0]}")
    scale_prune_indices = density.compute_prune_indices_by_degenerate_area(
        scales,
        min_area=config.min_surfel_area,
        trainable_mask=trainable_surfel_mask,
    )
    opacity_prune_mask = compute_opacity_prune_mask(opacities, trainable_surfel_mask, config.min_surfel_opacity)
    opacity_prune_indices = torch.nonzero(opacity_prune_mask, as_tuple=False).reshape(-1).cpu().numpy().astype(np.int64, copy=False)
    indices_to_remove = np.union1d(scale_prune_indices, opacity_prune_indices).astype(np.int64, copy=False)
    return scale_prune_indices, opacity_prune_indices, indices_to_remove.tolist()


def save_iteration_outputs(
        output_dir: Path,
        iteration: int,
        save_interval: int,
        final_iteration: int,
        all_camera_ids: list[str],
        forward_out: dict[str, dict],
        adjoint_images: dict[str, Any],
        renderer_settings: RendererSettingsConfig,
        save_rgb: bool = True,
        save_median_depth: bool = True,
        save_depth_distortion: bool = False,
        save_visible_normal: bool = False,
        save_normal_from_depth: bool = False,
        save_grad: bool = False,
        force: bool = False,
        save_linear_raw: bool = False,
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

    renders_dir = renders_output_dir(output_dir)
    for camera_name in all_camera_ids:
        camera_base_dir = renders_dir / camera_name

        if save_linear_raw:
            io_utils.save_linear_raw_exr(
                camera_base_dir / "linear_raw" / f"{iteration:04d}_linear_raw.exr",
                render.get_forward_linear_rgb(forward_out, camera_name),
            )

        if save_rgb:
            camera_render_dir = camera_base_dir / "render"
            camera_render_dir.mkdir(parents=True, exist_ok=True)

            io_utils.save_render(
                camera_render_dir / f"{iteration:04d}_render.png",
                render.get_forward_rgb(forward_out, camera_name),
            )

        if save_median_depth:
            camera_median_depth_dir = camera_base_dir / "median_depth"
            camera_median_depth_dir.mkdir(parents=True, exist_ok=True)

            save_median_depth_snapshot(
                camera_median_depth_dir / f"{iteration:04d}_median_depth.png",
                render.get_forward_median_depth(forward_out, camera_name),
                quantile=0.99,
                save_npy=False,
            )

        if save_depth_distortion:
            camera_depth_dir = camera_base_dir / "depth_distortion"
            camera_depth_dir.mkdir(parents=True, exist_ok=True)

            save_depth_distortion_snapshot(
                camera_depth_dir / f"{iteration:04d}_depth_distortion.png",
                render.get_forward_depth_distortion(forward_out, camera_name),
                quantile=0.99,
                save_npy=False,
            )

        if save_visible_normal:
            camera_visible_normal_dir = camera_base_dir / "visible_normal"
            camera_visible_normal_dir.mkdir(parents=True, exist_ok=True)

            save_normal_map_snapshot(
                camera_visible_normal_dir / f"{iteration:04d}_visible_normal.png",
                render.get_forward_visible_normal(forward_out, camera_name),
                save_npy=False,
            )

        if save_normal_from_depth:
            camera_depth_normal_dir = camera_base_dir / "normal_from_depth"
            camera_depth_normal_dir.mkdir(parents=True, exist_ok=True)

            save_normal_map_snapshot(
                camera_depth_normal_dir / f"{iteration:04d}_normal_from_depth.png",
                render.get_forward_normal_from_depth(forward_out, camera_name),
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

                io_utils.save_gradient_sign_png_py(
                    camera_grad_dir / f"{iteration:04d}_grad_099.png",
                    grad_img_np,
                    adjoint_spp=renderer_settings.adjoint_passes,
                    abs_quantile=0.999,
                    flip_y=False,
                )
