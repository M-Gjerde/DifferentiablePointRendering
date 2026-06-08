from __future__ import annotations

import copy
import csv
import select
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pale
import torch

from config import OptimizationConfig, RendererSettingsConfig
from density_control import (
    compute_prune_indices_by_degenerate_scale,
    compute_prune_indices_by_opacity,
    make_under_reconstruction_clones,
    project_gradient_to_surfel_tangent_plane_np,
)
from io_utils import (
    load_target_image,
    save_gaussians_to_ply,
    save_gradient_sign_png_py,
    save_render,
)
from losses import compute_l2_grad, compute_l2_loss
from optimizers import create_masked_optimizer
from render_hooks import *

PARAMETER_NAMES = (
    "position",
    "tangent_u",
    "tangent_v",
    "scale",
    "albedo",
    "opacity",
    "beta",
    "power",
)


def as_config_float(value: Any) -> float:
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError(f"Expected scalar or single-item tuple, got {value}")
        value = value[0]
    return float(value)


def make_named_parameter_dict(
        positions: torch.nn.Parameter,
        tangent_u: torch.nn.Parameter,
        tangent_v: torch.nn.Parameter,
        scales: torch.nn.Parameter,
        albedos: torch.nn.Parameter,
        opacities: torch.nn.Parameter,
        betas: torch.nn.Parameter,
        powers: torch.nn.Parameter,
) -> Dict[str, torch.nn.Parameter]:
    return {
        "position": positions,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
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
        tangent_u: torch.nn.Parameter,
        tangent_v: torch.nn.Parameter,
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
        if "tangent_u" in update:
            v = torch.as_tensor(update["tangent_u"], device=device, dtype=tangent_u.dtype)
            tangent_u.data[idx] = v.view_as(tangent_u.data[idx])
        if "tangent_v" in update:
            v = torch.as_tensor(update["tangent_v"], device=device, dtype=tangent_v.dtype)
            tangent_v.data[idx] = v.view_as(tangent_v.data[idx])
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
        new_params["tangent_u"],
        new_params["tangent_v"],
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
        grad_tangent_u_np: np.ndarray,
        grad_tangent_v_np: np.ndarray,
        grad_scales_np: np.ndarray,
        grad_albedos_np: np.ndarray,
        grad_opacities_np: np.ndarray,
        grad_betas_np: np.ndarray,
) -> None:
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
        f"{format_stat('tangent_u')}, "
        f"{format_stat('tangent_v')}, "
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

def make_loss_breakdown(loss_state: Dict[str, Any]) -> Dict[str, float]:
    rgb_loss = float(loss_state["total_rgb_loss_value"])
    depth_weighted = float(loss_state["total_depth_distortion_loss_weighted"])
    normal_weighted = float(loss_state["total_normal_loss_weighted"])
    visibility_opacity_weighted = float(loss_state["total_visibility_weighted_opacity_loss_weighted"])

    after_depth = rgb_loss + depth_weighted
    after_normal = after_depth + normal_weighted
    after_visibility_opacity = after_normal + visibility_opacity_weighted
    regularizer_total = depth_weighted + normal_weighted + visibility_opacity_weighted

    return {
        "before_regularizers": rgb_loss,
        "after_depth_distortion": after_depth,
        "after_normal_consistency": after_normal,
        "after_visibility_opacity": after_visibility_opacity,
        "regularizer_total": regularizer_total,
        "total": float(loss_state["total_loss_value"]),
    }


def format_loss_breakdown(loss_state: Dict[str, Any]) -> str:
    rgb_loss = float(loss_state["total_rgb_loss_value"])
    depth_weighted = float(loss_state["total_depth_distortion_loss_weighted"])
    normal_weighted = float(loss_state["total_normal_loss_weighted"])
    visibility_opacity_weighted = float(loss_state["total_visibility_weighted_opacity_loss_weighted"])
    total_loss = float(loss_state["total_loss_value"])

    after_depth = rgb_loss + depth_weighted
    after_normal = after_depth + normal_weighted
    after_visibility_opacity = after_normal + visibility_opacity_weighted
    regularizer_total = depth_weighted + normal_weighted + visibility_opacity_weighted

    return (
        "Loss stack:\n"
        f"  {'RGB only':<28} {rgb_loss:>12.3e}\n"
        f"  {'+ depth distortion':<28} {after_depth:>12.3e}  "
        f"(+{depth_weighted:.3e})\n"
        f"  {'+ normal consistency':<28} {after_normal:>12.3e}  "
        f"(+{normal_weighted:.3e})\n"
        f"  {'+ visibility opacity':<28} {after_visibility_opacity:>12.3e}  "
        f"(+{visibility_opacity_weighted:.3e})\n"
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
        total_time: float,
        num_points: int,
        loss_state: Dict[str, Any],
        lr_position: float,
        active_depth_distortion_weight: float,
        grad_pos_rms: float,
        grad_tanu_rms: float,
        grad_tanv_rms: float,
        grad_scale_rms: float,
        grad_albedo_rms: float,
        grad_opacity_rms: float,
        grad_beta_rms: float,
        grad_pos_max: float,
        grad_tanu_max: float,
        grad_tanv_max: float,
        grad_scale_max: float,
        grad_albedo_max: float,
        grad_opacity_max: float,
        grad_beta_max: float,
) -> str:
    iteration_rate = 1.0 / max(iteration_time, 1.0e-12)

    return (
        f"\n[Iter {iteration:04d}/{total_iterations}] "
        f"time={iteration_time:.3f}s total={total_time:.1f}s "
        f"it/s={iteration_rate:.2f} pts={num_points} adaptive_lr_pos={lr_position}\n"
        f"  losses:"
        f" rgb={loss_state['total_rgb_loss_value']:.3e}"
        f" depth_raw={loss_state['total_depth_distortion_loss_raw']:.3e}"
        f" depth_w={loss_state['total_depth_distortion_loss_weighted']:.3e}"
        f" depth_active_w={active_depth_distortion_weight:.3e}"
        f" normal_raw={loss_state['total_normal_loss_raw']:.3e}"
        f" normal_w={loss_state['total_normal_loss_weighted']:.3e}"
        f" vis_opacity_raw={loss_state['total_visibility_weighted_opacity_loss_raw']:.3e}"
        f" vis_opacity_w={loss_state['total_visibility_weighted_opacity_loss_weighted']:.3e}"
        f" total={loss_state['total_loss_value']:.3e}\n"
        f"  grad_rms:"
        f" pos={grad_pos_rms:.2e}"
        f" tu={grad_tanu_rms:.2e}"
        f" tv={grad_tanv_rms:.2e}"
        f" scale={grad_scale_rms:.2e}"
        f" albedo={grad_albedo_rms:.2e}"
        f" opacity={grad_opacity_rms:.2e}"
        f" beta={grad_beta_rms:.2e}\n"
        f"  grad_max:"
        f" pos={grad_pos_max:.2e}"
        f" tu={grad_tanu_max:.2e}"
        f" tv={grad_tanv_max:.2e}"
        f" scale={grad_scale_max:.2e}"
        f" albedo={grad_albedo_max:.2e}"
        f" opacity={grad_opacity_max:.2e}"
        f" beta={grad_beta_max:.2e}"
    )

def format_gradient_source_balance(
        loss_gradients: Dict[str, np.ndarray],
        depth_regularizer_gradients: Dict[str, np.ndarray],
        normal_regularizer_gradients: Dict[str, np.ndarray],
        visibility_opacity_gradients: Dict[str, np.ndarray],
        regularizer_gradients: Dict[str, np.ndarray],
        total_gradients: Dict[str, np.ndarray],
) -> str:
    keys = [
        ("position", "pos"),
        ("tangent_u", "tan_u"),
        ("tangent_v", "tan_v"),
        ("scale", "scale"),
        ("albedo", "albedo"),
        ("opacity", "opacity"),
        ("beta", "beta"),
    ]

    lines = [
        "Gradient source balance:",
        "  "
        f"{'param':<8}"
        f"{'loss':>11}"
        f"{'regularizers':>14}"
        f"{'total':>11}"
        f"{'regs%':>8}"
        f"{'depth%':>8}"
        f"{'normal%':>9}"
        f"{'vis_eta%':>10}"
        f"   {'regularizer components'}",
    ]

    for key, label in keys:
        loss_norm = gradient_norm_for_key(loss_gradients, key)

        depth_norm = gradient_norm_for_key(depth_regularizer_gradients, key)
        normal_norm = gradient_norm_for_key(normal_regularizer_gradients, key)
        visibility_opacity_norm = gradient_norm_for_key(visibility_opacity_gradients, key)

        regularizer_norm = gradient_norm_for_key(regularizer_gradients, key)
        total_norm = gradient_norm_for_key(total_gradients, key)

        loss_regularizer_denom = loss_norm + regularizer_norm
        regularizer_percent = (
            100.0 * regularizer_norm / loss_regularizer_denom
            if loss_regularizer_denom > 1.0e-20
            else 0.0
        )

        component_denom = depth_norm + normal_norm + visibility_opacity_norm
        depth_percent = 100.0 * depth_norm / component_denom if component_denom > 1.0e-20 else 0.0
        normal_percent = 100.0 * normal_norm / component_denom if component_denom > 1.0e-20 else 0.0
        visibility_opacity_percent = (
            100.0 * visibility_opacity_norm / component_denom
            if component_denom > 1.0e-20
            else 0.0
        )

        lines.append(
            "  "
            f"{label:<8}"
            f"{loss_norm:>11.2e}"
            f"{regularizer_norm:>14.2e}"
            f"{total_norm:>11.2e}"
            f"{regularizer_percent:>7.1f}%"
            f"{depth_percent:>7.1f}%"
            f"{normal_percent:>8.1f}%"
            f"{visibility_opacity_percent:>9.1f}%"
            f"   "
            f"depth={depth_norm:.2e}, "
            f"normal={normal_norm:.2e}, "
            f"vis_eta={visibility_opacity_norm:.2e}"
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
    tangent_u = torch.nn.Parameter(torch.tensor(updated["tangent_u"], device=device, dtype=torch.float32))
    tangent_v = torch.nn.Parameter(torch.tensor(updated["tangent_v"], device=device, dtype=torch.float32))
    scales = torch.nn.Parameter(torch.tensor(updated["scale"], device=device, dtype=torch.float32))
    albedos = torch.nn.Parameter(torch.tensor(updated["albedo"], device=device, dtype=torch.float32))
    opacities = torch.nn.Parameter(torch.tensor(updated["opacity"], device=device, dtype=torch.float32))
    betas = torch.nn.Parameter(torch.tensor(updated["beta"], device=device, dtype=torch.float32))
    powers = torch.nn.Parameter(torch.tensor(updated["power"], device=device, dtype=torch.float32))

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
    verify_tangents_inplace(tangent_u, tangent_v)
    verify_scales_inplace(scales)
    verify_positions_inplace(positions)
    verify_albedos_inplace(albedos)
    verify_opacities_inplace(opacities)
    verify_beta_inplace(betas, trainable_surfel_mask=trainable_surfel_mask)


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
    positions.grad = torch.tensor(grad_position_np, device=device, dtype=torch.float32)
    tangent_u.grad = torch.tensor(grad_tangent_u_np, device=device, dtype=torch.float32)
    tangent_v.grad = torch.tensor(grad_tangent_v_np, device=device, dtype=torch.float32)
    scales.grad = torch.tensor(grad_scales_np, device=device, dtype=torch.float32)
    albedos.grad = torch.tensor(grad_albedos_np, device=device, dtype=torch.float32)
    opacities.grad = torch.tensor(grad_opacities_np, device=device, dtype=torch.float32)
    betas.grad = torch.tensor(grad_betas_np, device=device, dtype=torch.float32)


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
    gradients_dir = output_dir / "gradients"
    gradients_dir.mkdir(parents=True, exist_ok=True)

    num_points = grad_position_np.shape[0]
    grad_opacity_flat = grad_opacities_np.reshape(num_points)
    grad_beta_flat = grad_betas_np.reshape(num_points)

    csv_path = gradients_dir / f"gradients_iter_{iteration:04d}.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "point_index",
                "grad_pos_x", "grad_pos_y", "grad_pos_z",
                "grad_tan_u_x", "grad_tan_u_y", "grad_tan_u_z",
                "grad_tan_v_x", "grad_tan_v_y", "grad_tan_v_z",
                "grad_scale_u", "grad_scale_v",
                "grad_albedo_r", "grad_albedo_g", "grad_albedo_b",
                "grad_opacity",
                "grad_beta",
            ]
        )

        for idx in range(num_points):
            gx, gy, gz = grad_position_np[idx]
            gux, guy, guz = grad_tangent_u_np[idx]
            gvx, gvy, gvz = grad_tangent_v_np[idx]
            gsu, gsv = grad_scales_np[idx]
            gcr, gcg, gcb = grad_albedos_np[idx]
            writer.writerow(
                [
                    idx,
                    gx, gy, gz,
                    gux, guy, guz,
                    gvx, gvy, gvz,
                    gsu, gsv,
                    gcr, gcg, gcb,
                    grad_opacity_flat[idx],
                    grad_beta_flat[idx],
                ]
            )

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


def save_manual_snapshot(
        renderer: pale.Renderer,
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


def save_iteration_point_cloud_snapshot(
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
) -> None:
    points_dir = output_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)

    ply_path = points_dir / f"iter_{iteration:05d}_points.ply"
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

    print(f"[Iter {iteration:04d}] Saved point cloud snapshot: {ply_path}")


def load_target_images(
        renderer: pale.Renderer,
        dataset_path: Path,
) -> tuple[Dict[str, np.ndarray], List[str], List[str]]:
    target_path = Path(dataset_path)
    if not target_path.is_dir():
        raise RuntimeError(f"Target path '{target_path}' must be a directory when multiple cameras are used.")

    training_camera_ids = get_training_camera_names(renderer)
    all_camera_ids = get_all_camera_names(renderer)
    target_images: Dict[str, np.ndarray] = {}

    print(f"Loading target images from directory: {target_path}")
    for camera_name in training_camera_ids:
        image_path = target_path / "images" / f"{camera_name}.png"
        if not image_path.is_file():
            raise RuntimeError(f"Missing target image for camera '{camera_name}': {image_path}")

        target_images[camera_name] = load_target_image(image_path)
        print(
            f"  Camera '{camera_name}': loaded target {image_path} "
            f"with shape {target_images[camera_name].shape}"
        )

    return target_images, training_camera_ids, all_camera_ids


def create_torch_parameters_from_initial(
        initial_params: Dict[str, np.ndarray],
        device: torch.device,
) -> Tuple[torch.nn.Parameter, ...]:
    positions = torch.nn.Parameter(torch.tensor(initial_params["position"], device=device, dtype=torch.float32))
    tangent_u = torch.nn.Parameter(torch.tensor(initial_params["tangent_u"], device=device, dtype=torch.float32))
    tangent_v = torch.nn.Parameter(torch.tensor(initial_params["tangent_v"], device=device, dtype=torch.float32))
    scales = torch.nn.Parameter(torch.tensor(initial_params["scale"], device=device, dtype=torch.float32))
    albedos = torch.nn.Parameter(torch.tensor(initial_params["albedo"], device=device, dtype=torch.float32))
    opacities = torch.nn.Parameter(torch.tensor(initial_params["opacity"], device=device, dtype=torch.float32))
    betas = torch.nn.Parameter(torch.tensor(initial_params["beta"], device=device, dtype=torch.float32))
    powers = torch.nn.Parameter(torch.tensor(initial_params["power"], device=device, dtype=torch.float32))
    return positions, tangent_u, tangent_v, scales, albedos, opacities, betas, powers


def make_initial_params_reference(initial_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "position": initial_params["position"].copy(),
        "tangent_u": initial_params["tangent_u"].copy(),
        "tangent_v": initial_params["tangent_v"].copy(),
        "scale": initial_params["scale"].copy(),
        "albedo": initial_params["albedo"].copy(),
        "opacity": initial_params["opacity"].copy(),
        "beta": initial_params["beta"].copy(),
        "power": initial_params["power"].copy(),
    }


def current_params_as_numpy(
        positions: torch.Tensor,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
) -> Dict[str, np.ndarray]:
    return {
        "position": positions.detach().cpu().numpy(),
        "tangent_u": tangent_u.detach().cpu().numpy(),
        "tangent_v": tangent_v.detach().cpu().numpy(),
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
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
        depth_distortion_weight: float,
        normal_consistency_weight: float,
        visibility_weighted_opacity_weight: float,
        use_depth_distortion: bool,
        use_normal_consistency: bool,
        use_visibility_weighted_opacity: bool,
) -> tuple[float, float, float, float, float, float, float]:
    initial_points_path = output_dir / "initial_points.ply"
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

    initial_rgb_loss = 0.0
    initial_depth_distortion_loss_raw = 0.0
    initial_normal_loss_raw = 0.0
    initial_visibility_weighted_opacity_loss_raw = 0.0

    for camera_name in all_camera_ids:
        img_np = get_forward_rgb(initial_images, camera_name)
        (output_dir / camera_name).mkdir(parents=True, exist_ok=True)
        save_render(output_dir / f"render_initial_{camera_name}.png", img_np)

        if camera_name not in target_images:
            print(f"Warning: no target image found for camera '{camera_name}', skipping target save and loss.")
            continue

        tgt_np = target_images[camera_name]
        initial_rgb_loss += float(compute_l2_loss(img_np, tgt_np))
        save_render(output_dir / f"render_target_{camera_name}.png", tgt_np)

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

        if use_visibility_weighted_opacity:
            visibility_opacity = get_forward_visibility_weighted_opacity(initial_images, camera_name)
            initial_visibility_weighted_opacity_loss_raw += float(visibility_opacity.mean())

    initial_depth_distortion_loss_weighted = depth_distortion_weight * initial_depth_distortion_loss_raw
    initial_normal_loss_weighted = normal_consistency_weight * initial_normal_loss_raw
    initial_visibility_weighted_opacity_loss_weighted = (
            visibility_weighted_opacity_weight * initial_visibility_weighted_opacity_loss_raw
    )

    initial_total_loss = (
            initial_rgb_loss
            + initial_depth_distortion_loss_weighted
            + initial_normal_loss_weighted
            + initial_visibility_weighted_opacity_loss_weighted
    )

    return (
        initial_rgb_loss,
        initial_depth_distortion_loss_raw,
        initial_depth_distortion_loss_weighted,
        initial_normal_loss_raw,
        initial_normal_loss_weighted,
        initial_visibility_weighted_opacity_loss_weighted,
        initial_total_loss,
    )


def print_loss_summary(
        prefix: str,
        rgb_loss: float,
        depth_distortion_loss_raw: float,
        depth_distortion_loss_weighted: float,
        normal_loss_raw: float,
        normal_loss_weighted: float,
        visibility_weighted_opacity_loss_weighted: float,
        total_loss: float,
) -> None:
    print(f"{prefix} RGB loss                               : {rgb_loss:.6e}")
    print(f"{prefix} depth distortion loss (raw)            : {depth_distortion_loss_raw:.6e}")
    print(f"{prefix} depth distortion loss (weighted)       : {depth_distortion_loss_weighted:.6e}")
    print(f"{prefix} normal consistency loss (raw)          : {normal_loss_raw:.6e}")
    print(f"{prefix} normal consistency loss (weighted)     : {normal_loss_weighted:.6e}")
    print(f"{prefix} visibility opacity loss (weighted)    : {visibility_weighted_opacity_loss_weighted:.6e}")
    print(f"{prefix} total loss                             : {total_loss:.6e}")

def compute_iteration_losses_and_adjoints(
        forward_out: Dict[str, dict],
        target_images: Dict[str, np.ndarray],
        training_camera_ids: List[str],
        depth_distortion_weight: float,
        normal_consistency_weight: float,
        visibility_weighted_opacity_weight: float,
        use_depth_distortion: bool,
        use_normal_consistency: bool,
        use_visibility_weighted_opacity: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "total_rgb_loss_value": 0.0,
        "total_depth_distortion_loss_raw": 0.0,
        "total_depth_distortion_loss_weighted": 0.0,
        "total_normal_loss_raw": 0.0,
        "total_normal_loss_weighted": 0.0,
        "total_visibility_weighted_opacity_loss_raw": 0.0,
        "total_visibility_weighted_opacity_loss_weighted": 0.0,
        "total_loss_value": 0.0,
        "loss_grad_images": {},
        "depth_distortion_grad_images": {},
        "visible_normal_adjoints": {},
        "depth_normal_adjoints": {},
        "depth_distortion_maps_for_logging": {},
        "visibility_weighted_opacity_maps_for_logging": {},
    }

    for camera_name in training_camera_ids:
        current_rgb_np = get_forward_rgb(forward_out, camera_name)
        target_rgb_np = target_images[camera_name]

        rgb_grad = compute_l2_grad(current_rgb_np, target_rgb_np)
        rgb_loss_value = float(compute_l2_loss(current_rgb_np, target_rgb_np))

        result["total_rgb_loss_value"] += rgb_loss_value
        result["total_loss_value"] += rgb_loss_value
        result["loss_grad_images"][camera_name] = rgb_grad

        if use_depth_distortion:
            current_depth_distortion_np = get_forward_depth_distortion(forward_out, camera_name)
            depth_distortion_loss_raw = float(current_depth_distortion_np.mean())
            depth_distortion_loss_weighted = depth_distortion_weight * depth_distortion_loss_raw

            result["depth_distortion_maps_for_logging"][camera_name] = current_depth_distortion_np
            result["total_depth_distortion_loss_raw"] += depth_distortion_loss_raw
            result["total_depth_distortion_loss_weighted"] += depth_distortion_loss_weighted
            result["total_loss_value"] += depth_distortion_loss_weighted
            result["depth_distortion_grad_images"][camera_name] = make_mean_reduction_adjoint_image(
                current_depth_distortion_np,
                depth_distortion_weight,
            )

        if use_normal_consistency:
            visible_normal = get_forward_visible_normal(forward_out, camera_name)
            normal_from_depth = get_forward_normal_from_depth(forward_out, camera_name)

            raw_normal_loss_value, dvis_raw, ddepth_raw = compute_normal_consistency_loss_and_adjoints(
                visible_normal,
                normal_from_depth,
                1.0,
            )

            weighted_normal_loss_value = normal_consistency_weight * raw_normal_loss_value
            result["total_normal_loss_raw"] += raw_normal_loss_value
            result["total_normal_loss_weighted"] += weighted_normal_loss_value
            result["total_loss_value"] += weighted_normal_loss_value
            result["visible_normal_adjoints"][camera_name] = (
                    normal_consistency_weight * dvis_raw
            ).astype(np.float32, copy=False)
            result["depth_normal_adjoints"][camera_name] = (
                    normal_consistency_weight * ddepth_raw
            ).astype(np.float32, copy=False)

        if use_visibility_weighted_opacity:
            visibility_opacity_np = get_forward_visibility_weighted_opacity(forward_out, camera_name)
            visibility_opacity_loss_raw = float(visibility_opacity_np.mean())
            visibility_opacity_loss_weighted = visibility_weighted_opacity_weight * visibility_opacity_loss_raw

            result["visibility_weighted_opacity_maps_for_logging"][camera_name] = visibility_opacity_np
            result["total_visibility_weighted_opacity_loss_raw"] += visibility_opacity_loss_raw
            result["total_visibility_weighted_opacity_loss_weighted"] += visibility_opacity_loss_weighted
            result["total_loss_value"] += visibility_opacity_loss_weighted

    return result


def extract_total_gradient_arrays(
        total_gradients: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grad_position_np = np.asarray(total_gradients["position"], dtype=np.float32, order="C")
    grad_tangent_u_np = np.asarray(total_gradients["tangent_u"], dtype=np.float32, order="C")
    grad_tangent_v_np = np.asarray(total_gradients["tangent_v"], dtype=np.float32, order="C")
    grad_scales_np = np.asarray(total_gradients["scale"], dtype=np.float32, order="C")
    grad_albedos_np = np.asarray(total_gradients["albedo"], dtype=np.float32, order="C")
    grad_opacities_np = np.asarray(total_gradients["opacity"], dtype=np.float32, order="C")
    grad_betas_np = np.asarray(total_gradients["beta"], dtype=np.float32, order="C")
    return (
        grad_position_np,
        grad_tangent_u_np,
        grad_tangent_v_np,
        grad_scales_np,
        grad_albedos_np,
        grad_opacities_np,
        grad_betas_np,
    )

def update_densification_statistics(
        iteration: int,
        densification_interval: int,
        densification_stats_warmup_iterations: int,
        densify_position_grad_accum_np: np.ndarray,
        densify_position_grad_denom_np: np.ndarray,
        densify_position_grad_vector_accum_np: np.ndarray,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        albedos: torch.Tensor,
        trainable_surfel_mask: torch.Tensor,
        densify_bsdf_floor: float,
        densify_bsdf_gamma: float,
        densify_position_grad_per_camera_np: np.ndarray,
        densify_position_grad_per_camera_count_np: np.ndarray,
) -> None:
    if densification_interval <= 0:
        return

    densification_phase = iteration % densification_interval
    should_accumulate = (
        densification_interval <= 1
        or (
            densification_phase >= densification_stats_warmup_iterations
            and densification_phase != 0
        )
    )
    if not should_accumulate:
        return

    if densify_position_grad_per_camera_np is None:
        raise RuntimeError(
            "update_densification_statistics requires renderer gradient stats: "
            "densify_position_grad_per_camera_np is None. "
            "Expected adjoint_images['gradient_stats']['position_per_camera']."
        )

    if densify_position_grad_per_camera_count_np is None:
        raise RuntimeError(
            "update_densification_statistics requires renderer gradient stats: "
            "densify_position_grad_per_camera_count_np is None. "
            "Expected adjoint_images['gradient_stats']['position_record_count_per_camera']."
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
        tangent_u_np = tangent_u.detach().cpu().numpy().astype(np.float32)
        tangent_v_np = tangent_v.detach().cpu().numpy().astype(np.float32)
        albedo_np = albedos.detach().cpu().numpy().astype(np.float32)
        trainable_np = trainable_surfel_mask.detach().cpu().numpy().astype(bool).reshape(-1)

    if tangent_u_np.shape != (point_count, 3):
        raise RuntimeError(f"tangent_u shape mismatch: expected {(point_count, 3)}, got {tangent_u_np.shape}")
    if tangent_v_np.shape != (point_count, 3):
        raise RuntimeError(f"tangent_v shape mismatch: expected {(point_count, 3)}, got {tangent_v_np.shape}")
    if trainable_np.shape[0] != point_count:
        raise RuntimeError(f"trainable mask length mismatch: expected {point_count}, got {trainable_np.shape[0]}")

    tangent_u_norm_np = np.linalg.norm(tangent_u_np, axis=1, keepdims=True)
    tangent_v_norm_np = np.linalg.norm(tangent_v_np, axis=1, keepdims=True)

    tangent_u_unit_np = tangent_u_np / np.maximum(tangent_u_norm_np, 1.0e-8)
    tangent_v_unit_np = tangent_v_np / np.maximum(tangent_v_norm_np, 1.0e-8)

    dot_u_np = np.sum(per_camera_grad_np * tangent_u_unit_np[:, None, :], axis=2, keepdims=True)
    dot_v_np = np.sum(per_camera_grad_np * tangent_v_unit_np[:, None, :], axis=2, keepdims=True)

    per_camera_tangent_grad_np = (
        dot_u_np * tangent_u_unit_np[:, None, :]
        + dot_v_np * tangent_v_unit_np[:, None, :]
    )

    visible_camera_mask_np = per_camera_count_np > 0
    active_camera_count_np = visible_camera_mask_np.sum(axis=1, keepdims=True).astype(np.float32)

    per_camera_tangent_grad_norm_np = np.linalg.norm(per_camera_tangent_grad_np, axis=2)
    per_camera_tangent_grad_norm_np = np.nan_to_num(
        per_camera_tangent_grad_norm_np,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    visible_mask_float_np = visible_camera_mask_np.astype(np.float32)

    safe_active_camera_count_np = np.maximum(active_camera_count_np, 1.0)

    # Scalar score:
    #     mean_visible(||project_tangent(g_camera)||)
    densify_position_signal_np = (
                                         per_camera_tangent_grad_norm_np * visible_mask_float_np
                                 ).sum(axis=1, keepdims=True) / safe_active_camera_count_np

    # Signed vector direction:
    #     mean_visible(project_tangent(g_camera))
    density_grad_position_vector_np = (
                                              per_camera_tangent_grad_np * visible_mask_float_np[:, :, None]
                                      ).sum(axis=1) / safe_active_camera_count_np

    density_grad_position_vector_np[active_camera_count_np[:, 0] == 0.0] = 0.0
    densify_position_signal_np[active_camera_count_np[:, 0] == 0.0] = 0.0

    linear_rgb_bsdf_scale_np = np.mean(albedo_np, axis=1)
    bsdf_normalizer_np = (
        np.maximum(linear_rgb_bsdf_scale_np, densify_bsdf_floor) ** densify_bsdf_gamma
    ).astype(np.float32)

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


def maybe_make_densification_result(
        iteration: int,
        config: OptimizationConfig,
        positions: torch.nn.Parameter,
        tangent_u: torch.nn.Parameter,
        tangent_v: torch.nn.Parameter,
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
        densify_until_iteration: int,
        densification_interval: int,
        densification_verbose: bool,
        densification_grad_quantile: float,
        densification_grad_abs_min: float,
) -> Optional[Dict[str, np.ndarray]]:
    if densification_interval <= 0:
        return None

    if not (
            densify_after <= iteration <= densify_until_iteration
            and iteration % densification_interval == 0
    ):
        return None

    with torch.no_grad():
        valid_denom_np = densify_position_grad_denom_np.reshape(-1) > 0.0
        avg_density_grad_norm_np = np.zeros((positions.shape[0],), dtype=np.float32)
        avg_density_grad_vector_np = np.zeros(tuple(positions.shape), dtype=np.float32)

        avg_density_grad_norm_np[valid_denom_np] = (
                densify_position_grad_accum_np.reshape(-1)[valid_denom_np]
                / densify_position_grad_denom_np.reshape(-1)[valid_denom_np]
        )

        avg_density_grad_vector_np[valid_denom_np] = (
                densify_position_grad_vector_accum_np[valid_denom_np]
                / densify_position_grad_denom_np.reshape(-1, 1)[valid_denom_np]
        )

        grad_pos_norm_np = np.nan_to_num(
            avg_density_grad_norm_np,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        trainable_np = trainable_surfel_mask.detach().cpu().numpy().astype(bool)
        finite_grad = np.isfinite(grad_pos_norm_np)
        candidate_mask_np = (
                valid_denom_np
                & finite_grad
                & trainable_np
                & (grad_pos_norm_np >= densification_grad_abs_min)
        )

        densification_result = None
        densify_reason = "not_attempted"
        n_new_from_densification = 0

        finite_count = int(np.count_nonzero(finite_grad))
        trainable_count = int(np.count_nonzero(trainable_np))
        above_abs_count = int(np.count_nonzero(grad_pos_norm_np >= densification_grad_abs_min))
        candidate_count = int(np.count_nonzero(candidate_mask_np))
        valid_denom_count = int(np.count_nonzero(valid_denom_np))

        grad_threshold = float("nan")
        grad_quantile_threshold = float("nan")

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

        if not np.any(valid_denom_np):
            densify_reason = "no_density_samples"
        elif candidate_count == 0:
            densify_reason = "no_candidates_after_grad_trainable"
        else:
            active_grad = grad_pos_norm_np[candidate_mask_np]
            grad_quantile_threshold = float(np.quantile(active_grad, densification_grad_quantile))
            grad_threshold = max(float(densification_grad_abs_min), grad_quantile_threshold)

            densify_mask_torch = torch.as_tensor(candidate_mask_np, device=positions.device, dtype=torch.bool)
            densification_result = make_under_reconstruction_clones(
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
                min_clone_scale=config.densification_scale_min,
            )

            if densification_result is not None:
                new_block = densification_result.get("new", None)
                if new_block is not None:
                    n_new_from_densification = int(new_block["position"].shape[0])
                    densify_reason = "clone_added"
                else:
                    densify_reason = "clone_result_without_new_block"
            else:
                densify_reason = "selected_candidates_but_clone_filter_rejected_all"

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
                f"abs_thr={densification_grad_abs_min:.3e}"
            )
        elif n_new_from_densification > 0:
            print(
                f"[Iter {iteration:04d}] Clone densification: "
                f"adding {n_new_from_densification} surfels | "
                f"grad_thr={grad_threshold:.3e}, "
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

    scale_prune_indices = compute_prune_indices_by_degenerate_scale(
        scales,
        min_scale=config.scale_prune_min_scale,
        trainable_mask=trainable_surfel_mask,
        min_points_to_keep=config.min_points_to_keep_after_scale_prune,
    )
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


def robust_normalize_scalar(values_np: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    values_np = np.nan_to_num(
        np.asarray(values_np, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    finite_values_np = values_np[np.isfinite(values_np)]
    if finite_values_np.size == 0:
        return np.zeros_like(values_np, dtype=np.float32)

    value_min = float(np.percentile(finite_values_np, lower_percentile))
    value_max = float(np.percentile(finite_values_np, upper_percentile))

    if value_max <= value_min + 1.0e-12:
        return np.zeros_like(values_np, dtype=np.float32)

    return np.clip((values_np - value_min) / (value_max - value_min), 0.0, 1.0).astype(np.float32)


def black_red_yellow_white_colormap(normalized_values_np: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(normalized_values_np, dtype=np.float32).reshape(-1, 1), 0.0, 1.0)

    red = np.clip(3.0 * t, 0.0, 1.0)
    green = np.clip(3.0 * t - 1.0, 0.0, 1.0)
    blue = np.clip(3.0 * t - 2.0, 0.0, 1.0)

    return np.clip(np.concatenate([red, green, blue], axis=1) * 255.0, 0.0, 255.0).astype(np.uint8)


def tensor_to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def save_scalar_colored_point_ply(
        output_path: Path,
        positions_np: np.ndarray,
        scalar_values_np: np.ndarray,
        scalar_name: str,
        normalize_by_max_value: float | None = None,
) -> None:
    positions_np = np.asarray(positions_np, dtype=np.float32)
    scalar_values_np = np.nan_to_num(
        np.asarray(scalar_values_np, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if positions_np.ndim != 2 or positions_np.shape[1] != 3:
        raise RuntimeError(f"positions_np must have shape (N, 3), got {positions_np.shape}")

    if scalar_values_np.shape[0] != positions_np.shape[0]:
        raise RuntimeError(
            f"Scalar length mismatch for {scalar_name}: "
            f"{scalar_values_np.shape[0]} vs positions {positions_np.shape[0]}"
        )

    if normalize_by_max_value is not None:
        normalized_np = np.clip(scalar_values_np / max(float(normalize_by_max_value), 1.0e-12), 0.0, 1.0)
    else:
        normalized_np = robust_normalize_scalar(scalar_values_np)

    colors_np = black_red_yellow_white_colormap(normalized_np)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {positions_np.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write(f"property float {scalar_name}\n")
        file.write("end_header\n")

        for position, color, scalar_value in zip(positions_np, colors_np, scalar_values_np):
            file.write(
                f"{position[0]:.9g} {position[1]:.9g} {position[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{float(scalar_value):.9g}\n"
            )


def average_densification_accumulators(
        densify_position_grad_accum_np: np.ndarray,
        densify_position_grad_denom_np: np.ndarray,
        densify_position_grad_vector_accum_np: np.ndarray,
        point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_denom_np = densify_position_grad_denom_np.reshape(-1) > 0.0

    avg_density_grad_norm_np = np.zeros((point_count,), dtype=np.float32)
    avg_density_grad_vector_np = np.zeros((point_count, 3), dtype=np.float32)

    avg_density_grad_norm_np[valid_denom_np] = (
        densify_position_grad_accum_np.reshape(-1)[valid_denom_np]
        / densify_position_grad_denom_np.reshape(-1)[valid_denom_np]
    )

    avg_density_grad_vector_np[valid_denom_np] = (
        densify_position_grad_vector_accum_np[valid_denom_np]
        / densify_position_grad_denom_np.reshape(-1, 1)[valid_denom_np]
    )

    avg_density_grad_vector_norm_np = np.linalg.norm(avg_density_grad_vector_np, axis=1).astype(np.float32)

    avg_density_grad_norm_np = np.nan_to_num(avg_density_grad_norm_np, nan=0.0, posinf=0.0, neginf=0.0)
    avg_density_grad_vector_norm_np = np.nan_to_num(avg_density_grad_vector_norm_np, nan=0.0, posinf=0.0, neginf=0.0)

    return avg_density_grad_norm_np, avg_density_grad_vector_np, avg_density_grad_vector_norm_np


def save_densification_gradient_diagnostics(
        output_dir: Path,
        iteration: int,
        positions: torch.Tensor,
        densify_position_grad_accum_np: np.ndarray,
        densify_position_grad_denom_np: np.ndarray,
        densify_position_grad_vector_accum_np: np.ndarray,
        photo_gradient_surfel_stats: dict,
        active_camera_count_max: int,
) -> None:
    positions_np = tensor_to_numpy_float32(positions)
    point_count = positions_np.shape[0]

    avg_density_grad_norm_np, _, avg_density_grad_vector_norm_np = average_densification_accumulators(
        densify_position_grad_accum_np=densify_position_grad_accum_np,
        densify_position_grad_denom_np=densify_position_grad_denom_np,
        densify_position_grad_vector_accum_np=densify_position_grad_vector_accum_np,
        point_count=point_count,
    )

    diagnostic_dir = output_dir / "gradient_stats" / f"iter_{iteration:06d}"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    save_scalar_colored_point_ply(
        output_path=diagnostic_dir / "gradient_geometric_pressure.ply",
        positions_np=positions_np,
        scalar_values_np=avg_density_grad_norm_np,
        scalar_name="gradient_geometric_pressure",
    )

    save_scalar_colored_point_ply(
        output_path=diagnostic_dir / "gradient_position_norm.ply",
        positions_np=positions_np,
        scalar_values_np=avg_density_grad_vector_norm_np,
        scalar_name="gradient_position_norm",
    )

    if "position_std" in photo_gradient_surfel_stats:
        position_std_np = np.asarray(photo_gradient_surfel_stats["position_std"], dtype=np.float32).reshape(-1)
        if position_std_np.shape[0] == point_count:
            save_scalar_colored_point_ply(
                output_path=diagnostic_dir / "gradient_position_std.ply",
                positions_np=positions_np,
                scalar_values_np=position_std_np,
                scalar_name="gradient_position_std",
            )

    if "position_active_camera_count" in photo_gradient_surfel_stats:
        active_camera_count_np = np.asarray(
            photo_gradient_surfel_stats["position_active_camera_count"],
            dtype=np.float32,
        ).reshape(-1)

        if active_camera_count_np.shape[0] == point_count:
            save_scalar_colored_point_ply(
                output_path=diagnostic_dir / "gradient_active_camera_count.ply",
                positions_np=positions_np,
                scalar_values_np=active_camera_count_np,
                scalar_name="gradient_active_camera_count",
                normalize_by_max_value=float(max(active_camera_count_max, 1)),
            )

    if "position_coherence" in photo_gradient_surfel_stats:
        position_coherence_np = np.asarray(photo_gradient_surfel_stats["position_coherence"], dtype=np.float32).reshape(-1)
        if position_coherence_np.shape[0] == point_count:
            save_scalar_colored_point_ply(
                output_path=diagnostic_dir / "gradient_position_coherence.ply",
                positions_np=positions_np,
                scalar_values_np=position_coherence_np,
                scalar_name="gradient_position_coherence",
                normalize_by_max_value=1.0,
            )

    if "position_disagreement" in photo_gradient_surfel_stats:
        position_disagreement_np = np.asarray(photo_gradient_surfel_stats["position_disagreement"], dtype=np.float32).reshape(-1)
        if position_disagreement_np.shape[0] == point_count:
            save_scalar_colored_point_ply(
                output_path=diagnostic_dir / "gradient_position_disagreement.ply",
                positions_np=positions_np,
                scalar_values_np=position_disagreement_np,
                scalar_name="gradient_position_disagreement",
            )

    print(f"[Iter {iteration:04d}] Saved gradient diagnostics: {diagnostic_dir}")


def save_iteration_outputs(
        output_dir: Path,
        iteration: int,
        save_interval: int,
        final_iteration: int,
        all_camera_ids: List[str],
        forward_out: Dict[str, dict],
        adjoint_images: Dict[str, Any],
        renderer_settings: RendererSettingsConfig,
) -> None:
    if iteration % save_interval != 0 and iteration != final_iteration:
        return

    for camera_name in all_camera_ids:
        camera_base_dir = output_dir / camera_name
        camera_render_dir = camera_base_dir / "render"
        camera_grad_dir = camera_base_dir / "grad"
        camera_depth_dir = camera_base_dir / "depth_distortion"
        camera_visible_normal_dir = camera_base_dir / "visible_normal"
        camera_depth_normal_dir = camera_base_dir / "normal_from_depth"
        camera_median_depth_dir = camera_base_dir / "median_depth"

        camera_render_dir.mkdir(parents=True, exist_ok=True)
        camera_grad_dir.mkdir(parents=True, exist_ok=True)
        camera_depth_dir.mkdir(parents=True, exist_ok=True)
        camera_visible_normal_dir.mkdir(parents=True, exist_ok=True)
        camera_depth_normal_dir.mkdir(parents=True, exist_ok=True)
        camera_median_depth_dir.mkdir(parents=True, exist_ok=True)

        save_render(
            camera_render_dir / f"{iteration:04d}_render.png",
            get_forward_rgb(forward_out, camera_name),
        )

        save_depth_distortion_snapshot(
            camera_depth_dir / f"{iteration:04d}_depth_distortion.png",
            get_forward_depth_distortion(forward_out, camera_name),
            quantile=0.99,
            save_npy=False,
        )

        save_median_depth_snapshot(
            camera_median_depth_dir / f"{iteration:04d}_median_depth.png",
            get_forward_median_depth(forward_out, camera_name),
            quantile=0.99,
            save_npy=False,
        )

        save_normal_map_snapshot(
            camera_visible_normal_dir / f"{iteration:04d}_visible_normal.png",
            get_forward_visible_normal(forward_out, camera_name),
            save_npy=False,
        )

        save_normal_map_snapshot(
            camera_depth_normal_dir / f"{iteration:04d}_normal_from_depth.png",
            get_forward_normal_from_depth(forward_out, camera_name),
            save_npy=False,
        )

        adjoint_source_images = adjoint_images.get("adjoint_source")
        if adjoint_source_images is not None and camera_name in adjoint_source_images:
            grad_img_np = np.asarray(adjoint_source_images[camera_name], dtype=np.float32, order="C")
            grad_img_np = np.nan_to_num(grad_img_np, nan=0.0, posinf=0.0, neginf=0.0)

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
            "camera_name",
            "loss_rgb_sum",
            "loss_depth_distortion_raw_sum",
            "loss_depth_distortion_weighted_sum",
            "loss_normal_consistency_raw_sum",
            "loss_normal_consistency_weighted_sum",
            "loss_visibility_weighted_opacity_raw_sum",
            "loss_visibility_weighted_opacity_weighted_sum",
            "loss_total_sum",
            "parameter_mse",
            "num_points",
            "iteration_time_sec",
            "total_time",
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
