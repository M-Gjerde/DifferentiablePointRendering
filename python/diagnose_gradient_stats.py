from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pale

from config import RendererSettingsConfig, parse_args
from render_hooks import fetch_parameters, get_training_camera_names
from training_helpers import load_target_images, compute_iteration_losses_and_adjoints


def robust_normalize(values: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return np.zeros_like(values, dtype=np.float32)

    value_min = float(np.percentile(finite_values, lower_percentile))
    value_max = float(np.percentile(finite_values, upper_percentile))

    if value_max <= value_min + 1.0e-12:
        return np.zeros_like(values, dtype=np.float32)

    return np.clip((values - value_min) / (value_max - value_min), 0.0, 1.0).astype(np.float32)


def heatmap_rgb(normalized_values: np.ndarray) -> np.ndarray:
    t = np.clip(normalized_values.reshape(-1, 1), 0.0, 1.0)
    red = np.clip(3.0 * t - 1.0, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(3.0 * t - 1.5), 0.0, 1.0)
    blue = np.clip(1.0 - 3.0 * t, 0.0, 1.0)
    return np.clip(np.concatenate([red, green, blue], axis=1) * 255.0, 0.0, 255.0).astype(np.uint8)


def compute_normals(tangent_u: np.ndarray, tangent_v: np.ndarray) -> np.ndarray:
    normals = np.cross(tangent_u, tangent_v)
    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_lengths, 1.0e-8)
    return np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def write_colored_point_ply(
        output_path: Path,
        positions: np.ndarray,
        tangent_u: np.ndarray,
        tangent_v: np.ndarray,
        colors: np.ndarray,
        scalar_values: np.ndarray,
        scalar_name: str,
) -> None:
    positions = np.asarray(positions, dtype=np.float32)
    normals = compute_normals(
        np.asarray(tangent_u, dtype=np.float32),
        np.asarray(tangent_v, dtype=np.float32),
    )
    colors = np.asarray(colors, dtype=np.uint8)
    scalar_values = np.nan_to_num(np.asarray(scalar_values, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)

    if positions.shape[0] != colors.shape[0] or positions.shape[0] != scalar_values.shape[0]:
        raise RuntimeError(
            f"PLY shape mismatch: positions={positions.shape}, colors={colors.shape}, scalar={scalar_values.shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {positions.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property float nx\n")
        file.write("property float ny\n")
        file.write("property float nz\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write(f"property float {scalar_name}\n")
        file.write("end_header\n")

        for position, normal, color, scalar_value in zip(positions, normals, colors, scalar_values):
            file.write(
                f"{position[0]:.9g} {position[1]:.9g} {position[2]:.9g} "
                f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{float(scalar_value):.9g}\n"
            )


def save_stat_ply(
        output_dir: Path,
        name: str,
        params: Dict[str, np.ndarray],
        values: np.ndarray,
        normalize_by_active_range: bool = False,
        active_camera_max: float = 1.0,
) -> None:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)

    if normalize_by_active_range:
        normalized = np.clip(values / max(float(active_camera_max), 1.0), 0.0, 1.0).astype(np.float32)
    else:
        normalized = robust_normalize(values)

    colors = heatmap_rgb(normalized)

    write_colored_point_ply(
        output_path=output_dir / f"{name}.ply",
        positions=params["position"],
        tangent_u=params["tangent_u"],
        tangent_v=params["tangent_v"],
        colors=colors,
        scalar_values=values,
        scalar_name=name,
    )


def main() -> None:
    config = parse_args()
    renderer_settings = RendererSettingsConfig()
    diagnostic_iterations = 30

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    scene_short = Path(config.scene_xml).stem

    base_output_dir = config.output_dir if config.output_dir.is_absolute() else config.assets_root / config.output_dir
    output_dir = base_output_dir / f"{timestamp}_gradient_stats_{scene_short}"
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = pale.Renderer(
        str(config.assets_root),
        config.scene_xml,
        config.pointcloud_ply,
        renderer_settings.as_dict(config),
    )

    target_images, training_camera_ids, _ = load_target_images(renderer, Path(config.dataset_path))
    camera_ids = get_training_camera_names(renderer)

    if len(training_camera_ids) == 0:
        raise RuntimeError("No training cameras found from target image dataset.")

    print("Gradient-stat diagnostic")
    print(f"  assets_root     : {config.assets_root}")
    print(f"  scene_xml       : {config.scene_xml}")
    print(f"  pointcloud      : {config.pointcloud_ply}")
    print(f"  dataset_path    : {config.dataset_path}")
    print(f"  output_dir      : {output_dir}")
    print(f"  cameras in scene: {camera_ids}")
    print(f"  training cameras: {training_camera_ids}")
    print(f"  iterations      : {diagnostic_iterations}")

    params = fetch_parameters(renderer)
    point_count = int(params["position"].shape[0])

    std_accum = np.zeros((point_count,), dtype=np.float64)
    pressure_accum = np.zeros((point_count,), dtype=np.float64)
    gradient_norm_accum = np.zeros((point_count,), dtype=np.float64)
    active_count_accum = np.zeros((point_count,), dtype=np.float64)

    valid_iteration_count = 0

    for iteration in range(1, diagnostic_iterations + 1):
        forward_out = renderer.render_forward()

        loss_state = compute_iteration_losses_and_adjoints(
            forward_out=forward_out,
            target_images=target_images,
            training_camera_ids=training_camera_ids,
            depth_distortion_weight=0.0,
            normal_consistency_weight=0.0,
            visibility_weighted_opacity_weight=0.0,
            use_depth_distortion=False,
            use_normal_consistency=False,
            use_visibility_weighted_opacity=False,
        )

        photo_gradients, adjoint_images = renderer.render_backward(loss_state["loss_grad_images"])
        gradient_stats = adjoint_images.get("gradient_stats", {})

        required_keys = [
            "position_std",
            "position_mean_norm",
            "position_active_camera_count",
        ]
        missing_keys = [key for key in required_keys if key not in gradient_stats]
        if missing_keys:
            raise RuntimeError(
                "render_backward did not return expected gradient_stats keys: "
                f"{missing_keys}. Make sure the C++ bindings add adjointImagesDictionary['gradient_stats']."
            )

        position_std = np.asarray(gradient_stats["position_std"], dtype=np.float32).reshape(-1)
        position_pressure = np.asarray(gradient_stats["position_mean_norm"], dtype=np.float32).reshape(-1)
        active_count = np.asarray(gradient_stats["position_active_camera_count"], dtype=np.float32).reshape(-1)

        position_gradient = np.asarray(photo_gradients["position"], dtype=np.float32, order="C")
        position_gradient_norm = np.linalg.norm(position_gradient, axis=1)

        if position_std.shape[0] != point_count:
            raise RuntimeError(f"position_std length mismatch: {position_std.shape[0]} vs {point_count}")

        std_accum += np.nan_to_num(position_std, nan=0.0, posinf=0.0, neginf=0.0)
        pressure_accum += np.nan_to_num(position_pressure, nan=0.0, posinf=0.0, neginf=0.0)
        gradient_norm_accum += np.nan_to_num(position_gradient_norm, nan=0.0, posinf=0.0, neginf=0.0)
        active_count_accum += np.nan_to_num(active_count, nan=0.0, posinf=0.0, neginf=0.0)

        valid_iteration_count += 1

        print(
            f"[{iteration:02d}/{diagnostic_iterations}] "
            f"loss={loss_state['total_rgb_loss_value']:.6e} "
            f"std_max={float(position_std.max()):.3e} "
            f"pressure_max={float(position_pressure.max()):.3e} "
            f"grad_norm_max={float(position_gradient_norm.max()):.3e} "
            f"active_max={float(active_count.max()):.0f}"
        )

    if valid_iteration_count == 0:
        raise RuntimeError("No valid backward iterations completed.")

    inv_count = 1.0 / float(valid_iteration_count)
    std_mean = (std_accum * inv_count).astype(np.float32)
    pressure_mean = (pressure_accum * inv_count).astype(np.float32)
    gradient_norm_mean = (gradient_norm_accum * inv_count).astype(np.float32)
    active_count_mean = (active_count_accum * inv_count).astype(np.float32)

    active_camera_max = max(float(len(training_camera_ids)), float(active_count_mean.max()), 1.0)

    params = fetch_parameters(renderer)

    save_stat_ply(output_dir, "gradient_position_std", params, std_mean)
    save_stat_ply(output_dir, "gradient_geometric_pressure", params, pressure_mean)
    save_stat_ply(output_dir, "gradient_position_norm", params, gradient_norm_mean)
    save_stat_ply(
        output_dir,
        "gradient_active_camera_count",
        params,
        active_count_mean,
        normalize_by_active_range=True,
        active_camera_max=active_camera_max,
    )

    print("Saved:")
    print(f"  {output_dir / 'gradient_position_std.ply'}")
    print(f"  {output_dir / 'gradient_geometric_pressure.ply'}")
    print(f"  {output_dir / 'gradient_position_norm.ply'}")
    print(f"  {output_dir / 'gradient_active_camera_count.ply'}")


if __name__ == "__main__":
    main()