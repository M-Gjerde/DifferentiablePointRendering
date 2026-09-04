from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_CLI_FLAGS: dict[str, str | tuple[str, str]] = {
    "device": "--device",
    "target_color_space": "--target-color-space",
    "checkpoint": "--checkpoint",
    "iterations": "--iterations",
    "optimizer_type": "--optimizer",
    "log_interval": "--log-interval",
    "save_interval": "--save-interval",
    "save_ply_files_interval": "--save-ply-files-interval",
    "enable_image_preview": ("--image-preview", "--no-image-preview"),
    "ground_truth": "--gt",
    "geometry_samples": "--geometry-samples",
    "geometry_seed": "--geometry-seed",
    "geometry_scale": "--geometry-scale",
    "geometry_use_vertices": ("--geometry-use-vertices", "--no-geometry-use-vertices"),
    "enable_metrics": ("--metrics", "--no-metrics"),
    "mesh_extraction_interval": "--mesh-extraction-interval",
    "mesh_extraction_depth_key": "--mesh-extraction-depth-key",
    "mesh_extraction_mesh_res": "--mesh-extraction-mesh-res",
    "mesh_extraction_num_cluster": "--mesh-extraction-num-cluster",
    "save_final_mesh": ("--save-final-mesh", "--no-save-final-mesh"),
    "learning_rate": "--lr",
    "learning_rate_position": "--lr-pos",
    "learning_rate_rotation": "--lr-rot",
    "learning_rate_scale": "--lr-scale",
    "learning_rate_albedo": "--lr-albedo",
    "learning_rate_opacity": "--lr-opacity",
    "learning_rate_beta": "--lr-beta",
    "use_global_lr_decay": ("--global-lr-decay", "--no-global-lr-decay"),
    "global_lr_scale_init": "--global-lr-scale-init",
    "global_lr_scale_final": "--global-lr-scale-final",
    "use_position_lr_decay": ("--position-lr-decay", "--no-position-lr-decay"),
    "position_lr_scale_init": "--position-lr-scale-init",
    "position_lr_scale_final": "--position-lr-scale-final",
    "lr_decay_start_iteration": "--lr-decay-start-iteration",
    "lr_decay_max_steps": "--lr-decay-max-steps",
    "ssim_weight": "--ssim-weight",
    "ssim_window_size": "--ssim-window-size",
    "ssim_sigma": "--ssim-sigma",
    "normal_consistency_weight": "--normal-consistency-weight",
    "normal_from_depth_use_mean_depth": (
        "--normal-from-depth-use-mean-depth",
        "--no-normal-from-depth-use-mean-depth",
    ),
    "depth_distort_weight": "--depth-distort-weight",
    "depth_distort_start_iteration": "--depth-distort-start-iteration",
    "opacity_prior_weight": "--opacity-prior-weight",
    "intra_slab_depth_weight": "--intra-slab-depth-weight",
    "curvature_scale_weight": "--curvature-scale-weight",
    "share_local_layer_direct_lighting": (
        "--share-local-layer-direct-lighting",
        "--no-share-local-layer-direct-lighting",
    ),
    "minimum_projected_footprint": (
        "--minimum-projected-footprint",
        "--no-minimum-projected-footprint",
    ),
    "minimum_projected_footprint_pixels": "--minimum-projected-footprint-pixels",
    "densification_interval": "--densification-interval",
    "prune_interval": "--prune-interval",
    "densify_after": "--densify-after",
    "prune_after": "--prune-after",
    "densification_verbose": ("--densification-verbose", "--no-densification-verbose"),
    "densification_grad_quantile": "--densification-grad-quantile",
    "densification_grad_abs_min": "--densification-grad-abs-min",
    "densification_grad_abs_min_final": "--densification-grad-abs-min-final",
    "densification_grad_abs_min_decay_start_iteration": "--densification-grad-abs-min-decay-start-iteration",
    "densification_grad_abs_min_decay_end_iteration": "--densification-grad-abs-min-decay-end-iteration",
    "curvature_violation_threshold": "--curvature-violation-threshold",
    "densification_scale_min": "--densification-scale-min",
    "densification_split_offset_scale": "--densification-split-offset-scale",
    "densification_split_scale_factor": "--densification-split-scale-factor",
    "densification_exact_clone_percent_dense": "--densification-exact-clone-percent-dense",
    "densification_scene_extent": "--densification-scene-extent",
    "densification_max_new_fraction": "--densification-max-new-fraction",
    "densification_stats_skip_interval_start": (
        "--densification-stats-skip-interval-start",
        "--no-densification-stats-skip-interval-start",
    ),
    "densification_downweight_normal_gradients": (
        "--densification-downweight-normal-gradients",
        "--no-densification-downweight-normal-gradients",
    ),
    "densification_tangent_only": (
        "--densification-tangent-only",
        "--no-densification-tangent-only",
    ),
    "densify_bsdf_floor": "--densify-bsdf-floor",
    "densify_bsdf_gamma": "--densify-bsdf-gamma",
    "opacity_prune_threshold": "--opacity-prune-threshold",
    "max_prune_fraction": "--max-prune-fraction",
    "min_surfel_area": "--min-surfel-area",
    "reset_opacity_interval": "--reset-opacity-interval",
    "reset_opacity_value": "--reset-opacity-value",
    "rebuild_bvh_interval": "--rebuild-bvh-interval",
    "inactive_transport_prune_cycles": "--inactive-transport-prune-cycles",
    "use_device_training_step": (
        "--device-training-step",
        "--no-device-training-step",
    ),
}


def parameter_digest(parameters: dict[str, Any]) -> str:
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:8]


def cli_args_for_parameter(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    if name not in CONFIG_CLI_FLAGS:
        raise KeyError(f"No CLI flag mapping for search parameter: {name}")

    flag = CONFIG_CLI_FLAGS[name]
    if isinstance(value, bool):
        if not isinstance(flag, tuple):
            raise TypeError(f"Boolean parameter {name} needs true/false CLI flags.")
        return [flag[0] if value else flag[1]]

    if isinstance(flag, tuple):
        raise TypeError(f"Parameter {name} uses boolean CLI flags but value is {value!r}.")

    if isinstance(value, (list, tuple)):
        return [flag, *[str(item) for item in value]]

    return [flag, str(value)]


def build_train_command(
    dataset_path: Path | None,
    output_dir: Path,
    parameters: dict[str, Any],
) -> list[str]:
    command = [sys.executable, str(PROJECT_ROOT / "main.py")]
    if dataset_path is not None:
        command.extend(["-s", str(dataset_path)])
    command.extend(["-o", str(output_dir)])

    for name in sorted(parameters):
        command.extend(cli_args_for_parameter(name, parameters[name]))

    return command
