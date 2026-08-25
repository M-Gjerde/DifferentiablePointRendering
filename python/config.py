from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import math


@dataclass
class RendererSettingsConfig:
    photons: float = 1e6
    bounces: int = 0
    adjoint_bounces: int = 1
    forward_passes: int = 1
    primal_shadow_rays: int = 1  # Li
    adjoint_shadow_rays: int = 1  # Li
    gather_passes: int = 1
    adjoint_passes: int = 3
    enable_adjoint_shadow_rays: bool = True
    adjoint_shadow_path_rays: int = 1  # p_i
    logging: int = 3

    def as_dict(self, config: "OptimizationConfig") -> Dict[str, float | int | bool]:
        return {
            "photons": self.photons,
            "bounces": self.bounces,
            "forward_passes": self.forward_passes,
            "gather_passes": self.gather_passes,
            "primal_shadow_rays": self.primal_shadow_rays,
            "adjoint_shadow_rays": self.adjoint_shadow_rays,
            "enable_adjoint_shadow_rays": self.enable_adjoint_shadow_rays,
            "adjoint_shadow_path_rays": self.adjoint_shadow_path_rays,
            "adjoint_bounces": self.adjoint_bounces,
            "adjoint_passes": self.adjoint_passes,
            "logging": self.logging,
            "depth_distort_weight": config.depth_distort_weight,
            "normal_consistency_weight": config.normal_consistency_weight,
            "normal_from_depth_use_mean_depth": config.normal_from_depth_use_mean_depth,
            "opacity_prior_weight": config.opacity_prior_weight,
        }


@dataclass
class OptimizationConfig:
    assets_root: Path = Path("../Assets")
    scene_xml: str = "cbox_custom.xml"
    pointcloud_ply: str = "initial.ply"
    dataset_path: Path = Path("./Output/target")
    output_dir: Path = Path("OptimizationOutput")
    output_dir_is_explicit: bool = False
    scene_xml_is_explicit: bool = False
    pointcloud_ply_is_explicit: bool = False
    checkpoint: Path | None = None
    resume_iteration_offset: int = 0

    device: str = "cpu"

    iterations: int = int(10.0e4)
    optimizer_type: str = "adam"
    # Learning rates
    learning_rate: float = 1.0
    learning_rate_position: float | None = None
    learning_rate_rotation: float | None = None
    max_rotation_step_radians: float = 0.01
    learning_rate_scale: float | None = None
    learning_rate_albedo: float | None = None
    learning_rate_opacity: float | None = None
    learning_rate_beta: float | None = None
    # Global LR scheduling
    use_global_lr_schedule: bool = True
    global_lr_scale_init: float = 10.0
    global_lr_scale_final: float = 2.0
    global_lr_start_iteration: int = 0
    global_lr_max_steps: int = int(2.0e4)

    depth_distort_weight: float = 2.0e3
    depth_distort_start_iteration: int = 0
    normal_consistency_weight: float = 0.05
    normal_from_depth_use_mean_depth: bool = False
    opacity_prior_weight: float = 0.0

    # Density control / EV-splitting
    # Ignore stats from the first half of each densification interval after cloning/pruning.
    densification_stats_skip_interval_start: bool = True
    densification_interval: int = 25
    prune_interval: int = 25
    densify_after: int = 0
    prune_after: int = 0
    densification_grad_quantile: float = 0.0
    densification_grad_abs_min: float = 6.0e-4
    densification_grad_abs_min_final: float = 1.25e-4
    densification_grad_abs_min_decay_start_iteration: int = 0
    densification_grad_abs_min_decay_end_iteration: int = 8_000
    densification_scale_min: float = 6.0e-3
    densification_split_offset_scale: float = 0.5
    densification_split_scale_factor: float = math.sqrt(4.0)
    densification_exact_clone_percent_dense: float = 0.003
    densification_scene_extent: float = 0.0
    densification_max_new_fraction: float = 1.0

    # More densification on radiometrically darker primitives
    densify_bsdf_floor = 0.1
    densify_bsdf_gamma = 1.0
    # Pruning
    opacity_prune_threshold: float = 0.10
    max_prune_fraction: float = 0.9
    min_surfel_area: float = math.pi * 5.0e-7
    inactive_gradient_prune_cycles: int = 1  # One cycle is one loop through all training cameras

    # Misc scheduling
    reset_opacity_interval: int = 0
    reset_opacity_value: float = 0.025
    reset_opacity_iterations: bool = False
    rebuild_bvh_interval: int = 1
    use_device_training_step: bool = True

    # Camera batching
    one_camera_per_iteration: bool = True
    camera_sampling_mode: str = "round_robin"  # "round_robin" or "random"
    camera_sampling_seed: int = 0
    scale_single_camera_gradients: bool = False

    # Logging
    log_interval: int = 5
    save_interval: int = 50
    save_ply_files_interval: int = save_interval

    # Mesh Extraction
    mesh_extraction_interval: int = 1_000
    mesh_extraction_depth_key: str = "median_depth"
    mesh_extraction_mesh_res: int = 1024
    mesh_extraction_num_cluster: int = 50
    save_final_mesh: bool = True
    # Iteration snapshot content
    save_snapshot_rgb: bool = True
    save_snapshot_median_depth: bool = True
    save_snapshot_depth_distortion: bool = False
    save_snapshot_visible_normal: bool = False
    save_snapshot_normal_from_depth: bool = True
    save_snapshot_grad: bool = False
    densification_verbose: bool = False


def resolve_learning_rates(config: OptimizationConfig) -> None:
    base_learning_rate = config.learning_rate

    if config.optimizer_type == "sgd":
        factor_position = 0.2
        factor_rotation = 0.1
        factor_scale = 0.005
        factor_albedo = 2.0
        factor_opacity = 1.0
        factor_beta = 0.00
    elif config.optimizer_type == "adam":
        factor_position = 0.0001
        factor_rotation = 0.001
        factor_scale = 0.0001
        factor_albedo = 0.0015
        factor_opacity = 0.0005
        factor_beta = 0.002
    else:
        raise ValueError(f"Unknown optimizer_type: {config.optimizer_type}")

    if config.learning_rate_position is None:
        config.learning_rate_position = factor_position * base_learning_rate
    if config.learning_rate_rotation is None:
        config.learning_rate_rotation = factor_rotation * base_learning_rate
    if config.learning_rate_scale is None:
        config.learning_rate_scale = factor_scale * base_learning_rate
    if config.learning_rate_albedo is None:
        config.learning_rate_albedo = factor_albedo * base_learning_rate
    if config.learning_rate_opacity is None:
        config.learning_rate_opacity = factor_opacity * base_learning_rate
    if config.learning_rate_beta is None:
        config.learning_rate_beta = factor_beta * base_learning_rate


def scale_iteration_interval_by_learning_rate(base_interval: int, learning_rate: float, ) -> int:
    if base_interval <= 0:
        return base_interval
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    return max(1, math.ceil(float(base_interval) / learning_rate))


def _load_checkpoint_run_config(checkpoint_dir: Path) -> dict:
    run_config_path = checkpoint_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"--checkpoint run directory is missing run_config.json: {run_config_path}")

    with run_config_path.open("r", encoding="utf-8") as run_config_file:
        return json.load(run_config_file)


def _checkpoint_config_value(run_config: dict, key: str):
    if key in run_config:
        return run_config[key]

    optimization_config = run_config.get("optimization_config", {})
    if isinstance(optimization_config, dict):
        return optimization_config.get(key)

    return None


def _last_metrics_iteration(checkpoint_dir: Path) -> int | None:
    metrics_csv_path = checkpoint_dir / "metrics.csv"
    if not metrics_csv_path.is_file():
        return None

    last_iteration: int | None = None
    with metrics_csv_path.open("r", encoding="utf-8", newline="") as metrics_file:
        for row in csv.DictReader(metrics_file):
            value = row.get("iteration")
            if value is None:
                continue
            try:
                last_iteration = max(last_iteration or 0, int(float(value)))
            except ValueError:
                continue

    return last_iteration


def _checkpoint_resume_iteration_offset(checkpoint_dir: Path, run_config: dict) -> int:
    last_metrics_iteration = _last_metrics_iteration(checkpoint_dir)
    if last_metrics_iteration is not None:
        return max(0, int(last_metrics_iteration))

    configured_iterations = _checkpoint_config_value(run_config, "iterations")
    if configured_iterations is None:
        return 0

    try:
        prior_offset = _checkpoint_config_value(run_config, "resume_iteration_offset")
        if prior_offset is None:
            prior_offset = 0
        return max(0, int(prior_offset) + int(configured_iterations))
    except (TypeError, ValueError):
        return 0


def configure_checkpoint(config: OptimizationConfig, cli_overrides: set[str]) -> None:
    if config.checkpoint is None:
        return

    checkpoint_dir = config.checkpoint.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"--checkpoint is not an existing run directory: {checkpoint_dir}")

    checkpoint_points_path = checkpoint_dir / "points_final.ply"
    if not checkpoint_points_path.is_file():
        raise FileNotFoundError(f"Could not find points_final.ply in checkpoint directory: {checkpoint_points_path}")

    run_config = _load_checkpoint_run_config(checkpoint_dir)

    if "assets_root" not in cli_overrides:
        assets_root = _checkpoint_config_value(run_config, "assets_root")
        if assets_root is not None:
            config.assets_root = Path(assets_root)

    if "scene_xml" not in cli_overrides:
        scene_xml = _checkpoint_config_value(run_config, "scene_xml")
        if scene_xml is None:
            raise KeyError(f"Checkpoint run_config.json is missing scene_xml: {checkpoint_dir / 'run_config.json'}")
        config.scene_xml = str(scene_xml)
        config.scene_xml_is_explicit = True

    if "dataset_path" not in cli_overrides:
        dataset_path = _checkpoint_config_value(run_config, "dataset_path")
        if dataset_path is None:
            raise KeyError(f"Checkpoint run_config.json is missing dataset_path: {checkpoint_dir / 'run_config.json'}")
        config.dataset_path = Path(dataset_path)

    config.checkpoint = checkpoint_dir
    config.pointcloud_ply = str(checkpoint_points_path)
    config.pointcloud_ply_is_explicit = True
    if "resume_iteration_offset" not in cli_overrides:
        config.resume_iteration_offset = _checkpoint_resume_iteration_offset(
            checkpoint_dir=checkpoint_dir,
            run_config=run_config,
        )

    print(f"[checkpoint] Run directory       : {checkpoint_dir}")
    print(f"[checkpoint] Scene XML           : {config.scene_xml}")
    print(f"[checkpoint] Dataset path        : {config.dataset_path}")
    print(f"[checkpoint] Initial point cloud : {checkpoint_points_path}")
    print(f"[checkpoint] Resume iteration   : {config.resume_iteration_offset}")


def parse_args() -> OptimizationConfig:
    parser = argparse.ArgumentParser(
        description="Optimize point positions using a custom differentiable renderer.",
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--scene", "--scene-xml", dest="scene_xml", type=str)
    parser.add_argument("--pointcloud", "--ply", dest="pointcloud_ply", type=str)
    parser.add_argument("-s", "--dataset-path", type=Path)
    parser.add_argument("--output", "-o", "-m", "--output-dir", dest="output_dir", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--optimizer", dest="optimizer_type", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--save-ply-files-interval", type=int)
    parser.add_argument(
        "--mesh-extraction-interval",
        "--mesh-checkpoint-interval",
        "--mesh-save-interval",
        "--mesh-interval",
        "--mesh-interval-save",
        dest="mesh_extraction_interval",
        type=int,
        help="Save a mesh checkpoint every N iterations. Use 0 to disable intermediate mesh checkpoints.",
    )
    parser.add_argument("--mesh-extraction-depth-key", type=str, choices=["median_depth", "mean_depth"])
    parser.add_argument("--mesh-extraction-mesh-res", type=int)
    parser.add_argument("--mesh-extraction-num-cluster", type=int)
    parser.add_argument("--save-final-mesh", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Prior optimization run directory. Reuses scene/dataset/assets from "
            "<checkpoint>/run_config.json and uses <checkpoint>/points_final.ply "
            "as this run's initial point cloud."
        ),
    )
    parser.add_argument(
        "--resume-iteration-offset",
        type=int,
        help=(
            "Global iteration offset for resumed schedules. Defaults to the "
            "checkpoint metrics.csv max iteration, falling back to the "
            "checkpoint run_config iterations."
        ),
    )
    parser.add_argument("--device", type=str)
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--lr-pos", dest="learning_rate_position", type=float)
    parser.add_argument("--lr-rot", dest="learning_rate_rotation", type=float)
    parser.add_argument("--max-rotation-step-radians", type=float)
    parser.add_argument("--lr-scale", dest="learning_rate_scale", type=float)
    parser.add_argument("--lr-albedo", dest="learning_rate_albedo", type=float)
    parser.add_argument("--lr-opacity", dest="learning_rate_opacity", type=float)
    parser.add_argument("--lr-beta", dest="learning_rate_beta", type=float)
    parser.add_argument("--global-lr-schedule", dest="use_global_lr_schedule",
                        action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--global-lr-scale-init", type=float)
    parser.add_argument("--global-lr-scale-final", type=float)
    parser.add_argument("--global-lr-start-iteration", type=int)
    parser.add_argument("--global-lr-max-steps", type=int)
    parser.add_argument("--normal-consistency-weight", dest="normal_consistency_weight", type=float)
    parser.add_argument("--normal-from-depth-use-mean-depth", dest="normal_from_depth_use_mean_depth",
                        action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--depth-distort-weight", dest="depth_distort_weight", type=float)
    parser.add_argument("--depth-distort-start-iteration", type=int)
    parser.add_argument(
        "--opacity-prior-weight",
        dest="opacity_prior_weight",
        type=float,
    )
    # Density control / EV-splitting
    parser.add_argument("--densification-interval", type=int)
    parser.add_argument("--prune-interval", type=int)
    parser.add_argument("--densify-after", type=int)
    parser.add_argument("--prune-after", type=int)
    parser.add_argument("--densification-verbose", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, )
    parser.add_argument("--densification-grad-quantile", type=float)
    parser.add_argument("--densification-grad-abs-min", type=float)
    parser.add_argument("--densification-grad-abs-min-final", type=float)
    parser.add_argument("--densification-scale-min", type=float)
    parser.add_argument("--densification-split-offset-scale", type=float)
    parser.add_argument("--densification-split-scale-factor", type=float)
    parser.add_argument(
        "--densification-exact-clone-percent-dense",
        "--densification-percent-dense",
        dest="densification_exact_clone_percent_dense",
        type=float,
    )
    parser.add_argument("--densification-scene-extent", type=float)
    parser.add_argument("--densification-max-new-fraction", type=float)
    parser.add_argument(
        "--densification-grad-abs-min-decay-start-iteration",
        "--densification-grad-abs-min-iter-start",
        dest="densification_grad_abs_min_decay_start_iteration",
        type=int,
    )
    parser.add_argument(
        "--densification-grad-abs-min-decay-end-iteration",
        "--densification-grad-abs-min-iter-end",
        dest="densification_grad_abs_min_decay_end_iteration",
        type=int,
    )
    parser.add_argument(
        "--densification-stats-skip-interval-start",
        "--densification-stats-warmup",
        dest="densification_stats_skip_interval_start",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help=(
            "Ignore densification gradient stats from the first half of each "
            "densification interval. The older --densification-stats-warmup "
            "spelling is accepted as an alias."
        ),
    )
    parser.add_argument("--densify-bsdf-floor", type=float)
    parser.add_argument("--densify-bsdf-gamma", type=float)
    # Pruning
    parser.add_argument("--opacity-prune-threshold", type=float)
    parser.add_argument("--max-prune-fraction", type=float)
    # Misc scheduling
    parser.add_argument("--reset-opacity-interval", type=int)
    parser.add_argument("--reset-opacity-value", type=float)
    parser.add_argument("--rebuild-bvh-interval", type=int)
    parser.add_argument("--inactive-gradient-prune-cycles", type=int)
    parser.add_argument(
        "--device-training-step",
        dest="use_device_training_step",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Use the device-resident fixed-topology RGB optimizer path when compatible.",
    )

    args = parser.parse_args()
    cli_overrides = set(vars(args).keys())

    config = OptimizationConfig()

    for parameter_name, parameter_value in vars(args).items():
        if not hasattr(config, parameter_name):
            raise RuntimeError(f"CLI argument produced unknown config field: {parameter_name}")
        setattr(config, parameter_name, parameter_value)

    config.output_dir_is_explicit = "output_dir" in cli_overrides
    config.scene_xml_is_explicit = "scene_xml" in cli_overrides
    config.pointcloud_ply_is_explicit = "pointcloud_ply" in cli_overrides

    configure_checkpoint(config, cli_overrides)

    resolve_learning_rates(config)

    return config
