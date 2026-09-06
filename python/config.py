from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RendererSettingsConfig:
    photons: float = 1e6
    bounces: int = 0
    adjoint_bounces: int = 1
    forward_passes: int = 1
    primal_shadow_rays: int = 1  # Li
    adjoint_shadow_rays: int = 1  # Li
    gather_passes: int = 1
    adjoint_passes: int = 2
    enable_adjoint_shadow_rays: bool = True
    adjoint_shadow_path_rays: int = 1  # p_i
    logging: int = 3

    def as_dict(self, config: "OptimizationConfig") -> dict[str, float | int | bool]:
        settings = asdict(self)
        settings.update({
            "depth_distort_weight": config.depth_distort_weight,
            "depth_distort_world_space": config.depth_distort_world_space,
            "normal_consistency_weight": config.normal_consistency_weight,
            "normal_from_depth_use_mean_depth": config.normal_from_depth_use_mean_depth,
            "opacity_prior_weight": config.opacity_prior_weight,
            "intra_slab_depth_weight": config.intra_slab_depth_weight,
            "curvature_scale_weight": config.curvature_scale_weight,
            "share_local_layer_direct_lighting": config.share_local_layer_direct_lighting,
            "enable_curvature_densification": (
                config.curvature_violation_threshold > 0.0
                and config.densification_interval > 0
            ),
            "enable_primal_activity_tracking": (
                config.inactive_transport_prune_cycles > 0
            ),
        })
        return settings


@dataclass
class OptimizationConfig:
    # Inputs and run location
    assets_root: Path = Path("../Assets")
    scene_xml: str = "cbox_custom.xml"
    pointcloud_ply: str = "initial.ply"
    dataset_path: Path = Path("./Output/target")
    target_color_space: str = "srgb"
    output_dir: Path = Path("OptimizationOutput")
    checkpoint: Path | None = None

    # Execution
    device: str = "cpu"
    iterations: int = 30_000
    optimizer_type: str = "adam"
    use_device_training_step: bool = True

    # Optimizer: base learning rates
    # Uniform multiplier applied to every component learning rate below.
    learning_rate: float = 1.0
    # Calibrated from the photometric-only global LR search (0.11x).
    learning_rate_position: float = 0.000055
    learning_rate_rotation: float = 0.005
    learning_rate_scale: float = 0.0005
    learning_rate_albedo: float = 0.0005
    learning_rate_opacity: float = 0.0002
    learning_rate_beta: float = 0.0005
    # Optimizer: learning-rate schedules
    # Multiplicative decay. All parameter groups receive the
    # global scale; position optionally receives a second position-only scale.
    use_global_lr_decay: bool = FalseK
    global_lr_scale_init: float = 1.0
    global_lr_scale_final: float = 0.25
    use_position_lr_decay: bool = True
    position_lr_scale_init: float = 10.0
    position_lr_scale_final: float = 1.0
    lr_decay_start_iteration: int = 0
    lr_decay_max_steps: int = 25_000

    # Objective: photometric loss
    ssim_weight: float = 0.00
    ssim_window_size: int = 5
    ssim_sigma: float = 0.75

    # Objective: geometric and parameter regularizers
    depth_distort_weight: float = 0.02
    # Use linear camera-forward depth in scene units instead of inverse-depth NDC.
    depth_distort_world_space: bool = True
    depth_distort_start_iteration: int = 0
    normal_consistency_weight: float = 0.005
    opacity_prior_weight: float = 0.0
    intra_slab_depth_weight: float = 1.0e-4
    curvature_scale_weight: float = 0.0e-0

    # Rendering model
    share_local_layer_direct_lighting: bool = True

    # Camera sampling
    one_camera_per_iteration: bool = True
    camera_sampling_mode: str = "round_robin"  # "round_robin" or "random"
    camera_sampling_seed: int = 0
    scale_single_camera_gradients: bool = False
    normal_from_depth_use_mean_depth: bool = False

    # Densification: schedule
    densification_interval: int = 200
    densify_after: int = 0
    densification_stats_skip_interval_start: bool = True

    # Densification: gradient signal
    # Auxiliary relative half-MSE statistics; parameter updates retain the RGB loss.
    densification_relative_error: bool = True
    densification_radiance_floor: float = 0.001  # linear RGB radiance units
    densification_full_position: bool = True
    densification_downweight_normal_gradients: bool = False
    # Legacy albedo normalization; ignored when relative-error statistics are enabled.
    densify_bsdf_floor: float = 0.01
    densify_bsdf_gamma: float = 1.0

    # Densification: base selection threshold
    # Absolute mode bypasses global and radiance-band score quantiles.
    # Both modes retain the bounded brightness preference below.
    densification_threshold_mode: str = "absolute"  # "absolute" or "quantile"
    densification_grad_abs_min: float = 5.0e-4
    densification_grad_abs_min_final: float = 5.0e-4
    densification_grad_abs_min_decay_start_iteration: int = 0
    densification_grad_abs_min_decay_end_iteration: int = 0

    # Densification: quantile selection (used only in "quantile" mode)
    densification_grad_quantile: float = 0.75
    # Apply the gradient quantile independently in log2 rendered/target
    # radiance bands. Values <= 1 disable radiance stratification.
    densification_radiance_quantile_bins: int = 16
    densification_radiance_quantile_min_bin_size: int = 16

    # Densification: radiance balancing (used in both threshold modes)
    # Divide final selection thresholds by a bounded, median-relative brightness
    # weight. Applied after threshold selection; strength 0 disables the bias.
    densification_radiance_bias_strength: float=  1.0
    densification_radiance_bias_min_weight: float = 0.25
    densification_radiance_bias_max_weight: float = 2.0

    # Densification: curvature trigger and clone/split policy
    # A non-positive value disables curvature-triggered densification.
    curvature_violation_threshold: float = -1
    densification_scale_min: float = 6.0e-3
    densification_exact_clone_percent_dense: float = 0.00
    densification_scene_extent: float = 0.0
    densification_split_offset_scale: float = 0.1
    densification_split_scale_factor: float = math.sqrt(2)
    # When false, position-triggered splits may use the full 3D gradient,
    # including the surfel-normal direction.
    densification_tangent_only: bool = False
    densification_max_new_fraction: float = 1.0
    densification_verbose: bool = False

    # Pruning and topology maintenance
    prune_interval: int = 100
    prune_after: int = 0
    opacity_prune_threshold: float = 0.0
    max_prune_fraction: float = 0.9
    min_surfel_area: float = math.pi * 2.0e-5
    inactive_transport_prune_cycles: int = 1
    reset_opacity_interval: int = 0
    reset_opacity_value: float = 0.025
    reset_opacity_iterations: bool = False
    rebuild_bvh_interval: int = densification_interval

    # Output and monitoring
    log_interval: int = 25
    # When enabled (> 0), save images on the first iteration, immediately before
    # each scheduled densification, and on the final iteration.
    save_interval: int = 100
    # When enabled (> 0), also save the first iteration, matching image snapshots.
    save_ply_files_interval: int = save_interval
    # Debug snapshots at the iteration immediately before the next scheduled
    # densification, replacing periodic PLY saves. Interval 0 still disables saves.
    save_ply_before_densification: bool = True
    save_snapshot_rgb: bool = True
    save_snapshot_median_depth: bool = False
    save_snapshot_depth_distortion: bool = False
    save_snapshot_visible_normal: bool = False
    save_snapshot_normal_from_depth: bool = False
    save_snapshot_grad: bool = False
    enable_metrics: bool = True
    enable_image_preview: bool = True

    # Mesh extraction and evaluation
    mesh_extraction_interval: int = 2_000
    mesh_extraction_depth_key: str = "median_depth"
    mesh_extraction_mesh_res: int = 768
    mesh_extraction_num_cluster: int = 50
    save_final_mesh: bool = True
    ground_truth: Path | None = None
    geometry_samples: int = 500_000
    geometry_seed: int = 0
    geometry_scale: float = 1.0
    geometry_use_vertices: bool = True

    # Internal CLI/checkpoint state
    output_dir_is_explicit: bool = False
    scene_xml_is_explicit: bool = False
    pointcloud_ply_is_explicit: bool = False
    resume_iteration_offset: int = 0


def resolve_learning_rates(config: OptimizationConfig) -> None:
    learning_rate_fields = (
        "learning_rate_position",
        "learning_rate_rotation",
        "learning_rate_scale",
        "learning_rate_albedo",
        "learning_rate_opacity",
        "learning_rate_beta",
    )
    multiplier = float(config.learning_rate)
    if not math.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError(f"learning_rate must be finite and non-negative, got {multiplier}")

    for field_name in learning_rate_fields:
        base_learning_rate = float(getattr(config, field_name))
        if not math.isfinite(base_learning_rate) or base_learning_rate < 0.0:
            raise ValueError(
                f"{field_name} must be finite and non-negative, got {base_learning_rate}"
            )
        setattr(config, field_name, base_learning_rate * multiplier)


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

    inherited_fields = (
        ("assets_root", Path, False),
        ("scene_xml", str, True),
        ("dataset_path", Path, True),
        ("target_color_space", str, False),
    )
    for field_name, convert, required in inherited_fields:
        if field_name in cli_overrides:
            continue
        value = _checkpoint_config_value(run_config, field_name)
        if value is None:
            if required:
                raise KeyError(
                    f"Checkpoint run_config.json is missing {field_name}: "
                    f"{checkpoint_dir / 'run_config.json'}"
                )
            continue
        setattr(config, field_name, convert(value))

    if "scene_xml" not in cli_overrides:
        config.scene_xml_is_explicit = True

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


def _add_typed_fields(parser, value_type, *field_names: str) -> None:
    for field_name in field_names:
        parser.add_argument(f"--{field_name.replace('_', '-')}", type=value_type)


def _add_boolean_argument(parser, *flags: str, **kwargs) -> None:
    parser.add_argument(
        *flags,
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        **kwargs,
    )


def parse_args() -> OptimizationConfig:
    parser = argparse.ArgumentParser(
        description="Optimize point positions using a custom differentiable renderer.",
        argument_default=argparse.SUPPRESS,
    )

    inputs = parser.add_argument_group("inputs and run")
    inputs.add_argument("--assets-root", type=Path)
    inputs.add_argument("--scene", "--scene-xml", dest="scene_xml", type=str)
    inputs.add_argument("--pointcloud", "--ply", dest="pointcloud_ply", type=str)
    inputs.add_argument("-s", "--dataset-path", type=Path)
    inputs.add_argument(
        "--target-color-space",
        choices=["auto", "srgb", "linear"],
        help=(
            "Target image encoding. 'auto' uses ICC metadata and file/sample conventions; "
            "all targets are converted to linear sRGB for optimization."
        ),
    )
    inputs.add_argument("--output", "-o", "-m", "--output-dir", dest="output_dir", type=Path)
    inputs.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Prior optimization run directory. Reuses scene/dataset/assets from "
            "<checkpoint>/run_config.json and uses <checkpoint>/points_final.ply "
            "as this run's initial point cloud."
        ),
    )
    inputs.add_argument(
        "--resume-iteration-offset",
        type=int,
        help=(
            "Global iteration offset for resumed schedules. Defaults to the "
            "checkpoint metrics.csv max iteration, falling back to the "
            "checkpoint run_config iterations."
        ),
    )
    inputs.add_argument("--device", type=str)

    optimizer = parser.add_argument_group("optimizer and learning-rate schedule")
    optimizer.add_argument("--iterations", type=int)
    optimizer.add_argument(
        "--optimizer",
        dest="optimizer_type",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
    )
    optimizer.add_argument(
        "--lr",
        "--learning-rate",
        dest="learning_rate",
        type=float,
        help="Uniform multiplier applied to all component learning rates defined in config.py.",
    )
    for suffix, field_name in (
        ("pos", "position"),
        ("rot", "rotation"),
        ("scale", "scale"),
        ("albedo", "albedo"),
        ("opacity", "opacity"),
        ("beta", "beta"),
    ):
        optimizer.add_argument(f"--lr-{suffix}", dest=f"learning_rate_{field_name}", type=float)
    _add_boolean_argument(optimizer, "--global-lr-decay", dest="use_global_lr_decay")
    _add_boolean_argument(optimizer, "--position-lr-decay", dest="use_position_lr_decay")
    _add_typed_fields(
        optimizer,
        float,
        "global_lr_scale_init",
        "global_lr_scale_final",
        "position_lr_scale_init",
        "position_lr_scale_final",
    )
    optimizer.add_argument(
        "--lr-decay-start-iteration",
        "--global-lr-start-iteration",
        dest="lr_decay_start_iteration",
        type=int,
    )
    optimizer.add_argument(
        "--lr-decay-max-steps",
        "--global-lr-max-steps",
        dest="lr_decay_max_steps",
        type=int,
    )
    _add_boolean_argument(
        optimizer,
        "--device-training-step",
        dest="use_device_training_step",
        help="Use the device-resident fixed-topology RGB optimizer path when compatible.",
    )

    objective = parser.add_argument_group("objective and regularizers")
    objective.add_argument(
        "--ssim-weight",
        type=float,
        help="DSSIM mixture weight in [0,1]; 0 restores the previous half-MSE-only RGB loss.",
    )
    objective.add_argument("--ssim-window-size", type=int)
    _add_typed_fields(
        objective,
        float,
        "ssim_sigma",
        "normal_consistency_weight",
        "depth_distort_weight",
        "opacity_prior_weight",
        "intra_slab_depth_weight",
        "curvature_scale_weight",
    )
    _add_boolean_argument(
        objective,
        "--depth-distort-world-space",
        help="Measure distortion using linear camera-forward depth in scene units instead of inverse-depth NDC; retune --depth-distort-weight when switching.",
    )
    objective.add_argument("--depth-distort-start-iteration", type=int)
    _add_boolean_argument(
        objective,
        "--normal-from-depth-use-mean-depth",
        dest="normal_from_depth_use_mean_depth",
    )

    rendering = parser.add_argument_group("rendering")
    _add_boolean_argument(
        rendering,
        "--share-local-layer-direct-lighting",
        dest="share_local_layer_direct_lighting",
        help="Share one point-light transport vertex and shadow connection across each local slab.",
    )

    densification_schedule = parser.add_argument_group("densification: schedule")
    _add_typed_fields(
        densification_schedule,
        int,
        "densification_interval",
        "densify_after",
    )
    _add_boolean_argument(
        densification_schedule,
        "--densification-stats-skip-interval-start",
        "--densification-stats-warmup",
        dest="densification_stats_skip_interval_start",
        help=(
            "Ignore densification gradient stats from the first half of each "
            "densification interval. The older --densification-stats-warmup "
            "spelling is accepted as an alias."
        ),
    )

    densification_signal = parser.add_argument_group("densification: gradient signal")
    _add_boolean_argument(
        densification_signal, "--densification-relative-error",
        help="Use a frozen per-pixel relative-MSE source for densification and disable the albedo boost.",
    )
    _add_boolean_argument(
        densification_signal, "--densification-full-position",
        help="Include all position derivatives in relative statistics; disable to retain the local footprint signal.",
    )
    densification_signal.add_argument(
        "--densification-radiance-floor", type=float,
        help="Positive linear-RGB floor in the symmetric relative-error normalizer.",
    )
    _add_boolean_argument(
        densification_signal,
        "--densification-downweight-normal-gradients",
        help=(
            "Downweight tangent densification statistics when position gradients "
            "point mostly along the surfel normal. Disabled by default."
        ),
    )
    _add_typed_fields(
        densification_signal,
        float,
        "densify_bsdf_floor",
        "densify_bsdf_gamma",
    )

    densification_selection = parser.add_argument_group("densification: candidate selection")
    densification_selection.add_argument(
        "--densification-threshold-mode", choices=["absolute", "quantile"],
        help="Use only the scheduled absolute threshold, or combine it with global/radiance-band quantiles; both retain brightness bias.",
    )
    _add_typed_fields(
        densification_selection,
        float,
        "densification_grad_abs_min",
        "densification_grad_abs_min_final",
    )
    densification_selection.add_argument(
        "--densification-grad-abs-min-decay-start-iteration",
        "--densification-grad-abs-min-iter-start",
        dest="densification_grad_abs_min_decay_start_iteration",
        type=int,
    )
    densification_selection.add_argument(
        "--densification-grad-abs-min-decay-end-iteration",
        "--densification-grad-abs-min-iter-end",
        dest="densification_grad_abs_min_decay_end_iteration",
        type=int,
    )

    densification_quantiles = parser.add_argument_group(
        'densification: quantile selection (mode="quantile")'
    )
    densification_quantiles.add_argument("--densification-grad-quantile", type=float)
    _add_typed_fields(
        densification_quantiles,
        int,
        "densification_radiance_quantile_bins",
        "densification_radiance_quantile_min_bin_size",
    )

    densification_radiance = parser.add_argument_group("densification: radiance balancing")
    _add_typed_fields(
        densification_radiance,
        float,
        "densification_radiance_bias_strength",
        "densification_radiance_bias_min_weight",
        "densification_radiance_bias_max_weight",
    )

    densification_split = parser.add_argument_group("densification: curvature and split policy")
    _add_typed_fields(
        densification_split,
        float,
        "densification_scale_min",
        "densification_split_offset_scale",
        "densification_split_scale_factor",
        "densification_scene_extent",
        "densification_max_new_fraction",
    )
    densification_split.add_argument(
        "--curvature-violation-threshold",
        type=float,
        help=(
            "Mean raw curvature-scale violation required to split a surfel; "
            "a non-positive value disables curvature densification."
        ),
    )
    densification_split.add_argument(
        "--densification-exact-clone-percent-dense",
        "--densification-percent-dense",
        dest="densification_exact_clone_percent_dense",
        type=float,
    )
    _add_boolean_argument(
        densification_split,
        "--densification-tangent-only",
        help=(
            "Restrict position-triggered densification displacement to the surfel "
            "tangent plane. Use --no-densification-tangent-only to retain the "
            "full 3D gradient direction."
        ),
    )
    _add_boolean_argument(densification_split, "--densification-verbose")

    pruning = parser.add_argument_group("pruning and topology maintenance")
    _add_typed_fields(
        pruning,
        int,
        "prune_interval",
        "prune_after",
        "reset_opacity_interval",
        "rebuild_bvh_interval",
        "inactive_transport_prune_cycles",
    )
    _add_typed_fields(
        pruning,
        float,
        "opacity_prune_threshold",
        "max_prune_fraction",
        "min_surfel_area",
        "reset_opacity_value",
    )

    output = parser.add_argument_group("output and monitoring")
    _add_typed_fields(output, int, "log_interval", "save_interval", "save_ply_files_interval")
    _add_boolean_argument(
        output, "--save-ply-before-densification",
        help="Save PLY one iteration before each scheduled densification instead of periodically; --save-ply-files-interval 0 disables these saves.",
    )
    _add_boolean_argument(
        output,
        "--metrics",
        dest="enable_metrics",
        help="Launch analyze/view_metrics_live.py alongside the optimization.",
    )
    _add_boolean_argument(
        output,
        "--image-preview",
        dest="enable_image_preview",
        help="Launch the live image preview helper while optimizing.",
    )

    mesh = parser.add_argument_group("mesh extraction and evaluation")
    mesh.add_argument(
        "--mesh-extraction-interval",
        "--mesh-checkpoint-interval",
        "--mesh-save-interval",
        "--mesh-interval",
        "--mesh-interval-save",
        dest="mesh_extraction_interval",
        type=int,
        help="Save a mesh checkpoint every N iterations. Use 0 to disable intermediate mesh checkpoints.",
    )
    mesh.add_argument("--mesh-extraction-depth-key", type=str, choices=["median_depth", "mean_depth"])
    _add_typed_fields(mesh, int, "mesh_extraction_mesh_res", "mesh_extraction_num_cluster")
    _add_boolean_argument(mesh, "--save-final-mesh")
    mesh.add_argument(
        "--ground-truth",
        "--gt",
        dest="ground_truth",
        type=Path,
        help=(
            "Optional ground-truth mesh or point cloud. When set, extracted mesh "
            "checkpoints are evaluated into geometry_metrics.csv."
        ),
    )
    _add_typed_fields(mesh, int, "geometry_samples", "geometry_seed")
    _add_typed_fields(mesh, float, "geometry_scale")
    _add_boolean_argument(
        mesh,
        "--geometry-use-vertices",
        help="Use reconstructed mesh vertices as point-to-triangle geometry queries.",
    )

    args = parser.parse_args()
    cli_overrides = set(vars(args).keys())

    config = OptimizationConfig()

    for parameter_name, parameter_value in vars(args).items():
        if not hasattr(config, parameter_name):
            raise RuntimeError(f"CLI argument produced unknown config field: {parameter_name}")
        setattr(config, parameter_name, parameter_value)

    config.output_dir_is_explicit = (
        config.output_dir_is_explicit or "output_dir" in cli_overrides
    )
    config.scene_xml_is_explicit = "scene_xml" in cli_overrides
    config.pointcloud_ply_is_explicit = "pointcloud_ply" in cli_overrides

    configure_checkpoint(config, cli_overrides)

    if not math.isfinite(config.densification_radiance_floor) or config.densification_radiance_floor <= 0:
        parser.error("--densification-radiance-floor must be finite and positive")
    if config.densification_radiance_quantile_bins < 1:
        parser.error("--densification-radiance-quantile-bins must be at least 1")
    if config.densification_radiance_quantile_min_bin_size < 1:
        parser.error("--densification-radiance-quantile-min-bin-size must be at least 1")
    if not math.isfinite(config.densification_radiance_bias_strength) or config.densification_radiance_bias_strength < 0:
        parser.error("--densification-radiance-bias-strength must be finite and non-negative")
    if not math.isfinite(config.densification_radiance_bias_min_weight) or not 0 < config.densification_radiance_bias_min_weight <= 1:
        parser.error("--densification-radiance-bias-min-weight must be finite and in (0, 1]")
    if not math.isfinite(config.densification_radiance_bias_max_weight) or config.densification_radiance_bias_max_weight < 1:
        parser.error("--densification-radiance-bias-max-weight must be finite and at least 1")

    resolve_learning_rates(config)

    return config
