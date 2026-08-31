from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from config import scheduled_iteration_interval
from optimizers import (create_learning_rate_schedules, update_optimizer_learning_rates, )
from training_helpers import *
from density_control import *


def extract_mesh_from_point_cloud(
        config: OptimizationConfig,
        points_path: Path,
        mesh_output_subdir: Path,
        log_prefix: str,
) -> None:
    extract_mesh_script = Path(__file__).resolve().with_name("extract_mesh.py")

    command = [
        sys.executable,
        str(extract_mesh_script),
        "--ply",
        str(points_path),
        "--mesh-output-subdir",
        str(mesh_output_subdir),
        "--depth-key",
        str(config.mesh_extraction_depth_key),
        "--mesh-res",
        str(int(config.mesh_extraction_mesh_res)),
        "--num-cluster",
        str(int(config.mesh_extraction_num_cluster)),
    ]

    print(
        f"{log_prefix} Extracting mesh to "
        f"{config.output_dir / mesh_output_subdir}"
    )

    result = subprocess.run(
        command,
        cwd=extract_mesh_script.parent,
        check=False,
    )

    if result.returncode != 0:
        print(
            f"{log_prefix} Mesh extraction failed "
            f"with exit code {result.returncode}: {shlex.join(command)}"
        )


def extract_mesh_checkpoint(config: OptimizationConfig, iteration: int, points_path: Path) -> None:
    extract_mesh_from_point_cloud(
        config=config,
        points_path=points_path,
        mesh_output_subdir=Path("mesh_checkpoints") / f"iter_{iteration:05d}",
        log_prefix=f"[Iter {iteration:04d}]",
    )


def extract_final_mesh(config: OptimizationConfig, points_path: Path) -> None:
    extract_mesh_from_point_cloud(
        config=config,
        points_path=points_path,
        mesh_output_subdir=Path("mesh"),
        log_prefix="[Final]",
    )


@dataclass
class IterationGradientResult:
    active_camera_name: str
    forward_out: Optional[Dict[str, dict]]
    loss_state: Dict[str, Any]
    averaged_loss_state: Dict[str, Any]
    photo_gradients: Dict[str, np.ndarray]
    depth_regularizer_gradients: Dict[str, np.ndarray]
    normal_regularizer_gradients: Dict[str, np.ndarray]
    opacity_prior_gradients: Dict[str, np.ndarray]
    intra_slab_depth_gradients: Dict[str, np.ndarray]
    curvature_scale_gradients: Dict[str, np.ndarray]
    surface_regularizer_gradients: Dict[str, np.ndarray]
    total_gradients: Dict[str, np.ndarray]
    adjoint_images: Dict[str, Any]
    photo_gradient_surfel_stats: Dict[str, Any]


DENSIFICATION_ORIGIN_INITIAL = np.uint8(0)
DENSIFICATION_ORIGIN_CLONE = np.uint8(1)
DENSIFICATION_ORIGIN_SPLIT = np.uint8(2)


def make_new_densification_origin_np(densification_result: dict | None, n_new: int) -> np.ndarray:
    new_origin_np = np.full((int(n_new),), DENSIFICATION_ORIGIN_INITIAL, dtype=np.uint8)
    if densification_result is None or n_new <= 0:
        return new_origin_np

    clone_count = max(0, int(densification_result.get("clone_count", 0)))
    split_count = max(0, int(densification_result.get("split_count", 0)))
    clone_count = min(clone_count, n_new)
    split_count = min(split_count, n_new - clone_count)

    if clone_count > 0:
        new_origin_np[:clone_count] = DENSIFICATION_ORIGIN_CLONE
    if split_count > 0:
        new_origin_np[clone_count:clone_count + split_count] = DENSIFICATION_ORIGIN_SPLIT

    return new_origin_np


def active_densification_origin_counts(densification_origin_np: np.ndarray) -> tuple[int, int]:
    return (
        int(np.count_nonzero(densification_origin_np == DENSIFICATION_ORIGIN_CLONE)),
        int(np.count_nonzero(densification_origin_np == DENSIFICATION_ORIGIN_SPLIT)),
    )


def active_densification_interval_for_iteration(
        config: OptimizationConfig,
        base_densification_interval: int,
        final_densification_interval: int,
        iteration: int,
) -> int:
    if config.use_global_lr_schedule:
        return scheduled_iteration_interval(
            initial_interval=base_densification_interval,
            final_interval=final_densification_interval,
            iteration=iteration,
            start_iteration=int(config.global_lr_start_iteration),
            max_steps=int(config.global_lr_max_steps),
        )
    return base_densification_interval


def densification_stats_skip_for_interval(config: OptimizationConfig, densification_interval: int) -> int:
    if not bool(
            getattr(
                config,
                "densification_stats_skip_interval_start",
                getattr(config, "densification_stats_warmup", True),
            )
    ):
        return 0
    return max(int(densification_interval) // 2, 0)


def next_densification_iteration_after(
        current_iteration: int,
        densify_after: int,
        densification_interval: int,
) -> Optional[int]:
    if densification_interval <= 0:
        return None
    return max(int(densify_after), int(current_iteration) + max(int(densification_interval), 1))


def active_camera_name_for_iteration(
        active_training_camera_ids: Sequence[str],
        one_camera_per_iteration: bool,
) -> str:
    return active_training_camera_ids[0] if one_camera_per_iteration else "ALL_CAMERAS"


def make_rgb_loss_state(
        active_training_camera_ids: Sequence[str],
        adjoint_images: Dict[str, Any],
) -> Dict[str, Any]:
    loss_state = make_zero_loss_values()
    loss_state.update({
        "depth_distortion_grad_images": {},
        "visible_normal_adjoints": {},
        "depth_normal_adjoints": {},
        "depth_distortion_maps_for_logging": {},
        "per_camera_loss_values": {},
    })

    rgb_loss_values = adjoint_images.get("loss_values", {})
    for camera_name in active_training_camera_ids:
        rgb_loss_value = float(rgb_loss_values[camera_name])
        camera_loss_values = make_zero_loss_values()
        camera_loss_values["total_rgb_loss_value"] = rgb_loss_value
        camera_loss_values["total_loss_value"] = rgb_loss_value
        loss_state["per_camera_loss_values"][camera_name] = camera_loss_values
        loss_state["total_rgb_loss_value"] += rgb_loss_value
        loss_state["total_loss_value"] += rgb_loss_value

    return loss_state


def add_regularizer_loss_state(
        loss_state: Dict[str, Any],
        regularizer_loss_state: Dict[str, Any],
) -> None:
    for loss_key in LOSS_VALUE_KEYS:
        loss_state[loss_key] += regularizer_loss_state[loss_key]

    for camera_name, camera_loss_values in regularizer_loss_state["per_camera_loss_values"].items():
        existing_camera_loss_values = loss_state["per_camera_loss_values"].setdefault(
            camera_name,
            make_zero_loss_values(),
        )
        for loss_key in LOSS_VALUE_KEYS:
            existing_camera_loss_values[loss_key] += camera_loss_values[loss_key]

    loss_state["depth_distortion_grad_images"] = regularizer_loss_state["depth_distortion_grad_images"]
    loss_state["visible_normal_adjoints"] = regularizer_loss_state["visible_normal_adjoints"]
    loss_state["depth_normal_adjoints"] = regularizer_loss_state["depth_normal_adjoints"]
    loss_state["depth_distortion_maps_for_logging"] = regularizer_loss_state["depth_distortion_maps_for_logging"]
    loss_state["intra_slab_depth_maps_for_logging"] = regularizer_loss_state.get(
        "intra_slab_depth_maps_for_logging", {}
    )
    loss_state["curvature_scale_maps_for_logging"] = regularizer_loss_state.get(
        "curvature_scale_maps_for_logging", {}
    )


def make_device_training_step_options(
        config: OptimizationConfig,
        active_learning_rates: Dict[str, float],
        camera_batch_scale: float,
        return_gradient_stats: bool = False,
        include_depth_distortion: bool = False,
        include_normal_consistency: bool = False,
        include_opacity_prior: bool = False,
        include_intra_slab_depth: bool = False,
        include_curvature_scale: bool = False,
) -> Dict[str, Any]:
    return {
        "optimizer": config.optimizer_type,
        "learning_rate_position": active_learning_rates.get(
            "position",
            float(config.learning_rate_position),
        ),
        "learning_rate_rotation": active_learning_rates.get(
            "rotation",
            float(config.learning_rate_rotation),
        ),
        "learning_rate_scale": active_learning_rates.get(
            "scale",
            float(config.learning_rate_scale),
        ),
        "learning_rate_albedo": active_learning_rates.get(
            "albedo",
            float(config.learning_rate_albedo),
        ),
        "learning_rate_opacity": active_learning_rates.get(
            "opacity",
            float(config.learning_rate_opacity),
        ),
        "learning_rate_beta": active_learning_rates.get(
            "beta",
            float(config.learning_rate_beta),
        ),
        "camera_batch_scale": camera_batch_scale,
        "return_gradient_stats": return_gradient_stats,
        "include_depth_distortion": include_depth_distortion,
        "include_normal_consistency": include_normal_consistency,
        "include_opacity_prior": include_opacity_prior,
        "include_intra_slab_depth": include_intra_slab_depth,
        "include_curvature_scale": include_curvature_scale,
    }


def active_surfel_mask_from_position_records(
        photo_gradient_surfel_stats: Dict[str, Any],
        point_count: int,
) -> np.ndarray:
    position_record_count_per_camera_np = photo_gradient_surfel_stats.get(
        "position_record_count_per_camera",
        None,
    )
    if position_record_count_per_camera_np is None:
        raise RuntimeError(
            "Inactive-surfel pruning requires "
            "adjoint_images['gradient_stats']['position_record_count_per_camera']."
        )

    position_record_count_per_camera_np = np.asarray(position_record_count_per_camera_np)

    if position_record_count_per_camera_np.shape[0] != point_count:
        raise RuntimeError(
            "Position-record-count shape mismatch: "
            f"expected {point_count}, got {position_record_count_per_camera_np.shape[0]}"
        )

    if position_record_count_per_camera_np.ndim == 1:
        return position_record_count_per_camera_np > 0

    return np.any(position_record_count_per_camera_np > 0, axis=1)


def compute_iteration_gradients(
        renderer: pale.Renderer,
        active_training_camera_ids: Sequence[str],
        latest_loss_values_by_camera: Dict[str, Dict[str, float]],
        training_camera_ids: Sequence[str],
        iteration: int,
        one_camera_per_iteration: bool,
        use_depth_distortion_gradients: bool,
        use_depth_distortion: bool,
        use_normal_consistency: bool,
        use_opacity_prior: bool,
        use_intra_slab_depth: bool,
        use_curvature_scale: bool,
        active_depth_distortion_weight: float,
        normal_consistency_weight: float,
        active_opacity_prior_weight: float,
        intra_slab_depth_weight: float,
        curvature_scale_weight: float,
) -> IterationGradientResult:
    active_camera_name = active_camera_name_for_iteration(
        active_training_camera_ids,
        one_camera_per_iteration,
    )
    photo_gradients, adjoint_images = renderer.render_rgb_loss_backward(
        list(active_training_camera_ids)
    )
    loss_state = make_rgb_loss_state(active_training_camera_ids, adjoint_images)

    forward_out = None
    depth_regularizer_gradients: Dict[str, np.ndarray] = {}
    normal_regularizer_gradients: Dict[str, np.ndarray] = {}
    opacity_prior_gradients: Dict[str, np.ndarray] = {}
    intra_slab_depth_gradients: Dict[str, np.ndarray] = {}
    curvature_scale_gradients: Dict[str, np.ndarray] = {}
    surface_regularizer_gradients: Dict[str, np.ndarray] = {}

    if (use_depth_distortion_gradients or use_normal_consistency or use_opacity_prior or
            use_intra_slab_depth or use_curvature_scale):
        if (
                hasattr(renderer, "render_forward_surface_regularizer_loss_and_adjoint")
                and hasattr(renderer, "render_surface_regularizers_backward_from_current_adjoint")
        ):
            regularizer_loss_state = renderer.render_forward_surface_regularizer_loss_and_adjoint(
                list(active_training_camera_ids),
                {
                    "depth_distortion_weight": active_depth_distortion_weight,
                    "normal_consistency_weight": normal_consistency_weight,
                    "opacity_prior_weight": active_opacity_prior_weight,
                    "intra_slab_depth_weight": intra_slab_depth_weight,
                    "curvature_scale_weight": curvature_scale_weight,
                    "use_depth_distortion": use_depth_distortion,
                    "use_normal_consistency": use_normal_consistency,
                    "use_opacity_prior": use_opacity_prior,
                    "use_intra_slab_depth": use_intra_slab_depth,
                    "use_curvature_scale": use_curvature_scale,
                },
            )
            add_regularizer_loss_state(loss_state, regularizer_loss_state)

            surface_regularizer_components = renderer.render_surface_regularizers_backward_from_current_adjoint(
                list(active_training_camera_ids),
                True,
            )
        else:
            forward_out = (
                renderer.render_forward(active_camera_name)
                if one_camera_per_iteration
                else renderer.render_forward()
            )

            regularizer_loss_state = compute_surface_regularizer_losses_and_adjoints(
                forward_out=forward_out,
                training_camera_ids=list(active_training_camera_ids),
                depth_distortion_weight=active_depth_distortion_weight,
                normal_consistency_weight=normal_consistency_weight,
                opacity_prior_weight=active_opacity_prior_weight,
                intra_slab_depth_weight=intra_slab_depth_weight,
                curvature_scale_weight=curvature_scale_weight,
                use_depth_distortion=use_depth_distortion,
                use_normal_consistency=use_normal_consistency,
                use_opacity_prior=use_opacity_prior,
                use_intra_slab_depth=use_intra_slab_depth,
                use_curvature_scale=use_curvature_scale,
            )
            add_regularizer_loss_state(loss_state, regularizer_loss_state)

            surface_regularizer_components = renderer.render_surface_regularizers_backward(
                list(active_training_camera_ids),
                loss_state["depth_distortion_grad_images"],
                loss_state["visible_normal_adjoints"],
                loss_state["depth_normal_adjoints"],
                loss_state["intra_slab_depth_grad_images"],
                loss_state["curvature_scale_grad_images"],
            )
        depth_regularizer_gradients = surface_regularizer_components["depth_distortion"]
        normal_regularizer_gradients = surface_regularizer_components["normal_consistency"]
        opacity_prior_gradients = surface_regularizer_components.get("opacity_prior", {})
        intra_slab_depth_gradients = surface_regularizer_components.get("intra_slab_depth", {})
        curvature_scale_gradients = surface_regularizer_components.get("curvature_scale", {})

        repair_nonfinite_gradient_dict_inplace("depth_regularizer_gradients", depth_regularizer_gradients, iteration)
        repair_nonfinite_gradient_dict_inplace("normal_regularizer_gradients", normal_regularizer_gradients, iteration)
        repair_nonfinite_gradient_dict_inplace("opacity_prior_gradients", opacity_prior_gradients, iteration)
        repair_nonfinite_gradient_dict_inplace(
            "intra_slab_depth_gradients", intra_slab_depth_gradients, iteration
        )
        repair_nonfinite_gradient_dict_inplace(
            "curvature_scale_gradients", curvature_scale_gradients, iteration
        )
        surface_regularizer_gradients = sum_gradient_dicts(
            depth_regularizer_gradients,
            normal_regularizer_gradients,
            opacity_prior_gradients,
            intra_slab_depth_gradients,
            curvature_scale_gradients,
        )

    photo_gradient_surfel_stats = adjoint_images.get("gradient_stats", {})

    repair_nonfinite_gradient_dict_inplace("photo_gradients", photo_gradients, iteration)
    repair_nonfinite_gradient_dict_inplace(
        "surface_regularizer_gradients",
        surface_regularizer_gradients,
        iteration,
    )
    total_gradients = sum_gradient_dicts(photo_gradients, surface_regularizer_gradients)

    for camera_name, camera_loss_values in loss_state["per_camera_loss_values"].items():
        latest_loss_values_by_camera[camera_name] = dict(camera_loss_values)

    averaged_loss_state = make_averaged_loss_state_from_camera_cache(
        latest_loss_values_by_camera=latest_loss_values_by_camera,
        expected_camera_ids=list(training_camera_ids),
    )

    return IterationGradientResult(
        active_camera_name=active_camera_name,
        forward_out=forward_out,
        loss_state=loss_state,
        averaged_loss_state=averaged_loss_state,
        photo_gradients=photo_gradients,
        depth_regularizer_gradients=depth_regularizer_gradients,
        normal_regularizer_gradients=normal_regularizer_gradients,
        opacity_prior_gradients=opacity_prior_gradients,
        intra_slab_depth_gradients=intra_slab_depth_gradients,
        curvature_scale_gradients=curvature_scale_gradients,
        surface_regularizer_gradients=surface_regularizer_gradients,
        total_gradients=total_gradients,
        adjoint_images=adjoint_images,
        photo_gradient_surfel_stats=photo_gradient_surfel_stats,
    )


def run_optimization(renderer: pale.Renderer, config: OptimizationConfig,
                     renderer_settings: RendererSettingsConfig) -> None:
    target_images, training_camera_ids, all_camera_ids = load_target_images(renderer, Path(config.dataset_path))

    depth_distortion_base_weight = float(getattr(config, "depth_distort_weight", 0.0))
    depth_distortion_start_iteration = int(getattr(config, "depth_distort_start_iteration", 0))

    normal_consistency_weight = float(getattr(config, "normal_consistency_weight", 0.0))
    opacity_prior_weight = float(getattr(config, "opacity_prior_weight", 0.0))
    intra_slab_depth_weight = float(getattr(config, "intra_slab_depth_weight", 0.0))
    curvature_scale_weight = float(getattr(config, "curvature_scale_weight", 0.0))
    save_ply_files_interval = int(config.save_ply_files_interval)
    mesh_extraction_interval = int(getattr(config, "mesh_extraction_interval", 1_000))

    if save_ply_files_interval < 0:
        raise ValueError(f"save_ply_files_interval must be >= 0, got {save_ply_files_interval}")
    if mesh_extraction_interval < 0:
        raise ValueError(f"mesh_extraction_interval must be >= 0, got {mesh_extraction_interval}")

    use_depth_distortion = depth_distortion_base_weight != 0.0
    use_normal_consistency = normal_consistency_weight != 0.0
    use_opacity_prior = opacity_prior_weight != 0.0
    use_intra_slab_depth = intra_slab_depth_weight != 0.0
    use_curvature_scale = curvature_scale_weight != 0.0

    print(
        "Loss terms: "
        f"depth_distortion={use_depth_distortion} "
        f"base_weight={depth_distortion_base_weight:.3e} "
        f"start_iter={depth_distortion_start_iteration}, "
        f"normal_consistency={use_normal_consistency} weight={normal_consistency_weight:.3e}, "
        f"opacity_prior={use_opacity_prior} weight={opacity_prior_weight:.3e}, "
        f"intra_slab_depth={use_intra_slab_depth} weight={intra_slab_depth_weight:.3e}, "
        f"curvature_scale={use_curvature_scale} weight={curvature_scale_weight:.3e}"
    )
    renderer.upload_training_targets(target_images)

    initial_params = fetch_parameters(renderer)
    initial_params_reference = make_initial_params_reference(initial_params)
    print(f"Fetched {initial_params['position'].shape[0]} initial points from PLY.")

    device = torch.device(config.device)

    positions, rotations, scales, albedos, opacities, betas, powers = create_torch_parameters_from_initial(
        initial_params, device)
    rotation_delta = torch.nn.Parameter(torch.zeros((positions.shape[0], 3), device=device, dtype=torch.float32))

    trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)
    frozen_surfel_count = int((~trainable_surfel_mask).sum().item())
    print(f"Frozen emissive surfels: {frozen_surfel_count} / {int(trainable_surfel_mask.numel())}")

    verify_parameters_inplane(
        positions,
        rotations,
        scales,
        albedos,
        opacities,
        betas,
        trainable_surfel_mask=trainable_surfel_mask,
    )

    apply_point_parameters(renderer, positions, rotations, scales, albedos, opacities, betas, powers)
    rebuild_bvh(renderer)

    positions, rotations, scales, albedos, opacities, betas, powers = refetch_parameters_as_torch(renderer, device)
    rotation_delta = torch.nn.Parameter(torch.zeros((positions.shape[0], 3), device=device, dtype=torch.float32))
    config.output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = create_masked_optimizer(config, positions, rotation_delta, scales, albedos, opacities, betas, powers)
    learning_rate_schedules = create_learning_rate_schedules(config)
    resume_iteration_offset = max(0, int(getattr(config, "resume_iteration_offset", 0)))
    final_global_iteration = resume_iteration_offset + int(config.iterations)
    if resume_iteration_offset > 0:
        print(
            "[checkpoint] Continuing iteration-dependent schedules from "
            f"iteration {resume_iteration_offset}; this run will end at "
            f"global iteration {final_global_iteration}."
        )
    active_learning_rates = update_optimizer_learning_rates(
        optimizer,
        learning_rate_schedules,
        resume_iteration_offset,
    )

    initial_images = renderer.render_forward()
    initial_depth_distortion_weight = scheduled_regularizer_weight(
        depth_distortion_base_weight,
        iteration=resume_iteration_offset,
        start_iteration=depth_distortion_start_iteration,
    )

    initial_loss_tuple = compute_initial_losses_and_save_outputs(
        output_dir=config.output_dir, initial_images=initial_images, target_images=target_images,
        all_camera_ids=all_camera_ids, positions=positions, rotations=rotations,
        scales=scales, albedos=albedos, opacities=opacities, betas=betas, powers=powers,
        depth_distortion_weight=initial_depth_distortion_weight,
        normal_consistency_weight=normal_consistency_weight,
        opacity_prior_weight=opacity_prior_weight,
        intra_slab_depth_weight=intra_slab_depth_weight,
        curvature_scale_weight=curvature_scale_weight,
        use_depth_distortion=use_depth_distortion,
        use_normal_consistency=use_normal_consistency,
        use_opacity_prior=use_opacity_prior,
        use_intra_slab_depth=use_intra_slab_depth,
        use_curvature_scale=use_curvature_scale,
    )

    print_loss_summary("Initial", *initial_loss_tuple)

    base_densification_interval = int(config.densification_interval)
    final_densification_interval = int(config.densification_interval_final)
    densification_interval = active_densification_interval_for_iteration(
        config=config,
        base_densification_interval=base_densification_interval,
        final_densification_interval=final_densification_interval,
        iteration=resume_iteration_offset,
    )
    prune_interval = int(config.prune_interval)
    densify_after = config.densify_after if config.densify_after >= 0 else densification_interval
    prune_after = config.prune_after if config.prune_after >= 0 else prune_interval
    densification_cycle_start_iteration = resume_iteration_offset
    densification_cycle_interval = densification_interval
    densification_stats_skip_iterations = densification_stats_skip_for_interval(
        config=config,
        densification_interval=densification_cycle_interval,
    )
    next_densification_iteration = next_densification_iteration_after(
        current_iteration=resume_iteration_offset,
        densify_after=densify_after,
        densification_interval=densification_interval,
    )

    opacity_prune_threshold = float(config.opacity_prune_threshold)
    max_prune_fraction = float(config.max_prune_fraction)
    inactive_gradient_prune_cycles = int(config.inactive_gradient_prune_cycles)
    reset_opacity_interval = int(config.reset_opacity_interval)
    reset_opacity_value = float(config.reset_opacity_value)
    densification_verbose = bool(config.densification_verbose)
    densification_grad_quantile = as_config_float(config.densification_grad_quantile)
    densification_grad_abs_min = float(config.densification_grad_abs_min)
    densification_grad_abs_min_final = float(config.densification_grad_abs_min_final)
    densification_grad_abs_min_decay_start_iteration = int(config.densification_grad_abs_min_decay_start_iteration)
    densification_grad_abs_min_decay_end_iteration = int(config.densification_grad_abs_min_decay_end_iteration)
    densify_bsdf_floor = float(config.densify_bsdf_floor)
    densify_bsdf_gamma = float(config.densify_bsdf_gamma)
    rebuild_bvh_interval = max(int(config.rebuild_bvh_interval), 1)
    use_device_training_step = bool(getattr(config, "use_device_training_step", True))
    device_training_disabled_reasons: list[str] = []
    required_device_training_methods = (
        "render_rgb_training_step",
        "render_rgb_backward_from_current_forward",
        "render_forward_surface_regularizer_loss_and_adjoint",
        "render_surface_regularizers_backward_from_current_adjoint",
        "apply_device_training_step",
        "reset_trainable_opacity_on_gpu",
        "sync_point_parameters_from_gpu",
        "capture_device_adam_state",
        "upload_device_adam_state",
    )
    missing_device_training_methods = [
        method_name
        for method_name in required_device_training_methods
        if not hasattr(renderer, method_name)
    ]
    if missing_device_training_methods:
        device_training_disabled_reasons.append(
            "renderer binding is missing "
            + ", ".join(missing_device_training_methods)
        )
    if str(config.optimizer_type).lower() != "adam":
        device_training_disabled_reasons.append("device path currently matches Adam only")
    use_device_training_step = use_device_training_step and not device_training_disabled_reasons
    if use_device_training_step:
        print("[device-training-step] Enabled device-resident optimizer path.")
    elif bool(getattr(config, "use_device_training_step", True)):
        print(
            "[device-training-step] Disabled: "
            + "; ".join(device_training_disabled_reasons)
        )

    densify_position_grad_accum_np = np.zeros((positions.shape[0], 1), dtype=np.float32)
    densify_position_grad_denom_np = np.zeros((positions.shape[0], 1), dtype=np.float32)
    # Stores local tangent coordinates (u, v, 0); converted to world space at clone time.
    densify_position_grad_vector_accum_np = np.zeros(tuple(positions.shape), dtype=np.float32)
    active_during_camera_cycle_np = np.zeros((positions.shape[0],), dtype=bool)
    inactive_gradient_cycle_count_np = np.zeros((positions.shape[0],), dtype=np.uint32, )
    densification_origin_np = np.full(
        (positions.shape[0],),
        DENSIFICATION_ORIGIN_INITIAL,
        dtype=np.uint8,
    )
    visited_training_camera_ids_this_cycle: set[str] = set()

    metrics_csv_path = config.output_dir / "metrics.csv"
    total_start_time = time.perf_counter()
    last_log_iteration = 0
    last_log_time = total_start_time
    iteration = 0
    latest_loss_values_by_camera: Dict[str, Dict[str, float]] = {}
    densification_clone_points_total = 0
    densification_split_points_total = 0

    with open(metrics_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        write_metrics_header(csv_writer)

        try:
            for iteration in range(1, config.iterations + 1):
                iteration_start = time.perf_counter()
                global_iteration = resume_iteration_offset + iteration

                densification_interval = active_densification_interval_for_iteration(
                    config=config,
                    base_densification_interval=base_densification_interval,
                    final_densification_interval=final_densification_interval,
                    iteration=global_iteration,
                )

                active_training_camera_ids = select_active_training_camera_ids(
                    training_camera_ids=training_camera_ids,
                    iteration=global_iteration,
                    config=config,
                )

                active_depth_distortion_weight = scheduled_regularizer_weight(
                    depth_distortion_base_weight,
                    iteration=global_iteration,
                    start_iteration=depth_distortion_start_iteration,
                )
                active_opacity_prior_weight = opacity_prior_weight
                active_densification_grad_abs_min = scheduled_densification_grad_abs_min(
                    initial_threshold=densification_grad_abs_min,
                    final_threshold=densification_grad_abs_min_final,
                    iteration=global_iteration,
                    start_iteration=densification_grad_abs_min_decay_start_iteration,
                    end_iteration=densification_grad_abs_min_decay_end_iteration,
                )

                use_depth_distortion_gradients = active_depth_distortion_weight != 0.0
                if use_device_training_step:
                    active_learning_rates = update_optimizer_learning_rates(
                        optimizer,
                        learning_rate_schedules,
                        global_iteration,
                    )
                    active_camera_name = active_camera_name_for_iteration(
                        active_training_camera_ids,
                        config.one_camera_per_iteration,
                    )
                    camera_batch_scale = (
                        float(len(training_camera_ids)) / float(len(active_training_camera_ids))
                        if config.one_camera_per_iteration and config.scale_single_camera_gradients
                        else 1.0
                    )
                    use_surface_regularizers = (
                            use_depth_distortion_gradients
                            or use_normal_consistency
                            or use_opacity_prior
                            or use_intra_slab_depth
                            or use_curvature_scale
                    )
                    needs_gradient_stats = (
                            densification_interval > 0
                            or inactive_gradient_prune_cycles > 0
                    )
                    device_step_options = make_device_training_step_options(
                        config=config,
                        active_learning_rates=active_learning_rates,
                        camera_batch_scale=camera_batch_scale,
                        return_gradient_stats=needs_gradient_stats,
                    )

                    regularizer_loss_state = None
                    if use_surface_regularizers:
                        regularizer_loss_state = renderer.render_forward_surface_regularizer_loss_and_adjoint(
                            list(active_training_camera_ids),
                            {
                                "depth_distortion_weight": active_depth_distortion_weight,
                                "normal_consistency_weight": normal_consistency_weight,
                                "opacity_prior_weight": active_opacity_prior_weight,
                                "intra_slab_depth_weight": intra_slab_depth_weight,
                                "curvature_scale_weight": curvature_scale_weight,
                                "use_depth_distortion": use_depth_distortion,
                                "use_normal_consistency": use_normal_consistency,
                                "use_opacity_prior": use_opacity_prior,
                                "use_intra_slab_depth": use_intra_slab_depth,
                                "use_curvature_scale": use_curvature_scale,
                            },
                        )
                        adjoint_images = renderer.render_rgb_backward_from_current_forward(
                            list(active_training_camera_ids),
                            {"return_gradient_stats": needs_gradient_stats},
                        )
                    else:
                        adjoint_images = renderer.render_rgb_training_step(
                            list(active_training_camera_ids),
                            device_step_options,
                        )

                    loss_state = make_rgb_loss_state(active_training_camera_ids, adjoint_images)

                    if use_surface_regularizers:
                        add_regularizer_loss_state(loss_state, regularizer_loss_state)

                        renderer.render_surface_regularizers_backward_from_current_adjoint(
                            list(active_training_camera_ids)
                        )

                        apply_result = renderer.apply_device_training_step(
                            make_device_training_step_options(
                                config=config,
                                active_learning_rates=active_learning_rates,
                                camera_batch_scale=camera_batch_scale,
                                include_depth_distortion=use_depth_distortion_gradients,
                                include_normal_consistency=use_normal_consistency,
                                include_opacity_prior=use_opacity_prior,
                                include_intra_slab_depth=use_intra_slab_depth,
                                include_curvature_scale=use_curvature_scale,
                            )
                        )
                        adjoint_images["point_count"] = apply_result.get(
                            "point_count",
                            adjoint_images.get("point_count", int(positions.shape[0])),
                        )
                        adjoint_images["optimizer_step"] = apply_result.get("optimizer_step", 0)

                    for camera_name, camera_loss_values in loss_state["per_camera_loss_values"].items():
                        latest_loss_values_by_camera[camera_name] = dict(camera_loss_values)

                    averaged_loss_state = make_averaged_loss_state_from_camera_cache(
                        latest_loss_values_by_camera=latest_loss_values_by_camera,
                        expected_camera_ids=list(training_camera_ids),
                    )

                    photo_gradient_surfel_stats = adjoint_images.get("gradient_stats", {})
                    if inactive_gradient_prune_cycles > 0:
                        active_during_camera_cycle_np |= active_surfel_mask_from_position_records(
                            photo_gradient_surfel_stats,
                            point_count=int(positions.shape[0]),
                        )
                    visited_training_camera_ids_this_cycle.update(active_training_camera_ids)
                    camera_cycle_complete = (
                            len(visited_training_camera_ids_this_cycle) == len(training_camera_ids)
                    )

                    clone_signal_per_camera_np = photo_gradient_surfel_stats.get("clone_signal_per_camera", None)
                    clone_signal_record_count_per_camera_np = photo_gradient_surfel_stats.get(
                        "clone_signal_record_count_per_camera",
                        None,
                    )
                    update_densification_statistics(
                        iteration=global_iteration,
                        densification_interval=densification_cycle_interval,
                        densification_cycle_start_iteration=densification_cycle_start_iteration,
                        densification_stats_skip_iterations=densification_stats_skip_iterations,
                        densify_position_grad_accum_np=densify_position_grad_accum_np,
                        densify_position_grad_denom_np=densify_position_grad_denom_np,
                        densify_position_grad_vector_accum_np=densify_position_grad_vector_accum_np,
                        rotations=rotations,
                        albedos=albedos,
                        trainable_surfel_mask=trainable_surfel_mask,
                        densify_bsdf_floor=densify_bsdf_floor,
                        densify_bsdf_gamma=densify_bsdf_gamma,
                        densify_position_grad_per_camera_np=clone_signal_per_camera_np,
                        densify_position_grad_per_camera_count_np=clone_signal_record_count_per_camera_np,
                    )

                    scheduled_opacity_reset = (
                            reset_opacity_interval > 0
                            and global_iteration % reset_opacity_interval == 0
                    )
                    manual_opacity_reset = bool(config.reset_opacity_iterations)
                    did_reset_opacity = scheduled_opacity_reset or manual_opacity_reset

                    if did_reset_opacity:
                        renderer.reset_trainable_opacity_on_gpu(float(reset_opacity_value))
                        print(f"[Iter {global_iteration:04d}] Resetting all opacities to {reset_opacity_value}")
                        config.reset_opacity_iterations = False

                    densification_is_due = (
                            densification_interval > 0
                            and next_densification_iteration is not None
                            and global_iteration >= next_densification_iteration
                    )
                    should_check_densification = not did_reset_opacity and densification_is_due
                    should_check_prune = (
                            not did_reset_opacity
                            and prune_interval > 0
                            and global_iteration >= prune_after
                            and global_iteration % prune_interval == 0
                    )
                    should_check_inactive_prune = (
                            not did_reset_opacity
                            and camera_cycle_complete
                            and global_iteration >= prune_after
                            and inactive_gradient_prune_cycles > 0
                    )
                    if (
                            should_check_densification
                            or should_check_prune
                            or should_check_inactive_prune
                    ):
                        renderer.sync_point_parameters_from_gpu()
                        positions, rotations, scales, albedos, opacities, betas, powers = (
                            refetch_parameters_as_torch(renderer, device)
                        )
                        rotation_delta = torch.nn.Parameter(
                            torch.zeros((positions.shape[0], 3), device=device, dtype=torch.float32)
                        )
                        trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)
                        verify_parameters_inplane(
                            positions,
                            rotations,
                            scales,
                            albedos,
                            opacities,
                            betas,
                            trainable_surfel_mask=trainable_surfel_mask,
                        )

                    densification_result = None
                    scale_prune_indices = []
                    opacity_prune_indices = []
                    indices_to_remove_list = []
                    inactive_camera_cycle_indices = np.zeros((0,), dtype=np.int64)
                    prune_scale_area_points = 0
                    prune_inactive_gradient_points = 0

                    if not did_reset_opacity:
                        if should_check_densification:
                            densification_result = maybe_make_densification_result(
                                iteration=global_iteration, config=config, positions=positions, rotations=rotations,
                                scales=scales, albedos=albedos, opacities=opacities,
                                betas=betas, powers=powers, trainable_surfel_mask=trainable_surfel_mask,
                                densify_position_grad_accum_np=densify_position_grad_accum_np,
                                densify_position_grad_denom_np=densify_position_grad_denom_np,
                                densify_position_grad_vector_accum_np=densify_position_grad_vector_accum_np,
                                densify_after=densify_after,
                                densification_interval=densification_interval,
                                densification_verbose=densification_verbose,
                                densification_grad_quantile=densification_grad_quantile,
                                densification_grad_abs_min=active_densification_grad_abs_min,
                                force_densification=True,
                            )

                        if should_check_prune:
                            scale_prune_indices, opacity_prune_indices, indices_to_remove_list = maybe_make_prune_indices(
                                iteration=global_iteration, config=config, scales=scales, opacities=opacities,
                                trainable_surfel_mask=trainable_surfel_mask, prune_after=prune_after,
                                prune_interval=prune_interval, reset_opacity_interval=reset_opacity_interval,
                                opacity_prune_threshold=opacity_prune_threshold,
                                max_prune_fraction=max_prune_fraction,
                            )

                        if should_check_inactive_prune:
                            trainable_surfel_mask_np = (
                                trainable_surfel_mask.detach().cpu().numpy().astype(bool).reshape(-1))
                            active_this_cycle_np = (trainable_surfel_mask_np & active_during_camera_cycle_np)
                            inactive_this_cycle_np = (trainable_surfel_mask_np & ~active_during_camera_cycle_np)
                            inactive_gradient_cycle_count_np[active_this_cycle_np] = 0
                            inactive_gradient_cycle_count_np[inactive_this_cycle_np] = np.minimum(
                                inactive_gradient_cycle_count_np[inactive_this_cycle_np] + 1,
                                inactive_gradient_prune_cycles,
                            )
                            inactive_camera_cycle_indices = np.flatnonzero(trainable_surfel_mask_np & (
                                    inactive_gradient_cycle_count_np >= inactive_gradient_prune_cycles)).astype(
                                np.int64)

                            if inactive_camera_cycle_indices.size > 0:
                                indices_to_remove_list.extend(int(index) for index in inactive_camera_cycle_indices)
                    else:
                        print(f"[Iter {global_iteration:04d}] Skipping densification/pruning due to opacity reset")

                    if indices_to_remove_list or densification_result is not None:
                        old_point_count_for_topology = int(positions.shape[0])
                        keep_mask_np = np.ones(old_point_count_for_topology, dtype=bool)
                        device_adam_state_snapshot = None

                        if densification_result is not None:
                            protected_src = np.asarray(
                                densification_result.get("source_index", np.zeros((0,), dtype=np.int64)),
                                dtype=np.int64).reshape(-1)
                            if protected_src.size > 0 and indices_to_remove_list:
                                protected_set = set(int(i) for i in protected_src)
                                indices_to_remove_list = [int(i) for i in indices_to_remove_list if
                                                          int(i) not in protected_set]

                        n_new_for_topology = 0
                        if densification_result is not None:
                            new_block_for_topology = densification_result.get("new", None)
                            if new_block_for_topology is not None:
                                n_new_for_topology = int(new_block_for_topology["position"].shape[0])

                        if indices_to_remove_list or n_new_for_topology > 0:
                            device_adam_state_snapshot = renderer.capture_device_adam_state()

                        if densification_result is not None:
                            apply_densification_source_updates_inplace(
                                densification_result, positions, rotations, scales,
                                albedos, opacities, betas, powers,
                            )
                            verify_parameters_inplane(
                                positions, rotations, scales, albedos, opacities, betas,
                                trainable_surfel_mask=trainable_surfel_mask,
                            )
                            apply_point_parameters(
                                renderer, positions, rotations, scales,
                                albedos, opacities, betas, powers,
                            )

                        if indices_to_remove_list:
                            scale_prune_set = set(int(i) for i in scale_prune_indices)
                            opacity_prune_set = set(int(i) for i in opacity_prune_indices)
                            overlap_set = scale_prune_set & opacity_prune_set
                            indices_to_remove = np.unique(np.asarray(indices_to_remove_list, dtype=np.int64))

                            inactive_cycle_prune_set = set(int(index) for index in inactive_camera_cycle_indices)
                            removed_index_set = set(int(index) for index in indices_to_remove)
                            prune_scale_area_points = len(scale_prune_set & removed_index_set)
                            prune_inactive_gradient_points = len(inactive_cycle_prune_set & removed_index_set)

                            if config.densification_verbose:
                                print(
                                    f"[Iter {global_iteration:04d}] Pruning {indices_to_remove.size} unique surfels | "
                                    f"scale={len(scale_prune_set)}, "
                                    f"opacity={len(opacity_prune_set)}, "
                                    f"inactive_gradient={len(inactive_cycle_prune_set)} "
                                    f"(threshold={inactive_gradient_prune_cycles} cycles), "
                                    f"both_scale_opacity={len(overlap_set)}, "
                                    f"scale_only={len(scale_prune_set - opacity_prune_set)}, "
                                    f"opacity_only={len(opacity_prune_set - scale_prune_set)}"
                                )

                            keep_mask_np[indices_to_remove] = False
                            remove_points(renderer, indices_to_remove)
                            densify_position_grad_accum_np = densify_position_grad_accum_np[keep_mask_np]
                            densify_position_grad_denom_np = densify_position_grad_denom_np[keep_mask_np]
                            densify_position_grad_vector_accum_np = densify_position_grad_vector_accum_np[keep_mask_np]
                            active_during_camera_cycle_np = active_during_camera_cycle_np[keep_mask_np]
                            inactive_gradient_cycle_count_np = inactive_gradient_cycle_count_np[keep_mask_np]
                            densification_origin_np = densification_origin_np[keep_mask_np]

                        source_index_for_new_np = None
                        if densification_result is not None:
                            new_block = densification_result.get("new", None)
                            if new_block is not None:
                                n_new = int(new_block["position"].shape[0])
                                add_new_points(renderer, densification_result)

                                source_index_for_new_np = np.asarray(
                                    densification_result.get("source_index", np.zeros((0,), dtype=np.int64)),
                                    dtype=np.int64,
                                ).reshape(-1)

                                if source_index_for_new_np.shape[0] != n_new:
                                    source_index_for_new_np = None

                                densify_position_grad_accum_np = np.concatenate(
                                    [densify_position_grad_accum_np, np.zeros((n_new, 1), dtype=np.float32)], axis=0)
                                densify_position_grad_denom_np = np.concatenate(
                                    [densify_position_grad_denom_np, np.zeros((n_new, 1), dtype=np.float32)], axis=0)
                                densify_position_grad_vector_accum_np = np.concatenate(
                                    [densify_position_grad_vector_accum_np, np.zeros((n_new, 3), dtype=np.float32)],
                                    axis=0)
                                active_during_camera_cycle_np = np.concatenate(
                                    [active_during_camera_cycle_np, np.ones((n_new,), dtype=bool), ], axis=0)
                                inactive_gradient_cycle_count_np = np.concatenate(
                                    [inactive_gradient_cycle_count_np, np.zeros((n_new,), dtype=np.uint32), ], axis=0)
                                densification_origin_np = np.concatenate(
                                    [
                                        densification_origin_np,
                                        make_new_densification_origin_np(densification_result, n_new),
                                    ],
                                    axis=0,
                                )

                        rebuild_bvh(renderer)
                        positions, rotations, scales, albedos, opacities, betas, powers = refetch_parameters_as_torch(
                            renderer, device)
                        rotation_delta = torch.nn.Parameter(
                            torch.zeros((positions.shape[0], 3), device=device, dtype=torch.float32))

                        if densify_position_grad_accum_np.shape[0] != positions.shape[0]:
                            raise RuntimeError(
                                "Densification scalar accumulator length mismatch after topology change: "
                                f"{densify_position_grad_accum_np.shape[0]} vs {positions.shape[0]}"
                            )

                        if densify_position_grad_vector_accum_np.shape[0] != positions.shape[0]:
                            raise RuntimeError(
                                "Densification vector accumulator length mismatch after topology change: "
                                f"{densify_position_grad_vector_accum_np.shape[0]} vs {positions.shape[0]}"
                            )

                        if densification_origin_np.shape[0] != positions.shape[0]:
                            raise RuntimeError(
                                "Densification origin length mismatch after topology change: "
                                f"{densification_origin_np.shape[0]} vs {positions.shape[0]}"
                            )

                        trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)
                        verify_parameters_inplane(
                            positions, rotations, scales, albedos, opacities, betas,
                            trainable_surfel_mask=trainable_surfel_mask,
                        )
                        apply_point_parameters(renderer, positions, rotations, scales, albedos, opacities, betas,
                                               powers)
                        rebuild_bvh(renderer)

                        migrated_device_adam_state = migrate_device_adam_state_snapshot(
                            device_adam_state_snapshot,
                            keep_mask_np=keep_mask_np,
                            new_point_count=int(positions.shape[0]),
                            source_index_for_new_np=source_index_for_new_np,
                            copy_source_state_to_new=False,
                        )
                        if migrated_device_adam_state is not None:
                            renderer.upload_device_adam_state(migrated_device_adam_state)

                        optimizer = create_masked_optimizer(
                            config,
                            positions,
                            rotation_delta,
                            scales,
                            albedos,
                            opacities,
                            betas,
                            powers,
                        )
                        active_learning_rates = update_optimizer_learning_rates(
                            optimizer,
                            learning_rate_schedules,
                            global_iteration,
                        )
                        trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)
                        frozen_surfel_count = int((~trainable_surfel_mask).sum().item())
                        #print(f"Frozen emissive surfels: {frozen_surfel_count} / {int(trainable_surfel_mask.numel())}")

                    if camera_cycle_complete:
                        active_during_camera_cycle_np = np.zeros((positions.shape[0],), dtype=bool, )
                        visited_training_camera_ids_this_cycle.clear()

                    if densification_is_due:
                        densify_position_grad_accum_np[:] = 0.0
                        densify_position_grad_denom_np[:] = 0.0
                        densify_position_grad_vector_accum_np[:] = 0.0
                        densification_cycle_start_iteration = global_iteration
                        densification_cycle_interval = densification_interval
                        densification_stats_skip_iterations = densification_stats_skip_for_interval(
                            config=config,
                            densification_interval=densification_cycle_interval,
                        )
                        next_densification_iteration = next_densification_iteration_after(
                            current_iteration=global_iteration,
                            densify_after=densify_after,
                            densification_interval=densification_interval,
                        )

                    grad_position_renderer_norm = 0.0
                    grad_position_renderer_max = 0.0
                    grad_position_surface_regularizer_norm = 0.0
                    grad_position_surface_regularizer_max = 0.0
                    grad_position_total_norm = 0.0
                    grad_position_total_max = 0.0
                    grad_opacity_total_norm = 0.0
                    grad_opacity_total_max = 0.0

                    save_interval = int(config.save_interval)
                    should_save_iteration_outputs = (
                            save_interval > 0 and (
                            global_iteration % save_interval == 0 or iteration == config.iterations)
                    )

                    if should_save_iteration_outputs:
                        save_forward_out = renderer.render_forward()
                        snapshot_adjoint_images = adjoint_images
                        if config.save_snapshot_grad:
                            snapshot_adjoint_images = compute_snapshot_adjoint_images(
                                renderer=renderer,
                                forward_out=save_forward_out,
                                target_images=target_images,
                                camera_ids=training_camera_ids,
                            )

                        save_iteration_outputs(
                            output_dir=config.output_dir,
                            iteration=global_iteration,
                            save_interval=config.save_interval,
                            final_iteration=final_global_iteration,
                            all_camera_ids=all_camera_ids,
                            forward_out=save_forward_out,
                            adjoint_images=snapshot_adjoint_images,
                            renderer_settings=renderer_settings,
                            save_rgb=config.save_snapshot_rgb,
                            save_median_depth=config.save_snapshot_median_depth,
                            save_depth_distortion=config.save_snapshot_depth_distortion,
                            save_visible_normal=config.save_snapshot_visible_normal,
                            save_normal_from_depth=config.save_snapshot_normal_from_depth,
                            save_grad=config.save_snapshot_grad,
                        )

                    iteration_point_cloud_path = None
                    should_extract_mesh_checkpoint = (
                            mesh_extraction_interval > 0
                            and global_iteration % mesh_extraction_interval == 0
                    )
                    needs_parameter_snapshot = (
                            (save_ply_files_interval > 0 and global_iteration % save_ply_files_interval == 0)
                            or should_extract_mesh_checkpoint
                    )
                    if needs_parameter_snapshot:
                        renderer.sync_point_parameters_from_gpu()
                        positions, rotations, scales, albedos, opacities, betas, powers = (
                            refetch_parameters_as_torch(renderer, device)
                        )
                        rotation_delta = torch.nn.Parameter(
                            torch.zeros((positions.shape[0], 3), device=device, dtype=torch.float32)
                        )
                        trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)
                        verify_parameters_inplane(
                            positions,
                            rotations,
                            scales,
                            albedos,
                            opacities,
                            betas,
                            trainable_surfel_mask=trainable_surfel_mask,
                        )

                    if save_ply_files_interval > 0 and global_iteration % save_ply_files_interval == 0:
                        iteration_point_cloud_path = save_iteration_point_cloud_snapshot(
                            config.output_dir,
                            global_iteration,
                            positions,
                            rotations,
                            scales,
                            albedos,
                            opacities,
                            betas,
                            powers,
                        )

                    if should_extract_mesh_checkpoint:
                        if iteration_point_cloud_path is None:
                            iteration_point_cloud_path = save_iteration_point_cloud_snapshot(
                                config.output_dir,
                                global_iteration,
                                positions,
                                rotations,
                                scales,
                                albedos,
                                opacities,
                                betas,
                                powers,
                            )
                        extract_mesh_checkpoint(config, global_iteration, iteration_point_cloud_path)

                    num_points = int(positions.shape[0])
                    iteration_end = time.perf_counter()
                    iteration_time = iteration_end - iteration_start
                    total_time = iteration_end - total_start_time
                    densification_clone_points = (
                        int(densification_result.get("clone_count", 0))
                        if densification_result is not None
                        else 0
                    )
                    densification_split_points = (
                        int(densification_result.get("split_count", 0))
                        if densification_result is not None
                        else 0
                    )
                    densification_new_points = densification_clone_points + densification_split_points
                    densification_clone_points_total += densification_clone_points
                    densification_split_points_total += densification_split_points
                    (
                        densification_clone_points_active,
                        densification_split_points_active,
                    ) = active_densification_origin_counts(densification_origin_np)

                    csv_writer.writerow(
                        [
                            global_iteration,
                            active_camera_name,
                            len(active_training_camera_ids),
                            averaged_loss_state["loss_metric_camera_count"],
                            averaged_loss_state["loss_metric_expected_camera_count"],
                            averaged_loss_state["loss_metric_is_complete"],
                            averaged_loss_state["total_rgb_loss_value"],
                            averaged_loss_state["total_depth_distortion_loss_raw"],
                            averaged_loss_state["total_depth_distortion_loss_weighted"],
                            averaged_loss_state["total_normal_loss_raw"],
                            averaged_loss_state["total_normal_loss_weighted"],
                            averaged_loss_state["total_opacity_prior_loss_raw"],
                            averaged_loss_state["total_opacity_prior_loss_weighted"],
                            averaged_loss_state["total_intra_slab_depth_loss_raw"],
                            averaged_loss_state["total_intra_slab_depth_loss_weighted"],
                            averaged_loss_state["total_curvature_scale_loss_raw"],
                            averaged_loss_state["total_curvature_scale_loss_weighted"],
                            averaged_loss_state["total_loss_value"],
                            num_points,
                            densification_new_points,
                            densification_clone_points,
                            densification_split_points,
                            densification_clone_points_total,
                            densification_split_points_total,
                            densification_clone_points_active,
                            densification_split_points_active,
                            prune_scale_area_points,
                            prune_inactive_gradient_points,
                            iteration_time,
                            total_time,
                            grad_position_renderer_norm,
                            grad_position_renderer_max,
                            grad_position_surface_regularizer_norm,
                            grad_position_surface_regularizer_max,
                            grad_position_total_norm,
                            grad_position_total_max,
                            grad_opacity_total_norm,
                            grad_opacity_total_max,
                        ]
                    )
                    csv_file.flush()

                    log_interval = int(config.log_interval)
                    if iteration == 1 or (log_interval > 0 and iteration % log_interval == 0):
                        logged_iteration_count = iteration - last_log_iteration
                        log_elapsed_time = iteration_end - last_log_time
                        iteration_rate = float(logged_iteration_count) / max(log_elapsed_time, 1.0e-12)
                        last_log_iteration = iteration
                        last_log_time = iteration_end

                        lr_position = active_learning_rates.get("position", float(config.learning_rate_position))
                        exact_clone_scale_threshold = exact_clone_scale_threshold_for_positions(
                            config=config,
                            positions=positions,
                            trainable_surfel_mask=trainable_surfel_mask,
                        )
                        minimum_splittable_scale = minimum_splittable_scale_for_config(config)
                        print(
                            format_training_iteration_log(
                                iteration=global_iteration,
                                total_iterations=final_global_iteration,
                                iteration_time=iteration_time,
                                iteration_rate=iteration_rate,
                                total_time=total_time,
                                num_points=num_points,
                                loss_state=averaged_loss_state,
                                lr_position=lr_position,
                                active_densification_interval=densification_interval,
                                active_prune_interval=prune_interval,
                                active_densification_grad_abs_min=active_densification_grad_abs_min,
                                active_depth_distortion_weight=active_depth_distortion_weight,
                                active_normal_consistency_weight=normal_consistency_weight,
                                active_opacity_prior_weight=active_opacity_prior_weight,
                                exact_clone_scale_threshold=exact_clone_scale_threshold,
                                minimum_splittable_scale=minimum_splittable_scale,
                                grad_pos_rms=0.0,
                                grad_rotation_rms=0.0,
                                grad_scale_rms=0.0,
                                grad_albedo_rms=0.0,
                                grad_opacity_rms=0.0,
                                grad_beta_rms=0.0,
                                grad_pos_max=0.0,
                                grad_rotation_max=0.0,
                                grad_scale_max=0.0,
                                grad_albedo_max=0.0,
                                grad_opacity_max=0.0,
                                grad_beta_max=0.0,
                            )
                        )
                        print(format_loss_breakdown(averaged_loss_state))
                        # print("[device-training-step] Host gradient arrays skipped in fixed-topology path.")

                        hotkey = poll_hotkey()
                        if hotkey == "s":
                            renderer.sync_point_parameters_from_gpu()
                            positions, rotations, scales, albedos, opacities, betas, powers = (
                                refetch_parameters_as_torch(renderer, device)
                            )
                            rotation_delta = torch.nn.Parameter(
                                torch.zeros((positions.shape[0], 3), device=device, dtype=torch.float32)
                            )
                            manual_points_path = save_manual_snapshot(
                                renderer,
                                config.output_dir,
                                global_iteration,
                                positions,
                                rotations,
                                scales,
                                albedos,
                                opacities,
                                betas,
                                powers,
                                training_camera_ids,
                            )
                            extract_mesh_checkpoint(config, global_iteration, manual_points_path)
                        elif hotkey == "g":
                            print(
                                "[device-training-step] Gradient snapshot skipped; host gradients were not downloaded.")

                    continue

                iteration_gradients = compute_iteration_gradients(
                    renderer=renderer,
                    active_training_camera_ids=active_training_camera_ids,
                    latest_loss_values_by_camera=latest_loss_values_by_camera,
                    training_camera_ids=training_camera_ids,
                    iteration=global_iteration,
                    one_camera_per_iteration=config.one_camera_per_iteration,
                    use_depth_distortion_gradients=use_depth_distortion_gradients,
                    use_depth_distortion=use_depth_distortion,
                    use_normal_consistency=use_normal_consistency,
                    use_opacity_prior=use_opacity_prior,
                    use_intra_slab_depth=use_intra_slab_depth,
                    use_curvature_scale=use_curvature_scale,
                    active_depth_distortion_weight=active_depth_distortion_weight,
                    normal_consistency_weight=normal_consistency_weight,
                    active_opacity_prior_weight=active_opacity_prior_weight,
                    intra_slab_depth_weight=intra_slab_depth_weight,
                    curvature_scale_weight=curvature_scale_weight,
                )

                active_camera_name = iteration_gradients.active_camera_name
                forward_out = iteration_gradients.forward_out
                loss_state = iteration_gradients.loss_state
                averaged_loss_state = iteration_gradients.averaged_loss_state
                photo_gradients = iteration_gradients.photo_gradients
                depth_regularizer_gradients = iteration_gradients.depth_regularizer_gradients
                normal_regularizer_gradients = iteration_gradients.normal_regularizer_gradients
                opacity_prior_gradients = iteration_gradients.opacity_prior_gradients
                intra_slab_depth_gradients = iteration_gradients.intra_slab_depth_gradients
                curvature_scale_gradients = iteration_gradients.curvature_scale_gradients
                surface_regularizer_gradients = iteration_gradients.surface_regularizer_gradients
                total_gradients = iteration_gradients.total_gradients
                adjoint_images = iteration_gradients.adjoint_images
                photo_gradient_surfel_stats = iteration_gradients.photo_gradient_surfel_stats

                # Any positive per-camera record means the surfel contributed at least one
                # position-gradient record during this render-backward call.
                active_during_camera_cycle_np |= active_surfel_mask_from_position_records(
                    photo_gradient_surfel_stats,
                    point_count=int(positions.shape[0]),
                )
                visited_training_camera_ids_this_cycle.update(active_training_camera_ids)
                camera_cycle_complete = (len(visited_training_camera_ids_this_cycle) == len(training_camera_ids))

                camera_batch_scale = (
                    float(len(training_camera_ids)) / float(len(active_training_camera_ids))
                    if config.one_camera_per_iteration and config.scale_single_camera_gradients
                    else 1.0
                )

                if camera_batch_scale != 1.0:
                    photo_gradients = scale_gradient_dict(photo_gradients, camera_batch_scale)
                    depth_regularizer_gradients = scale_gradient_dict(
                        depth_regularizer_gradients,
                        camera_batch_scale,
                    )
                    normal_regularizer_gradients = scale_gradient_dict(
                        normal_regularizer_gradients,
                        camera_batch_scale,
                    )
                    opacity_prior_gradients = scale_gradient_dict(
                        opacity_prior_gradients,
                        camera_batch_scale,
                    )
                    intra_slab_depth_gradients = scale_gradient_dict(
                        intra_slab_depth_gradients,
                        camera_batch_scale,
                    )
                    curvature_scale_gradients = scale_gradient_dict(
                        curvature_scale_gradients,
                        camera_batch_scale,
                    )
                    surface_regularizer_gradients = scale_gradient_dict(
                        surface_regularizer_gradients,
                        camera_batch_scale,
                    )
                    total_gradients = sum_gradient_dicts(
                        photo_gradients,
                        surface_regularizer_gradients,
                    )

                (grad_position_np, grad_rotation_np, grad_scales_np, grad_albedos_np, grad_opacities_np,
                 grad_betas_np,) = extract_total_gradient_arrays(total_gradients)

                grad_opacity_total_norm = gradient_l2_norm(grad_opacities_np)
                grad_opacity_total_max = max_point_norm(grad_opacities_np)

                grad_position_renderer_norm, grad_position_renderer_max = position_gradient_norm_stats_or_zero(
                    photo_gradients)

                grad_position_surface_regularizer_norm, grad_position_surface_regularizer_max = position_gradient_norm_stats_or_zero(
                    surface_regularizer_gradients
                )
                grad_position_total_norm = gradient_l2_norm(grad_position_np)
                grad_position_total_max = max_point_norm(grad_position_np)

                if grad_position_np.shape != tuple(positions.shape):
                    raise RuntimeError(
                        f"Gradient shape mismatch for position: expected {tuple(positions.shape)}, got {grad_position_np.shape}")

                clone_signal_per_camera_np = photo_gradient_surfel_stats.get("clone_signal_per_camera", None)
                clone_signal_record_count_per_camera_np = photo_gradient_surfel_stats.get(
                    "clone_signal_record_count_per_camera",
                    None,
                )
                update_densification_statistics(
                    iteration=global_iteration,
                    densification_interval=densification_cycle_interval,
                    densification_cycle_start_iteration=densification_cycle_start_iteration,
                    densification_stats_skip_iterations=densification_stats_skip_iterations,
                    densify_position_grad_accum_np=densify_position_grad_accum_np,
                    densify_position_grad_denom_np=densify_position_grad_denom_np,
                    densify_position_grad_vector_accum_np=densify_position_grad_vector_accum_np,
                    rotations=rotations,
                    albedos=albedos,
                    trainable_surfel_mask=trainable_surfel_mask,
                    densify_bsdf_floor=densify_bsdf_floor,
                    densify_bsdf_gamma=densify_bsdf_gamma,
                    densify_position_grad_per_camera_np=clone_signal_per_camera_np,
                    densify_position_grad_per_camera_count_np=clone_signal_record_count_per_camera_np,
                )

                optimizer.zero_grad(set_to_none=True)

                zero_frozen_surfel_gradients_np(
                    trainable_surfel_mask, grad_position_np, grad_rotation_np,
                    grad_scales_np, grad_albedos_np, grad_opacities_np, grad_betas_np,
                )

                assign_numpy_gradients_to_tensors(
                    device, positions, rotation_delta, scales, albedos, opacities, betas,
                    grad_position_np, grad_rotation_np, grad_scales_np,
                    grad_albedos_np, grad_opacities_np, grad_betas_np,
                )

                active_learning_rates = update_optimizer_learning_rates(
                    optimizer,
                    learning_rate_schedules,
                    global_iteration,
                )
                optimizer.step()
                apply_local_rotation_update_to_quaternions_inplace(
                    rotations,
                    rotation_delta,
                    trainable_surfel_mask=trainable_surfel_mask,
                )

                scheduled_opacity_reset = (
                        reset_opacity_interval > 0
                        and global_iteration % reset_opacity_interval == 0
                )
                manual_opacity_reset = bool(config.reset_opacity_iterations)
                did_reset_opacity = scheduled_opacity_reset or manual_opacity_reset

                if did_reset_opacity:
                    with torch.no_grad():
                        opacities[trainable_surfel_mask] = float(reset_opacity_value)

                    print(f"[Iter {global_iteration:04d}] Resetting all opacities to {reset_opacity_value}")
                    config.reset_opacity_iterations = False

                densification_is_due = (
                        densification_interval > 0
                        and next_densification_iteration is not None
                        and global_iteration >= next_densification_iteration
                )

                verify_parameters_inplane(positions, rotations, scales, albedos, opacities, betas,
                                          trainable_surfel_mask=trainable_surfel_mask)
                apply_point_parameters(renderer, positions, rotations, scales, albedos, opacities, betas, powers)

                if global_iteration % rebuild_bvh_interval == 0:
                    rebuild_bvh(renderer)

                densification_result = None
                scale_prune_indices = []
                opacity_prune_indices = []
                indices_to_remove_list = []
                inactive_camera_cycle_indices = np.zeros((0,), dtype=np.int64)
                prune_scale_area_points = 0
                prune_inactive_gradient_points = 0

                if not did_reset_opacity:
                    if densification_is_due:
                        densification_result = maybe_make_densification_result(
                            iteration=global_iteration, config=config, positions=positions, rotations=rotations,
                            scales=scales, albedos=albedos, opacities=opacities,
                            betas=betas, powers=powers, trainable_surfel_mask=trainable_surfel_mask,
                            densify_position_grad_accum_np=densify_position_grad_accum_np,
                            densify_position_grad_denom_np=densify_position_grad_denom_np,
                            densify_position_grad_vector_accum_np=densify_position_grad_vector_accum_np,
                            densify_after=densify_after,
                            densification_interval=densification_interval, densification_verbose=densification_verbose,
                            densification_grad_quantile=densification_grad_quantile,
                            densification_grad_abs_min=active_densification_grad_abs_min,
                            force_densification=True,
                        )

                    scale_prune_indices, opacity_prune_indices, indices_to_remove_list = maybe_make_prune_indices(
                        iteration=global_iteration, config=config, scales=scales, opacities=opacities,
                        trainable_surfel_mask=trainable_surfel_mask, prune_after=prune_after,
                        prune_interval=prune_interval, reset_opacity_interval=reset_opacity_interval,
                        opacity_prune_threshold=opacity_prune_threshold, max_prune_fraction=max_prune_fraction,
                    )

                    if (
                            camera_cycle_complete
                            and global_iteration >= prune_after
                            and inactive_gradient_prune_cycles > 0
                    ):
                        trainable_surfel_mask_np = (
                            trainable_surfel_mask.detach().cpu().numpy().astype(bool).reshape(-1))
                        active_this_cycle_np = (trainable_surfel_mask_np & active_during_camera_cycle_np)
                        inactive_this_cycle_np = (trainable_surfel_mask_np & ~active_during_camera_cycle_np)
                        # A surfel with any position-gradient record in this
                        # complete camera cycle is no longer considered inactive.
                        inactive_gradient_cycle_count_np[active_this_cycle_np] = 0
                        # Count consecutive complete cycles with no position-gradient
                        # record. Saturate at the threshold; larger values are irrelevant.
                        inactive_gradient_cycle_count_np[inactive_this_cycle_np] = np.minimum(
                            inactive_gradient_cycle_count_np[inactive_this_cycle_np] + 1,
                            inactive_gradient_prune_cycles, )
                        inactive_camera_cycle_indices = np.flatnonzero(trainable_surfel_mask_np & (
                                inactive_gradient_cycle_count_np >= inactive_gradient_prune_cycles)).astype(
                            np.int64)

                        if inactive_camera_cycle_indices.size > 0:
                            indices_to_remove_list.extend(int(index) for index in inactive_camera_cycle_indices)
                else:
                    print(f"[Iter {global_iteration:04d}] Skipping densification/pruning due to opacity reset")

                if indices_to_remove_list or densification_result is not None:
                    old_params_for_optimizer = make_named_parameter_dict(positions, rotation_delta, scales, albedos,
                                                                         opacities, betas, powers, )
                    old_optimizer_for_migration = optimizer
                    old_point_count_for_migration = int(positions.shape[0])
                    keep_mask_np = np.ones(old_point_count_for_migration, dtype=bool)

                    if densification_result is not None:
                        protected_src = np.asarray(
                            densification_result.get("source_index", np.zeros((0,), dtype=np.int64)),
                            dtype=np.int64).reshape(-1)
                        if protected_src.size > 0 and indices_to_remove_list:
                            protected_set = set(int(i) for i in protected_src)
                            indices_to_remove_list = [int(i) for i in indices_to_remove_list if
                                                      int(i) not in protected_set]

                    if densification_result is not None:
                        apply_densification_source_updates_inplace(densification_result, positions, rotations, scales,
                                                                   albedos, opacities, betas, powers, )
                        verify_parameters_inplane(positions, rotations, scales, albedos, opacities, betas,
                                                  trainable_surfel_mask=trainable_surfel_mask, )
                        apply_point_parameters(renderer, positions, rotations, scales, albedos, opacities, betas,
                                               powers, )

                    if indices_to_remove_list:
                        scale_prune_set = set(int(i) for i in scale_prune_indices)
                        opacity_prune_set = set(int(i) for i in opacity_prune_indices)
                        overlap_set = scale_prune_set & opacity_prune_set
                        indices_to_remove = np.unique(np.asarray(indices_to_remove_list, dtype=np.int64))

                        inactive_cycle_prune_set = set(int(index) for index in inactive_camera_cycle_indices)
                        removed_index_set = set(int(index) for index in indices_to_remove)
                        prune_scale_area_points = len(scale_prune_set & removed_index_set)
                        prune_inactive_gradient_points = len(inactive_cycle_prune_set & removed_index_set)
                        if config.densification_verbose:
                            print(
                                f"[Iter {global_iteration:04d}] Pruning {indices_to_remove.size} unique surfels | "
                                f"scale={len(scale_prune_set)}, "
                                f"opacity={len(opacity_prune_set)}, "
                                f"inactive_gradient={len(inactive_cycle_prune_set)} "
                                f"(threshold={inactive_gradient_prune_cycles} cycles), "
                                f"both_scale_opacity={len(overlap_set)}, "
                                f"scale_only={len(scale_prune_set - opacity_prune_set)}, "
                                f"opacity_only={len(opacity_prune_set - scale_prune_set)}"
                            )

                        keep_mask_np[indices_to_remove] = False
                        remove_points(renderer, indices_to_remove)
                        densify_position_grad_accum_np = densify_position_grad_accum_np[keep_mask_np]
                        densify_position_grad_denom_np = densify_position_grad_denom_np[keep_mask_np]
                        densify_position_grad_vector_accum_np = densify_position_grad_vector_accum_np[keep_mask_np]
                        active_during_camera_cycle_np = active_during_camera_cycle_np[keep_mask_np]
                        inactive_gradient_cycle_count_np = (inactive_gradient_cycle_count_np[keep_mask_np])
                        densification_origin_np = densification_origin_np[keep_mask_np]
                    source_index_for_new_np = None

                    if densification_result is not None:
                        new_block = densification_result.get("new", None)
                        if new_block is not None:
                            n_new = int(new_block["position"].shape[0])
                            add_new_points(renderer, densification_result)

                            source_index_for_new_np = np.asarray(
                                densification_result.get("source_index", np.zeros((0,), dtype=np.int64)),
                                dtype=np.int64,
                            ).reshape(-1)

                            if source_index_for_new_np.shape[0] != n_new:
                                source_index_for_new_np = None

                            densify_position_grad_accum_np = np.concatenate(
                                [densify_position_grad_accum_np, np.zeros((n_new, 1), dtype=np.float32)], axis=0)
                            densify_position_grad_denom_np = np.concatenate(
                                [densify_position_grad_denom_np, np.zeros((n_new, 1), dtype=np.float32)], axis=0)
                            densify_position_grad_vector_accum_np = np.concatenate(
                                [densify_position_grad_vector_accum_np, np.zeros((n_new, 3), dtype=np.float32)], axis=0)
                            active_during_camera_cycle_np = np.concatenate(
                                [active_during_camera_cycle_np, np.ones((n_new,), dtype=bool), ], axis=0)
                            inactive_gradient_cycle_count_np = np.concatenate(
                                [inactive_gradient_cycle_count_np, np.zeros((n_new,), dtype=np.uint32), ], axis=0, )
                            densification_origin_np = np.concatenate(
                                [
                                    densification_origin_np,
                                    make_new_densification_origin_np(densification_result, n_new),
                                ],
                                axis=0,
                            )

                    rebuild_bvh(renderer)
                    positions, rotations, scales, albedos, opacities, betas, powers = refetch_parameters_as_torch(
                        renderer, device)
                    rotation_delta = torch.nn.Parameter(
                        torch.zeros((positions.shape[0], 3), device=device, dtype=torch.float32))

                    if densify_position_grad_accum_np.shape[0] != positions.shape[0]:
                        raise RuntimeError(
                            "Densification scalar accumulator length mismatch after topology change: "
                            f"{densify_position_grad_accum_np.shape[0]} vs {positions.shape[0]}"
                        )

                    if densify_position_grad_vector_accum_np.shape[0] != positions.shape[0]:
                        raise RuntimeError(
                            "Densification vector accumulator length mismatch after topology change: "
                            f"{densify_position_grad_vector_accum_np.shape[0]} vs {positions.shape[0]}"
                        )

                    if densification_origin_np.shape[0] != positions.shape[0]:
                        raise RuntimeError(
                            "Densification origin length mismatch after topology change: "
                            f"{densification_origin_np.shape[0]} vs {positions.shape[0]}"
                        )

                    trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)
                    verify_parameters_inplane(positions, rotations, scales, albedos, opacities, betas,
                                              trainable_surfel_mask=trainable_surfel_mask, )

                    apply_point_parameters(renderer, positions, rotations, scales, albedos, opacities, betas, powers, )
                    rebuild_bvh(renderer)

                    new_params_for_optimizer = make_named_parameter_dict(positions, rotation_delta, scales, albedos,
                                                                         opacities, betas, powers, )
                    optimizer = rebuild_optimizer_preserving_state(
                        config=config, old_optimizer=old_optimizer_for_migration,
                        old_params=old_params_for_optimizer, new_params=new_params_for_optimizer,
                        keep_mask_np=keep_mask_np, source_index_for_new_np=source_index_for_new_np,
                        copy_source_state_to_new=False,
                    )

                    active_learning_rates = update_optimizer_learning_rates(
                        optimizer,
                        learning_rate_schedules,
                        global_iteration,
                    )
                    trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)
                    frozen_surfel_count = int((~trainable_surfel_mask).sum().item())
                    #print(f"Frozen emissive surfels: {frozen_surfel_count} / {int(trainable_surfel_mask.numel())}")

                if camera_cycle_complete:
                    active_during_camera_cycle_np = np.zeros((positions.shape[0],), dtype=bool, )
                    visited_training_camera_ids_this_cycle.clear()

                if densification_is_due:
                    densify_position_grad_accum_np[:] = 0.0
                    densify_position_grad_denom_np[:] = 0.0
                    densify_position_grad_vector_accum_np[:] = 0.0
                    densification_cycle_start_iteration = global_iteration
                    densification_cycle_interval = densification_interval
                    densification_stats_skip_iterations = densification_stats_skip_for_interval(
                        config=config,
                        densification_interval=densification_cycle_interval,
                    )
                    next_densification_iteration = next_densification_iteration_after(
                        current_iteration=global_iteration,
                        densify_after=densify_after,
                        densification_interval=densification_interval,
                    )

                save_interval = int(config.save_interval)
                should_save_iteration_outputs = (
                        save_interval > 0
                        and (global_iteration % save_interval == 0 or iteration == config.iterations)
                )

                if should_save_iteration_outputs:
                    if forward_out is None or config.one_camera_per_iteration or config.save_snapshot_grad:
                        save_forward_out = renderer.render_forward()
                    else:
                        save_forward_out = forward_out

                    snapshot_adjoint_images = adjoint_images
                    if config.save_snapshot_grad:
                        snapshot_adjoint_images = compute_snapshot_adjoint_images(
                            renderer=renderer,
                            forward_out=save_forward_out,
                            target_images=target_images,
                            camera_ids=training_camera_ids,
                        )

                    save_iteration_outputs(
                        output_dir=config.output_dir,
                        iteration=global_iteration,
                        save_interval=config.save_interval,
                        final_iteration=final_global_iteration,
                        all_camera_ids=all_camera_ids,
                        forward_out=save_forward_out,
                        adjoint_images=snapshot_adjoint_images,
                        renderer_settings=renderer_settings,
                        save_rgb=config.save_snapshot_rgb,
                        save_median_depth=config.save_snapshot_median_depth,
                        save_depth_distortion=config.save_snapshot_depth_distortion,
                        save_visible_normal=config.save_snapshot_visible_normal,
                        save_normal_from_depth=config.save_snapshot_normal_from_depth,
                        save_grad=config.save_snapshot_grad,
                    )

                iteration_point_cloud_path = None
                should_extract_mesh_checkpoint = (
                        mesh_extraction_interval > 0
                        and global_iteration % mesh_extraction_interval == 0
                )

                if save_ply_files_interval > 0 and global_iteration % save_ply_files_interval == 0:
                    iteration_point_cloud_path = save_iteration_point_cloud_snapshot(
                        config.output_dir,
                        global_iteration,
                        positions,
                        rotations,
                        scales,
                        albedos,
                        opacities,
                        betas,
                        powers,
                    )

                if should_extract_mesh_checkpoint:
                    if iteration_point_cloud_path is None:
                        iteration_point_cloud_path = save_iteration_point_cloud_snapshot(
                            config.output_dir,
                            global_iteration,
                            positions,
                            rotations,
                            scales,
                            albedos,
                            opacities,
                            betas,
                            powers,
                        )

                    extract_mesh_checkpoint(config, global_iteration, iteration_point_cloud_path)

                num_points = positions.shape[0]
                iteration_end = time.perf_counter()
                iteration_time = iteration_end - iteration_start
                total_time = iteration_end - total_start_time
                densification_clone_points = (
                    int(densification_result.get("clone_count", 0))
                    if densification_result is not None
                    else 0
                )
                densification_split_points = (
                    int(densification_result.get("split_count", 0))
                    if densification_result is not None
                    else 0
                )
                densification_new_points = densification_clone_points + densification_split_points
                densification_clone_points_total += densification_clone_points
                densification_split_points_total += densification_split_points
                (
                    densification_clone_points_active,
                    densification_split_points_active,
                ) = active_densification_origin_counts(densification_origin_np)

                csv_writer.writerow(
                    [
                        global_iteration,
                        active_camera_name,
                        len(active_training_camera_ids),
                        averaged_loss_state["loss_metric_camera_count"],
                        averaged_loss_state["loss_metric_expected_camera_count"],
                        averaged_loss_state["loss_metric_is_complete"],
                        averaged_loss_state["total_rgb_loss_value"],
                        averaged_loss_state["total_depth_distortion_loss_raw"],
                        averaged_loss_state["total_depth_distortion_loss_weighted"],
                        averaged_loss_state["total_normal_loss_raw"],
                        averaged_loss_state["total_normal_loss_weighted"],
                        averaged_loss_state["total_opacity_prior_loss_raw"],
                        averaged_loss_state["total_opacity_prior_loss_weighted"],
                        averaged_loss_state["total_intra_slab_depth_loss_raw"],
                        averaged_loss_state["total_intra_slab_depth_loss_weighted"],
                        averaged_loss_state["total_curvature_scale_loss_raw"],
                        averaged_loss_state["total_curvature_scale_loss_weighted"],
                        averaged_loss_state["total_loss_value"],
                        num_points,
                        densification_new_points,
                        densification_clone_points,
                        densification_split_points,
                        densification_clone_points_total,
                        densification_split_points_total,
                        densification_clone_points_active,
                        densification_split_points_active,
                        prune_scale_area_points,
                        prune_inactive_gradient_points,
                        iteration_time,
                        total_time,
                        grad_position_renderer_norm,
                        grad_position_renderer_max,
                        grad_position_surface_regularizer_norm,
                        grad_position_surface_regularizer_max,
                        grad_position_total_norm,
                        grad_position_total_max,
                        grad_opacity_total_norm,
                        grad_opacity_total_max,
                    ]
                )

                csv_file.flush()

                log_interval = int(config.log_interval)
                if iteration == 1 or (log_interval > 0 and iteration % log_interval == 0):
                    logged_iteration_count = iteration - last_log_iteration
                    log_elapsed_time = iteration_end - last_log_time
                    iteration_rate = float(logged_iteration_count) / max(log_elapsed_time, 1.0e-12)
                    last_log_iteration = iteration
                    last_log_time = iteration_end

                    grad_pos_rms = rms_point(grad_position_np)
                    grad_rotation_rms = rms_point(grad_rotation_np)
                    grad_scale_rms = rms_point(grad_scales_np)
                    grad_albedo_rms = rms_point(grad_albedos_np)
                    grad_opacity_rms = rms_point(grad_opacities_np)
                    grad_beta_rms = rms_point(grad_betas_np)

                    grad_pos_max = max_point_norm(grad_position_np)
                    grad_rotation_max = max_point_norm(grad_rotation_np)
                    grad_scale_max = max_point_norm(grad_scales_np)
                    grad_albedo_max = max_point_norm(grad_albedos_np)
                    grad_opacity_max = max_point_norm(grad_opacities_np)
                    grad_beta_max = max_point_norm(grad_betas_np)
                    lr_position = active_learning_rates.get("position", float(config.learning_rate_position))
                    photo_gradient_stats = gradient_stats_from_dict(photo_gradients)
                    surface_regularizer_gradient_stats = (
                        gradient_stats_from_dict(surface_regularizer_gradients)
                        if surface_regularizer_gradients
                        else {}
                    )

                    exact_clone_scale_threshold = exact_clone_scale_threshold_for_positions(
                        config=config,
                        positions=positions,
                        trainable_surfel_mask=trainable_surfel_mask,
                    )
                    minimum_splittable_scale = minimum_splittable_scale_for_config(config)
                    print(
                        format_training_iteration_log(
                            iteration=global_iteration,
                            total_iterations=final_global_iteration,
                            iteration_time=iteration_time,
                            iteration_rate=iteration_rate,
                            total_time=total_time,
                            num_points=num_points,
                            loss_state=averaged_loss_state,
                            lr_position=lr_position,
                            active_densification_interval=densification_interval,
                            active_prune_interval=prune_interval,
                            active_densification_grad_abs_min=active_densification_grad_abs_min,
                            active_depth_distortion_weight=active_depth_distortion_weight,
                            active_normal_consistency_weight=normal_consistency_weight,
                            active_opacity_prior_weight=active_opacity_prior_weight,
                            exact_clone_scale_threshold=exact_clone_scale_threshold,
                            minimum_splittable_scale=minimum_splittable_scale,
                            grad_pos_rms=grad_pos_rms,
                            grad_rotation_rms=grad_rotation_rms,
                            grad_scale_rms=grad_scale_rms,
                            grad_albedo_rms=grad_albedo_rms,
                            grad_opacity_rms=grad_opacity_rms,
                            grad_beta_rms=grad_beta_rms,
                            grad_pos_max=grad_pos_max,
                            grad_rotation_max=grad_rotation_max,
                            grad_scale_max=grad_scale_max,
                            grad_albedo_max=grad_albedo_max,
                            grad_opacity_max=grad_opacity_max,
                            grad_beta_max=grad_beta_max,
                        )
                    )

                    print(format_loss_breakdown(averaged_loss_state))

                    print(
                        format_gradient_source_balance(
                            loss_gradients=photo_gradients,
                            depth_regularizer_gradients=depth_regularizer_gradients,
                            normal_regularizer_gradients=normal_regularizer_gradients,
                            opacity_prior_gradients=opacity_prior_gradients,
                            intra_slab_depth_gradients=intra_slab_depth_gradients,
                            curvature_scale_gradients=curvature_scale_gradients,
                            surface_regularizer_gradients=surface_regularizer_gradients,
                            total_gradients=total_gradients,
                        )
                    )
                    print(format_gradient_stats("render_grads", photo_gradient_stats))

                    if surface_regularizer_gradients:
                        print(format_gradient_stats("surface_regularizers", surface_regularizer_gradient_stats))
                    if use_depth_distortion and loss_state["depth_distortion_maps_for_logging"]:
                        print(summarize_depth_distortion_maps(loss_state["depth_distortion_maps_for_logging"],
                                                              loss_state["depth_distortion_grad_images"]))

                    hotkey = poll_hotkey()
                    if hotkey == "s":
                        manual_points_path = save_manual_snapshot(
                            renderer,
                            config.output_dir,
                            global_iteration,
                            positions,
                            rotations,
                            scales,
                            albedos,
                            opacities,
                            betas,
                            powers,
                            training_camera_ids,
                        )
                        extract_mesh_checkpoint(config, global_iteration, manual_points_path)
                    elif hotkey == "g":
                        save_gradients_snapshot(config.output_dir, global_iteration, grad_position_np, grad_rotation_np,
                                                grad_scales_np, grad_albedos_np, grad_opacities_np,
                                                grad_betas_np)

        except KeyboardInterrupt:
            elapsed = time.perf_counter() - total_start_time
            stopped_global_iteration = resume_iteration_offset + int(iteration)
            print(
                f"\nCtrl+C detected at iteration {stopped_global_iteration:04d}. "
                f"Total elapsed time: {elapsed:.1f} s. "
                "Stopping optimization loop and saving current result..."
            )

    if use_device_training_step:
        renderer.sync_point_parameters_from_gpu()
        positions, rotations, scales, albedos, opacities, betas, powers = refetch_parameters_as_torch(renderer, device)
        rotation_delta = torch.nn.Parameter(torch.zeros((positions.shape[0], 3), device=device, dtype=torch.float32))
        trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)
    else:
        apply_point_parameters(renderer, positions, rotations, scales, albedos, opacities, betas, powers)
    final_images = renderer.render_forward()

    final_rgb_loss = 0.0
    final_depth_distortion_loss_raw = 0.0
    final_depth_distortion_loss_weighted = 0.0
    final_normal_loss_raw = 0.0
    final_normal_loss_weighted = 0.0
    final_opacity_prior_loss_raw = 0.0
    final_opacity_prior_loss_weighted = 0.0
    final_intra_slab_depth_loss_raw = 0.0
    final_intra_slab_depth_loss_weighted = 0.0
    final_curvature_scale_loss_raw = 0.0
    final_curvature_scale_loss_weighted = 0.0
    final_total_loss = 0.0

    final_depth_distortion_weight = scheduled_regularizer_weight(
        depth_distortion_base_weight,
        iteration=resume_iteration_offset + int(iteration),
        start_iteration=depth_distortion_start_iteration,
    )

    for camera_name in training_camera_ids:
        img_np = get_forward_rgb(final_images, camera_name)
        tgt_np = target_images[camera_name]
        rgb_loss_cam = float(compute_l2_loss(img_np, tgt_np))
        final_rgb_loss += rgb_loss_cam
        final_total_loss += rgb_loss_cam
        save_render(config.output_dir / f"render_final_{camera_name}.png", img_np)

        if use_depth_distortion:
            dist_np = get_forward_depth_distortion(final_images, camera_name)
            dist_loss_cam_raw = float(dist_np.mean())
            dist_loss_cam_weighted = final_depth_distortion_weight * dist_loss_cam_raw
            final_depth_distortion_loss_raw += dist_loss_cam_raw
            final_depth_distortion_loss_weighted += dist_loss_cam_weighted
            final_total_loss += dist_loss_cam_weighted
            save_depth_distortion_snapshot(config.output_dir / f"depth_distortion_final_{camera_name}.png", dist_np,
                                           quantile=0.99, save_npy=False)

        if use_normal_consistency:
            median_depth = get_forward_median_depth(final_images, camera_name)
            visible_normal = get_forward_visible_normal(final_images, camera_name)
            normal_from_depth = get_forward_normal_from_depth(final_images, camera_name)
            normal_loss_cam_raw, _, _ = compute_normal_consistency_loss_and_adjoints(visible_normal, normal_from_depth,
                                                                                     1.0)
            normal_loss_cam_weighted = normal_consistency_weight * normal_loss_cam_raw

            final_normal_loss_raw += normal_loss_cam_raw
            final_normal_loss_weighted += normal_loss_cam_weighted
            final_total_loss += normal_loss_cam_weighted

            save_median_depth_snapshot(config.output_dir / f"median_depth_final_{camera_name}.png", median_depth,
                                       save_npy=False)
            save_normal_map_snapshot(config.output_dir / f"visible_normal_final_{camera_name}.png", visible_normal,
                                     save_npy=False)
            save_normal_map_snapshot(config.output_dir / f"normal_from_depth_final_{camera_name}.png",
                                     normal_from_depth, save_npy=False)

        if use_opacity_prior:
            opacity_prior_cam_raw = float(get_forward_opacity_prior(final_images, camera_name).mean())
            opacity_prior_cam_weighted = opacity_prior_weight * opacity_prior_cam_raw

            final_opacity_prior_loss_raw += opacity_prior_cam_raw
            final_opacity_prior_loss_weighted += opacity_prior_cam_weighted
            final_total_loss += opacity_prior_cam_weighted

        if use_intra_slab_depth:
            intra_slab_depth_map = get_forward_intra_slab_depth(final_images, camera_name)
            intra_slab_active_count = max(
                1,
                int(get_forward_intra_slab_depth_active_slab_count(
                    final_images,
                    camera_name,
                ).sum(dtype=np.uint64)),
            )
            intra_slab_depth_cam_raw = float(
                intra_slab_depth_map.sum() / intra_slab_active_count
            )
            intra_slab_depth_cam_weighted = (
                intra_slab_depth_weight * intra_slab_depth_cam_raw
            )
            final_intra_slab_depth_loss_raw += intra_slab_depth_cam_raw
            final_intra_slab_depth_loss_weighted += intra_slab_depth_cam_weighted
            final_total_loss += intra_slab_depth_cam_weighted

        if use_curvature_scale:
            curvature_scale_map = get_forward_curvature_scale(final_images, camera_name)
            curvature_scale_active_count = max(
                1,
                int(get_forward_curvature_scale_active_slab_count(
                    final_images,
                    camera_name,
                ).sum(dtype=np.uint64)),
            )
            curvature_scale_cam_raw = float(
                curvature_scale_map.sum() / curvature_scale_active_count
            )
            curvature_scale_cam_weighted = (
                curvature_scale_weight * curvature_scale_cam_raw
            )
            final_curvature_scale_loss_raw += curvature_scale_cam_raw
            final_curvature_scale_loss_weighted += curvature_scale_cam_weighted
            final_total_loss += curvature_scale_cam_weighted

    print_loss_summary("Initial", *initial_loss_tuple)
    print_loss_summary(
        "Final",
        final_rgb_loss,
        final_depth_distortion_loss_raw,
        final_depth_distortion_loss_weighted,
        final_normal_loss_raw,
        final_normal_loss_weighted,
        final_opacity_prior_loss_raw,
        final_opacity_prior_loss_weighted,
        final_intra_slab_depth_loss_raw,
        final_intra_slab_depth_loss_weighted,
        final_curvature_scale_loss_raw,
        final_curvature_scale_loss_weighted,
        final_total_loss,
    )
    ply_path = config.output_dir / "points_final.ply"
    save_gaussians_to_ply(ply_path, positions, rotations, scales, albedos, opacities, betas, powers,
                          shape_default=0.0)

    print(f"Final parameters written to PLY: {ply_path}")
    if bool(getattr(config, "save_final_mesh", True)):
        extract_final_mesh(config, ply_path)

    print("\nOptimization completed.")
    print(f"Outputs saved in: {config.output_dir.resolve()}")
    print(f"Total optimization wall time: {time.perf_counter() - total_start_time:.1f} s")
