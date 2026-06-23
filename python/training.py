from __future__ import annotations

import time
from optimizers import (create_learning_rate_schedules, update_optimizer_learning_rates, )
from training_helpers import *
from density_control import *


def run_optimization(renderer: pale.Renderer, config: OptimizationConfig,
                     renderer_settings: RendererSettingsConfig) -> None:
    target_images, training_camera_ids, all_camera_ids = load_target_images(renderer, Path(config.dataset_path))

    depth_distortion_base_weight = float(getattr(config, "depth_distort_weight", 0.0))
    depth_distortion_start_iteration = int(getattr(config, "depth_distort_start_iteration", 0))

    normal_consistency_weight = float(getattr(config, "normal_consistency_weight", 0.0))
    visibility_weighted_opacity_weight = float(config.visibility_weighted_opacity_weight)
    bsdf_decay_weight = float(config.bsdf_decay_weight)
    use_bsdf_decay = bsdf_decay_weight != 0.0
    save_ply_files_interval = float(config.save_ply_files_interval)

    use_depth_distortion = depth_distortion_base_weight != 0.0
    use_normal_consistency = normal_consistency_weight != 0.0
    use_visibility_weighted_opacity = visibility_weighted_opacity_weight != 0.0

    print(
        "Loss terms: "
        f"depth_distortion={use_depth_distortion} "
        f"base_weight={depth_distortion_base_weight:.3e} "
        f"start_iter={depth_distortion_start_iteration}, "
        f"normal_consistency={use_normal_consistency} weight={normal_consistency_weight:.3e}, "
        f"visibility_weighted_opacity={use_visibility_weighted_opacity} "
        f"weight={visibility_weighted_opacity_weight:.3e}, "
        f"bsdf_decay={use_bsdf_decay} "
        f"weight={bsdf_decay_weight:.3e}"
    )
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
    active_learning_rates = update_optimizer_learning_rates(optimizer, learning_rate_schedules, 0)

    initial_images = renderer.render_forward()
    initial_depth_distortion_weight = scheduled_regularizer_weight(
        depth_distortion_base_weight,
        iteration=0,
        start_iteration=depth_distortion_start_iteration,
    )

    initial_loss_tuple = compute_initial_losses_and_save_outputs(
        output_dir=config.output_dir, initial_images=initial_images, target_images=target_images,
        all_camera_ids=all_camera_ids, positions=positions, rotations=rotations,
        scales=scales, albedos=albedos, opacities=opacities, betas=betas, powers=powers,
        depth_distortion_weight=initial_depth_distortion_weight,
        normal_consistency_weight=normal_consistency_weight,
        visibility_weighted_opacity_weight=visibility_weighted_opacity_weight,
        use_depth_distortion=use_depth_distortion,
        use_normal_consistency=use_normal_consistency,
        use_visibility_weighted_opacity=use_visibility_weighted_opacity,
    )

    print_loss_summary("Initial", *initial_loss_tuple)

    densification_interval = int(config.densification_interval)
    prune_interval = int(config.prune_interval)
    densification_stats_warmup_iterations = max(densification_interval // 2, 0)
    densify_after = config.densify_after if config.densify_after >= 0 else densification_interval
    prune_after = config.prune_after if config.prune_after >= 0 else prune_interval
    densify_until_iteration = int(config.densify_until_iteration) if config.densify_until_iteration >= 0 else int(
        config.densify_until_fraction * config.iterations)

    opacity_prune_threshold = float(config.opacity_prune_threshold)
    max_prune_fraction = float(config.max_prune_fraction)
    inactive_gradient_prune_cycles = int(config.inactive_gradient_prune_cycles)
    reset_opacity_interval = int(config.reset_opacity_interval)
    reset_opacity_value = float(config.reset_opacity_value)
    reset_scale_interval = int(config.reset_scale_interval)
    reset_scale_shrink_factor = float(config.reset_scale_shrink_factor)
    densification_verbose = bool(config.densification_verbose)
    densification_grad_quantile = as_config_float(config.densification_grad_quantile)
    densification_grad_abs_min = float(config.densification_grad_abs_min)
    densify_bsdf_floor = float(config.densify_bsdf_floor)
    densify_bsdf_gamma = float(config.densify_bsdf_gamma)
    rebuild_bvh_interval = max(int(config.rebuild_bvh_interval), 1)
    save_gradient_diagnostics = bool(getattr(config, "save_gradient_diagnostics", False))

    densify_position_grad_accum_np = np.zeros((positions.shape[0], 1), dtype=np.float32)
    densify_position_grad_denom_np = np.zeros((positions.shape[0], 1), dtype=np.float32)
    densify_position_grad_vector_accum_np = np.zeros(tuple(positions.shape), dtype=np.float32)
    active_during_camera_cycle_np = np.zeros((positions.shape[0],), dtype=bool)
    inactive_gradient_cycle_count_np = np.zeros((positions.shape[0],), dtype=np.uint32, )
    visited_training_camera_ids_this_cycle: set[str] = set()

    metrics_csv_path = config.output_dir / "metrics.csv"
    total_start_time = time.perf_counter()
    iteration = 0
    latest_loss_values_by_camera: Dict[str, Dict[str, float]] = {}

    with open(metrics_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        write_metrics_header(csv_writer)

        try:
            for iteration in range(1, config.iterations + 1):
                iteration_start = time.perf_counter()

                active_training_camera_ids = select_active_training_camera_ids(
                    training_camera_ids=training_camera_ids,
                    iteration=iteration,
                    config=config,
                )

                if config.one_camera_per_iteration:
                    active_camera_name = active_training_camera_ids[0]
                    forward_out = renderer.render_forward(active_camera_name)
                else:
                    active_camera_name = "ALL_CAMERAS"
                    forward_out = renderer.render_forward()

                active_depth_distortion_weight = scheduled_regularizer_weight(
                    depth_distortion_base_weight,
                    iteration=iteration,
                    start_iteration=depth_distortion_start_iteration,
                )

                active_densification_grad_abs_min = scheduled_densification_grad_abs_min(config, iteration)
                use_depth_distortion_gradients = active_depth_distortion_weight != 0.0

                loss_state = compute_iteration_losses_and_adjoints(
                    forward_out=forward_out,
                    target_images=target_images,
                    training_camera_ids=active_training_camera_ids,
                    depth_distortion_weight=active_depth_distortion_weight,
                    normal_consistency_weight=normal_consistency_weight,
                    visibility_weighted_opacity_weight=visibility_weighted_opacity_weight,
                    use_depth_distortion=use_depth_distortion,
                    use_normal_consistency=use_normal_consistency,
                    use_visibility_weighted_opacity=use_visibility_weighted_opacity,
                )
                (
                    bsdf_decay_loss_raw,
                    bsdf_decay_loss_weighted,
                    bsdf_decay_grad_albedos_np,
                ) = compute_global_bsdf_decay_loss_and_gradient(
                    albedos=albedos,
                    trainable_surfel_mask=trainable_surfel_mask,
                    weight=bsdf_decay_weight,
                )

                add_global_bsdf_decay_loss_to_loss_state(
                    loss_state=loss_state,
                    raw_loss=bsdf_decay_loss_raw,
                    weighted_loss=bsdf_decay_loss_weighted,
                )

                for camera_name, camera_loss_values in loss_state["per_camera_loss_values"].items():
                    latest_loss_values_by_camera[camera_name] = dict(camera_loss_values)

                averaged_loss_state = make_averaged_loss_state_from_camera_cache(
                    latest_loss_values_by_camera=latest_loss_values_by_camera,
                    expected_camera_ids=training_camera_ids,
                )

                photo_gradients, adjoint_images = renderer.render_backward(loss_state["loss_grad_images"])
                photo_gradient_surfel_stats = adjoint_images.get("gradient_stats", {})

                active_camera_count_np = photo_gradient_surfel_stats.get(
                    "position_active_camera_count", None, )
                if active_camera_count_np is None:
                    raise RuntimeError(
                        "Inactive-surfel pruning requires "                        "adjoint_images['gradient_stats']['position_active_camera_count'].")
                active_camera_count_np = np.asarray(active_camera_count_np, dtype=np.uint32, ).reshape(-1)

                if active_camera_count_np.shape[0] != positions.shape[0]:
                    raise RuntimeError(
                        "Active-camera-count shape mismatch: "
                        f"expected {positions.shape[0]}, got {active_camera_count_np.shape[0]}"
                    )

                # `activeCameraCount > 0` means the surfel contributed at least one
                # position-gradient record during this render-backward call.
                active_during_camera_cycle_np |= active_camera_count_np > 0
                visited_training_camera_ids_this_cycle.update(active_training_camera_ids)
                camera_cycle_complete = (len(visited_training_camera_ids_this_cycle) == len(training_camera_ids))
                surface_regularizer_gradients: Dict[str, np.ndarray] = {}
                use_surface_regularizers = (
                        use_depth_distortion_gradients
                        or use_normal_consistency
                        or use_visibility_weighted_opacity
                )

                depth_regularizer_gradients: Dict[str, np.ndarray] = {}
                normal_regularizer_gradients: Dict[str, np.ndarray] = {}
                visibility_opacity_gradients: Dict[str, np.ndarray] = {}
                surface_regularizer_gradients: Dict[str, np.ndarray] = {}

                if use_surface_regularizers:
                    surface_regularizer_components = renderer.render_surface_regularizers_backward(
                        active_training_camera_ids,
                        loss_state["depth_distortion_grad_images"],
                        loss_state["visible_normal_adjoints"],
                        loss_state["depth_normal_adjoints"],
                    )

                    depth_regularizer_gradients = surface_regularizer_components["depth_distortion"]
                    normal_regularizer_gradients = surface_regularizer_components["normal_consistency"]
                    visibility_opacity_gradients = surface_regularizer_components["visibility_weighted_opacity"]

                    repair_nonfinite_gradient_dict_inplace("depth_regularizer_gradients", depth_regularizer_gradients,
                                                           iteration)
                    repair_nonfinite_gradient_dict_inplace("normal_regularizer_gradients", normal_regularizer_gradients,
                                                           iteration)
                    repair_nonfinite_gradient_dict_inplace("visibility_opacity_gradients", visibility_opacity_gradients,
                                                           iteration)
                    surface_regularizer_gradients = sum_gradient_dicts(depth_regularizer_gradients,
                                                                       normal_regularizer_gradients,
                                                                       visibility_opacity_gradients)

                photo_gradient_stats = gradient_stats_from_dict(photo_gradients)
                surface_regularizer_gradient_stats = (
                    gradient_stats_from_dict(surface_regularizer_gradients)
                    if surface_regularizer_gradients
                    else {}
                )

                repair_nonfinite_gradient_dict_inplace("photo_gradients", photo_gradients, iteration)
                repair_nonfinite_gradient_dict_inplace(
                    "surface_regularizer_gradients",
                    surface_regularizer_gradients,
                    iteration,
                )

                camera_batch_scale = (
                    float(len(training_camera_ids)) / float(len(active_training_camera_ids))
                    if config.one_camera_per_iteration and config.scale_single_camera_gradients
                    else 1.0
                )

                if camera_batch_scale != 1.0:
                    photo_gradients = scale_gradient_dict(photo_gradients, camera_batch_scale)
                    surface_regularizer_gradients = scale_gradient_dict(
                        surface_regularizer_gradients,
                        camera_batch_scale,
                    )

                bsdf_decay_gradients = make_albedo_only_gradient_dict(
                    reference_gradients=photo_gradients,
                    albedo_gradient=bsdf_decay_grad_albedos_np,
                )

                total_gradients = sum_gradient_dicts(
                    photo_gradients,
                    surface_regularizer_gradients,
                    bsdf_decay_gradients,
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

                photo_gradient_surfel_stats = adjoint_images.get("gradient_stats", {})
                position_per_camera_np = photo_gradient_surfel_stats.get("position_per_camera", None)
                position_record_count_per_camera_np = photo_gradient_surfel_stats.get(
                    "position_record_count_per_camera",
                    None,
                )
                if iteration == 1 or iteration % densification_interval == 0:
                    print(
                        "[densify-debug] "
                        f"gradient_stats_keys={list(photo_gradient_surfel_stats.keys())} "
                        f"position_per_camera_shape={None if position_per_camera_np is None else np.asarray(position_per_camera_np).shape} "
                        f"count_shape={None if position_record_count_per_camera_np is None else np.asarray(position_record_count_per_camera_np).shape}"
                        f"grad_abs_min={active_densification_grad_abs_min:.3e}"

                    )
                bsdf_decay_gradient_stats = gradient_stats_from_dict(bsdf_decay_gradients)

                update_densification_statistics(
                    iteration=iteration,
                    densification_interval=densification_interval,
                    densification_stats_warmup_iterations=densification_stats_warmup_iterations,
                    densify_position_grad_accum_np=densify_position_grad_accum_np,
                    densify_position_grad_denom_np=densify_position_grad_denom_np,
                    densify_position_grad_vector_accum_np=densify_position_grad_vector_accum_np,
                    rotations=rotations,
                    albedos=albedos,
                    trainable_surfel_mask=trainable_surfel_mask,
                    densify_bsdf_floor=densify_bsdf_floor,
                    densify_bsdf_gamma=densify_bsdf_gamma,
                    densify_position_grad_per_camera_np=position_per_camera_np,
                    densify_position_grad_per_camera_count_np=position_record_count_per_camera_np,
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

                active_learning_rates = update_optimizer_learning_rates(optimizer, learning_rate_schedules, iteration)
                optimizer.step()
                apply_local_rotation_update_to_quaternions_inplace(
                    rotations,
                    rotation_delta,
                    trainable_surfel_mask=trainable_surfel_mask,
                    max_rotation_step_radians=float(config.max_rotation_step_radians),
                )

                if (
                        reset_scale_interval > 0 and iteration % reset_scale_interval == 0) or config.reset_scale_iterations:
                    with torch.no_grad():
                        scales[trainable_surfel_mask] *= reset_scale_shrink_factor
                    print(
                        f"[Iter {iteration:04d}] Shrinking trainable surfel scales by factor "
                        f"{reset_scale_shrink_factor:.3g}"
                    )
                    config.reset_scale_iterations = False

                scheduled_opacity_reset = reset_opacity_interval > 0 and iteration % reset_opacity_interval == 0
                manual_opacity_reset = bool(config.reset_opacity_iterations)
                did_reset_opacity = scheduled_opacity_reset or manual_opacity_reset

                if did_reset_opacity:
                    with torch.no_grad():
                        opacities[trainable_surfel_mask] = float(reset_opacity_value)

                    print(f"[Iter {iteration:04d}] Resetting all opacities to {reset_opacity_value}")
                    config.reset_opacity_iterations = False

                verify_parameters_inplane(positions, rotations, scales, albedos, opacities, betas,
                                          trainable_surfel_mask=trainable_surfel_mask)
                apply_point_parameters(renderer, positions, rotations, scales, albedos, opacities, betas,
                                       powers)

                if iteration % rebuild_bvh_interval == 0:
                    rebuild_bvh(renderer)

                densification_result = None
                scale_prune_indices = []
                opacity_prune_indices = []
                indices_to_remove_list = []

                if not did_reset_opacity:
                    densification_result = maybe_make_densification_result(
                        iteration=iteration, config=config, positions=positions, rotations=rotations, scales=scales,
                        albedos=albedos, opacities=opacities,
                        betas=betas, powers=powers, trainable_surfel_mask=trainable_surfel_mask,
                        densify_position_grad_accum_np=densify_position_grad_accum_np,
                        densify_position_grad_denom_np=densify_position_grad_denom_np,
                        densify_position_grad_vector_accum_np=densify_position_grad_vector_accum_np,
                        densify_after=densify_after, densify_until_iteration=densify_until_iteration,
                        densification_interval=densification_interval, densification_verbose=densification_verbose,
                        densification_grad_quantile=densification_grad_quantile,
                        densification_grad_abs_min=active_densification_grad_abs_min,
                    )

                    if (
                            save_gradient_diagnostics
                            and config.save_interval > 0
                            and densify_after <= iteration <= densify_until_iteration
                            and iteration % config.save_interval == 0
                    ):
                        save_densification_gradient_diagnostics(
                            output_dir=config.output_dir,
                            iteration=iteration,
                            positions=positions,
                            densify_position_grad_accum_np=densify_position_grad_accum_np,
                            densify_position_grad_denom_np=densify_position_grad_denom_np,
                            densify_position_grad_vector_accum_np=densify_position_grad_vector_accum_np,
                            photo_gradient_surfel_stats=photo_gradient_surfel_stats,
                            active_camera_count_max=len(training_camera_ids),
                        )

                    scale_prune_indices, opacity_prune_indices, indices_to_remove_list = maybe_make_prune_indices(
                        iteration=iteration, config=config, scales=scales, opacities=opacities,
                        trainable_surfel_mask=trainable_surfel_mask, prune_after=prune_after,
                        prune_interval=prune_interval, reset_opacity_interval=reset_opacity_interval,
                        opacity_prune_threshold=opacity_prune_threshold, max_prune_fraction=max_prune_fraction,
                    )

                    inactive_camera_cycle_indices = np.zeros((0,), dtype=np.int64)
                    if (camera_cycle_complete and iteration >= prune_after and inactive_gradient_prune_cycles > 0):
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
                    print(f"[Iter {iteration:04d}] Skipping densification/pruning due to opacity reset")

                if indices_to_remove_list or densification_result is not None:
                    old_params_for_optimizer = make_named_parameter_dict(positions, rotation_delta, scales, albedos,
                                                                         opacities, betas, powers, )
                    old_optimizer_for_migration = optimizer
                    old_point_count_for_migration = int(positions.shape[0])
                    keep_mask_np = np.ones(old_point_count_for_migration, dtype=bool)

                    if densification_result is not None and "update_source" in densification_result:
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

                        print(
                            f"[Iter {iteration:04d}] Pruning {indices_to_remove.size} unique surfels | "
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
                        copy_source_state_to_new=True,
                    )

                    active_learning_rates = update_optimizer_learning_rates(optimizer, learning_rate_schedules,
                                                                            iteration)
                    trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)
                    frozen_surfel_count = int((~trainable_surfel_mask).sum().item())
                    print(f"Frozen emissive surfels: {frozen_surfel_count} / {int(trainable_surfel_mask.numel())}")

                if camera_cycle_complete:
                    active_during_camera_cycle_np = np.zeros((positions.shape[0],), dtype=bool, )
                    visited_training_camera_ids_this_cycle.clear()

                if densification_interval > 0 and densify_after <= iteration <= densify_until_iteration and iteration % densification_interval == 0:
                    densify_position_grad_accum_np[:] = 0.0
                    densify_position_grad_denom_np[:] = 0.0
                    densify_position_grad_vector_accum_np[:] = 0.0

                save_interval = int(config.save_interval)
                should_save_iteration_outputs = (
                        save_interval > 0 and (iteration % save_interval == 0 or iteration == config.iterations))

                if should_save_iteration_outputs:
                    if config.one_camera_per_iteration:
                        save_forward_out = renderer.render_forward()
                    else:
                        save_forward_out = forward_out

                    save_iteration_outputs(
                        output_dir=config.output_dir,
                        iteration=iteration,
                        save_interval=config.save_interval,
                        final_iteration=config.iterations,
                        all_camera_ids=all_camera_ids,
                        forward_out=save_forward_out,
                        adjoint_images=adjoint_images,
                        renderer_settings=renderer_settings,
                        save_rgb=config.save_snapshot_rgb,
                        save_median_depth=config.save_snapshot_median_depth,
                        save_depth_distortion=config.save_snapshot_depth_distortion,
                        save_visible_normal=config.save_snapshot_visible_normal,
                        save_normal_from_depth=config.save_snapshot_normal_from_depth,
                        save_grad=config.save_snapshot_grad,
                    )

                if save_ply_files_interval > 0 and iteration % save_ply_files_interval == 0:
                    save_iteration_point_cloud_snapshot(config.output_dir, iteration, positions, rotations,
                                                        scales, albedos, opacities, betas, powers)

                num_points = positions.shape[0]
                iteration_end = time.perf_counter()
                iteration_time = iteration_end - iteration_start
                total_time = iteration_end - total_start_time

                csv_writer.writerow(
                    [
                        iteration,
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
                        averaged_loss_state["total_visibility_weighted_opacity_loss_raw"],
                        averaged_loss_state["total_visibility_weighted_opacity_loss_weighted"],
                        averaged_loss_state["total_bsdf_decay_loss_raw"],
                        averaged_loss_state["total_bsdf_decay_loss_weighted"],
                        averaged_loss_state["total_loss_value"],
                        num_points,
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

                if iteration % config.log_interval == 0 or iteration == 1:
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

                    print(
                        format_training_iteration_log(
                            iteration=iteration,
                            total_iterations=config.iterations,
                            iteration_time=iteration_time,
                            total_time=total_time,
                            num_points=num_points,
                            loss_state=averaged_loss_state,
                            lr_position=lr_position,
                            active_depth_distortion_weight=active_depth_distortion_weight,
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
                            visibility_opacity_gradients=visibility_opacity_gradients,
                            bsdf_decay_gradients=bsdf_decay_gradients,
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

                    if use_visibility_weighted_opacity and loss_state["visibility_weighted_opacity_maps_for_logging"]:
                        visibility_maps = loss_state["visibility_weighted_opacity_maps_for_logging"]
                        visibility_loss_mean = np.mean([float(v.mean()) for v in visibility_maps.values()])
                        print(f"visibility_weighted_opacity_raw_mean={visibility_loss_mean:.3e}")

                    if use_bsdf_decay:
                        print(format_gradient_stats("bsdf_decay", bsdf_decay_gradient_stats))

                    hotkey = poll_hotkey()
                    if hotkey == "s":
                        save_manual_snapshot(renderer, config.output_dir, iteration, positions, rotations,
                                             scales, albedos, opacities, betas, powers, training_camera_ids)
                    elif hotkey == "g":
                        save_gradients_snapshot(config.output_dir, iteration, grad_position_np, grad_rotation_np,
                                                grad_scales_np, grad_albedos_np, grad_opacities_np,
                                                grad_betas_np)

        except KeyboardInterrupt:
            elapsed = time.perf_counter() - total_start_time
            print(
                f"\nCtrl+C detected at iteration {iteration:04d}. "
                f"Total elapsed time: {elapsed:.1f} s. "
                "Stopping optimization loop and saving current result..."
            )

    apply_point_parameters(renderer, positions, rotations, scales, albedos, opacities, betas, powers)
    final_images = renderer.render_forward()

    final_rgb_loss = 0.0
    final_depth_distortion_loss_raw = 0.0
    final_depth_distortion_loss_weighted = 0.0
    final_normal_loss_raw = 0.0
    final_normal_loss_weighted = 0.0
    final_total_loss = 0.0

    final_depth_distortion_weight = scheduled_regularizer_weight(
        depth_distortion_base_weight,
        iteration=iteration,
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

    final_visibility_weighted_opacity_loss_raw = 0.0
    final_visibility_weighted_opacity_loss_weighted = 0.0

    if use_visibility_weighted_opacity:
        for camera_name in training_camera_ids:
            visibility_opacity_np = get_forward_visibility_weighted_opacity(final_images, camera_name)
            visibility_loss_raw = float(visibility_opacity_np.mean())
            visibility_loss_weighted = visibility_weighted_opacity_weight * visibility_loss_raw
            final_visibility_weighted_opacity_loss_raw += visibility_loss_raw
            final_visibility_weighted_opacity_loss_weighted += visibility_loss_weighted
            final_total_loss += visibility_loss_weighted

    print_loss_summary("Initial", *initial_loss_tuple)
    print_loss_summary(
        "Final",
        final_rgb_loss,
        final_depth_distortion_loss_raw,
        final_depth_distortion_loss_weighted,
        final_normal_loss_raw,
        final_normal_loss_weighted,
        final_visibility_weighted_opacity_loss_weighted,
        final_total_loss,
    )
    ply_path = config.output_dir / "points_final.ply"
    save_gaussians_to_ply(ply_path, positions, rotations, scales, albedos, opacities, betas, powers,
                          shape_default=0.0)

    print(f"Final parameters written to PLY: {ply_path}")
    print("\nOptimization completed.")
    print(f"Outputs saved in: {config.output_dir.resolve()}")
    print(f"Total optimization wall time: {time.perf_counter() - total_start_time:.1f} s")
