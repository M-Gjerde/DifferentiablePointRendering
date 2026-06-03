from __future__ import annotations

import time
from losses import compute_parameter_mse
from optimizers import create_learning_rate_schedules, update_optimizer_learning_rates
from render_hooks import apply_point_parameters, rebuild_bvh, remove_points, add_new_points
from training_helpers import *


def run_optimization(renderer: pale.Renderer, config: OptimizationConfig,
                     renderer_settings: RendererSettingsConfig) -> None:
    target_images, training_camera_ids, all_camera_ids = load_target_images(renderer, Path(config.dataset_path))

    depth_distortion_base_weight = float(getattr(config, "depth_distort_weight", 0.0))
    depth_distortion_start_iteration = int(getattr(config, "depth_distort_start_iteration", 0))

    normal_consistency_weight = float(getattr(config, "normal_consistency_weight", 0.0))
    visibility_weighted_opacity_weight = float(config.visibility_weighted_opacity_weight)

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
        f"weight={visibility_weighted_opacity_weight:.3e}"
    )
    initial_params = fetch_parameters(renderer)
    initial_params_reference = make_initial_params_reference(initial_params)
    print(f"Fetched {initial_params['position'].shape[0]} initial points from PLY.")

    device = torch.device(config.device)
    positions, tangent_u, tangent_v, scales, albedos, opacities, betas, powers = create_torch_parameters_from_initial(
        initial_params, device)

    trainable_surfel_mask = make_trainable_surfel_mask_from_powers(powers)
    frozen_surfel_count = int((~trainable_surfel_mask).sum().item())
    print(f"Frozen emissive surfels: {frozen_surfel_count} / {int(trainable_surfel_mask.numel())}")

    verify_parameters_inplane(positions, tangent_u, tangent_v, scales, albedos, opacities, betas,
                              trainable_surfel_mask=trainable_surfel_mask)
    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, albedos, opacities, betas, powers)
    rebuild_bvh(renderer)

    positions, tangent_u, tangent_v, scales, albedos, opacities, betas, powers = refetch_parameters_as_torch(renderer,
                                                                                                             device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = create_masked_optimizer(config, positions, tangent_u, tangent_v, scales, albedos, opacities, betas,
                                        powers)
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
        all_camera_ids=all_camera_ids, positions=positions, tangent_u=tangent_u, tangent_v=tangent_v, scales=scales,
        albedos=albedos, opacities=opacities, betas=betas, powers=powers,
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
    reset_opacity_interval = int(config.reset_opacity_interval)
    reset_opacity_value = float(config.reset_opacity_value)
    densification_verbose = bool(config.densification_verbose)
    densification_grad_quantile = as_config_float(config.densification_grad_quantile)
    densification_grad_abs_min = float(config.densification_grad_abs_min)
    densify_bsdf_floor = float(config.densify_bsdf_floor)
    densify_bsdf_gamma = float(config.densify_bsdf_gamma)
    rebuild_bvh_interval = max(int(config.rebuild_bvh_interval), 1)

    densify_position_grad_accum_np = np.zeros((positions.shape[0], 1), dtype=np.float32)
    densify_position_grad_denom_np = np.zeros((positions.shape[0], 1), dtype=np.float32)
    densify_position_grad_vector_accum_np = np.zeros(tuple(positions.shape), dtype=np.float32)

    metrics_csv_path = config.output_dir / "metrics.csv"
    total_start_time = time.perf_counter()
    iteration = 0

    with open(metrics_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        write_metrics_header(csv_writer)

        try:
            for iteration in range(1, config.iterations + 1):
                iteration_start = time.perf_counter()
                forward_out = renderer.render_forward()

                active_depth_distortion_weight = scheduled_regularizer_weight(
                    depth_distortion_base_weight,
                    iteration=iteration,
                    start_iteration=depth_distortion_start_iteration,
                )

                use_depth_distortion_gradients = active_depth_distortion_weight != 0.0

                loss_state = compute_iteration_losses_and_adjoints(
                    forward_out=forward_out,
                    target_images=target_images,
                    training_camera_ids=training_camera_ids,
                    depth_distortion_weight=active_depth_distortion_weight,
                    normal_consistency_weight=normal_consistency_weight,
                    visibility_weighted_opacity_weight=visibility_weighted_opacity_weight,
                    use_depth_distortion=use_depth_distortion,
                    use_normal_consistency=use_normal_consistency,
                    use_visibility_weighted_opacity=use_visibility_weighted_opacity,
                )

                photo_gradients, adjoint_images = renderer.render_backward(loss_state["loss_grad_images"])

                surface_regularizer_gradients: Dict[str, np.ndarray] = {}
                use_surface_regularizers = (
                        use_depth_distortion_gradients
                        or use_normal_consistency
                        or use_visibility_weighted_opacity
                )

                if use_surface_regularizers:
                    surface_regularizer_gradients = renderer.render_surface_regularizers_backward(
                        training_camera_ids,
                        loss_state["depth_distortion_grad_images"],
                        loss_state["visible_normal_adjoints"],
                        loss_state["depth_normal_adjoints"],
                    )

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

                total_gradients = sum_gradient_dicts(photo_gradients, surface_regularizer_gradients)

                grad_position_np, grad_tangent_u_np, grad_tangent_v_np, grad_scales_np, grad_albedos_np, grad_opacities_np, grad_betas_np = extract_total_gradient_arrays(
                    total_gradients
                )

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

                update_densification_statistics(
                    iteration=iteration, densification_interval=densification_interval,
                    densification_stats_warmup_iterations=densification_stats_warmup_iterations,
                    densify_position_grad_accum_np=densify_position_grad_accum_np,
                    densify_position_grad_denom_np=densify_position_grad_denom_np,
                    densify_position_grad_vector_accum_np=densify_position_grad_vector_accum_np,
                    total_gradients=total_gradients, tangent_u=tangent_u, tangent_v=tangent_v,
                    albedos=albedos, trainable_surfel_mask=trainable_surfel_mask,
                    densify_bsdf_floor=densify_bsdf_floor, densify_bsdf_gamma=densify_bsdf_gamma,
                )

                optimizer.zero_grad(set_to_none=True)

                zero_frozen_surfel_gradients_np(
                    trainable_surfel_mask, grad_position_np, grad_tangent_u_np, grad_tangent_v_np,
                    grad_scales_np, grad_albedos_np, grad_opacities_np, grad_betas_np,
                )

                assign_numpy_gradients_to_tensors(
                    device, positions, tangent_u, tangent_v, scales, albedos, opacities, betas,
                    grad_position_np, grad_tangent_u_np, grad_tangent_v_np, grad_scales_np,
                    grad_albedos_np, grad_opacities_np, grad_betas_np,
                )

                active_learning_rates = update_optimizer_learning_rates(optimizer, learning_rate_schedules, iteration)
                optimizer.step()

                if reset_opacity_interval > 0 and iteration % reset_opacity_interval == 0:
                    with torch.no_grad():
                        opacities[trainable_surfel_mask] = float(reset_opacity_value)
                    print(f"[Iter {iteration:04d}] Resetting all opacities to {reset_opacity_value}")

                verify_parameters_inplane(positions, tangent_u, tangent_v, scales, albedos, opacities, betas,
                                          trainable_surfel_mask=trainable_surfel_mask)
                apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, albedos, opacities, betas,
                                       powers)

                if iteration % rebuild_bvh_interval == 0:
                    rebuild_bvh(renderer)

                densification_result = maybe_make_densification_result(
                    iteration=iteration, config=config, positions=positions, tangent_u=tangent_u,
                    tangent_v=tangent_v, scales=scales, albedos=albedos, opacities=opacities,
                    betas=betas, powers=powers, trainable_surfel_mask=trainable_surfel_mask,
                    densify_position_grad_accum_np=densify_position_grad_accum_np,
                    densify_position_grad_denom_np=densify_position_grad_denom_np,
                    densify_position_grad_vector_accum_np=densify_position_grad_vector_accum_np,
                    densify_after=densify_after, densify_until_iteration=densify_until_iteration,
                    densification_interval=densification_interval, densification_verbose=densification_verbose,
                    densification_grad_quantile=densification_grad_quantile,
                    densification_grad_abs_min=densification_grad_abs_min,
                )

                scale_prune_indices, opacity_prune_indices, indices_to_remove_list = maybe_make_prune_indices(
                    iteration=iteration, config=config, scales=scales, opacities=opacities,
                    trainable_surfel_mask=trainable_surfel_mask, prune_after=prune_after,
                    prune_interval=prune_interval, reset_opacity_interval=reset_opacity_interval,
                    opacity_prune_threshold=opacity_prune_threshold, max_prune_fraction=max_prune_fraction,
                )

                if indices_to_remove_list or densification_result is not None:
                    old_params_for_optimizer = make_named_parameter_dict(positions, tangent_u, tangent_v, scales,
                                                                         albedos, opacities, betas, powers)
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
                        apply_densification_source_updates_inplace(densification_result, positions, tangent_u,
                                                                   tangent_v, scales, albedos, opacities, betas, powers)
                        verify_parameters_inplane(positions, tangent_u, tangent_v, scales, albedos, opacities, betas,
                                                  trainable_surfel_mask=trainable_surfel_mask)
                        apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, albedos, opacities,
                                               betas, powers)

                    if indices_to_remove_list:
                        scale_prune_set = set(int(i) for i in scale_prune_indices)
                        opacity_prune_set = set(int(i) for i in opacity_prune_indices)
                        overlap_set = scale_prune_set & opacity_prune_set
                        indices_to_remove = np.unique(np.asarray(indices_to_remove_list, dtype=np.int64))

                        print(
                            f"[Iter {iteration:04d}] Pruning {indices_to_remove.size} unique surfels | "
                            f"scale={len(scale_prune_set)}, opacity={len(opacity_prune_set)}, "
                            f"both={len(overlap_set)}, scale_only={len(scale_prune_set - opacity_prune_set)}, "
                            f"opacity_only={len(opacity_prune_set - scale_prune_set)}"
                        )

                        keep_mask_np[indices_to_remove] = False
                        remove_points(renderer, indices_to_remove)
                        densify_position_grad_accum_np = densify_position_grad_accum_np[keep_mask_np]
                        densify_position_grad_denom_np = densify_position_grad_denom_np[keep_mask_np]
                        densify_position_grad_vector_accum_np = densify_position_grad_vector_accum_np[keep_mask_np]

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

                    rebuild_bvh(renderer)
                    positions, tangent_u, tangent_v, scales, albedos, opacities, betas, powers = refetch_parameters_as_torch(
                        renderer, device)

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
                    verify_parameters_inplane(positions, tangent_u, tangent_v, scales, albedos, opacities, betas,
                                              trainable_surfel_mask=trainable_surfel_mask)
                    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, albedos, opacities, betas,
                                           powers)
                    rebuild_bvh(renderer)

                    new_params_for_optimizer = make_named_parameter_dict(positions, tangent_u, tangent_v, scales,
                                                                         albedos, opacities, betas, powers)
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

                if densification_interval > 0 and densify_after <= iteration <= densify_until_iteration and iteration % densification_interval == 0:
                    densify_position_grad_accum_np[:] = 0.0
                    densify_position_grad_denom_np[:] = 0.0
                    densify_position_grad_vector_accum_np[:] = 0.0

                save_iteration_outputs(
                    output_dir=config.output_dir, iteration=iteration, save_interval=config.save_interval,
                    final_iteration=config.iterations, all_camera_ids=all_camera_ids,
                    forward_out=forward_out, adjoint_images=adjoint_images, renderer_settings=renderer_settings,
                )

                if prune_interval > 0 and iteration % prune_interval == 0:
                    save_iteration_point_cloud_snapshot(config.output_dir, iteration, positions, tangent_u, tangent_v,
                                                        scales, albedos, opacities, betas, powers)

                num_points = positions.shape[0]
                iteration_end = time.perf_counter()
                iteration_time = iteration_end - iteration_start
                total_time = iteration_end - total_start_time

                parameter_mse = compute_parameter_mse(
                    current_params_as_numpy(positions, tangent_u, tangent_v, scales, albedos, opacities, betas, powers),
                    initial_params_reference)

                csv_writer.writerow(
                    [
                        iteration,
                        "ALL_CAMERAS",
                        loss_state["total_rgb_loss_value"],
                        loss_state["total_depth_distortion_loss_raw"],
                        loss_state["total_depth_distortion_loss_weighted"],
                        loss_state["total_normal_loss_raw"],
                        loss_state["total_normal_loss_weighted"],
                        loss_state["total_visibility_weighted_opacity_loss_raw"],
                        loss_state["total_visibility_weighted_opacity_loss_weighted"],
                        loss_state["total_loss_value"],
                        parameter_mse,
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
                    lr_position = active_learning_rates.get("position", float(config.learning_rate_position))

                    print(
                        f"[Iter {iteration:04d}/{config.iterations}] "
                        f"RGB={loss_state['total_rgb_loss_value']:.3e}, "
                        f"DdistRaw={loss_state['total_depth_distortion_loss_raw']:.3e}, "
                        f"DdistW={loss_state['total_depth_distortion_loss_weighted']:.3e}, "
                        f"DdistActiveW={active_depth_distortion_weight:.3e}, "
                        f"NconsRaw={loss_state['total_normal_loss_raw']:.3e}, "
                        f"NconsW={loss_state['total_normal_loss_weighted']:.3e}, "
                        f"VisOpacityRaw={loss_state['total_visibility_weighted_opacity_loss_raw']:.3e}, "
                        f"VisOpacityW={loss_state['total_visibility_weighted_opacity_loss_weighted']:.3e}, "
                        f"Total={loss_state['total_loss_value']:.3e}, "
                        f"lr_pos={lr_position:.3e}, t={iteration_time:.3f} s, "
                        f"pos_rms={grad_pos_rms:.2e}, tu_rms={grad_tanu_rms:.2e}, "
                        f"tv_rms={grad_tanv_rms:.2e}, su,sv_rms={grad_scale_rms:.2e}, "
                        f"rho_rms={grad_albedo_rms:.2e}, eta_rms={grad_opacity_rms:.2e}, "
                        f"beta_rms={grad_beta_rms:.2e}, pos_max={grad_pos_max:.2e}, "
                        f"tu_max={grad_tanu_max:.2e}, tv_max={grad_tanv_max:.2e}, "
                        f"su,sv_max={grad_scale_max:.2e}, rho_max={grad_albedo_max:.2e}, "
                        f"eta_max={grad_opacity_max:.2e}, beta_max={grad_beta_max:.2e}, "
                        f"pts={num_points}, t_total={total_time:.1f} s, "
                        f"it/s={1.0 / max(iteration_time, 1.0e-12):.2f}"
                    )

                    print(format_loss_breakdown(loss_state))
                    print(
                        format_gradient_source_balance(photo_gradients, surface_regularizer_gradients, total_gradients))

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

                    hotkey = poll_hotkey()
                    if hotkey == "s":
                        save_manual_snapshot(renderer, config.output_dir, iteration, positions, tangent_u, tangent_v,
                                             scales, albedos, opacities, betas, powers, training_camera_ids)
                    elif hotkey == "g":
                        save_gradients_snapshot(config.output_dir, iteration, grad_position_np, grad_tangent_u_np,
                                                grad_tangent_v_np, grad_scales_np, grad_albedos_np, grad_opacities_np,
                                                grad_betas_np)

        except KeyboardInterrupt:
            elapsed = time.perf_counter() - total_start_time
            print(
                f"\nCtrl+C detected at iteration {iteration:04d}. "
                f"Total elapsed time: {elapsed:.1f} s. "
                "Stopping optimization loop and saving current result..."
            )

    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, albedos, opacities, betas, powers)
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
    save_gaussians_to_ply(ply_path, positions, tangent_u, tangent_v, scales, albedos, opacities, betas, powers,
                          shape_default=0.0)

    print(f"Final parameters written to PLY: {ply_path}")
    print("\nOptimization completed.")
    print(f"Outputs saved in: {config.output_dir.resolve()}")
    print(f"Total optimization wall time: {time.perf_counter() - total_start_time:.1f} s")
