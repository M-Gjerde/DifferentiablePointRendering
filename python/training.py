from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch

import pale
from config import OptimizationConfig, RendererSettingsConfig
from io_utils import (
    load_target_image,
    save_positions_numpy,
    save_render,
    save_gradient_sign_png_py,
    save_loss_image,
    save_gaussians_to_ply,
)
from losses import (
    compute_l2_loss,
    compute_l2_loss_and_grad,
    compute_l2_ssim_loss_and_grad,
    compute_parameter_mse,
)
from optimizers import create_optimizer
from density_control import (
    densify_points_long_axis_split,
    compute_prune_indices_by_opacity,
)
from render_hooks import (
    remove_points,
    fetch_parameters,
    apply_point_parameters,
    orthonormalize_tangents_inplace,
    verify_scales_inplace,
    verify_colors_inplace,
    verify_opacities_inplace,
    verify_beta_inplace,
    add_new_points,
    rebuild_bvh,
    get_camera_names,
)
from debug_init_utils import add_debug_noise_to_initial_parameters
from repulsion import compute_elliptical_repulsion_loss  # noqa: F401  (kept for future use)

import sys
import select


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def poll_save_hotkey() -> bool:
    """
    Non-blocking check for a single-line keyboard input.
    Returns True if the user typed 's' (case-insensitive) and pressed Enter.
    """
    if not sys.stdin.isatty():
        return False

    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return False

    line = sys.stdin.readline().strip()
    return line.lower() == "s"


def save_manual_snapshot(
        output_dir: Path,
        iteration: int,
        positions: torch.Tensor,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        current_rgb_np: np.ndarray,
) -> None:
    """
    Save the current state using the same filenames as the final output.
    These will be overwritten later, both by future manual saves and at the end.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save current render
    save_render(output_dir / "render_final.png", current_rgb_np)

    # Save full parameter set as PLY
    ply_path = output_dir / "points_final.ply"
    save_gaussians_to_ply(
        ply_path,
        positions,
        tangent_u,
        tangent_v,
        scales,
        colors,
        opacities,
        betas,
        shape_default=0.0,
    )
    print(
        f"[Iter {iteration:04d}] Hotkey 's' pressed -> "
        f"saved render_final.png and points_final.ply"
    )


def clear_output_dir(output_dir: Path) -> None:
    """
    Remove all files in the given directory (and direct subdirectories),
    then ensure the directory exists.
    """
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                for sub_item in item.iterdir():
                    if sub_item.is_file():
                        sub_item.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def refetch_parameters_as_torch(
        renderer: pale.Renderer,
        device: torch.device,
) -> Tuple[torch.nn.Parameter, ...]:
    """
    Fetch parameters from the renderer, convert them to torch.nn.Parameter tensors.
    Returns (positions, tangent_u, tangent_v, scales, colors, opacities).
    """
    updated = fetch_parameters(renderer)

    positions = torch.nn.Parameter(
        torch.tensor(updated["position"], device=device, dtype=torch.float32)
    )
    tangent_u = torch.nn.Parameter(
        torch.tensor(updated["tangent_u"], device=device, dtype=torch.float32)
    )
    tangent_v = torch.nn.Parameter(
        torch.tensor(updated["tangent_v"], device=device, dtype=torch.float32)
    )
    scales = torch.nn.Parameter(
        torch.tensor(updated["scale"], device=device, dtype=torch.float32)
    )
    colors = torch.nn.Parameter(
        torch.tensor(updated["color"], device=device, dtype=torch.float32)
    )
    opacities = torch.nn.Parameter(
        torch.tensor(updated["opacity"], device=device, dtype=torch.float32)
    )
    betas = torch.nn.Parameter(
        torch.tensor(updated["beta"], device=device, dtype=torch.float32)
    )

    return positions, tangent_u, tangent_v, scales, colors, opacities, betas


def verify_parameters_inplane(
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
) -> None:
    """
    Enforce parameter constraints in-place:
    - orthonormal tangents
    - valid scales, colors, and opacities
    """
    orthonormalize_tangents_inplace(tangent_u, tangent_v)
    verify_scales_inplace(scales)
    verify_colors_inplace(colors)
    verify_opacities_inplace(opacities)
    verify_beta_inplace(betas)


def assign_numpy_gradients_to_tensors(
        device: torch.device,
        positions: torch.nn.Parameter,
        tangent_u: torch.nn.Parameter,
        tangent_v: torch.nn.Parameter,
        scales: torch.nn.Parameter,
        colors: torch.nn.Parameter,
        opacities: torch.nn.Parameter,
        betas: torch.nn.Parameter,
        grad_position_np: np.ndarray,
        grad_tangent_u_np: np.ndarray,
        grad_tangent_v_np: np.ndarray,
        grad_scales_np: np.ndarray,
        grad_colors_np: np.ndarray,
        grad_opacities_np: np.ndarray,
        grad_betas_np: np.ndarray,
) -> None:
    """
    Copy numpy gradient arrays into the .grad fields of the given torch Parameters.
    """
    positions.grad = torch.tensor(grad_position_np, device=device, dtype=torch.float32)
    tangent_u.grad = torch.tensor(grad_tangent_u_np, device=device, dtype=torch.float32)
    tangent_v.grad = torch.tensor(grad_tangent_v_np, device=device, dtype=torch.float32)
    scales.grad = torch.tensor(grad_scales_np, device=device, dtype=torch.float32)
    colors.grad = torch.tensor(grad_colors_np, device=device, dtype=torch.float32)
    opacities.grad = torch.tensor(grad_opacities_np, device=device, dtype=torch.float32)
    betas.grad = torch.tensor(grad_betas_np, device=device, dtype=torch.float32)


def compute_density_importance(
        grad_position_np: np.ndarray,
        grad_scales_np: np.ndarray,
        grad_opacities_np: np.ndarray,
) -> torch.Tensor:
    """
    Construct a scalar importance per point from position, scale, and opacity gradients.
    """
    grad_pos_norm = np.linalg.norm(grad_position_np, axis=1)
    grad_scale_norm = np.linalg.norm(grad_scales_np, axis=1)
    grad_opacity_abs = np.abs(grad_opacities_np).reshape(-1)

    importance_np = (
            grad_pos_norm
            + 0.3 * grad_scale_norm
            + 0.1 * grad_opacity_abs
    ).astype(np.float32)

    return torch.from_numpy(importance_np)


# ---------------------------------------------------------------------------
# Main optimization
# ---------------------------------------------------------------------------

def run_optimization(
        renderer: Pale.Renderer,
        config: OptimizationConfig,
        renderer_settings: RendererSettingsConfig,
) -> None:

    # ------------------------------------------------------------------
    # 1a. Load target images per camera
    # ------------------------------------------------------------------
    target_path = Path(config.dataset_path)
    target_images: Dict[str, np.ndarray] = {}

    # Multi-camera mode: interpret dataset path as a directory
    if not target_path.is_dir():
        raise RuntimeError(
            f"Target path '{target_path}' must be a directory when multiple cameras are used."
        )

    print(f"Loading target images from directory: {target_path}")
    camera_ids = get_camera_names(renderer)
    for camera_name in camera_ids:
        image_path = target_path / f"{camera_name}" / "out_photonmap.png"
        if not image_path.is_file():
            raise RuntimeError(
                f"Missing target image for camera '{camera_name}': {image_path}"
            )
        target_images[camera_name] = load_target_image(image_path)
        print(
            f"  Camera '{camera_name}': loaded target {image_path} "
            f"with shape {target_images[camera_name].shape}"
        )

    # ------------------------------------------------------------------
    # Fetch initial parameters from renderer
    # ------------------------------------------------------------------
    initial_params = fetch_parameters(renderer)
    initial_positions_np = initial_params["position"]
    initial_tangent_u_np = initial_params["tangent_u"]
    initial_tangent_v_np = initial_params["tangent_v"]
    initial_scale_np = initial_params["scale"]
    initial_color_np = initial_params["color"]
    initial_opacity_np = initial_params["opacity"]
    initial_beta_np = initial_params["beta"]

    num_points_initial = initial_positions_np.shape[0]
    print(f"Fetched {num_points_initial} initial points from PLY.")

    # Immutable reference for parameter MSE
    initial_params_reference: Dict[str, np.ndarray] = {
        "position": initial_positions_np.copy(),
        "tangent_u": initial_tangent_u_np.copy(),
        "tangent_v": initial_tangent_v_np.copy(),
        "scale": initial_scale_np.copy(),
        "color": initial_color_np.copy(),
        "opacity": initial_opacity_np.copy(),
        "beta": initial_beta_np.copy(),
    }

    # (Optionally apply debug noise as before...)
    apply_noise = False
    if apply_noise:
        (
            initial_positions_np,
            initial_tangent_u_np,
            initial_tangent_v_np,
            initial_scale_np,
            initial_color_np,
            initial_opacity_np,
            initial_beta_np,
        ) = add_debug_noise_to_initial_parameters(
            initial_positions_np,
            initial_tangent_u_np,
            initial_tangent_v_np,
            initial_scale_np,
            initial_color_np,
            initial_opacity_np,
            initial_beta_np,
        )
        print("Initial parameters perturbed by debug Gaussian noise.")

    device = torch.device(config.device)

    # Create initial trainable tensors from numpy (unchanged)
    positions = torch.nn.Parameter(
        torch.tensor(initial_positions_np, device=device, dtype=torch.float32)
    )
    tangent_u = torch.nn.Parameter(
        torch.tensor(initial_tangent_u_np, device=device, dtype=torch.float32)
    )
    tangent_v = torch.nn.Parameter(
        torch.tensor(initial_tangent_v_np, device=device, dtype=torch.float32)
    )
    scales = torch.nn.Parameter(
        torch.tensor(initial_scale_np, device=device, dtype=torch.float32)
    )
    colors = torch.nn.Parameter(
        torch.tensor(initial_color_np, device=device, dtype=torch.float32)
    )
    opacities = torch.nn.Parameter(
        torch.tensor(initial_opacity_np, device=device, dtype=torch.float32)
    )
    betas = torch.nn.Parameter(
        torch.tensor(initial_beta_np, device=device, dtype=torch.float32)
    )

    # ------------------------------------------------------------------
    # 2. Initial reparameterization and sync with renderer
    # ------------------------------------------------------------------
    verify_parameters_inplane(tangent_u, tangent_v, scales, colors, opacities, betas)

    apply_point_parameters(
        renderer, positions, tangent_u, tangent_v, scales, colors, opacities, betas
    )
    rebuild_bvh(renderer)

    (
        positions,
        tangent_u,
        tangent_v,
        scales,
        colors,
        opacities,
        betas,
    ) = refetch_parameters_as_torch(renderer, device)

    optimizer = create_optimizer(
        config,
        positions,
        tangent_u,
        tangent_v,
        scales,
        colors,
        opacities,
        betas,
    )

    # ------------------------------------------------------------------
    # 3. Initial loss and output dir setup (multi-camera)
    # ------------------------------------------------------------------
    # Forward pass: dict[name -> HxWx3]
    initial_images = renderer.render_forward()

    initial_loss = 0.0
    main_camera = camera_ids[0]

    clear_output_dir(config.output_dir)

    for camera_name in camera_ids:
        img_np = np.asarray(initial_images[camera_name], dtype=np.float32, order="C")
        tgt_np = target_images[camera_name]

        loss_cam, _ = compute_l2_loss_and_grad(img_np, tgt_np)
        initial_loss += loss_cam

        camera_base_dir = config.output_dir / camera_name
        camera_base_dir.mkdir(parents=True, exist_ok=True)

        # Save per-camera initial render and target
        save_render(
            config.output_dir / f"render_initial_{camera_name}.png",
            img_np,
        )
        save_render(
            config.output_dir / f"render_target_{camera_name}.png",
            tgt_np,
        )

    print(f"Initial loss (L2, summed over cameras): {initial_loss:.6e}")

    # ------------------------------------------------------------------
    # 4. Density control / scheduling hyperparameters
    # ------------------------------------------------------------------
    iteration = 0

    densification_interval = 200
    prune_interval = 100
    burnin_iterations = 0
    reset_opacity_interval = int(1e10)

    opacity_prune_threshold = 0.5
    max_prune_fraction = 0.3
    rebuild_bvh_interval = 1

    metrics_csv_path = config.output_dir / "metrics.csv"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["iteration", "loss_l2", "parameter_mse", "num_points", "iteration_time_sec"]
        )

        try:
            for iteration in range(1, config.iterations + 1):
                iteration_start = time.perf_counter()

                # --------------------------------------------------------------
                # 5. Forward pass and image-space loss (multi-camera)
                # --------------------------------------------------------------
                # dict[name -> HxWx3]
                current_images = renderer.render_forward()

                total_loss_value = 0.0
                loss_grad_images: Dict[str, np.ndarray] = {}
                loss_images: Dict[str, np.ndarray] = {}

                for camera_name in camera_ids:
                    current_rgb_np = np.asarray(
                        current_images[camera_name],
                        dtype=np.float32,
                        order="C",
                    )
                    target_rgb_np = target_images[camera_name]

                    loss_value_cam, loss_grad_cam, loss_image_cam = compute_l2_loss_and_grad(
                        current_rgb_np,
                        target_rgb_np,
                        return_loss_image=True,
                    )

                    total_loss_value += float(loss_value_cam)
                    loss_grad_images[camera_name] = loss_grad_cam
                    loss_images[camera_name] = loss_image_cam

                # Use main camera image for manual snapshot and some logs
                current_main_rgb_np = np.asarray(
                    current_images[main_camera],
                    dtype=np.float32,
                    order="C",
                )

                # --------------------------------------------------------------
                # 6. Backward pass in renderer (multi-camera)
                # --------------------------------------------------------------
                gradients, adjoint_images = renderer.render_backward(loss_grad_images)
                # `adjoint_images` is dict[name -> HxWx4 float]

                # Extract numpy gradients
                grad_position_np = np.asarray(gradients["position"], dtype=np.float32, order="C")
                grad_tangent_u_np = np.asarray(gradients["tangent_u"], dtype=np.float32, order="C")
                grad_tangent_v_np = np.asarray(gradients["tangent_v"], dtype=np.float32, order="C")
                grad_scales_np = np.asarray(gradients["scale"], dtype=np.float32, order="C")
                grad_colors_np = np.asarray(gradients["color"], dtype=np.float32, order="C")
                grad_opacities_np = np.asarray(gradients["opacity"], dtype=np.float32, order="C")
                grad_betas_np = np.asarray(gradients["beta"], dtype=np.float32, order="C")

                # Zero the first row for each
                #grad_position_np[0, :] = 0
                #grad_tangent_u_np[0, :] = 0
                #grad_tangent_v_np[0, :] = 0
                #grad_scales_np[0, :] = 0
                #grad_colors_np[0, :] = 0
                #grad_opacities_np[0] = 0
                #grad_betas_np[0] = 0

                # Sanity check shapes
                current_positions_shape = tuple(positions.shape)
                if grad_position_np.shape != current_positions_shape:
                    raise RuntimeError(
                        f"Gradient shape mismatch for position: expected {current_positions_shape}, "
                        f"got {grad_position_np.shape}"
                    )

                # --------------------------------------------------------------
                # 7. Optimizer step
                # --------------------------------------------------------------
                optimizer.zero_grad(set_to_none=True)

                assign_numpy_gradients_to_tensors(
                    device,
                    positions,
                    tangent_u,
                    tangent_v,
                    scales,
                    colors,
                    opacities,
                    betas,
                    grad_position_np,
                    grad_tangent_u_np,
                    grad_tangent_v_np,
                    grad_scales_np,
                    grad_colors_np,
                    grad_opacities_np,
                    grad_betas_np,
                )

                optimizer.step()

                # Reset opacities on schedule (unchanged)
                if iteration % reset_opacity_interval == 0:
                    with torch.no_grad():
                        opacities[:] = 0.1
                    print(f"[Iter {iteration:04d}] Resetting all opacities to 0.1")

                # --------------------------------------------------------------
                # 8. Reparameterization, sync, BVH (unchanged logic)
                # --------------------------------------------------------------
                verify_parameters_inplane(tangent_u, tangent_v, scales, colors, opacities, betas)

                apply_point_parameters(
                    renderer,
                    positions,
                    tangent_u,
                    tangent_v,
                    scales,
                    colors,
                    opacities,
                    betas,
                )

                if iteration % rebuild_bvh_interval == 0:
                    rebuild_bvh(renderer)
                    (
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                        betas,
                    ) = refetch_parameters_as_torch(renderer, device)

                    optimizer = create_optimizer(
                        config,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                        betas,
                    )

                # --------------------------------------------------------------
                # 9. Densification + pruning (unchanged)
                # --------------------------------------------------------------
                densification_result: Optional[Dict[str, np.ndarray]] = None
                indices_to_remove_list: List[int] = []

                if iteration >= burnin_iterations and iteration % densification_interval == 0:
                    importance_tensor = compute_density_importance(
                        grad_position_np,
                        grad_scales_np,
                        grad_opacities_np,
                    )

                    max_new_points = max(
                        int(0.2 * positions.shape[0]),
                        5,
                    )

                    densification_result = densify_points_long_axis_split(
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                        betas,
                        importance=importance_tensor,
                        max_new_points=max_new_points,
                        split_distance=0.15,
                        minor_axis_shrink=0.85,
                        opacity_reduction=0.6,
                        min_long_axis_scale=0.03,
                        grad_threshold=1e-7,
                    )

                    if densification_result is not None:
                        parent_indices = densification_result["prune_indices"]
                        if parent_indices is not None and len(parent_indices) > 0:
                            indices_to_remove_list.extend(int(i) for i in parent_indices)

                if iteration >= burnin_iterations and iteration % prune_interval == 0:
                    opacity_prune_indices = compute_prune_indices_by_opacity(
                        opacities,
                        min_opacity=opacity_prune_threshold,
                        use_quantile=False,
                        max_fraction_to_prune=max_prune_fraction,
                    )
                    if opacity_prune_indices.size > 0:
                        indices_to_remove_list.extend(int(i) for i in opacity_prune_indices)

                if indices_to_remove_list or densification_result is not None:
                    if indices_to_remove_list:
                        indices_to_remove = np.unique(
                            np.asarray(indices_to_remove_list, dtype=np.int64)
                        )
                        remove_points(renderer, indices_to_remove)
                        rebuild_bvh(renderer)

                    if densification_result is not None:
                        add_new_points(renderer, densification_result)

                    (
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                        betas,
                    ) = refetch_parameters_as_torch(renderer, device)

                    verify_parameters_inplane(tangent_u, tangent_v, scales, colors, opacities, betas)
                    apply_point_parameters(
                        renderer,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                        betas,
                    )

                    rebuild_bvh(renderer)
                    optimizer = create_optimizer(
                        config,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                        betas,
                    )

                # --------------------------------------------------------------
                # 10. Snapshots (per-camera images)
                # --------------------------------------------------------------
                if iteration % config.save_interval == 0 or iteration == config.iterations:
                    for camera_name in camera_ids:
                        camera_base_dir = config.output_dir / camera_name
                        camera_render_dir = camera_base_dir / "render"
                        camera_grad_dir = camera_base_dir / "grad"
                        camera_render_dir.mkdir(parents=True, exist_ok=True)
                        camera_grad_dir.mkdir(parents=True, exist_ok=True)

                        # Per-camera render
                        image_numpy = np.asarray(
                            current_images[camera_name],
                            dtype=np.float32,
                            order="C",
                        )
                        render_path = (
                                camera_render_dir
                                / f"{iteration:04d}_render.png"
                        )
                        save_render(render_path, image_numpy)

                        # Per-camera adjoint/gradient visualization
                        grad_image_numpy = np.asarray(
                            adjoint_images[camera_name],
                            dtype=np.float32,
                            order="C",
                        )
                        grad_image_numpy = np.nan_to_num(
                            grad_image_numpy,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )

                        grad_path = (
                                camera_grad_dir
                                / f"{iteration:04d}_grad_099.png"
                        )
                        save_gradient_sign_png_py(
                            grad_path,
                            grad_image_numpy,
                            adjoint_spp=renderer_settings.adjoint_passes,
                            abs_quantile=0.999,
                            flip_y=False,
                        )

                        # Loss image: store under main camera
                        main_loss_image = loss_images[camera_name]
                        main_camera_loss_root = config.output_dir / camera_name
                        save_loss_image(main_camera_loss_root, main_loss_image, iteration)

                # --------------------------------------------------------------
                # 11. Metrics and logging
                # --------------------------------------------------------------
                num_points = positions.shape[0]
                iteration_end = time.perf_counter()
                iteration_time = iteration_end - iteration_start

                current_params_np = {
                    "position": positions.detach().cpu().numpy(),
                    "tangent_u": tangent_u.detach().cpu().numpy(),
                    "tangent_v": tangent_v.detach().cpu().numpy(),
                    "scale": scales.detach().cpu().numpy(),
                    "color": colors.detach().cpu().numpy(),
                    "opacity": opacities.detach().cpu().numpy(),
                    "beta": betas.detach().cpu().numpy(),
                }
                parameter_mse = compute_parameter_mse(
                    current_params_np,
                    initial_params_reference,
                )

                csv_writer.writerow(
                    [iteration, total_loss_value, parameter_mse, num_points, iteration_time]
                )
                csv_file.flush()

                if iteration % config.log_interval == 0 or iteration == 1:
                    denom = max(num_points, 1)
                    grad_norm = float(np.linalg.norm(grad_position_np) / denom)
                    grad_tanu = float(np.linalg.norm(grad_tangent_u_np) / denom)
                    grad_tanv = float(np.linalg.norm(grad_tangent_v_np) / denom)
                    grad_scale = float(np.linalg.norm(grad_scales_np) / denom)
                    grad_color = float(np.linalg.norm(grad_colors_np) / denom)
                    grad_opacity = float(np.linalg.norm(grad_opacities_np) / denom)
                    grad_beta = float(np.linalg.norm(grad_betas_np) / denom)

                    print(
                        f"[Iter {iteration:04d}/{config.iterations}] "
                        f"Loss = {total_loss_value:.6e}, "
                        f"|trans| = {grad_norm:.3e}, "
                        f"|tu| = {grad_tanu:.3e}, "
                        f"|tv| = {grad_tanv:.3e}, "
                        f"|su,sv| = {grad_scale:.3e}, "
                        f"|color| = {grad_color:.3e}, "
                        f"|opacity| = {grad_opacity:.3e}, "
                        f"|beta| = {grad_beta:.3e}, "
                        f"|pts| = {num_points}, "
                        f"t = {iteration_time:.3f} s"
                    )

                # Hotkey snapshot: use main camera for the image
                if poll_save_hotkey():
                    save_manual_snapshot(
                        config.output_dir,
                        iteration,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                        betas,
                        current_main_rgb_np,
                    )

        except KeyboardInterrupt:
            print(
                f"\nCtrl+C detected at iteration {iteration:04d}. "
                "Stopping optimization loop and saving current result..."
            )

    # ------------------------------------------------------------------
    # 12. Final render and export (multi-camera)
    # ------------------------------------------------------------------
    apply_point_parameters(
        renderer,
        positions,
        tangent_u,
        tangent_v,
        scales,
        colors,
        opacities,
        betas,
    )

    final_images = renderer.render_forward()

    final_loss = 0.0
    for camera_name in camera_ids:
        img_np = np.asarray(
            final_images[camera_name],
            dtype=np.float32,
            order="C",
        )
        tgt_np = target_images[camera_name]
        final_loss += compute_l2_loss(img_np, tgt_np)

        save_render(
            config.output_dir / f"render_final_{camera_name}.png",
            img_np,
        )

    save_positions_numpy(
        config.output_dir / "positions_final.npy",
        positions.detach().cpu().numpy(),
    )

    ply_path = config.output_dir / "points_final.ply"
    save_gaussians_to_ply(
        ply_path,
        positions,
        tangent_u,
        tangent_v,
        scales,
        colors,
        opacities,
        betas,
        shape_default=0.0,
    )
    print(f"Final parameters written to PLY: {ply_path}")

    print("\nOptimization completed.")
    print(f"Initial loss (sum over cameras): {initial_loss:.6e}")
    print(f"Final loss   (sum over cameras): {final_loss:.6e}")
    print(f"Outputs saved in: {config.output_dir.resolve()}")
