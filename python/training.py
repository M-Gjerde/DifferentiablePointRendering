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

    # Save positions as numpy
    save_positions_numpy(
        output_dir / "positions_final.npy",
        positions.detach().cpu().numpy(),
    )

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
        f"saved render_final.png, positions_final.npy and points_final.ply"
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
        config: OptimizationConfig,
        renderer_settings: RendererSettingsConfig,
) -> None:
    # ------------------------------------------------------------------
    # 1. Initialize renderer and target
    # ------------------------------------------------------------------
    renderer = pale.Renderer(
        str(config.assets_root),
        config.scene_xml,
        config.pointcloud_ply,
        renderer_settings.as_dict(),
    )

    target_rgb = load_target_image(config.target_image_path)
    print(f"Loaded target image: {config.target_image_path} with shape {target_rgb.shape}")

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

    #Optional debug noise
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

    # Create initial trainable tensors from numpy
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

    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, colors, opacities, betas)
    rebuild_bvh(renderer)

    # Ensure canonical ordering from renderer
    (
        positions,
        tangent_u,
        tangent_v,
        scales,
        colors,
        opacities,
        betas
    ) = refetch_parameters_as_torch(renderer, device)

    optimizer = create_optimizer(
        config,
        positions,
        tangent_u,
        tangent_v,
        scales,
        colors,
        opacities,
        betas
    )

    # ------------------------------------------------------------------
    # 3. Initial loss and output dir setup
    # ------------------------------------------------------------------
    initial_rgb = renderer.render_forward()
    initial_rgb_np = np.asarray(initial_rgb, dtype=np.float32, order="C")
    initial_loss, _ = compute_l2_loss_and_grad(initial_rgb_np, target_rgb)
    print(f"Initial loss (L2): {initial_loss:.6e}")

    clear_output_dir(config.output_dir)

    save_render(config.output_dir / "render_initial.png", initial_rgb_np)
    save_render(config.output_dir / "render_target.png", target_rgb)
    save_positions_numpy(config.output_dir / "positions_initial.npy", initial_positions_np)

    # ------------------------------------------------------------------
    # 4. Density control / scheduling hyperparameters
    # ------------------------------------------------------------------
    iteration = 0

    # Density control settings (currently effectively disabled with large intervals)
    densification_interval = int(1e10)
    prune_interval = int(1e10)
    burnin_iterations = 5
    reset_opacity_interval = int(1e10)

    # Opacity-based pruning parameters
    opacity_prune_threshold = 0.5
    max_prune_fraction = 0.3

    # BVH rebuild schedule
    rebuild_bvh_interval = 1

    # Metrics logging
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
                # 5. Forward pass and image-space loss
                # --------------------------------------------------------------
                current_rgb = renderer.render_forward()
                current_rgb_np = np.asarray(current_rgb, dtype=np.float32, order="C")

                loss_value, loss_grad_image, loss_image = compute_l2_loss_and_grad(
                    current_rgb_np,
                    target_rgb,
                    return_loss_image=True,
                )

                gradients, grad_img = renderer.render_backward(loss_grad_image)

                # Extract numpy gradients
                grad_position_np = np.asarray(gradients["position"], dtype=np.float32, order="C")
                grad_tangent_u_np = np.asarray(gradients["tangent_u"], dtype=np.float32, order="C")
                grad_tangent_v_np = np.asarray(gradients["tangent_v"], dtype=np.float32, order="C")
                grad_scales_np = np.asarray(gradients["scale"], dtype=np.float32, order="C")
                grad_colors_np = np.asarray(gradients["color"], dtype=np.float32, order="C")
                grad_opacities_np = np.asarray(gradients["opacity"], dtype=np.float32, order="C")
                grad_betas_np = np.asarray(gradients["beta"], dtype=np.float32, order="C")

                # Sanity check shapes
                current_positions_shape = tuple(positions.shape)
                if grad_position_np.shape != current_positions_shape:
                    raise RuntimeError(
                        f"Gradient shape mismatch for position: expected {current_positions_shape}, "
                        f"got {grad_position_np.shape}"
                    )

                # --------------------------------------------------------------
                # 6. Optimizer step
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

                # Optional repulsion term (currently disabled)
                # repulsion_loss = compute_elliptical_repulsion_loss(
                #     positions=positions,
                #     tangent_u=tangent_u,
                #     tangent_v=tangent_v,
                #     scales=scales,
                #     radius_factor=1.0,
                #     repulsion_weight=1e-2,
                #     contact_distance=2.0,
                # )
                # if torch.isfinite(repulsion_loss) and repulsion_loss.item() != 0.0:
                #     repulsion_loss.backward()

                optimizer.step()

                # Reset opacities on schedule
                if iteration % reset_opacity_interval == 0:
                    with torch.no_grad():
                        opacities[:] = 0.1
                    print(f"[Iter {iteration:04d}] Resetting all opacities to 0.1")

                # --------------------------------------------------------------
                # 7. Reparameterization and sync
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
                    betas
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
                        betas
                    )

                # --------------------------------------------------------------
                # 8. Densification + pruning
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
                        importance=importance_tensor,
                        max_new_points=max_new_points,
                        split_distance=0.4,
                        minor_axis_shrink=0.85,
                        opacity_reduction=0.8,
                        min_long_axis_scale=0.03,
                        grad_threshold=1e-4,
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
                    # 1) Remove points
                    if indices_to_remove_list:
                        indices_to_remove = np.unique(
                            np.asarray(indices_to_remove_list, dtype=np.int64)
                        )
                        remove_points(renderer, indices_to_remove)
                        rebuild_bvh(renderer)

                    # 2) Add children from long-axis split
                    if densification_result is not None:
                        add_new_points(renderer, densification_result)

                    # 3) Refetch parameters in canonical order
                    (
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                        betas
                    ) = refetch_parameters_as_torch(renderer, device)

                    # 4) Reparameterize and sync
                    verify_parameters_inplane(tangent_u, tangent_v, scales, colors, opacities, betas)
                    apply_point_parameters(
                        renderer,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                        betas
                    )

                    # 5) Rebuild BVH and recreate optimizer
                    rebuild_bvh(renderer)
                    optimizer = create_optimizer(
                        config,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                        betas
                    )

                # --------------------------------------------------------------
                # 9. Snapshots
                # --------------------------------------------------------------
                if iteration % config.save_interval == 0 or iteration == config.iterations:
                    render_path = config.output_dir / "render" / f"{iteration:04d}_render.png"
                    save_render(render_path, current_rgb_np)

                    grad_img_np = np.asarray(grad_img, dtype=np.float32, order="C")
                    grad_img_np = np.nan_to_num(grad_img_np, nan=0.0, posinf=0.0, neginf=0.0)

                    render_grad_quantile = (
                            config.output_dir / "grad" / f"{iteration:04d}_grad_099.png"
                    )
                    save_gradient_sign_png_py(
                        render_grad_quantile,
                        grad_img_np,
                        adjoint_spp=renderer_settings.adjoint_passes,
                        abs_quantile=0.999,
                        flip_y=False,
                    )

                    save_loss_image(config.output_dir, loss_image, iteration)

                # --------------------------------------------------------------
                # 10. Metrics and logging
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
                    [iteration, loss_value, parameter_mse, num_points, iteration_time]
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
                        f"Loss = {loss_value:.6e}, "
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

                # Hotkey snapshot
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
                        current_rgb_np,
                    )

        except KeyboardInterrupt:
            print(
                f"\nCtrl+C detected at iteration {iteration:04d}. "
                "Stopping optimization loop and saving current result..."
            )

    # ------------------------------------------------------------------
    # 11. Final render and export
    # ------------------------------------------------------------------
    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, colors, opacities, betas)
    final_rgb = renderer.render_forward()
    final_rgb_np = np.asarray(final_rgb, dtype=np.float32, order="C")
    final_loss = compute_l2_loss(final_rgb_np, target_rgb)

    save_render(config.output_dir / "render_final.png", final_rgb_np)
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
    print(f"Initial loss: {initial_loss:.6e}")
    print(f"Final loss:   {final_loss:.6e}")
    print(f"Outputs saved in: {config.output_dir.resolve()}")
