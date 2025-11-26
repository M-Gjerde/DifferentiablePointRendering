from __future__ import annotations

import csv
import os
import time
from pathlib import Path

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
    save_gaussians_to_ply
)
from losses import (
    compute_l2_loss,
    compute_l2_loss_and_grad,
    compute_parameter_mse,
)

from optimizers import create_optimizer
from density_control import (densify_points_long_axis_split, compute_prune_indices_by_opacity)
from render_hooks import remove_points
from render_hooks import (
    fetch_parameters,
    apply_point_parameters,
    orthonormalize_tangents_inplace,
    verify_scales_inplace,
    verify_colors_inplace,
    verify_opacities_inplace,
    add_new_points,
    rebuild_bvh
)
from debug_init_utils import add_debug_noise_to_initial_parameters

from repulsion import compute_elliptical_repulsion_loss
import sys
import select


def poll_save_hotkey() -> bool:
    """
    Non-blocking check for a single-line keyboard input.
    Returns True if the user typed 's' (case-insensitive) and pressed Enter.
    """
    if not sys.stdin.isatty():
        return False

    # select with timeout=0 -> non-blocking
    rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not rlist:
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
        beta_default=0.0,
        shape_default=0.0,
    )
    print(
        f"[Iter {iteration:04d}] Hotkey 's' pressed -> "
        f"saved render_final.png, positions_final.npy and points_final.ply"
    )


def clear_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                for item2 in item.iterdir():
                    if item2.is_file():
                        item2.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def run_optimization(
        config: OptimizationConfig,
        renderer_settings: RendererSettingsConfig,
) -> None:
    # --- Initialize renderer ---
    renderer = pale.Renderer(
        str(config.assets_root),
        config.scene_xml,
        config.pointcloud_ply,
        renderer_settings.as_dict(),
    )

    # --- Load target image ---
    target_rgb = load_target_image(config.target_image_path)
    print(f"Loaded target image: {config.target_image_path} with shape {target_rgb.shape}")

    # --- Fetch initial parameters from renderer ---
    initial_params = fetch_parameters(renderer)
    initial_positions_np = initial_params["position"]
    initial_tangent_u_np = initial_params["tangent_u"]
    initial_tangent_v_np = initial_params["tangent_v"]
    initial_scale_np = initial_params["scale"]
    initial_color_np = initial_params["color"]
    initial_opacity_np = initial_params["opacity"]

    num_points = initial_positions_np.shape[0]
    print(f"Fetched {num_points} initial points from PLY.")

    # For parameter-MSE reference, keep an immutable copy
    initial_params_reference = {
        "position": initial_positions_np.copy(),
        "tangent_u": initial_tangent_u_np.copy(),
        "tangent_v": initial_tangent_v_np.copy(),
        "scale": initial_scale_np.copy(),
        "color": initial_color_np.copy(),
        "opacity": initial_opacity_np.copy(),
    }

    # --- Debug noise injection (can be removed later) ---
    # (initial_positions_np,
    # initial_tangent_u_np,
    # initial_tangent_v_np,
    # initial_scale_np,
    # initial_color_np,
    # initial_opacity_np) = add_debug_noise_to_initial_parameters(
    #    initial_positions_np,
    #    initial_tangent_u_np,
    #    initial_tangent_v_np,
    #    initial_scale_np,
    #    initial_color_np,
    #    initial_opacity_np)
    # print("Initial parameters perturbed by debug Gaussian noise.")

    device = torch.device(config.device)

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

    # --- Initial snapshot ---
    verify_scales_inplace(scales)
    verify_colors_inplace(colors)
    verify_opacities_inplace(opacities)
    orthonormalize_tangents_inplace(tangent_u, tangent_v)

    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, colors, opacities)
    rebuild_bvh(renderer)
    # Re-fetch parameters from renderer to get canonical ordering
    updated_parameters = fetch_parameters(renderer)
    new_positions_np = updated_parameters["position"]
    new_tangent_u_np = updated_parameters["tangent_u"]
    new_tangent_v_np = updated_parameters["tangent_v"]
    new_scale_np = updated_parameters["scale"]
    new_color_np = updated_parameters["color"]
    new_opacity_np = updated_parameters["opacity"]
    positions = torch.nn.Parameter(
        torch.tensor(new_positions_np, device=device, dtype=torch.float32)
    )
    tangent_u = torch.nn.Parameter(
        torch.tensor(new_tangent_u_np, device=device, dtype=torch.float32)
    )
    tangent_v = torch.nn.Parameter(
        torch.tensor(new_tangent_v_np, device=device, dtype=torch.float32)
    )
    scales = torch.nn.Parameter(
        torch.tensor(new_scale_np, device=device, dtype=torch.float32)
    )
    colors = torch.nn.Parameter(
        torch.tensor(new_color_np, device=device, dtype=torch.float32)
    )
    opacities = torch.nn.Parameter(
        torch.tensor(new_opacity_np, device=device, dtype=torch.float32)
    )
    optimizer = create_optimizer(
        config,
        positions,
        tangent_u,
        tangent_v,
        scales,
        colors,
        opacities
    )

    initial_rgb = renderer.render_forward()
    initial_rgb_np = np.asarray(initial_rgb, dtype=np.float32, order="C")
    initial_loss, _ = compute_l2_loss_and_grad(
        initial_rgb_np,
        target_rgb,
    )
    print(f"Initial loss (L2): {initial_loss:.6e}")

    clear_output_dir(config.output_dir)

    save_render(config.output_dir / "render_initial.png", initial_rgb_np)
    save_render(config.output_dir / "render_target.png", target_rgb)
    save_positions_numpy(config.output_dir / "positions_initial.npy", initial_positions_np)

    iteration = 0
    # Heuristics for density control:
    # - Start after a short burn-in so geometry and lighting stabilize a bit.
    # - Densify less often than you optimize.
    # - Prune slightly more often than you densify.
    densification_interval = 10  # densify every 100 iterations
    prune_interval = 5  # prune every 50 iterations
    burnin_iterations = 5  # do not touch topology before this
    reset_opacity_interval = 200  # do not touch topology before this

    # Opacity-based pruning parameters
    opacity_prune_threshold = 0.5  # prune lowest 15% opacities (quantile mode)
    max_prune_fraction = 0.3  # never prune more than 30% in one step

    rebuild_bvh_interval = 1  # keep this if you want per-iter rebuild
    metrics_csv_path = config.output_dir / "metrics.csv"
    clear_output_dir(config.output_dir)

    # initial renders and saves (unchanged)
    save_render(config.output_dir / "render_initial.png", initial_rgb_np)
    save_render(config.output_dir / "render_target.png", target_rgb)
    save_positions_numpy(config.output_dir / "positions_initial.npy", initial_positions_np)

    with open(metrics_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "iteration",
            "loss_l2",
            "parameter_mse",
            "num_points",
            "iteration_time_sec",
        ])

        try:
            for iteration in range(1, config.iterations + 1):
                iteration_start = time.perf_counter()

                # 1) Forward
                current_rgb = renderer.render_forward()
                current_rgb_np = np.asarray(current_rgb, dtype=np.float32, order="C")

                # 2) Loss + backward image
                loss_value, loss_grad_image, loss_image = compute_l2_loss_and_grad(
                    current_rgb_np,
                    target_rgb,
                    return_loss_image=True,
                )

                gradients, grad_img = renderer.render_backward(loss_grad_image)

                # 3) Extract gradients
                grad_position_np = np.asarray(gradients["position"], dtype=np.float32, order="C")
                grad_tangent_u_np = np.asarray(gradients["tangent_u"], dtype=np.float32, order="C")
                grad_tangent_v_np = np.asarray(gradients["tangent_v"], dtype=np.float32, order="C")
                grad_scales_np = np.asarray(gradients["scale"], dtype=np.float32, order="C")
                grad_colors_np = np.asarray(gradients["color"], dtype=np.float32, order="C")
                grad_opacities_np = np.asarray(gradients["opacity"], dtype=np.float32, order="C")

                # Optional: sanity check vs CURRENT parameter shapes
                current_positions_shape = tuple(positions.shape)
                if grad_position_np.shape != current_positions_shape:
                    raise RuntimeError(
                        f"Gradient shape mismatch for position: expected {current_positions_shape}, "
                        f"got {grad_position_np.shape}"
                    )

                # 4) Apply gradients to current parameters
                optimizer.zero_grad(set_to_none=True)
                positions.grad = torch.tensor(
                    grad_position_np, device=device, dtype=torch.float32
                )
                tangent_u.grad = torch.tensor(
                    grad_tangent_u_np, device=device, dtype=torch.float32
                )
                tangent_v.grad = torch.tensor(
                    grad_tangent_v_np, device=device, dtype=torch.float32
                )
                scales.grad = torch.tensor(
                    grad_scales_np, device=device, dtype=torch.float32
                )
                colors.grad = torch.tensor(
                    grad_colors_np, device=device, dtype=torch.float32
                )
                opacities.grad = torch.tensor(
                    grad_opacities_np, device=device, dtype=torch.float32
                )

                # --- Elliptical repulsion term (on positions only) ---------------
                repulsion_loss = compute_elliptical_repulsion_loss(
                    positions=positions,
                    tangent_u=tangent_u,
                    tangent_v=tangent_v,
                    scales=scales,
                    radius_factor=1.0,  # 1 surfel-radius in the normalization
                    repulsion_weight=1e-2,  # tune relative to image loss
                    contact_distance=2.0,  # ~no overlap at d_hat >= 2
                )

                if torch.isfinite(repulsion_loss) and repulsion_loss.item() != 0.0:
                    repulsion_loss.backward()

                optimizer.step()

                if iteration % reset_opacity_interval == 0:
                    with torch.no_grad():
                        opacities[:] = 0.1
                    print(f"[Iter {iteration:04d}] Resetting all opacities to 0.1")

                # 5) Reparameterization
                orthonormalize_tangents_inplace(tangent_u, tangent_v)
                verify_scales_inplace(scales)
                verify_colors_inplace(colors)
                verify_opacities_inplace(opacities)

                # 6) Upload updated parameters (and rebuild if topology changed)
                apply_point_parameters(
                    renderer,
                    positions,
                    tangent_u,
                    tangent_v,
                    scales,
                    colors,
                    opacities
                )

                if iteration % rebuild_bvh_interval == 0:
                    rebuild_bvh(renderer)
                    # Re-fetch parameters from renderer to get canonical ordering
                    updated_parameters = fetch_parameters(renderer)
                    new_positions_np = updated_parameters["position"]
                    new_tangent_u_np = updated_parameters["tangent_u"]
                    new_tangent_v_np = updated_parameters["tangent_v"]
                    new_scale_np = updated_parameters["scale"]
                    new_color_np = updated_parameters["color"]
                    new_opacity_np = updated_parameters["opacity"]

                    positions = torch.nn.Parameter(
                        torch.tensor(new_positions_np, device=device, dtype=torch.float32)
                    )
                    tangent_u = torch.nn.Parameter(
                        torch.tensor(new_tangent_u_np, device=device, dtype=torch.float32)
                    )
                    tangent_v = torch.nn.Parameter(
                        torch.tensor(new_tangent_v_np, device=device, dtype=torch.float32)
                    )
                    scales = torch.nn.Parameter(
                        torch.tensor(new_scale_np, device=device, dtype=torch.float32)
                    )
                    colors = torch.nn.Parameter(
                        torch.tensor(new_color_np, device=device, dtype=torch.float32)
                    )
                    opacities = torch.nn.Parameter(
                        torch.tensor(new_opacity_np, device=device, dtype=torch.float32)
                    )
                    optimizer = create_optimizer(
                        config,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities
                    )

                # 7) Densification + pruning AFTER step, at interval
                densification_result = None
                indices_to_remove_list: list[int] = []

                if iteration >= burnin_iterations and iteration % densification_interval == 0:
                    # Build EDC-like importance from current gradients
                    grad_pos_norm = np.linalg.norm(grad_position_np, axis=1)
                    grad_scale_norm = np.linalg.norm(grad_scales_np, axis=1)
                    grad_opacity_abs = np.abs(grad_opacities_np).reshape(-1)

                    importance_np = (
                            grad_pos_norm +
                            0.3 * grad_scale_norm +
                            0.1 * grad_opacity_abs
                    )

                    importance_np = np.asarray(importance_np, dtype=np.float32)  # final guarantee
                    importance_tensor = torch.from_numpy(importance_np)

                    # Reasonable budget, e.g. at most 20% new points
                    max_new_points = max(int(0.2 * positions.shape[0]), int(5))  # or a fixed number

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
                    # 1) Remove all chosen indices (split parents + low-opacity points)
                    if indices_to_remove_list:
                        indices_to_remove = np.unique(
                            np.asarray(indices_to_remove_list, dtype=np.int64)
                        )
                        remove_points(renderer, indices_to_remove)
                        rebuild_bvh(renderer)
                    # 2) Add children from long-axis split (if any)
                    if densification_result is not None:
                        add_new_points(renderer, densification_result)

                    # 3) Re-fetch parameters from renderer to get canonical ordering,
                    #    now including the newly appended children and without pruned ones.
                    updated_parameters = fetch_parameters(renderer)
                    new_positions_np = updated_parameters["position"]
                    new_tangent_u_np = updated_parameters["tangent_u"]
                    new_tangent_v_np = updated_parameters["tangent_v"]
                    new_scale_np = updated_parameters["scale"]
                    new_color_np = updated_parameters["color"]
                    new_opacity_np = updated_parameters["opacity"]

                    positions = torch.nn.Parameter(
                        torch.tensor(new_positions_np, device=device, dtype=torch.float32)
                    )
                    tangent_u = torch.nn.Parameter(
                        torch.tensor(new_tangent_u_np, device=device, dtype=torch.float32)
                    )
                    tangent_v = torch.nn.Parameter(
                        torch.tensor(new_tangent_v_np, device=device, dtype=torch.float32)
                    )
                    scales = torch.nn.Parameter(
                        torch.tensor(new_scale_np, device=device, dtype=torch.float32)
                    )
                    colors = torch.nn.Parameter(
                        torch.tensor(new_color_np, device=device, dtype=torch.float32)
                    )
                    opacities = torch.nn.Parameter(
                        torch.tensor(new_opacity_np, device=device, dtype=torch.float32)
                    )

                    # 4) Reparameterization on the new tensor state
                    orthonormalize_tangents_inplace(tangent_u, tangent_v)
                    verify_scales_inplace(scales)
                    verify_colors_inplace(colors)
                    verify_opacities_inplace(opacities)

                    # 5) Push back to renderer
                    apply_point_parameters(
                        renderer,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                    )

                    # 6) Rebuild BVH (topology changed) and recreate optimizer
                    rebuild_bvh(renderer)
                    optimizer = create_optimizer(
                        config,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities,
                    )

                # snapshots (unchanged)
                if iteration % config.save_interval == 0 or iteration == config.iterations:
                    snapshot_rgb_np = current_rgb_np
                    render_path = config.output_dir / "render" / f"{iteration:04d}_render.png"
                    save_render(render_path, snapshot_rgb_np)

                    img = np.asarray(grad_img, dtype=np.float32, order="C")
                    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

                    render_grad_quantile = (
                            config.output_dir / "grad" / f"{iteration:04d}_grad_099.png"
                    )
                    save_gradient_sign_png_py(
                        render_grad_quantile,
                        img,
                        adjoint_spp=renderer_settings.adjoint_passes,
                        abs_quantile=0.999,
                        flip_y=False,
                    )

                    save_loss_image(config.output_dir, loss_image, iteration)

                num_points = positions.shape[0]
                iteration_end = time.perf_counter()
                iteration_time = iteration_end - iteration_start

                # --- Parameter MSE vs. initial parameters -----------------------
                current_params_np = {
                    "position": np.array(positions.detach().cpu().numpy()),
                    "tangent_u": np.array(tangent_u.detach().cpu().numpy()),
                    "tangent_v": np.array(tangent_v.detach().cpu().numpy()),
                    "scale": np.array(scales.detach().cpu().numpy()),
                    "color": np.array(colors.detach().cpu().numpy()),
                    "opacity": np.array(opacities.detach().cpu().numpy()),
                }
                parameter_mse = compute_parameter_mse(
                    current_params_np,
                    initial_params_reference,
                )

                # --- Write CSV row ----------------------------------------------
                csv_writer.writerow([
                    iteration,
                    loss_value,
                    parameter_mse,
                    num_points,
                    iteration_time,
                ])
                csv_file.flush()

                # --- Console logging (unchanged, now uses num_points, iteration_time) ---
                if iteration % config.log_interval == 0 or iteration == 1:
                    grad_norm = float(np.linalg.norm(grad_position_np) / max(num_points, 1))
                    grad_tanu = float(np.linalg.norm(grad_tangent_u_np) / max(num_points, 1))
                    grad_tanv = float(np.linalg.norm(grad_tangent_v_np) / max(num_points, 1))
                    grad_scale = float(np.linalg.norm(grad_scales_np) / max(num_points, 1))
                    grad_color = float(np.linalg.norm(grad_colors_np) / max(num_points, 1))
                    grad_opacity = float(np.linalg.norm(grad_opacities_np) / max(num_points, 1))
                    print(
                        f"[Iter {iteration:04d}/{config.iterations}] "
                        f"Loss = {loss_value:.6e}, "
                        f"|trans| = {grad_norm:.3e}, "
                        f"|tu| = {grad_tanu:.3e}, "
                        f"|tv| = {grad_tanv:.3e}, "
                        f"|su,sv| = {grad_scale:.3e}, "
                        f"|color| = {grad_color:.3e}, "
                        f"|opacity| = {grad_opacity:.3e}, "
                        f"|pts| = {num_points}, "
                        f"t = {iteration_time:.3f} s"
                    )
                # --- Hotkey: press 's' + Enter in the terminal to save current state ---
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
                        current_rgb_np,
                    )

        except KeyboardInterrupt:
            print(
                f"\nCtrl+C detected at iteration {iteration:04d}. "
                "Stopping optimization loop and saving current result..."
            )

    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, colors, opacities)
    final_rgb = renderer.render_forward()
    final_rgb_np = np.asarray(final_rgb, dtype=np.float32, order="C")
    final_loss = compute_l2_loss(final_rgb_np, target_rgb)

    save_render(config.output_dir / "render_final.png", final_rgb_np)
    save_positions_numpy(
        config.output_dir / "positions_final.npy",
        positions.detach().cpu().numpy(),
    )

    # --- Save full final parameter set as ASCII PLY for inspection ---
    ply_path = config.output_dir / "points_final.ply"
    save_gaussians_to_ply(
        ply_path,
        positions,
        tangent_u,
        tangent_v,
        scales,
        colors,
        opacities,
        beta_default=0.0,
        shape_default=0.0,
    )
    print(f"Final parameters written to PLY: {ply_path}")

    print("\nOptimization completed.")
    print(f"Initial loss: {initial_loss:.6e}")
    print(f"Final loss:   {final_loss:.6e}")
    print(f"Outputs saved in: {config.output_dir.resolve()}")
