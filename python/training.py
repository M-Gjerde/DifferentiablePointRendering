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
from density_control import densify_points
from render_hooks import (
    fetch_parameters,
    apply_point_parameters,
    orthonormalize_tangents_inplace,
    verify_scales_inplace,
    verify_colors_inplace,
    verify_opacities_inplace,
    apply_new_points,
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
    (initial_positions_np,
     initial_tangent_u_np,
     initial_tangent_v_np,
     initial_scale_np,
     initial_color_np,
     initial_opacity_np) = add_debug_noise_to_initial_parameters(
        initial_positions_np,
        initial_tangent_u_np,
        initial_tangent_v_np,
        initial_scale_np,
        initial_color_np,
        initial_opacity_np)
    print("Initial parameters perturbed by debug Gaussian noise.")

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
    densification_interval = 1e100
    rebuild_bvh_interval = 5

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

                optimizer.step()

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

                # 7) Densification AFTER step, at interval
                perform_density_update = (iteration % densification_interval == 0)
                if perform_density_update:
                    densification_result = densify_points(
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                    )
                else:
                    densification_result = None

                if densification_result is not None:
                    # 7a) Append new points on the C++ side (children only)
                    apply_new_points(renderer, densification_result)

                    # 7b) Re-fetch parameters from renderer to get canonical ordering,
                    #     now including the newly appended children.
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

                    # 7c) Apply parent updates from densification in Python
                    updated_block = densification_result.get("updated")
                    if updated_block is not None:
                        # updated_block["indices"]: 1D array of parent indices
                        # updated_block["scale"]:   (K,2) array of new scales for those parents
                        parent_indices = torch.as_tensor(
                            updated_block["indices"],
                            device=device,
                            dtype=torch.long,
                        )
                        parent_scales = torch.as_tensor(
                            updated_block["scale"],
                            device=device,
                            dtype=torch.float32,
                        )  # shape (K, 2)

                        # Overwrite selected parent scales in the tensor
                        scales.data[parent_indices] = parent_scales

                    # 7d) Reparameterization on the new tensor state
                    orthonormalize_tangents_inplace(tangent_u, tangent_v)
                    verify_scales_inplace(scales)
                    verify_colors_inplace(colors)
                    verify_opacities_inplace(opacities)

                    # Apply the changed points
                    apply_point_parameters(
                        renderer,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities
                    )

                    # 7e) Recreate optimizer for the new parameter set (N has changed)
                    optimizer = create_optimizer(
                        config,
                        positions,
                        tangent_u,
                        tangent_v,
                        scales,
                        colors,
                        opacities
                    )

                # snapshots (unchanged)
                if iteration % config.save_interval == 0 or iteration == config.iterations:
                    snapshot_rgb_np = current_rgb_np
                    render_path = config.output_dir / "render" / f"render_iter_{iteration:04d}.png"
                    save_render(render_path, snapshot_rgb_np)

                    img = np.asarray(grad_img, dtype=np.float32, order="C")
                    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

                    render_grad_quantile = (
                            config.output_dir / "grad" / f"render_iter_grad_099_{iteration:04d}.png"
                    )
                    save_gradient_sign_png_py(
                        render_grad_quantile,
                        img,
                        adjoint_spp=8,
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
