from __future__ import annotations

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
from losses import compute_l2_loss, compute_l2_loss_and_grad
from optimizers import create_optimizer
from density_control import densify_and_prune_points
from render_hooks import (
    fetch_initial_parameters,
    apply_point_parameters,
    orthonormalize_tangents_inplace,
    verify_scales_inplace,
    verify_colors_inplace,
)
from debug_init_utils import add_debug_noise_to_initial_parameters

from repulsion import compute_elliptical_repulsion_loss


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
    initial_params = fetch_initial_parameters(renderer)
    initial_positions_np = initial_params["position"]
    initial_tangent_u_np = initial_params["tangent_u"]
    initial_tangent_v_np = initial_params["tangent_v"]
    initial_scale_np = initial_params["scale"]
    initial_color_np = initial_params["color"]

    num_points = initial_positions_np.shape[0]
    print(f"Fetched {num_points} initial points from renderer.")

    # --- Debug noise injection (can be removed later) ---
    #(
    #    initial_positions_np,
    #    initial_tangent_u_np,
    #    initial_tangent_v_np,
    #    initial_scale_np,
    #    initial_color_np,
    #) = add_debug_noise_to_initial_parameters(
    #    initial_positions_np,
    #    initial_tangent_u_np,
    #    initial_tangent_v_np,
    #    initial_scale_np,
    #    initial_color_np,
    #)
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

    optimizer = create_optimizer(
        config,
        positions,
        tangent_u,
        tangent_v,
        scales,
        colors,
    )

    # --- Initial snapshot ---
    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, colors)
    orthonormalize_tangents_inplace(tangent_u, tangent_v)
    verify_scales_inplace(scales)
    verify_colors_inplace(colors)

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
    densification_interval = 20
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

            # Optional: sanity check vs CURRENT parameter shapes
            current_positions_shape = tuple(positions.shape)
            if grad_position_np.shape != current_positions_shape:
                raise RuntimeError(
                    f"Gradient shape mismatch for position: expected {current_positions_shape}, "
                    f"got {grad_position_np.shape}"
                )

            # 4) Apply gradients to current parameters
            optimizer.zero_grad(set_to_none=True)
            # Renderer gradients (from your custom adjoint)
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

            # --- Elliptical repulsion term (on positions only) ---------------
            repulsion_loss = compute_elliptical_repulsion_loss(
                positions=positions,
                tangent_u=tangent_u,
                tangent_v=tangent_v,
                scales=scales,
                radius_factor=2.0,  # ~ footprint size in uv; tune
                repulsion_weight=1e-2,  # strength; tune
            )

            # This will ADD to positions.grad (since we already set it),
            # but won't touch tangent/scales/colors because we detach them.
            if repulsion_loss.item() != 0.0:
                repulsion_loss.backward()

            optimizer.step()

            # 5) Reparameterization
            orthonormalize_tangents_inplace(tangent_u, tangent_v)
            verify_scales_inplace(scales)
            verify_colors_inplace(colors)

            # 6) Densification / pruning AFTER step
            perform_density_update = (iteration >= 5 and iteration % densification_interval == 1)

            if perform_density_update:
                (
                    positions,
                    tangent_u,
                    tangent_v,
                    scales,
                    colors,
                    optimizer,
                    num_points,
                    did_change_topology,
                ) = densify_and_prune_points(
                    positions=positions,
                    tangent_u=tangent_u,
                    tangent_v=tangent_v,
                    scales=scales,
                    colors=colors,
                    grad_position_np=grad_position_np,
                    grad_scales_np=grad_scales_np,
                    optimizer=optimizer,
                    config=config,
                    device=device,
                )
            else:
                did_change_topology = False

            # 7) Upload updated parameters (and rebuild if topology changed)
            apply_point_parameters(
                renderer,
                positions,
                tangent_u,
                tangent_v,
                scales,
                colors,
                did_change_topology,
            )

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

            if iteration % config.log_interval == 0 or iteration == 1:
                grad_norm = float(np.linalg.norm(grad_position_np) / max(num_points, 1))
                grad_tanu = float(np.linalg.norm(grad_tangent_u_np) / max(num_points, 1))
                grad_tanv = float(np.linalg.norm(grad_tangent_v_np) / max(num_points, 1))
                grad_scale = float(np.linalg.norm(grad_scales_np) / max(num_points, 1))
                grad_color = float(np.linalg.norm(grad_colors_np) / max(num_points, 1))
                print(
                    f"[Iter {iteration:04d}/{config.iterations}] "
                    f"Loss = {loss_value:.6e}, "
                    f"|translation| = {grad_norm:.3e}, "
                    f"|tan_u| = {grad_tanu:.3e}, "
                    f"|tan_v| = {grad_tanv:.3e}, "
                    f"|scale| = {grad_scale:.3e}, "
                    f"|color| = {grad_color:.3e}, "
                    f"|Points| = {num_points}, "
                    f"Time = {iteration_time:.3f} s"
                )

    except KeyboardInterrupt:
        print(
            f"\nCtrl+C detected at iteration {iteration:04d}. "
            "Stopping optimization loop and saving current result..."
        )

    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, colors, False)
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
        opacity_default=1.0,
        beta_default=0.0,
        shape_default=0.0,
    )
    print(f"Final parameters written to PLY: {ply_path}")

    print("\nOptimization completed.")
    print(f"Initial loss: {initial_loss:.6e}")
    print(f"Final loss:   {final_loss:.6e}")
    print(f"Outputs saved in: {config.output_dir.resolve()}")
