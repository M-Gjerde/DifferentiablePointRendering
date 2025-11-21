from __future__ import annotations

import os
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
)
from losses import compute_l2_loss, compute_l2_loss_and_grad
from optimizers import create_optimizer
from render_hooks import (
    fetch_initial_parameters,
    apply_point_parameters,
    orthonormalize_tangents_inplace,
    verify_scales_inplace,
    verify_colors_inplace,
)


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return arr / norms


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
    gt_position0 = initial_positions_np[0].copy()

    rng = np.random.default_rng(12)

    # Position noise
    noise_sigma_translation = 0.03
    noisy_positions_np = initial_positions_np.copy()
    noisy_positions_np += rng.normal(
        loc=0.0,
        scale=noise_sigma_translation,
        size=noisy_positions_np.shape,
    )
    initial_positions_np = noisy_positions_np.astype(np.float32)
    print("Initial positions perturbed by Gaussian noise:", noise_sigma_translation)

    # Tangent noise
    noise_sigma_tan = 0.03
    initial_tangent_u_np = initial_tangent_u_np.copy()
    initial_tangent_v_np = initial_tangent_v_np.copy()

    initial_tangent_u_np += rng.normal(
        0.0,
        noise_sigma_tan,
        initial_tangent_u_np.shape,
    )
    initial_tangent_v_np += rng.normal(
        0.0,
        noise_sigma_tan,
        initial_tangent_v_np.shape,
    )

    initial_tangent_u_np = normalize_rows(initial_tangent_u_np).astype(np.float32)
    initial_tangent_v_np = normalize_rows(initial_tangent_v_np).astype(np.float32)

    # Scale noise
    noise_sigma_scale = 0.04
    noisy_scale_np = initial_scale_np.copy()
    noisy_scale_np += rng.normal(
        0.0,
        noise_sigma_scale,
        initial_scale_np.shape,
    )
    initial_scale_np = noisy_scale_np.astype(np.float32)

    # Color noise (grayscale per-point)
    noise_sigma_color = 0.1
    rng = np.random.default_rng()
    color_np = initial_color_np.copy()
    gray_noise = rng.normal(0.0, noise_sigma_color, size=(initial_color_np.shape[0], 1))
    color_np += gray_noise
    initial_color_np = color_np.astype(np.float32)

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

    initial_param_mse0 = float(
        np.mean((initial_positions_np[0] - gt_position0) ** 2)
    )
    print(f"Initial position MSE (point 0): {initial_param_mse0:.6e}")

    iteration = 0
    try:
        for iteration in range(1, config.iterations + 1):
            apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, colors)

            current_rgb = renderer.render_forward()
            current_rgb_np = np.asarray(current_rgb, dtype=np.float32, order="C")

            loss_value, loss_grad_image, loss_image = compute_l2_loss_and_grad(
                current_rgb_np,
                target_rgb,
                return_loss_image=True,
            )

            gradients, grad_img = renderer.render_backward(loss_grad_image)

            grad_position_np = np.asarray(
                gradients["position"], dtype=np.float32, order="C"
            )
            grad_tangent_u_np = np.asarray(
                gradients["tangent_u"], dtype=np.float32, order="C"
            )
            grad_tangent_v_np = np.asarray(
                gradients["tangent_v"], dtype=np.float32, order="C"
            )
            grad_scales_np = np.asarray(
                gradients["scale"], dtype=np.float32, order="C"
            )
            grad_colors_np = np.asarray(
                gradients["color"], dtype=np.float32, order="C"
            )

            if grad_position_np.shape != initial_positions_np.shape:
                raise RuntimeError(
                    f"Gradient shape mismatch for position: expected {initial_positions_np.shape}, "
                    f"got {grad_position_np.shape}"
                )
            if grad_tangent_u_np.shape != initial_tangent_u_np.shape:
                raise RuntimeError(
                    f"Gradient shape mismatch for tangent_u: expected {initial_tangent_u_np.shape}, "
                    f"got {grad_tangent_u_np.shape}"
                )
            if grad_tangent_v_np.shape != initial_tangent_v_np.shape:
                raise RuntimeError(
                    f"Gradient shape mismatch for tangent_v: expected {initial_tangent_v_np.shape}, "
                    f"got {grad_tangent_v_np.shape}"
                )

            optimizer.zero_grad(set_to_none=True)
            positions.grad = torch.tensor(grad_position_np, device=device, dtype=torch.float32)
            tangent_u.grad = torch.tensor(grad_tangent_u_np, device=device, dtype=torch.float32)
            tangent_v.grad = torch.tensor(grad_tangent_v_np, device=device, dtype=torch.float32)
            scales.grad = torch.tensor(grad_scales_np, device=device, dtype=torch.float32)
            colors.grad = torch.tensor(grad_colors_np, device=device, dtype=torch.float32)
            optimizer.step()

            orthonormalize_tangents_inplace(tangent_u, tangent_v)
            verify_scales_inplace(scales)
            verify_colors_inplace(colors)

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
                    f"|color| = {grad_color:.3e}"
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

    except KeyboardInterrupt:
        print(
            f"\nCtrl+C detected at iteration {iteration:04d}. "
            "Stopping optimization loop and saving current result..."
        )

    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales, colors)
    final_rgb = renderer.render_forward()
    final_rgb_np = np.asarray(final_rgb, dtype=np.float32, order="C")
    final_loss = compute_l2_loss(final_rgb_np, target_rgb)

    save_render(config.output_dir / "render_final.png", final_rgb_np)
    save_positions_numpy(
        config.output_dir / "positions_final.npy",
        positions.detach().cpu().numpy(),
    )

    print("\nOptimization completed.")
    print(f"Initial loss: {initial_loss:.6e}")
    print(f"Final loss:   {final_loss:.6e}")
    print(f"Outputs saved in: {config.output_dir.resolve()}")
