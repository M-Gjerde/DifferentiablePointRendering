#!/usr/bin/env python3
"""
optimize_points.py

Drive optimization of point parameters using a custom differentiable renderer
(`pale`) and a PyTorch optimizer. The renderer is responsible for both the
forward render and gradient computation; PyTorch is used only to manage
parameters and optimization algorithms (Adam, SGD, etc.).

Typical usage:

    python optimize_points.py \
        --assets-root ../Assets \
        --scene-xml cbox_custom.xml \
        --pointcloud initial.ply \
        --target-image ../Assets/targets/cbox_target.exr \
        --iterations 200 \
        --lr 0.01
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import torch
import imageio.v3 as iio

import pale  # custom renderer bindings
from trimesh.permutate import noise
from typing import Sequence


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass
class RendererSettingsConfig:
    photons: float = 5e3
    bounces: int = 4
    forward_passes: int = 50
    gather_passes: int = 8
    adjoint_bounces: int = 1
    adjoint_passes: int = 8
    logging: int = 4 # Spdlog enums

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "photons": self.photons,
            "bounces": self.bounces,
            "forward_passes": self.forward_passes,
            "gather_passes": self.gather_passes,
            "adjoint_bounces": self.adjoint_bounces,
            "adjoint_passes": self.adjoint_passes,
            "logging": self.logging,
        }


@dataclass
class OptimizationConfig:
    assets_root: Path
    scene_xml: str
    pointcloud_ply: str
    target_image_path: Path

    output_dir: Path

    iterations: int = 10
    learning_rate: float = 1e-2          # base LR (for convenience / default)
    learning_rate_position: float = 1e-2 # LR for positions
    learning_rate_tangent: float = 1e-2  # LR for tangents
    learning_rate_scale: float = 1e-2    # LR for scales
    optimizer_type: str = "adam"         # "adam" or "sgd"
    log_interval: int = 1
    save_interval: int = 5

    device: str = "cpu"  # torch device for parameter storage


# --------------------------------------------------------------------------------------
# Renderer parameter access hooks
# --------------------------------------------------------------------------------------

def fetch_initial_parameters(renderer: pale.Renderer) -> Dict[str, np.ndarray]:
    """
    Fetch all point parameters from the renderer as a dict of NumPy arrays.

    Expected keys and shapes (matching the C++ bindings):
        "position"   : (N,3)
        "tangent_u"  : (N,3)
        "tangent_v"  : (N,3)
        "scale"      : (N,2)
        "color"      : (N,3)
        "opacity"    : (N,)
        "beta"       : (N,)
        "shape"      : (N,)
    """
    params = renderer.get_point_parameters()
    out: Dict[str, np.ndarray] = {}
    for key, value in params.items():
        out[key] = np.asarray(value, dtype=np.float32, order="C")
    return out

def orthonormalize_tangents_inplace(
    tangent_u: torch.Tensor,
    tangent_v: torch.Tensor,
) -> dict[str, float]:
    """
    In-place Gram–Schmidt on (tangent_u, tangent_v) rows, enforcing:

        |tangent_u| = |tangent_v| = 1
        tangent_u ⟂ tangent_v
        n = tangent_u × tangent_v  (right-handed frame)

    Returns some diagnostics.
    """
    with torch.no_grad():
        # 1) normalize tangent_u
        tu = tangent_u.data
        tv = tangent_v.data

        tu_norm = tu.norm(dim=1, keepdim=True).clamp(min=1e-8)
        tu_unit = tu / tu_norm

        # 2) make tangent_v orthogonal to tangent_u
        tv_proj = (tv * tu_unit).sum(dim=1, keepdim=True) * tu_unit
        tv_orth = tv - tv_proj

        tv_norm = tv_orth.norm(dim=1, keepdim=True).clamp(min=1e-8)
        tv_unit = tv_orth / tv_norm

        # write back
        tangent_u.data.copy_(tu_unit)
        tangent_v.data.copy_(tv_unit)

        # 3) diagnostics
        dot_uv = (tangent_u * tangent_v).sum(dim=1)
        norm_u = tangent_u.norm(dim=1)
        norm_v = tangent_v.norm(dim=1)
        cross = torch.cross(tangent_u, tangent_v, dim=1)
        cross_norm = cross.norm(dim=1)

        stats = {
            "max_dev_norm_u": float((norm_u - 1.0).abs().max().item()),
            "max_dev_norm_v": float((norm_v - 1.0).abs().max().item()),
            "max_abs_dot_uv": float(dot_uv.abs().max().item()),
            "min_cross_norm": float(cross_norm.min().item()),
        }
        return stats


def verify_scales_inplace(
    scales: torch.Tensor,
) -> dict[str, float]:
    """
    In-place verification/clamping of scale values.

    Enforces:
        0.001 <= s_u, s_v <= 0.1

    Returns diagnostics about how much correction was applied.
    """
    with torch.no_grad():
        # Direct reference to underlying storage
        s = scales.data

        before_min = float(s.min().item())
        before_max = float(s.max().item())

        # Clamp in-place
        s_clamped = torch.clamp(s, min=0.001, max=1.0)
        s.copy_(s_clamped)

        after_min = float(s.min().item())
        after_max = float(s.max().item())

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min":  after_min,
            "after_max":  after_max,
        }



def apply_point_parameters(
    renderer: pale.Renderer,
    positions: torch.Tensor,
    tangent_u: torch.Tensor,
    tangent_v: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    """
    Push updated positions, tangent_u, and tangent_v into the renderer.

    Expects tensors of shape (N,3) on any device.
    """
    positions_np = np.asarray(
        positions.detach().cpu().numpy(), dtype=np.float32, order="C"
    )
    tangent_u_np = np.asarray(
        tangent_u.detach().cpu().numpy(), dtype=np.float32, order="C"
    )
    tangent_v_np = np.asarray(
        tangent_v.detach().cpu().numpy(), dtype=np.float32, order="C"
    )
    scales_np = np.asarray(
        scales.detach().cpu().numpy(), dtype=np.float32, order="C"
    )

    if positions_np.shape != tangent_u_np.shape or positions_np.shape != tangent_v_np.shape:
        raise RuntimeError(
            f"Shape mismatch between position {positions_np.shape}, "
            f"tangent_u {tangent_u_np.shape}, tangent_v {tangent_v_np.shape}"
        )

    # Optional: sanity check dtypes
    # print("dtypes:", positions_np.dtype, tangent_u_np.dtype, tangent_v_np.dtype)

    renderer.set_point_parameters(
        {
            "position": positions_np,
            "tangent_u": tangent_u_np,
            "tangent_v": tangent_v_np,
            "scale": scales_np,
        }
    )


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------


def save_gradient_sign_png_py(
    file_path: Path,
    rgba32f: np.ndarray,            # (H,W,4) float32
    adjoint_spp: float = 32.0,
    abs_quantile: float = 0.99,
    flip_y: bool = True,
) -> bool:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(rgba32f, dtype=np.float32, order="C")
    if img.ndim != 3 or img.shape[2] < 3:
        return False

    # 1) scalar = mean(R,G,B) / SPP
    rgb = img[..., :3] / float(max(adjoint_spp, 1e-8))
    scalar = np.mean(rgb, axis=2)
    scalar[~np.isfinite(scalar)] = 0.0

    # 2) symmetric robust scale using |scalar| quantile
    finite_abs = np.abs(scalar[np.isfinite(scalar)])
    if finite_abs.size:
        q = np.clip(abs_quantile, 0.0, 1.0)
        scale_abs = np.quantile(finite_abs, q) if q < 1.0 else finite_abs.max()
        if not (np.isfinite(scale_abs) and scale_abs > 0.0):
            scale_abs = 1.0
    else:
        scale_abs = 1.0
    norm = np.clip(scalar / scale_abs, -1.0, 1.0)

    # 3) map [-1,1] -> [0,1], apply matplotlib seismic
    cmap = matplotlib.colormaps["seismic"]
    t = 0.5 * (norm + 1.0)                    # [0,1]
    rgba = cmap(t, bytes=True)                # uint8 RGBA
    out = rgba[..., :3]                       # drop alpha

    # 4) flip and save
    if flip_y:
        out = np.flipud(out)
    iio.imwrite(str(file_path), out)
    return True



def load_target_image(path: Path) -> np.ndarray:
    """
    Load a target RGB image as float32 array (H,W,3).

    Supports LDR (PNG, JPG) and HDR (EXR) as long as imageio can read them.
    """
    img = iio.imread(path.as_posix())
    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 2:
        # grayscale -> RGB
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]

    if img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError(f"Target image must be HxWx3, got shape {img.shape}")

    # Normalize if needed (e.g. uint8 [0,255] -> [0,1])
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    if img.max() > 1.0 + 1e-4:
        img = img / 255.0

    return np.ascontiguousarray(img)


def save_positions_numpy(path: Path, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(positions, dtype=np.float32, order="C"))


def save_render(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(rgb, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    iio.imwrite(path.as_posix(), img_u8)

# --------------------------------------------------------------------------------------
# Loss and optimization loop
# --------------------------------------------------------------------------------------

def compute_l2_loss(
    rendered: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Simple L2 loss between rendered and target RGB images.

    Both inputs must be (H,W,3) float32.
    """
    if rendered.shape != target.shape:
        raise RuntimeError(
            f"Shape mismatch: rendered {rendered.shape}, target {target.shape}"
        )
    diff = rendered - target
    return float(np.mean(diff * diff))


def compute_l2_loss_and_grad(
    rendered: np.ndarray,
    target: np.ndarray,
    return_loss_image: bool = False,
) -> tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
    """
    L2 loss and gradient w.r.t. rendered image.

        C = mean((rendered - target)^2)
        dC/d(rendered) = 2 * (rendered - target) / (H * W * 3)

    If return_loss_image=True:
        Also returns an (H,W,3) per-pixel loss image: (rendered - target)^2
    """

    if rendered.shape != target.shape:
        raise RuntimeError(
            f"Shape mismatch: rendered {rendered.shape}, target {target.shape}"
        )

    diff = rendered - target                     # (H,W,3)
    loss_image = diff * diff                     # (H,W,3)
    loss = float(np.mean(loss_image))            # scalar

    num_elements = diff.size
    grad_image = (2.0 / float(num_elements)) * diff

    if return_loss_image:
        return loss, grad_image, loss_image

    return loss, grad_image



def create_optimizer(
    config: OptimizationConfig,
    positions: torch.nn.Parameter,
    tangent_u: torch.nn.Parameter,
    tangent_v: torch.nn.Parameter,
    scales: torch.nn.Parameter,
) -> torch.optim.Optimizer:
    """
    Create an optimizer with per-parameter learning rates.

    Falls back to config.learning_rate if a specific LR is not set.
    """
    opt_type = config.optimizer_type.lower()

    lr_pos = config.learning_rate_position or config.learning_rate
    lr_tan = config.learning_rate_tangent or config.learning_rate
    lr_scale = config.learning_rate_scale or config.learning_rate

    param_groups = [
        {
            "params": [positions],
            "lr": lr_pos,
        },
        {
            "params": [tangent_u, tangent_v],
            "lr": lr_tan,
        },
        {
            "params": [scales],
            "lr": lr_scale,
        },
    ]

    if opt_type == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.8)
    elif opt_type == "adam":
        return torch.optim.Adam(param_groups)
    else:
        raise ValueError(f"Unknown optimizer_type: {config.optimizer_type}")



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

    # --- Fetch initial parameters from renderer (dict of arrays) ---
    initial_params = fetch_initial_parameters(renderer)
    initial_positions_np = initial_params["position"]    # (N,3)
    initial_tangent_u_np = initial_params["tangent_u"]   # (N,3)
    initial_tangent_v_np = initial_params["tangent_v"]   # (N,3)
    initial_scale_np = initial_params["scale"]   # (N,2)

    num_points = initial_positions_np.shape[0]
    print(f"Fetched {num_points} initial points from renderer.")
    gt_position0 = initial_positions_np[0].copy()

    rng = np.random.default_rng(12)

    # Add small Gaussian noise to positions (you can leave tangents as-is)
    noise_sigma_translation = 0.1
    noisy_positions_np = initial_positions_np.copy()
    noisy_positions_np += rng.normal(
        loc=0.0,
        scale=noise_sigma_translation,
        size=noisy_positions_np.shape,
    )
    initial_positions_np = noisy_positions_np.astype(np.float32)
    print("Initial positions perturbed by Gaussian noise on point 0:", noise_sigma_translation)

    # Tangent noise (much smaller recommended)
    noise_sigma_tan = 0.1
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
    # Re-normalize tangent vectors
    # (each one defines a direction in world space)
    def normalize_rows(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        return arr / norms

    initial_tangent_u_np = normalize_rows(initial_tangent_u_np).astype(np.float32)
    initial_tangent_v_np = normalize_rows(initial_tangent_v_np).astype(np.float32)

    noise_sigma_scale = 0.04
    noisy_scale_np = initial_scale_np.copy()

    noisy_scale_np += rng.normal(
        0.0,
        noise_sigma_scale,
        initial_scale_np.shape,
    )

    initial_scale_np = noisy_scale_np.astype(np.float32)

    device = torch.device(config.device)

    # Parameters: positions, tangent_u, tangent_v
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

    optimizer = create_optimizer(
        config,
        positions,
        tangent_u,
        tangent_v,
        scales,
    )
    # --- Initial snapshot render ---
    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales)
    initial_rgb = renderer.render_forward()
    initial_rgb_np = np.asarray(initial_rgb, dtype=np.float32, order="C")
    initial_loss, loss_grad_image = compute_l2_loss_and_grad(
        initial_rgb_np,
        target_rgb,
    )
    print(f"Initial loss (L2): {initial_loss:.6e}")

    # --- Clear output directory (remove files but keep folders) ---
    if config.output_dir.exists():
        for item in config.output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                for item2 in item.iterdir():
                    if item2.is_file():
                        item2.unlink()

    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    save_render(config.output_dir / "render_initial.png", initial_rgb_np)
    save_render(config.output_dir / "render_target.png", target_rgb)
    save_positions_numpy(config.output_dir / "positions_initial.npy", initial_positions_np)

    # Initial parameter MSE for point 0
    initial_param_mse0 = float(
        np.mean((initial_positions_np[0] - gt_position0) ** 2)
    )
    print(f"Initial position MSE (point 0): {initial_param_mse0:.6e}")

    iteration = 0
    # --- Optimization loop ---
    try:
        for iteration in range(1, config.iterations + 1):
            # 1) Push current positions into renderer
            apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales)

            # 2) Forward render (for logging)
            current_rgb = renderer.render_forward()
            current_rgb_np = np.asarray(current_rgb, dtype=np.float32, order="C")

            # 3) Compute loss + dC/dI (adjoint image)
            loss_value, loss_grad_image, loss_image = compute_l2_loss_and_grad(
                current_rgb_np,
                target_rgb,
                return_loss_image=True,
            )

            # 4) Backward pass: renderer computes dC/d(position), dC/d(tangent_u), dC/d(tangent_v)
            gradients, grad_img = renderer.render_backward(loss_grad_image)
            grad_scale = 1.0

            grad_position_np = np.asarray(
                gradients["position"] * grad_scale, dtype=np.float32, order="C"
            )
            grad_tangent_u_np = np.asarray(
                gradients["tangent_u"] * grad_scale, dtype=np.float32, order="C"
            )
            grad_tangent_v_np = np.asarray(
                gradients["tangent_v"] * grad_scale, dtype=np.float32, order="C"
            )
            grad_scales_v_np = np.asarray(
                gradients["scale"] * grad_scale, dtype=np.float32, order="C"
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

            # 5) Set torch gradients and step optimizer
            optimizer.zero_grad(set_to_none=True)
            positions.grad = torch.tensor(grad_position_np).to(device=device, dtype=torch.float32)
            tangent_u.grad = torch.tensor(grad_tangent_u_np).to(device=device, dtype=torch.float32)
            tangent_v.grad = torch.tensor(grad_tangent_v_np).to(device=device, dtype=torch.float32)
            scales.grad =    torch.tensor(grad_scales_v_np).to(device=device, dtype=torch.float32)
            optimizer.step()

            # 5b) Re-orthonormalize tangents and collect diagnostics
            ortho_stats = orthonormalize_tangents_inplace(tangent_u, tangent_v)
            verify_scales_inplace(scales)

            # --- Logging ---
            if iteration % config.log_interval == 0 or iteration == 1:
                grad_norm = float(np.linalg.norm(grad_position_np) / max(num_points, 1))
                grad_tanu = float(np.linalg.norm(grad_tangent_u_np) / max(num_points, 1))
                grad_tanv = float(np.linalg.norm(grad_tangent_v_np) / max(num_points, 1))
                grad_scale = float(np.linalg.norm(grad_scales_v_np) / max(num_points, 1))
                print(
                    f"[Iter {iteration:04d}/{config.iterations}] "
                    f"Loss = {loss_value:.6e}, "
                    f"mean |translation| = {grad_norm:.3e}, "
                    f"mean |tan_u| = {grad_tanu:.3e}, "
                    f"mean |tan_v| = {grad_tanv:.3e}, "
                    f"mean |scale| = {grad_scale:.3e}, "
                    #f"ortho: max_dev_norm_u={ortho_stats['max_dev_norm_u']:.2e}, "
                    #f"max_dev_norm_v={ortho_stats['max_dev_norm_v']:.2e}, "
                    #f"max_abs_dot_uv={ortho_stats['max_abs_dot_uv']:.2e}, "
                    #f"min_cross_norm={ortho_stats['min_cross_norm']:.2e}"
                )

            # --- Periodic saving ---
            if iteration % config.save_interval == 0 or iteration == config.iterations:
                #apply_positions(renderer, positions.detach().cpu().numpy())
                #snapshot_rgb = renderer.render_forward()
                snapshot_rgb_np = np.asarray(current_rgb, dtype=np.float32, order="C")

                render_path = config.output_dir / "render" / f"render_iter_{iteration:04d}.png"
                #pos_path = config.output_dir / f"positions_iter_{iteration:04d}.npy"
                save_render(render_path, snapshot_rgb_np)
                #save_positions_numpy(pos_path, positions.detach().cpu().numpy())
                img = np.asarray(grad_img, dtype=np.float32, order="C")  # (H,W,4)
                img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

                render_grad = config.output_dir / "grad" / f"render_iter_grad_{iteration:04d}.png"
                render_grad_quantile = config.output_dir / "grad" / f"render_iter_grad_099_{iteration:04d}.png"
                save_gradient_sign_png_py(
                    render_grad_quantile,
                    img,
                    adjoint_spp=8,
                    abs_quantile=0.999,
                    flip_y=False,
                )

                # Save visualization (tonemap + robust scaling)
                loss_image_vis = np.clip(loss_image / np.percentile(loss_image, 99.0), 0.0, 1.0)
                loss_image_u8 = (loss_image_vis * 255).astype(np.uint8)
                os.makedirs(config.output_dir / "loss", exist_ok=True)
                iio.imwrite(
                    (config.output_dir / "loss" / f"loss_image_iter_{iteration:04d}.png").as_posix(),
                    loss_image_u8,
                )
    except KeyboardInterrupt:
        print(
            f"\nCtrl+C detected at iteration {iteration:04d}. "
            "Stopping optimization loop and saving current result..."
        )


    # --- Final summary ---
    apply_point_parameters(renderer, positions, tangent_u, tangent_v, scales)
    final_rgb = renderer.render_forward()
    final_rgb_np = np.asarray(final_rgb, dtype=np.float32, order="C")
    final_loss = compute_l2_loss(final_rgb_np, target_rgb)

    save_render(config.output_dir / "render_final.png", final_rgb_np)
    save_positions_numpy(config.output_dir / "positions_final.npy", positions.detach().cpu().numpy())

    print("\nOptimization completed.")
    print(f"Initial loss: {initial_loss:.6e}")
    print(f"Final loss:   {final_loss:.6e}")
    print(f"Outputs saved in: {config.output_dir.resolve()}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> OptimizationConfig:
    parser = argparse.ArgumentParser(
        description="Optimize point positions using a custom differentiable renderer."
    )

    parser.add_argument(
        "--assets-root",
        type=Path,
        required=False,
        default="../Assets",
        help="Path to the Assets directory used by the renderer.",
    )
    parser.add_argument(
        "--scene-xml",
        type=str,
        default="cbox_custom.xml",
        help="Scene XML file name (relative to assets-root).",
    )
    parser.add_argument(
        "--pointcloud",
        type=str,
        default="initial.ply",
        help="Point cloud PLY file used by the renderer.",
    )
    parser.add_argument(
        "--target-image",
        type=Path,
        required=False,
        default="./Output/target/out_photonmap.png",
        help="Path to target RGB image (PNG, JPG, EXR, etc.).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("OptimizationOutput"),
        help="Directory where intermediate and final outputs are saved.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of optimization iterations.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["adam", "sgd"],
        help="Which optimizer to use.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Print log every N iterations.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save render and positions every N iterations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for parameters (e.g. 'cpu' or 'cuda').",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=1.0,
        help="Base learning rate (used if --lr-pos / --lr-tan / --lr-scale are not given).",
    )
    parser.add_argument(
        "--lr-pos",
        dest="learning_rate_position",
        type=float,
        default=1.0,
        help="Learning rate for positions (defaults to base LR if omitted).",
    )
    parser.add_argument(
        "--lr-tan",
        dest="learning_rate_tangent",
        type=float,
        default=0.5,
        help="Learning rate for tangents (tangent_u, tangent_v; defaults to base LR if omitted).",
    )
    parser.add_argument(
        "--lr-scale",
        dest="learning_rate_scale",
        type=float,
        default=0.1,
        help="Learning rate for scales (u, v; defaults to base LR if omitted).",
    )


    # ... optimizer, log-interval, save-interval, device ...

    args = parser.parse_args()

    lr_base = args.learning_rate

    lr_pos = args.learning_rate_position
    if lr_pos is None:
        lr_pos = lr_base

    lr_tan = args.learning_rate_tangent
    if lr_tan is None:
        lr_tan = lr_base

    lr_scale = args.learning_rate_scale
    if lr_scale is None:
        lr_scale = lr_base

    return OptimizationConfig(
        assets_root=args.assets_root,
        scene_xml=args.scene_xml,
        pointcloud_ply=args.pointcloud,
        target_image_path=args.target_image,
        output_dir=args.output_dir,
        iterations=args.iterations,
        learning_rate=lr_base,
        learning_rate_position=lr_pos,
        learning_rate_tangent=lr_tan,
        learning_rate_scale=lr_scale,
        optimizer_type=args.optimizer,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device,
    )



def main() -> None:
    config = parse_args()
    renderer_settings = RendererSettingsConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting optimization with configuration:")
    print(f"  assets_root   : {config.assets_root}")
    print(f"  scene_xml     : {config.scene_xml}")
    print(f"  pointcloud    : {config.pointcloud_ply}")
    print(f"  target_image  : {config.target_image_path}")
    print(f"  iterations    : {config.iterations}")
    print(f"  learning_rate_pos : {config.learning_rate_position}")
    print(f"  learning_rate_tan : {config.learning_rate_tangent}")
    print(f"  learning_rate_scale : {config.learning_rate_scale}")
    print(f"  optimizer     : {config.optimizer_type}")
    print(f"  output_dir    : {config.output_dir}")
    print(f"  device        : {config.device}")

    run_optimization(config, renderer_settings)


if __name__ == "__main__":
    main()
