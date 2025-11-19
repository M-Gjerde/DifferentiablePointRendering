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


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass
class RendererSettingsConfig:
    photons: float = 1e4
    bounces: int = 4
    forward_passes: int = 40
    gather_passes: int = 16
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
    learning_rate: float = 1e-2
    optimizer_type: str = "adam"  # "adam" or "sgd"
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


def apply_positions(renderer: pale.Renderer, positions: np.ndarray) -> None:
    """
    Push updated positions into the renderer.

    Uses set_point_parameters with only the 'position' key; other parameters
    remain unchanged on the C++ side.
    """
    positions_np = np.asarray(positions, dtype=np.float32, order="C")
    if positions_np.ndim != 2 or positions_np.shape[1] != 3:
        raise RuntimeError(
            f"Expected positions of shape (N,3), got {positions_np.shape}"
        )

    renderer.set_point_parameters({"position": positions_np})


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
    parameter: torch.nn.Parameter,
    config: OptimizationConfig,
) -> torch.optim.Optimizer:
    if config.optimizer_type.lower() == "sgd":
        return torch.optim.SGD([parameter], lr=config.learning_rate, momentum=0.9)
    elif config.optimizer_type.lower() == "adam":
        return torch.optim.Adam([parameter], lr=config.learning_rate)
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
    initial_positions_np = initial_params["position"]  # (N,3)
    num_points = initial_positions_np.shape[0]
    print(f"Fetched {num_points} initial points from renderer.")
    gt_position0 = initial_positions_np[0].copy()
    # Add small Gaussian noise to the first position only
    noise_sigma = 0.05 # try 0.005–0.05 depending on scene scale
    rng = np.random.default_rng(42)

    noisy_positions_np = initial_positions_np.copy()
    noisy_positions_np += rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=(3,),
    )

    #noisy_positions_np[0] += (0.05, 0, 0)

    initial_positions_np = noisy_positions_np.astype(np.float32)
    print("Initial positions perturbed by Gaussian noise on point 0:", noise_sigma)


    # --- Wrap parameters in torch (currently only positions are optimized) ---
    device = torch.device(config.device)
    positions = torch.nn.Parameter(
        torch.tensor(initial_positions_np, device=device, dtype=torch.float32)
    )

    optimizer = create_optimizer(positions, config)

    # --- Initial snapshot render ---
    apply_positions(renderer, positions.detach().cpu().numpy())
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

    # --- Optimization loop ---
    for iteration in range(1, config.iterations + 1):
        # 1) Push current positions into renderer
        apply_positions(renderer, positions.detach().cpu().numpy())

        # 2) Forward render (for logging)
        current_rgb = renderer.render_forward()
        current_rgb_np = np.asarray(current_rgb, dtype=np.float32, order="C")

        # 3) Compute loss + dC/dI (adjoint image)
        loss_value, loss_grad_image, loss_image = compute_l2_loss_and_grad(
            current_rgb_np,
            target_rgb,
            return_loss_image=True,
        )

        # 4) Backward pass: renderer computes dC/d(position)
        gradients, grad_img = renderer.render_backward(loss_grad_image)
        grad_scale = 1e3
        grad_position_np = np.asarray(gradients["position"] * grad_scale, dtype=np.float32, order="C")

        if grad_position_np.shape != initial_positions_np.shape:
            raise RuntimeError(
                f"Gradient shape mismatch: expected {initial_positions_np.shape}, "
                f"got {grad_position_np.shape}"
            )

        # 5) Set torch gradients and step optimizer
        optimizer.zero_grad(set_to_none=True)
        positions.grad = torch.tensor(grad_position_np).to(
            device=device,
            dtype=torch.float32,
        )
        optimizer.step()

        # --- Parameter loss (MSE) for point 0 ---
        current_positions_np = positions.detach().cpu().numpy()
        current_positions_np = np.asarray(current_positions_np, dtype=np.float32, order="C")
        param_mse0 = float(
            np.mean((current_positions_np[0] - gt_position0) ** 2)
        )

        # --- Logging ---
        if iteration % config.log_interval == 0 or iteration == 1:
            grad_norm = float(np.linalg.norm(grad_position_np) / max(num_points, 1))
            print(
                f"[Iter {iteration:04d}/{config.iterations}] "
                f"Loss = {loss_value:.6e}, "
                f"mean |grad_position| = {grad_norm:.6e}, "
                f"param MSE[0] = {param_mse0:.6e}"
                f"param (x, y, z) = ({current_positions_np[0][0]:.2f}, {current_positions_np[0][1]:.2f}, {current_positions_np[0][2]:.2f})"
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


    # --- Final summary ---
    apply_positions(renderer, positions.detach().cpu().numpy())
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
        required=True,
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
        required=True,
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
        "--lr",
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for the optimizer.",
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

    args = parser.parse_args()

    return OptimizationConfig(
        assets_root=args.assets_root,
        scene_xml=args.scene_xml,
        pointcloud_ply=args.pointcloud,
        target_image_path=args.target_image,
        output_dir=args.output_dir,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
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
    print(f"  learning_rate : {config.learning_rate}")
    print(f"  optimizer     : {config.optimizer_type}")
    print(f"  output_dir    : {config.output_dir}")
    print(f"  device        : {config.device}")

    run_optimization(config, renderer_settings)


if __name__ == "__main__":
    main()
