from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class RendererSettingsConfig:
    photons: float = 1e5
    bounces: int = 4
    forward_passes: int = 10
    gather_passes: int = 2
    adjoint_bounces: int = 2
    adjoint_passes: int = 2
    logging: int = 3  # Spdlog enums

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
    learning_rate: float = 1e-2  # base LR (for convenience / default)
    learning_rate_position: float = 1e-2
    learning_rate_tangent: float = 1e-2
    learning_rate_scale: float = 1e-2
    learning_rate_color: float = 1e-2
    optimizer_type: str = "adam"  # "adam" or "sgd"
    log_interval: int = 1
    save_interval: int = 5
    device: str = "cpu"  # torch device for parameter storage


def parse_args() -> OptimizationConfig:
    parser = argparse.ArgumentParser(
        description="Optimize point positions using a custom differentiable renderer."
    )

    parser.add_argument(
        "--assets-root",
        type=Path,
        required=False,
        default=Path("../Assets"),
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
        default=Path("./Output/target/out_photonmap.png"),
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
        help="Base learning rate before per-parameter multipliers.",
    )
    parser.add_argument(
        "--lr-multiplier",
        "--learning-rate-multiplier",
        dest="learning_rate_multiplier",
        type=float,
        default=1.0,
        help="Global multiplier applied to all per-parameter learning rates.",
    )
    parser.add_argument(
        "--lr-pos",
        dest="learning_rate_position",
        type=float,
        default=None,  # derive from base LR if not set
        help="Learning rate for positions (defaults to ~0.1 * base LR if omitted).",
    )
    parser.add_argument(
        "--lr-tan",
        dest="learning_rate_tangent",
        type=float,
        default=None,
        help="Learning rate for tangents (defaults to ~0.2 * base LR if omitted).",
    )
    parser.add_argument(
        "--lr-scale",
        dest="learning_rate_scale",
        type=float,
        default=None,
        help="Learning rate for scales (defaults to ~0.2 * base LR if omitted).",
    )
    parser.add_argument(
        "--lr-color",
        dest="learning_rate_color",
        type=float,
        default=None,
        help="Learning rate for colors (defaults to ~0.5 * base LR if omitted).",
    )

    args = parser.parse_args()

    lr_base = args.learning_rate
    base_lr = args.learning_rate * args.learning_rate_multiplier

    lr_pos = args.learning_rate_position or (0.5 * base_lr)
    lr_tan = args.learning_rate_tangent or  (0.1 * base_lr)
    lr_scale = args.learning_rate_scale or  (0.1 * base_lr)
    lr_color = args.learning_rate_color or  (0.1 * base_lr)

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
        learning_rate_color=lr_color,
        optimizer_type=args.optimizer,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device,
    )
