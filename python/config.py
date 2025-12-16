from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class RendererSettingsConfig:
    photons: float = 1e5
    bounces: int = 3
    forward_passes: int = 10
    gather_passes: int = 1
    adjoint_bounces: int = 2
    adjoint_passes: int = 1
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
    dataset_path: Path
    output_dir: Path
    personal_suffix: str = ""

    iterations: int = 50000
    learning_rate: float = 1e-2  # base LR (for convenience / default)
    # Defaults chosen to match 3DGS absolute values when base LR = 1.6e-4
    learning_rate_position: float = 1.6e-4      # 1.0 × base
    learning_rate_tangent: float = 1.0e-3       # 6.25 × base
    learning_rate_scale: float = 5.0e-3         # 31.25 × base
    learning_rate_albedo: float = 2.5e-3        # 15.625 × base
    learning_rate_opacity: float = 5.0e-2       # 312.5 × base
    learning_rate_beta: float = 5.0e-2       # 312.5 × base
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
        "--dataset-path",
        type=Path,
        required=False,
        default=Path("./Output/target"),
        help="Path to target RGB image (PNG, JPG, EXR, etc.).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("OptimizationOutput"),
        help="Directory where intermediate and final outputs are saved.",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional string appended to the run output folder (e.g. 'no_shadows', 'debug', 'v2').",
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
        "--lr-albedo",
        dest="learning_rate_albedo",
        type=float,
        default=None,
        help="Learning rate for albedos (defaults to ~0.5 * base LR if omitted).",
    )
    parser.add_argument(
        "--lr-opacity",
        dest="learning_rate_opacity",
        type=float,
        default=None,
        help="Learning rate for opacities (defaults to ~0.5 * base LR if omitted).",
    )
    parser.add_argument(
        "--lr-beta",
        dest="learning_rate_beta",
        type=float,
        default=None,
        help="Learning rate for opacities (defaults to ~0.5 * base LR if omitted).",
    )

    args = parser.parse_args()

    # Base LR (position LR), with optional global multiplier
    base_lr = args.learning_rate * args.learning_rate_multiplier
    lr_base = args.learning_rate  # store the *unmultiplied* base, if you want to log i

    lr_scale = 1
    if args.optimizer == "sgd":
        lr_scale = 1000

    # 3DGS-inspired relative factors w.r.t. position LR
    factor_position = lr_scale * 0.01  # ~rotation_lr / position_lr
    factor_tangent  = lr_scale * 0.10   # ~rotation_lr / position_lr
    factor_scale    = lr_scale * 0.0001   # ~scaling_lr / position_lr
    factor_albedo   = lr_scale * 10.0    # ~feature_lr / position_lr
    factor_opacity  = lr_scale * 10.0    # ~opacity_lr / position_lr
    factor_beta     = lr_scale * 1.0  # ~beta_lr / position_lr


    #factor_position = lr_scale * 0  # ~rotation_lr / position_lr
    #factor_tangent  = lr_scale * 0  # ~rotation_lr / position_lr
    #factor_scale    = lr_scale * 0  # ~scaling_lr / position_lr
    #factor_albedo   = lr_scale * 0  # ~feature_lr / position_lr
    #factor_opacity  = lr_scale * 0  # ~opacity_lr / position_lr
    #factor_beta     = lr_scale * 0  # ~beta_lr / position_lr
#
    lr_pos = args.learning_rate_position or (factor_position *  base_lr)
    lr_tan = args.learning_rate_tangent or (factor_tangent * base_lr)
    lr_scale = args.learning_rate_scale or (factor_scale * base_lr)
    lr_albedo = args.learning_rate_albedo or (factor_albedo * base_lr)
    lr_opacity = args.learning_rate_opacity or (factor_opacity * base_lr)
    lr_beta = args.learning_rate_beta or (factor_beta * base_lr)

    return OptimizationConfig(
        assets_root=args.assets_root,
        scene_xml=args.scene_xml,
        pointcloud_ply=args.pointcloud,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        iterations=args.iterations,
        learning_rate=lr_base,
        learning_rate_position=lr_pos,
        learning_rate_tangent=lr_tan,
        learning_rate_scale=lr_scale,
        learning_rate_albedo=lr_albedo,
        learning_rate_opacity=lr_opacity,
        learning_rate_beta=lr_beta,
        optimizer_type=args.optimizer,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device,
        personal_suffix=args.suffix,
    )
