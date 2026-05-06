from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class RendererSettingsConfig:
    photons: float = 1e6
    bounces: int = 2
    forward_passes: int = 5
    primal_shadow_rays: int =  4 # Li
    adjoint_shadow_rays: int = 4 # Li
    gather_passes: int = 1
    adjoint_bounces: int = 3
    adjoint_passes: int = 6
    enable_adjoint_shadow_rays: bool = True
    adjoint_shadow_path_rays: int = 2 #Pi
    useDepthDistortion: bool = True
    useNormalConsistency: bool = True
    logging: int = 4

    def as_dict(self, config: "OptimizationConfig") -> Dict[str, float | int]:
        return {
            "photons": self.photons,
            "bounces": self.bounces,
            "forward_passes": self.forward_passes,
            "gather_passes": self.gather_passes,
            "primal_shadow_rays": self.primal_shadow_rays,
            "adjoint_shadow_rays": self.adjoint_shadow_rays,
            "enable_adjoint_shadow_rays": self.enable_adjoint_shadow_rays,
            "adjoint_shadow_path_rays": self.adjoint_shadow_path_rays,
            "adjoint_bounces": self.adjoint_bounces,
            "adjoint_passes": self.adjoint_passes,
            "logging": self.logging,
            "use_depth_distortion": self.useDepthDistortion,
            "use_normal_consistency": self.useNormalConsistency,
            "depth_distort_weight": config.depth_distort_weight,
            "normal_consistency_weight": config.normal_consistency_weight,
        }


@dataclass
class OptimizationConfig:
    assets_root: Path
    scene_xml: str
    pointcloud_ply: str
    dataset_path: Path
    output_dir: Path
    personal_suffix: str = ""
    personal_prefix: str = ""

    iterations: int = 50000
    learning_rate: float = 1e-2
    learning_rate_position: float = 0
    learning_rate_tangent: float = 0
    learning_rate_scale: float = 0
    learning_rate_albedo: float = 0
    learning_rate_opacity: float = 0
    learning_rate_beta: float = 0

    depth_distort_weight: float = 0.2
    normal_consistency_weight: float = 0.05
    opacity_loss_weight: float = 10

    optimizer_type: str = "adam"
    log_interval: int = 1
    save_interval: int = 5
    device: str = "cpu"


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
        help="Path to target RGB image directory.",
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
        help="Optional string appended to the run output folder.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional string prepended to the run output folder.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=int(1e5),
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
        help="Torch device for parameter storage.",
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
        "--lr-pos",
        dest="learning_rate_position",
        type=float,
        default=None,
        help="Learning rate for positions.",
    )
    parser.add_argument(
        "--lr-tan",
        dest="learning_rate_tangent",
        type=float,
        default=None,
        help="Learning rate for tangents.",
    )
    parser.add_argument(
        "--lr-scale",
        dest="learning_rate_scale",
        type=float,
        default=None,
        help="Learning rate for scales.",
    )
    parser.add_argument(
        "--lr-albedo",
        dest="learning_rate_albedo",
        type=float,
        default=None,
        help="Learning rate for albedos.",
    )
    parser.add_argument(
        "--lr-opacity",
        dest="learning_rate_opacity",
        type=float,
        default=None,
        help="Learning rate for opacities.",
    )
    parser.add_argument(
        "--lr-beta",
        dest="learning_rate_beta",
        type=float,
        default=None,
        help="Learning rate for beta.",
    )

    parser.add_argument(
        "--normal-consistency-weight",
        dest="normal_consistency_weight",
        type=float,
        default=0.5,
        help="Weight for the normal consistency regularizer.",
    )

    parser.add_argument(
        "--depth-distort-weight",
        dest="depth_distort_weight",
        type=float,
        default=100.0,
        help="Weight for the depth distortion regularizer.",
    )
    parser.add_argument(
        "--opacity-weight",
        dest="opacity_loss_weight",
        type=float,
        default=10,
        help="Weight for the favoring opacity = 1.",
    )

    args = parser.parse_args()

    base_lr = args.learning_rate
    lr_base = args.learning_rate

    if args.optimizer == "sgd":
        # Tuned for your current gradient magnitudes:
        # - reduce position step substantially
        # - reduce tangent step drastically
        # - keep scale/albedo/opacity responsive
        factor_position = 1.0
        factor_tangent = 10.0
        factor_scale = 0.5
        factor_albedo = 200.0
        factor_opacity = 200.0
        factor_beta = 0.25
    else:
        factor_position = 0.0005
        factor_tangent = 0.002
        factor_scale = 0.0002
        factor_albedo = 0.005
        factor_opacity = 0.001
        factor_beta = 0.000

    lr_pos = args.learning_rate_position or (factor_position * base_lr)
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
        depth_distort_weight=args.depth_distort_weight,
        normal_consistency_weight=args.normal_consistency_weight,
        opacity_loss_weight=args.opacity_loss_weight,
        optimizer_type=args.optimizer,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device,
        personal_suffix=args.suffix,
        personal_prefix=args.prefix,
    )