from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import math


@dataclass
class RendererSettingsConfig:
    photons: float = 1e6
    bounces: int = 1
    adjoint_bounces: int = 1
    forward_passes: int = 1
    primal_shadow_rays: int = 1  # Li
    adjoint_shadow_rays: int = 1  # Li
    gather_passes: int = 1
    adjoint_passes: int = 1
    enable_adjoint_shadow_rays: bool = True
    adjoint_shadow_path_rays: int = 1  # p_i
    logging: int = 3

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
            "depth_distort_weight": config.depth_distort_weight,
            "normal_consistency_weight": config.normal_consistency_weight,
            "visibility_weighted_opacity_weight": config.visibility_weighted_opacity_weight,
        }


@dataclass
class OptimizationConfig:
    assets_root: Path = Path("../Assets")
    scene_xml: str = "cbox_custom.xml"
    pointcloud_ply: str = "initial.ply"
    dataset_path: Path = Path("./Output/target")
    output_dir: Path = Path("OptimizationOutput")

    iterations: int = int(1e5)

    optimizer_type: str = "sgd"
    learning_rate: float = 1.0

    learning_rate_position: float | None = None
    learning_rate_tangent: float | None = None
    learning_rate_scale: float | None = None
    learning_rate_albedo: float | None = None
    learning_rate_opacity: float | None = None
    learning_rate_beta: float | None = None
    # Position-only exponential LR schedule.
    use_position_lr_schedule: bool = False
    position_lr_scale_init: float = 2.0
    position_lr_scale_final: float = 0.5
    position_lr_max_steps: int = 10_000

    depth_distort_weight: float = 1000
    depth_distort_start_iteration: int = 0
    normal_consistency_weight: float = 0.01
    visibility_weighted_opacity_weight: float = 0.1

    log_interval: int = 1
    save_interval: int = 5
    device: str = "cpu"

    # Density control / EV-splitting
    densification_interval: int = 100
    prune_interval: int = 25
    densify_after: int = -1
    prune_after: int = -1
    densify_until_iteration: int = -1
    densify_until_fraction: float = 0.8

    densification_verbose: bool = True
    densification_grad_quantile: float = 0.0,
    densification_grad_abs_min: float = 1e-2
    densification_scale_min: float = 1.5e-2

    # More densification on radiometrically darker primitives
    densify_bsdf_floor: float = 0.15
    densify_bsdf_gamma: float = 1.2

    # Pruning
    opacity_prune_threshold: float = 0.05
    max_prune_fraction: float = 0.9
    scale_prune_min_scale: float = 1.0e-3
    min_points_to_keep_after_scale_prune: int = 1

    # Misc scheduling
    reset_opacity_interval: int = 1500
    reset_opacity_value: float = 0.0
    rebuild_bvh_interval: int = 1


def resolve_learning_rates(config: OptimizationConfig) -> None:
    base_learning_rate = config.learning_rate

    if config.optimizer_type == "sgd":
        factor_position = 0.2
        factor_tangent = 0.1
        factor_scale = 0.005
        factor_albedo = 2.0
        factor_opacity = 1.0
        factor_beta = 0.00
    elif config.optimizer_type == "adam":
        factor_position = 0.0005
        factor_tangent = 0.005
        factor_scale = 0.002
        factor_albedo = 0.002
        factor_opacity = 0.01
        factor_beta = 0.000
    else:
        raise ValueError(f"Unknown optimizer_type: {config.optimizer_type}")

    if config.learning_rate_position is None:
        config.learning_rate_position = factor_position * base_learning_rate
    if config.learning_rate_tangent is None:
        config.learning_rate_tangent = factor_tangent * base_learning_rate
    if config.learning_rate_scale is None:
        config.learning_rate_scale = factor_scale * base_learning_rate
    if config.learning_rate_albedo is None:
        config.learning_rate_albedo = factor_albedo * base_learning_rate
    if config.learning_rate_opacity is None:
        config.learning_rate_opacity = factor_opacity * base_learning_rate
    if config.learning_rate_beta is None:
        config.learning_rate_beta = factor_beta * base_learning_rate


def scale_iteration_interval_by_learning_rate(base_interval: int, learning_rate: float, ) -> int:
    if base_interval <= 0:
        return base_interval
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    return max(1, math.ceil(float(base_interval) / learning_rate))


def resolve_iteration_schedules(config: OptimizationConfig, cli_overrides: set[str], ) -> None:
    if "densification_interval" not in cli_overrides:
        config.densification_interval = scale_iteration_interval_by_learning_rate(config.densification_interval,
                                                                                  config.learning_rate,
                                                                                  )


def parse_args() -> OptimizationConfig:
    parser = argparse.ArgumentParser(
        description="Optimize point positions using a custom differentiable renderer.",
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--scene-xml", type=str)
    parser.add_argument("--pointcloud", dest="pointcloud_ply", type=str)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path)

    parser.add_argument("--iterations", type=int)
    parser.add_argument("--optimizer", dest="optimizer_type", type=str, choices=["adam", "sgd"])
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--device", type=str)

    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--lr-pos", dest="learning_rate_position", type=float)
    parser.add_argument("--lr-tan", dest="learning_rate_tangent", type=float)
    parser.add_argument("--lr-scale", dest="learning_rate_scale", type=float)
    parser.add_argument("--lr-albedo", dest="learning_rate_albedo", type=float)
    parser.add_argument("--lr-opacity", dest="learning_rate_opacity", type=float)
    parser.add_argument("--lr-beta", dest="learning_rate_beta", type=float)
    parser.add_argument("--position-lr-schedule", dest="use_position_lr_schedule",
                        action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--position-lr-scale-init", type=float)
    parser.add_argument("--position-lr-scale-final", type=float)
    parser.add_argument("--position-lr-max-steps", type=int)

    parser.add_argument("--normal-consistency-weight", dest="normal_consistency_weight", type=float)
    parser.add_argument("--depth-distort-weight", dest="depth_distort_weight", type=float)
    parser.add_argument("--depth-distort-start-iteration", type=int)
    parser.add_argument("--visibility-weighted-opacity-weight", dest="visibility_weighted_opacity_weight", type=float, )
    # Density control / EV-splitting
    parser.add_argument("--densification-interval", type=int)
    parser.add_argument("--prune-interval", type=int)
    parser.add_argument("--densify-after", type=int)
    parser.add_argument("--prune-after", type=int)
    parser.add_argument("--densify-until-iteration", type=int)
    parser.add_argument("--densify-until-fraction", type=float)

    parser.add_argument(
        "--densification-verbose",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )

    parser.add_argument("--densification-grad-quantile", type=float)
    parser.add_argument("--densification-grad-abs-min", type=float)

    parser.add_argument("--densify-bsdf-floor", type=float)
    parser.add_argument("--densify-bsdf-gamma", type=float)

    parser.add_argument("--max-split-fraction", type=float)

    parser.add_argument(
        "--evsplit-preserve-integrated-opacity",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )

    parser.add_argument("--evsplit-min-scale", type=float)

    # Pruning
    parser.add_argument("--opacity-prune-threshold", type=float)
    parser.add_argument("--max-prune-fraction", type=float)
    parser.add_argument("--scale-prune-min-scale", type=float)
    parser.add_argument("--min-points-to-keep-after-scale-prune", type=int)

    # Misc scheduling
    parser.add_argument("--reset-opacity-interval", type=int)
    parser.add_argument("--reset-opacity-value", type=float)
    parser.add_argument("--rebuild-bvh-interval", type=int)

    args = parser.parse_args()

    config = OptimizationConfig()

    for parameter_name, parameter_value in vars(args).items():
        if not hasattr(config, parameter_name):
            raise RuntimeError(
                f"CLI argument produced unknown config field: {parameter_name}"
            )
        setattr(config, parameter_name, parameter_value)

    resolve_learning_rates(config)

    # for parameter_name, parameter_value in vars(args).items():
    #    if not hasattr(config, parameter_name):
    #        raise RuntimeError(
    #            f"CLI argument produced unknown config field: {parameter_name}"
    #        )
    #    setattr(config, parameter_name, parameter_value)
    #
    ##resolve_iteration_schedules(config, cli_overrides)
    # resolve_learning_rates(config)

    return config
