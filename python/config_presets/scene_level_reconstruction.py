"""Scene level reconstruction defaults; parsing and validation live in config."""

from dataclasses import dataclass

from config import (
    OptimizationConfig as BaseOptimizationConfig,
    RendererSettingsConfig as BaseRendererSettingsConfig,
    configure_checkpoint,
    parse_args as parse_config_args,
    resolve_learning_rates,
)


@dataclass
class RendererSettingsConfig(BaseRendererSettingsConfig):
    adjoint_passes: int = 4


@dataclass
class OptimizationConfig(BaseOptimizationConfig):
    learning_rate_position: float = 0.0005
    learning_rate_scale: float = 0.0035
    position_lr_scale_init: float = 6.0
    position_lr_scale_final: float = 0.5
    lr_decay_max_steps: int = 25000
    depth_distort_weight: float = 0.0025
    normal_consistency_weight: float = 0.005
    densification_interval: int = 1000
    densification_stats_skip_interval_start: bool = True
    densification_diagnostics: bool = True
    densification_split_scale_factor: float = 1.2
    rebuild_bvh_interval: int = 1000
    mesh_extraction_mesh_res: int = 1024
    save_snapshot_rgb: bool = True


def parse_args() -> OptimizationConfig:
    return parse_config_args(OptimizationConfig)
