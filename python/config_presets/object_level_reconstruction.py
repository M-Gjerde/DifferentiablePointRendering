"""Object level reconstruction defaults; parsing and validation live in config."""

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
    adjoint_passes: int = 16


@dataclass
class OptimizationConfig(BaseOptimizationConfig):
    iterations: int = 20000
    learning_rate_position: float = 0.0005
    learning_rate_scale: float = 0.0035
    position_lr_scale_init: float = 1.5
    position_lr_scale_final: float = 0.5
    lr_decay_max_steps: int = 18000
    densification_stats_skip_interval_start: bool = True
    densification_diagnostics: bool = True
    densification_split_scale_factor: float = 1.2
    save_snapshot_rgb: bool = True


def parse_args() -> OptimizationConfig:
    return parse_config_args(OptimizationConfig)
