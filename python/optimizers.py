from __future__ import annotations

import torch

from config import OptimizationConfig


def create_optimizer(
    config: OptimizationConfig,
    positions: torch.nn.Parameter,
    tangent_u: torch.nn.Parameter,
    tangent_v: torch.nn.Parameter,
    scales: torch.nn.Parameter,
    albedos: torch.nn.Parameter,
    opacities: torch.nn.Parameter,
    betas: torch.nn.Parameter,
) -> torch.optim.Optimizer:
    """
    Create an optimizer with per-parameter learning rates.

    Falls back to config.learning_rate if a specific LR is not set.
    """
    opt_type = config.optimizer_type.lower()

    lr_pos = config.learning_rate_position or config.learning_rate
    lr_tan = config.learning_rate_tangent or config.learning_rate
    lr_scale = config.learning_rate_scale or config.learning_rate
    lr_albedo = config.learning_rate_albedo or config.learning_rate
    lr_opacity = config.learning_rate_opacity or config.learning_rate
    lr_beta = config.learning_rate_beta or config.learning_rate

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
        {
            "params": [albedos],
            "lr": lr_albedo,
        },
        {
            "params": [opacities],
            "lr": lr_opacity,
        },
        {
            "params": [betas],
            "lr": lr_beta,
        },
    ]

    if opt_type == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.8)
    if opt_type == "adam":
        return torch.optim.Adam(param_groups)

    raise ValueError(f"Unknown optimizer_type: {config.optimizer_type}")
