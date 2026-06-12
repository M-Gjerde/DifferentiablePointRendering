from __future__ import annotations

import torch

from typing import Any, Callable, Iterable, Optional
from config import OptimizationConfig

import numpy as np


def create_optimizer(
    config: OptimizationConfig,
    positions: torch.nn.Parameter,
    rotation: torch.nn.Parameter,
    scales: torch.nn.Parameter,
    albedos: torch.nn.Parameter,
    opacities: torch.nn.Parameter,
    betas: torch.nn.Parameter,
) -> torch.optim.Optimizer:
    """Create an optimizer with per-parameter learning rates."""
    opt_type = config.optimizer_type.lower()
    param_groups = [
        {"params": [positions], "lr": config.learning_rate_position, "name": "position"},
        {"params": [rotation], "lr": config.learning_rate_rotation, "name": "rotation"},
        {"params": [scales], "lr": config.learning_rate_scale, "name": "scale"},
        {"params": [albedos], "lr": config.learning_rate_albedo, "name": "albedo"},
        {"params": [opacities], "lr": config.learning_rate_opacity, "name": "opacity"},
        {"params": [betas], "lr": config.learning_rate_beta, "name": "beta"},
    ]
    if opt_type == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.5)
    if opt_type == "adam":
        return torch.optim.Adam(param_groups)
    raise ValueError(f"Unknown optimizer_type: {config.optimizer_type}")




class MaskedAdam(torch.optim.Optimizer):
    """
    Adam that supports per-surfel masked updates.

    Expected attributes on each Parameter before step():
      - param.updateMask: bool tensor broadcastable to param.grad shape
      - param.surfelMask: bool tensor of shape (N,) (one bool per surfel/row)

    Behavior:
      - Only rows with surfelMask=True have their Adam moments updated.
      - Only entries with updateMask=True are updated in the parameter tensor.
      - Per-surfel bias correction uses a per-surfel step counter (shape (N,)).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("MaskedAdam expects dense gradients (got sparse).")

                update_mask = getattr(param, "updateMask", None)
                surfel_mask = getattr(param, "surfelMask", None)

                # If no mask provided, behave like standard Adam (single global step)
                if update_mask is None or surfel_mask is None:
                    state = self.state[param]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    step = state["step"]

                    if weight_decay != 0.0:
                        grad = grad.add(param, alpha=weight_decay)

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - (beta1 ** step)
                    bias_correction2 = 1.0 - (beta2 ** step)

                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                    step_size = lr / bias_correction1
                    param.addcdiv_(exp_avg, denom, value=-step_size)
                    continue

                # Masked mode (per-surfel steps)
                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    # One step counter per surfel (row)
                    state["surfel_step"] = torch.zeros(
                        (surfel_mask.shape[0],),
                        device=param.device,
                        dtype=torch.int64,
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                surfel_step = state["surfel_step"]  # (N,)

                # Weight decay only where we update
                if weight_decay != 0.0:
                    grad = torch.where(update_mask, grad + weight_decay * param, grad)

                # Update moments only where masked-in
                exp_avg.copy_(torch.where(update_mask, exp_avg * beta1 + (1.0 - beta1) * grad, exp_avg))
                exp_avg_sq.copy_(torch.where(update_mask, exp_avg_sq * beta2 + (1.0 - beta2) * (grad * grad), exp_avg_sq))

                # Increment per-surfel step only for surfels being updated
                surfel_step.add_(surfel_mask.to(torch.int64))

                # Bias correction per surfel, then broadcast to parameter shape
                # (N,) -> (N,1) or (N,1,1,...) to match param dims
                if grad.ndim == 1:
                    t = surfel_step.to(torch.float32)  # (N,)
                    bias_correction1 = 1.0 - torch.pow(torch.tensor(beta1, device=param.device), t)
                    bias_correction2 = 1.0 - torch.pow(torch.tensor(beta2, device=param.device), t)

                    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
                    step_size = lr / bias_correction1.clamp_min(1e-12)

                    update = step_size * (exp_avg / denom)
                    param.copy_(torch.where(update_mask, param - update, param))
                else:
                    expand_shape = (surfel_step.shape[0],) + (1,) * (grad.ndim - 1)
                    t = surfel_step.view(expand_shape).to(torch.float32)

                    beta1_t = torch.pow(torch.tensor(beta1, device=param.device), t)
                    beta2_t = torch.pow(torch.tensor(beta2, device=param.device), t)
                    bias_correction1 = 1.0 - beta1_t
                    bias_correction2 = 1.0 - beta2_t

                    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
                    step_size = lr / bias_correction1.clamp_min(1e-12)

                    update = step_size * (exp_avg / denom)
                    param.copy_(torch.where(update_mask, param - update, param))

        return loss

def compute_surfel_update_mask(
    grad_position_np: np.ndarray,
    grad_rotation_np: np.ndarray,
    grad_scales_np: np.ndarray,
    grad_albedos_np: np.ndarray,
    grad_opacities_np: np.ndarray,
    grad_betas_np: np.ndarray,
    eps: float = 0.0,
) -> np.ndarray:
    """Return a per-surfel update mask based on all gradient blocks."""
    sq = np.zeros((grad_position_np.shape[0],), dtype=np.float32)

    def add_sq(a: np.ndarray) -> None:
        aa = np.asarray(a, dtype=np.float32)
        if aa.ndim == 1:
            sq[:] += aa * aa
        else:
            sq[:] += np.sum(aa * aa, axis=1)

    add_sq(grad_position_np)
    add_sq(grad_rotation_np)
    add_sq(grad_scales_np)
    add_sq(grad_albedos_np)
    add_sq(grad_opacities_np.reshape(-1))
    add_sq(grad_betas_np.reshape(-1))

    if eps <= 0.0:
        return sq != 0.0
    return sq > (eps * eps)


def assign_numpy_gradients_to_tensors_masked(
    device: torch.device,
    positions: torch.nn.Parameter,
    rotation: torch.nn.Parameter,
    scales: torch.nn.Parameter,
    albedos: torch.nn.Parameter,
    opacities: torch.nn.Parameter,
    betas: torch.nn.Parameter,
    grad_position_np: np.ndarray,
    grad_rotation_np: np.ndarray,
    grad_scales_np: np.ndarray,
    grad_albedos_np: np.ndarray,
    grad_opacities_np: np.ndarray,
    grad_betas_np: np.ndarray,
    surfel_update_mask_np: np.ndarray,
) -> None:
    """Copy numpy gradients into .grad and attach a per-surfel update mask."""
    surfel_update_mask_t = torch.tensor(surfel_update_mask_np, device=device, dtype=torch.bool)

    def set_grad_and_mask(param: torch.nn.Parameter, grad_np: np.ndarray) -> None:
        grad_t = torch.tensor(grad_np, device=device, dtype=torch.float32)
        param.grad = grad_t
        param.surfelMask = surfel_update_mask_t
        if grad_t.ndim == 1:
            param.updateMask = surfel_update_mask_t
        else:
            expand_shape = (surfel_update_mask_t.shape[0],) + (1,) * (grad_t.ndim - 1)
            param.updateMask = surfel_update_mask_t.view(expand_shape).expand_as(grad_t)

    set_grad_and_mask(positions, grad_position_np)
    set_grad_and_mask(rotation, grad_rotation_np)
    set_grad_and_mask(scales, grad_scales_np)
    set_grad_and_mask(albedos, grad_albedos_np)
    set_grad_and_mask(opacities, grad_opacities_np.reshape(-1))
    set_grad_and_mask(betas, grad_betas_np.reshape(-1))



def create_masked_optimizer(
    config: OptimizationConfig,
    positions: torch.nn.Parameter,
    rotation: torch.nn.Parameter,
    scales: torch.nn.Parameter,
    albedos: torch.nn.Parameter,
    opacities: torch.nn.Parameter,
    betas: torch.nn.Parameter,
    powers: torch.nn.Parameter,
) -> torch.optim.Optimizer:
    opt_type = config.optimizer_type.lower()

    param_groups = [
        {"params": [positions], "lr": config.learning_rate_position, "name": "position"},
        {"params": [rotation], "lr": config.learning_rate_rotation, "name": "rotation"},
        {"params": [scales], "lr": config.learning_rate_scale, "name": "scale"},
        {"params": [albedos], "lr": config.learning_rate_albedo, "name": "albedo"},
        {"params": [opacities], "lr": config.learning_rate_opacity, "name": "opacity"},
        {"params": [betas], "lr": config.learning_rate_beta, "name": "beta"},
        {"params": [powers], "lr": 0.0, "name": "power"},
    ]

    if opt_type == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.8)
    if opt_type == "adam":
        return torch.optim.Adam(param_groups)
    if opt_type in ("masked_adam", "nullgrad_adam", "sparse_adam"):
        return MaskedAdam(param_groups)
    raise ValueError(f"Unknown optimizer_type: {config.optimizer_type}")

def make_constant_scale_func() -> Callable[[int], float]:
    def schedule(_: int) -> float:
        return 1.0

    return schedule


def make_exponential_scale_func(
        scale_init: float,
        scale_final: float,
        max_steps: int,
        start_iteration: int = 0,
) -> Callable[[int], float]:
    if scale_init <= 0.0:
        raise ValueError(f"scale_init must be positive, got {scale_init}")
    if scale_final <= 0.0:
        raise ValueError(f"scale_final must be positive, got {scale_final}")
    if max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {max_steps}")

    if max_steps == 0:
        def immediate_schedule(iteration: int) -> float:
            return scale_init if iteration < start_iteration else scale_final

        return immediate_schedule

    def schedule(iteration: int) -> float:
        t = np.clip(float(iteration - start_iteration) / float(max_steps), 0.0, 1.0)
        return float(
            np.exp(
                np.log(scale_init) * (1.0 - t)
                + np.log(scale_final) * t
            )
        )

    return schedule


def get_required_learning_rate(config: OptimizationConfig, attribute_name: str) -> float:
    learning_rate = getattr(config, attribute_name)
    if learning_rate is None:
        raise ValueError(f"config.{attribute_name} must be resolved before creating LR schedules.")
    return float(learning_rate)


def create_learning_rate_schedules(config: OptimizationConfig) -> dict[str, object]:
    base_learning_rates = {
        "position": get_required_learning_rate(config, "learning_rate_position"),
        "rotation": get_required_learning_rate(config, "learning_rate_rotation"),
        "scale": get_required_learning_rate(config, "learning_rate_scale"),
        "albedo": get_required_learning_rate(config, "learning_rate_albedo"),
        "opacity": get_required_learning_rate(config, "learning_rate_opacity"),
        "beta": get_required_learning_rate(config, "learning_rate_beta"),
        "power": 0.0,
    }

    use_global_lr_schedule = bool(getattr(config, "use_global_lr_schedule", False))
    if use_global_lr_schedule:
        global_lr_scale_func = make_exponential_scale_func(
            scale_init=float(getattr(config, "global_lr_scale_init", 1.0)),
            scale_final=float(getattr(config, "global_lr_scale_final", 1.0)),
            start_iteration=int(getattr(config, "global_lr_start_iteration", 0)),
            max_steps=int(getattr(config, "global_lr_max_steps", 0)),
        )
    else:
        global_lr_scale_func = make_constant_scale_func()

    parameter_lr_scale_funcs: dict[str, Callable[[int], float]] = {}

    if bool(getattr(config, "use_position_lr_schedule", False)):
        parameter_lr_scale_funcs["position"] = make_exponential_scale_func(
            scale_init=float(getattr(config, "position_lr_scale_init", 1.0)),
            scale_final=float(getattr(config, "position_lr_scale_final", 1.0)),
            start_iteration=0,
            max_steps=int(getattr(config, "position_lr_max_steps", 0)),
        )

    return {
        "base_learning_rates": base_learning_rates,
        "global_lr_scale_func": global_lr_scale_func,
        "parameter_lr_scale_funcs": parameter_lr_scale_funcs,
    }


def update_optimizer_learning_rates(
        optimizer: torch.optim.Optimizer,
        learning_rate_schedules: dict[str, object],
        iteration: int,
) -> dict[str, float]:
    base_learning_rates = learning_rate_schedules.get("base_learning_rates", {})
    global_lr_scale_func = learning_rate_schedules.get("global_lr_scale_func", make_constant_scale_func())
    parameter_lr_scale_funcs = learning_rate_schedules.get("parameter_lr_scale_funcs", {})

    if not callable(global_lr_scale_func):
        raise RuntimeError("global_lr_scale_func must be callable.")

    global_lr_scale = float(global_lr_scale_func(iteration))
    active_learning_rates: dict[str, float] = {
        "global_lr_scale": global_lr_scale,
    }

    for parameter_group in optimizer.param_groups:
        group_name = parameter_group.get("name", None)
        if group_name is None:
            raise RuntimeError(
                "Optimizer parameter group is missing a 'name'. "
                "Add names such as 'position', 'rotation', 'scale', 'albedo', 'opacity', 'beta', or 'power'."
            )

        group_name = str(group_name)

        if group_name not in base_learning_rates:
            raise RuntimeError(f"No base learning rate registered for parameter group '{group_name}'.")

        base_learning_rate = float(base_learning_rates[group_name])
        parameter_lr_scale_func = parameter_lr_scale_funcs.get(group_name, make_constant_scale_func())

        if not callable(parameter_lr_scale_func):
            raise RuntimeError(f"LR scale function for parameter group '{group_name}' must be callable.")

        parameter_lr_scale = float(parameter_lr_scale_func(iteration))
        learning_rate = base_learning_rate * global_lr_scale * parameter_lr_scale

        parameter_group["lr"] = learning_rate
        active_learning_rates[group_name] = learning_rate

    return active_learning_rates