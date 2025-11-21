import numpy as np
import torch

from config import OptimizationConfig
from optimizers import create_optimizer


def densify_and_prune_points(
    positions: torch.nn.Parameter,
    tangent_u: torch.nn.Parameter,
    tangent_v: torch.nn.Parameter,
    scales: torch.nn.Parameter,
    colors: torch.nn.Parameter,
    grad_position_np: np.ndarray,
    grad_scales_np: np.ndarray,
    optimizer: torch.optim.Optimizer,
    config: OptimizationConfig,
    device: torch.device,
) -> tuple[
    torch.nn.Parameter,
    torch.nn.Parameter,
    torch.nn.Parameter,
    torch.nn.Parameter,
    torch.nn.Parameter,
    torch.optim.Optimizer,
    int,
    bool,
]:
    """
    Gradient-based adaptive density control for surfels with 2D scale.

    - Uses ||dL/dp|| and ||dL/ds|| as cues.
    - Densifies (splits/clones) high-gradient surfels.
    - Prunes tiny, low-gradient surfels.
    """

    with torch.no_grad():
        positions_np = positions.detach().cpu().numpy()
        tangent_u_np = tangent_u.detach().cpu().numpy()
        tangent_v_np = tangent_v.detach().cpu().numpy()
        scales_np = scales.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy()

        num_points = positions_np.shape[0]
        if num_points == 0:
            return (
                positions,
                tangent_u,
                tangent_v,
                scales,
                colors,
                optimizer,
                num_points,
                False,
            )

        # ------------------------------------------------------------------
        # 1) Compute per-point gradient metric
        # ------------------------------------------------------------------
        grad_position_norm = np.linalg.norm(grad_position_np, axis=1)
        grad_scale_norm = np.linalg.norm(grad_scales_np, axis=1)

        grad_scale_weight = 0.1  # position dominates
        combined_gradient_metric = (
            grad_position_norm + grad_scale_weight * grad_scale_norm
        )

        # --------------------------------------------------------------
        # 2) Densification: fixed gradient threshold
        # --------------------------------------------------------------
        fixed_grad_threshold = 3e-3  # ← tune this value

        densify_indices = np.where(combined_gradient_metric > fixed_grad_threshold)[0]

        new_positions_list = [positions_np]
        new_tangent_u_list = [tangent_u_np]
        new_tangent_v_list = [tangent_v_np]
        new_scales_list = [scales_np]
        new_colors_list = [colors_np]

        # Parameters for splitting
        scale_shrink_factor = 0.3
        rng = np.random.default_rng()

        for index in densify_indices:
            parent_position = positions_np[index]
            parent_tangent_u = tangent_u_np[index]
            parent_tangent_v = tangent_v_np[index]
            parent_scale = scales_np[index]  # (su, sv)
            parent_color = colors_np[index]

            # Random offset inside ellipsoidal footprint in tangent plane
            random_u = rng.uniform(-0.5, 0.5) * 2
            random_v = rng.uniform(-0.5, 0.5) * 2
            offset_vector = (
                random_u * parent_scale[0] * parent_tangent_u
                + random_v * parent_scale[1] * parent_tangent_v
            )

            child_position = parent_position + offset_vector
            child_tangent_u = parent_tangent_u.copy()
            child_tangent_v = parent_tangent_v.copy()
            child_scale = parent_scale * scale_shrink_factor
            child_color = parent_color.copy()

            # Optionally also shrink parent scale to focus detail
            scales_np[index] = parent_scale * scale_shrink_factor

            new_positions_list.append(child_position[None, :])
            new_tangent_u_list.append(child_tangent_u[None, :])
            new_tangent_v_list.append(child_tangent_v[None, :])
            new_scales_list.append(child_scale[None, :])
            new_colors_list.append(child_color[None, :])

        positions_np_extended = np.concatenate(new_positions_list, axis=0)
        tangent_u_np_extended = np.concatenate(new_tangent_u_list, axis=0)
        tangent_v_np_extended = np.concatenate(new_tangent_v_list, axis=0)
        scales_np_extended = np.concatenate(new_scales_list, axis=0)
        colors_np_extended = np.concatenate(new_colors_list, axis=0)

        # ------------------------------------------------------------------
        # 3) Pruning: tiny, low-gradient surfels
        # ------------------------------------------------------------------
        extended_grad_position_norm = np.concatenate(
            [
                grad_position_norm,
                np.zeros(
                    positions_np_extended.shape[0] - num_points, dtype=np.float32
                ),
            ],
            axis=0,
        )
        extended_grad_scale_norm = np.concatenate(
            [
                grad_scale_norm,
                np.zeros(
                    scales_np_extended.shape[0] - num_points, dtype=np.float32
                ),
            ],
            axis=0,
        )

        scale_norm_extended = np.linalg.norm(scales_np_extended, axis=1)

        # Heuristic thresholds
        min_scale_threshold = 0.001  # surfels much smaller than this get pruned
        prune_grad_threshold = np.percentile(
            combined_gradient_metric, 25.0
        )  # lower quartile

        prune_mask = (
            (scale_norm_extended < min_scale_threshold)
            & (extended_grad_position_norm < prune_grad_threshold)
            & (extended_grad_scale_norm < prune_grad_threshold)
        )

        # Always keep at least some points
        if np.all(prune_mask):
            prune_mask[:] = False

        keep_mask = ~prune_mask

        positions_np_final = positions_np_extended[keep_mask]
        tangent_u_np_final = tangent_u_np_extended[keep_mask]
        tangent_v_np_final = tangent_v_np_extended[keep_mask]
        scales_np_final = scales_np_extended[keep_mask]
        colors_np_final = colors_np_extended[keep_mask]

        new_num_points = positions_np_final.shape[0]
        topology_changed = new_num_points != num_points

        # ------------------------------------------------------------------
        # 4) Rebuild Parameters and Optimizer if topology changed
        # ------------------------------------------------------------------
        if topology_changed:
            positions_new = torch.nn.Parameter(
                torch.tensor(
                    positions_np_final, device=device, dtype=torch.float32
                )
            )
            tangent_u_new = torch.nn.Parameter(
                torch.tensor(
                    tangent_u_np_final, device=device, dtype=torch.float32
                )
            )
            tangent_v_new = torch.nn.Parameter(
                torch.tensor(
                    tangent_v_np_final, device=device, dtype=torch.float32
                )
            )
            scales_new = torch.nn.Parameter(
                torch.tensor(
                    scales_np_final, device=device, dtype=torch.float32
                )
            )
            colors_new = torch.nn.Parameter(
                torch.tensor(
                    colors_np_final, device=device, dtype=torch.float32
                )
            )

            optimizer_new = create_optimizer(
                config,
                positions_new,
                tangent_u_new,
                tangent_v_new,
                scales_new,
                colors_new,
            )

            return (
                positions_new,
                tangent_u_new,
                tangent_v_new,
                scales_new,
                colors_new,
                optimizer_new,
                new_num_points,
                True,
            )
        else:
            # No topology change: keep old parameters / optimizer
            return (
                positions,
                tangent_u,
                tangent_v,
                scales,
                colors,
                optimizer,
                num_points,
                False,
            )
