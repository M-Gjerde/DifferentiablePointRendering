# repulsion.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_elliptical_repulsion_loss(
    positions: torch.Tensor,
    tangent_u: torch.Tensor,
    tangent_v: torch.Tensor,
    scales: torch.Tensor,
    radius_factor: float = 2.0,
    repulsion_weight: float = 1e-3,
) -> torch.Tensor:
    """
    Elliptical repulsion in each surfel's local frame.

    For surfel i we define a normalized coordinate of point j:

        delta_ij = x_i - x_j
        u_ij = <delta_ij, t_u_i>
        v_ij = <delta_ij, t_v_i>
        w_ij = <delta_ij, n_i>,      n_i = normalize(t_u_i x t_v_i)

        su_i, sv_i = scales_i
        smax_i = max(su_i, sv_i)

        u_hat = u_ij / (radius_factor * su_i)
        v_hat = v_ij / (radius_factor * sv_i)
        w_hat = w_ij / (radius_factor * smax_i)

        d_hat_ij = sqrt(u_hat^2 + v_hat^2 + w_hat^2)

    We penalize pairs with d_hat_ij < 1:

        E = 0.5 * λ * sum_{i<j, d_hat_ij < 1} (1 - d_hat_ij)^2

    Notes:
      - Only depends on positions for gradients; tangents/scales are detached.
      - Complexity O(N^2); okay for moderate N.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N,3), got {tuple(positions.shape)}")

    num_points = positions.shape[0]
    if num_points <= 1:
        return positions.new_tensor(0.0)

    # Detach frames and scales: we only want gradients w.r.t. positions
    t_u = F.normalize(tangent_u.detach(), dim=-1)
    t_v = F.normalize(tangent_v.detach(), dim=-1)
    n = F.normalize(torch.cross(t_u, t_v, dim=-1), dim=-1)
    s = scales.detach()

    su = s[:, 0].clamp_min(1e-6)  # (N,)
    sv = s[:, 1].clamp_min(1e-6)
    smax = torch.max(su, sv)

    # Pairwise differences delta_ij = x_i - x_j
    # shapes: (N,1,3) - (1,N,3) -> (N,N,3)
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)

    # Project onto surfel i's local frame
    t_u_i = t_u.unsqueeze(1)  # (N,1,3)
    t_v_i = t_v.unsqueeze(1)
    n_i = n.unsqueeze(1)

    u_ij = (diff * t_u_i).sum(dim=-1)  # (N,N)
    v_ij = (diff * t_v_i).sum(dim=-1)
    w_ij = (diff * n_i).sum(dim=-1)

    # Normalize by surfel scales (per i, broadcast over j)
    su_i = su.view(-1, 1)      # (N,1)
    sv_i = sv.view(-1, 1)
    smax_i = smax.view(-1, 1)

    u_hat = u_ij / (radius_factor * su_i)
    v_hat = v_ij / (radius_factor * sv_i)
    w_hat = w_ij / (radius_factor * smax_i)

    d_hat_sq = u_hat * u_hat + v_hat * v_hat + w_hat * w_hat
    d_hat = torch.sqrt(d_hat_sq + 1e-9)

    # We only count each pair once: i < j
    device = positions.device
    pair_mask = torch.triu(torch.ones_like(d_hat, dtype=torch.bool, device=device), diagonal=1)

    # Only repulse pairs inside the ellipse (d_hat < 1)
    inside_mask = pair_mask & (d_hat < 1.0)

    if not inside_mask.any():
        return positions.new_tensor(0.0)

    d_inside = d_hat[inside_mask]

    # Smooth quadratic penalty toward boundary d_hat = 1
    loss_pairs = 0.5 * (1.0 - d_inside) ** 2  # (M,)
    loss = repulsion_weight * loss_pairs.sum()

    return loss
