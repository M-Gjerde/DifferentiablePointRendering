# repulsion.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_elliptical_repulsion_loss(
    positions: torch.Tensor,
    tangent_u: torch.Tensor,
    tangent_v: torch.Tensor,
    scales: torch.Tensor,
    radius_factor: float = 1.0,
    repulsion_weight: float = 1e-3,
    contact_distance: float = 2.0,  # d_hat at which overlap ~ 0%
) -> torch.Tensor:
    """
    Elliptical repulsion in each surfel's local frame.

    We treat d_hat as a normalized center distance:
        d_hat ~  distance / (radius_factor * surfel_radius)

    - If d_hat >= contact_distance:  0 penalty   (no overlap)
    - If d_hat <  contact_distance:  penalty grows as centers get closer.

    contact_distance ≈ 2.0 means “two radii apart”: just-touching → 0 penalty,
    with increasing penalty as overlap grows.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N,3), got {tuple(positions.shape)}")

    num_points = positions.shape[0]
    if num_points <= 1:
        return positions.new_tensor(0.0)

    # Detach frames and scales: gradients only w.r.t. positions
    t_u = F.normalize(tangent_u.detach(), dim=-1)
    t_v = F.normalize(tangent_v.detach(), dim=-1)
    n = F.normalize(torch.cross(t_u, t_v, dim=-1), dim=-1)
    s = scales.detach()

    su = s[:, 0].clamp_min(1e-6)
    sv = s[:, 1].clamp_min(1e-6)
    smax = torch.max(su, sv)

    # Pairwise differences delta_ij = x_i - x_j
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N,N,3)

    # Project onto surfel i's local frame
    t_u_i = t_u.unsqueeze(1)
    t_v_i = t_v.unsqueeze(1)
    n_i = n.unsqueeze(1)

    u_ij = (diff * t_u_i).sum(dim=-1)  # (N,N)
    v_ij = (diff * t_v_i).sum(dim=-1)
    w_ij = (diff * n_i).sum(dim=-1)

    su_i = su.view(-1, 1)
    sv_i = sv.view(-1, 1)
    smax_i = smax.view(-1, 1)

    u_hat = u_ij / (radius_factor * su_i)
    v_hat = v_ij / (radius_factor * sv_i)
    w_hat = w_ij / (radius_factor * smax_i)

    d_hat_sq = u_hat * u_hat + v_hat * v_hat + w_hat * w_hat
    d_hat = torch.sqrt(d_hat_sq + 1e-9)

    device = positions.device
    pair_mask = torch.triu(
        torch.ones_like(d_hat, dtype=torch.bool, device=device),
        diagonal=1,
    )

    # Only pairs *inside* the contact distance get penalized
    inside_mask = pair_mask & (d_hat < contact_distance)

    if not inside_mask.any():
        return positions.new_tensor(0.0)

    d_inside = d_hat[inside_mask]

    # Overlap depth in normalized units: 0 at contact_distance, 1 at d_hat = 0
    depth = (contact_distance - d_inside) / contact_distance  # in (0, 1]
    loss_pairs = 0.5 * depth * depth
    loss = repulsion_weight * loss_pairs.sum()

    return loss
