import numpy as np
import torch
from typing import Optional, Dict, Any


def densify_points_long_axis_split(
    positions: torch.nn.Parameter,
    tangent_u: torch.nn.Parameter,
    tangent_v: torch.nn.Parameter,
    scales: torch.nn.Parameter,
    colors: torch.nn.Parameter,
    opacities: torch.nn.Parameter,
    *,
    importance: Optional[torch.Tensor] = None,  # (N,) scores, EDC-style
    max_new_points: Optional[int] = None,
    split_distance: float = 0.6,
    minor_axis_shrink: float = 0.85,
    opacity_reduction: float = 0.6,
    min_long_axis_scale: float = 0.0,  # ignore “dust” surfels if desired
    grad_threshold: float = 0.0,       # minimal importance to be considered
) -> Optional[Dict[str, Any]]:
    """
    Long-Axis Split style densification for your 2D surfels.

    - No area_threshold prefilter.
    - Uses `importance` (EDC-like) if given.
    - Uses `max_new_points` as a budget: each parent -> 2 children.
    - Optionally skips tiny surfels via `min_long_axis_scale`.

    Returns:
        None if no densification.

        Otherwise:
        {
            "prune_indices": np.ndarray [K],       # parent indices to remove
            "new": {
                "position":  np.ndarray [2K, 3],
                "tangent_u": np.ndarray [2K, 3],
                "tangent_v": np.ndarray [2K, 3],
                "scale":     np.ndarray [2K, 2],
                "color":     np.ndarray [2K, 3],
                "opacity":   np.ndarray [2K],
            }
        }
    """

    with torch.no_grad():
        positions_np = positions.detach().cpu().numpy()
        tangent_u_np = tangent_u.detach().cpu().numpy()
        tangent_v_np = tangent_v.detach().cpu().numpy()
        scales_np = scales.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy()
        opacities_np = opacities.detach().cpu().numpy()

        if importance is not None:
            importance_np = importance.detach().cpu().numpy().reshape(-1)
        else:
            importance_np = None

    number_of_points = positions_np.shape[0]
    if number_of_points == 0:
        return None

    # --- 1) Basic geometry-derived info: long-axis scale per point ---
    long_axis_scales = np.maximum(scales_np[:, 0], scales_np[:, 1])  # (N,)

    # Start from all indices
    candidate_mask = long_axis_scales >= min_long_axis_scale
    candidate_indices = np.nonzero(candidate_mask)[0].astype(np.int64)

    if candidate_indices.size == 0:
        return None

    # --- 2) If we have importance (EDC-style), restrict by grad_threshold ---
    if importance_np is not None:
        candidate_scores = importance_np[candidate_indices]
        # Filter by gradient threshold
        print(candidate_scores)
        if grad_threshold > 0.0:
            grad_mask = candidate_scores >= grad_threshold
            candidate_indices = candidate_indices[grad_mask]
            candidate_scores = candidate_scores[grad_mask]
            if candidate_indices.size == 0:
                return None
    else:
        candidate_scores = None  # fallback: use geometry only later

    # --- 3) Apply budget on parents: each parent → 2 children ---
    if max_new_points is not None:
        max_parents = max_new_points // 2
        if max_parents <= 0:
            return None

        if candidate_indices.size > max_parents:
            if candidate_scores is not None:
                # Select top-k by importance score (descending)
                order = np.argsort(-candidate_scores)  # descending
                selected = order[:max_parents]
                candidate_indices = candidate_indices[selected]
            else:
                # Fallback: largest long-axis scales first
                ca_long = long_axis_scales[candidate_indices]
                order = np.argsort(-ca_long)  # descending
                selected = order[:max_parents]
                candidate_indices = candidate_indices[selected]

    # Canonical, sorted parent indices (nice for remove_points)
    parent_indices = np.unique(candidate_indices.astype(np.int64))
    if parent_indices.size == 0:
        return None

    # --- 4) Perform long-axis split ---
    new_positions_list = []
    new_tangent_u_list = []
    new_tangent_v_list = []
    new_scales_list = []
    new_colors_list = []
    new_opacities_list = []

    for parent_index in parent_indices:
        parent_position = positions_np[parent_index]
        parent_tangent_u = tangent_u_np[parent_index]
        parent_tangent_v = tangent_v_np[parent_index]
        parent_scale = scales_np[parent_index]  # [s_u, s_v]
        parent_color = colors_np[parent_index]
        parent_opacity = float(opacities_np[parent_index])

        scale_u = float(parent_scale[0])
        scale_v = float(parent_scale[1])

        # Determine long axis in tangent plane
        if scale_u >= scale_v:
            long_axis_vector = parent_tangent_u
            long_scale = scale_u
            short_scale = scale_v
            long_is_u = True
        else:
            long_axis_vector = parent_tangent_v
            long_scale = scale_v
            short_scale = scale_u
            long_is_u = False

        # Offset distance along long axis (≈3σ * split_distance)
        effective_radius = 3.0 * long_scale
        offset_magnitude = split_distance * effective_radius
        offset_vector = offset_magnitude * long_axis_vector

        child_position_pos = parent_position + offset_vector
        child_position_neg = parent_position - offset_vector

        # Child scales
        new_long_scale = long_scale * (1.0 - split_distance)
        new_short_scale = short_scale * minor_axis_shrink

        if long_is_u:
            child_scale = np.array([new_long_scale, new_short_scale], dtype=np.float32)
        else:
            child_scale = np.array([new_short_scale, new_long_scale], dtype=np.float32)

        # Child opacity
        child_opacity = np.clip(parent_opacity * opacity_reduction, 0.0, 1.0)

        # Record two children
        for child_position in (child_position_pos, child_position_neg):
            new_positions_list.append(child_position)
            new_tangent_u_list.append(parent_tangent_u)
            new_tangent_v_list.append(parent_tangent_v)
            new_scales_list.append(child_scale)
            new_colors_list.append(parent_color)
            new_opacities_list.append(child_opacity)

    if not new_positions_list:
        return None

    new_positions_np = np.asarray(new_positions_list, dtype=np.float32)
    new_tangent_u_np = np.asarray(new_tangent_u_list, dtype=np.float32)
    new_tangent_v_np = np.asarray(new_tangent_v_list, dtype=np.float32)
    new_scales_np = np.asarray(new_scales_list, dtype=np.float32)
    new_colors_np = np.asarray(new_colors_list, dtype=np.float32)
    new_opacities_np = np.asarray(new_opacities_list, dtype=np.float32)

    return {
        "prune_indices": parent_indices,
        "new": {
            "position": new_positions_np,
            "tangent_u": new_tangent_u_np,
            "tangent_v": new_tangent_v_np,
            "scale": new_scales_np,
            "color": new_colors_np,
            "opacity": new_opacities_np,
        },
    }


def compute_prune_indices_by_opacity(
    opacities: torch.Tensor,
    min_opacity: float,
    use_quantile: bool = True,
    max_fraction_to_prune: float = 0.3,
    min_points_to_keep: int = 1,
) -> np.ndarray:
    """
    Decide which points to prune based on opacity (EDC-style).

    opacities:
        Torch tensor of shape (N,) or (N,1) with values in [0,1].

    If use_quantile = True:
        min_opacity is interpreted as a quantile q in [0,1].
        We prune the lowest-q opacities, but never more than
        max_fraction_to_prune * N, and we always keep at least
        min_points_to_keep points.

    If use_quantile = False:
        min_opacity is an absolute threshold in [0,1], and we prune
        opacities < min_opacity (capped as above).

    Returns:
        np.ndarray[int64] of indices to prune (possibly empty).
    """
    with torch.no_grad():
        opa = opacities.detach().cpu().numpy().reshape(-1)  # shape (N,)

    num_points = opa.shape[0]
    if num_points == 0:
        return np.zeros((0,), dtype=np.int64)

    # Determine threshold
    if use_quantile:
        q = float(min_opacity)
        q = max(0.0, min(1.0, q))
        threshold = float(np.quantile(opa, q))
    else:
        threshold = float(min_opacity)

    # Candidates whose opacity is below threshold
    candidate_mask = opa < threshold
    candidate_indices = np.nonzero(candidate_mask)[0]  # int64

    if candidate_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)

    # Cap how many we can prune
    max_prune_by_fraction = int(max_fraction_to_prune * num_points)
    # Ensure we keep at least min_points_to_keep
    max_prune_by_min_points = max(0, num_points - min_points_to_keep)
    max_prune = min(max_prune_by_fraction, max_prune_by_min_points)

    if max_prune <= 0:
        return np.zeros((0,), dtype=np.int64)

    # If not too many candidates, prune all of them (still obeying cap above)
    if candidate_indices.size <= max_prune:
        return candidate_indices.astype(np.int64)

    # Otherwise, prune the lowest-opacity subset
    order = np.argsort(opa[candidate_indices])  # ascending by opacity
    selected = candidate_indices[order[:max_prune]]
    return selected.astype(np.int64)
