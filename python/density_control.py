import numpy as np
import torch
from typing import Optional, Any


def project_gradient_to_surfel_tangent_plane_np(
        grad_position_np: np.ndarray,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
) -> np.ndarray:
    """
    Project world-space position gradients onto each surfel's tangent plane.

    Returns:
        projected_grad_np with shape (N, 3).
    """
    grad_np = np.asarray(grad_position_np, dtype=np.float32, order="C")

    with torch.no_grad():
        device = tangent_u.device

        g = torch.as_tensor(grad_np, device=device, dtype=torch.float32)

        tu = torch.nn.functional.normalize(tangent_u.detach(), dim=1)
        tv = torch.nn.functional.normalize(tangent_v.detach(), dim=1)

        projected = (
            torch.sum(g * tu, dim=1, keepdim=True) * tu
            +
            torch.sum(g * tv, dim=1, keepdim=True) * tv
        )

        return projected.detach().cpu().numpy().astype(np.float32)

def make_under_reconstruction_clones(
        positions,
        tangent_u,
        tangent_v,
        scales,
        albedos,
        opacities,
        betas,
        powers,
        grad_position_np,
        trainable_surfel_mask,
        grad_threshold,
        small_scale_threshold,
        max_clone_fraction=0.05,
        clone_offset_scale=0.25,
        clone_scale_factor=1.6,
        normal_perturbation=2.0e-5,
        tangent_project_position_grad=True,
        selection_score_np=None
) -> dict[str, dict[str, Any] | Any] | None:
    """
    Clone-only densification for under-reconstruction.

    Selection:
        large position-gradient magnitude
        small surfel scale
        trainable surfel

    The clone copies all local parameters and is optionally displaced
    a small distance along the local tangent-projected descent direction.
    """

    with torch.no_grad():
        device = positions.device


        grad_pos = torch.as_tensor(
            grad_position_np,
            device=device,
            dtype=torch.float32,
        )

        if selection_score_np is None:
            selection_score = torch.linalg.norm(grad_pos, dim=1)
        else:
            selection_score = torch.as_tensor(
                selection_score_np,
                device=device,
                dtype=torch.float32,
            ).reshape(-1)
        max_scale = torch.max(scales, dim=1).values

        selected = (
                torch.isfinite(selection_score)
                & (selection_score >= grad_threshold)
                & (max_scale <= small_scale_threshold)
                & trainable_surfel_mask
        )

        selected_idx = torch.nonzero(selected, as_tuple=False).flatten()
        if selected_idx.numel() == 0:
            return None

        # Cap clone count to avoid explosive growth.
        n_points = positions.shape[0]
        max_new = max(1, int(max_clone_fraction * float(n_points)))

        if selected_idx.numel() > max_new:
            # Keep strongest-gradient candidates.
            selected_grad = selection_score[selected_idx]
            keep = torch.topk(selected_grad, k=max_new, largest=True).indices
            selected_idx = selected_idx[keep]

        p = positions[selected_idx].detach().clone()
        tu = tangent_u[selected_idx].detach().clone()
        tv = tangent_v[selected_idx].detach().clone()
        sc = scales[selected_idx].detach().clone()
        alb = albedos[selected_idx].detach().clone()
        opa = opacities[selected_idx].detach().clone()
        be = betas[selected_idx].detach().clone()
        pow_ = powers[selected_idx].detach().clone()

        g = grad_pos[selected_idx]

        # For a minimization objective, grad_position_np = dL/dp.
        # The optimizer moves in -grad direction.
        descent = -g

        if tangent_project_position_grad:
            tu_n = torch.nn.functional.normalize(tu, dim=1)
            tv_n = torch.nn.functional.normalize(tv, dim=1)

            descent = (
                torch.sum(descent * tu_n, dim=1, keepdim=True) * tu_n
                +
                torch.sum(descent * tv_n, dim=1, keepdim=True) * tv_n
            )

        descent_norm = torch.linalg.norm(descent, dim=1, keepdim=True)
        valid_dir = descent_norm > 1.0e-12

        direction = torch.zeros_like(descent)
        direction[valid_dir[:, 0]] = descent[valid_dir[:, 0]] / descent_norm[valid_dir[:, 0]]

        # Move by a fraction of local surfel size in the tangent plane.
        local_radius = torch.min(sc, dim=1).values[:, None]
        tangent_offset = clone_offset_scale * local_radius * direction

        # Small normal perturbation to avoid BVH self-intersection / coincident primitive issues.
        normal = torch.cross(tu, tv, dim=1)
        normal_norm = torch.linalg.norm(normal, dim=1, keepdim=True)
        valid_normal = normal_norm > 1.0e-12

        normal_dir = torch.zeros_like(normal)
        normal_dir[valid_normal[:, 0]] = normal[valid_normal[:, 0]] / normal_norm[valid_normal[:, 0]]

        normal_offset = float(normal_perturbation) * normal_dir

        new_positions = p + tangent_offset + normal_offset

        # Optional 3DGS split-like shrinkage for the clone.
        new_sc = sc / float(clone_scale_factor)

        return {
            "new": {
                "position": new_positions.detach().cpu().numpy().astype(np.float32),
                "tangent_u": tu.detach().cpu().numpy().astype(np.float32),
                "tangent_v": tv.detach().cpu().numpy().astype(np.float32),
                "scale": new_sc.detach().cpu().numpy().astype(np.float32),
                "albedo": alb.detach().cpu().numpy().astype(np.float32),
                "opacity": opa.detach().cpu().numpy().reshape(-1).astype(np.float32),
                "beta": be.detach().cpu().numpy().reshape(-1).astype(np.float32),
                "power": pow_.detach().cpu().numpy().reshape(-1).astype(np.float32),
            },
            "source_index": selected_idx.detach().cpu().numpy().astype(np.int64),
            "grad_norm": selection_score[selected_idx].detach().cpu().numpy().astype(np.float32),
        }

def add_densification_stats_np(
        grad_position_np: np.ndarray,
        trainable_surfel_mask: torch.Tensor,
        accum_np: np.ndarray,
        denom_np: np.ndarray,
        update_only_nonzero: bool = True,
) -> None:
    """
    Accumulate per-primitive position-gradient magnitudes for densification.

    This is a density-control statistic, not the optimizer gradient.
    Recommended input: photo_gradients["position"], not total_gradients["position"].
    """
    grad_position_np = np.asarray(grad_position_np, dtype=np.float32, order="C")

    grad_norm_np = np.linalg.norm(grad_position_np, axis=1, keepdims=True)
    grad_norm_np = np.nan_to_num(
        grad_norm_np,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    trainable_np = trainable_surfel_mask.detach().cpu().numpy().astype(bool).reshape(-1, 1)

    update_mask_np = trainable_np & np.isfinite(grad_norm_np)

    if update_only_nonzero:
        update_mask_np = update_mask_np & (grad_norm_np > 0.0)

    accum_np[update_mask_np] += grad_norm_np[update_mask_np]
    denom_np[update_mask_np] += 1.0

def compute_prune_indices_by_opacity(
        opacities: torch.Tensor,
        min_opacity: float,
        use_quantile: bool = False,
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
    candidate_mask = opa <= threshold
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


def compute_prune_indices_by_degenerate_scale(
        scales: torch.Tensor,
        *,
        min_scale: float = 1.0e-5,
        trainable_mask: Optional[torch.Tensor] = None,
        min_points_to_keep: int = 1,
) -> np.ndarray:
    """
    Prune surfels where either scale parameter is degenerate.

    A surfel is considered degenerate if:

        scale_u <= min_scale OR scale_v <= min_scale

    Non-finite scale values are also treated as degenerate.

    Args:
        scales:
            Torch tensor with shape (N, 2).

        min_scale:
            Minimum valid scale. Use 1.0e-5 to prune zero-scale and near-zero-scale surfels.

        trainable_mask:
            Optional bool tensor with shape (N,). If provided, only trainable surfels
            are eligible for pruning. This protects emissive/light surfels.

        min_points_to_keep:
            Safety guard to avoid pruning all points.

    Returns:
        np.ndarray[int64] of indices to prune.
    """
    with torch.no_grad():
        scales_np = scales.detach().cpu().numpy().astype(np.float32, copy=False)

        if trainable_mask is not None:
            trainable_mask_np = trainable_mask.detach().cpu().numpy().astype(bool)
        else:
            trainable_mask_np = np.ones((scales_np.shape[0],), dtype=bool)

    if scales_np.ndim != 2 or scales_np.shape[1] != 2:
        raise ValueError(
            f"Expected scales to have shape (N, 2), got: {scales_np.shape}"
        )

    num_points = scales_np.shape[0]
    if num_points == 0:
        return np.zeros((0,), dtype=np.int64)

    finite_mask = np.isfinite(scales_np).all(axis=1)
    degenerate_mask = (
        (~finite_mask)
        | (scales_np[:, 0] <= min_scale)
        | (scales_np[:, 1] <= min_scale)
    )

    candidate_mask = degenerate_mask & trainable_mask_np
    candidate_indices = np.nonzero(candidate_mask)[0].astype(np.int64)

    if candidate_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)

    max_prune_by_min_points = max(0, num_points - min_points_to_keep)

    if max_prune_by_min_points <= 0:
        return np.zeros((0,), dtype=np.int64)

    if candidate_indices.size <= max_prune_by_min_points:
        return candidate_indices.astype(np.int64)

    return candidate_indices[:max_prune_by_min_points].astype(np.int64)