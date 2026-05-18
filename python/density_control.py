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
        max_clone_fraction=0.05,
        clone_offset_scale=0.25,
        clone_scale_factor=1.6,
        normal_perturbation_min=1.0e-5,
        normal_perturbation_max=3.0e-5,
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

        selected = (
                torch.isfinite(selection_score)
                & (selection_score >= grad_threshold)
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

        # Random signed normal perturbation to avoid systematic drift along +normal.
        # This is only for BVH/coincident-primitive robustness, not density placement.
        eps_min = float(normal_perturbation_min)
        eps_max = float(normal_perturbation_max)

        if eps_min < 0.0 or eps_max < 0.0:
            raise ValueError("normal perturbation bounds must be non-negative")

        if eps_max < eps_min:
            raise ValueError(
                f"normal_perturbation_max must be >= normal_perturbation_min, "
                f"got {eps_max} < {eps_min}"
            )

        random_unit = torch.rand(
            (selected_idx.numel(), 1),
            device=device,
            dtype=torch.float32,
        )

        random_magnitude = eps_min + (eps_max - eps_min) * random_unit

        random_sign = torch.where(
            torch.rand(
                (selected_idx.numel(), 1),
                device=device,
                dtype=torch.float32,
            ) < 0.5,
            torch.full((selected_idx.numel(), 1), -1.0, device=device, dtype=torch.float32),
            torch.full((selected_idx.numel(), 1), 1.0, device=device, dtype=torch.float32),
        )

        normal_offset = random_sign * random_magnitude * normal_dir
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

import math
import numpy as np
import torch
from typing import Any


def make_under_reconstruction_evsplits(
        positions: torch.Tensor,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
        grad_position_np: np.ndarray,
        trainable_surfel_mask: torch.Tensor,
        grad_threshold: float,
        max_split_fraction: float = 0.05,
        selection_score_np: np.ndarray | None = None,
        min_scale: float = 1.0e-6,
        min_opacity: float = 1.0e-5,
        max_opacity: float = 0.99,
        preserve_integrated_opacity: bool = True,
) -> dict[str, dict[str, Any] | Any] | None:
    """
    EV-Splitting-style densification for Gaussian surfels.

    This replaces each selected parent surfel by two children. It is not clone-additive.

    Local measure:
        2D world tangent-plane area measure.

    Local model:
        g(y) = alpha * exp(-0.5 y^T Sigma^{-1} y),
        Sigma = diag(scale_u^2, scale_v^2) in the local tangent frame.

    Split:
        centered tangent-plane split with normal chosen from the projected descent
        direction. If the descent direction is degenerate, split along the largest
        local scale axis.

    Returned structure:
        Compatible with your existing add_new_points(...), plus:
            replace_source = True
            source_index = parent indices to remove
    """

    with torch.no_grad():
        device = positions.device

        grad_pos = torch.as_tensor(
            np.asarray(grad_position_np, dtype=np.float32, order="C"),
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

        selected = (
            torch.isfinite(selection_score)
            & (selection_score >= float(grad_threshold))
            & trainable_surfel_mask
        )

        selected_idx = torch.nonzero(selected, as_tuple=False).flatten()
        if selected_idx.numel() == 0:
            return None

        n_points = int(positions.shape[0])

        # EV split creates two children and removes one parent.
        # Net growth is +1 per selected surfel.
        max_splits = max(1, int(float(max_split_fraction) * float(n_points)))

        if selected_idx.numel() > max_splits:
            selected_score = selection_score[selected_idx]
            keep = torch.topk(selected_score, k=max_splits, largest=True).indices
            selected_idx = selected_idx[keep]

        p = positions[selected_idx].detach().clone()
        tu = tangent_u[selected_idx].detach().clone()
        tv = tangent_v[selected_idx].detach().clone()
        sc = scales[selected_idx].detach().clone()
        alb = albedos[selected_idx].detach().clone()
        opa = opacities[selected_idx].detach().clone().reshape(-1)
        be = betas[selected_idx].detach().clone().reshape(-1)
        pow_ = powers[selected_idx].detach().clone().reshape(-1)

        g = grad_pos[selected_idx]

        eps = 1.0e-12

        tu_n = torch.nn.functional.normalize(tu, dim=1, eps=eps)
        tv_n = torch.nn.functional.normalize(tv, dim=1, eps=eps)

        old_normal = torch.cross(tu_n, tv_n, dim=1)
        old_normal = torch.nn.functional.normalize(old_normal, dim=1, eps=eps)

        # For minimization, grad = dL/dp, so descent is -grad.
        descent = -g

        # Local tangent components of descent.
        a_u = torch.sum(descent * tu_n, dim=1)
        a_v = torch.sum(descent * tv_n, dim=1)
        a = torch.stack([a_u, a_v], dim=1)

        a_norm = torch.linalg.norm(a, dim=1, keepdim=True)
        valid_direction = a_norm[:, 0] > 1.0e-10

        # Fallback: split along largest local scale.
        major_is_u = sc[:, 0] >= sc[:, 1]
        fallback_a = torch.stack(
            [
                major_is_u.to(torch.float32),
                (~major_is_u).to(torch.float32),
            ],
            dim=1,
        )

        a_unit = torch.where(
            valid_direction[:, None],
            a / torch.clamp(a_norm, min=eps),
            fallback_a,
        )

        su2 = torch.clamp(sc[:, 0], min=min_scale) ** 2
        sv2 = torch.clamp(sc[:, 1], min=min_scale) ** 2

        sigma = torch.zeros((selected_idx.numel(), 2, 2), device=device, dtype=torch.float32)
        sigma[:, 0, 0] = su2
        sigma[:, 1, 1] = sv2

        sigma_a = torch.bmm(sigma, a_unit[:, :, None]).squeeze(-1)
        tau2 = torch.sum(a_unit * sigma_a, dim=1)
        tau2 = torch.clamp(tau2, min=min_scale ** 2)
        tau = torch.sqrt(tau2)

        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)

        # Centered EV split displacement in local tangent coordinates.
        delta_local = sqrt_2_over_pi * sigma_a / tau[:, None]

        # Centered EV child covariance.
        outer = sigma_a[:, :, None] * sigma_a[:, None, :]
        child_sigma = sigma - (2.0 / math.pi) * outer / tau2[:, None, None]
        child_sigma = 0.5 * (child_sigma + child_sigma.transpose(1, 2))

        eigval, eigvec = torch.linalg.eigh(child_sigma)

        # Sort descending so scale[:, 0] is the major axis.
        order = torch.argsort(eigval, dim=1, descending=True)
        eigval = torch.gather(eigval, 1, order)
        eigvec = torch.gather(
            eigvec,
            2,
            order[:, None, :].expand(-1, 2, -1),
        )

        eigval = torch.clamp(eigval, min=min_scale ** 2)
        child_sc = torch.sqrt(eigval)

        # Eigenvectors are local 2D directions. Convert them to world tangent directions.
        e0 = eigvec[:, 0, 0:1] * tu_n + eigvec[:, 1, 0:1] * tv_n
        e1 = eigvec[:, 0, 1:2] * tu_n + eigvec[:, 1, 1:2] * tv_n

        e0 = torch.nn.functional.normalize(e0, dim=1, eps=eps)
        e1 = torch.nn.functional.normalize(e1, dim=1, eps=eps)

        # Preserve original normal orientation.
        child_normal = torch.cross(e0, e1, dim=1)
        wrong_handed = torch.sum(child_normal * old_normal, dim=1) < 0.0
        e1 = torch.where(wrong_handed[:, None], -e1, e1)

        offset_world = delta_local[:, 0:1] * tu_n + delta_local[:, 1:2] * tv_n

        pos_l = p - offset_world
        pos_r = p + offset_world

        if preserve_integrated_opacity:
            det_parent = torch.clamp(su2 * sv2, min=min_scale ** 4)
            det_child = torch.clamp(eigval[:, 0] * eigval[:, 1], min=min_scale ** 4)

            # Each child receives half the normalized mass, converted back to peak opacity.
            opacity_factor = 0.5 * torch.sqrt(det_parent / det_child)
        else:
            # Use only if your renderer already interprets opacity as normalized mass.
            opacity_factor = torch.full_like(opa, 0.5)

        child_opa = torch.clamp(
            opa * opacity_factor,
            min=float(min_opacity),
            max=float(max_opacity),
        )

        new_positions = torch.cat([pos_l, pos_r], dim=0)
        new_tu = torch.cat([e0, e0], dim=0)
        new_tv = torch.cat([e1, e1], dim=0)
        new_scales = torch.cat([child_sc, child_sc], dim=0)

        new_albedos = torch.cat([alb, alb], dim=0)
        new_opacities = torch.cat([child_opa, child_opa], dim=0)
        new_betas = torch.cat([be, be], dim=0)
        new_powers = torch.cat([pow_, pow_], dim=0)

        return {
            "new": {
                "position": new_positions.detach().cpu().numpy().astype(np.float32),
                "tangent_u": new_tu.detach().cpu().numpy().astype(np.float32),
                "tangent_v": new_tv.detach().cpu().numpy().astype(np.float32),
                "scale": new_scales.detach().cpu().numpy().astype(np.float32),
                "albedo": new_albedos.detach().cpu().numpy().astype(np.float32),
                "opacity": new_opacities.detach().cpu().numpy().reshape(-1).astype(np.float32),
                "beta": new_betas.detach().cpu().numpy().reshape(-1).astype(np.float32),
                "power": new_powers.detach().cpu().numpy().reshape(-1).astype(np.float32),
            },
            "source_index": selected_idx.detach().cpu().numpy().astype(np.int64),
            "replace_source": True,
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