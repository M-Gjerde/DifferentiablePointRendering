import numpy as np
import torch
from typing import Optional, Any
import math

def normalize_quaternions_torch(q: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    finite = torch.isfinite(q).all(dim=1, keepdim=True)
    n = torch.linalg.norm(q, dim=1, keepdim=True)
    fallback = torch.zeros_like(q)
    fallback[:, 0] = 1.0
    qn = q / n.clamp_min(eps)
    qn = torch.where(finite & (n > eps), qn, fallback)
    qn = torch.where(qn[:, 0:1] < 0.0, -qn, qn)
    return qn

def quaternion_to_tangent_frame_torch(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = normalize_quaternions_torch(q)
    qw, qx, qy, qz = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    tu = torch.cat([
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy + qz * qw),
        2.0 * (qx * qz - qy * qw),
    ], dim=1)
    tv = torch.cat([
        2.0 * (qx * qy - qz * qw),
        1.0 - 2.0 * (qx * qx + qz * qz),
        2.0 * (qy * qz + qx * qw),
    ], dim=1)
    tw = torch.cat([
        2.0 * (qx * qz + qy * qw),
        2.0 * (qy * qz - qx * qw),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ], dim=1)
    tu = torch.nn.functional.normalize(tu, dim=1, eps=1.0e-8)
    tv = tv - torch.sum(tv * tu, dim=1, keepdim=True) * tu
    tv = torch.nn.functional.normalize(tv, dim=1, eps=1.0e-8)
    tw = torch.nn.functional.normalize(torch.cross(tu, tv, dim=1), dim=1, eps=1.0e-8)
    return tu, tv, tw

def quaternion_multiply_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bw, bx, by, bz = b[:, 0:1], b[:, 1:2], b[:, 2:3], b[:, 3:4]
    return torch.cat([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=1)

def quaternion_exp_from_local_delta(delta: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    angle = torch.linalg.norm(delta, dim=1, keepdim=True)
    half_angle = 0.5 * angle
    sin_half_over_angle = torch.where(
        angle > eps,
        torch.sin(half_angle) / angle.clamp_min(eps),
        0.5 - (angle * angle) / 48.0,
    )
    dq = torch.cat([torch.cos(half_angle), delta * sin_half_over_angle], dim=1)
    return normalize_quaternions_torch(dq)

def apply_local_rotation_update_to_quaternions_inplace(
        rotations: torch.Tensor,
        rotation_delta: torch.Tensor,
        trainable_surfel_mask: Optional[torch.Tensor] = None,
        max_rotation_step_radians: float = 0.0,
) -> dict[str, float]:
    if rotations.ndim != 2 or rotations.shape[1] != 4:
        raise ValueError(f"rotations must be (N,4), got {tuple(rotations.shape)}")
    if rotation_delta.ndim != 2 or rotation_delta.shape[1] != 3 or rotation_delta.shape[0] != rotations.shape[0]:
        raise ValueError(f"rotation_delta must be (N,3), got {tuple(rotation_delta.shape)}")
    with torch.no_grad():
        delta = rotation_delta.detach().to(device=rotations.device, dtype=rotations.dtype)
        if trainable_surfel_mask is not None:
            mask = trainable_surfel_mask.to(device=rotations.device, dtype=torch.bool).view(-1, 1)
            delta = torch.where(mask, delta, torch.zeros_like(delta))
        if max_rotation_step_radians is not None and max_rotation_step_radians > 0.0:
            delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
            clamp_scale = torch.clamp(float(max_rotation_step_radians) / delta_norm.clamp_min(1.0e-12), max=1.0)
            delta = delta * clamp_scale
        before = normalize_quaternions_torch(rotations.detach())
        dq = quaternion_exp_from_local_delta(delta)
        after = normalize_quaternions_torch(quaternion_multiply_wxyz(before, dq))
        rotations.copy_(after)
        rotation_delta.zero_()
        return {
            "max_quat_norm_error": float((torch.linalg.norm(rotations, dim=1) - 1.0).abs().max().item()),
            "max_delta_norm": float(torch.linalg.norm(delta, dim=1).max().item()) if delta.numel() else 0.0,
        }

def quaternion_from_tangent_frame_torch(tangent_u: torch.Tensor, tangent_v: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-12
    u = torch.nn.functional.normalize(tangent_u, dim=1, eps=eps)
    v = tangent_v - torch.sum(tangent_v * u, dim=1, keepdim=True) * u
    v = torch.nn.functional.normalize(v, dim=1, eps=eps)
    w = torch.nn.functional.normalize(torch.cross(u, v, dim=1), dim=1, eps=eps)
    m00, m01, m02 = u[:, 0], v[:, 0], w[:, 0]
    m10, m11, m12 = u[:, 1], v[:, 1], w[:, 1]
    m20, m21, m22 = u[:, 2], v[:, 2], w[:, 2]
    q = torch.zeros((u.shape[0], 4), device=u.device, dtype=u.dtype)
    trace = m00 + m11 + m22
    mask = trace > 0.0
    if mask.any():
        s = torch.sqrt(trace[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (m21[mask] - m12[mask]) / s
        q[mask, 2] = (m02[mask] - m20[mask]) / s
        q[mask, 3] = (m10[mask] - m01[mask]) / s
    mask_x = (~mask) & (m00 > m11) & (m00 > m22)
    if mask_x.any():
        s = torch.sqrt(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x]) * 2.0
        q[mask_x, 0] = (m21[mask_x] - m12[mask_x]) / s
        q[mask_x, 1] = 0.25 * s
        q[mask_x, 2] = (m01[mask_x] + m10[mask_x]) / s
        q[mask_x, 3] = (m02[mask_x] + m20[mask_x]) / s
    mask_y = (~mask) & (~mask_x) & (m11 > m22)
    if mask_y.any():
        s = torch.sqrt(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y]) * 2.0
        q[mask_y, 0] = (m02[mask_y] - m20[mask_y]) / s
        q[mask_y, 1] = (m01[mask_y] + m10[mask_y]) / s
        q[mask_y, 2] = 0.25 * s
        q[mask_y, 3] = (m12[mask_y] + m21[mask_y]) / s
    mask_z = (~mask) & (~mask_x) & (~mask_y)
    if mask_z.any():
        s = torch.sqrt(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z]) * 2.0
        q[mask_z, 0] = (m10[mask_z] - m01[mask_z]) / s
        q[mask_z, 1] = (m02[mask_z] + m20[mask_z]) / s
        q[mask_z, 2] = (m12[mask_z] + m21[mask_z]) / s
        q[mask_z, 3] = 0.25 * s
    return normalize_quaternions_torch(q)

def project_gradient_to_surfel_tangent_plane_np(
        grad_position_np: np.ndarray,
        rotations: torch.Tensor,
) -> np.ndarray:
    grad_np = np.asarray(grad_position_np, dtype=np.float32, order="C")
    with torch.no_grad():
        device = rotations.device
        g = torch.as_tensor(grad_np, device=device, dtype=torch.float32)
        tu, tv, _ = quaternion_to_tangent_frame_torch(rotations.detach())
        projected = torch.sum(g * tu, dim=1, keepdim=True) * tu + torch.sum(g * tv, dim=1, keepdim=True) * tv
        return projected.detach().cpu().numpy().astype(np.float32)

def make_under_reconstruction_clones(
        positions,
        rotations,
        scales,
        albedos,
        opacities,
        betas,
        powers,
        grad_position_np,
        trainable_surfel_mask,
        grad_threshold,
        max_clone_fraction=1.0,
        clone_offset_scale=0.60,
        clone_scale_factor=1.8,
        min_clone_scale=5.0e-2,
        normal_perturbation_min=1.0e-5,
        normal_perturbation_max=3.0e-5,
        tangent_project_position_grad=True,
        normal_shift_on_clone=True,
        normal_shift_scale=0.25,
        max_normal_shift_fraction=0.50,
        selection_score_np=None,
) -> dict[str, dict[str, Any] | Any] | None:
    """
    Conservative clone/split densification for under-reconstruction.

    This turns one surfel into two children:

        source child: existing surfel updated in-place
        clone child : newly appended surfel

    Both children:
        - inherit material parameters
        - receive the same tangent frame
        - keep the current opacity behavior
        - receive scale / clone_scale_factor

    Position update:
        - tangent component controls clone separation
        - normal component optionally shifts both children together

    This avoids using the normal gradient as clone separation, while still allowing
    the reconstructed surface to move along the normal direction.
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

        min_source_scale = torch.min(scales, dim=1).values

        selected = (
                torch.isfinite(selection_score)
                & (selection_score >= grad_threshold)
                & trainable_surfel_mask
                & (min_source_scale >= float(min_clone_scale))
        )

        selected_idx = torch.nonzero(selected, as_tuple=False).flatten()
        if selected_idx.numel() == 0:
            return None

        n_points = positions.shape[0]
        max_new = max(1, int(max_clone_fraction * float(n_points)))

        if selected_idx.numel() > max_new:
            selected_grad = selection_score[selected_idx]
            keep = torch.topk(selected_grad, k=max_new, largest=True).indices
            selected_idx = selected_idx[keep]

        tu_all, tv_all, _ = quaternion_to_tangent_frame_torch(rotations.detach())

        p = positions[selected_idx].detach().clone()
        rot = rotations[selected_idx].detach().clone()
        tu = tu_all[selected_idx].detach().clone()
        tv = tv_all[selected_idx].detach().clone()
        sc = scales[selected_idx].detach().clone()
        alb = albedos[selected_idx].detach().clone()
        opa = opacities[selected_idx].detach().clone()
        be = betas[selected_idx].detach().clone()
        pow_ = powers[selected_idx].detach().clone()

        g = grad_pos[selected_idx]

        # For minimization, the optimizer moves along -grad.
        descent = -g

        tu_n = torch.nn.functional.normalize(tu, dim=1, eps=1.0e-12)
        tv_n = torch.nn.functional.normalize(tv, dim=1, eps=1.0e-12)

        normal = torch.cross(tu_n, tv_n, dim=1)
        normal_norm = torch.linalg.norm(normal, dim=1, keepdim=True)
        valid_normal = normal_norm > 1.0e-12

        normal_dir = torch.zeros_like(normal)
        normal_dir[valid_normal[:, 0]] = (
                normal[valid_normal[:, 0]] / normal_norm[valid_normal[:, 0]]
        )

        tangent_descent = (
                torch.sum(descent * tu_n, dim=1, keepdim=True) * tu_n
                + torch.sum(descent * tv_n, dim=1, keepdim=True) * tv_n
        )

        normal_descent = (
                torch.sum(descent * normal_dir, dim=1, keepdim=True) * normal_dir
        )

        if tangent_project_position_grad:
            split_descent = tangent_descent
        else:
            split_descent = descent

        split_descent_norm = torch.linalg.norm(split_descent, dim=1, keepdim=True)
        valid_split_dir = split_descent_norm > 1.0e-12

        split_direction = torch.zeros_like(split_descent)
        split_direction[valid_split_dir[:, 0]] = (
                split_descent[valid_split_dir[:, 0]]
                / split_descent_norm[valid_split_dir[:, 0]]
        )

        local_radius = torch.min(sc, dim=1).values[:, None]
        tangent_offset = clone_offset_scale * local_radius * split_direction

        normal_shift = torch.zeros_like(p)

        if normal_shift_on_clone:
            normal_shift_scale = float(normal_shift_scale)
            max_normal_shift_fraction = float(max_normal_shift_fraction)

            if normal_shift_scale < 0.0:
                raise ValueError("normal_shift_scale must be non-negative")

            if max_normal_shift_fraction < 0.0:
                raise ValueError("max_normal_shift_fraction must be non-negative")

            descent_norm = torch.linalg.norm(descent, dim=1, keepdim=True)
            normal_descent_norm = torch.linalg.norm(normal_descent, dim=1, keepdim=True)

            normal_fraction = normal_descent_norm / torch.clamp(descent_norm, min=1.0e-12)
            normal_fraction = torch.clamp(normal_fraction, min=0.0, max=1.0)

            normal_shift_magnitude = normal_shift_scale * local_radius * normal_fraction
            max_normal_shift = max_normal_shift_fraction * local_radius
            normal_shift_magnitude = torch.minimum(normal_shift_magnitude, max_normal_shift)

            normal_shift_direction = torch.zeros_like(normal_descent)
            valid_normal_shift_dir = normal_descent_norm > 1.0e-12
            normal_shift_direction[valid_normal_shift_dir[:, 0]] = (
                    normal_descent[valid_normal_shift_dir[:, 0]]
                    / normal_descent_norm[valid_normal_shift_dir[:, 0]]
            )

            normal_shift = normal_shift_magnitude * normal_shift_direction

        # Asymmetric split.
        # Parent keeps most opacity, so it moves less.
        child_weight = 0.75
        source_positions = p + normal_shift - (1.0 - child_weight) * tangent_offset
        clone_positions = p + normal_shift + child_weight * tangent_offset

        # Small normal perturbation only for the appended clone.
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

        clone_positions = clone_positions + random_sign * random_magnitude * normal_dir

        # Simple clone/split scale rule.
        safe_clone_scale_factor = max(float(clone_scale_factor), 1.0)
        child_sc = sc / safe_clone_scale_factor

        # Avoid producing surfels below the clone minimum.
        child_sc = torch.clamp(child_sc, min=float(min_clone_scale))

        # Keep current opacity behavior unchanged.
        parent_opacity = torch.clamp(opa, min=0.2)
        child_opacity = torch.clamp(opa, min=0.2)

        selected_idx_np = selected_idx.detach().cpu().numpy().astype(np.int64)

        return {
            "update_source": {
                "index": selected_idx_np,
                "position": source_positions.detach().cpu().numpy().astype(np.float32),
                "scale": child_sc.detach().cpu().numpy().astype(np.float32),
                "opacity": parent_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32),
            },

            "new": {
                "position": clone_positions.detach().cpu().numpy().astype(np.float32),
                "rotation": rot.detach().cpu().numpy().astype(np.float32),
                "scale": child_sc.detach().cpu().numpy().astype(np.float32),
                "albedo": alb.detach().cpu().numpy().astype(np.float32),
                "opacity": child_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32),
                "beta": be.detach().cpu().numpy().reshape(-1).astype(np.float32),
                "power": pow_.detach().cpu().numpy().reshape(-1).astype(np.float32),
            },

            "source_index": selected_idx_np,

            "grad_norm": selection_score[selected_idx].detach().cpu().numpy().astype(np.float32),

            "replace_source": False,
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


def compute_prune_indices_by_degenerate_area(
        scales: torch.Tensor,
        *,
        min_area: float = math.pi * 1.0e-10,
        trainable_mask: Optional[torch.Tensor] = None,
        min_points_to_keep: int = 1,
) -> np.ndarray:
    """
    Prune surfels with degenerate in-plane ellipse area.

    The geometric area is:

        area = pi * scale_u * scale_v

    This preserves elongated surfels as long as their total support area remains
    meaningful. Non-finite or non-positive scale values are always degenerate.
    """
    if min_area < 0.0:
        raise ValueError(f"min_area must be non-negative, got {min_area}")

    with torch.no_grad():
        scales_np = scales.detach().cpu().numpy().astype(np.float32, copy=False)

        if trainable_mask is not None:
            trainable_mask_np = trainable_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        else:
            trainable_mask_np = np.ones((scales_np.shape[0],), dtype=bool)

    if scales_np.ndim != 2 or scales_np.shape[1] != 2:
        raise ValueError(f"Expected scales to have shape (N, 2), got {scales_np.shape}")

    num_points = scales_np.shape[0]
    if num_points == 0:
        return np.zeros((0,), dtype=np.int64)

    scale_u = scales_np[:, 0]
    scale_v = scales_np[:, 1]

    finite_mask = np.isfinite(scales_np).all(axis=1)
    positive_scale_mask = (scale_u > 0.0) & (scale_v > 0.0)

    surfel_area = np.full((num_points,), np.nan, dtype=np.float32)
    valid_area_mask = finite_mask & positive_scale_mask
    surfel_area[valid_area_mask] = (
        np.float32(math.pi)
        * scale_u[valid_area_mask]
        * scale_v[valid_area_mask]
    )

    degenerate_mask = (
        ~finite_mask
        | ~positive_scale_mask
        | (surfel_area <= float(min_area))
    )

    candidate_indices = np.nonzero(
        degenerate_mask & trainable_mask_np
    )[0].astype(np.int64)

    if candidate_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)

    max_prune_by_min_points = max(0, num_points - int(min_points_to_keep))
    if max_prune_by_min_points <= 0:
        return np.zeros((0,), dtype=np.int64)

    if candidate_indices.size <= max_prune_by_min_points:
        return candidate_indices

    # Prefer pruning the smallest-area surfels first if the safety cap applies.
    candidate_area = np.nan_to_num(
        surfel_area[candidate_indices],
        nan=-np.inf,
        posinf=np.inf,
        neginf=-np.inf,
    )
    order = np.argsort(candidate_area)
    return candidate_indices[order[:max_prune_by_min_points]]
