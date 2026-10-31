import math

import numpy as np
import torch


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
        trainable_surfel_mask: torch.Tensor | None = None,
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
        before = normalize_quaternions_torch(rotations.detach())
        dq = quaternion_exp_from_local_delta(delta)
        after = normalize_quaternions_torch(quaternion_multiply_wxyz(before, dq))
        rotations.copy_(after)
        rotation_delta.zero_()
        return {
            "max_quat_norm_error": float((torch.linalg.norm(rotations, dim=1) - 1.0).abs().max().item()),
            "max_delta_norm": float(torch.linalg.norm(delta, dim=1).max().item()) if delta.numel() else 0.0,
        }


def _sample_beta_footprint_positions(positions, rotations, scales, betas, count):
    """Draw independent positions from each parent's normalized tangent-plane kernel."""
    tu, tv, _ = quaternion_to_tangent_frame_torch(rotations)
    # K(u,v) = (1-u^2-v^2)^b, b = 4*exp(beta), on the unit disk.
    # Including the polar area Jacobian gives r^2 ~ Beta(1, b+1).
    beta_exponent = 4.0 * torch.exp(betas.reshape(-1))
    uniforms = torch.rand((positions.shape[0], count, 2), device=positions.device, dtype=positions.dtype)
    # Stable inverse CDF: 1 - (1-U)^(1/(b+1)), including sharp kernels.
    radius = torch.sqrt(-torch.expm1(
        torch.log1p(-uniforms[:, :, 0]) / (beta_exponent[:, None] + 1.0)
    ))
    angle = (2.0 * math.pi) * uniforms[:, :, 1]
    offset_u = scales[:, None, 0] * radius * torch.cos(angle)
    offset_v = scales[:, None, 1] * radius * torch.sin(angle)
    offsets = offset_u[:, :, None] * tu[:, None, :] + offset_v[:, :, None] * tv[:, None, :]
    return positions[:, None, :] + offsets


def make_under_reconstruction_clones(
        positions,
        rotations,
        scales,
        albedos,
        opacities,
        betas,
        powers,
        trainable_surfel_mask,
        grad_threshold,
        max_clone_fraction=1.00,
        clone_scale_factor=math.sqrt(2.0),
        min_clone_scale=5.0e-2,
        tangent_project_position_grad=False,
        selection_score_np=None,
        clone_max_scale=None,
        grad_position_np=None,
):
    """Optionally clone small surfels within their footprint and split larger ones.

    The selection score is a window reduction made before each optimizer step.
    Children independently sample the current parent's beta footprint in
    its tangent plane; the gradient only selects parents, not child placement.
    With clone_max_scale=None, retain the existing split-only policy. Otherwise,
    max(su,sv) <= clone_max_scale selects clones, retaining parent and child scales
    and leaving the parent unchanged.
    """
    if clone_max_scale is not None and (
        not math.isfinite(float(clone_max_scale)) or float(clone_max_scale) <= 0
    ):
        raise ValueError("clone_max_scale must be finite and positive, or None")
    with torch.no_grad():
        device = positions.device
        point_count = int(positions.shape[0])

        if selection_score_np is None:
            grad_pos = torch.as_tensor(grad_position_np, device=device, dtype=positions.dtype)
            if grad_pos.shape != positions.shape:
                raise ValueError("grad_position_np must match positions shape")
            if tangent_project_position_grad:
                tu, tv, _ = quaternion_to_tangent_frame_torch(rotations.detach())
                position_signal = torch.sqrt(
                    torch.sum(grad_pos * tu, dim=1).square()
                    + torch.sum(grad_pos * tv, dim=1).square())
            else:
                position_signal = torch.linalg.norm(grad_pos, dim=1)
        else:
            position_signal = torch.as_tensor(
                selection_score_np,
                device=device,
                dtype=torch.float32,
            ).reshape(-1)
            if position_signal.numel() != point_count:
                raise ValueError(
                    "selection_score_np must contain one value per surfel, "
                    f"got {position_signal.numel()} for {point_count} surfels"
                )

        position_signal = torch.where(
            torch.isfinite(position_signal),
            torch.clamp(position_signal, min=0.0),
            torch.zeros_like(position_signal),
        )
        position_threshold = float(grad_threshold)
        if math.isfinite(position_threshold) and position_threshold > 0.0:
            position_score = position_signal / position_threshold
        else:
            position_score = torch.zeros_like(position_signal)


        selection_score = position_score

        safe_clone_scale_factor = max(float(clone_scale_factor), 1.0)
        minimum_splittable_scale = float(min_clone_scale) * safe_clone_scale_factor * (1.0 + 1.0e-4)

        min_source_scale = torch.min(scales, dim=1).values
        clone_mask = (
            torch.max(scales, dim=1).values <= float(clone_max_scale)
            if clone_max_scale is not None else torch.zeros(point_count, device=device, dtype=torch.bool)
        )
        size_eligible = (clone_mask & (min_source_scale > 0)) | (
            ~clone_mask & (min_source_scale >= minimum_splittable_scale)
        )

        selected = (
                torch.isfinite(selection_score)
                & (selection_score >= 1.0)
                & trainable_surfel_mask
                & torch.all(torch.isfinite(positions), dim=1)
                & torch.all(torch.isfinite(rotations), dim=1)
                & torch.all(torch.isfinite(scales), dim=1)
                & torch.isfinite(betas.detach().reshape(-1))
                & size_eligible
        )

        selected_idx = torch.nonzero(selected, as_tuple=False).flatten()
        if selected_idx.numel() == 0:
            return None

        n_points = positions.shape[0]
        safe_max_clone_fraction = max(float(max_clone_fraction), 0.0)
        if safe_max_clone_fraction <= 0.0:
            return None
        max_new = max(1, int(safe_max_clone_fraction * float(n_points)))

        if selected_idx.numel() > max_new:
            selected_score = selection_score[selected_idx]
            keep = torch.topk(selected_score, k=max_new, largest=True).indices
            selected_idx = selected_idx[keep]


        clone_idx = selected_idx[clone_mask[selected_idx]]
        split_idx = selected_idx[~clone_mask[selected_idx]]

        # Clones precede splits so provenance labels retain their existing order.
        source_index_t = torch.cat((clone_idx, split_idx))
        parameters = dict(position=positions, rotation=rotations, scale=scales,
                          albedo=albedos, opacity=opacities, beta=betas, power=powers)
        children = {key: value[source_index_t].detach().clone()
                    for key, value in parameters.items()}
        update_source = None

        if split_idx.numel() > 0:
            sampled = _sample_beta_footprint_positions(
                positions[split_idx], rotations[split_idx], scales[split_idx], betas[split_idx], 2,
            )
            child_sc = scales[split_idx] / safe_clone_scale_factor

            parent_opacity = torch.clamp(opacities[split_idx].detach().clone(), 0.0, 1.0)
            update_source = {
                "index": split_idx.detach().cpu().numpy().astype(np.int64),
                "position": sampled[:, 0].cpu().numpy().astype(np.float32),
                "scale": child_sc.detach().cpu().numpy().astype(np.float32),
                "opacity": parent_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32),
            }

            split_start = clone_idx.numel()
            children["position"][split_start:] = sampled[:, 1]
            children["scale"][split_start:] = child_sc
            children["opacity"][split_start:] = parent_opacity

        if clone_idx.numel() > 0:
            children["position"][:clone_idx.numel()] = _sample_beta_footprint_positions(
                positions[clone_idx], rotations[clone_idx], scales[clone_idx], betas[clone_idx], 1,
            )[:, 0]

        scores = selection_score[source_index_t].cpu().numpy().astype(np.float32, copy=False)
        result = {
            "new": {key: value.cpu().numpy().astype(np.float32, copy=False)
                    .reshape((-1,) if key in ("opacity", "beta", "power") else value.shape)
                    for key, value in children.items()},
            "source_index": source_index_t.cpu().numpy().astype(np.int64, copy=False),
            "grad_norm": scores,
            "selection_score": scores,
            "clone_count": int(clone_idx.numel()),
            "split_count": int(split_idx.numel()),
            "position_trigger_count": int(torch.count_nonzero(position_score[selected_idx] >= 1.0).item()),
            "position_split_count": int(split_idx.numel()),
            "split_sampling": "beta_footprint",
            "split_scale_factor": safe_clone_scale_factor,
            "clone_max_scale": clone_max_scale,
            "replace_source": False,
        }

        if update_source is not None:
            result["update_source"] = update_source

        return result


def compute_prune_indices_by_degenerate_area(
        scales: torch.Tensor,
        *,
        min_area: float = math.pi * 1.0e-10,
        trainable_mask: torch.Tensor | None = None,
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
