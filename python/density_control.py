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
        max_clone_fraction=1.00,
        clone_offset_scale=0.3,
        clone_scale_factor=math.sqrt(2.0),
        min_clone_scale=5.0e-2,
        min_split_coherence=0.05,
        normal_perturbation_min=0.0,
        normal_perturbation_max=0.0,
        tangent_project_position_grad=False,
        normal_shift_on_clone=False,
        normal_shift_scale=0.0,
        max_normal_shift_fraction=0.50,
        exact_clone_scale_threshold=0.0,
        selection_score_np=None,
        curvature_violation_np=None,
        curvature_direction_uu_np=None,
        curvature_direction_uv_np=None,
        curvature_direction_vv_np=None,
        curvature_violation_threshold=0.0,
):
    with torch.no_grad():
        device = positions.device
        point_count = int(positions.shape[0])

        grad_pos = torch.as_tensor(grad_position_np, device=device, dtype=torch.float32)
        if tuple(grad_pos.shape) != tuple(positions.shape):
            raise ValueError(
                "grad_position_np must match positions shape, "
                f"got {tuple(grad_pos.shape)} and {tuple(positions.shape)}"
            )

        tu_all, tv_all, _ = quaternion_to_tangent_frame_torch(rotations.detach())
        tu_all = torch.nn.functional.normalize(tu_all, dim=1, eps=1.0e-12)
        tv_all = tv_all - torch.sum(tv_all * tu_all, dim=1, keepdim=True) * tu_all
        tv_all = torch.nn.functional.normalize(tv_all, dim=1, eps=1.0e-12)

        tangent_grad = (
                torch.sum(grad_pos * tu_all, dim=1, keepdim=True) * tu_all
                + torch.sum(grad_pos * tv_all, dim=1, keepdim=True) * tv_all
        )

        if selection_score_np is None:
            position_signal = torch.linalg.norm(
                tangent_grad if tangent_project_position_grad else grad_pos,
                dim=1,
            )
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

        curvature_threshold = float(curvature_violation_threshold)
        curvature_enabled = (
                math.isfinite(curvature_threshold)
                and curvature_threshold > 0.0
                and curvature_violation_np is not None
        )
        curvature_violation = torch.zeros(point_count, device=device, dtype=torch.float32)
        curvature_uu = torch.zeros_like(curvature_violation)
        curvature_uv = torch.zeros_like(curvature_violation)
        curvature_vv = torch.zeros_like(curvature_violation)

        if curvature_enabled:
            curvature_violation = torch.as_tensor(
                curvature_violation_np,
                device=device,
                dtype=torch.float32,
            ).reshape(-1)
            if curvature_violation.numel() != point_count:
                raise ValueError(
                    "curvature_violation_np must contain one value per surfel, "
                    f"got {curvature_violation.numel()} for {point_count} surfels"
                )

            tensor_inputs = (
                curvature_direction_uu_np,
                curvature_direction_uv_np,
                curvature_direction_vv_np,
            )
            if all(value is not None for value in tensor_inputs):
                curvature_uu = torch.as_tensor(
                    curvature_direction_uu_np, device=device, dtype=torch.float32
                ).reshape(-1)
                curvature_uv = torch.as_tensor(
                    curvature_direction_uv_np, device=device, dtype=torch.float32
                ).reshape(-1)
                curvature_vv = torch.as_tensor(
                    curvature_direction_vv_np, device=device, dtype=torch.float32
                ).reshape(-1)
                if any(value.numel() != point_count for value in
                       (curvature_uu, curvature_uv, curvature_vv)):
                    raise ValueError(
                        "curvature direction tensors must contain one value per surfel"
                    )

            curvature_violation = torch.where(
                torch.isfinite(curvature_violation),
                torch.clamp(curvature_violation, min=0.0),
                torch.zeros_like(curvature_violation),
            )
            curvature_uu = torch.where(
                torch.isfinite(curvature_uu), curvature_uu, torch.zeros_like(curvature_uu)
            )
            curvature_uv = torch.where(
                torch.isfinite(curvature_uv), curvature_uv, torch.zeros_like(curvature_uv)
            )
            curvature_vv = torch.where(
                torch.isfinite(curvature_vv), curvature_vv, torch.zeros_like(curvature_vv)
            )

        curvature_score = (
            curvature_violation / curvature_threshold
            if curvature_enabled
            else torch.zeros_like(position_score)
        )
        selection_score = torch.maximum(position_score, curvature_score)

        safe_clone_scale_factor = max(float(clone_scale_factor), 1.0)
        minimum_splittable_scale = float(min_clone_scale) * safe_clone_scale_factor * (1.0 + 1.0e-4)

        min_source_scale = torch.min(scales, dim=1).values

        selected = (
                torch.isfinite(selection_score)
                & (selection_score >= 1.0)
                & trainable_surfel_mask
                & torch.all(torch.isfinite(positions), dim=1)
                & torch.all(torch.isfinite(rotations), dim=1)
                & torch.all(torch.isfinite(scales), dim=1)
                & (min_source_scale >= minimum_splittable_scale)
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

        curvature_requested_all = curvature_score >= 1.0

        selected_scale_max = torch.max(scales[selected_idx].detach(), dim=1).values
        exact_clone_scale_threshold_value = float(exact_clone_scale_threshold)
        if exact_clone_scale_threshold_value > 0.0:
            # Curvature creates degrees of freedom through splitting only; it
            # never enters the legacy exact-clone branch.
            clone_mask = (
                    (selected_scale_max <= exact_clone_scale_threshold_value)
                    & ~curvature_requested_all[selected_idx]
            )
        else:
            clone_mask = torch.zeros_like(selected_scale_max, dtype=torch.bool)
        split_mask = ~clone_mask

        clone_idx = selected_idx[clone_mask]
        split_idx = selected_idx[split_mask]

        new_positions = []
        new_rotations = []
        new_scales = []
        new_albedos = []
        new_opacities = []
        new_betas = []
        new_powers = []
        source_index_chunks = []
        grad_norm_chunks = []
        update_source = None
        curvature_direction_count = 0

        if clone_idx.numel() > 0:
            clone_grad = grad_pos[clone_idx]

            if tangent_project_position_grad:
                clone_grad = tangent_grad[clone_idx]

            clone_grad_norm = torch.linalg.norm(clone_grad, dim=1, keepdim=True)
            clone_direction = torch.where(
                    clone_grad_norm > 1.0e-12,
                    clone_grad / torch.clamp(clone_grad_norm, min=1.0e-12),
                    torch.zeros_like(clone_grad),
            )

            clone_spatial_scale = torch.max(scales[clone_idx].detach(), dim=1, keepdim=True).values
            clone_offset = clone_offset_scale * clone_spatial_scale

            # grad_pos is assumed to contain dL/dposition, so move toward descent.
            clone_positions = (
                    positions[clone_idx].detach()
                    - clone_offset * clone_direction
            )

            new_positions.append(clone_positions)
            new_rotations.append(rotations[clone_idx].detach().clone())
            new_scales.append(scales[clone_idx].detach().clone())
            new_albedos.append(albedos[clone_idx].detach().clone())
            new_opacities.append(
                    torch.clamp(opacities[clone_idx].detach().clone(), 0.0, 1.0)
            )
            new_betas.append(betas[clone_idx].detach().clone())
            new_powers.append(powers[clone_idx].detach().clone())
            source_index_chunks.append(clone_idx)
            grad_norm_chunks.append(selection_score[clone_idx])

        if split_idx.numel() > 0:
            p = positions[split_idx].detach().clone()
            sc = scales[split_idx].detach().clone()
            tu = tu_all[split_idx].detach().clone()
            tv = tv_all[split_idx].detach().clone()

            tu_n = torch.nn.functional.normalize(tu, dim=1, eps=1.0e-12)
            tv_n = tv - torch.sum(tv * tu_n, dim=1, keepdim=True) * tu_n
            tv_n = torch.nn.functional.normalize(tv_n, dim=1, eps=1.0e-12)

            # Split displacement is always tangent-space. The configuration
            # option controls the position score, while this projected descent
            # direction remains geometrically valid for the surfel ellipse.
            position_descent = -tangent_grad[split_idx]
            position_descent_norm = torch.linalg.norm(position_descent, dim=1, keepdim=True)
            position_direction = torch.where(
                position_descent_norm > 1.0e-12,
                position_descent / torch.clamp(position_descent_norm, min=1.0e-12),
                tu_n,
            )

            tensor_uu = curvature_uu[split_idx]
            tensor_uv = curvature_uv[split_idx]
            tensor_vv = curvature_vv[split_idx]
            tensor_trace = tensor_uu + tensor_vv
            tensor_anisotropy = torch.sqrt(
                torch.clamp(
                    torch.square(tensor_uu - tensor_vv) + 4.0 * torch.square(tensor_uv),
                    min=0.0,
                )
            )
            tensor_finite = (
                    torch.isfinite(tensor_uu)
                    & torch.isfinite(tensor_uv)
                    & torch.isfinite(tensor_vv)
                    & torch.isfinite(tensor_anisotropy)
            )
            tensor_valid = (
                    tensor_finite
                    & (tensor_trace > 1.0e-12)
                    & (tensor_anisotropy > 1.0e-6 * torch.clamp(tensor_trace, min=1.0e-12))
            )

            theta = 0.5 * torch.atan2(2.0 * tensor_uv, tensor_uu - tensor_vv)
            curvature_axis_u = torch.cos(theta)
            curvature_axis_v = torch.sin(theta)
            curvature_direction = (
                    curvature_axis_u[:, None] * tu_n
                    + curvature_axis_v[:, None] * tv_n
            )
            curvature_direction = torch.nn.functional.normalize(
                curvature_direction, dim=1, eps=1.0e-12
            )

            curvature_requested = curvature_requested_all[split_idx]
            use_curvature_direction = curvature_requested & tensor_valid
            curvature_direction_count = int(
                torch.count_nonzero(use_curvature_direction).item()
            )
            split_direction = torch.where(
                use_curvature_direction[:, None],
                curvature_direction,
                position_direction,
            )
            split_direction = torch.nn.functional.normalize(
                split_direction, dim=1, eps=1.0e-12
            )

            axis_u = torch.sum(split_direction * tu_n, dim=1)
            axis_v = torch.sum(split_direction * tv_n, dim=1)
            axis_length = torch.sqrt(torch.clamp(axis_u * axis_u + axis_v * axis_v, min=1.0e-24))
            axis_u = axis_u / axis_length
            axis_v = axis_v / axis_length

            safe_scale_u = torch.clamp(torch.abs(sc[:, 0]), min=1.0e-8)
            safe_scale_v = torch.clamp(torch.abs(sc[:, 1]), min=1.0e-8)
            inverse_radius_squared = (
                    torch.square(axis_u) / torch.square(safe_scale_u)
                    + torch.square(axis_v) / torch.square(safe_scale_v)
            )
            split_radius = torch.rsqrt(torch.clamp(inverse_radius_squared, min=1.0e-24))
            tangent_offset = (
                    float(clone_offset_scale)
                    * split_radius[:, None]
                    * split_direction
            )

            source_positions = p - 0.5 * tangent_offset
            child_positions = p + 0.5 * tangent_offset

            child_sc = sc / safe_clone_scale_factor

            parent_opacity = torch.clamp(opacities[split_idx].detach().clone(), 0.0, 1.0)
            update_source = {
                "index": split_idx.detach().cpu().numpy().astype(np.int64),
                "position": source_positions.detach().cpu().numpy().astype(np.float32),
                "scale": child_sc.detach().cpu().numpy().astype(np.float32),
                "opacity": parent_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32),
            }

            new_positions.append(child_positions)
            new_rotations.append(rotations[split_idx].detach().clone())
            new_scales.append(child_sc)
            new_albedos.append(albedos[split_idx].detach().clone())
            new_opacities.append(parent_opacity.clone())
            new_betas.append(betas[split_idx].detach().clone())
            new_powers.append(powers[split_idx].detach().clone())
            source_index_chunks.append(split_idx)
            grad_norm_chunks.append(selection_score[split_idx])

        if not new_positions:
            return None

        new_position_t = torch.cat(new_positions, dim=0)
        new_rotation_t = torch.cat(new_rotations, dim=0)
        new_scale_t = torch.cat(new_scales, dim=0)
        new_albedo_t = torch.cat(new_albedos, dim=0)
        new_opacity_t = torch.cat(new_opacities, dim=0)
        new_beta_t = torch.cat(new_betas, dim=0)
        new_power_t = torch.cat(new_powers, dim=0)
        source_index_t = torch.cat(source_index_chunks, dim=0)
        grad_norm_t = torch.cat(grad_norm_chunks, dim=0)

        result = {
            "new": {
                "position": new_position_t.detach().cpu().numpy().astype(np.float32),
                "rotation": new_rotation_t.detach().cpu().numpy().astype(np.float32),
                "scale": new_scale_t.detach().cpu().numpy().astype(np.float32),
                "albedo": new_albedo_t.detach().cpu().numpy().astype(np.float32),
                "opacity": new_opacity_t.detach().cpu().numpy().reshape(-1).astype(np.float32),
                "beta": new_beta_t.detach().cpu().numpy().reshape(-1).astype(np.float32),
                "power": new_power_t.detach().cpu().numpy().reshape(-1).astype(np.float32),
            },
            "source_index": source_index_t.detach().cpu().numpy().astype(np.int64),
            "grad_norm": grad_norm_t.detach().cpu().numpy().astype(np.float32),
            "selection_score": grad_norm_t.detach().cpu().numpy().astype(np.float32),
            "clone_count": int(clone_idx.numel()),
            "split_count": int(split_idx.numel()),
            "position_trigger_count": int(torch.count_nonzero(position_score[selected_idx] >= 1.0).item()),
            "curvature_trigger_count": int(torch.count_nonzero(curvature_requested_all[selected_idx]).item()),
            # Exclusive split attribution. Curvature has direction priority when
            # both thresholds fire, so the two counts sum exactly to split_count.
            "position_split_count": int(
                torch.count_nonzero(~curvature_requested_all[split_idx]).item()
            ),
            "curvature_split_count": int(
                torch.count_nonzero(curvature_requested_all[split_idx]).item()
            ),
            "split_trigger_is_curvature": (
                curvature_requested_all[split_idx]
                .detach().cpu().numpy().astype(bool)
            ),
            "curvature_direction_count": curvature_direction_count,
            "exact_clone_scale_threshold": exact_clone_scale_threshold_value,
            "split_offset_scale": float(clone_offset_scale),
            "split_scale_factor": safe_clone_scale_factor,
            "replace_source": False,
        }

        if update_source is not None:
            result["update_source"] = update_source

        return result


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
