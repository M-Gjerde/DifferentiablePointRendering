from __future__ import annotations

from typing import Dict

import numpy as np
import torch

import pale  # custom renderer bindings


MIN_SURFEL_SCALE = 1.0e-6


def _finite_min_max(x: torch.Tensor) -> tuple[float, float]:
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return float("nan"), float("nan")
    return float(finite.min().item()), float(finite.max().item())


def _assert_finite_np(name: str, array: np.ndarray) -> None:
    if np.isfinite(array).all():
        return
    bad_mask = ~np.isfinite(array)
    first_bad_flat = int(np.flatnonzero(bad_mask)[0])
    first_bad_index = np.unravel_index(first_bad_flat, array.shape)
    raise RuntimeError(
        f"{name} contains non-finite values: "
        f"bad_count={int(np.count_nonzero(bad_mask))}, "
        f"first_bad_index={first_bad_index}, "
        f"first_bad_value={array[first_bad_index]}"
    )


def _assert_positive_np(name: str, array: np.ndarray, min_value: float) -> None:
    bad_mask = ~(array >= min_value)
    if not np.any(bad_mask):
        return
    first_bad_flat = int(np.flatnonzero(bad_mask)[0])
    first_bad_index = np.unravel_index(first_bad_flat, array.shape)
    raise RuntimeError(
        f"{name} must be >= {min_value:g}: "
        f"bad_count={int(np.count_nonzero(bad_mask))}, "
        f"first_bad_index={first_bad_index}, "
        f"first_bad_value={array[first_bad_index]}"
    )



def get_forward_rgba(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    return np.asarray(forward_out[camera_name]["image"], dtype=np.float32, order="C")


def get_forward_rgb(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    return get_forward_rgba(forward_out, camera_name)[..., :3]


def get_forward_linear_rgba(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    return np.asarray(forward_out[camera_name]["raw"], dtype=np.float32, order="C")


def get_forward_linear_rgb(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    return get_forward_linear_rgba(forward_out, camera_name)[..., :3]


def _infer_hw_from_forward(forward_out: Dict[str, dict], camera_name: str) -> tuple[int, int]:
    image = np.asarray(forward_out[camera_name]["image"], dtype=np.float32, order="C")
    return int(image.shape[0]), int(image.shape[1])


def get_forward_depth_distortion(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "depth_distortion" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w), dtype=np.float32)

    depth = np.asarray(camera_out["depth_distortion"], dtype=np.float32, order="C")
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def get_forward_opacity_prior(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "opacity_prior" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w), dtype=np.float32)

    opacity_prior = np.asarray(camera_out["opacity_prior"], dtype=np.float32, order="C")
    return np.nan_to_num(opacity_prior, nan=0.0, posinf=0.0, neginf=0.0)


def get_forward_intra_slab_depth(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "intra_slab_depth" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w), dtype=np.float32)
    values = np.asarray(camera_out["intra_slab_depth"], dtype=np.float32, order="C")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def get_forward_intra_slab_depth_active_slab_count(
        forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "intra_slab_depth_active_slab_count" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w), dtype=np.uint32)
    return np.asarray(
        camera_out["intra_slab_depth_active_slab_count"],
        dtype=np.uint32,
        order="C",
    )


def get_forward_curvature_scale(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "curvature_scale" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w), dtype=np.float32)
    values = np.asarray(camera_out["curvature_scale"], dtype=np.float32, order="C")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def get_forward_curvature_scale_active_slab_count(
        forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "curvature_scale_active_slab_count" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w), dtype=np.uint32)
    return np.asarray(
        camera_out["curvature_scale_active_slab_count"],
        dtype=np.uint32,
        order="C",
    )


def get_forward_visible_normal(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "visible_normal" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w, 4), dtype=np.float32)

    normal = np.asarray(camera_out["visible_normal"], dtype=np.float32, order="C")
    return np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)


def get_forward_normal_from_depth(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "normal_from_depth" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w, 4), dtype=np.float32)

    normal = np.asarray(camera_out["normal_from_depth"], dtype=np.float32, order="C")
    return np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)


def get_forward_median_depth(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if "median_depth" not in camera_out:
        h, w = _infer_hw_from_forward(forward_out, camera_name)
        return np.zeros((h, w), dtype=np.float32)

    depth = np.asarray(camera_out["median_depth"], dtype=np.float32, order="C")
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def fetch_parameters(renderer: pale.Renderer) -> Dict[str, np.ndarray]:
    """
    Fetch all point parameters from the renderer as a dict of NumPy arrays.

    Expected keys and shapes:
        "position" : (N,3)
        "rotation" : (N,4), quaternion w,x,y,z
        "scale"    : (N,2)
        "albedo"   : (N,3)
        "opacity"  : (N,)
        "beta"     : (N,)
        "shape"    : (N,)
        "power"    : (N,)
        "densification_origin" : (N,), optional diagnostic provenance
        "primitive_age" : (N,), optional iterations since creation/last split
    """
    params = renderer.get_point_parameters()
    out: Dict[str, np.ndarray] = {}
    for key, value in params.items():
        out[key] = np.asarray(value, dtype=np.float32, order="C")
    return out

def orthonormalize_tangents_inplace(
    tangentU: torch.Tensor,
    tangentV: torch.Tensor,
    referenceDirection: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    if tangentU.ndim != 2 or tangentU.shape[1] != 3:
        raise ValueError(f"tangentU must be (N, 3), got {tuple(tangentU.shape)}")
    if tangentV.ndim != 2 or tangentV.shape[1] != 3:
        raise ValueError(f"tangentV must be (N, 3), got {tuple(tangentV.shape)}")
    if tangentU.shape != tangentV.shape:
        raise ValueError(
            f"tangentU and tangentV must have same shape, got "
            f"{tuple(tangentU.shape)} and {tuple(tangentV.shape)}"
        )

    with torch.no_grad():
        eps = 1.0e-8
        device = tangentU.device
        dtype = tangentU.dtype

        oldU = tangentU.clone()
        oldV = tangentV.clone()

        oldNormal = torch.cross(oldU, oldV, dim=1)
        oldNormalNorm = oldNormal.norm(dim=1, keepdim=True)
        hasValidOldNormal = oldNormalNorm.squeeze(1) > eps
        oldNormal = oldNormal / oldNormalNorm.clamp_min(eps)

        uNorm = tangentU.norm(dim=1, keepdim=True)
        badU = uNorm.squeeze(1) < eps

        if badU.any():
            tangentU[badU] = torch.tensor(
                [1.0, 0.0, 0.0],
                device=device,
                dtype=dtype,
            )
            uNorm = tangentU.norm(dim=1, keepdim=True)

        u = tangentU / uNorm.clamp_min(eps)

        v = tangentV - (tangentV * u).sum(dim=1, keepdim=True) * u
        vNorm = v.norm(dim=1, keepdim=True)
        badV = vNorm.squeeze(1) < eps

        if badV.any():
            uBad = u[badV]

            useXAxis = torch.abs(uBad[:, 0]) < 0.9
            aux = torch.zeros_like(uBad)
            aux[useXAxis] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
            aux[~useXAxis] = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

            aux = aux - (aux * uBad).sum(dim=1, keepdim=True) * uBad
            aux = torch.nn.functional.normalize(aux, dim=1, eps=eps)
            v[badV] = aux
            vNorm = v.norm(dim=1, keepdim=True)

        v = v / vNorm.clamp_min(eps)

        # Preserve the original normal orientation, if the old frame had one.
        newNormal = torch.cross(u, v, dim=1)
        newNormal = newNormal / newNormal.norm(dim=1, keepdim=True).clamp_min(eps)

        preserveMask = hasValidOldNormal
        flipMask = preserveMask & ((newNormal * oldNormal).sum(dim=1) < 0.0)

        v[flipMask] = -v[flipMask]
        newNormal[flipMask] = -newNormal[flipMask]

        # Optional explicit orientation only when requested.
        if referenceDirection is not None:
            ref = referenceDirection.to(device=device, dtype=dtype)

            if ref.ndim == 1:
                ref = ref.view(1, 3)
            if ref.shape[0] == 1:
                ref = ref.expand_as(u)
            if ref.shape != u.shape:
                raise ValueError(
                    f"referenceDirection must be (3,), (1,3), or (N,3), got {tuple(referenceDirection.shape)}"
                )

            ref = ref / ref.norm(dim=1, keepdim=True).clamp_min(eps)
            newNormal = torch.cross(u, v, dim=1)
            newNormal = newNormal / newNormal.norm(dim=1, keepdim=True).clamp_min(eps)

            refFlipMask = (newNormal * ref).sum(dim=1) < 0.0
            v[refFlipMask] = -v[refFlipMask]
            newNormal[refFlipMask] = -newNormal[refFlipMask]

        tangentU.copy_(u)
        tangentV.copy_(v)

        dotUV = (tangentU * tangentV).sum(dim=1)
        normU = tangentU.norm(dim=1)
        normV = tangentV.norm(dim=1)
        crossProduct = torch.cross(tangentU, tangentV, dim=1)
        crossNorm = crossProduct.norm(dim=1)

        return {
            "max_dev_norm_u": float((normU - 1.0).abs().max().item()),
            "max_dev_norm_v": float((normV - 1.0).abs().max().item()),
            "max_abs_dot_uv": float(dotUV.abs().max().item()),
            "min_cross_norm": float(crossNorm.min().item()),
        }

def verify_tangents_inplace(
    tangent_u: torch.Tensor,
    tangent_v: torch.Tensor,
    eps: float = 1e-8,
) -> None:
    """
    Enforce an orthonormal in-plane frame in-place:
    - normalize tangent_u
    - make tangent_v orthogonal to tangent_u
    - normalize tangent_v

    Expects shape [N, 3].
    """

    if tangent_u.ndim != 2 or tangent_v.ndim != 2 or tangent_u.shape != tangent_v.shape:
        raise ValueError(
            f"Expected tangent_u and tangent_v to have same shape [N, 3], "
            f"got {tangent_u.shape=} and {tangent_v.shape=}"
        )
    if tangent_u.shape[1] != 3:
        raise ValueError(f"Expected tangent tensors of shape [N, 3], got {tangent_u.shape}")

    with torch.no_grad():
        # Normalize u, with fallback for degenerate rows
        u_norm = torch.linalg.norm(tangent_u, dim=1, keepdim=True)
        bad_u = u_norm.squeeze(1) < eps

        if bad_u.any():
            tangent_u[bad_u] = torch.tensor(
                [1.0, 0.0, 0.0], device=tangent_u.device, dtype=tangent_u.dtype
            )
            u_norm = torch.linalg.norm(tangent_u, dim=1, keepdim=True)

        tangent_u.div_(u_norm.clamp_min(eps))

        # Remove projection of v onto u: v <- v - (u·v)u
        proj = torch.sum(tangent_v * tangent_u, dim=1, keepdim=True)
        tangent_v.sub_(proj * tangent_u)

        # Normalize v, with fallback if v became degenerate
        v_norm = torch.linalg.norm(tangent_v, dim=1, keepdim=True)
        bad_v = v_norm.squeeze(1) < eps

        if bad_v.any():
            u_bad = tangent_u[bad_v]

            # Build a safe auxiliary axis not parallel to u
            use_x = torch.abs(u_bad[:, 0]) < 0.9
            aux = torch.zeros_like(u_bad)
            aux[use_x] = torch.tensor([1.0, 0.0, 0.0], device=tangent_v.device, dtype=tangent_v.dtype)
            aux[~use_x] = torch.tensor([0.0, 1.0, 0.0], device=tangent_v.device, dtype=tangent_v.dtype)

            # Project aux into plane orthogonal to u
            aux = aux - torch.sum(aux * u_bad, dim=1, keepdim=True) * u_bad
            aux = F.normalize(aux, dim=1, eps=eps)

            tangent_v[bad_v] = aux
            v_norm = torch.linalg.norm(tangent_v, dim=1, keepdim=True)

        tangent_v.div_(v_norm.clamp_min(eps))

def verify_scales_inplace(scales: torch.Tensor) -> dict[str, float]:
    """
    In-place verification/clamping of scale values.

    Enforces:
        MIN_SURFEL_SCALE <= s_u, s_v <= 1.0
    """
    with torch.no_grad():
        s = scales.data
        before_min, before_max = _finite_min_max(s)
        nonfinite_count = int(torch.count_nonzero(~torch.isfinite(s)).item())

        s_clean = torch.nan_to_num(s, nan=MIN_SURFEL_SCALE, posinf=1.0, neginf=MIN_SURFEL_SCALE)
        s_clamped = torch.clamp(s_clean, min=MIN_SURFEL_SCALE, max=1.0)
        s.copy_(s_clamped)

        after_min, after_max = _finite_min_max(s)

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min": after_min,
            "after_max": after_max,
            "nonfinite_count": nonfinite_count,
        }

def verify_positions_inplace(positions: torch.Tensor) -> dict[str, float]:
    """
    In-place verification/clamping of position values.

    Enforces:
        -10.0 <= x, y, z <= 10.0
    """
    with torch.no_grad():
        p = positions.data
        before_min, before_max = _finite_min_max(p)
        nonfinite_count = int(torch.count_nonzero(~torch.isfinite(p)).item())

        p_clean = torch.nan_to_num(p, nan=0.0, posinf=5.0, neginf=-5.0)
        p_clamped = torch.clamp(p_clean, min=-5.0, max=5.0)
        p.copy_(p_clamped)

        after_min, after_max = _finite_min_max(p)

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min": after_min,
            "after_max": after_max,
            "nonfinite_count": nonfinite_count,
        }


def verify_albedos_inplace(albedos: torch.Tensor) -> dict[str, float]:
    """
    In-place verification/clamping of albedo values.

    Enforces:
        0.0 <= c <= 1.0
    """
    with torch.no_grad():
        s = albedos.data
        before_min, before_max = _finite_min_max(s)
        nonfinite_count = int(torch.count_nonzero(~torch.isfinite(s)).item())

        s_clean = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
        s_clamped = torch.clamp(s_clean, min=0.0, max=1.0)
        s.copy_(s_clamped)

        after_min, after_max = _finite_min_max(s)

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min": after_min,
            "after_max": after_max,
            "nonfinite_count": nonfinite_count,
        }


def verify_opacities_inplace(opacities: torch.Tensor) -> dict[str, float]:
    """
    In-place verification/clamping of albedo values.

    Enforces:
        0.0 <= c <= 1.0
    """
    with torch.no_grad():
        s = opacities.data
        before_min, before_max = _finite_min_max(s)
        nonfinite_count = int(torch.count_nonzero(~torch.isfinite(s)).item())

        s_clean = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
        s_clamped = torch.clamp(s_clean, min=0.0, max=1.0)
        s.copy_(s_clamped)

        after_min, after_max = _finite_min_max(s)

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min": after_min,
            "after_max": after_max,
            "nonfinite_count": nonfinite_count,
        }

def verify_beta_inplace(
        betas: torch.Tensor,
        trainable_surfel_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """
    In-place verification/clamping of beta values.

    Enforces:
        -2.0 <= beta <= 5.0

    If trainable_surfel_mask is provided, only trainable surfels are verified.
    Frozen surfels are left untouched.
    """
    min_beta_value = -2.5
    max_beta_value = 0.0
    with torch.no_grad():
        beta_values = betas.data

        before_min, before_max = _finite_min_max(beta_values)
        nonfinite_count = int(torch.count_nonzero(~torch.isfinite(beta_values)).item())
        beta_values.copy_(torch.nan_to_num(beta_values, nan=1.0, posinf=max_beta_value, neginf=min_beta_value))

        if trainable_surfel_mask is None:
            beta_values.clamp_(min=min_beta_value, max=max_beta_value)
        else:
            mask = trainable_surfel_mask.to(
                device=beta_values.device,
                dtype=torch.bool,
            )

            if mask.ndim != 1:
                raise RuntimeError(
                    f"trainable_surfel_mask must be 1D, got shape {tuple(mask.shape)}"
                )

            if beta_values.shape[0] != mask.shape[0]:
                raise RuntimeError(
                    "Beta/mask shape mismatch: "
                    f"betas has {beta_values.shape[0]} surfels, "
                    f"mask has {mask.shape[0]}"
                )

            beta_values[mask] = torch.clamp(
                beta_values[mask],
                min=min_beta_value,
                max=5.0,
            )

        after_min, after_max = _finite_min_max(beta_values)

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min": after_min,
            "after_max": after_max,
            "nonfinite_count": nonfinite_count,
        }
def apply_point_parameters(
        renderer: pale.Renderer,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
) -> None:
    positions_np = np.asarray(positions.detach().cpu().numpy(), dtype=np.float32, order="C")
    rotations_np = np.asarray(rotations.detach().cpu().numpy(), dtype=np.float32, order="C")
    scales_np = np.asarray(scales.detach().cpu().numpy(), dtype=np.float32, order="C")
    albedos_np = np.asarray(albedos.detach().cpu().numpy(), dtype=np.float32, order="C")
    opacities_np = np.asarray(opacities.detach().cpu().numpy(), dtype=np.float32, order="C")
    betas_np = np.asarray(betas.detach().cpu().numpy(), dtype=np.float32, order="C")
    powers_np = np.asarray(powers.detach().cpu().numpy(), dtype=np.float32, order="C")

    if positions_np.ndim != 2 or positions_np.shape[1] != 3:
        raise RuntimeError(f"position must have shape (N,3), got {positions_np.shape}")
    if rotations_np.ndim != 2 or rotations_np.shape[1] != 4:
        raise RuntimeError(f"rotation must have shape (N,4), got {rotations_np.shape}")
    if positions_np.shape[0] != rotations_np.shape[0]:
        raise RuntimeError(f"position/rotation point-count mismatch: {positions_np.shape[0]} vs {rotations_np.shape[0]}")

    arrays_to_check = {
        "position": positions_np,
        "rotation": rotations_np,
        "scale": scales_np,
        "albedo": albedos_np,
        "opacity": opacities_np,
        "beta": betas_np,
        "power": powers_np,
    }
    for name, array in arrays_to_check.items():
        _assert_finite_np(name, array)
    _assert_positive_np("scale", scales_np, MIN_SURFEL_SCALE)

    renderer.apply_point_optimization(
        {
            "position": positions_np,
            "rotation": rotations_np,
            "scale": scales_np,
            "albedo": albedos_np,
            "opacity": opacities_np,
            "beta": betas_np,
            "power": powers_np,
        }
    )


def _as_float32_contiguous(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32, order="C")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr


def _as_flat_float32(x: np.ndarray, name: str) -> np.ndarray:
    arr = _as_float32_contiguous(x, name)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise RuntimeError(f"{name} must have shape (N,) or (N,1), got {arr.shape}")
    return arr

def add_new_points(renderer, densification_result: dict | None) -> None:
    """
    Append newly created surfels to the C++ renderer.

    Expected format:

        densification_result = {
            "new": {
                "position": (N,3) float32,
                "rotation": (N,4) float32 quaternion w,x,y,z,
                "scale":    (N,2) float32,
                "albedo":   (N,3) float32,
                "opacity":  (N,) or (N,1) float32,
                "beta":     (N,) or (N,1) float32,
                "power":    (N,) or (N,1) float32, optional,
            }
        }
    """
    if densification_result is None:
        return

    new_block = densification_result.get("new")
    if new_block is None and "position" in densification_result:
        new_block = densification_result
    if new_block is None:
        return

    required_keys = ["position", "rotation", "scale", "albedo", "opacity", "beta"]
    for key in required_keys:
        if key not in new_block:
            raise RuntimeError(f"add_new_points: missing densification_result['new']['{key}']")

    position = _as_float32_contiguous(new_block["position"], "new.position")
    rotation = _as_float32_contiguous(new_block["rotation"], "new.rotation")
    scale = _as_float32_contiguous(new_block["scale"], "new.scale")
    albedo = _as_float32_contiguous(new_block["albedo"], "new.albedo")
    opacity = _as_flat_float32(new_block["opacity"], "new.opacity")
    beta = _as_flat_float32(new_block["beta"], "new.beta")

    if position.ndim != 2 or position.shape[1] != 3:
        raise RuntimeError(f"new.position must have shape (N,3), got {position.shape}")

    n_new = position.shape[0]
    if n_new == 0:
        return

    expected_shapes = {
        "new.rotation": (n_new, 4),
        "new.scale": (n_new, 2),
        "new.albedo": (n_new, 3),
    }

    arrays_to_check = {
        "new.position": position,
        "new.rotation": rotation,
        "new.scale": scale,
        "new.albedo": albedo,
        "new.opacity": opacity,
        "new.beta": beta,
    }

    for name, expected_shape in expected_shapes.items():
        if arrays_to_check[name].shape != expected_shape:
            raise RuntimeError(f"{name} must have shape {expected_shape}, got {arrays_to_check[name].shape}")

    if opacity.shape[0] != n_new:
        raise RuntimeError(f"new.opacity must have length {n_new}, got {opacity.shape[0]}")
    if beta.shape[0] != n_new:
        raise RuntimeError(f"new.beta must have length {n_new}, got {beta.shape[0]}")

    if "power" in new_block:
        power = _as_flat_float32(new_block["power"], "new.power")
        if power.shape[0] != n_new:
            raise RuntimeError(f"new.power must have length {n_new}, got {power.shape[0]}")
    else:
        power = np.zeros((n_new,), dtype=np.float32)

    arrays_to_check["new.power"] = power
    for name, array in arrays_to_check.items():
        _assert_finite_np(name, array)
    _assert_positive_np("new.scale", scale, MIN_SURFEL_SCALE)

    renderer.add_points(
        {
            "new": {
                "position": position,
                "rotation": rotation,
                "scale": scale,
                "albedo": albedo,
                "opacity": opacity,
                "beta": beta,
                "power": power,
            }
        }
    )


def remove_points(renderer: pale.Renderer, indices_to_remove: np.ndarray) -> None:
    """
    Remove points by index from the renderer's canonical point cloud.

    indices_to_remove:
        1D array-like of int (int32 or int64). Indices are in the current
        canonical ordering of the renderer (i.e., after the most recent
        fetch_parameters call).
    """
    indices_np = np.asarray(indices_to_remove, dtype=np.int64)
    if indices_np.ndim != 1:
        raise ValueError("remove_points: indices_to_remove must be 1D")

    renderer.remove_points({"indices": indices_np})

def rebuild_bvh(renderer: pale.Renderer) -> None:
    """
    Rebuild the renderer BVH/GPU buffers after point-cloud topology changes.
    """
    renderer.rebuild_bvh()

def get_training_camera_names(renderer: pale.Renderer) -> dict:
    return renderer.get_training_camera_names()  # C++ binding you implement

def get_all_camera_names(renderer: pale.Renderer) -> dict:
    return renderer.get_camera_names()  # C++ binding you implement
