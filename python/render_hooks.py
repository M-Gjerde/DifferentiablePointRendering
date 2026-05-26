from __future__ import annotations

from typing import Dict

import numpy as np
import torch

import pale  # custom renderer bindings


def fetch_parameters(renderer: pale.Renderer) -> Dict[str, np.ndarray]:
    """
    Fetch all point parameters from the renderer as a dict of NumPy arrays.

    Expected keys and shapes (matching the C++ bindings):
        "position"   : (N,3)
        "tangent_u"  : (N,3)
        "tangent_v"  : (N,3)
        "scale"      : (N,2)
        "albedo"      : (N,3)
        "opacity"    : (N,)
        "beta"       : (N,)
        "shape"      : (N,)
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
    """
    In-place Gram–Schmidt on (tangentU, tangentV) rows, enforcing:

        |tangentU| = |tangentV| = 1
        tangentU ⟂ tangentV
        n = tangentU × tangentV   (right-handed frame)
        dot(n, referenceDirection) >= 0 (orientation consistency)

    Args:
        tangentU: (N, 3) tensor of primary tangent directions.
        tangentV: (N, 3) tensor of secondary tangent directions.
        referenceDirection: (3,) or (1, 3) tensor specifying the desired
                            normal orientation hemisphere. If None, uses +Z.

    Returns:
        Dictionary with diagnostics about norms, orthogonality, and orientation.
    """
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
        epsilon = 1e-6
        device = tangentU.device
        dtype = tangentU.dtype

        primaryTangent = tangentU
        secondaryTangent = tangentV

        # 1. Normalize tangentU
        primaryNorm = primaryTangent.norm(dim=1, keepdim=True).clamp(min=1e-8)
        primaryUnit = primaryTangent / primaryNorm

        # 2. Gram–Schmidt orthogonalize tangentV against tangentU
        secondaryProjection = (secondaryTangent * primaryUnit).sum(dim=1, keepdim=True) * primaryUnit
        secondaryOrtho = secondaryTangent - secondaryProjection

        # Degeneracy fix: if secondaryOrtho is too small, choose a stable orthogonal vector
        squaredLengthSecondaryOrtho = (secondaryOrtho * secondaryOrtho).sum(dim=1, keepdim=True)
        degenerateMask = squaredLengthSecondaryOrtho < epsilon

        if degenerateMask.any():
            # Choose (0, 1, 0) if primaryUnit.y is not ~1, otherwise (1, 0, 0)
            useYAxisMask = (primaryUnit[:, 1].abs() < 0.9).view(-1, 1)

            yAxisVector = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3)
            xAxisVector = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 3)

            yAxisBatch = yAxisVector.expand_as(primaryUnit)
            xAxisBatch = xAxisVector.expand_as(primaryUnit)

            arbitraryDirection = torch.where(useYAxisMask, yAxisBatch, xAxisBatch)
            arbitraryDirection = arbitraryDirection - (
                (arbitraryDirection * primaryUnit).sum(dim=1, keepdim=True) * primaryUnit
            )

            secondaryOrtho = torch.where(degenerateMask, arbitraryDirection, secondaryOrtho)

        # Normalize tangentV
        secondaryNorm = secondaryOrtho.norm(dim=1, keepdim=True).clamp(min=1e-8)
        secondaryUnit = secondaryOrtho / secondaryNorm

        # 3. Compute right-handed normal
        normalVector = torch.cross(primaryUnit, secondaryUnit, dim=1)
        normalNorm = normalVector.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalUnit = normalVector / normalNorm

        # 4. Enforce consistent orientation w.r.t. referenceDirection
        if referenceDirection is None:
            referenceDirection = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

        if referenceDirection.ndim == 1:
            referenceDirection = referenceDirection.view(1, 3)
        elif referenceDirection.ndim != 2 or referenceDirection.shape[1] != 3:
            raise ValueError(
                f"referenceDirection must be (3,) or (1, 3) or (N, 3), "
                f"got {tuple(referenceDirection.shape)}"
            )

        if referenceDirection.shape[0] == 1:
            referenceDirection = referenceDirection.expand_as(primaryUnit)
        elif referenceDirection.shape[0] != primaryUnit.shape[0]:
            raise ValueError(
                f"referenceDirection batch size must be 1 or N={primaryUnit.shape[0]}, "
                f"got {referenceDirection.shape[0]}"
            )

        normalDotReference = (normalUnit * referenceDirection).sum(dim=1, keepdim=True)
        flipMask = normalDotReference < 0.0

        secondaryUnit = torch.where(flipMask, -secondaryUnit, secondaryUnit)
        normalUnit = torch.where(flipMask, -normalUnit, normalUnit)

        # 5. Write back in place
        tangentU.copy_(primaryUnit)
        tangentV.copy_(secondaryUnit)

        # Diagnostics
        dotUV = (tangentU * tangentV).sum(dim=1)
        normU = tangentU.norm(dim=1)
        normV = tangentV.norm(dim=1)
        crossProduct = torch.cross(tangentU, tangentV, dim=1)
        crossNorm = crossProduct.norm(dim=1)
        normalDotReferenceFinal = (normalUnit * referenceDirection).sum(dim=1)

        diagnostics: Dict[str, float] = {
            "max_dev_norm_u": float((normU - 1.0).abs().max().item()),
            "max_dev_norm_v": float((normV - 1.0).abs().max().item()),
            "max_abs_dot_uv": float(dotUV.abs().max().item()),
            "min_cross_norm": float(crossNorm.min().item()),
            "min_normal_dot_ref": float(normalDotReferenceFinal.min().item()),
        }
        return diagnostics

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
        0.001 <= s_u, s_v <= 1.0
    """
    with torch.no_grad():
        s = scales.data
        before_min = float(s.min().item())
        before_max = float(s.max().item())

        s_clamped = torch.clamp(s, min=0.00, max=1.0) ## TODO Enforcing min size matching photon map min resolution
        s.copy_(s_clamped)

        after_min = float(s.min().item())
        after_max = float(s.max().item())

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min": after_min,
            "after_max": after_max,
        }

def verify_positions_inplace(positions: torch.Tensor) -> dict[str, float]:
    """
    In-place verification/clamping of position values.

    Enforces:
        -10.0 <= x, y, z <= 10.0
    """
    with torch.no_grad():
        p = positions.data
        before_min = float(p.min().item())
        before_max = float(p.max().item())

        p_clamped = torch.clamp(p, min=-10.0, max=10.0)
        p.copy_(p_clamped)

        after_min = float(p.min().item())
        after_max = float(p.max().item())

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min": after_min,
            "after_max": after_max,
        }


def verify_albedos_inplace(albedos: torch.Tensor) -> dict[str, float]:
    """
    In-place verification/clamping of albedo values.

    Enforces:
        0.0 <= c <= 1.0
    """
    with torch.no_grad():
        s = albedos.data
        before_min = float(s.min().item())
        before_max = float(s.max().item())

        s_clamped = torch.clamp(s, min=0.0, max=1.0)
        s.copy_(s_clamped)

        after_min = float(s.min().item())
        after_max = float(s.max().item())

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min": after_min,
            "after_max": after_max,
        }


def verify_opacities_inplace(opacities: torch.Tensor) -> dict[str, float]:
    """
    In-place verification/clamping of albedo values.

    Enforces:
        0.0 <= c <= 1.0
    """
    with torch.no_grad():
        s = opacities.data
        before_min = float(s.min().item())
        before_max = float(s.max().item())

        s_clamped = torch.clamp(s, min=0.0, max=1.0)
        s.copy_(s_clamped)

        after_min = float(s.min().item())
        after_max = float(s.max().item())

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min": after_min,
            "after_max": after_max,
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
    min_beta_value = -1.5
    with torch.no_grad():
        beta_values = betas.data

        before_min = float(beta_values.min().item())
        before_max = float(beta_values.max().item())

        if trainable_surfel_mask is None:
            beta_values.clamp_(min=min_beta_value, max=5.0)
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

        after_min = float(beta_values.min().item())
        after_max = float(beta_values.max().item())

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min": after_min,
            "after_max": after_max,
        }

def apply_point_parameters(
        renderer: pale.Renderer,
        positions: torch.Tensor,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scales: torch.Tensor,
        albedos: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
) -> None:
    """
    Push updated positions, tangent_u, tangent_v, scales, and albedos into the renderer.

    Expects tensors of shape (N,3) for position/tangents, (N,2) for scales,
    (N,3) for albedos, on any device.
    """
    positions_np = np.asarray(
        positions.detach().cpu().numpy(), dtype=np.float32, order="C"
    )
    tangent_u_np = np.asarray(
        tangent_u.detach().cpu().numpy(), dtype=np.float32, order="C"
    )
    tangent_v_np = np.asarray(
        tangent_v.detach().cpu().numpy(), dtype=np.float32, order="C"
    )
    scales_np = np.asarray(
        scales.detach().cpu().numpy(), dtype=np.float32, order="C"
    )
    albedos_np = np.asarray(
        albedos.detach().cpu().numpy(), dtype=np.float32, order="C"
    )

    opacities_np = np.asarray(
        opacities.detach().cpu().numpy(), dtype=np.float32, order="C"
    )
    betas_np = np.asarray(
        betas.detach().cpu().numpy(), dtype=np.float32, order="C"
    )
    powers_np = np.asarray(
        powers.detach().cpu().numpy(), dtype=np.float32, order="C"
    )

    if positions_np.shape != tangent_u_np.shape or positions_np.shape != tangent_v_np.shape:
        raise RuntimeError(
            f"Shape mismatch between position {positions_np.shape}, "
            f"tangent_u {tangent_u_np.shape}, tangent_v {tangent_v_np.shape}"
        )

    renderer.apply_point_optimization(
        {
            "position": positions_np,
            "tangent_u": tangent_u_np,
            "tangent_v": tangent_v_np,
            "scale": scales_np,
            "albedo": albedos_np,
            "opacity": opacities_np,
            "beta": betas_np,
            "power": powers_np
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
    Append newly created surfels/Gaussians to the C++ renderer.

    Expected preferred format:

        densification_result = {
            "new": {
                "position":  (N,3) float32,
                "tangent_u": (N,3) float32,
                "tangent_v": (N,3) float32,
                "scale":     (N,2) float32,
                "albedo":    (N,3) float32,
                "opacity":   (N,) or (N,1) float32,
                "beta":      (N,) or (N,1) float32,
                "power":     (N,) or (N,1) float32, optional,
            }
        }

    For compatibility, this also accepts a flat densification_result
    containing the same keys directly.
    """
    if densification_result is None:
        return

    # Preferred format: {"new": {...}}
    new_block = densification_result.get("new")

    # Backward-compatible format: {"position": ..., "scale": ..., ...}
    if new_block is None and "position" in densification_result:
        new_block = densification_result

    if new_block is None:
        return

    required_keys = [
        "position",
        "tangent_u",
        "tangent_v",
        "scale",
        "albedo",
        "opacity",
        "beta",
    ]

    for key in required_keys:
        if key not in new_block:
            raise RuntimeError(f"add_new_points: missing densification_result['new']['{key}']")

    position = _as_float32_contiguous(new_block["position"], "new.position")
    tangent_u = _as_float32_contiguous(new_block["tangent_u"], "new.tangent_u")
    tangent_v = _as_float32_contiguous(new_block["tangent_v"], "new.tangent_v")
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
        "new.tangent_u": (n_new, 3),
        "new.tangent_v": (n_new, 3),
        "new.scale": (n_new, 2),
        "new.albedo": (n_new, 3),
    }

    arrays_to_check = {
        "new.tangent_u": tangent_u,
        "new.tangent_v": tangent_v,
        "new.scale": scale,
        "new.albedo": albedo,
    }

    for name, expected_shape in expected_shapes.items():
        if arrays_to_check[name].shape != expected_shape:
            raise RuntimeError(
                f"{name} must have shape {expected_shape}, got {arrays_to_check[name].shape}"
            )

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

    parameters_for_cpp = {
        "new": {
            "position": position,
            "tangent_u": tangent_u,
            "tangent_v": tangent_v,
            "scale": scale,
            "albedo": albedo,
            "opacity": opacity,
            "beta": beta,
            "power": power,
        }
    }

    renderer.add_points(parameters_for_cpp)


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
    new_points dict has keys: position, tangent_u, tangent_v, scale, albedo.
    This function is responsible for telling the renderer to append these
    to its point cloud asset and rebuild its BVH/GPU buffers.
    """
    renderer.rebuild_bvh()  # C++ binding you implement

def get_training_camera_names(renderer: pale.Renderer) -> dict:
    return renderer.get_training_camera_names()  # C++ binding you implement

def get_all_camera_names(renderer: pale.Renderer) -> dict:
    return renderer.get_camera_names()  # C++ binding you implement
