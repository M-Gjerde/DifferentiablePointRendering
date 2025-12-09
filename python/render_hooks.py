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

        s_clamped = torch.clamp(s, min=0.00, max=1.0) ## TODO Enforcing a max size for the gaussians. Look if its possible to avoid this.
        s.copy_(s_clamped)

        after_min = float(s.min().item())
        after_max = float(s.max().item())

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

        s_clamped = torch.clamp(s, min=0.1, max=1.0)
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

def verify_beta_inplace(betas: torch.Tensor) -> dict[str, float]:
    """
    In-place verification/clamping of albedo values.

    Enforces:
        0.0 <= c <= 1.0
    """
    with torch.no_grad():
        s = betas.data
        before_min = float(s.min().item())
        before_max = float(s.max().item())

        s_clamped = torch.clamp(s, min=-3.0, max=1.0)
        s.copy_(s_clamped)

        after_min = float(s.min().item())
        after_max = float(s.max().item())

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
            "beta": betas_np
        }
    )


def add_new_points(renderer, densification_result: dict | None) -> None:
    if densification_result is None:
        return

    # densification_result is assumed to have already been applied to the
    # PyTorch tensors (positions, scales, etc.) via your Python logic.
    # Here we only care about actually appending new Gaussians.

    new_block = densification_result.get("new")
    if new_block is None:
        # No new points to append
        return

    # new_block should already be numpy arrays with shapes
    # (N,3) for position/tangent_u/tangent_v/albedo, (N,2) for scale
    parameters_for_cpp = {
        "new": {
            "position": new_block["position"],
            "tangent_u": new_block["tangent_u"],
            "tangent_v": new_block["tangent_v"],
            "scale": new_block["scale"],
            "albedo": new_block["albedo"],
            "opacity": new_block["opacity"],
            "beta": new_block["beta"],
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

def get_camera_names(renderer: pale.Renderer) -> dict:
    return renderer.get_camera_names()  # C++ binding you implement
