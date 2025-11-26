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
        "color"      : (N,3)
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
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
) -> dict[str, float]:
    """
    In-place Gram–Schmidt on (tangent_u, tangent_v) rows, enforcing:

        |tangent_u| = |tangent_v| = 1
        tangent_u ⟂ tangent_v
        n = tangent_u × tangent_v  (right-handed frame)

    Returns some diagnostics.
    """
    with torch.no_grad():
        tu = tangent_u.data
        tv = tangent_v.data

        tu_norm = tu.norm(dim=1, keepdim=True).clamp(min=1e-8)
        tu_unit = tu / tu_norm

        tv_proj = (tv * tu_unit).sum(dim=1, keepdim=True) * tu_unit
        tv_orth = tv - tv_proj

        tv_norm = tv_orth.norm(dim=1, keepdim=True).clamp(min=1e-8)
        tv_unit = tv_orth / tv_norm

        tangent_u.data.copy_(tu_unit)
        tangent_v.data.copy_(tv_unit)

        dot_uv = (tangent_u * tangent_v).sum(dim=1)
        norm_u = tangent_u.norm(dim=1)
        norm_v = tangent_v.norm(dim=1)
        cross = torch.cross(tangent_u, tangent_v, dim=1)
        cross_norm = cross.norm(dim=1)

        return {
            "max_dev_norm_u": float((norm_u - 1.0).abs().max().item()),
            "max_dev_norm_v": float((norm_v - 1.0).abs().max().item()),
            "max_abs_dot_uv": float(dot_uv.abs().max().item()),
            "min_cross_norm": float(cross_norm.min().item()),
        }


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

        s_clamped = torch.clamp(s, min=0.00, max=0.3) ## TODO Enforcing a max size for the gaussians. Look if its possible to avoid this.
        s.copy_(s_clamped)

        after_min = float(s.min().item())
        after_max = float(s.max().item())

        return {
            "before_min": before_min,
            "before_max": before_max,
            "after_min": after_min,
            "after_max": after_max,
        }


def verify_colors_inplace(colors: torch.Tensor) -> dict[str, float]:
    """
    In-place verification/clamping of color values.

    Enforces:
        0.0 <= c <= 1.0
    """
    with torch.no_grad():
        s = colors.data
        before_min = float(s.min().item())
        before_max = float(s.max().item())

        s_clamped = torch.clamp(s, min=0.0, max=0.99)
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
    In-place verification/clamping of color values.

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


def apply_point_parameters(
        renderer: pale.Renderer,
        positions: torch.Tensor,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
) -> None:
    """
    Push updated positions, tangent_u, tangent_v, scales, and colors into the renderer.

    Expects tensors of shape (N,3) for position/tangents, (N,2) for scales,
    (N,3) for colors, on any device.
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
    colors_np = np.asarray(
        colors.detach().cpu().numpy(), dtype=np.float32, order="C"
    )

    opacities_np = np.asarray(
        opacities.detach().cpu().numpy(), dtype=np.float32, order="C"
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
            "color": colors_np,
            "opacity": opacities_np
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
    # (N,3) for position/tangent_u/tangent_v/color, (N,2) for scale
    parameters_for_cpp = {
        "new": {
            "position": new_block["position"],
            "tangent_u": new_block["tangent_u"],
            "tangent_v": new_block["tangent_v"],
            "scale": new_block["scale"],
            "color": new_block["color"],
            "opacity": new_block["opacity"],
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
    new_points dict has keys: position, tangent_u, tangent_v, scale, color.
    This function is responsible for telling the renderer to append these
    to its point cloud asset and rebuild its BVH/GPU buffers.
    """
    renderer.rebuild_bvh()  # C++ binding you implement
