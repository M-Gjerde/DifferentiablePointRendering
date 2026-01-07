#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ======================================================================================
# 2DGS PLY loader (binary_little_endian point_cloud.ply)
# ======================================================================================
def find_latest_2dgs_ply(output_root_path: Path) -> Path:
    if not output_root_path.exists():
        raise FileNotFoundError(f"Path '{output_root_path}' does not exist.")

    if output_root_path.is_file():
        if output_root_path.suffix.lower() == ".ply":
            return output_root_path
        raise ValueError(f"Expected a .ply file, got: {output_root_path}")

    candidate_plys: List[Path] = []
    for ply_path in output_root_path.rglob("point_cloud.ply"):
        if ply_path.is_file():
            candidate_plys.append(ply_path)

    if not candidate_plys:
        raise FileNotFoundError(f"No 'point_cloud.ply' found under '{output_root_path}'.")

    latest_ply = max(candidate_plys, key=lambda p: p.stat().st_mtime)
    return latest_ply


def read_ply_header_and_get_vertex_count_and_format(ply_path: Path) -> Tuple[int, str, int]:
    header_byte_length = 0
    header_lines: List[bytes] = []

    with ply_path.open("rb") as file_handle:
        while True:
            line = file_handle.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading PLY header.")
            header_lines.append(line)
            header_byte_length += len(line)
            if line.strip() == b"end_header":
                break

    fmt: Optional[str] = None
    vertex_count: Optional[int] = None

    for raw_line in header_lines:
        line = raw_line.decode("ascii", errors="ignore").strip()
        if line.startswith("format "):
            parts = line.split()
            if len(parts) >= 2:
                fmt = parts[1]
        if line.startswith("element vertex "):
            parts = line.split()
            if len(parts) == 3:
                vertex_count = int(parts[2])

    if vertex_count is None or fmt is None:
        raise RuntimeError("Failed to parse PLY header (missing format or element vertex).")
    if fmt != "binary_little_endian":
        raise RuntimeError(f"Expected binary_little_endian, got '{fmt}'")

    return vertex_count, fmt, header_byte_length


def parse_2dgs_binary_little_endian(
    ply_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads 2DGS point_cloud.ply (binary_little_endian) assuming 61 float32 per vertex:
      x y z
      nx ny nz
      f_dc_0 f_dc_1 f_dc_2
      f_rest_0..f_rest_44 (ignored)
      opacity
      scale_0 scale_1
      rot_0 rot_1 rot_2 rot_3   (assumed w,x,y,z)
    """
    vertex_count, _, header_len = read_ply_header_and_get_vertex_count_and_format(ply_path)

    floats_per_vertex = 3 + 3 + 3 + 45 + 1 + 2 + 4  # 61
    vertex_stride_bytes = floats_per_vertex * 4

    with ply_path.open("rb") as file_handle:
        file_handle.seek(header_len)
        raw = file_handle.read(vertex_count * vertex_stride_bytes)

    expected_bytes = vertex_count * vertex_stride_bytes
    if len(raw) != expected_bytes:
        raise RuntimeError(f"Unexpected data size: got {len(raw)} bytes, expected {expected_bytes} bytes")

    data = np.frombuffer(raw, dtype="<f4").reshape(vertex_count, floats_per_vertex)

    positions = data[:, 0:3]
    f_dc = data[:, 6:9]
    opacity_raw = data[:, 54]
    scale_0 = data[:, 55]
    scale_1 = data[:, 56]
    rot_wxyz = data[:, 57:61]

    # Opacity: sometimes stored as logit
    opacity_is_logit = (opacity_raw.min() < -1e-3) or (opacity_raw.max() > 1.0 + 1e-3)
    if opacity_is_logit:
        opacities01 = 1.0 / (1.0 + np.exp(-opacity_raw))
    else:
        opacities01 = np.clip(opacity_raw, 0.0, 1.0)

    # Scales: sometimes stored as log-scale
    scale_is_log = (np.percentile(scale_0, 10) < 0.0) or (np.percentile(scale_1, 10) < 0.0)
    if scale_is_log:
        scales_uv = np.stack([np.exp(scale_0), np.exp(scale_1)], axis=1)
    else:
        scales_uv = np.stack([scale_0, scale_1], axis=1)

    # Color decode: keep your current heuristic (swap if you have your exact mapping)
    c0 = 0.28209479177387814
    albedo01 = np.clip(f_dc.astype(np.float32) * c0 + 0.7, 0.0, 1.0)

    quats_wxyz = rot_wxyz.astype(np.float32)
    quats_wxyz = quats_wxyz / (np.linalg.norm(quats_wxyz, axis=1, keepdims=True) + 1e-12)

    return (
        positions.astype(np.float32),
        albedo01.astype(np.float32),
        opacities01.astype(np.float32),
        scales_uv.astype(np.float32),
        quats_wxyz.astype(np.float32),
    )


# ======================================================================================
# Quaternion -> (tu, tv)
# ======================================================================================
def rotate_vectors_by_quaternion_wxyz(quats_wxyz: np.ndarray, vectors_xyz: np.ndarray) -> np.ndarray:
    """
    Rotate vectors by unit quaternions.
    quats_wxyz: (N,4) as [w,x,y,z]
    vectors_xyz: (N,3)
    """
    q = quats_wxyz.astype(np.float32)
    v = vectors_xyz.astype(np.float32)

    w = q[:, 0:1]
    q_xyz = q[:, 1:4]

    # v' = v + 2*cross(q_xyz, cross(q_xyz, v) + w*v)
    t = 2.0 * np.cross(q_xyz, v)
    v_rot = v + w * t + np.cross(q_xyz, t)
    return v_rot


def quaternions_to_tangent_frame(quats_wxyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpret quaternion as orientation of the surfel local frame:
      local +X -> tu
      local +Y -> tv

    Returns orthonormal (tu, tv).
    """
    n = quats_wxyz.shape[0]
    local_x = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (n, 1))
    local_y = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (n, 1))

    tu = rotate_vectors_by_quaternion_wxyz(quats_wxyz, local_x)
    tv = rotate_vectors_by_quaternion_wxyz(quats_wxyz, local_y)

    tu = tu / (np.linalg.norm(tu, axis=1, keepdims=True) + 1e-12)
    tv = tv - (np.sum(tv * tu, axis=1, keepdims=True) * tu)
    tv = tv / (np.linalg.norm(tv, axis=1, keepdims=True) + 1e-12)

    return tu.astype(np.float32), tv.astype(np.float32)


# ======================================================================================
# Beta/shape initialization
# ======================================================================================
def compute_beta(
    su: np.ndarray,
    sv: np.ndarray,
    beta_mode: str,
    beta_constant: float,
    beta_scale_factor: float,
) -> np.ndarray:
    """
    beta_mode:
      - "constant": beta = beta_constant
      - "from_scale": beta ~ beta_scale_factor / (mean_sigma^2 + eps)
    """
    beta_mode = beta_mode.lower().strip()
    if beta_mode == "constant":
        return np.full_like(su, float(beta_constant), dtype=np.float32)

    if beta_mode == "from_scale":
        mean_sigma = 0.5 * (su + sv)
        eps = 1e-8
        beta = float(beta_scale_factor) / (mean_sigma * mean_sigma + eps)
        return beta.astype(np.float32)

    raise ValueError(f"Unsupported beta_mode '{beta_mode}' (expected constant|from_scale).")


# ======================================================================================
# Output: EXACT ASCII layout required by generate_volume_ply
# ======================================================================================
@dataclass(frozen=True)
class BetaSurfelVolumeLayout:
    positions: np.ndarray   # (N,3)
    tu: np.ndarray          # (N,3)
    tv: np.ndarray          # (N,3)
    su: np.ndarray          # (N,)
    sv: np.ndarray          # (N,)
    albedo: np.ndarray      # (N,3)
    opacity: np.ndarray     # (N,)
    beta: np.ndarray        # (N,)
    shape: np.ndarray       # (N,)


def convert_2dgs_to_volume_layout(
    input_path: Path,
    opacity_threshold: float,
    z_min: float,
    area_threshold: float,
    max_points: int,
    scale_mult: float,
    opacity_mult: float,
    beta_mode: str,
    beta_constant: float,
    beta_scale_factor: float,
    shape_constant: float,
) -> Tuple[BetaSurfelVolumeLayout, Path]:
    ply_path = find_latest_2dgs_ply(input_path)
    positions, albedo01, opacity01, scales_uv, quats_wxyz = parse_2dgs_binary_little_endian(ply_path)

    # Filters: opacity + z
    keep_opacity = opacity01 >= float(opacity_threshold)
    keep_z = positions[:, 2] >= float(z_min)
    keep = keep_opacity & keep_z

    positions = positions[keep]
    albedo01 = albedo01[keep]
    opacity01 = opacity01[keep]
    scales_uv = scales_uv[keep]
    quats_wxyz = quats_wxyz[keep]

    if positions.shape[0] == 0:
        raise RuntimeError("No 2DGS points left after opacity/z filtering. Lower thresholds.")

    # Area filter
    ellipse_area = scales_uv[:, 0] * scales_uv[:, 1]
    keep_area = ellipse_area >= float(area_threshold)

    positions = positions[keep_area]
    albedo01 = albedo01[keep_area]
    opacity01 = opacity01[keep_area]
    scales_uv = scales_uv[keep_area]
    quats_wxyz = quats_wxyz[keep_area]

    if positions.shape[0] == 0:
        raise RuntimeError("No 2DGS points left after area filtering. Lower --area-threshold.")

    # Clamp count (deterministic)
    if max_points and positions.shape[0] > int(max_points):
        positions = positions[:max_points]
        albedo01 = albedo01[:max_points]
        opacity01 = opacity01[:max_points]
        scales_uv = scales_uv[:max_points]
        quats_wxyz = quats_wxyz[:max_points]

    # Quaternion -> (tu, tv)
    tu, tv = quaternions_to_tangent_frame(quats_wxyz)

    # su/sv (after multiplier)
    su = (scales_uv[:, 0] * float(scale_mult)).astype(np.float32)
    sv = (scales_uv[:, 1] * float(scale_mult)).astype(np.float32)

    # opacity (after multiplier)
    opacity01 = np.clip(opacity01 * float(opacity_mult), 0.0, 1.0).astype(np.float32)

    # beta + shape
    beta = compute_beta(
        su=su,
        sv=sv,
        beta_mode=beta_mode,
        beta_constant=beta_constant,
        beta_scale_factor=beta_scale_factor,
    ).astype(np.float32)
    shape = np.full_like(beta, float(shape_constant), dtype=np.float32)

    out = BetaSurfelVolumeLayout(
        positions=positions.astype(np.float32),
        tu=tu.astype(np.float32),
        tv=tv.astype(np.float32),
        su=su,
        sv=sv,
        albedo=albedo01.astype(np.float32),
        opacity=opacity01,
        beta=beta,
        shape=shape,
    )
    return out, ply_path


def write_beta_surfels_ascii_ply_volume_layout(output_path: Path, cloud: BetaSurfelVolumeLayout) -> None:
    """
    EXACTLY matches generate_volume_ply layout:

    Header:
      x y z
      tu_x tu_y tu_z
      tv_x tv_y tv_z
      su sv
      albedo_r albedo_g albedo_b
      opacity
      beta
      shape
    """
    n = int(cloud.positions.shape[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.extend(
        [
            "ply",
            "format ascii 1.0",
            "comment Converted from 2DGS point_cloud.ply",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property float tu_x",
            "property float tu_y",
            "property float tu_z",
            "property float tv_x",
            "property float tv_y",
            "property float tv_z",
            "property float su",
            "property float sv",
            "property float albedo_r",
            "property float albedo_g",
            "property float albedo_b",
            "property float opacity",
            "property float beta",
            "property float shape",
            "end_header",
        ]
    )

    # Use the same numeric formatting style as your generator (.6f)
    for i in range(n):
        x, y, z = cloud.positions[i].tolist()
        tu_x, tu_y, tu_z = cloud.tu[i].tolist()
        tv_x, tv_y, tv_z = cloud.tv[i].tolist()
        su = float(cloud.su[i])
        sv = float(cloud.sv[i])
        r, g, b = cloud.albedo[i].tolist()
        opacity = float(cloud.opacity[i])
        beta = float(cloud.beta[i])
        shape = float(cloud.shape[i])

        lines.append(
            f"{x:.6f} {y:.6f} {z:.6f} "
            f"{tu_x:.6f} {tu_y:.6f} {tu_z:.6f} "
            f"{tv_x:.6f} {tv_y:.6f} {tv_z:.6f} "
            f"{su:.6f} {sv:.6f} "
            f"{r:.6f} {g:.6f} {b:.6f} "
            f"{opacity:.6f} {beta:.6f} {shape:.6f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ======================================================================================
# CLI
# ======================================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 2DGS point_cloud.ply to beta surfels in the generate_volume_ply ASCII layout."
    )
    parser.add_argument("--input", type=Path, required=True, help="2DGS run directory OR direct point_cloud.ply")
    parser.add_argument("--output", type=Path, required=True, help="Output ASCII PLY path")

    # Filters
    parser.add_argument("--opacity-threshold", type=float, default=0.0)
    parser.add_argument("--z-min", type=float, default=-1e30)
    parser.add_argument("--area-threshold", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=0)

    # Multipliers
    parser.add_argument("--scale-mult", type=float, default=1.0)
    parser.add_argument("--opacity-mult", type=float, default=1.0)

    # Beta/shape
    parser.add_argument("--beta-mode", type=str, default="constant", help="constant|from_scale")
    parser.add_argument("--beta-constant", type=float, default=0.0)
    parser.add_argument("--beta-scale-factor", type=float, default=1.0)
    parser.add_argument("--shape-constant", type=float, default=0.0)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    cloud, source_ply = convert_2dgs_to_volume_layout(
        input_path=args.input,
        opacity_threshold=float(args.opacity_threshold),
        z_min=float(args.z_min),
        area_threshold=float(args.area_threshold),
        max_points=int(args.max_points),
        scale_mult=float(args.scale_mult),
        opacity_mult=float(args.opacity_mult),
        beta_mode=str(args.beta_mode),
        beta_constant=float(args.beta_constant),
        beta_scale_factor=float(args.beta_scale_factor),
        shape_constant=float(args.shape_constant),
    )

    write_beta_surfels_ascii_ply_volume_layout(args.output, cloud)

    print(f"Source: {source_ply}")
    print(f"Wrote:  {args.output}")
    print(
        f"Count: {cloud.positions.shape[0]}\n"
        f"Filters: opacity>={args.opacity_threshold}, z>={args.z_min}, area>={args.area_threshold}\n"
        f"Multipliers: scale_mult={args.scale_mult}, opacity_mult={args.opacity_mult}\n"
        f"Beta: mode={args.beta_mode}, constant={args.beta_constant}, scale_factor={args.beta_scale_factor}\n"
        f"Shape: constant={args.shape_constant}"
    )


if __name__ == "__main__":
    main()
