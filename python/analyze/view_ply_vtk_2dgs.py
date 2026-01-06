#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
import vtk


def find_latest_2dgs_ply(output_root_path: Path) -> Path:
    if not output_root_path.exists():
        raise FileNotFoundError(f"Path '{output_root_path}' does not exist.")

    if output_root_path.is_file():
        print(f"Using PLY file: {output_root_path}")
        return output_root_path

    # Common 2DGS layout: <run>/point_cloud/iteration_xxxxx/point_cloud.ply
    candidate_plys: List[Path] = []
    for ply_path in output_root_path.rglob("point_cloud.ply"):
        if ply_path.is_file():
            candidate_plys.append(ply_path)

    if not candidate_plys:
        raise FileNotFoundError(f"No 'point_cloud.ply' found under '{output_root_path}'.")

    latest_ply = max(candidate_plys, key=lambda p: p.stat().st_mtime)
    print(f"Using latest: {latest_ply}")
    return latest_ply


def read_ply_header_and_get_vertex_count_and_format(ply_path: Path) -> Tuple[int, str, int]:
    """
    Returns:
      vertex_count
      format_str: 'ascii' or 'binary_little_endian'
      header_byte_length: number of bytes up to and including end_header newline
    """
    header_lines: List[bytes] = []
    header_byte_length = 0
    with ply_path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading PLY header.")
            header_lines.append(line)
            header_byte_length += len(line)
            if line.strip() == b"end_header":
                # header may or may not have a trailing newline; our counting includes it already
                break

    fmt = None
    vertex_count = None

    for raw_line in header_lines:
        line = raw_line.decode("ascii", errors="ignore").strip()
        if line.startswith("format "):
            # format binary_little_endian 1.0
            parts = line.split()
            if len(parts) >= 2:
                fmt = parts[1]
        if line.startswith("element vertex "):
            parts = line.split()
            if len(parts) == 3:
                vertex_count = int(parts[2])

    if vertex_count is None or fmt is None:
        raise RuntimeError("Failed to parse PLY header (missing format or element vertex).")

    if fmt not in ("ascii", "binary_little_endian"):
        raise RuntimeError(f"Unsupported PLY format: '{fmt}'")

    return vertex_count, fmt, header_byte_length


def parse_2dgs_binary_little_endian(
    ply_path: Path,
    opacity_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads 2DGS point_cloud.ply (binary_little_endian) with:
      x y z
      nx ny nz
      f_dc_0 f_dc_1 f_dc_2
      f_rest_0..f_rest_44 (ignored)
      opacity
      scale_0 scale_1
      rot_0 rot_1 rot_2 rot_3
    Returns:
      positions (N,3) float32
      colors01  (N,3) float32 in [0,1] (using f_dc as-is, clipped)
      opacities01 (N,) float32 in [0,1]   (sigmoid if needed? see note below)
      scales_uv (N,2) float32 (exp if log-scale, see note below)
      quats_wxyz (N,4) float32
    """
    vertex_count, fmt, header_len = read_ply_header_and_get_vertex_count_and_format(ply_path)
    if fmt != "binary_little_endian":
        raise RuntimeError(f"Expected binary_little_endian PLY, got '{fmt}'")

    floats_per_vertex = 3 + 3 + 3 + 45 + 1 + 2 + 4  # 61
    vertex_stride_bytes = floats_per_vertex * 4

    with ply_path.open("rb") as f:
        f.seek(header_len)
        raw = f.read(vertex_count * vertex_stride_bytes)

    if len(raw) != vertex_count * vertex_stride_bytes:
        raise RuntimeError(
            f"Unexpected data size: got {len(raw)} bytes, expected {vertex_count * vertex_stride_bytes} bytes"
        )

    data = np.frombuffer(raw, dtype="<f4").reshape(vertex_count, floats_per_vertex)

    positions = data[:, 0:3]
    normals = data[:, 3:6]  # not used for glyph orientation here (we use quaternion)
    f_dc = data[:, 6:9]
    # f_rest = data[:, 9:54]  # ignored (45)
    opacity = data[:, 54]
    scale_0 = data[:, 55]
    scale_1 = data[:, 56]
    rot = data[:, 57:61]

    # ----- conventions handling -----
    # Many 3DGS/2DGS pipelines store:
    #   opacity = logit(alpha)
    #   scales = log(s)
    # Some store already-activated values.
    #
    # We provide a robust heuristic:
    # - If opacity has values outside [0,1], treat as logit and sigmoid it.
    # - If scales are often negative, treat as log and exp it.
    opacity_is_logit = (opacity.min() < -1e-3) or (opacity.max() > 1.0 + 1e-3)
    if opacity_is_logit:
        opacities01 = 1.0 / (1.0 + np.exp(-opacity))
    else:
        opacities01 = np.clip(opacity, 0.0, 1.0)

    scale_is_log = (np.percentile(scale_0, 10) < 0.0) or (np.percentile(scale_1, 10) < 0.0)
    if scale_is_log:
        scales_uv = np.stack([np.exp(scale_0), np.exp(scale_1)], axis=1)
    else:
        scales_uv = np.stack([scale_0, scale_1], axis=1)

    colors01 = np.clip(f_dc.astype(np.float32), 0.0, 1.0)

    # Quaternion: assume file order is (rot_0..3) = (w,x,y,z)
    quats_wxyz = rot.astype(np.float32)
    quat_norm = np.linalg.norm(quats_wxyz, axis=1, keepdims=True) + 1e-12
    quats_wxyz = quats_wxyz / quat_norm

    # Filter by opacity threshold (in activated alpha space)
    keep = opacities01 >= float(opacity_threshold)
    positions = positions[keep].astype(np.float32)
    colors01 = colors01[keep].astype(np.float32)
    opacities01 = opacities01[keep].astype(np.float32)
    scales_uv = scales_uv[keep].astype(np.float32)
    quats_wxyz = quats_wxyz[keep].astype(np.float32)

    if positions.shape[0] == 0:
        raise RuntimeError("No points left after opacity filtering. Lower --opacity-threshold.")

    print(
        f"Loaded {positions.shape[0]} / {vertex_count} points from {ply_path}\n"
        f"Opacity encoding: {'logit->sigmoid' if opacity_is_logit else 'raw'}\n"
        f"Scale encoding: {'log->exp' if scale_is_log else 'raw'}"
    )
    return positions, colors01, opacities01, scales_uv, quats_wxyz


def numpy_rgba_u8(name: str, rgb01: np.ndarray, alpha01: np.ndarray) -> vtk.vtkUnsignedCharArray:
    rgb_u8 = (np.asarray(rgb01, dtype=np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    a_u8 = (np.asarray(alpha01, dtype=np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.concatenate([rgb_u8, a_u8.reshape(-1, 1)], axis=1)

    arr = vtk.vtkUnsignedCharArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(4)
    arr.SetNumberOfTuples(rgba.shape[0])
    for i in range(rgba.shape[0]):
        arr.SetTuple4(i, int(rgba[i, 0]), int(rgba[i, 1]), int(rgba[i, 2]), int(rgba[i, 3]))
    return arr


def numpy_to_vtk_float_array(name: str, data: np.ndarray, num_components: int) -> vtk.vtkFloatArray:
    flat = np.asarray(data, dtype=np.float32).reshape(data.shape[0], num_components)
    arr = vtk.vtkFloatArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(num_components)
    arr.SetNumberOfTuples(flat.shape[0])
    for i in range(flat.shape[0]):
        arr.SetTuple(i, flat[i].tolist())
    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VTK viewer for 2DGS point_cloud.ply (binary_little_endian).")
    parser.add_argument("--input", type=Path, required=True, help="Run dir or point_cloud.ply path.")
    parser.add_argument("--opacity-threshold", type=float, default=0.0)
    parser.add_argument("--area-threshold", type=float, default=0.0, help="Filter by scale_0*scale_1 (after decoding).")
    parser.add_argument("--max-ellipses", type=int, default=0)
    parser.add_argument("--disk-resolution", type=int, default=16)
    parser.add_argument("--alpha-mult", type=float, default=1.0, help="Multiply alpha for visualization.")
    parser.add_argument("--scale-mult", type=float, default=1.0, help="Multiply scales for visualization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ply_path = find_latest_2dgs_ply(args.input)

    positions, colors01, opacities01, scales_uv, quats_wxyz = parse_2dgs_binary_little_endian(
        ply_path, opacity_threshold=args.opacity_threshold
    )

    # Area filtering
    ellipse_area = scales_uv[:, 0] * scales_uv[:, 1]
    keep_area = ellipse_area >= float(args.area_threshold)

    positions = positions[keep_area]
    colors01 = colors01[keep_area]
    opacities01 = opacities01[keep_area]
    scales_uv = scales_uv[keep_area]
    quats_wxyz = quats_wxyz[keep_area]

    if args.max_ellipses and positions.shape[0] > args.max_ellipses:
        positions = positions[: args.max_ellipses]
        colors01 = colors01[: args.max_ellipses]
        opacities01 = opacities01[: args.max_ellipses]
        scales_uv = scales_uv[: args.max_ellipses]
        quats_wxyz = quats_wxyz[: args.max_ellipses]

    print(f"Rendering {positions.shape[0]} 2DGS ellipses")

    # Build vtkPolyData
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToFloat()
    vtk_points.SetNumberOfPoints(int(positions.shape[0]))
    for i in range(int(positions.shape[0])):
        vtk_points.SetPoint(i, float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2]))

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)

    # Orientation quaternion (w,x,y,z) expected by SetOrientationModeToQuaternion
    poly.GetPointData().AddArray(numpy_to_vtk_float_array("orientation", quats_wxyz, 4))

    # Scale vector: (su, sv, 1)
    scales = np.stack(
        [scales_uv[:, 0] * float(args.scale_mult), scales_uv[:, 1] * float(args.scale_mult), np.ones_like(scales_uv[:, 0])],
        axis=1,
    ).astype(np.float32)
    poly.GetPointData().AddArray(numpy_to_vtk_float_array("scale", scales, 3))

    # Color with alpha
    alpha_vis = np.clip(opacities01 * float(args.alpha_mult), 0.0, 1.0)
    poly.GetPointData().AddArray(numpy_rgba_u8("color_rgba", colors01, alpha_vis))

    # Glyph source: disk in XY plane with radius 1
    disk = vtk.vtkDiskSource()
    disk.SetInnerRadius(0.0)
    disk.SetOuterRadius(1.0)
    disk.SetRadialResolution(1)
    disk.SetCircumferentialResolution(int(args.disk_resolution))
    disk.Update()

    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(poly)
    mapper.SetSourceConnection(disk.GetOutputPort())

    mapper.SetOrientationArray("orientation")
    mapper.SetOrientationModeToQuaternion()

    mapper.SetScaleArray("scale")
    mapper.SetScaleModeToScaleByVectorComponents()
    mapper.ScalingOn()

    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("color_rgba")
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().SetAmbient(0.25)
    actor.GetProperty().SetDiffuse(0.75)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.2, 0.2, 0.25)
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(0.1)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1200, 900)
    window.SetAlphaBitPlanes(True)
    window.SetMultiSamples(0)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)

    renderer.ResetCamera()
    window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
