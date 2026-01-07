#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as et
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import vtk


# ======================================================================================
# Camera parsing (Mitsuba XML)  (unchanged)
# ======================================================================================
@dataclass(frozen=False)
class ParsedSensor:
    sensor_id: str
    fov_degrees: float
    fov_axis: str  # "x" or "y"
    near_clip: float
    far_clip: float
    width: int
    height: int
    origin: np.ndarray  # (3,)
    target: np.ndarray  # (3,)
    up: np.ndarray      # (3,)


def _parse_vec3_csv(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: '{text}'")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def parse_mitsuba_scene_sensors(xml_path: Path) -> List[ParsedSensor]:
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML scene not found: {xml_path}")

    tree = et.parse(str(xml_path))
    root = tree.getroot()

    sensors: List[ParsedSensor] = []
    for sensor_elem in root.findall("./sensor"):
        if sensor_elem.attrib.get("type", "") != "perspective":
            continue

        sensor_id = sensor_elem.attrib.get("id", "")
        if not sensor_id:
            continue

        fov_axis = "y"
        fov_degrees: Optional[float] = None
        near_clip: float = 0.01
        far_clip: float = 1000.0
        width: int = 800
        height: int = 600

        for child in list(sensor_elem):
            tag = child.tag
            name = child.attrib.get("name", "")

            if tag == "string" and name == "fov_axis":
                fov_axis = child.attrib.get("value", "y").strip()
            elif tag == "float" and name == "fov":
                fov_degrees = float(child.attrib["value"])
            elif tag == "float" and name == "near_clip":
                near_clip = float(child.attrib["value"])
            elif tag == "float" and name == "far_clip":
                far_clip = float(child.attrib["value"])
            elif tag == "film":
                width_elem = child.find("./integer[@name='width']")
                height_elem = child.find("./integer[@name='height']")
                if width_elem is not None:
                    width = int(width_elem.attrib["value"])
                if height_elem is not None:
                    height = int(height_elem.attrib["value"])

        if fov_degrees is None:
            raise ValueError(f"Sensor '{sensor_id}' has no <float name='fov' ...>")

        lookat_elem = sensor_elem.find("./transform[@name='to_world']/lookat")
        if lookat_elem is None:
            raise ValueError(f"Sensor '{sensor_id}' missing to_world/lookat")

        origin = _parse_vec3_csv(lookat_elem.attrib["origin"])
        target = _parse_vec3_csv(lookat_elem.attrib["target"])
        up = _parse_vec3_csv(lookat_elem.attrib["up"])

        sensors.append(
            ParsedSensor(
                sensor_id=sensor_id,
                fov_degrees=float(fov_degrees),
                fov_axis=fov_axis,
                near_clip=float(near_clip),
                far_clip=float(far_clip),
                width=int(width),
                height=int(height),
                origin=origin,
                target=target,
                up=up,
            )
        )

    sensors.sort(key=lambda s: s.sensor_id)
    if not sensors:
        raise RuntimeError(f"No perspective <sensor> found in: {xml_path}")
    return sensors


def _compute_vtk_view_angle_degrees(sensor: ParsedSensor) -> float:

    print(sensor.fov_axis.lower())
    aspect = float(sensor.width) / float(sensor.height)
    half_fov_x = math.radians(sensor.fov_degrees) * 0.5
    half_fov_y = math.atan(math.tan(half_fov_x) / max(aspect, 1e-12))
    return math.degrees(2.0 * half_fov_y) - 3


def apply_sensor_to_vtk_camera(vtk_camera: vtk.vtkCamera, sensor: ParsedSensor) -> None:
    vtk_camera.SetPosition(float(sensor.origin[0]), float(sensor.origin[1]), float(sensor.origin[2]))
    vtk_camera.SetFocalPoint(float(sensor.target[0]), float(sensor.target[1]), float(sensor.target[2]))
    vtk_camera.SetViewUp(float(sensor.up[0]), float(sensor.up[1]), float(sensor.up[2]))
    vtk_camera.SetViewAngle(_compute_vtk_view_angle_degrees(sensor))
    vtk_camera.SetClippingRange(float(sensor.near_clip), float(sensor.far_clip))


# ======================================================================================
# Shared VTK helpers
# ======================================================================================
def numpy_to_vtk_float_array(name: str, data: np.ndarray, num_components: int) -> vtk.vtkFloatArray:
    flat = np.asarray(data, dtype=np.float32).reshape(data.shape[0], num_components)
    array_handle = vtk.vtkFloatArray()
    array_handle.SetName(name)
    array_handle.SetNumberOfComponents(num_components)
    array_handle.SetNumberOfTuples(flat.shape[0])
    for i in range(flat.shape[0]):
        array_handle.SetTuple(i, flat[i].tolist())
    return array_handle


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


def write_render_to_png(render_window: vtk.vtkRenderWindow, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.ReadFrontBufferOff()
    window_to_image.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(window_to_image.GetOutputPort())
    writer.Write()


# ======================================================================================
# Beta-surfel PLY loader (your ASCII layout)  (unchanged)
# ======================================================================================
def find_latest_beta_surfel_points_ply(output_root_path: Path) -> Path:
    if not output_root_path.exists():
        raise FileNotFoundError(f"Path '{output_root_path}' does not exist.")

    if output_root_path.is_file():
        print(f"Using PLY file: {output_root_path}")
        return output_root_path

    points_in_root = output_root_path / "points_final.ply"
    if points_in_root.is_file():
        print(f"Using points_final.ply in run directory: {output_root_path}")
        return points_in_root

    candidate_run_dirs: List[Path] = []
    for child_path in output_root_path.iterdir():
        if child_path.is_dir() and (child_path / "points_final.ply").is_file():
            candidate_run_dirs.append(child_path)

    if not candidate_run_dirs:
        raise FileNotFoundError(f"No subdirectories with points_final.ply found under '{output_root_path}'.")

    latest_run_dir = max(candidate_run_dirs, key=lambda run_path: (run_path / "points_final.ply").stat().st_mtime)
    latest_ply_path = latest_run_dir / "points_final.ply"
    print(f"Using latest beta-surfel run directory: {latest_run_dir}")
    print(f"points_final.ply: {latest_ply_path}")
    return latest_ply_path


def load_beta_surfels_from_ply_ascii(
    ply_path: Path,
    opacity_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pk_x: List[float] = []
    pk_y: List[float] = []
    pk_z: List[float] = []

    tu_x: List[float] = []
    tu_y: List[float] = []
    tu_z: List[float] = []

    tv_x: List[float] = []
    tv_y: List[float] = []
    tv_z: List[float] = []

    su_values: List[float] = []
    sv_values: List[float] = []

    color0: List[float] = []
    color1: List[float] = []
    color2: List[float] = []
    opacity_values: List[float] = []

    with ply_path.open("r", encoding="utf-8") as file_handle:
        header_finished = False
        for line in file_handle:
            if not header_finished:
                if line.strip() == "end_header":
                    header_finished = True
                continue

            parts = line.strip().split()
            if not parts or len(parts) < 15:
                continue

            opacity_value = float(parts[14])
            if opacity_value < opacity_threshold:
                continue

            opacity_values.append(opacity_value)

            pk_x.append(float(parts[0]))
            pk_y.append(float(parts[1]))
            pk_z.append(float(parts[2]))

            tu_x.append(float(parts[3]))
            tu_y.append(float(parts[4]))
            tu_z.append(float(parts[5]))

            tv_x.append(float(parts[6]))
            tv_y.append(float(parts[7]))
            tv_z.append(float(parts[8]))

            su_values.append(float(parts[9]))
            sv_values.append(float(parts[10]))

            color0.append(float(parts[11]))
            color1.append(float(parts[12]))
            color2.append(float(parts[13]))

    if len(pk_x) == 0:
        raise RuntimeError(f"No points loaded from '{ply_path}'. Try lowering --opacity-threshold.")

    positions = np.stack([pk_x, pk_y, pk_z], axis=1).astype(np.float32)
    tangent_u = np.stack([tu_x, tu_y, tu_z], axis=1).astype(np.float32)
    tangent_v = np.stack([tv_x, tv_y, tv_z], axis=1).astype(np.float32)
    su = np.asarray(su_values, dtype=np.float32) * 0.5
    sv = np.asarray(sv_values, dtype=np.float32) * 0.5
    colors = np.stack([color0, color1, color2], axis=1).astype(np.float32).clip(0.0, 1.0)
    opacities = np.asarray(opacity_values, dtype=np.float32).clip(0.0, 1.0)

    print(f"Loaded {positions.shape[0]} beta-surfels from {ply_path}")
    return positions, tangent_u, tangent_v, su, sv, colors, opacities


def rotation_matrix_to_quaternion_wxyz(rotation_matrices: np.ndarray) -> np.ndarray:
    r = rotation_matrices
    q = np.zeros((r.shape[0], 4), dtype=np.float32)

    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    positive = trace > 0.0

    t = trace[positive]
    s = np.sqrt(t + 1.0) * 2.0
    q[positive, 0] = 0.25 * s
    q[positive, 1] = (r[positive, 2, 1] - r[positive, 1, 2]) / s
    q[positive, 2] = (r[positive, 0, 2] - r[positive, 2, 0]) / s
    q[positive, 3] = (r[positive, 1, 0] - r[positive, 0, 1]) / s

    neg = ~positive
    if np.any(neg):
        r_neg = r[neg]
        diag = np.stack([r_neg[:, 0, 0], r_neg[:, 1, 1], r_neg[:, 2, 2]], axis=1)
        max_index = np.argmax(diag, axis=1)

        for k in (0, 1, 2):
            mask = max_index == k
            if not np.any(mask):
                continue
            rk = r_neg[mask]

            if k == 0:
                s = np.sqrt(1.0 + rk[:, 0, 0] - rk[:, 1, 1] - rk[:, 2, 2]) * 2.0
                qw = (rk[:, 2, 1] - rk[:, 1, 2]) / s
                qx = 0.25 * s
                qy = (rk[:, 0, 1] + rk[:, 1, 0]) / s
                qz = (rk[:, 0, 2] + rk[:, 2, 0]) / s
            elif k == 1:
                s = np.sqrt(1.0 + rk[:, 1, 1] - rk[:, 0, 0] - rk[:, 2, 2]) * 2.0
                qw = (rk[:, 0, 2] - rk[:, 2, 0]) / s
                qx = (rk[:, 0, 1] + rk[:, 1, 0]) / s
                qy = 0.25 * s
                qz = (rk[:, 1, 2] + rk[:, 2, 1]) / s
            else:
                s = np.sqrt(1.0 + rk[:, 2, 2] - rk[:, 0, 0] - rk[:, 1, 1]) * 2.0
                qw = (rk[:, 1, 0] - rk[:, 0, 1]) / s
                qx = (rk[:, 0, 2] + rk[:, 2, 0]) / s
                qy = (rk[:, 1, 2] + rk[:, 2, 1]) / s
                qz = 0.25 * s

            out_idx = np.where(neg)[0][mask]
            q[out_idx, 0] = qw
            q[out_idx, 1] = qx
            q[out_idx, 2] = qy
            q[out_idx, 3] = qz

    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return q


def build_orientation_quaternions_wxyz(tangent_u: np.ndarray, tangent_v: np.ndarray) -> np.ndarray:
    u = tangent_u.astype(np.float32)
    v = tangent_v.astype(np.float32)

    u_hat = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    v_hat = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    n = np.cross(u_hat, v_hat)
    n_hat = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

    v_ortho = v_hat - (np.sum(v_hat * u_hat, axis=1, keepdims=True) * u_hat)
    v_ortho = v_ortho / (np.linalg.norm(v_ortho, axis=1, keepdims=True) + 1e-12)

    rotation_matrices = np.zeros((u_hat.shape[0], 3, 3), dtype=np.float32)
    rotation_matrices[:, :, 0] = u_hat
    rotation_matrices[:, :, 1] = v_ortho
    rotation_matrices[:, :, 2] = n_hat

    return rotation_matrix_to_quaternion_wxyz(rotation_matrices)


# ======================================================================================
# 2DGS PLY loader (binary_little_endian point_cloud.ply)
# ======================================================================================
def find_latest_2dgs_ply(output_root_path: Path) -> Path:
    if not output_root_path.exists():
        raise FileNotFoundError(f"Path '{output_root_path}' does not exist.")

    if output_root_path.is_file():
        print(f"Using 2DGS PLY file: {output_root_path}")
        return output_root_path

    candidate_plys: List[Path] = []
    for ply_path in output_root_path.rglob("point_cloud.ply"):
        if ply_path.is_file():
            candidate_plys.append(ply_path)

    if not candidate_plys:
        raise FileNotFoundError(f"No 'point_cloud.ply' found under '{output_root_path}'.")

    latest_ply = max(candidate_plys, key=lambda p: p.stat().st_mtime)
    print(f"Using latest 2DGS: {latest_ply}")
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
    if fmt not in ("ascii", "binary_little_endian"):
        raise RuntimeError(f"Unsupported PLY format: '{fmt}'")

    return vertex_count, fmt, header_byte_length


def parse_2dgs_binary_little_endian(
    ply_path: Path,
    opacity_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertex_count, fmt, header_len = read_ply_header_and_get_vertex_count_and_format(ply_path)
    if fmt != "binary_little_endian":
        raise RuntimeError(f"Expected binary_little_endian PLY for 2DGS, got '{fmt}'")

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
    opacity = data[:, 54]
    scale_0 = data[:, 55]
    scale_1 = data[:, 56]
    rot_wxyz = data[:, 57:61]

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

    C0 = 0.28209479177387814
    colors01 = np.clip(f_dc.astype(np.float32) * C0 + 0.7, 0.0, 1.0)

    quats_wxyz = rot_wxyz.astype(np.float32)
    quats_wxyz = quats_wxyz / (np.linalg.norm(quats_wxyz, axis=1, keepdims=True) + 1e-12)

    z_min = 0.23
    opacity_keep = opacities01 >= float(opacity_threshold)
    z_keep = positions[:, 2] >= float(z_min)
    keep = opacity_keep & z_keep

    positions = positions[keep].astype(np.float32)
    colors01 = colors01[keep].astype(np.float32)
    opacities01 = opacities01[keep].astype(np.float32)
    scales_uv = scales_uv[keep].astype(np.float32) * 0.7
    quats_wxyz = quats_wxyz[keep].astype(np.float32)

    if positions.shape[0] == 0:
        raise RuntimeError("No 2DGS points left after opacity filtering. Lower --opacity-threshold.")

    print(
        f"Loaded {positions.shape[0]} / {vertex_count} 2DGS points from {ply_path}\n"
        f"Opacity encoding: {'logit->sigmoid' if opacity_is_logit else 'raw'}\n"
        f"Scale encoding: {'log->exp' if scale_is_log else 'raw'}"
    )
    return positions, colors01, opacities01, scales_uv, quats_wxyz


# ======================================================================================
# Unified point cloud input
# ======================================================================================
@dataclass(frozen=True)
class GlyphCloud:
    positions: np.ndarray     # (N,3)
    quats_wxyz: np.ndarray    # (N,4)
    scales_uv: np.ndarray     # (N,2)  (su, sv)
    colors01: np.ndarray      # (N,3)
    opacities01: np.ndarray   # (N,)


def load_glyph_cloud_from_input(
    point_cloud_input: Path,
    point_cloud_type: str,  # "auto" | "beta_surfel" | "2dgs"
    opacity_threshold: float,
    area_threshold: float,
    max_ellipses: int,
    scale_mult: float,
    alpha_mult: float,
) -> GlyphCloud:
    chosen_type = point_cloud_type.lower().strip()

    ply_path: Path
    if chosen_type == "2dgs":
        ply_path = find_latest_2dgs_ply(point_cloud_input)
        positions, colors01, opacities01, scales_uv, quats_wxyz = parse_2dgs_binary_little_endian(
            ply_path, opacity_threshold=opacity_threshold
        )
    elif chosen_type == "beta_surfel":
        ply_path = find_latest_beta_surfel_points_ply(point_cloud_input)
        positions, tangent_u, tangent_v, su, sv, colors01, opacities01 = load_beta_surfels_from_ply_ascii(
            ply_path, opacity_threshold=opacity_threshold
        )
        quats_wxyz = build_orientation_quaternions_wxyz(tangent_u, tangent_v)
        scales_uv = np.stack([su, sv], axis=1).astype(np.float32)
    elif chosen_type == "auto":
        # Heuristic:
        # - If file is named point_cloud.ply OR directory contains point_cloud.ply -> 2dgs
        # - Else -> beta surfel points_final.ply
        if point_cloud_input.is_file() and point_cloud_input.name == "point_cloud.ply":
            ply_path = point_cloud_input
            positions, colors01, opacities01, scales_uv, quats_wxyz = parse_2dgs_binary_little_endian(
                ply_path, opacity_threshold=opacity_threshold
            )
        else:
            try:
                ply_path = find_latest_2dgs_ply(point_cloud_input)
                positions, colors01, opacities01, scales_uv, quats_wxyz = parse_2dgs_binary_little_endian(
                    ply_path, opacity_threshold=opacity_threshold
                )
            except Exception:
                ply_path = find_latest_beta_surfel_points_ply(point_cloud_input)
                positions, tangent_u, tangent_v, su, sv, colors01, opacities01 = load_beta_surfels_from_ply_ascii(
                    ply_path, opacity_threshold=opacity_threshold
                )
                quats_wxyz = build_orientation_quaternions_wxyz(tangent_u, tangent_v)
                scales_uv = np.stack([su, sv], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported --point-cloud-type '{point_cloud_type}' (expected auto|beta_surfel|2dgs)")

    # Area filtering (on decoded scales)
    ellipse_area = scales_uv[:, 0] * scales_uv[:, 1]
    keep_area = ellipse_area >= float(area_threshold)

    positions = positions[keep_area]
    colors01 = colors01[keep_area]
    opacities01 = opacities01[keep_area]
    scales_uv = scales_uv[keep_area]
    quats_wxyz = quats_wxyz[keep_area]

    # Clamp length
    if max_ellipses and positions.shape[0] > max_ellipses:
        positions = positions[:max_ellipses]
        colors01 = colors01[:max_ellipses]
        opacities01 = opacities01[:max_ellipses]
        scales_uv = scales_uv[:max_ellipses]
        quats_wxyz = quats_wxyz[:max_ellipses]

    # Apply visualization multipliers (do this AFTER filtering)
    scales_uv = scales_uv * float(scale_mult)
    opacities01 = np.clip(opacities01 * float(alpha_mult), 0.0, 1.0)

    if positions.shape[0] == 0:
        raise RuntimeError("No points left after filtering. Lower thresholds or increase --max-ellipses.")

    print(f"Rendering {positions.shape[0]} ellipses from {ply_path}")
    return GlyphCloud(
        positions=positions.astype(np.float32),
        quats_wxyz=quats_wxyz.astype(np.float32),
        scales_uv=scales_uv.astype(np.float32),
        colors01=colors01.astype(np.float32),
        opacities01=opacities01.astype(np.float32),
    )


# ======================================================================================
# Args + main (auto flip-through + save images)
# ======================================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VTK renderer: surfel/2DGS ellipses + Mitsuba XML cameras to PNGs.")

    parser.add_argument("--scene-xml", type=Path, required=True)

    # One input that can be:
    #  - your beta-surfel run dir (contains points_final.ply)
    #  - a direct points_final.ply
    #  - your 2DGS run dir (contains .../point_cloud.ply)
    #  - a direct point_cloud.ply
    parser.add_argument("--point-cloud", type=Path, required=True)

    parser.add_argument("--point-cloud-type", type=str, default="auto", help="auto|beta_surfel|2dgs")

    parser.add_argument("--opacity-threshold", type=float, default=0.0)
    parser.add_argument("--area-threshold", type=float, default=0.0)
    parser.add_argument("--max-ellipses", type=int, default=0)
    parser.add_argument("--disk-resolution", type=int, default=16)

    parser.add_argument("--scale-mult", type=float, default=1.0)
    parser.add_argument("--alpha-mult", type=float, default=1.0)

    parser.add_argument("--images-dir", type=Path, required=True, help="Output directory for per-camera PNGs.")
    parser.add_argument(
        "--camera-id-prefix",
        type=str,
        default="",
        help="Optional filter: only render sensors whose id starts with this prefix.",
    )
    parser.add_argument("--background", type=float, nargs=3, default=(0.2, 0.2, 0.25))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    sensors = parse_mitsuba_scene_sensors(args.scene_xml)
    if args.camera_id_prefix:
        sensors = [s for s in sensors if s.sensor_id.startswith(args.camera_id_prefix)]
        if not sensors:
            raise RuntimeError(f"No sensors matched --camera-id-prefix '{args.camera_id_prefix}'")

    glyph_cloud = load_glyph_cloud_from_input(
        point_cloud_input=args.point_cloud,
        point_cloud_type=args.point_cloud_type,
        opacity_threshold=float(args.opacity_threshold),
        area_threshold=float(args.area_threshold),
        max_ellipses=int(args.max_ellipses),
        scale_mult=float(args.scale_mult),
        alpha_mult=float(args.alpha_mult),
    )

    # Build vtkPolyData
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToFloat()
    vtk_points.SetNumberOfPoints(int(glyph_cloud.positions.shape[0]))
    for i in range(int(glyph_cloud.positions.shape[0])):
        p = glyph_cloud.positions[i]
        vtk_points.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)

    poly.GetPointData().AddArray(numpy_to_vtk_float_array("orientation", glyph_cloud.quats_wxyz, 4))

    scales_3 = np.stack(
        [glyph_cloud.scales_uv[:, 0], glyph_cloud.scales_uv[:, 1], np.ones_like(glyph_cloud.scales_uv[:, 0])],
        axis=1,
    ).astype(np.float32)
    poly.GetPointData().AddArray(numpy_to_vtk_float_array("scale", scales_3, 3))

    poly.GetPointData().AddArray(numpy_rgba_u8("color_rgba", glyph_cloud.colors01, glyph_cloud.opacities01))

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
    renderer.SetBackground(float(args.background[0]), float(args.background[1]), float(args.background[2]))
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetAlphaBitPlanes(True)
    render_window.SetMultiSamples(0)

    # Auto flip-through + save
    args.images_dir.mkdir(parents=True, exist_ok=True)
    render_window.SetOffScreenRendering(1)

    for sensor_index, sensor in enumerate(sensors):
        sensor.fov_degrees = sensor.fov_degrees - 7
        render_window.SetSize(int(sensor.width), int(sensor.height))
        print(sensor)
        apply_sensor_to_vtk_camera(renderer.GetActiveCamera(), sensor)

        renderer.GetActiveCamera().SetExplicitAspectRatio(1.2)
        renderer.GetActiveCamera().SetUseExplicitAspectRatio(True)
        render_window.Render()

        output_path = args.images_dir / f"{sensor_index:03d}_{sensor.sensor_id}.png"
        write_render_to_png(render_window, output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
