#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import vtk


def find_latest_points_ply(output_root_path: Path) -> Path:
    if not output_root_path.exists():
        raise FileNotFoundError(f"Output root '{output_root_path}' does not exist.")

    if output_root_path.is_file():
        return output_root_path

    points_in_root = output_root_path / "points_final.ply"
    if points_in_root.is_file():
        return points_in_root

    candidate_run_dirs: List[Path] = []
    for child_path in output_root_path.iterdir():
        if child_path.is_dir() and (child_path / "points_final.ply").is_file():
            candidate_run_dirs.append(child_path)

    if not candidate_run_dirs:
        raise FileNotFoundError(f"No subdirectories with points_final.ply found under '{output_root_path}'.")

    latest_run_dir = max(candidate_run_dirs, key=lambda run_path: (run_path / "points_final.ply").stat().st_mtime)
    return latest_run_dir / "points_final.ply"


def numpy_rgb01_and_alpha01_to_vtk_u8_rgba(name: str, rgb01: np.ndarray, alpha01: np.ndarray) -> vtk.vtkUnsignedCharArray:
    rgb_u8 = (np.asarray(rgb01, dtype=np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    a_u8 = (np.asarray(alpha01, dtype=np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.concatenate([rgb_u8, a_u8.reshape(-1, 1)], axis=1)

    array_handle = vtk.vtkUnsignedCharArray()
    array_handle.SetName(name)
    array_handle.SetNumberOfComponents(4)
    array_handle.SetNumberOfTuples(rgba.shape[0])
    for i in range(rgba.shape[0]):
        array_handle.SetTuple4(i, int(rgba[i, 0]), int(rgba[i, 1]), int(rgba[i, 2]), int(rgba[i, 3]))
    return array_handle


def load_surfels_from_ply(
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
            if len(parts) < 15:
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
    su = np.asarray(su_values, dtype=np.float32)
    sv = np.asarray(sv_values, dtype=np.float32)
    colors = np.stack([color0, color1, color2], axis=1).astype(np.float32).clip(0.0, 1.0)
    opacities = np.asarray(opacity_values, dtype=np.float32).clip(0.0, 1.0)
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


def numpy_to_vtk_float_array(name: str, data: np.ndarray, num_components: int) -> vtk.vtkFloatArray:
    flat = np.asarray(data, dtype=np.float32).reshape(data.shape[0], num_components)
    array_handle = vtk.vtkFloatArray()
    array_handle.SetName(name)
    array_handle.SetNumberOfComponents(num_components)
    array_handle.SetNumberOfTuples(flat.shape[0])
    for i in range(flat.shape[0]):
        array_handle.SetTuple(i, flat[i].tolist())
    return array_handle


# -----------------------------------------------------------------------------
# XML parsing + VTK camera application
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorLookAt:
    sensor_id: str
    fov_axis: str
    fov_degrees: float
    near_clip: float
    far_clip: float
    film_width: Optional[int]
    film_height: Optional[int]
    origin: np.ndarray
    target: np.ndarray
    up: np.ndarray


def _parse_vec3_csv(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated floats, got: '{text}'")
    return np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def parse_sensors_from_scene_xml(scene_xml_path: Path) -> Dict[str, SensorLookAt]:
    tree = ET.parse(scene_xml_path)
    root = tree.getroot()

    sensors: Dict[str, SensorLookAt] = {}
    for sensor in root.findall("sensor"):
        if sensor.attrib.get("type") != "perspective":
            continue

        sensor_id = sensor.attrib.get("id", "")
        fov_axis = "y"
        fov_degrees: Optional[float] = None
        near_clip = 0.01
        far_clip = 100.0
        film_width: Optional[int] = None
        film_height: Optional[int] = None
        origin: Optional[np.ndarray] = None
        target: Optional[np.ndarray] = None
        up: Optional[np.ndarray] = None

        for child in list(sensor):
            if child.tag == "string" and child.attrib.get("name") == "fov_axis":
                fov_axis = str(child.attrib.get("value", "y")).strip().lower()
            elif child.tag == "float" and child.attrib.get("name") == "fov":
                fov_degrees = float(child.attrib["value"])
            elif child.tag == "float" and child.attrib.get("name") == "near_clip":
                near_clip = float(child.attrib["value"])
            elif child.tag == "float" and child.attrib.get("name") == "far_clip":
                far_clip = float(child.attrib["value"])
            elif child.tag == "film":
                for film_child in list(child):
                    if film_child.tag == "integer" and film_child.attrib.get("name") == "width":
                        film_width = int(film_child.attrib["value"])
                    if film_child.tag == "integer" and film_child.attrib.get("name") == "height":
                        film_height = int(film_child.attrib["value"])
            elif child.tag == "transform" and child.attrib.get("name") == "to_world":
                lookat = child.find("lookat")
                if lookat is not None:
                    origin = _parse_vec3_csv(lookat.attrib["origin"])
                    target = _parse_vec3_csv(lookat.attrib["target"])
                    up = _parse_vec3_csv(lookat.attrib["up"])

        if fov_degrees is None:
            raise ValueError(f"Sensor '{sensor_id}' missing <float name='fov' .../>")
        if origin is None or target is None or up is None:
            raise ValueError(f"Sensor '{sensor_id}' missing <transform name='to_world'><lookat .../></transform>")

        sensors[sensor_id] = SensorLookAt(
            sensor_id=sensor_id,
            fov_axis=fov_axis,
            fov_degrees=float(fov_degrees),
            near_clip=float(near_clip),
            far_clip=float(far_clip),
            film_width=film_width,
            film_height=film_height,
            origin=origin,
            target=target,
            up=up,
        )

    if not sensors:
        raise ValueError(f"No <sensor type='perspective' ...> found in '{scene_xml_path}'")

    return sensors


def _horizontal_fov_to_vertical_fov_degrees(horizontal_fov_degrees: float, aspect_width_over_height: float) -> float:
    h = math.radians(horizontal_fov_degrees)
    v = 2.0 * math.atan(math.tan(h / 2.0) / max(aspect_width_over_height, 1e-12))
    return math.degrees(v)


def apply_sensor_to_vtk_camera(sensor: SensorLookAt, renderer: vtk.vtkRenderer, render_window: vtk.vtkRenderWindow) -> None:
    camera = renderer.GetActiveCamera()
    camera.SetPosition(float(sensor.origin[0]), float(sensor.origin[1]), float(sensor.origin[2]))
    camera.SetFocalPoint(float(sensor.target[0]), float(sensor.target[1]), float(sensor.target[2]))
    camera.SetViewUp(float(sensor.up[0]), float(sensor.up[1]), float(sensor.up[2]))
    camera.OrthogonalizeViewUp()

    if sensor.fov_axis == "y":
        vertical_fov_degrees = sensor.fov_degrees
    elif sensor.fov_axis == "x":
        if sensor.film_width is None or sensor.film_height is None:
            raise ValueError("fov_axis='x' requires film width/height to convert to vertical FOV for VTK.")
        aspect = float(sensor.film_width) / float(sensor.film_height)
        vertical_fov_degrees = _horizontal_fov_to_vertical_fov_degrees(sensor.fov_degrees, aspect)
    else:
        raise ValueError(f"Unsupported fov_axis='{sensor.fov_axis}' (expected 'x' or 'y').")

    camera.SetViewAngle(float(vertical_fov_degrees) - 7)
    camera.SetClippingRange(float(sensor.near_clip), float(sensor.far_clip))

    if sensor.film_width is not None and sensor.film_height is not None:
        render_window.SetSize(int(sensor.film_width), int(sensor.film_height))


# -----------------------------------------------------------------------------
# Rendering to file
# -----------------------------------------------------------------------------
def save_render_window_to_png_rgba_opaque(render_window: vtk.vtkRenderWindow, output_png_path: Path) -> None:
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetInputBufferTypeToRGBA()
    window_to_image.ReadFrontBufferOff()
    window_to_image.Update()

    image = window_to_image.GetOutput()

    # Force alpha channel to 255 for all pixels
    width, height, _ = image.GetDimensions()
    rgba = vtk.vtkUnsignedCharArray()
    rgba.DeepCopy(image.GetPointData().GetScalars())  # N*4 u8

    num_pixels = width * height
    for pixel_index in range(num_pixels):
        rgba.SetComponent(pixel_index, 3, 255)

    image.GetPointData().SetScalars(rgba)

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(output_png_path))
    writer.SetInputData(image)
    writer.Write()


# -----------------------------------------------------------------------------
# CLI + main
# -----------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VTK viewer: surfel glyphs + Mitsuba camera XML + per-camera screenshot export.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--camera-id", type=str, default="", help="If set: use only this sensor id. Empty => iterate all sensors.")
    parser.add_argument("--export-dir", type=Path, default=Path("vtk_renders"), help="Where to save PNG renders.")
    parser.add_argument("--no-interactive", action="store_true", help="Export renders and exit (no interactor).")

    parser.add_argument("--opacity-threshold", type=float, default=0.0)
    parser.add_argument("--area-threshold", type=float, default=0.0)
    parser.add_argument("--max-ellipses", type=int, default=0)
    parser.add_argument("--disk-resolution", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--background", type=float, nargs=3, default=(0.08, 0.08, 0.10))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    ply_path = find_latest_points_ply(args.output_root)

    positions, tangent_u, tangent_v, su, sv, colors, opacities = load_surfels_from_ply(
        ply_path,
        opacity_threshold=args.opacity_threshold,
    )

    ellipse_area = su * sv
    ellipse_mask = ellipse_area >= float(args.area_threshold)

    positions = positions[ellipse_mask]
    tangent_u = tangent_u[ellipse_mask]
    tangent_v = tangent_v[ellipse_mask]
    su = su[ellipse_mask] * float(args.scale)
    sv = sv[ellipse_mask] * float(args.scale)
    colors = colors[ellipse_mask]
    opacities = opacities[ellipse_mask]

    if args.max_ellipses and positions.shape[0] > args.max_ellipses:
        positions = positions[: args.max_ellipses]
        tangent_u = tangent_u[: args.max_ellipses]
        tangent_v = tangent_v[: args.max_ellipses]
        su = su[: args.max_ellipses]
        sv = sv[: args.max_ellipses]
        colors = colors[: args.max_ellipses]
        opacities = opacities[: args.max_ellipses]

    # PolyData
    points = vtk.vtkPoints()
    points.SetDataTypeToFloat()
    points.SetNumberOfPoints(int(positions.shape[0]))
    for i in range(int(positions.shape[0])):
        points.SetPoint(i, float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2]))

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)

    quaternions = build_orientation_quaternions_wxyz(tangent_u, tangent_v)
    poly_data.GetPointData().AddArray(numpy_to_vtk_float_array("orientation", quaternions, 4))

    scale_triples = np.stack([su, sv, np.ones_like(su)], axis=1).astype(np.float32)
    poly_data.GetPointData().AddArray(numpy_to_vtk_float_array("scale", scale_triples, 3))

    poly_data.GetPointData().AddArray(
        numpy_rgb01_and_alpha01_to_vtk_u8_rgba("color_rgba", colors, opacities)
    )

    disk = vtk.vtkDiskSource()
    disk.SetInnerRadius(0.0)
    disk.SetOuterRadius(1.0)
    disk.SetRadialResolution(1)
    disk.SetCircumferentialResolution(int(args.disk_resolution))
    disk.Update()

    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(poly_data)
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
    render_window.SetSize(1200, 900)
    render_window.SetAlphaBitPlanes(True)
    render_window.SetMultiSamples(0)

    sensors = parse_sensors_from_scene_xml(args.scene_xml)

    if args.camera_id:
        if args.camera_id not in sensors:
            available = ", ".join(sorted(sensors.keys()))
            raise KeyError(f"camera-id '{args.camera_id}' not found. Available: {available}")
        sensor_ids = [args.camera_id]
    else:
        sensor_ids = sorted(sensors.keys())

    # Render + save per camera
    for sensor_id in sensor_ids:
        sensor = sensors[sensor_id]
        apply_sensor_to_vtk_camera(sensor, renderer, render_window)
        render_window.Render()

        output_png_path = args.export_dir / f"{sensor_id}.png"
        save_render_window_to_png_rgba_opaque(render_window, output_png_path)
        print(f"Saved {output_png_path}")

    if args.no_interactive:
        return

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)

    # Leave the last camera active for interactive viewing
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
