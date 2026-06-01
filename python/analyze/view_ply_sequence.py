#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import vtk


def parse_iteration_from_ply_name(ply_path: Path) -> int:
    match = re.search(r"iter_(\d+)_points\.ply$", ply_path.name)
    if match is None:
        return -1

    return int(match.group(1))


def find_latest_optimization_run_dir(
    output_root_path: Path,
    run_index: int = 0,
) -> Path:
    if run_index < 0:
        raise ValueError(f"run_index must be >= 0, got {run_index}.")

    if not output_root_path.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root_path}")

    if not output_root_path.is_dir():
        raise NotADirectoryError(f"Output root is not a directory: {output_root_path}")

    candidate_run_dirs: List[Path] = []

    for child_path in output_root_path.iterdir():
        if not child_path.is_dir():
            continue

        points_dir = child_path / "points"
        if not points_dir.is_dir():
            continue

        if any(points_dir.glob("*.ply")):
            candidate_run_dirs.append(child_path)

    if not candidate_run_dirs:
        raise FileNotFoundError(
            f"No optimization folders with a non-empty ./points folder found under: {output_root_path}"
        )

    sorted_run_dirs = sorted(
        candidate_run_dirs,
        key=lambda run_dir: run_dir.stat().st_mtime,
        reverse=True,
    )

    if run_index >= len(sorted_run_dirs):
        raise IndexError(
            f"Requested run_index={run_index}, but only {len(sorted_run_dirs)} valid run folders were found."
        )

    return sorted_run_dirs[run_index]


def find_points_sequence_dir(
    output_root_path: Path,
    run_dir: Path | None,
    run_index: int,
) -> Path:
    if run_dir is not None:
        resolved_run_dir = run_dir.expanduser().resolve()
        points_dir = resolved_run_dir / "points"

        if not resolved_run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {resolved_run_dir}")

        if not points_dir.is_dir():
            raise FileNotFoundError(f"Run directory has no ./points folder: {points_dir}")

        return points_dir

    resolved_output_root_path = output_root_path.expanduser().resolve()
    latest_run_dir = find_latest_optimization_run_dir(
        output_root_path=resolved_output_root_path,
        run_index=run_index,
    )

    return latest_run_dir / "points"


def find_ply_sequence(points_dir: Path) -> List[Path]:
    ply_paths = list(points_dir.glob("*.ply"))

    if not ply_paths:
        raise FileNotFoundError(f"No .ply files found in points folder: {points_dir}")

    return sorted(
        ply_paths,
        key=lambda path: (
            parse_iteration_from_ply_name(path),
            path.name,
        ),
    )


def numpy_rgb01_and_alpha01_to_vtk_u8_rgba(
    name: str,
    rgb01: np.ndarray,
    alpha01: np.ndarray,
) -> vtk.vtkUnsignedCharArray:
    rgb_u8 = (np.asarray(rgb01, dtype=np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    alpha_u8 = (np.asarray(alpha01, dtype=np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)

    rgba = np.concatenate([rgb_u8, alpha_u8.reshape(-1, 1)], axis=1)

    array_handle = vtk.vtkUnsignedCharArray()
    array_handle.SetName(name)
    array_handle.SetNumberOfComponents(4)
    array_handle.SetNumberOfTuples(rgba.shape[0])

    for point_index in range(rgba.shape[0]):
        array_handle.SetTuple4(
            point_index,
            int(rgba[point_index, 0]),
            int(rgba[point_index, 1]),
            int(rgba[point_index, 2]),
            int(rgba[point_index, 3]),
        )

    return array_handle


def load_surfels_from_ply(
    ply_path: Path,
    opacity_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position_x_values: List[float] = []
    position_y_values: List[float] = []
    position_z_values: List[float] = []

    tangent_u_x_values: List[float] = []
    tangent_u_y_values: List[float] = []
    tangent_u_z_values: List[float] = []

    tangent_v_x_values: List[float] = []
    tangent_v_y_values: List[float] = []
    tangent_v_z_values: List[float] = []

    scale_u_values: List[float] = []
    scale_v_values: List[float] = []

    color_r_values: List[float] = []
    color_g_values: List[float] = []
    color_b_values: List[float] = []

    opacity_values: List[float] = []

    with ply_path.open("r", encoding="utf-8") as file_handle:
        header_finished = False

        for line in file_handle:
            if not header_finished:
                if line.strip() == "end_header":
                    header_finished = True
                continue

            parts = line.strip().split()
            if not parts:
                continue

            # Expected layout:
            # 0..2   position
            # 3..5   tangent_u
            # 6..8   tangent_v
            # 9..10  scale_u, scale_v
            # 11..13 albedo/color
            # 14     opacity
            if len(parts) < 15:
                continue

            opacity_value = float(parts[14])
            if opacity_value < opacity_threshold:
                continue

            position_x_values.append(float(parts[0]))
            position_y_values.append(float(parts[1]))
            position_z_values.append(float(parts[2]))

            tangent_u_x_values.append(float(parts[3]))
            tangent_u_y_values.append(float(parts[4]))
            tangent_u_z_values.append(float(parts[5]))

            tangent_v_x_values.append(float(parts[6]))
            tangent_v_y_values.append(float(parts[7]))
            tangent_v_z_values.append(float(parts[8]))

            scale_u_values.append(float(parts[9]))
            scale_v_values.append(float(parts[10]))

            color_r_values.append(float(parts[11]))
            color_g_values.append(float(parts[12]))
            color_b_values.append(float(parts[13]))

            opacity_values.append(opacity_value)

    if len(position_x_values) == 0:
        raise RuntimeError(f"No points loaded from '{ply_path}'. Try lowering --opacity-threshold.")

    positions = np.stack(
        [position_x_values, position_y_values, position_z_values],
        axis=1,
    ).astype(np.float32)

    tangent_u = np.stack(
        [tangent_u_x_values, tangent_u_y_values, tangent_u_z_values],
        axis=1,
    ).astype(np.float32)

    tangent_v = np.stack(
        [tangent_v_x_values, tangent_v_y_values, tangent_v_z_values],
        axis=1,
    ).astype(np.float32)

    scale_u = np.asarray(scale_u_values, dtype=np.float32)
    scale_v = np.asarray(scale_v_values, dtype=np.float32)

    colors = np.stack(
        [color_r_values, color_g_values, color_b_values],
        axis=1,
    ).astype(np.float32).clip(0.0, 1.0)

    opacities = np.asarray(opacity_values, dtype=np.float32).clip(0.0, 1.0)

    return positions, tangent_u, tangent_v, scale_u, scale_v, colors, opacities


def rotation_matrix_to_quaternion_wxyz(rotation_matrices: np.ndarray) -> np.ndarray:
    rotation = rotation_matrices
    quaternions = np.zeros((rotation.shape[0], 4), dtype=np.float32)

    trace = rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]
    positive_trace_mask = trace > 0.0

    positive_trace = trace[positive_trace_mask]
    positive_scale = np.sqrt(positive_trace + 1.0) * 2.0

    quaternions[positive_trace_mask, 0] = 0.25 * positive_scale
    quaternions[positive_trace_mask, 1] = (
        rotation[positive_trace_mask, 2, 1] - rotation[positive_trace_mask, 1, 2]
    ) / positive_scale
    quaternions[positive_trace_mask, 2] = (
        rotation[positive_trace_mask, 0, 2] - rotation[positive_trace_mask, 2, 0]
    ) / positive_scale
    quaternions[positive_trace_mask, 3] = (
        rotation[positive_trace_mask, 1, 0] - rotation[positive_trace_mask, 0, 1]
    ) / positive_scale

    non_positive_trace_mask = ~positive_trace_mask

    if np.any(non_positive_trace_mask):
        rotation_negative = rotation[non_positive_trace_mask]

        diagonal = np.stack(
            [
                rotation_negative[:, 0, 0],
                rotation_negative[:, 1, 1],
                rotation_negative[:, 2, 2],
            ],
            axis=1,
        )

        max_diagonal_index = np.argmax(diagonal, axis=1)

        for diagonal_index in (0, 1, 2):
            current_mask = max_diagonal_index == diagonal_index
            if not np.any(current_mask):
                continue

            current_rotation = rotation_negative[current_mask]

            if diagonal_index == 0:
                current_scale = np.sqrt(
                    1.0
                    + current_rotation[:, 0, 0]
                    - current_rotation[:, 1, 1]
                    - current_rotation[:, 2, 2]
                ) * 2.0

                quaternion_w = (
                    current_rotation[:, 2, 1] - current_rotation[:, 1, 2]
                ) / current_scale
                quaternion_x = 0.25 * current_scale
                quaternion_y = (
                    current_rotation[:, 0, 1] + current_rotation[:, 1, 0]
                ) / current_scale
                quaternion_z = (
                    current_rotation[:, 0, 2] + current_rotation[:, 2, 0]
                ) / current_scale

            elif diagonal_index == 1:
                current_scale = np.sqrt(
                    1.0
                    + current_rotation[:, 1, 1]
                    - current_rotation[:, 0, 0]
                    - current_rotation[:, 2, 2]
                ) * 2.0

                quaternion_w = (
                    current_rotation[:, 0, 2] - current_rotation[:, 2, 0]
                ) / current_scale
                quaternion_x = (
                    current_rotation[:, 0, 1] + current_rotation[:, 1, 0]
                ) / current_scale
                quaternion_y = 0.25 * current_scale
                quaternion_z = (
                    current_rotation[:, 1, 2] + current_rotation[:, 2, 1]
                ) / current_scale

            else:
                current_scale = np.sqrt(
                    1.0
                    + current_rotation[:, 2, 2]
                    - current_rotation[:, 0, 0]
                    - current_rotation[:, 1, 1]
                ) * 2.0

                quaternion_w = (
                    current_rotation[:, 1, 0] - current_rotation[:, 0, 1]
                ) / current_scale
                quaternion_x = (
                    current_rotation[:, 0, 2] + current_rotation[:, 2, 0]
                ) / current_scale
                quaternion_y = (
                    current_rotation[:, 1, 2] + current_rotation[:, 2, 1]
                ) / current_scale
                quaternion_z = 0.25 * current_scale

            output_indices = np.where(non_positive_trace_mask)[0][current_mask]

            quaternions[output_indices, 0] = quaternion_w
            quaternions[output_indices, 1] = quaternion_x
            quaternions[output_indices, 2] = quaternion_y
            quaternions[output_indices, 3] = quaternion_z

    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True) + 1e-12

    return quaternions


def build_orientation_quaternions_wxyz(
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
) -> np.ndarray:
    tangent_u_float = tangent_u.astype(np.float32)
    tangent_v_float = tangent_v.astype(np.float32)

    unit_tangent_u = tangent_u_float / (
        np.linalg.norm(tangent_u_float, axis=1, keepdims=True) + 1e-12
    )

    tangent_v_normalized = tangent_v_float / (
        np.linalg.norm(tangent_v_float, axis=1, keepdims=True) + 1e-12
    )

    normal = np.cross(unit_tangent_u, tangent_v_normalized)
    unit_normal = normal / (
        np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    )

    tangent_v_orthogonal = tangent_v_normalized - (
        np.sum(tangent_v_normalized * unit_tangent_u, axis=1, keepdims=True)
        * unit_tangent_u
    )

    unit_tangent_v = tangent_v_orthogonal / (
        np.linalg.norm(tangent_v_orthogonal, axis=1, keepdims=True) + 1e-12
    )

    rotation_matrices = np.zeros((unit_tangent_u.shape[0], 3, 3), dtype=np.float32)
    rotation_matrices[:, :, 0] = unit_tangent_u
    rotation_matrices[:, :, 1] = unit_tangent_v
    rotation_matrices[:, :, 2] = unit_normal

    return rotation_matrix_to_quaternion_wxyz(rotation_matrices)


def numpy_to_vtk_float_array(
    name: str,
    data: np.ndarray,
    num_components: int,
) -> vtk.vtkFloatArray:
    flat_data = np.asarray(data, dtype=np.float32).reshape(
        data.shape[0],
        num_components,
    )

    array_handle = vtk.vtkFloatArray()
    array_handle.SetName(name)
    array_handle.SetNumberOfComponents(num_components)
    array_handle.SetNumberOfTuples(flat_data.shape[0])

    for point_index in range(flat_data.shape[0]):
        array_handle.SetTuple(point_index, flat_data[point_index].tolist())

    return array_handle


def build_poly_data_from_ply(
    ply_path: Path,
    opacity_threshold: float,
    area_threshold: float,
    max_ellipses: int,
    scale_multiplier: float,
    alpha_multiplier: float,
    solid: bool,
) -> vtk.vtkPolyData:
    positions, tangent_u, tangent_v, scale_u, scale_v, colors, opacities = load_surfels_from_ply(
        ply_path=ply_path,
        opacity_threshold=opacity_threshold,
    )

    ellipse_area = scale_u * scale_v
    ellipse_mask = ellipse_area >= float(area_threshold)

    positions = positions[ellipse_mask]
    tangent_u = tangent_u[ellipse_mask]
    tangent_v = tangent_v[ellipse_mask]
    scale_u = scale_u[ellipse_mask] * scale_multiplier
    scale_v = scale_v[ellipse_mask] * scale_multiplier
    colors = colors[ellipse_mask]
    opacities = opacities[ellipse_mask]

    if solid:
        opacities = np.ones_like(opacities)
    else:
        opacities = opacities * float(alpha_multiplier)

    if max_ellipses > 0 and positions.shape[0] > max_ellipses:
        positions = positions[:max_ellipses]
        tangent_u = tangent_u[:max_ellipses]
        tangent_v = tangent_v[:max_ellipses]
        scale_u = scale_u[:max_ellipses]
        scale_v = scale_v[:max_ellipses]
        colors = colors[:max_ellipses]
        opacities = opacities[:max_ellipses]

    print(f"Loaded {positions.shape[0]} visible surfels from: {ply_path.name}")

    points = vtk.vtkPoints()
    points.SetDataTypeToFloat()
    points.SetNumberOfPoints(int(positions.shape[0]))

    for point_index in range(int(positions.shape[0])):
        points.SetPoint(
            point_index,
            float(positions[point_index, 0]),
            float(positions[point_index, 1]),
            float(positions[point_index, 2]),
        )

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)

    quaternions = build_orientation_quaternions_wxyz(tangent_u, tangent_v)

    poly_data.GetPointData().AddArray(
        numpy_to_vtk_float_array("orientation", quaternions, 4)
    )

    scale_triples = np.stack(
        [scale_u, scale_v, np.ones_like(scale_u)],
        axis=1,
    ).astype(np.float32)

    poly_data.GetPointData().AddArray(
        numpy_to_vtk_float_array("scale", scale_triples, 3)
    )

    poly_data.GetPointData().AddArray(
        numpy_rgb01_and_alpha01_to_vtk_u8_rgba("color_rgba", colors, opacities)
    )

    poly_data.Modified()

    return poly_data


def make_status_text_actor() -> vtk.vtkTextActor:
    text_actor = vtk.vtkTextActor()
    text_actor.SetDisplayPosition(20, 20)

    text_property = text_actor.GetTextProperty()
    text_property.SetFontSize(18)
    text_property.SetColor(1.0, 1.0, 1.0)
    text_property.SetBackgroundColor(0.0, 0.0, 0.0)
    text_property.SetBackgroundOpacity(0.5)

    return text_actor


def update_status_text(
    text_actor: vtk.vtkTextActor,
    current_index: int,
    ply_paths: List[Path],
    points_dir: Path,
) -> None:
    current_ply_path = ply_paths[current_index]
    iteration = parse_iteration_from_ply_name(current_ply_path)

    if iteration >= 0:
        iteration_text = f"iteration {iteration}"
    else:
        iteration_text = current_ply_path.stem

    text_actor.SetInput(
        f"{current_index + 1}/{len(ply_paths)} | {iteration_text}\n"
        f"{current_ply_path.name}\n"
        f"{points_dir}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "VTK viewer for cycling through optimization point-cloud checkpoints "
            "from the latest run's ./points folder."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../Assets/OptimizationOutput"),
        help="Root folder containing timestamped optimization output folders.",
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit optimization run directory. If omitted, the latest run under --output-root is used.",
    )

    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="0 means latest run, 1 means second latest run, etc. Ignored when --run-dir is provided.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Initial PLY index inside the points folder.",
    )

    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--area-threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-ellipses",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--disk-resolution",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--solid",
        action="store_true",
    )

    parser.add_argument(
        "--window-width",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--window-height",
        type=int,
        default=900,
    )

    parser.add_argument(
        "--reset-camera-on-start",
        action="store_true",
        help="Frame the first point cloud once at startup. Camera is still preserved when cycling.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    points_dir = find_points_sequence_dir(
        output_root_path=args.output_root,
        run_dir=args.run_dir,
        run_index=args.run_index,
    )

    ply_paths = find_ply_sequence(points_dir)

    if args.start_index < 0 or args.start_index >= len(ply_paths):
        raise IndexError(
            f"--start-index must be in [0, {len(ply_paths) - 1}], got {args.start_index}."
        )

    current_index = args.start_index

    print(f"Using points folder: {points_dir}")
    print(f"Found {len(ply_paths)} PLY files.")
    print("Controls: Right/Left arrow = next/previous PLY, Home/End = first/last, r = reload file list, q/Escape = quit")

    poly_data = build_poly_data_from_ply(
        ply_path=ply_paths[current_index],
        opacity_threshold=args.opacity_threshold,
        area_threshold=args.area_threshold,
        max_ellipses=args.max_ellipses,
        scale_multiplier=args.scale,
        alpha_multiplier=args.alpha,
        solid=args.solid,
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
    renderer.SetBackground(0.2, 0.2, 0.25)
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(0.1)

    status_text_actor = make_status_text_actor()
    update_status_text(
        text_actor=status_text_actor,
        current_index=current_index,
        ply_paths=ply_paths,
        points_dir=points_dir,
    )
    renderer.AddActor2D(status_text_actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(int(args.window_width), int(args.window_height))
    render_window.SetAlphaBitPlanes(True)
    render_window.SetMultiSamples(0)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)

    camera = renderer.GetActiveCamera()
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.SetFocalPoint(0.0, 0.0, 0.25)

    if args.reset_camera_on_start:
        renderer.ResetCamera()

    renderer.ResetCameraClippingRange()

    def load_current_index() -> None:
        nonlocal current_index

        current_ply_path = ply_paths[current_index]

        new_poly_data = build_poly_data_from_ply(
            ply_path=current_ply_path,
            opacity_threshold=args.opacity_threshold,
            area_threshold=args.area_threshold,
            max_ellipses=args.max_ellipses,
            scale_multiplier=args.scale,
            alpha_multiplier=args.alpha,
            solid=args.solid,
        )

        mapper.SetInputData(new_poly_data)
        mapper.Modified()

        update_status_text(
            text_actor=status_text_actor,
            current_index=current_index,
            ply_paths=ply_paths,
            points_dir=points_dir,
        )

        render_window.SetWindowName(
            f"{current_index + 1}/{len(ply_paths)} - {current_ply_path.name}"
        )

        renderer.ResetCameraClippingRange()

        # Important:
        # Do not call renderer.ResetCamera() here.
        # This preserves the current camera view while cycling.
        render_window.Render()

        print(f"Showing {current_index + 1}/{len(ply_paths)}: {current_ply_path.name}")

    def reload_ply_file_list() -> None:
        nonlocal ply_paths
        nonlocal current_index

        current_ply_name = ply_paths[current_index].name

        updated_ply_paths = find_ply_sequence(points_dir)
        ply_paths = updated_ply_paths

        matching_indices = [
            index
            for index, ply_path in enumerate(ply_paths)
            if ply_path.name == current_ply_name
        ]

        if matching_indices:
            current_index = matching_indices[0]
        else:
            current_index = min(current_index, len(ply_paths) - 1)

        print(f"Reloaded PLY file list. Found {len(ply_paths)} files.")
        load_current_index()

    def on_key_press(caller, event_name) -> None:
        nonlocal current_index

        key_symbol = interactor.GetKeySym()

        if key_symbol in ("Right", "d", "D"):
            current_index = (current_index + 1) % len(ply_paths)
            load_current_index()
            return

        if key_symbol in ("Left", "a", "A"):
            current_index = (current_index - 1) % len(ply_paths)
            load_current_index()
            return

        if key_symbol == "Home":
            current_index = 0
            load_current_index()
            return

        if key_symbol == "End":
            current_index = len(ply_paths) - 1
            load_current_index()
            return

        if key_symbol in ("r", "R"):
            reload_ply_file_list()
            return

        if key_symbol in ("q", "Q", "Escape"):
            interactor.GetRenderWindow().Finalize()
            interactor.TerminateApp()
            return

    interactor.AddObserver("KeyPressEvent", on_key_press)

    render_window.SetWindowName(
        f"{current_index + 1}/{len(ply_paths)} - {ply_paths[current_index].name}"
    )

    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()