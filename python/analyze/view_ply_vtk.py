#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import vtk


def find_latest_points_ply(
    output_root_path: Path,
    use_initial: bool,
    index: int = 0,
    verbose: bool = True,
) -> Path:
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}.")

    if not output_root_path.exists():
        raise FileNotFoundError(f"Output root '{output_root_path}' does not exist.")

    target_filename = "initial_points.ply" if use_initial else "points_final.ply"

    if output_root_path.is_file():
        if verbose:
            print(f"Using PLY file: {output_root_path}")
        return output_root_path

    points_in_root = output_root_path / target_filename
    if points_in_root.is_file():
        if index != 0:
            raise ValueError(
                f"output_root_path points directly to a run directory containing {target_filename}, "
                f"so only index=0 is valid. Got index={index}."
            )

        if verbose:
            print(f"Using {target_filename} in run directory: {output_root_path}")
        return points_in_root

    candidate_run_dirs: List[Path] = []
    for child_path in output_root_path.iterdir():
        candidate_ply_path = child_path / target_filename
        if child_path.is_dir() and candidate_ply_path.is_file():
            candidate_run_dirs.append(child_path)

    if not candidate_run_dirs:
        raise FileNotFoundError(
            f"No subdirectories with {target_filename} found under '{output_root_path}'."
        )

    sorted_run_dirs = sorted(
        candidate_run_dirs,
        key=lambda run_path: (run_path / target_filename).stat().st_mtime,
        reverse=True,
    )

    if index >= len(sorted_run_dirs):
        raise IndexError(
            f"Requested index={index}, but only {len(sorted_run_dirs)} run directories "
            f"with {target_filename} were found under '{output_root_path}'."
        )

    selected_run_dir = sorted_run_dirs[index]
    selected_ply_path = selected_run_dir / target_filename

    if verbose:
        print(f"Using run index {index}: {selected_run_dir}")
        print(f"{target_filename}: {selected_ply_path}")

    return selected_ply_path


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

            opacity_values.append(opacity_value)

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

    print(f"Loaded {positions.shape[0]} points from {ply_path}")

    return positions, tangent_u, tangent_v, scale_u, scale_v, colors, opacities


def rotation_matrix_to_quaternion_wxyz(rotation_matrices: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices with shape (N, 3, 3) to quaternions in (w, x, y, z).
    """
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

                quaternion_w = (current_rotation[:, 2, 1] - current_rotation[:, 1, 2]) / current_scale
                quaternion_x = 0.25 * current_scale
                quaternion_y = (current_rotation[:, 0, 1] + current_rotation[:, 1, 0]) / current_scale
                quaternion_z = (current_rotation[:, 0, 2] + current_rotation[:, 2, 0]) / current_scale

            elif diagonal_index == 1:
                current_scale = np.sqrt(
                    1.0
                    + current_rotation[:, 1, 1]
                    - current_rotation[:, 0, 0]
                    - current_rotation[:, 2, 2]
                ) * 2.0

                quaternion_w = (current_rotation[:, 0, 2] - current_rotation[:, 2, 0]) / current_scale
                quaternion_x = (current_rotation[:, 0, 1] + current_rotation[:, 1, 0]) / current_scale
                quaternion_y = 0.25 * current_scale
                quaternion_z = (current_rotation[:, 1, 2] + current_rotation[:, 2, 1]) / current_scale

            else:
                current_scale = np.sqrt(
                    1.0
                    + current_rotation[:, 2, 2]
                    - current_rotation[:, 0, 0]
                    - current_rotation[:, 1, 1]
                ) * 2.0

                quaternion_w = (current_rotation[:, 1, 0] - current_rotation[:, 0, 1]) / current_scale
                quaternion_x = (current_rotation[:, 0, 2] + current_rotation[:, 2, 0]) / current_scale
                quaternion_y = (current_rotation[:, 1, 2] + current_rotation[:, 2, 1]) / current_scale
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
    """
    Build a local surfel frame.

    X axis = tangent_u
    Y axis = tangent_v re-orthogonalized against tangent_u
    Z axis = normal = cross(tangent_u, tangent_v)

    The VTK disk source is in the XY plane, so this rotates it into the surfel plane.
    """
    tangent_u_float = tangent_u.astype(np.float32)
    tangent_v_float = tangent_v.astype(np.float32)

    unit_tangent_u = tangent_u_float / (
        np.linalg.norm(tangent_u_float, axis=1, keepdims=True) + 1e-12
    )

    tangent_v_normalized = tangent_v_float / (
        np.linalg.norm(tangent_v_float, axis=1, keepdims=True) + 1e-12
    )

    normal = np.cross(unit_tangent_u, tangent_v_normalized)
    unit_normal = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)

    tangent_v_orthogonal = tangent_v_normalized - (
        np.sum(tangent_v_normalized * unit_tangent_u, axis=1, keepdims=True) * unit_tangent_u
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
    flat_data = np.asarray(data, dtype=np.float32).reshape(data.shape[0], num_components)

    array_handle = vtk.vtkFloatArray()
    array_handle.SetName(name)
    array_handle.SetNumberOfComponents(num_components)
    array_handle.SetNumberOfTuples(flat_data.shape[0])

    for point_index in range(flat_data.shape[0]):
        array_handle.SetTuple(point_index, flat_data[point_index].tolist())

    return array_handle


def build_poly_data_from_ply(
    ply_path: Path,
    args: argparse.Namespace,
) -> vtk.vtkPolyData:
    positions, tangent_u, tangent_v, scale_u, scale_v, colors, opacities = load_surfels_from_ply(
        ply_path=ply_path,
        opacity_threshold=args.opacity_threshold,
    )

    ellipse_area = scale_u * scale_v
    ellipse_mask = ellipse_area >= float(args.area_threshold)

    positions = positions[ellipse_mask]
    tangent_u = tangent_u[ellipse_mask]
    tangent_v = tangent_v[ellipse_mask]
    scale_u = scale_u[ellipse_mask] * args.scale
    scale_v = scale_v[ellipse_mask] * args.scale
    colors = colors[ellipse_mask]
    opacities = opacities[ellipse_mask]

    if args.solid:
        opacities = np.ones_like(opacities)

    if args.max_ellipses and positions.shape[0] > args.max_ellipses:
        positions = positions[: args.max_ellipses]
        tangent_u = tangent_u[: args.max_ellipses]
        tangent_v = tangent_v[: args.max_ellipses]
        scale_u = scale_u[: args.max_ellipses]
        scale_v = scale_v[: args.max_ellipses]
        colors = colors[: args.max_ellipses]
        opacities = opacities[: args.max_ellipses]

    print(f"Rendering {positions.shape[0]} surfels")

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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VTK viewer: render surfels as oriented ellipses and reload when the PLY changes."
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=False,
        default=Path("../Assets/OptimizationOutput"),
    )

    parser.add_argument(
        "--initial",
        action="store_true",
        help="Load initial_points.ply instead of points_final.ply.",
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
        "--index",
        type=int,
        default=0,
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
        "--reload-ms",
        type=int,
        default=1000,
        help="How often to check whether the selected PLY file changed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    ply_path = find_latest_points_ply(
        output_root_path=args.output_root,
        use_initial=args.initial,
        index=args.index,
        verbose=True,
    )

    last_loaded_mtime_ns = ply_path.stat().st_mtime_ns

    poly_data = build_poly_data_from_ply(
        ply_path=ply_path,
        args=args,
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

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 900)
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

    renderer.ResetCameraClippingRange()

    def reload_points_if_changed(caller, event_name) -> None:
        nonlocal ply_path
        nonlocal last_loaded_mtime_ns

        try:
            current_ply_path = find_latest_points_ply(
                output_root_path=args.output_root,
                use_initial=args.initial,
                index=args.index,
                verbose=False,
            )

            current_mtime_ns = current_ply_path.stat().st_mtime_ns

            if current_ply_path == ply_path and current_mtime_ns == last_loaded_mtime_ns:
                return

            new_poly_data = build_poly_data_from_ply(
                ply_path=current_ply_path,
                args=args,
            )

            mapper.SetInputData(new_poly_data)
            mapper.Modified()

            ply_path = current_ply_path
            last_loaded_mtime_ns = current_mtime_ns

            renderer.ResetCameraClippingRange()

            # Important:
            # Do not call renderer.ResetCamera() here.
            # That would change the user's current camera view.
            render_window.Render()

            print(f"Reloaded point cloud: {current_ply_path}")

        except Exception as exception:
            # This can happen if the optimizer is currently writing the PLY.
            # The next timer tick will try again.
            print(f"Could not reload point cloud yet: {exception}")

    interactor.AddObserver("TimerEvent", reload_points_if_changed)
    interactor.CreateRepeatingTimer(int(args.reload_ms))

    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
