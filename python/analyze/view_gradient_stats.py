#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import vtk


COLORMAP_NAMES = [
    "BLUE -> GREEN -> RED",
    "black_red_yellow_white",
    "grayscale",
    "viridis",
    "magma",
    "turbo",
]


def contains_gradient_plys(path: Path) -> bool:
    return path.is_dir() and any(path.glob("gradient_*.ply"))


def gradient_dir_sort_key(path: Path) -> float:
    ply_paths = list(path.glob("gradient_*.ply"))
    if ply_paths:
        return max(ply_path.stat().st_mtime for ply_path in ply_paths)
    return path.stat().st_mtime


def find_latest_iteration_gradient_stats_dir(run_output_dir: Path) -> Path:
    run_output_dir = run_output_dir.expanduser().resolve()

    if not run_output_dir.exists():
        raise FileNotFoundError(f"Run output dir does not exist: {run_output_dir}")
    if not run_output_dir.is_dir():
        raise NotADirectoryError(f"Run output dir is not a directory: {run_output_dir}")

    # Case 1: user directly passed an iteration folder containing gradient_*.ply.
    if contains_gradient_plys(run_output_dir):
        return run_output_dir

    gradient_stats_root = run_output_dir / "gradient_stats"
    if not gradient_stats_root.exists():
        raise FileNotFoundError(
            f"Run output dir does not contain a gradient_stats folder: {run_output_dir}"
        )
    if not gradient_stats_root.is_dir():
        raise NotADirectoryError(f"gradient_stats path is not a directory: {gradient_stats_root}")

    # Preferred layout:
    #   run_output_dir/gradient_stats/iter_000450/gradient_*.ply
    candidate_dirs: List[Path] = [
        child_path
        for child_path in gradient_stats_root.iterdir()
        if contains_gradient_plys(child_path)
    ]

    if candidate_dirs:
        candidate_dirs = sorted(candidate_dirs, key=gradient_dir_sort_key, reverse=True)
        return candidate_dirs[0]

    # Fallback supported layout:
    #   run_output_dir/gradient_stats/gradient_*.ply
    if contains_gradient_plys(gradient_stats_root):
        return gradient_stats_root

    raise FileNotFoundError(
        f"No gradient-stat folders containing gradient_*.ply found under: {gradient_stats_root}"
    )


def infer_output_dir_from_gradient_stats_dir(gradient_stats_dir: Path) -> Path:
    gradient_stats_dir = gradient_stats_dir.expanduser().resolve()

    if gradient_stats_dir.parent.name == "gradient_stats":
        return gradient_stats_dir.parent.parent

    if gradient_stats_dir.name == "gradient_stats":
        return gradient_stats_dir.parent

    return gradient_stats_dir


def find_latest_output_dir(output_root: Path, run_index: int = 0) -> Path:
    output_root = output_root.expanduser().resolve()

    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")
    if not output_root.is_dir():
        raise NotADirectoryError(f"Output root is not a directory: {output_root}")

    candidate_pairs: list[tuple[Path, Path]] = []

    for child_path in output_root.iterdir():
        if not child_path.is_dir():
            continue

        try:
            latest_gradient_stats_dir = find_latest_iteration_gradient_stats_dir(child_path)
        except (FileNotFoundError, NotADirectoryError):
            continue

        candidate_pairs.append((child_path, latest_gradient_stats_dir))

    if candidate_pairs:
        candidate_pairs = sorted(
            candidate_pairs,
            key=lambda pair: gradient_dir_sort_key(pair[1]),
            reverse=True,
        )

        if run_index < 0 or run_index >= len(candidate_pairs):
            raise IndexError(
                f"run_index must be in [0, {len(candidate_pairs) - 1}], got {run_index}"
            )

        return candidate_pairs[run_index][0]

    # Support passing a single run dir as --output-root.
    try:
        find_latest_iteration_gradient_stats_dir(output_root)
        return output_root
    except (FileNotFoundError, NotADirectoryError):
        pass

    raise FileNotFoundError(
        f"No output run folders with gradient diagnostics found under: {output_root}"
    )


def resolve_gradient_stats_dir(
        output_root: Path,
        run_dir: Path | None,
        run_index: int = 0,
) -> tuple[Path, Path]:
    if run_dir is not None:
        selected_path = run_dir.expanduser().resolve()

        if not selected_path.is_dir():
            raise NotADirectoryError(f"--run-dir is not a directory: {selected_path}")

        selected_gradient_stats_dir = find_latest_iteration_gradient_stats_dir(selected_path)
        selected_output_dir = infer_output_dir_from_gradient_stats_dir(selected_gradient_stats_dir)

        return selected_output_dir, selected_gradient_stats_dir

    selected_output_dir = find_latest_output_dir(output_root, run_index=run_index)
    selected_gradient_stats_dir = find_latest_iteration_gradient_stats_dir(selected_output_dir)

    return selected_output_dir, selected_gradient_stats_dir


def find_gradient_plys(gradient_stats_dir: Path) -> List[Path]:
    preferred_order = [
        "gradient_position_std.ply",
        "gradient_geometric_pressure.ply",
        "gradient_position_norm.ply",
        "gradient_active_camera_count.ply",
    ]

    existing_by_name = {path.name: path for path in gradient_stats_dir.glob("*.ply")}
    ordered_paths = [existing_by_name[name] for name in preferred_order if name in existing_by_name]
    remaining_paths = sorted(
        [path for path in gradient_stats_dir.glob("*.ply") if path.name not in preferred_order],
        key=lambda path: path.name,
    )

    ply_paths = ordered_paths + remaining_paths
    if not ply_paths:
        raise FileNotFoundError(f"No .ply files found in: {gradient_stats_dir}")

    return ply_paths


def robust_normalize(values: np.ndarray, lower_percentile: float, upper_percentile: float) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return np.zeros_like(values, dtype=np.float32)

    value_min = float(np.percentile(finite_values, lower_percentile))
    value_max = float(np.percentile(finite_values, upper_percentile))

    if value_max <= value_min + 1.0e-12:
        return np.zeros_like(values, dtype=np.float32)

    return np.clip((values - value_min) / (value_max - value_min), 0.0, 1.0).astype(np.float32)


def interpolate_colormap(
        normalized_values: np.ndarray,
        control_points: list[tuple[float, tuple[float, float, float]]],
) -> np.ndarray:
    t = np.clip(np.asarray(normalized_values, dtype=np.float32).reshape(-1), 0.0, 1.0)

    control_t = np.asarray([point[0] for point in control_points], dtype=np.float32)
    control_rgb = np.asarray([point[1] for point in control_points], dtype=np.float32)

    red = np.interp(t, control_t, control_rgb[:, 0])
    green = np.interp(t, control_t, control_rgb[:, 1])
    blue = np.interp(t, control_t, control_rgb[:, 2])

    return np.stack([red, green, blue], axis=1)


def apply_colormap(
        scalar_values: np.ndarray,
        file_colors_u8: np.ndarray,
        colormap_name: str,
        lower_percentile: float,
        upper_percentile: float,
) -> np.ndarray:
    if colormap_name == "BLUE -> GREEN -> RED":
        return np.asarray(file_colors_u8, dtype=np.uint8)

    normalized_values = robust_normalize(
        scalar_values,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    if colormap_name == "grayscale":
        rgb = np.repeat(normalized_values.reshape(-1, 1), 3, axis=1)

    elif colormap_name == "black_red_yellow_white":
        rgb = interpolate_colormap(
            normalized_values,
            [
                (0.00, (0.00, 0.00, 0.00)),
                (0.35, (0.70, 0.00, 0.00)),
                (0.70, (1.00, 0.75, 0.00)),
                (1.00, (1.00, 1.00, 1.00)),
            ],
        )

    elif colormap_name == "viridis":
        rgb = interpolate_colormap(
            normalized_values,
            [
                (0.00, (0.267, 0.005, 0.329)),
                (0.25, (0.230, 0.322, 0.546)),
                (0.50, (0.128, 0.567, 0.551)),
                (0.75, (0.369, 0.789, 0.383)),
                (1.00, (0.993, 0.906, 0.144)),
            ],
        )

    elif colormap_name == "magma":
        rgb = interpolate_colormap(
            normalized_values,
            [
                (0.00, (0.001, 0.000, 0.014)),
                (0.25, (0.316, 0.071, 0.486)),
                (0.50, (0.716, 0.215, 0.475)),
                (0.75, (0.986, 0.535, 0.382)),
                (1.00, (0.987, 0.991, 0.750)),
            ],
        )

    elif colormap_name == "turbo":
        rgb = interpolate_colormap(
            normalized_values,
            [
                (0.00, (0.190, 0.072, 0.232)),
                (0.20, (0.115, 0.383, 0.842)),
                (0.40, (0.176, 0.737, 0.586)),
                (0.60, (0.777, 0.853, 0.207)),
                (0.80, (0.985, 0.496, 0.128)),
                (1.00, (0.480, 0.016, 0.011)),
            ],
        )

    else:
        raise ValueError(f"Unknown colormap: {colormap_name}")

    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def vtk_unsigned_char_rgb_array(name: str, colors_u8: np.ndarray) -> vtk.vtkUnsignedCharArray:
    colors_u8 = np.asarray(colors_u8, dtype=np.uint8).reshape(-1, 3)

    color_array = vtk.vtkUnsignedCharArray()
    color_array.SetName(name)
    color_array.SetNumberOfComponents(3)
    color_array.SetNumberOfTuples(colors_u8.shape[0])

    for point_index in range(colors_u8.shape[0]):
        color_array.SetTuple3(
            point_index,
            int(colors_u8[point_index, 0]),
            int(colors_u8[point_index, 1]),
            int(colors_u8[point_index, 2]),
        )

    return color_array


def vtk_float_scalar_array(name: str, values: np.ndarray) -> vtk.vtkFloatArray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    scalar_array = vtk.vtkFloatArray()
    scalar_array.SetName(name)
    scalar_array.SetNumberOfComponents(1)
    scalar_array.SetNumberOfTuples(values.shape[0])

    for point_index in range(values.shape[0]):
        scalar_array.SetValue(point_index, float(values[point_index]))

    return scalar_array


def parse_ascii_ply_vertices(ply_path: Path) -> tuple[list[str], np.ndarray]:
    vertex_count = None
    property_names: list[str] = []
    inside_vertex_element = False

    with ply_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped = line.strip()

            if stripped.startswith("element "):
                parts = stripped.split()
                inside_vertex_element = len(parts) >= 3 and parts[1] == "vertex"
                if inside_vertex_element:
                    vertex_count = int(parts[2])
                continue

            if inside_vertex_element and stripped.startswith("property "):
                parts = stripped.split()
                property_names.append(parts[-1])
                continue

            if stripped == "end_header":
                if vertex_count is None:
                    raise RuntimeError(f"No vertex count found in PLY header: {ply_path}")

                vertex_rows = []
                for _ in range(vertex_count):
                    vertex_line = file_handle.readline()
                    if not vertex_line:
                        break
                    vertex_rows.append([float(value) for value in vertex_line.strip().split()])

                if len(vertex_rows) != vertex_count:
                    raise RuntimeError(f"Expected {vertex_count} vertices, read {len(vertex_rows)} from {ply_path}")

                return property_names, np.asarray(vertex_rows, dtype=np.float32)

    raise RuntimeError(f"No end_header found in PLY: {ply_path}")


def infer_scalar_property_name(
        ply_path: Path,
        property_names: list[str],
        explicit_scalar_name: str | None,
) -> str:
    if explicit_scalar_name is not None:
        if explicit_scalar_name not in property_names:
            raise RuntimeError(
                f"Scalar property '{explicit_scalar_name}' not found in {ply_path.name}. "
                f"Available: {property_names}"
            )
        return explicit_scalar_name

    if ply_path.stem in property_names:
        return ply_path.stem

    known_properties = {
        "x", "y", "z",
        "nx", "ny", "nz",
        "red", "green", "blue",
        "alpha", "opacity",
        "rot_w", "rot_x", "rot_y", "rot_z",
        "su", "sv",
        "albedo_r", "albedo_g", "albedo_b",
        "beta", "shape", "power",
    }

    candidate_names = [name for name in property_names if name not in known_properties]
    if candidate_names:
        return candidate_names[-1]

    raise RuntimeError(f"Could not infer scalar property in {ply_path.name}. Available: {property_names}")


def read_ply_as_points(
        ply_path: Path,
        colormap_name: str,
        scalar_name: str | None,
        lower_percentile: float,
        upper_percentile: float,
) -> vtk.vtkPolyData:
    property_names, vertex_data = parse_ascii_ply_vertices(ply_path)
    property_index = {name: index for index, name in enumerate(property_names)}

    for required_name in ("x", "y", "z"):
        if required_name not in property_index:
            raise RuntimeError(f"PLY is missing property '{required_name}': {ply_path}")

    positions = np.stack(
        [
            vertex_data[:, property_index["x"]],
            vertex_data[:, property_index["y"]],
            vertex_data[:, property_index["z"]],
        ],
        axis=1,
    ).astype(np.float32)

    if all(name in property_index for name in ("red", "green", "blue")):
        file_colors_u8 = np.stack(
            [
                vertex_data[:, property_index["red"]],
                vertex_data[:, property_index["green"]],
                vertex_data[:, property_index["blue"]],
            ],
            axis=1,
        ).clip(0.0, 255.0).astype(np.uint8)
    else:
        file_colors_u8 = np.full((positions.shape[0], 3), 255, dtype=np.uint8)

    scalar_property_name = infer_scalar_property_name(
        ply_path=ply_path,
        property_names=property_names,
        explicit_scalar_name=scalar_name,
    )
    scalar_values = vertex_data[:, property_index[scalar_property_name]].astype(np.float32)

    display_colors_u8 = apply_colormap(
        scalar_values=scalar_values,
        file_colors_u8=file_colors_u8,
        colormap_name=colormap_name,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToFloat()
    vtk_points.SetNumberOfPoints(int(positions.shape[0]))

    for point_index in range(positions.shape[0]):
        vtk_points.SetPoint(
            point_index,
            float(positions[point_index, 0]),
            float(positions[point_index, 1]),
            float(positions[point_index, 2]),
        )

    raw_poly_data = vtk.vtkPolyData()
    raw_poly_data.SetPoints(vtk_points)
    raw_poly_data.GetPointData().AddArray(vtk_float_scalar_array(scalar_property_name, scalar_values))
    raw_poly_data.GetPointData().SetScalars(vtk_unsigned_char_rgb_array("display_colors", display_colors_u8))

    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(raw_poly_data)
    vertex_filter.Update()

    poly_data = vtk.vtkPolyData()
    poly_data.DeepCopy(vertex_filter.GetOutput())
    poly_data.GetPointData().SetScalars(poly_data.GetPointData().GetArray("display_colors"))

    return poly_data


def make_status_text_actor() -> vtk.vtkTextActor:
    text_actor = vtk.vtkTextActor()
    text_actor.SetDisplayPosition(20, 20)

    text_property = text_actor.GetTextProperty()
    text_property.SetFontSize(18)
    text_property.SetColor(1.0, 1.0, 1.0)
    text_property.SetBackgroundColor(0.0, 0.0, 0.0)
    text_property.SetBackgroundOpacity(0.55)

    return text_actor


def update_status_text(
        text_actor: vtk.vtkTextActor,
        current_index: int,
        ply_paths: List[Path],
        selected_output_dir: Path,
        gradient_stats_dir: Path,
        point_size: float,
        colormap_name: str,
) -> None:
    current_path = ply_paths[current_index]

    text_actor.SetInput(
        f"{current_index + 1}/{len(ply_paths)} | point size {point_size:.1f} | colormap: {colormap_name}\n"
        f"{current_path.name}\n"
        f"Space = cycle colormap | Left/Right = cycle PLY\n"
        f"run: {selected_output_dir.name}\n"
        f"stats: {gradient_stats_dir.name}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple VTK viewer for cycling through gradient-stat colored PLY files."
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../Assets/OptimizationOutput"),
        help=(
            "Root folder containing optimization run folders. "
            "The latest run with gradient_stats/iter_*/gradient_*.ply is selected."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Explicit run folder, gradient_stats folder, or exact iter_* folder. "
            "If this is a run folder, the latest gradient_stats/iter_* folder is selected."
        ),
    )
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=15.0)
    parser.add_argument("--point-size-step", type=float, default=1.0)
    parser.add_argument("--window-width", type=int, default=1200)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument("--reset-camera-on-load", action="store_true")
    parser.add_argument("--navigation-repeat-seconds", type=float, default=0.15)
    parser.add_argument("--colormap", type=str, default="BLUE -> GREEN -> RED", choices=COLORMAP_NAMES)
    parser.add_argument("--scalar-name", type=str, default=None)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    selected_output_dir, gradient_stats_dir = resolve_gradient_stats_dir(
        output_root=args.output_root,
        run_dir=args.run_dir,
        run_index=args.run_index,
    )

    ply_paths = find_gradient_plys(gradient_stats_dir)

    if args.start_index < 0 or args.start_index >= len(ply_paths):
        raise IndexError(f"--start-index must be in [0, {len(ply_paths) - 1}], got {args.start_index}")

    current_index = int(args.start_index)
    point_size = float(args.point_size)
    colormap_index = COLORMAP_NAMES.index(args.colormap)

    print(f"Using output folder       : {selected_output_dir}")
    print(f"Using gradient-stat folder: {gradient_stats_dir}")
    print(f"Found {len(ply_paths)} PLY files:")
    for index, path in enumerate(ply_paths):
        print(f"  {index}: {path.name}")
    print(
        "Controls: Left/Right = previous/next PLY, Space = cycle colormap, "
        "+/- = point size, R = reload/latest, Q/Escape = quit"
    )

    poly_data = read_ply_as_points(
        ply_path=ply_paths[current_index],
        colormap_name=COLORMAP_NAMES[colormap_index],
        scalar_name=args.scalar_name,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
    )

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.SetScalarModeToUsePointData()
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(point_size)
    actor.GetProperty().SetRenderPointsAsSpheres(True)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.15, 0.15, 0.18)
    renderer.AddActor(actor)

    status_text_actor = make_status_text_actor()
    update_status_text(
        text_actor=status_text_actor,
        current_index=current_index,
        ply_paths=ply_paths,
        selected_output_dir=selected_output_dir,
        gradient_stats_dir=gradient_stats_dir,
        point_size=point_size,
        colormap_name=COLORMAP_NAMES[colormap_index],
    )
    renderer.AddActor2D(status_text_actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(int(args.window_width), int(args.window_height))

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)

    queued_action: str | None = None
    held_navigation_action: str | None = None
    pressed_keys: set[str] = set()
    last_navigation_action_time = 0.0
    navigation_repeat_seconds = max(float(args.navigation_repeat_seconds), 0.01)

    def reload_file_list_preserving_current() -> None:
        nonlocal selected_output_dir
        nonlocal gradient_stats_dir
        nonlocal ply_paths
        nonlocal current_index

        current_name = ply_paths[current_index].name if ply_paths else None

        try:
            selected_output_dir, gradient_stats_dir = resolve_gradient_stats_dir(
                output_root=args.output_root,
                run_dir=args.run_dir,
                run_index=args.run_index,
            )
            updated_paths = find_gradient_plys(gradient_stats_dir)
        except Exception as exception:
            print(f"Warning: could not reload latest gradient stats ({exception})")
            return

        ply_paths = updated_paths

        if current_name is not None:
            matching_indices = [i for i, path in enumerate(ply_paths) if path.name == current_name]
            current_index = matching_indices[0] if matching_indices else min(current_index, len(ply_paths) - 1)
        else:
            current_index = 0

    def load_current_ply(reset_camera: bool = False) -> None:
        nonlocal poly_data

        current_path = ply_paths[current_index]
        current_colormap = COLORMAP_NAMES[colormap_index]

        poly_data = read_ply_as_points(
            ply_path=current_path,
            colormap_name=current_colormap,
            scalar_name=args.scalar_name,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
        )

        mapper.SetInputData(poly_data)
        mapper.Modified()

        actor.GetProperty().SetPointSize(point_size)

        update_status_text(
            text_actor=status_text_actor,
            current_index=current_index,
            ply_paths=ply_paths,
            selected_output_dir=selected_output_dir,
            gradient_stats_dir=gradient_stats_dir,
            point_size=point_size,
            colormap_name=current_colormap,
        )

        render_window.SetWindowName(
            f"{current_index + 1}/{len(ply_paths)} - "
            f"{current_path.name} - {current_colormap}"
        )

        if reset_camera:
            renderer.ResetCamera()

        renderer.ResetCameraClippingRange()
        render_window.Render()

        print(
            f"Showing {current_index + 1}/{len(ply_paths)}: "
            f"{current_path.name} | colormap={current_colormap} | dir={gradient_stats_dir}"
        )

    def execute_action(action: str) -> None:
        nonlocal current_index
        nonlocal point_size
        nonlocal colormap_index

        if action == "next":
            reload_file_list_preserving_current()
            current_index = (current_index + 1) % len(ply_paths)
            load_current_ply(reset_camera=args.reset_camera_on_load)
            return

        if action == "previous":
            reload_file_list_preserving_current()
            current_index = (current_index - 1) % len(ply_paths)
            load_current_ply(reset_camera=args.reset_camera_on_load)
            return

        if action == "first":
            reload_file_list_preserving_current()
            current_index = 0
            load_current_ply(reset_camera=args.reset_camera_on_load)
            return

        if action == "last":
            reload_file_list_preserving_current()
            current_index = len(ply_paths) - 1
            load_current_ply(reset_camera=args.reset_camera_on_load)
            return

        if action == "reload":
            reload_file_list_preserving_current()
            load_current_ply(reset_camera=False)
            return

        if action == "next_colormap":
            colormap_index = (colormap_index + 1) % len(COLORMAP_NAMES)
            load_current_ply(reset_camera=False)
            return

        if action == "point_size_up":
            point_size = min(point_size + float(args.point_size_step), 50.0)
            load_current_ply(reset_camera=False)
            return

        if action == "point_size_down":
            point_size = max(point_size - float(args.point_size_step), 1.0)
            load_current_ply(reset_camera=False)
            return

        if action == "quit":
            interactor.GetRenderWindow().Finalize()
            interactor.TerminateApp()
            return

    def queue_action(action: str) -> None:
        nonlocal queued_action
        queued_action = action

    def key_symbol_to_navigation_action(key_symbol: str) -> str | None:
        if key_symbol in ("Right", "d", "D"):
            return "next"
        if key_symbol in ("Left", "a", "A"):
            return "previous"
        return None

    def on_key_press(caller, event_name) -> None:
        nonlocal held_navigation_action

        key_symbol = interactor.GetKeySym()
        first_press = key_symbol not in pressed_keys
        pressed_keys.add(key_symbol)

        navigation_action = key_symbol_to_navigation_action(key_symbol)
        if navigation_action is not None:
            held_navigation_action = navigation_action
            if first_press:
                queue_action(navigation_action)
            return

        if not first_press:
            return

        if key_symbol == "space":
            queue_action("next_colormap")
            return

        if key_symbol == "Home":
            queue_action("first")
            return

        if key_symbol == "End":
            queue_action("last")
            return

        if key_symbol in ("r", "R"):
            queue_action("reload")
            return

        if key_symbol in ("plus", "equal", "KP_Add"):
            queue_action("point_size_up")
            return

        if key_symbol in ("minus", "underscore", "KP_Subtract"):
            queue_action("point_size_down")
            return

        if key_symbol in ("q", "Q", "Escape"):
            queue_action("quit")
            return

    def on_key_release(caller, event_name) -> None:
        nonlocal held_navigation_action

        key_symbol = interactor.GetKeySym()
        pressed_keys.discard(key_symbol)

        navigation_action = key_symbol_to_navigation_action(key_symbol)
        if navigation_action is not None and held_navigation_action == navigation_action:
            held_navigation_action = None

    def on_timer_event(caller, event_name) -> None:
        nonlocal queued_action
        nonlocal last_navigation_action_time

        action = queued_action
        if action is not None:
            queued_action = None
            execute_action(action)

            if action in ("next", "previous"):
                last_navigation_action_time = time.monotonic()

            return

        if held_navigation_action is None:
            return

        current_time = time.monotonic()
        if current_time - last_navigation_action_time < navigation_repeat_seconds:
            return

        execute_action(held_navigation_action)
        last_navigation_action_time = current_time

    def clear_key_state(caller, event_name) -> None:
        nonlocal held_navigation_action
        pressed_keys.clear()
        held_navigation_action = None

    render_window.SetWindowName(
        f"{current_index + 1}/{len(ply_paths)} - "
        f"{ply_paths[current_index].name} - {COLORMAP_NAMES[colormap_index]}"
    )

    interactor.AddObserver("KeyPressEvent", on_key_press)
    interactor.AddObserver("KeyReleaseEvent", on_key_release)
    interactor.AddObserver("TimerEvent", on_timer_event)
    interactor.AddObserver("LeaveEvent", clear_key_state)

    interactor.Initialize()
    camera = renderer.GetActiveCamera()
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.SetFocalPoint(0.0, 0.0, 0.25)

    #renderer.ResetCamera()
    renderer.ResetCameraClippingRange()
    render_window.Render()
    interactor.CreateRepeatingTimer(50)
    interactor.Start()


if __name__ == "__main__":
    main()