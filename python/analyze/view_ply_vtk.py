#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import vtk


def find_latest_points_ply(output_root_path: Path, use_initial: bool) -> Path:
    if not output_root_path.exists():
        raise FileNotFoundError(f"Output root '{output_root_path}' does not exist.")

    target_filename = "initial_points.ply" if use_initial else "points_final.ply"

    if output_root_path.is_file():
        print(f"Using PLY file: {output_root_path}")
        return output_root_path

    points_in_root = output_root_path / target_filename
    if points_in_root.is_file():
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

    latest_run_dir = max(
        candidate_run_dirs,
        key=lambda run_path: (run_path / target_filename).stat().st_mtime,
    )
    latest_ply_path = latest_run_dir / target_filename

    print(f"Using latest run directory: {latest_run_dir}")
    print(f"{target_filename}: {latest_ply_path}")
    return latest_ply_path


def numpy_rgb01_and_alpha01_to_vtk_u8_rgba(name: str, rgb01: np.ndarray,
                                           alpha01: np.ndarray) -> vtk.vtkUnsignedCharArray:
    rgbU8 = (np.asarray(rgb01, dtype=np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    aU8 = (np.asarray(alpha01, dtype=np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)

    rgba = np.concatenate([rgbU8, aU8.reshape(-1, 1)], axis=1)

    arrayHandle = vtk.vtkUnsignedCharArray()
    arrayHandle.SetName(name)
    arrayHandle.SetNumberOfComponents(4)
    arrayHandle.SetNumberOfTuples(rgba.shape[0])
    for i in range(rgba.shape[0]):
        arrayHandle.SetTuple4(i, int(rgba[i, 0]), int(rgba[i, 1]), int(rgba[i, 2]), int(rgba[i, 3]))
    return arrayHandle


# -----------------------------------------------------------------------------
# Reused loader style, corrected for your PLY layout (albedo at 11..13, opacity at 14)
# -----------------------------------------------------------------------------
def load_surfels_from_ply(
        plyPath: Path,
        opacityThreshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pkX: List[float] = []
    pkY: List[float] = []
    pkZ: List[float] = []

    tuX: List[float] = []
    tuY: List[float] = []
    tuZ: List[float] = []

    tvX: List[float] = []
    tvY: List[float] = []
    tvZ: List[float] = []

    suValues: List[float] = []
    svValues: List[float] = []

    color0: List[float] = []
    color1: List[float] = []
    color2: List[float] = []
    opacityValues: List[float] = []

    with plyPath.open("r", encoding="utf-8") as fileHandle:
        headerFinished = False
        for line in fileHandle:
            if not headerFinished:
                if line.strip() == "end_header":
                    headerFinished = True
                continue

            parts = line.strip().split()
            if not parts:
                continue

            # Need at least index 14 for opacity
            if len(parts) < 15:
                continue

            opacityValue = float(parts[14])
            if opacityValue < opacityThreshold:
                continue

            opacityValues.append(opacityValue)

            pkX.append(float(parts[0]))
            pkY.append(float(parts[1]))
            pkZ.append(float(parts[2]))

            tuX.append(float(parts[3]))
            tuY.append(float(parts[4]))
            tuZ.append(float(parts[5]))

            tvX.append(float(parts[6]))
            tvY.append(float(parts[7]))
            tvZ.append(float(parts[8]))

            suValues.append(float(parts[9]))
            svValues.append(float(parts[10]))

            # Albedo at 11..13
            color0.append(float(parts[11]))
            color1.append(float(parts[12]))
            color2.append(float(parts[13]))

            # color0.append(float(0.7))
            # color1.append(float(0.99))
            # color2.append(float(0.99))

    if len(pkX) == 0:
        raise RuntimeError(f"No points loaded from '{plyPath}'. Try lowering --opacity-threshold.")

    positions = np.stack([pkX, pkY, pkZ], axis=1).astype(np.float32)
    tangentU = np.stack([tuX, tuY, tuZ], axis=1).astype(np.float32)
    tangentV = np.stack([tvX, tvY, tvZ], axis=1).astype(np.float32)
    su = np.asarray(suValues, dtype=np.float32)
    sv = np.asarray(svValues, dtype=np.float32)
    colors = np.stack([color0, color1, color2], axis=1).astype(np.float32).clip(0.0, 1.0)
    opacities = np.asarray(opacityValues, dtype=np.float32).clip(0.0, 1.0)

    print(f"Loaded {positions.shape[0]} points from {plyPath}")
    return positions, tangentU, tangentV, su, sv, colors, opacities


# -----------------------------------------------------------------------------
# Quaternion orientation (for VTK versions without matrix orientation mode)
# -----------------------------------------------------------------------------
def rotation_matrix_to_quaternion_wxyz(rotationMatrices: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices (N,3,3) to quaternions in (w,x,y,z).
    Numerically stable branch-based conversion.
    """
    r = rotationMatrices
    q = np.zeros((r.shape[0], 4), dtype=np.float32)

    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    positive = trace > 0.0

    # trace > 0
    t = trace[positive]
    s = np.sqrt(t + 1.0) * 2.0
    q[positive, 0] = 0.25 * s
    q[positive, 1] = (r[positive, 2, 1] - r[positive, 1, 2]) / s
    q[positive, 2] = (r[positive, 0, 2] - r[positive, 2, 0]) / s
    q[positive, 3] = (r[positive, 1, 0] - r[positive, 0, 1]) / s

    # trace <= 0
    neg = ~positive
    if np.any(neg):
        rNeg = r[neg]
        diag = np.stack([rNeg[:, 0, 0], rNeg[:, 1, 1], rNeg[:, 2, 2]], axis=1)
        maxIndex = np.argmax(diag, axis=1)

        for k in (0, 1, 2):
            mask = maxIndex == k
            if not np.any(mask):
                continue
            rk = rNeg[mask]

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

            outIdx = np.where(neg)[0][mask]
            q[outIdx, 0] = qw
            q[outIdx, 1] = qx
            q[outIdx, 2] = qy
            q[outIdx, 3] = qz

    # Normalize
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return q


def build_orientation_quaternions_wxyz(tangentU: np.ndarray, tangentV: np.ndarray) -> np.ndarray:
    """
    Build a local frame:
      X axis = u
      Y axis = v (re-orthogonalized)
      Z axis = n = cross(u,v)
    Disk source is in XY plane, so this rotates it into the surfel plane.
    Returns quaternions as (w,x,y,z) per point for VTK quaternion mode.
    """
    u = tangentU.astype(np.float32)
    v = tangentV.astype(np.float32)

    uHat = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    vHat = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    n = np.cross(uHat, vHat)
    nHat = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

    vOrtho = vHat - (np.sum(vHat * uHat, axis=1, keepdims=True) * uHat)
    vOrtho = vOrtho / (np.linalg.norm(vOrtho, axis=1, keepdims=True) + 1e-12)

    # Rotation matrix columns = [u v n]
    rotationMatrices = np.zeros((uHat.shape[0], 3, 3), dtype=np.float32)
    rotationMatrices[:, :, 0] = uHat
    rotationMatrices[:, :, 1] = vOrtho
    rotationMatrices[:, :, 2] = nHat

    return rotation_matrix_to_quaternion_wxyz(rotationMatrices)


def numpy_to_vtk_float_array(name: str, data: np.ndarray, numComponents: int) -> vtk.vtkFloatArray:
    flat = np.asarray(data, dtype=np.float32).reshape(data.shape[0], numComponents)
    arrayHandle = vtk.vtkFloatArray()
    arrayHandle.SetName(name)
    arrayHandle.SetNumberOfComponents(numComponents)
    arrayHandle.SetNumberOfTuples(flat.shape[0])
    for i in range(flat.shape[0]):
        arrayHandle.SetTuple(i, flat[i].tolist())
    return arrayHandle


def numpy_rgb01_to_vtk_u8_rgb(name: str, rgb01: np.ndarray) -> vtk.vtkUnsignedCharArray:
    rgbU8 = (np.asarray(rgb01, dtype=np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    arrayHandle = vtk.vtkUnsignedCharArray()
    arrayHandle.SetName(name)
    arrayHandle.SetNumberOfComponents(3)
    arrayHandle.SetNumberOfTuples(rgbU8.shape[0])
    for i in range(rgbU8.shape[0]):
        arrayHandle.SetTuple3(i, int(rgbU8[i, 0]), int(rgbU8[i, 1]), int(rgbU8[i, 2]))
    return arrayHandle


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VTK viewer: render surfels as oriented ellipses (glyphs).")
    parser.add_argument("--output-root", type=Path, required=False, default=Path("../Assets/OptimizationOutput"))
    parser.add_argument("--initial", action="store_true",
                        help="Load initial_points.ply instead of points_final.ply.")
    parser.add_argument("--opacity-threshold", type=float, default=0.0)
    parser.add_argument("--area-threshold", type=float, default=0.0)
    parser.add_argument("--max-ellipses", type=int, default=0)
    parser.add_argument("--disk-resolution", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--scale", type=float, default=1)
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    ply_path = find_latest_points_ply(args.output_root, use_initial=args.initial)

    positions, tangent_u, tangent_v, su, sv, colors, opacities = load_surfels_from_ply(
        ply_path,
        opacityThreshold=args.opacity_threshold,
    )

    ellipse_area = su * sv
    ellipse_mask = ellipse_area >= float(args.area_threshold)

    positions = positions[ellipse_mask]
    tangent_u = tangent_u[ellipse_mask]
    tangent_v = tangent_v[ellipse_mask]
    su = su[ellipse_mask] * args.scale
    sv = sv[ellipse_mask] * args.scale
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

    print(f"Rendering {positions.shape[0]} surfels")

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
    renderer.SetBackground(0.2, 0.2, 0.25)
    renderer.SetUseDepthPeeling(True)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 900)
    render_window.SetAlphaBitPlanes(True)
    render_window.SetMultiSamples(0)

    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(0.1)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)

    camera = renderer.GetActiveCamera()
    camera.SetViewUp(0.0, 1.0, 0.0)

    renderer.ResetCamera()
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
