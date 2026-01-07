#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh


@dataclass(frozen=True)
class DistanceStats:
    count: int
    mean: float
    rms: float
    median: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float


def compute_stats(distances: np.ndarray) -> DistanceStats:
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("distances must be a non-empty 1D array")

    distances_sorted = np.sort(distances)
    rms = float(np.sqrt(np.mean(distances_sorted * distances_sorted)))

    return DistanceStats(
        count=int(distances_sorted.size),
        mean=float(np.mean(distances_sorted)),
        rms=rms,
        median=float(np.median(distances_sorted)),
        p90=float(np.percentile(distances_sorted, 90.0)),
        p95=float(np.percentile(distances_sorted, 95.0)),
        p99=float(np.percentile(distances_sorted, 99.0)),
        min=float(distances_sorted[0]),
        max=float(distances_sorted[-1]),
    )


def load_mesh_from_obj(obj_path: Path) -> trimesh.Trimesh:
    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    loaded = trimesh.load(obj_path, force="scene", process=False)

    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"OBJ scene has no geometry: {obj_path}")
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"OBJ scene has no Trimesh geometry: {obj_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        raise TypeError(f"Unsupported OBJ load result type: {type(loaded)}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("GT mesh has no faces; point-to-surface requires triangles.")
    return mesh


def _ply_numpy_dtype(ply_type: str) -> np.dtype:
    ply_type = ply_type.lower()
    mapping = {
        "char": np.int8,
        "int8": np.int8,
        "uchar": np.uint8,
        "uint8": np.uint8,
        "short": np.int16,
        "int16": np.int16,
        "ushort": np.uint16,
        "uint16": np.uint16,
        "int": np.int32,
        "int32": np.int32,
        "uint": np.uint32,
        "uint32": np.uint32,
        "float": np.float32,
        "float32": np.float32,
        "double": np.float64,
        "float64": np.float64,
    }
    if ply_type not in mapping:
        raise ValueError(f"Unsupported PLY scalar type: '{ply_type}'")
    return np.dtype(mapping[ply_type])


def load_points_from_ply_vertices(
    ply_path: Path,
    opacity_threshold: float = 0.5,
    z_min_threshold_binary: float = 0.22,
) -> np.ndarray:
    """
    Robust PLY reader that tolerates non-standard trailing tokens after property names.

    Supports:
      - format ascii 1.0
      - format binary_little_endian 1.0

    Filtering:
      - If 'opacity' exists: keep points with opacity >= opacity_threshold
      - If binary_little_endian (2DGS): additionally require z > z_min_threshold_binary

    Returns:
      vertices_xyz: (N_filtered, 3) float64
    """
    if not ply_path.is_file():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    with ply_path.open("rb") as f:
        header_lines: List[str] = []
        while True:
            line_bytes = f.readline()
            if not line_bytes:
                raise ValueError("Unexpected EOF while reading PLY header")
            line = line_bytes.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        if not header_lines or header_lines[0] != "ply":
            raise ValueError("Not a PLY file (missing 'ply' magic)")

        ply_format: str | None = None
        vertex_count: int | None = None
        vertex_properties: List[Tuple[str, np.dtype]] = []
        in_vertex_element = False

        for line in header_lines[1:]:
            if line.startswith("format "):
                tokens = line.split()
                if len(tokens) < 3:
                    raise ValueError(f"Malformed format line: '{line}'")
                ply_format = tokens[1].lower()

            elif line.startswith("element "):
                tokens = line.split()
                if len(tokens) != 3:
                    continue
                in_vertex_element = (tokens[1].lower() == "vertex")
                if in_vertex_element:
                    vertex_count = int(tokens[2])
                    vertex_properties = []

            elif line.startswith("property ") and in_vertex_element:
                tokens = line.split()
                if len(tokens) < 3:
                    continue
                prop_type = tokens[1]
                prop_name = tokens[2]
                if prop_type.lower() == "list":
                    raise ValueError("List properties are not supported.")
                vertex_properties.append((prop_name, _ply_numpy_dtype(prop_type)))

        if ply_format is None:
            raise ValueError("PLY header missing 'format'")
        if vertex_count is None:
            raise ValueError("PLY header missing 'element vertex'")
        if not vertex_properties:
            raise ValueError("PLY vertex element has no scalar properties")

        property_names = [n for (n, _) in vertex_properties]

        try:
            x_index = property_names.index("x")
            y_index = property_names.index("y")
            z_index = property_names.index("z")
        except ValueError as exc:
            raise ValueError(f"x/y/z not found. Found: {property_names[:32]}") from exc

        opacity_index = property_names.index("opacity") if "opacity" in property_names else None

        # ---------- ASCII ----------
        if ply_format == "ascii":
            with ply_path.open("r", encoding="ascii", errors="replace") as tf:
                for _ in range(len(header_lines)):
                    tf.readline()
                data = np.loadtxt(tf, dtype=np.float64, max_rows=vertex_count)

            if data.ndim == 1:
                data = data.reshape(1, -1)

            vertices_xyz = data[:, [x_index, y_index, z_index]]

            keep_mask = np.ones(vertices_xyz.shape[0], dtype=bool)

            # z filtering (same threshold as binary)
            #keep_mask &= vertices_xyz[:, 2] > z_min_threshold_binary

            if opacity_index is not None:
                keep_mask &= data[:, opacity_index] >= opacity_threshold

            vertices_xyz = vertices_xyz[keep_mask]

        # ---------- BINARY (2DGS) ----------
        elif ply_format == "binary_little_endian":
            dtype_fields = [(name, dt.newbyteorder("<")) for (name, dt) in vertex_properties]
            structured_dtype = np.dtype(dtype_fields)

            vertex_records = np.fromfile(f, dtype=structured_dtype, count=vertex_count)

            x_vals = vertex_records[property_names[x_index]].astype(np.float64)
            y_vals = vertex_records[property_names[y_index]].astype(np.float64)
            z_vals = vertex_records[property_names[z_index]].astype(np.float64)

            keep_mask = z_vals > z_min_threshold_binary

            if opacity_index is not None:
                opacity_vals = vertex_records[property_names[opacity_index]].astype(np.float64)
                keep_mask &= opacity_vals >= opacity_threshold

            vertices_xyz = np.stack(
                [x_vals[keep_mask], y_vals[keep_mask], z_vals[keep_mask]],
                axis=1,
            )

        else:
            raise ValueError(f"Unsupported PLY format: '{ply_format}'")

    if vertices_xyz.ndim != 2 or vertices_xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 vertices, got shape {vertices_xyz.shape}")

    return vertices_xyz




def point_to_surface_distances(mesh: trimesh.Trimesh, points_xyz: np.ndarray) -> np.ndarray:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must be Nx3, got {points_xyz.shape}")

    _, distances, _ = trimesh.proximity.closest_point(mesh, points_xyz)
    return np.asarray(distances, dtype=np.float64)


def save_distances_csv(output_csv: Path, distances: np.ndarray) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_csv, distances.reshape(-1, 1), delimiter=",", header="distance", comments="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute unsigned point-to-surface distances from a point cloud PLY to a GT OBJ mesh."
    )
    parser.add_argument("--gt_obj", type=Path, required=True, help="Path to ground-truth .obj mesh")
    parser.add_argument("--points_ply", type=Path, required=True, help="Path to point cloud .ply (beta surfel or 2DGS)")
    parser.add_argument("--out_csv", type=Path, default=None, help="Optional: save per-point distances to CSV")
    args = parser.parse_args()

    gt_mesh = load_mesh_from_obj(args.gt_obj)
    points_xyz = load_points_from_ply_vertices(args.points_ply)

    distances = point_to_surface_distances(gt_mesh, points_xyz)
    stats = compute_stats(distances)

    print(f"GT mesh:  {args.gt_obj}")
    print(f"Points:   {args.points_ply}")
    print("")
    print("Point-to-surface distance stats (unsigned, OBJ units):")
    print(f"  count : {stats.count}")
    print(f"  mean  : {stats.mean:.6e}")
    print(f"  rms   : {stats.rms:.6e}")
    print(f"  median: {stats.median:.6e}")
    print(f"  p90   : {stats.p90:.6e}")
    print(f"  p95   : {stats.p95:.6e}")
    print(f"  p99   : {stats.p99:.6e}")
    print(f"  min   : {stats.min:.6e}")
    print(f"  max   : {stats.max:.6e}")

    if args.out_csv is not None:
        save_distances_csv(args.out_csv, distances)
        print(f"\nWrote per-point distances to: {args.out_csv}")


if __name__ == "__main__":
    main()
