from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class VertexTable:
    property_names: list[str]
    values: np.ndarray

    @property
    def property_lookup(self) -> dict[str, int]:
        return {property_name.lower(): index for index, property_name in enumerate(self.property_names)}


def parse_run_timestamp(path_name: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", path_name)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")


def find_run(output_root: Path, run_index: int) -> tuple[Path, Path]:
    output_root = output_root.resolve()

    if output_root.is_file():
        if output_root.suffix.lower() != ".ply":
            raise ValueError(f"Expected a .ply point cloud, got {output_root}")
        return output_root.parent, output_root

    if (output_root / "points_final.ply").is_file():
        return output_root, output_root / "points_final.ply"

    candidates: list[tuple[bool, datetime, float, Path]] = []
    for run_dir in output_root.iterdir():
        points_path = run_dir / "points_final.ply"
        if not run_dir.is_dir() or not points_path.is_file():
            continue
        timestamp = parse_run_timestamp(run_dir.name)
        candidates.append((timestamp is not None, timestamp or datetime.min, points_path.stat().st_mtime, run_dir))

    if not candidates:
        raise FileNotFoundError(f"No run folders with points_final.ply found under {output_root}")

    candidates.sort(reverse=True)

    if run_index < 0 or run_index >= len(candidates):
        available_runs = "\n".join(f"[{i}] {item[3].name}" for i, item in enumerate(candidates))
        raise IndexError(f"--index {run_index} is out of range.\nAvailable runs:\n{available_runs}")

    run_dir = candidates[run_index][3]
    return run_dir, run_dir / "points_final.ply"


def format_suffix_value(value: float) -> str:
    return f"{value:.9g}".replace("-", "m").replace(".", "p")


def read_ascii_ply_vertex_table(points_path: Path) -> VertexTable:
    points_path = points_path.resolve()

    vertex_count: int | None = None
    vertex_property_names: list[str] = []
    current_element = None
    header_line_count = 0
    ply_format: str | None = None

    with points_path.open("r", encoding="utf-8") as file:
        for line in file:
            header_line_count += 1
            stripped_line = line.strip()
            parts = stripped_line.split()

            if not parts:
                continue

            if parts[0] == "format":
                if len(parts) < 2:
                    raise ValueError(f"Malformed PLY format line in {points_path}")
                ply_format = parts[1]

            if parts[0] == "element":
                if len(parts) < 3:
                    raise ValueError(f"Malformed PLY element line: {stripped_line}")
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                continue

            if parts[0] == "property" and current_element == "vertex":
                if len(parts) < 3:
                    raise ValueError(f"Malformed PLY property line: {stripped_line}")
                vertex_property_names.append(parts[-1])
                continue

            if stripped_line == "end_header":
                break

    if ply_format != "ascii":
        raise ValueError(
            f"{points_path} is {ply_format!r}. This script expects ASCII PLY because surfel metadata "
            "uses custom vertex properties. Export ASCII PLY, or add plyfile/binary support."
        )

    if vertex_count is None:
        raise ValueError(f"{points_path} does not contain a vertex element.")

    if not vertex_property_names:
        raise ValueError(f"{points_path} does not contain vertex properties.")

    values = np.loadtxt(str(points_path), dtype=np.float64, skiprows=header_line_count, max_rows=vertex_count)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.shape[1] < len(vertex_property_names):
        raise ValueError(f"Loaded {values.shape[1]} vertex columns, but PLY header declares {len(vertex_property_names)}.")

    values = values[:, :len(vertex_property_names)]
    return VertexTable(property_names=vertex_property_names, values=values)


def find_property_indices(vertex_table: VertexTable, property_names: list[str]) -> list[int] | None:
    lookup = vertex_table.property_lookup
    indices: list[int] = []

    for property_name in property_names:
        index = lookup.get(property_name.lower())
        if index is None:
            return None
        indices.append(index)

    return indices


def find_property_triplet(vertex_table: VertexTable, candidates: list[tuple[str, str, str]]) -> np.ndarray | None:
    for candidate in candidates:
        indices = find_property_indices(vertex_table, list(candidate))
        if indices is not None:
            return vertex_table.values[:, indices]
    return None


def find_property_pair(vertex_table: VertexTable, candidates: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray] | None:
    for candidate in candidates:
        indices = find_property_indices(vertex_table, list(candidate))
        if indices is not None:
            return vertex_table.values[:, indices[0]], vertex_table.values[:, indices[1]]
    return None


def find_single_property(vertex_table: VertexTable, candidates: list[str]) -> np.ndarray | None:
    lookup = vertex_table.property_lookup
    for candidate in candidates:
        index = lookup.get(candidate.lower())
        if index is not None:
            return vertex_table.values[:, index]
    return None


def make_vector_property_candidates(base_names: list[str]) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    suffix_groups = [
        ("_x", "_y", "_z"),
        ("x", "y", "z"),
        ("_X", "_Y", "_Z"),
        ("X", "Y", "Z"),
        ("_0", "_1", "_2"),
        ("0", "1", "2"),
    ]

    for base_name in base_names:
        for suffix_group in suffix_groups:
            candidates.append(tuple(base_name + suffix for suffix in suffix_group))

    return candidates


def safe_normalize(vectors: np.ndarray, fallback: np.ndarray | None = None, eps: float = 1.0e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = norms[:, 0] > eps

    normalized = np.zeros_like(vectors)
    normalized[valid] = vectors[valid] / norms[valid]

    if fallback is not None:
        fallback = np.asarray(fallback, dtype=np.float64)
        if fallback.ndim == 1:
            fallback = np.repeat(fallback.reshape(1, 3), vectors.shape[0], axis=0)
        normalized[~valid] = fallback[~valid]

    return normalized


def load_positions(vertex_table: VertexTable) -> np.ndarray:
    position_candidates = [
        ("x", "y", "z"),
        ("position_x", "position_y", "position_z"),
        ("pos_x", "pos_y", "pos_z"),
        ("mean_x", "mean_y", "mean_z"),
        ("means_x", "means_y", "means_z"),
    ]

    positions = find_property_triplet(vertex_table, position_candidates)
    if positions is None:
        if vertex_table.values.shape[1] < 3:
            raise ValueError("Could not infer x/y/z columns from PLY file.")
        positions = vertex_table.values[:, :3]

    return np.ascontiguousarray(positions, dtype=np.float64)


def load_colors(vertex_table: VertexTable) -> np.ndarray | None:
    color_candidates = [
        ("red", "green", "blue"),
        ("r", "g", "b"),
        ("color_r", "color_g", "color_b"),
        ("albedo_r", "albedo_g", "albedo_b"),
        ("diffuse_r", "diffuse_g", "diffuse_b"),
        ("base_color_r", "base_color_g", "base_color_b"),
    ]

    colors = find_property_triplet(vertex_table, color_candidates)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.max(initial=0.0) > 1.5:
            colors = colors / 255.0
        return np.clip(colors, 0.0, 1.0)

    sh_dc_colors = find_property_triplet(vertex_table, [("f_dc_0", "f_dc_1", "f_dc_2")])
    if sh_dc_colors is not None:
        spherical_harmonics_c0 = 0.28209479177387814
        colors = 0.5 + spherical_harmonics_c0 * sh_dc_colors
        return np.clip(colors, 0.0, 1.0)

    return None


def load_existing_normals(vertex_table: VertexTable) -> np.ndarray | None:
    normal_candidates = [
        ("nx", "ny", "nz"),
        ("normal_x", "normal_y", "normal_z"),
        ("norm_x", "norm_y", "norm_z"),
        ("n_x", "n_y", "n_z"),
        ("normal_0", "normal_1", "normal_2"),
        ("normal0", "normal1", "normal2"),
    ]

    normals = find_property_triplet(vertex_table, normal_candidates)
    if normals is None:
        return None

    return safe_normalize(normals)


def load_tangent_vectors(vertex_table: VertexTable) -> tuple[np.ndarray, np.ndarray] | None:
    tangent_u_candidates = make_vector_property_candidates(["tanU", "tanu", "tan_u", "tangentU", "tangent_u", "tu", "axis_u", "basis_u"])
    tangent_v_candidates = make_vector_property_candidates(["tanV", "tanv", "tan_v", "tangentV", "tangent_v", "tv", "axis_v", "basis_v"])

    tangent_u = find_property_triplet(vertex_table, tangent_u_candidates)
    tangent_v = find_property_triplet(vertex_table, tangent_v_candidates)

    if tangent_u is None or tangent_v is None:
        return None

    return np.asarray(tangent_u, dtype=np.float64), np.asarray(tangent_v, dtype=np.float64)


def load_tangent_normals(vertex_table: VertexTable) -> np.ndarray | None:
    tangent_vectors = load_tangent_vectors(vertex_table)
    if tangent_vectors is None:
        return None

    tangent_u, tangent_v = tangent_vectors
    normals = np.cross(tangent_u, tangent_v)
    return safe_normalize(normals)


def construct_tangent_basis_from_normals(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = safe_normalize(normals, fallback=np.array([0.0, 0.0, 1.0]))

    helper = np.zeros_like(normals)
    use_z_helper = np.abs(normals[:, 2]) < 0.9
    helper[use_z_helper] = np.array([0.0, 0.0, 1.0])
    helper[~use_z_helper] = np.array([1.0, 0.0, 0.0])

    tangent_u = safe_normalize(np.cross(helper, normals), fallback=np.array([1.0, 0.0, 0.0]))
    tangent_v = safe_normalize(np.cross(normals, tangent_u), fallback=np.array([0.0, 1.0, 0.0]))

    return tangent_u, tangent_v


def orthonormalize_tangent_basis(tangent_u: np.ndarray, tangent_v: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    generated_tangent_u, generated_tangent_v = construct_tangent_basis_from_normals(normals)

    tangent_u = tangent_u - np.sum(tangent_u * normals, axis=1, keepdims=True) * normals
    tangent_u = safe_normalize(tangent_u, fallback=generated_tangent_u)

    tangent_v = tangent_v - np.sum(tangent_v * normals, axis=1, keepdims=True) * normals
    tangent_v = safe_normalize(tangent_v, fallback=generated_tangent_v)

    orientation = np.sum(np.cross(tangent_u, tangent_v) * normals, axis=1)
    tangent_v[orientation < 0.0] *= -1.0

    return tangent_u, tangent_v


def estimate_pca_normals(points: np.ndarray, pca_knn: int) -> np.ndarray:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=pca_knn), fast_normal_computation=False)
    return np.asarray(point_cloud.normals, dtype=np.float64)


def orient_normals_outward(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0, keepdims=True)
    radial_vectors = points - center
    inward_mask = np.sum(radial_vectors * normals, axis=1) < 0.0
    normals = normals.copy()
    normals[inward_mask] *= -1.0
    return normals


def build_normals(points: np.ndarray, vertex_table: VertexTable, normal_mode: str) -> tuple[np.ndarray, str]:
    normals: np.ndarray | None = None
    normal_source = "unknown"

    if normal_mode in {"auto", "existing"}:
        normals = load_existing_normals(vertex_table)
        if normals is not None:
            normal_source = "existing PLY normals"

    if normals is None and normal_mode in {"auto", "tangent"}:
        normals = load_tangent_normals(vertex_table)
        if normals is not None:
            normal_source = "cross(tanU, tanV)"

    if normals is None:
        raise ValueError(
            f"Could not construct normals with --normal-mode {normal_mode!r}. "
            "Use --normal-mode pca, or export nx/ny/nz or tanU/tanV properties."
        )

    normals = safe_normalize(normals, fallback=np.array([0.0, 0.0, 1.0]))
    normals = orient_normals_outward(points, normals)
    normals = safe_normalize(normals, fallback=np.array([0.0, 0.0, 1.0]))

    return normals, normal_source


def apply_scale_property_mode(scale_values: np.ndarray, scale_property_mode: str) -> np.ndarray:
    scale_values = np.asarray(scale_values, dtype=np.float64)

    if scale_property_mode == "linear":
        return scale_values

    if scale_property_mode == "exp":
        return np.exp(scale_values)

    if scale_property_mode != "auto":
        raise ValueError(f"Unknown scale property mode: {scale_property_mode}")

    finite_values = scale_values[np.isfinite(scale_values)]
    if finite_values.size == 0:
        return scale_values

    if np.nanpercentile(finite_values, 5.0) < 0.0:
        return np.exp(scale_values)

    return scale_values


def load_scale_pair(vertex_table: VertexTable, scale_property_mode: str, max_sample_radius: float) -> tuple[np.ndarray, np.ndarray] | None:
    scale_pair_candidates = [
        ("scale_u", "scale_v"),
        ("scaleU", "scaleV"),
        ("scaleu", "scalev"),
        ("su", "sv"),
        ("s_u", "s_v"),
        ("radius_u", "radius_v"),
        ("radiusU", "radiusV"),
        ("scale_0", "scale_1"),
        ("scale0", "scale1"),
    ]

    scale_pair = find_property_pair(vertex_table, scale_pair_candidates)
    if scale_pair is not None:
        radius_u, radius_v = scale_pair
        radius_u = apply_scale_property_mode(radius_u, scale_property_mode)
        radius_v = apply_scale_property_mode(radius_v, scale_property_mode)
    else:
        single_scale = find_single_property(vertex_table, ["scale", "radius", "s"])
        if single_scale is not None:
            radius_u = apply_scale_property_mode(single_scale, scale_property_mode)
            radius_v = radius_u.copy()
        else:
            return None

    radius_u = np.abs(np.asarray(radius_u, dtype=np.float64))
    radius_v = np.abs(np.asarray(radius_v, dtype=np.float64))

    if max_sample_radius > 0.0:
        radius_u = np.minimum(radius_u, max_sample_radius)
        radius_v = np.minimum(radius_v, max_sample_radius)

    return radius_u, radius_v


def expand_surfel_samples(
        points: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray | None,
        vertex_table: VertexTable,
        samples_per_surfel: int,
        scale_property_mode: str,
        max_sample_radius: float,
        sample_radial_exponent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    if samples_per_surfel <= 1:
        return points, normals, colors, "center-only"

    if sample_radial_exponent <= 0.0:
        raise ValueError("--sample-radial-exponent must be positive.")

    scale_pair = load_scale_pair(
        vertex_table=vertex_table,
        scale_property_mode=scale_property_mode,
        max_sample_radius=max_sample_radius,
    )

    if scale_pair is None:
        return points, normals, colors, "center-only: no scale/radius properties found"

    radius_u, radius_v = scale_pair

    tangent_vectors = load_tangent_vectors(vertex_table)
    if tangent_vectors is None:
        tangent_u, tangent_v = construct_tangent_basis_from_normals(normals)
        tangent_source = "basis from normals"
    else:
        tangent_u, tangent_v = tangent_vectors
        tangent_u, tangent_v = orthonormalize_tangent_basis(tangent_u, tangent_v, normals)
        tangent_source = "PLY tangent basis"

    sample_offsets: list[tuple[float, float]] = [(0.0, 0.0)]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for sample_index in range(samples_per_surfel - 1):
        normalized_index = (sample_index + 0.5) / max(samples_per_surfel - 1, 1)
        radial_coordinate = normalized_index ** sample_radial_exponent
        angular_coordinate = sample_index * golden_angle
        sample_offsets.append((radial_coordinate * np.cos(angular_coordinate), radial_coordinate * np.sin(angular_coordinate)))

    expanded_points: list[np.ndarray] = []
    expanded_normals: list[np.ndarray] = []
    expanded_colors: list[np.ndarray] = []

    for offset_u, offset_v in sample_offsets:
        offset = offset_u * radius_u[:, None] * tangent_u + offset_v * radius_v[:, None] * tangent_v
        expanded_points.append(points + offset)
        expanded_normals.append(normals)
        if colors is not None:
            expanded_colors.append(colors)

    sampled_points = np.ascontiguousarray(np.vstack(expanded_points), dtype=np.float64)
    sampled_normals = np.ascontiguousarray(np.vstack(expanded_normals), dtype=np.float64)
    sampled_colors = np.ascontiguousarray(np.vstack(expanded_colors), dtype=np.float64) if colors is not None else None

    return (
        sampled_points,
        sampled_normals,
        sampled_colors,
        f"{samples_per_surfel} samples/surfel using {tangent_source}, radial exponent {sample_radial_exponent:g}",
    )


def apply_valid_point_mask(vertex_table: VertexTable, points: np.ndarray, colors: np.ndarray | None, mask: np.ndarray) -> tuple[VertexTable, np.ndarray, np.ndarray | None]:
    filtered_table = VertexTable(property_names=vertex_table.property_names, values=vertex_table.values[mask])
    filtered_points = points[mask]
    filtered_colors = colors[mask] if colors is not None else None
    return filtered_table, filtered_points, filtered_colors


def make_point_cloud(points: np.ndarray, normals: np.ndarray, colors: np.ndarray | None) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    point_cloud.normals = o3d.utility.Vector3dVector(safe_normalize(normals, fallback=np.array([0.0, 0.0, 1.0])))

    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    return point_cloud


def normalize_point_cloud_normals(point_cloud: o3d.geometry.PointCloud) -> None:
    normals = np.asarray(point_cloud.normals)
    if normals.size == 0:
        raise ValueError("Point cloud has no normals. Poisson reconstruction requires oriented normals.")
    point_cloud.normals = o3d.utility.Vector3dVector(safe_normalize(normals, fallback=np.array([0.0, 0.0, 1.0])))


def orient_point_cloud_normals_outward(point_cloud: o3d.geometry.PointCloud) -> None:
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    if points.size == 0:
        return

    if normals.shape != points.shape:
        raise ValueError("Point cloud normal count does not match point count.")

    center = np.mean(points, axis=0, keepdims=True)
    radial_vectors = points - center
    inward_mask = np.sum(radial_vectors * normals, axis=1) < 0.0
    normals = normals.copy()
    normals[inward_mask] *= -1.0
    point_cloud.normals = o3d.utility.Vector3dVector(safe_normalize(normals, fallback=np.array([0.0, 0.0, 1.0])))


def split_point_cloud_components(point_cloud: o3d.geometry.PointCloud, component_eps: float, component_min_points: int) -> list[o3d.geometry.PointCloud]:
    if component_eps <= 0.0:
        return [point_cloud]

    labels = np.asarray(point_cloud.cluster_dbscan(eps=component_eps, min_points=component_min_points, print_progress=True))
    valid_labels = sorted(label for label in np.unique(labels) if label >= 0)

    components: list[o3d.geometry.PointCloud] = []
    for label in valid_labels:
        indices = np.where(labels == label)[0]
        if indices.size == 0:
            continue
        components.append(point_cloud.select_by_index(indices.tolist()))

    noise_indices = np.where(labels < 0)[0]
    if noise_indices.size > 0:
        print(f"DBSCAN marked {noise_indices.size} points as noise")

    return components


def remove_low_density_vertices(mesh: o3d.geometry.TriangleMesh, densities: o3d.utility.DoubleVector, density_quantile: float) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(mesh)

    if density_quantile <= 0.0:
        return mesh

    densities_np = np.asarray(densities)
    if densities_np.size == 0:
        return mesh

    density_threshold = np.quantile(densities_np, density_quantile)
    remove_mask = densities_np < density_threshold
    mesh.remove_vertices_by_mask(remove_mask.tolist())
    mesh.remove_unreferenced_vertices()

    return mesh


def crop_mesh_to_point_bounds(mesh: o3d.geometry.TriangleMesh, reference_points: np.ndarray, bbox_padding_factor: float) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(mesh)

    if bbox_padding_factor < 0.0 or len(mesh.vertices) == 0:
        return mesh

    min_bound = np.min(reference_points, axis=0)
    max_bound = np.max(reference_points, axis=0)
    diagonal_length = float(np.linalg.norm(max_bound - min_bound))
    padding = bbox_padding_factor * diagonal_length

    min_bound = min_bound - padding
    max_bound = max_bound + padding

    vertices = np.asarray(mesh.vertices)
    remove_mask = np.any((vertices < min_bound) | (vertices > max_bound), axis=1)
    mesh.remove_vertices_by_mask(remove_mask.tolist())
    mesh.remove_unreferenced_vertices()

    return mesh


def keep_largest_triangle_clusters(mesh: o3d.geometry.TriangleMesh, cluster_count_to_keep: int) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(mesh)

    if cluster_count_to_keep <= 0 or len(mesh.triangles) == 0:
        return mesh

    triangle_clusters, cluster_triangle_counts, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_triangle_counts = np.asarray(cluster_triangle_counts)

    keep_count = min(cluster_count_to_keep, len(cluster_triangle_counts))
    keep_clusters = np.argsort(cluster_triangle_counts)[-keep_count:]

    remove_mask = ~np.isin(triangle_clusters, keep_clusters)
    mesh.remove_triangles_by_mask(remove_mask.tolist())
    mesh.remove_unreferenced_vertices()

    return mesh


def post_process_mesh(mesh: o3d.geometry.TriangleMesh, cluster_count_to_keep: int) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(mesh)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh = keep_largest_triangle_clusters(mesh, cluster_count_to_keep)

    if len(mesh.vertices) > 0:
        mesh.compute_vertex_normals()

    return mesh


def create_poisson_mesh(point_cloud: o3d.geometry.PointCloud, poisson_depth: int, poisson_scale: float) -> tuple[o3d.geometry.TriangleMesh, o3d.utility.DoubleVector]:
    return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=poisson_depth,
        width=0.0,
        scale=poisson_scale,
        linear_fit=False,
    )


def reconstruct_poisson_component(
        point_cloud: o3d.geometry.PointCloud,
        poisson_depth: int,
        poisson_scale: float,
        density_quantile: float,
        bbox_padding_factor: float,
) -> o3d.geometry.TriangleMesh:
    orient_point_cloud_normals_outward(point_cloud)
    normalize_point_cloud_normals(point_cloud)

    component_reference_points = np.asarray(point_cloud.points)
    mesh, densities = create_poisson_mesh(point_cloud=point_cloud, poisson_depth=poisson_depth, poisson_scale=poisson_scale)
    mesh.compute_vertex_normals()
    mesh = remove_low_density_vertices(mesh=mesh, densities=densities, density_quantile=density_quantile)
    mesh = crop_mesh_to_point_bounds(mesh=mesh, reference_points=component_reference_points, bbox_padding_factor=bbox_padding_factor)

    if len(mesh.vertices) > 0:
        mesh.compute_vertex_normals()

    return mesh


def merge_meshes(meshes: list[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    merged_mesh = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        merged_mesh += mesh
    if len(merged_mesh.vertices) > 0:
        merged_mesh.compute_vertex_normals()
    return merged_mesh


def build_mesh_suffix(args: argparse.Namespace) -> str:
    suffix_parts: list[str] = []

    # if args.samples_per_surfel != 16:
    #     suffix_parts.append(f"samples_{args.samples_per_surfel}")
    #
    # if args.sample_radial_exponent != 0.5:
    #     suffix_parts.append(f"radial_exp_{format_suffix_value(args.sample_radial_exponent)}")
    #
    # if args.poisson_depth != 9:
    #     suffix_parts.append(f"depth_{args.poisson_depth}")
    #
    # if args.density_quantile != 0.02:
    #     suffix_parts.append(f"density_q_{format_suffix_value(args.density_quantile)}")
    #
    # if args.bbox_padding_factor != 0.05:
    #     suffix_parts.append(f"bbox_{format_suffix_value(args.bbox_padding_factor)}")
    #
    # if getattr(args, "split_components", False):
    #     suffix_parts.append(f"split_eps_{format_suffix_value(args.component_eps)}")

    return f"_{'_'.join(suffix_parts)}" if suffix_parts else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poisson mesh extraction from optimized surfel/point primitives.")
    parser.add_argument("--output_root", type=Path, default=Path("../Assets/OptimizationOutput"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--normal-mode", type=str, default="tangent", choices=["auto", "existing", "tangent"])
    parser.add_argument("--samples-per-surfel", type=int, default=24)
    parser.add_argument(
        "--sample-radial-exponent",
        type=float,
        default=1.2,
        help=(
            "Controls how surfel samples are distributed radially. "
            "0.5 preserves the old uniform-area disk sampling. "
            "Larger values concentrate samples closer to the surfel center."
        ),
    )
    parser.add_argument("--scale-property-mode", type=str, default="linear", choices=["auto", "linear", "exp"])
    parser.add_argument("--max-sample-radius", type=float, default=0.00)
    parser.add_argument("--poisson-depth", type=int, default=10)
    parser.add_argument("--poisson-scale", type=float, default=1.1)
    parser.add_argument("--density-quantile", type=float, default=0.05)
    parser.add_argument("--bbox-padding-factor", type=float, default=0.05)
    parser.add_argument("--num-cluster", type=int, default=50)
    parser.add_argument("--split-components", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--component-eps", type=float, default=0.05)
    parser.add_argument("--component-min-points", type=int, default=25)
    parser.add_argument("--save-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir, points_path = find_run(args.output_root, args.index)
    mesh_dir = run_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_name_suffix = build_mesh_suffix(args)

    print(f"Extracting Poisson mesh from {run_dir}")
    print(f"Using point cloud {points_path}")

    vertex_table = read_ascii_ply_vertex_table(points_path)
    points = load_positions(vertex_table)
    colors = load_colors(vertex_table)

    finite_mask = np.all(np.isfinite(points), axis=1)
    if colors is not None:
        finite_mask &= np.all(np.isfinite(colors), axis=1)

    vertex_table, points, colors = apply_valid_point_mask(vertex_table, points, colors, finite_mask)

    if points.shape[0] == 0:
        raise RuntimeError("No valid points remain after filtering.")

    normals, normal_source = build_normals(points=points, vertex_table=vertex_table, normal_mode=args.normal_mode)

    sampled_points, sampled_normals, sampled_colors, sampling_source = expand_surfel_samples(
        points=points,
        normals=normals,
        colors=colors,
        vertex_table=vertex_table,
        samples_per_surfel=args.samples_per_surfel,
        scale_property_mode=args.scale_property_mode,
        max_sample_radius=args.max_sample_radius,
        sample_radial_exponent=args.sample_radial_exponent,
    )

    point_cloud = make_point_cloud(sampled_points, sampled_normals, sampled_colors)

    print(f"Input surfels: {points.shape[0]}")
    print(f"Poisson samples: {len(point_cloud.points)}")
    print(f"Normal source: {normal_source}")
    print(f"Sampling: {sampling_source}")

    if args.save_samples:
        samples_path = mesh_dir / f"poisson_samples{mesh_name_suffix}.ply"
        o3d.io.write_point_cloud(str(samples_path), point_cloud, write_ascii=True)
        print(f"Poisson sample point cloud saved at {samples_path}")

    if args.split_components:
        components = split_point_cloud_components(
            point_cloud=point_cloud,
            component_eps=args.component_eps,
            component_min_points=args.component_min_points,
        )

        if not components:
            raise RuntimeError("No point-cloud components found before Poisson reconstruction. Increase --component-eps or lower --component-min-points.")

        print(f"Running component-wise Poisson reconstruction for {len(components)} components ...")
        component_meshes: list[o3d.geometry.TriangleMesh] = []

        for component_index, component_point_cloud in enumerate(components):
            print(f"Component {component_index}: {len(component_point_cloud.points)} points")
            component_mesh = reconstruct_poisson_component(
                point_cloud=component_point_cloud,
                poisson_depth=args.poisson_depth,
                poisson_scale=args.poisson_scale,
                density_quantile=args.density_quantile,
                bbox_padding_factor=args.bbox_padding_factor,
            )
            component_meshes.append(component_mesh)

        mesh = merge_meshes(component_meshes)
    else:
        print("Running Poisson reconstruction ...")
        mesh = reconstruct_poisson_component(
            point_cloud=point_cloud,
            poisson_depth=args.poisson_depth,
            poisson_scale=args.poisson_scale,
            density_quantile=args.density_quantile,
            bbox_padding_factor=args.bbox_padding_factor,
        )

    mesh_path = mesh_dir / f"poisson{mesh_name_suffix}.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"Poisson mesh saved at {mesh_path}")

    mesh_post = post_process_mesh(mesh, cluster_count_to_keep=args.num_cluster)
    mesh_post_path = mesh_dir / f"poisson_post{mesh_name_suffix}.ply"
    o3d.io.write_triangle_mesh(str(mesh_post_path), mesh_post)
    print(f"Poisson post processed mesh saved at {mesh_post_path}")


if __name__ == "__main__":
    main()