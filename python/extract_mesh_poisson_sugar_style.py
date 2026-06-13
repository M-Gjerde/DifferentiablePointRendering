from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poisson mesh extraction from optimized quaternion surfel primitives.")
    parser.add_argument("--output_root", type=Path, default=Path("../Assets/OptimizationOutput"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--input-path", type=Path, default=None)

    parser.add_argument("--samples-per-surfel", type=int, default=16)
    parser.add_argument("--area-weighted-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--area-sample-power", type=float, default=1.0)
    parser.add_argument("--min-samples-per-surfel", type=int, default=1)
    parser.add_argument("--max-samples-per-surfel", type=int, default=512)
    parser.add_argument("--area-reference-quantile", type=float, default=0.50)

    parser.add_argument("--correct-normal-flips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normal-neighbor-count", type=int, default=32)
    parser.add_argument("--normal-flip-threshold", type=float, default=0.0)

    parser.add_argument("--smooth-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normal-smoothing-mode", type=str, default="pca", choices=["pca", "bilateral"])
    parser.add_argument("--normal-smoothing-neighbor-count", type=int, default=32)
    parser.add_argument("--normal-smoothing-radius-factor", type=float, default=3.0)
    parser.add_argument("--normal-smoothing-max-angle-deg", type=float, default=35.0)
    parser.add_argument("--normal-smoothing-plane-sigma-factor", type=float, default=0.5)
    parser.add_argument("--normal-smoothing-distance-sigma-factor", type=float, default=1.0)
    parser.add_argument("--normal-smoothing-iterations", type=int, default=5)

    parser.add_argument("--poisson-depth", type=int, default=11)
    parser.add_argument("--poisson-scale", type=float, default=1.1)
    parser.add_argument("--density-quantile", type=float, default=0.1)
    parser.add_argument("--bbox-padding-factor", type=float, default=0.05)

    parser.add_argument("--num-cluster", type=int, default=50)
    parser.add_argument("--split-components", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--component-eps", type=float, default=0.05)
    parser.add_argument("--component-min-points", type=int, default=50)

    parser.add_argument("--save-samples", action="store_true")

    args = parser.parse_args()

    if args.samples_per_surfel <= 0:
        raise ValueError("--samples-per-surfel must be greater than 0")

    if args.normal_neighbor_count < 0:
        raise ValueError("--normal-neighbor-count must be >= 0")

    if args.normal_smoothing_neighbor_count < 1:
        raise ValueError("--normal-smoothing-neighbor-count must be >= 1")

    if args.normal_smoothing_iterations < 1:
        raise ValueError("--normal-smoothing-iterations must be >= 1")

    if args.normal_smoothing_radius_factor <= 0.0:
        raise ValueError("--normal-smoothing-radius-factor must be > 0")

    if args.normal_smoothing_plane_sigma_factor <= 0.0:
        raise ValueError("--normal-smoothing-plane-sigma-factor must be > 0")

    if args.normal_smoothing_distance_sigma_factor <= 0.0:
        raise ValueError("--normal-smoothing-distance-sigma-factor must be > 0")

    if args.normal_smoothing_max_angle_deg <= 0.0 or args.normal_smoothing_max_angle_deg >= 180.0:
        raise ValueError("--normal-smoothing-max-angle-deg must be in the range (0, 180)")

    return args


def get_latest_output_dir(output_root: Path, index: int) -> Path:
    output_root = output_root.expanduser().resolve()

    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    output_dirs = [path for path in output_root.iterdir() if path.is_dir()]

    if not output_dirs:
        raise FileNotFoundError(f"No output directories found in: {output_root}")

    output_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    if index < 0 or index >= len(output_dirs):
        raise IndexError(f"Output index {index} is outside available range [0, {len(output_dirs) - 1}]")

    return output_dirs[index]


def find_input_ply(output_dir: Path) -> Path:
    search_dirs = [output_dir / "mesh", output_dir]
    preferred_names = ["points_final.ply", "initial_points.ply", "points.ply", "point_cloud.ply", "optimized_points.ply", "surfels.ply"]

    candidates: list[Path] = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for preferred_name in preferred_names:
            preferred_path = search_dir / preferred_name
            if preferred_path.exists():
                candidates.append(preferred_path)

        candidates.extend(search_dir.glob("*points*.ply"))
        candidates.extend(search_dir.glob("*surfels*.ply"))

    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
    candidates = [
        path for path in candidates
        if "sample" not in path.stem.lower()
        and "poisson" not in path.stem.lower()
        and "post" not in path.stem.lower()
    ]

    if not candidates:
        raise FileNotFoundError(f"Could not find an input point/surfel PLY in: {output_dir}")

    return candidates[0]


def read_vertex_properties(ply_path: Path) -> dict[str, np.ndarray]:
    try:
        from plyfile import PlyData

        ply_data = PlyData.read(str(ply_path))
        vertex_data = ply_data["vertex"].data

        return {name: np.asarray(vertex_data[name]) for name in vertex_data.dtype.names}
    except ImportError:
        return read_ascii_vertex_properties(ply_path)
    except Exception as exception:
        raise RuntimeError(f"Failed to read PLY file: {ply_path}") from exception


def read_ascii_vertex_properties(ply_path: Path) -> dict[str, np.ndarray]:
    with ply_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    property_names: list[str] = []
    vertex_count = None
    header_end_index = None
    inside_vertex_element = False

    for line_index, line in enumerate(lines):
        stripped_line = line.strip()

        if stripped_line.startswith("element vertex"):
            vertex_count = int(stripped_line.split()[-1])
            inside_vertex_element = True
        elif stripped_line.startswith("element "):
            inside_vertex_element = False
        elif inside_vertex_element and stripped_line.startswith("property"):
            property_names.append(stripped_line.split()[-1])
        elif stripped_line == "end_header":
            header_end_index = line_index + 1
            break

    if vertex_count is None or header_end_index is None:
        raise ValueError(f"Invalid ASCII PLY header: {ply_path}")

    vertex_lines = lines[header_end_index:header_end_index + vertex_count]
    vertex_array = np.loadtxt(vertex_lines, dtype=np.float32)

    if vertex_array.ndim == 1:
        vertex_array = vertex_array[None, :]

    return {
        name: vertex_array[:, property_index]
        for property_index, name in enumerate(property_names)
    }


def require_property(properties: dict[str, np.ndarray], names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in properties:
            return np.asarray(properties[name], dtype=np.float32)

    raise KeyError(f"Missing required property. Tried: {list(names)}")


def optional_property(properties: dict[str, np.ndarray], names: Iterable[str], default_value: float, count: int) -> np.ndarray:
    for name in names:
        if name in properties:
            return np.asarray(properties[name], dtype=np.float32)

    return np.full(count, default_value, dtype=np.float32)


def normalize_vectors(vectors: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector_lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid_mask = vector_lengths[:, 0] > 1e-12

    normalized_vectors = np.zeros_like(vectors, dtype=np.float32)
    normalized_vectors[valid_mask] = vectors[valid_mask] / vector_lengths[valid_mask]

    if fallback is not None:
        normalized_vectors[~valid_mask] = fallback[~valid_mask]

    return normalized_vectors


def normalize_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    quaternion_lengths = np.linalg.norm(quaternions, axis=1, keepdims=True)
    valid_mask = quaternion_lengths[:, 0] > 1e-12

    normalized_quaternions = np.zeros_like(quaternions, dtype=np.float32)
    normalized_quaternions[valid_mask] = quaternions[valid_mask] / quaternion_lengths[valid_mask]
    normalized_quaternions[~valid_mask] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    return normalized_quaternions


def quaternion_wxyz_to_frame(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quaternions = normalize_quaternions_wxyz(quaternions)

    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    tangent_u = np.empty((quaternions.shape[0], 3), dtype=np.float32)
    tangent_v = np.empty((quaternions.shape[0], 3), dtype=np.float32)
    normals = np.empty((quaternions.shape[0], 3), dtype=np.float32)

    tangent_u[:, 0] = 1.0 - 2.0 * (y * y + z * z)
    tangent_u[:, 1] = 2.0 * (x * y + w * z)
    tangent_u[:, 2] = 2.0 * (x * z - w * y)

    tangent_v[:, 0] = 2.0 * (x * y - w * z)
    tangent_v[:, 1] = 1.0 - 2.0 * (x * x + z * z)
    tangent_v[:, 2] = 2.0 * (y * z + w * x)

    normals[:, 0] = 2.0 * (x * z + w * y)
    normals[:, 1] = 2.0 * (y * z - w * x)
    normals[:, 2] = 1.0 - 2.0 * (x * x + y * y)

    return normalize_vectors(tangent_u), normalize_vectors(tangent_v), normalize_vectors(normals)


def compute_samples_per_surfel(
    scales: np.ndarray,
    base_samples_per_surfel: int,
    area_weighted_sampling: bool,
    area_sample_power: float,
    min_samples_per_surfel: int,
    max_samples_per_surfel: int,
    area_reference_quantile: float,
) -> np.ndarray:
    surfel_count = int(scales.shape[0])

    if not area_weighted_sampling:
        return np.full(surfel_count, int(base_samples_per_surfel), dtype=np.int32)

    surfel_areas = np.maximum(scales[:, 0] * scales[:, 1], 1.0e-12)
    reference_area = float(np.quantile(surfel_areas, area_reference_quantile))
    reference_area = max(reference_area, 1.0e-12)

    relative_area = surfel_areas / reference_area
    sample_counts = float(base_samples_per_surfel) * np.power(relative_area, float(area_sample_power))
    sample_counts = np.ceil(sample_counts).astype(np.int32)
    sample_counts = np.clip(sample_counts, int(min_samples_per_surfel), int(max_samples_per_surfel))

    return sample_counts


def sample_unit_disk(sample_count: int) -> np.ndarray:
    if sample_count == 1:
        return np.zeros((1, 2), dtype=np.float32)

    sample_indices = np.arange(sample_count, dtype=np.float32)

    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    radii = np.sqrt((sample_indices + 0.5) / float(sample_count))
    angles = sample_indices * golden_angle

    samples = np.stack(
        [
            radii * np.cos(angles),
            radii * np.sin(angles),
        ],
        axis=1,
    ).astype(np.float32)

    samples[0, :] = 0.0

    return samples


def load_surfels(properties: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.stack(
        [
            require_property(properties, ["x"]),
            require_property(properties, ["y"]),
            require_property(properties, ["z"]),
        ],
        axis=1,
    ).astype(np.float32)

    quaternions = np.stack(
        [
            require_property(properties, ["rot_w"]),
            require_property(properties, ["rot_x"]),
            require_property(properties, ["rot_y"]),
            require_property(properties, ["rot_z"]),
        ],
        axis=1,
    ).astype(np.float32)

    surfel_count = positions.shape[0]

    scale_u = optional_property(properties, ["su", "scale_u", "scale_x", "scale_0"], 1.0, surfel_count)
    scale_v = optional_property(properties, ["sv", "scale_v", "scale_y", "scale_1"], 1.0, surfel_count)
    opacity = optional_property(properties, ["opacity"], 1.0, surfel_count)

    albedo_r = optional_property(properties, ["albedo_r", "red", "r"], 0.7, surfel_count)
    albedo_g = optional_property(properties, ["albedo_g", "green", "g"], 0.7, surfel_count)
    albedo_b = optional_property(properties, ["albedo_b", "blue", "b"], 0.7, surfel_count)

    colors = np.stack([albedo_r, albedo_g, albedo_b], axis=1).astype(np.float32)

    if float(np.nanmax(colors)) > 1.0:
        colors = colors / 255.0

    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)

    tangent_u, tangent_v, normals = quaternion_wxyz_to_frame(quaternions)

    valid_mask = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(quaternions).all(axis=1)
        & np.isfinite(tangent_u).all(axis=1)
        & np.isfinite(tangent_v).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & np.isfinite(scale_u)
        & np.isfinite(scale_v)
        & np.isfinite(opacity)
        & np.isfinite(colors).all(axis=1)
        & (scale_u > 0.0)
        & (scale_v > 0.0)
        & (opacity > 0.0)
    )

    scales = np.stack([scale_u, scale_v], axis=1).astype(np.float32)

    return (
        positions[valid_mask],
        tangent_u[valid_mask],
        tangent_v[valid_mask],
        normals[valid_mask],
        scales[valid_mask],
        colors[valid_mask],
    )


def correct_isolated_normal_flips(
    positions: np.ndarray,
    normals: np.ndarray,
    neighbor_count: int,
    flip_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if positions.shape[0] <= 1 or neighbor_count <= 0:
        return normals.astype(np.float32), np.zeros(positions.shape[0], dtype=bool)

    corrected_normals = np.asarray(normals, dtype=np.float32).copy()
    normal_lengths = np.linalg.norm(corrected_normals, axis=1, keepdims=True)
    corrected_normals = corrected_normals / np.maximum(normal_lengths, 1.0e-12)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64))

    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    effective_neighbor_count = min(int(neighbor_count), int(positions.shape[0]) - 1)
    flip_mask = np.zeros(int(positions.shape[0]), dtype=bool)

    for surfel_index in range(int(positions.shape[0])):
        _, neighbor_indices, _ = kdtree.search_knn_vector_3d(
            point_cloud.points[surfel_index],
            effective_neighbor_count + 1,
        )

        neighbor_indices = [int(index) for index in neighbor_indices if int(index) != surfel_index]

        if not neighbor_indices:
            continue

        neighborhood_normal = np.mean(corrected_normals[neighbor_indices], axis=0)
        neighborhood_normal_length = float(np.linalg.norm(neighborhood_normal))

        if neighborhood_normal_length <= 1.0e-12:
            continue

        neighborhood_normal = neighborhood_normal / neighborhood_normal_length
        local_alignment = float(np.dot(corrected_normals[surfel_index], neighborhood_normal))

        if local_alignment < -abs(float(flip_threshold)):
            flip_mask[surfel_index] = True

    corrected_normals[flip_mask] *= -1.0

    return corrected_normals.astype(np.float32), flip_mask


def estimate_reference_radius_from_scales(scales: np.ndarray, radius_factor: float) -> float:
    effective_radii = np.sqrt(np.maximum(scales[:, 0] * scales[:, 1], 1.0e-12))
    reference_radius = float(np.median(effective_radii))
    return max(reference_radius * float(radius_factor), 1.0e-8)


def weighted_pca_normal(
    neighbor_positions: np.ndarray,
    neighbor_weights: np.ndarray,
    fallback_normal: np.ndarray,
) -> np.ndarray:
    weight_sum = float(np.sum(neighbor_weights))

    if weight_sum <= 1.0e-12 or neighbor_positions.shape[0] < 3:
        return fallback_normal.astype(np.float32)

    weighted_center = np.sum(neighbor_positions * neighbor_weights[:, None], axis=0) / weight_sum
    centered_positions = neighbor_positions - weighted_center[None, :]

    covariance = (
        centered_positions.T
        @ (centered_positions * neighbor_weights[:, None])
    ) / weight_sum

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return fallback_normal.astype(np.float32)

    normal = eigenvectors[:, int(np.argmin(eigenvalues))].astype(np.float32)
    normal_length = float(np.linalg.norm(normal))

    if normal_length <= 1.0e-12:
        return fallback_normal.astype(np.float32)

    normal /= normal_length

    if float(np.dot(normal, fallback_normal)) < 0.0:
        normal *= -1.0

    return normal.astype(np.float32)


def smooth_surfel_normals_same_layer(
    positions: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray,
    neighbor_count: int,
    radius_factor: float,
    max_angle_degrees: float,
    plane_sigma_factor: float,
    distance_sigma_factor: float,
    mode: str,
    iterations: int,
) -> np.ndarray:
    if positions.shape[0] <= 1:
        return normals.astype(np.float32)

    smoothed_normals = normalize_vectors(normals.astype(np.float32), fallback=normals.astype(np.float32))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64))

    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    reference_radius = estimate_reference_radius_from_scales(scales, radius_factor)
    distance_sigma = max(reference_radius * float(distance_sigma_factor), 1.0e-8)
    plane_sigma = max(reference_radius * float(plane_sigma_factor), 1.0e-8)

    max_angle_cosine = float(np.cos(np.deg2rad(float(max_angle_degrees))))
    effective_neighbor_count = min(int(neighbor_count), int(positions.shape[0]) - 1)

    for _ in range(int(iterations)):
        previous_normals = smoothed_normals.copy()
        next_normals = previous_normals.copy()

        for surfel_index in range(int(positions.shape[0])):
            query_position = positions[surfel_index]
            query_normal = previous_normals[surfel_index]

            _, neighbor_indices, _ = kdtree.search_knn_vector_3d(
                point_cloud.points[surfel_index],
                effective_neighbor_count + 1,
            )

            neighbor_indices = np.asarray(
                [int(index) for index in neighbor_indices if int(index) != surfel_index],
                dtype=np.int32,
            )

            if neighbor_indices.size == 0:
                continue

            neighbor_positions = positions[neighbor_indices]
            neighbor_normals = previous_normals[neighbor_indices]

            position_offsets = neighbor_positions - query_position[None, :]
            distances_squared = np.sum(position_offsets * position_offsets, axis=1)

            normal_alignment = neighbor_normals @ query_normal
            same_orientation_mask = normal_alignment > max_angle_cosine

            point_to_plane_offsets = position_offsets @ query_normal
            same_layer_mask = np.abs(point_to_plane_offsets) < (3.0 * plane_sigma)

            valid_mask = same_orientation_mask & same_layer_mask

            if np.count_nonzero(valid_mask) < 2:
                continue

            valid_positions = neighbor_positions[valid_mask]
            valid_normals = neighbor_normals[valid_mask]
            valid_distances_squared = distances_squared[valid_mask]
            valid_normal_alignment = np.clip(normal_alignment[valid_mask], 0.0, 1.0)
            valid_plane_offsets = point_to_plane_offsets[valid_mask]

            distance_weights = np.exp(
                -valid_distances_squared / (2.0 * distance_sigma * distance_sigma)
            )

            normal_weights = valid_normal_alignment * valid_normal_alignment

            plane_weights = np.exp(
                -(valid_plane_offsets * valid_plane_offsets) / (2.0 * plane_sigma * plane_sigma)
            )

            weights = distance_weights * normal_weights * plane_weights
            weight_sum = float(np.sum(weights))

            if weight_sum <= 1.0e-12:
                continue

            if mode == "bilateral":
                normal = np.sum(valid_normals * weights[:, None], axis=0)
                normal_length = float(np.linalg.norm(normal))

                if normal_length <= 1.0e-12:
                    continue

                normal = normal / normal_length

                if float(np.dot(normal, query_normal)) < 0.0:
                    normal *= -1.0

                next_normals[surfel_index] = normal.astype(np.float32)

            elif mode == "pca":
                pca_positions = np.concatenate(
                    [query_position[None, :], valid_positions],
                    axis=0,
                )

                pca_weights = np.concatenate(
                    [np.array([weight_sum], dtype=np.float32), weights.astype(np.float32)],
                    axis=0,
                )

                next_normals[surfel_index] = weighted_pca_normal(
                    neighbor_positions=pca_positions,
                    neighbor_weights=pca_weights,
                    fallback_normal=query_normal,
                )
            else:
                raise ValueError(f"Unsupported normal smoothing mode: {mode}")

        smoothed_normals = normalize_vectors(next_normals, fallback=previous_normals)

    return smoothed_normals.astype(np.float32)


def sample_surfels(
    positions: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples_per_surfel = compute_samples_per_surfel(
        scales=scales,
        base_samples_per_surfel=args.samples_per_surfel,
        area_weighted_sampling=args.area_weighted_sampling,
        area_sample_power=args.area_sample_power,
        min_samples_per_surfel=args.min_samples_per_surfel,
        max_samples_per_surfel=args.max_samples_per_surfel,
        area_reference_quantile=args.area_reference_quantile,
    )

    sample_points_list: list[np.ndarray] = []
    sample_normals_list: list[np.ndarray] = []
    sample_colors_list: list[np.ndarray] = []

    for surfel_index in range(int(positions.shape[0])):
        current_sample_count = int(samples_per_surfel[surfel_index])
        sample_uv = sample_unit_disk(current_sample_count)

        sample_points = (
            positions[surfel_index][None, :]
            + tangent_u[surfel_index][None, :] * (sample_uv[:, 0:1] * scales[surfel_index, 0])
            + tangent_v[surfel_index][None, :] * (sample_uv[:, 1:2] * scales[surfel_index, 1])
        )

        sample_normals = np.repeat(normals[surfel_index][None, :], current_sample_count, axis=0)
        sample_colors = np.repeat(colors[surfel_index][None, :], current_sample_count, axis=0)

        sample_points_list.append(sample_points.astype(np.float32))
        sample_normals_list.append(sample_normals.astype(np.float32))
        sample_colors_list.append(sample_colors.astype(np.float32))

    sample_points = np.concatenate(sample_points_list, axis=0)
    sample_normals = np.concatenate(sample_normals_list, axis=0)
    sample_colors = np.concatenate(sample_colors_list, axis=0)

    return sample_points, sample_normals, sample_colors, samples_per_surfel


def create_point_cloud(points: np.ndarray, normals: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    point_cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return point_cloud


def filter_sample_components(point_cloud: o3d.geometry.PointCloud, args: argparse.Namespace) -> o3d.geometry.PointCloud:
    if not args.split_components:
        return point_cloud

    labels = np.asarray(
        point_cloud.cluster_dbscan(
            eps=args.component_eps,
            min_points=args.component_min_points,
            print_progress=False,
        )
    )

    if labels.size == 0 or np.max(labels) < 0:
        return point_cloud

    unique_labels, label_counts = np.unique(labels[labels >= 0], return_counts=True)
    sorted_labels = unique_labels[np.argsort(label_counts)[::-1]]
    kept_labels = set(int(label) for label in sorted_labels[:args.num_cluster])
    kept_indices = np.where(np.isin(labels, list(kept_labels)))[0]

    return point_cloud.select_by_index(kept_indices)


def remove_low_density_vertices(
    mesh: o3d.geometry.TriangleMesh,
    densities: np.ndarray,
    density_quantile: float,
) -> o3d.geometry.TriangleMesh:
    if density_quantile <= 0.0:
        return mesh

    density_threshold = np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(densities < density_threshold)
    mesh.remove_unreferenced_vertices()

    return mesh


def crop_to_source_bbox(
    mesh: o3d.geometry.TriangleMesh,
    source_points: np.ndarray,
    padding_factor: float,
) -> o3d.geometry.TriangleMesh:
    min_bound = source_points.min(axis=0)
    max_bound = source_points.max(axis=0)

    diagonal = max_bound - min_bound
    padding = np.maximum(diagonal * padding_factor, 1e-6)

    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound - padding, max_bound + padding)

    return mesh.crop(bounding_box)



def transfer_point_cloud_colors_to_mesh(
    mesh: o3d.geometry.TriangleMesh,
    point_cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:
    if not point_cloud.has_colors():
        return mesh

    mesh_vertices = np.asarray(mesh.vertices)

    if mesh_vertices.size == 0:
        return mesh

    point_colors = np.asarray(point_cloud.colors)

    if point_colors.size == 0:
        return mesh

    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    vertex_colors = np.zeros((mesh_vertices.shape[0], 3), dtype=np.float64)

    for vertex_index, vertex in enumerate(mesh_vertices):
        _, neighbor_indices, _ = kdtree.search_knn_vector_3d(vertex.astype(np.float64), 1)

        if len(neighbor_indices) == 0:
            vertex_colors[vertex_index] = np.array([0.7, 0.7, 0.7], dtype=np.float64)
        else:
            vertex_colors[vertex_index] = point_colors[int(neighbor_indices[0])]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return mesh


def keep_largest_mesh_components(
    mesh: o3d.geometry.TriangleMesh,
    max_components: int,
    min_triangles: int,
) -> o3d.geometry.TriangleMesh:
    if max_components <= 0:
        return mesh

    triangle_labels, triangle_counts, _ = mesh.cluster_connected_triangles()
    triangle_labels = np.asarray(triangle_labels)
    triangle_counts = np.asarray(triangle_counts)

    if triangle_labels.size == 0 or triangle_counts.size == 0:
        return mesh

    sorted_component_indices = np.argsort(triangle_counts)[::-1]
    kept_component_indices = {
        int(index)
        for index in sorted_component_indices[:max_components]
        if triangle_counts[index] >= min_triangles
    }

    if not kept_component_indices:
        return mesh

    remove_mask = np.array(
        [int(label) not in kept_component_indices for label in triangle_labels],
        dtype=bool,
    )

    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    return mesh


def main() -> None:
    args = parse_args()

    output_dir = get_latest_output_dir(args.output_root, args.index)
    input_path = args.input_path.expanduser().resolve() if args.input_path is not None else find_input_ply(output_dir)

    mesh_dir = output_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using output directory: {output_dir}")
    print(f"Using input PLY: {input_path}")

    properties = read_vertex_properties(input_path)
    positions, tangent_u, tangent_v, normals, scales, colors = load_surfels(properties)

    print(f"Loaded surfels: {positions.shape[0]}")
    print(f"Samples per surfel: {args.samples_per_surfel}")

    if args.correct_normal_flips:
        normals, normal_flip_mask = correct_isolated_normal_flips(
            positions=positions,
            normals=normals,
            neighbor_count=args.normal_neighbor_count,
            flip_threshold=args.normal_flip_threshold,
        )

        print(
            f"Corrected isolated normal flips: {int(np.count_nonzero(normal_flip_mask))}/{positions.shape[0]} "
            f"| neighbor_count={args.normal_neighbor_count} "
            f"| flip_threshold={args.normal_flip_threshold}"
        )
    else:
        print("Normal flip correction disabled.")

    if args.smooth_normals:
        normals_before_smoothing = normals.copy()

        normals = smooth_surfel_normals_same_layer(
            positions=positions,
            normals=normals,
            scales=scales,
            neighbor_count=args.normal_smoothing_neighbor_count,
            radius_factor=args.normal_smoothing_radius_factor,
            max_angle_degrees=args.normal_smoothing_max_angle_deg,
            plane_sigma_factor=args.normal_smoothing_plane_sigma_factor,
            distance_sigma_factor=args.normal_smoothing_distance_sigma_factor,
            mode=args.normal_smoothing_mode,
            iterations=args.normal_smoothing_iterations,
        )

        normal_cosines = np.sum(normals_before_smoothing * normals, axis=1)
        normal_cosines = np.clip(normal_cosines, -1.0, 1.0)
        normal_angle_changes = np.rad2deg(np.arccos(normal_cosines))

        print(
            "Smoothed normals: "
            f"mode={args.normal_smoothing_mode} "
            f"| iterations={args.normal_smoothing_iterations} "
            f"| neighbor_count={args.normal_smoothing_neighbor_count} "
            f"| max_angle_deg={args.normal_smoothing_max_angle_deg} "
            f"| median_change_deg={float(np.median(normal_angle_changes)):.3f} "
            f"| p95_change_deg={float(np.quantile(normal_angle_changes, 0.95)):.3f} "
            f"| max_change_deg={float(np.max(normal_angle_changes)):.3f}"
        )
    else:
        print("Normal smoothing disabled.")

    sample_points, sample_normals, sample_colors, samples_per_surfel = sample_surfels(
        positions=positions,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        normals=normals,
        scales=scales,
        colors=colors,
        args=args,
    )

    print(
        "Sample count per surfel: "
        f"min={int(samples_per_surfel.min())}, "
        f"median={float(np.median(samples_per_surfel)):.1f}, "
        f"mean={float(np.mean(samples_per_surfel)):.1f}, "
        f"max={int(samples_per_surfel.max())}, "
        f"total={int(np.sum(samples_per_surfel))}"
    )

    point_cloud = create_point_cloud(sample_points, sample_normals, sample_colors)

    print(f"Generated samples: {np.asarray(point_cloud.points).shape[0]}")

    if args.save_samples:
        sample_path = mesh_dir / "poisson_samples.ply"
        o3d.io.write_point_cloud(str(sample_path), point_cloud, write_ascii=False)
        print(f"Saved samples: {sample_path}")

    point_cloud = filter_sample_components(point_cloud, args)

    print(f"Samples after component filtering: {np.asarray(point_cloud.points).shape[0]}")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=args.poisson_depth,
        scale=args.poisson_scale,
    )

    mesh = remove_low_density_vertices(mesh, np.asarray(densities), args.density_quantile)
    mesh = crop_to_source_bbox(mesh, positions, args.bbox_padding_factor)
    mesh = keep_largest_mesh_components(mesh, args.num_cluster, args.component_min_points)
    mesh = clean_mesh(mesh)
    mesh = transfer_point_cloud_colors_to_mesh(mesh, point_cloud)

    output_path = mesh_dir / "poisson_post.ply"
    o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False)

    print(f"Saved mesh: {output_path}")
    print(f"Vertices: {np.asarray(mesh.vertices).shape[0]}")
    print(f"Triangles: {np.asarray(mesh.triangles).shape[0]}")


if __name__ == "__main__":
    main()