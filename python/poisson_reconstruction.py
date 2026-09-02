"""Opacity-measure surfel sampling and Screened Poisson reconstruction.

For a surfel with normalized elliptical radius ``r``, PBDR evaluates the
geometric opacity profile ``(1-r^2)^(4 exp(beta))``.  This module samples that
same surface measure, including the ellipse area Jacobian and learned opacity,
and supplies the resulting oriented points to Open3D's Screened Poisson solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class QuaternionSurfelCloud:
    positions: np.ndarray
    quaternions_wxyz: np.ndarray
    scale_u: np.ndarray
    scale_v: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    betas: np.ndarray
    source_indices: np.ndarray

    @property
    def surfel_count(self) -> int:
        return int(self.positions.shape[0])


@dataclass(frozen=True)
class PoissonSamplingSettings:
    sample_count: int = 500_000
    seed: int = 0
    minimum_samples_per_surfel: int = 0
    use_beta_profile: bool = True


@dataclass(frozen=True)
class PoissonReconstructionSettings:
    depth: int = 8
    scale: float = 1.1
    linear_fit: bool = True
    n_threads: int = -1
    density_quantile: float = 0.01
    coverage_trim_cells: float = 4.0
    crop_padding_cells: float = 4.0


def _read_ascii_ply_vertices(points_path: Path) -> tuple[dict[str, np.ndarray], int]:
    points_path = points_path.expanduser().resolve()
    property_names: list[str] = []
    vertex_count: int | None = None
    current_element: str | None = None
    header_line_count = 0
    ply_format: str | None = None

    with points_path.open("r", encoding="ascii") as file_handle:
        for line in file_handle:
            header_line_count += 1
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "format":
                if len(parts) < 2:
                    raise ValueError(f"Malformed PLY format declaration in {points_path}")
                ply_format = parts[1]
            elif parts[0] == "element":
                if len(parts) < 3:
                    raise ValueError(f"Malformed PLY element declaration in {points_path}")
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError(
                        f"Only scalar vertex properties are supported in {points_path}: {line.strip()}"
                    )
                property_names.append(parts[2])
            elif parts[0] == "end_header":
                break
        else:
            raise ValueError(f"Missing end_header in {points_path}")

    if ply_format != "ascii":
        raise ValueError(
            f"Poisson surfel extraction currently requires an ASCII PLY, got {ply_format!r}"
        )
    if vertex_count is None or vertex_count <= 0:
        raise ValueError(f"PLY has no non-empty vertex element: {points_path}")
    if not property_names:
        raise ValueError(f"PLY vertex element has no scalar properties: {points_path}")

    vertex_table = np.loadtxt(
        points_path,
        dtype=np.float64,
        skiprows=header_line_count,
        max_rows=vertex_count,
        ndmin=2,
    )
    if vertex_table.shape != (vertex_count, len(property_names)):
        raise ValueError(
            f"PLY vertex table shape mismatch in {points_path}: got {vertex_table.shape}, "
            f"expected {(vertex_count, len(property_names))}"
        )

    return {
        name: vertex_table[:, property_index]
        for property_index, name in enumerate(property_names)
    }, vertex_count


def _require_properties(
    properties: dict[str, np.ndarray],
    names: Sequence[str],
    points_path: Path,
) -> list[np.ndarray]:
    missing = [name for name in names if name not in properties]
    if missing:
        raise ValueError(
            f"Quaternion surfel PLY {points_path} is missing properties: {', '.join(missing)}"
        )
    return [properties[name] for name in names]


def normalize_quaternions_wxyz(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Expected quaternions with shape (N,4), got {quaternions.shape}")

    norms = np.linalg.norm(quaternions, axis=1)
    valid = np.isfinite(quaternions).all(axis=1) & np.isfinite(norms) & (norms > 1.0e-12)
    normalized = np.zeros_like(quaternions)
    normalized[:, 0] = 1.0
    normalized[valid] = quaternions[valid] / norms[valid, None]
    normalized[normalized[:, 0] < 0.0] *= -1.0
    return normalized, valid


def quaternion_tangent_frames_wxyz(
    quaternions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, valid = normalize_quaternions_wxyz(quaternions)
    if not np.all(valid):
        raise ValueError("Cannot construct tangent frames from invalid quaternions")

    qw, qx, qy, qz = q.T
    tangent_u = np.column_stack(
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy + qz * qw),
            2.0 * (qx * qz - qy * qw),
        )
    )
    tangent_v = np.column_stack(
        (
            2.0 * (qx * qy - qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz + qx * qw),
        )
    )

    tangent_u /= np.maximum(np.linalg.norm(tangent_u, axis=1, keepdims=True), 1.0e-12)
    tangent_v -= np.sum(tangent_v * tangent_u, axis=1, keepdims=True) * tangent_u
    tangent_v /= np.maximum(np.linalg.norm(tangent_v, axis=1, keepdims=True), 1.0e-12)
    normals = np.cross(tangent_u, tangent_v)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    return tangent_u, tangent_v, normals


def load_quaternion_surfel_cloud(
    points_path: Path,
    opacity_threshold: float = 1.0e-3,
    emitter_power_epsilon: float = 1.0e-8,
) -> tuple[QuaternionSurfelCloud, dict[str, Any]]:
    if not math.isfinite(opacity_threshold) or opacity_threshold < 0.0:
        raise ValueError(
            f"opacity_threshold must be finite and non-negative, got {opacity_threshold}"
        )
    if not math.isfinite(emitter_power_epsilon) or emitter_power_epsilon < 0.0:
        raise ValueError(
            "emitter_power_epsilon must be finite and non-negative, got "
            f"{emitter_power_epsilon}"
        )

    points_path = points_path.expanduser().resolve()
    properties, vertex_count = _read_ascii_ply_vertices(points_path)

    position_columns = _require_properties(properties, ("x", "y", "z"), points_path)
    quaternion_columns = _require_properties(
        properties, ("rot_w", "rot_x", "rot_y", "rot_z"), points_path
    )
    scale_u, scale_v = _require_properties(properties, ("su", "sv"), points_path)
    color_columns = _require_properties(
        properties, ("albedo_r", "albedo_g", "albedo_b"), points_path
    )
    opacity, beta, power = _require_properties(
        properties, ("opacity", "beta", "power"), points_path
    )

    positions = np.column_stack(position_columns)
    quaternions = np.column_stack(quaternion_columns)
    colors = np.column_stack(color_columns)
    normalized_quaternions, valid_quaternion = normalize_quaternions_wxyz(quaternions)

    finite = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(scale_u)
        & np.isfinite(scale_v)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(opacity)
        & np.isfinite(beta)
        & np.isfinite(power)
        & valid_quaternion
    )
    positive_scale = (scale_u > 1.0e-12) & (scale_v > 1.0e-12)
    non_emissive = np.abs(power) <= float(emitter_power_epsilon)
    opaque_enough = opacity >= float(opacity_threshold)
    keep = finite & positive_scale & non_emissive & opaque_enough

    if not np.any(keep):
        raise RuntimeError(
            f"No reconstructable surfels remain after filtering {points_path}; "
            "check opacity and emitter thresholds."
        )

    cloud = QuaternionSurfelCloud(
        positions=np.ascontiguousarray(positions[keep], dtype=np.float64),
        quaternions_wxyz=np.ascontiguousarray(normalized_quaternions[keep], dtype=np.float64),
        scale_u=np.ascontiguousarray(scale_u[keep], dtype=np.float64),
        scale_v=np.ascontiguousarray(scale_v[keep], dtype=np.float64),
        colors=np.ascontiguousarray(np.clip(colors[keep], 0.0, 1.0), dtype=np.float64),
        opacities=np.ascontiguousarray(np.clip(opacity[keep], 0.0, 1.0), dtype=np.float64),
        betas=np.ascontiguousarray(beta[keep], dtype=np.float64),
        source_indices=np.flatnonzero(keep).astype(np.int64),
    )
    report: dict[str, Any] = {
        "input_vertices": int(vertex_count),
        "kept_surfels": cloud.surfel_count,
        "invalid_or_nonfinite": int(np.count_nonzero(~finite)),
        "nonpositive_scale": int(np.count_nonzero(finite & ~positive_scale)),
        "emissive": int(np.count_nonzero(finite & positive_scale & ~non_emissive)),
        "below_opacity_threshold": int(
            np.count_nonzero(finite & positive_scale & non_emissive & ~opaque_enough)
        ),
        "opacity_threshold": float(opacity_threshold),
        "emitter_power_epsilon": float(emitter_power_epsilon),
    }
    return cloud, report


def _vectors_to_nearest_cameras(
    positions: np.ndarray,
    camera_locations: np.ndarray,
    chunk_size: int = 32_768,
) -> np.ndarray:
    """Return camera-position minus point without materializing an NxCx3 array."""
    nearest_vectors = np.empty_like(positions)
    camera_count = camera_locations.shape[0]
    if camera_count == 0:
        raise ValueError("At least one camera is required for camera normal orientation")

    for begin in range(0, positions.shape[0], chunk_size):
        end = min(begin + chunk_size, positions.shape[0])
        point_chunk = positions[begin:end]
        camera_vectors = camera_locations[None, :, :] - point_chunk[:, None, :]
        squared_distances = np.einsum(
            "nci,nci->nc", camera_vectors, camera_vectors, optimize=True
        )
        nearest_indices = np.argmin(squared_distances, axis=1)
        nearest_vectors[begin:end] = camera_vectors[
            np.arange(end - begin), nearest_indices
        ]

    return nearest_vectors


def orient_surfel_normals(
    positions: np.ndarray,
    normals: np.ndarray,
    mode: str,
    consistent_neighbor_count: int = 20,
    camera_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    positions = np.asarray(positions, dtype=np.float64)
    oriented = np.asarray(normals, dtype=np.float64).copy()
    if positions.shape != oriented.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Positions and normals must both have shape (N,3), got {positions.shape} and {oriented.shape}"
        )
    if mode not in {"surfel", "camera", "consistent", "consistent-camera"}:
        raise ValueError(f"Unknown normal orientation mode: {mode}")

    lengths = np.linalg.norm(oriented, axis=1)
    if not np.isfinite(oriented).all() or np.any(lengths <= 1.0e-12):
        raise ValueError("Normal orientation received invalid or near-zero normals")
    oriented /= lengths[:, None]

    use_camera_reference = mode in {"camera", "consistent-camera"}
    use_consistency = mode in {"consistent", "consistent-camera"}
    camera_locations: np.ndarray | None = None
    camera_flips = 0
    nearest_camera_vectors: np.ndarray | None = None

    if use_camera_reference:
        if camera_positions is None:
            raise ValueError(f"Normal orientation mode {mode!r} requires camera positions")
        camera_locations = np.asarray(camera_positions, dtype=np.float64)
        if camera_locations.ndim != 2 or camera_locations.shape[1] != 3 or camera_locations.shape[0] == 0:
            raise ValueError(
                f"Camera positions must have shape (C,3), got {camera_locations.shape}"
            )
        if not np.isfinite(camera_locations).all():
            raise ValueError("Camera positions must all be finite")
        nearest_camera_vectors = _vectors_to_nearest_cameras(positions, camera_locations)
        flip_mask = np.sum(oriented * nearest_camera_vectors, axis=1) < 0.0
        oriented[flip_mask] *= -1.0
        camera_flips = int(np.count_nonzero(flip_mask))

    consistency_k = 0
    consistency_used_joggle = False
    if use_consistency and positions.shape[0] >= 4:
        consistency_k = min(max(int(consistent_neighbor_count), 3), positions.shape[0] - 1)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(positions)
        point_cloud.normals = o3d.utility.Vector3dVector(oriented)
        try:
            point_cloud.orient_normals_consistent_tangent_plane(consistency_k)
        except RuntimeError as exception:
            # Open3D builds a Delaunay/Riemannian graph and Qhull rejects exact
            # planes or other cospherical configurations. Joggle only the graph
            # points by a deterministic numerical epsilon; reconstruction samples
            # continue to use the original surfel centers.
            if "QH" not in str(exception) and "Qhull" not in str(exception):
                raise
            extent = np.ptp(positions, axis=0)
            joggle_scale = max(float(np.max(extent)), 1.0) * 1.0e-8
            joggle_rng = np.random.default_rng(0)
            graph_positions = positions + joggle_rng.standard_normal(positions.shape) * joggle_scale
            point_cloud.points = o3d.utility.Vector3dVector(graph_positions)
            point_cloud.normals = o3d.utility.Vector3dVector(oriented)
            point_cloud.orient_normals_consistent_tangent_plane(consistency_k)
            consistency_used_joggle = True
        oriented = np.asarray(point_cloud.normals).copy()

    global_camera_flip = False
    camera_facing_fraction: float | None = None
    if nearest_camera_vectors is not None:
        facing = np.sum(oriented * nearest_camera_vectors, axis=1)
        if float(np.sum(facing)) < 0.0:
            oriented *= -1.0
            facing *= -1.0
            global_camera_flip = True
        camera_facing_fraction = float(np.mean(facing >= 0.0))

    report: dict[str, Any] = {
        "mode": mode,
        "consistent_neighbor_count": int(consistency_k),
        "consistent_orientation_used_epsilon_joggle": bool(consistency_used_joggle),
        "initial_camera_flips": int(camera_flips),
        "global_camera_flip_after_consistency": bool(global_camera_flip),
        "camera_facing_fraction": camera_facing_fraction,
    }
    return np.ascontiguousarray(oriented), report


def integrated_surfel_opacity_mass(
    cloud: QuaternionSurfelCloud,
    use_beta_profile: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if use_beta_profile:
        beta_exponent = 4.0 * np.exp(np.clip(cloud.betas, -20.0, 20.0))
    else:
        beta_exponent = np.zeros(cloud.surfel_count, dtype=np.float64)

    # Integral over the unit disk of (1-r^2)^a is pi/(a+1).
    masses = (
        cloud.opacities
        * math.pi
        * cloud.scale_u
        * cloud.scale_v
        / (beta_exponent + 1.0)
    )
    masses = np.where(np.isfinite(masses) & (masses > 0.0), masses, 0.0)
    return masses, beta_exponent


def allocate_weighted_sample_counts(
    weights: np.ndarray,
    sample_count: int,
    minimum_samples_per_item: int = 0,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        raise ValueError("Cannot allocate samples across an empty weight array")
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("Sampling weights must be finite, non-negative, and have positive sum")
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    minimum = int(minimum_samples_per_item)
    if minimum < 0:
        raise ValueError(
            f"minimum_samples_per_item must be non-negative, got {minimum_samples_per_item}"
        )
    required = minimum * weights.size
    if required > sample_count:
        raise ValueError(
            f"sample_count={sample_count} cannot provide {minimum} samples to each of "
            f"{weights.size} surfels"
        )

    counts = np.full(weights.shape, minimum, dtype=np.int64)
    remaining = int(sample_count - required)
    if remaining == 0:
        return counts

    expected = remaining * weights / float(weights.sum())
    additions = np.floor(expected).astype(np.int64)
    counts += additions
    leftover = int(remaining - int(additions.sum()))
    if leftover > 0:
        fractional = expected - additions
        # Stable sorting makes ties deterministic across runs and platforms.
        selected = np.argsort(-fractional, kind="stable")[:leftover]
        counts[selected] += 1

    if int(counts.sum()) != int(sample_count):
        raise RuntimeError("Internal weighted sample allocation did not preserve the sample budget")
    return counts


def sample_opacity_weighted_surfel_surfaces(
    cloud: QuaternionSurfelCloud,
    oriented_normals: np.ndarray,
    settings: PoissonSamplingSettings,
) -> tuple[o3d.geometry.PointCloud, dict[str, Any]]:
    if int(settings.sample_count) <= 0:
        raise ValueError(f"sample_count must be positive, got {settings.sample_count}")
    if int(settings.minimum_samples_per_surfel) < 0:
        raise ValueError(
            "minimum_samples_per_surfel must be non-negative, got "
            f"{settings.minimum_samples_per_surfel}"
        )

    tangent_u, tangent_v, quaternion_normals = quaternion_tangent_frames_wxyz(
        cloud.quaternions_wxyz
    )
    oriented_normals = np.asarray(oriented_normals, dtype=np.float64)
    if oriented_normals.shape != quaternion_normals.shape:
        raise ValueError(
            f"Oriented normals have shape {oriented_normals.shape}, expected {quaternion_normals.shape}"
        )

    masses, beta_exponent = integrated_surfel_opacity_mass(
        cloud, use_beta_profile=settings.use_beta_profile
    )
    positive_mass = masses > 0.0
    if not np.all(positive_mass):
        raise ValueError("Filtered surfel cloud still contains zero opacity-mass surfels")

    counts = allocate_weighted_sample_counts(
        masses,
        sample_count=int(settings.sample_count),
        minimum_samples_per_item=int(settings.minimum_samples_per_surfel),
    )
    sampled_surfel_indices = np.repeat(np.arange(cloud.surfel_count, dtype=np.int64), counts)
    if sampled_surfel_indices.size != int(settings.sample_count):
        raise RuntimeError("Sample source index count does not match requested Poisson sample count")

    starts = np.cumsum(counts, dtype=np.int64) - counts
    repeated_starts = np.repeat(starts, counts)
    local_sample_indices = np.arange(sampled_surfel_indices.size, dtype=np.int64) - repeated_starts
    repeated_counts = counts[sampled_surfel_indices]
    radial_quantile = (local_sample_indices.astype(np.float64) + 0.5) / repeated_counts

    if settings.use_beta_profile:
        repeated_exponent = beta_exponent[sampled_surfel_indices]
        # Inverse CDF for p(r) proportional to r*(1-r^2)^a on r in [0,1].
        radius_squared = -np.expm1(
            np.log1p(-radial_quantile) / (repeated_exponent + 1.0)
        )
    else:
        radius_squared = radial_quantile
    radius = np.sqrt(np.clip(radius_squared, 0.0, 1.0))

    random_generator = np.random.default_rng(int(settings.seed))
    angular_offsets = random_generator.random(cloud.surfel_count)
    golden_ratio_conjugate = 0.6180339887498948482
    angle_cycles = np.mod(
        local_sample_indices * golden_ratio_conjugate
        + angular_offsets[sampled_surfel_indices],
        1.0,
    )
    angles = 2.0 * math.pi * angle_cycles
    local_u = radius * np.cos(angles)
    local_v = radius * np.sin(angles)

    source = sampled_surfel_indices
    sample_positions = (
        cloud.positions[source]
        + (local_u * cloud.scale_u[source])[:, None] * tangent_u[source]
        + (local_v * cloud.scale_v[source])[:, None] * tangent_v[source]
    )
    sample_normals = oriented_normals[source]
    sample_colors = cloud.colors[source]

    finite_samples = (
        np.isfinite(sample_positions).all(axis=1)
        & np.isfinite(sample_normals).all(axis=1)
        & np.isfinite(sample_colors).all(axis=1)
    )
    if not np.all(finite_samples):
        raise RuntimeError(
            f"Generated {np.count_nonzero(~finite_samples)} non-finite Poisson samples"
        )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(sample_positions)
    point_cloud.normals = o3d.utility.Vector3dVector(sample_normals)
    point_cloud.colors = o3d.utility.Vector3dVector(np.clip(sample_colors, 0.0, 1.0))

    positive_counts = counts[counts > 0]
    report: dict[str, Any] = {
        "requested_samples": int(settings.sample_count),
        "generated_samples": int(sample_positions.shape[0]),
        "seed": int(settings.seed),
        "minimum_samples_per_surfel": int(settings.minimum_samples_per_surfel),
        "use_beta_profile": bool(settings.use_beta_profile),
        "integrated_opacity_mass": float(masses.sum()),
        "sampled_surfels": int(np.count_nonzero(counts)),
        "unsampled_surfels": int(np.count_nonzero(counts == 0)),
        "minimum_positive_samples_per_surfel": int(positive_counts.min()),
        "maximum_samples_per_surfel": int(positive_counts.max()),
        "mean_samples_per_surfel": float(counts.mean()),
    }
    return point_cloud, report


def reconstruct_screened_poisson(
    samples: o3d.geometry.PointCloud,
    settings: PoissonReconstructionSettings,
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh, dict[str, Any]]:
    sample_count = len(samples.points)
    if sample_count < 4:
        raise ValueError(f"Screened Poisson reconstruction needs at least four points, got {sample_count}")
    if not samples.has_normals():
        raise ValueError("Screened Poisson reconstruction requires oriented sample normals")
    if settings.depth < 2:
        raise ValueError(f"Poisson depth must be >= 2, got {settings.depth}")
    if not math.isfinite(settings.scale) or settings.scale <= 1.0:
        raise ValueError(f"Poisson scale must be finite and > 1, got {settings.scale}")
    if not 0.0 <= settings.density_quantile < 1.0:
        raise ValueError(
            f"Poisson density quantile must lie in [0,1), got {settings.density_quantile}"
        )
    if not math.isfinite(settings.coverage_trim_cells):
        raise ValueError(
            f"coverage_trim_cells must be finite, got {settings.coverage_trim_cells}"
        )
    if not math.isfinite(settings.crop_padding_cells):
        raise ValueError(
            f"crop_padding_cells must be finite, got {settings.crop_padding_cells}"
        )

    mesh, density_values = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        samples,
        depth=int(settings.depth),
        scale=float(settings.scale),
        linear_fit=bool(settings.linear_fit),
        n_threads=int(settings.n_threads),
    )
    if mesh.has_vertex_colors():
        mesh_colors = np.asarray(mesh.vertex_colors)
        mesh_colors[:] = np.clip(
            np.nan_to_num(mesh_colors, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
    raw_mesh = o3d.geometry.TriangleMesh(mesh)
    densities = np.asarray(density_values, dtype=np.float64)
    vertices = np.asarray(mesh.vertices)
    if vertices.shape[0] == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("Screened Poisson reconstruction returned an empty mesh")
    if densities.shape != (vertices.shape[0],):
        raise RuntimeError(
            f"Poisson density count {densities.shape} does not match vertices {vertices.shape[0]}"
        )

    sample_positions = np.asarray(samples.points)
    sample_min = sample_positions.min(axis=0)
    sample_max = sample_positions.max(axis=0)
    sample_extent = sample_max - sample_min
    reconstruction_cube_width = float(np.max(sample_extent) * settings.scale)
    finest_cell_width = reconstruction_cube_width / float(2 ** int(settings.depth))

    keep = np.isfinite(vertices).all(axis=1) & np.isfinite(densities)
    density_threshold: float | None = None
    if settings.density_quantile > 0.0:
        density_threshold = float(np.quantile(densities[np.isfinite(densities)], settings.density_quantile))
        keep &= densities >= density_threshold

    crop_padding = max(float(settings.crop_padding_cells), 0.0) * finest_cell_width
    keep &= np.all(vertices >= sample_min[None, :] - crop_padding, axis=1)
    keep &= np.all(vertices <= sample_max[None, :] + crop_padding, axis=1)

    coverage_threshold: float | None = None
    nearest_sample_distances: np.ndarray | None = None
    if settings.coverage_trim_cells > 0.0:
        coverage_threshold = float(settings.coverage_trim_cells) * finest_cell_width
        vertex_cloud = o3d.geometry.PointCloud()
        vertex_cloud.points = o3d.utility.Vector3dVector(vertices)
        nearest_sample_distances = np.asarray(
            vertex_cloud.compute_point_cloud_distance(samples), dtype=np.float64
        )
        keep &= np.isfinite(nearest_sample_distances)
        keep &= nearest_sample_distances <= coverage_threshold

    trimmed_mesh = o3d.geometry.TriangleMesh(mesh)
    trimmed_mesh.remove_vertices_by_mask((~keep).tolist())
    trimmed_mesh.remove_unreferenced_vertices()
    trimmed_mesh.remove_duplicated_vertices()
    trimmed_mesh.remove_duplicated_triangles()
    trimmed_mesh.remove_degenerate_triangles()
    if len(trimmed_mesh.vertices) == 0 or len(trimmed_mesh.triangles) == 0:
        raise RuntimeError(
            "Poisson support trimming removed the complete mesh; reduce density or "
            "coverage trimming."
        )
    trimmed_mesh.compute_vertex_normals()

    finite_densities = densities[np.isfinite(densities)]
    density_quantiles = np.quantile(
        finite_densities, (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
    )
    report: dict[str, Any] = {
        "sample_count": int(sample_count),
        "depth": int(settings.depth),
        "scale": float(settings.scale),
        "linear_fit": bool(settings.linear_fit),
        "n_threads": int(settings.n_threads),
        "density_quantile": float(settings.density_quantile),
        "density_threshold": density_threshold,
        "coverage_trim_cells": float(settings.coverage_trim_cells),
        "coverage_threshold": coverage_threshold,
        "crop_padding_cells": float(settings.crop_padding_cells),
        "finest_cell_width": float(finest_cell_width),
        "raw_vertices": int(len(raw_mesh.vertices)),
        "raw_triangles": int(len(raw_mesh.triangles)),
        "trimmed_vertices": int(len(trimmed_mesh.vertices)),
        "trimmed_triangles": int(len(trimmed_mesh.triangles)),
        "density_quantiles": {
            name: float(value)
            for name, value in zip(
                ("min", "p01", "p05", "p50", "p95", "p99", "max"),
                density_quantiles,
            )
        },
        "nearest_sample_distance_quantiles": None,
    }
    if nearest_sample_distances is not None and nearest_sample_distances.size > 0:
        distance_quantiles = np.quantile(
            nearest_sample_distances[np.isfinite(nearest_sample_distances)],
            (0.0, 0.5, 0.9, 0.95, 0.99, 1.0),
        )
        report["nearest_sample_distance_quantiles"] = {
            name: float(value)
            for name, value in zip(
                ("min", "p50", "p90", "p95", "p99", "max"),
                distance_quantiles,
            )
        }

    return raw_mesh, trimmed_mesh, report
