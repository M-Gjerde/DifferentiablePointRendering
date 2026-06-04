from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_points_from_ply(
    ply_path: Path,
    sample_count: int,
    use_vertices: bool,
) -> np.ndarray:
    if not ply_path.exists():
        raise FileNotFoundError(f"File does not exist: {ply_path}")

    mesh = o3d.io.read_triangle_mesh(str(ply_path))

    if not mesh.is_empty() and len(mesh.vertices) > 0:
        vertex_points = np.asarray(mesh.vertices, dtype=np.float64)
        triangle_count = len(mesh.triangles)

        if use_vertices or triangle_count == 0:
            return validate_points(vertex_points, ply_path)

        sampled_point_cloud = mesh.sample_points_uniformly(number_of_points=sample_count)
        sampled_points = np.asarray(sampled_point_cloud.points, dtype=np.float64)
        return validate_points(sampled_points, ply_path)

    point_cloud = o3d.io.read_point_cloud(str(ply_path))

    if point_cloud.is_empty() or len(point_cloud.points) == 0:
        raise ValueError(f"Could not load vertices or point cloud from: {ply_path}")

    points = np.asarray(point_cloud.points, dtype=np.float64)
    return validate_points(points, ply_path)


def validate_points(points: np.ndarray, ply_path: Path) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points from {ply_path}, got shape {points.shape}")

    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    if len(points) == 0:
        raise ValueError(f"No valid finite points found in: {ply_path}")

    return points


def nearest_neighbor_distances(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    target_tree = cKDTree(target_points)
    distances, _ = target_tree.query(source_points, k=1, workers=-1)
    return distances


def compute_chamfer_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
    squared: bool,
) -> dict[str, float]:
    distances_a_to_b = nearest_neighbor_distances(points_a, points_b)
    distances_b_to_a = nearest_neighbor_distances(points_b, points_a)

    if squared:
        distances_a_to_b = distances_a_to_b * distances_a_to_b
        distances_b_to_a = distances_b_to_a * distances_b_to_a

    chamfer_a_to_b = float(np.mean(distances_a_to_b))
    chamfer_b_to_a = float(np.mean(distances_b_to_a))

    return {
        "chamfer_a_to_b": chamfer_a_to_b,
        "chamfer_b_to_a": chamfer_b_to_a,
        "chamfer_symmetric_sum": chamfer_a_to_b + chamfer_b_to_a,
        "chamfer_symmetric_mean": 0.5 * (chamfer_a_to_b + chamfer_b_to_a),
        "median_a_to_b": float(np.median(distances_a_to_b)),
        "median_b_to_a": float(np.median(distances_b_to_a)),
        "max_a_to_b": float(np.max(distances_a_to_b)),
        "max_b_to_a": float(np.max(distances_b_to_a)),
    }


def format_float(value: float) -> str:
    return f"{value:.12f}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Chamfer distance between two PLY models.")
    parser.add_argument("ground_truth", type=Path, help="Path to ground-truth .ply mesh or point cloud.")
    parser.add_argument("reconstruction", type=Path, help="Path to reconstructed .ply mesh or point cloud.")
    parser.add_argument("--samples", type=int, default=500_000, help="Number of sampled surface points per mesh.")
    parser.add_argument("--use_vertices", action="store_true", help="Use raw vertices instead of surface sampling.")
    parser.add_argument("--squared", action="store_true", help="Use squared distances.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    reconstruction_points = load_points_from_ply(
        ply_path=args.reconstruction.expanduser(),
        sample_count=args.samples,
        use_vertices=args.use_vertices,
    )

    ground_truth_points = load_points_from_ply(
        ply_path=args.ground_truth.expanduser(),
        sample_count=args.samples,
        use_vertices=args.use_vertices,
    )

    results = compute_chamfer_distance(
        points_a=reconstruction_points,
        points_b=ground_truth_points,
        squared=args.squared,
    )

    distance_type = "squared" if args.squared else "linear"

    print("Chamfer distance")
    print(f"  Reconstruction: {args.reconstruction}")
    print(f"  Ground truth:   {args.ground_truth}")
    print(f"  Reconstruction points: {len(reconstruction_points)}")
    print(f"  Ground-truth points:   {len(ground_truth_points)}")
    print(f"  Distance type: {distance_type}")
    print()
    print("Main results")
    print(f"  Reconstruction -> GT: {format_float(results['chamfer_a_to_b'])}")
    print(f"  GT -> Reconstruction: {format_float(results['chamfer_b_to_a'])}")
    print(f"  Symmetric sum:        {format_float(results['chamfer_symmetric_sum'])}")
    print(f"  Symmetric mean:       {format_float(results['chamfer_symmetric_mean'])}")
    print()
    print("Diagnostics")
    print(f"  Median Reconstruction -> GT: {format_float(results['median_a_to_b'])}")
    print(f"  Median GT -> Reconstruction: {format_float(results['median_b_to_a'])}")
    print(f"  Max Reconstruction -> GT:    {format_float(results['max_a_to_b'])}")
    print(f"  Max GT -> Reconstruction:    {format_float(results['max_b_to_a'])}")


if __name__ == "__main__":
    main()