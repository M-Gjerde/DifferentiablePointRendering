#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d


def find_latest_points_ply(outputRootPath: Path) -> Path:
    if not outputRootPath.exists():
        raise FileNotFoundError(f"Output root '{outputRootPath}' does not exist.")

    if outputRootPath.is_file():
        print(f"Using PLY file: {outputRootPath}")
        return outputRootPath

    pointsInRoot = outputRootPath / "points_final.ply"
    if pointsInRoot.is_file():
        print(f"Using points_final.ply in run directory: {outputRootPath}")
        return pointsInRoot

    candidateRunDirs: List[Path] = []
    for childPath in outputRootPath.iterdir():
        if childPath.is_dir() and (childPath / "points_final.ply").is_file():
            candidateRunDirs.append(childPath)

    if not candidateRunDirs:
        raise FileNotFoundError(f"No subdirectories with points_final.ply found under '{outputRootPath}'.")

    latestRunDir = max(candidateRunDirs, key=lambda runPath: (runPath / "points_final.ply").stat().st_mtime)
    latestPlyPath = latestRunDir / "points_final.ply"
    print(f"Using latest run directory: {latestRunDir}")
    print(f"points_final.ply: {latestPlyPath}")
    return latestPlyPath


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
            if not parts or len(parts) < 15:
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

            color0.append(float(parts[11]))
            color1.append(float(parts[12]))
            color2.append(float(parts[13]))

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


def normalize_rows(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (lengths + eps)


def build_normals_from_tangents(tangentU: np.ndarray, tangentV: np.ndarray) -> np.ndarray:
    uHat = normalize_rows(tangentU.astype(np.float32))
    vHat = normalize_rows(tangentV.astype(np.float32))
    normals = np.cross(uHat, vHat)
    normals = normalize_rows(normals)
    return normals

def split_point_cloud_into_components(
    point_cloud: o3d.geometry.PointCloud,
    cluster_eps: float,
    cluster_eps_mult: float,
    min_component_points: int,
) -> list[o3d.geometry.PointCloud]:
    if cluster_eps <= 0.0:
        median_spacing = estimate_knn_spacing(point_cloud, knn=1)
        if median_spacing <= 0.0:
            raise RuntimeError("Could not estimate point spacing for clustering.")

        cluster_eps = cluster_eps_mult * median_spacing

    labels = np.asarray(
        point_cloud.cluster_dbscan(
            eps=float(cluster_eps),
            min_points=10,
            print_progress=True,
        )
    )

    valid_labels = sorted(label for label in set(labels.tolist()) if label >= 0)

    components: list[o3d.geometry.PointCloud] = []

    for label in valid_labels:
        component_indices = np.where(labels == label)[0]

        if component_indices.size < min_component_points:
            continue

        component = point_cloud.select_by_index(component_indices.tolist())
        components.append(component)

    components.sort(
        key=lambda component: np.asarray(component.points).shape[0],
        reverse=True,
    )

    print(f"DBSCAN: eps={cluster_eps:.6g}, components={len(components)}")

    for component_index, component in enumerate(components):
        print(
            f"  component {component_index}: "
            f"{np.asarray(component.points).shape[0]} points"
        )

    return components

def estimate_knn_spacing(pointCloud: o3d.geometry.PointCloud, knn: int) -> float:
    distances = pointCloud.compute_nearest_neighbor_distance()
    if len(distances) == 0:
        return 0.0
    return float(np.median(np.asarray(distances, dtype=np.float64)))
def reconstruct_mesh_from_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    method: str,
    poisson_depth: int,
    poisson_scale: float,
    poisson_linear_fit: bool,
    poisson_density_quantile: float,
    bpa_radius_mult: float,
    bpa_radii_count: int,
) -> o3d.geometry.TriangleMesh:
    if method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud,
            depth=int(poisson_depth),
            scale=float(poisson_scale),
            linear_fit=bool(poisson_linear_fit),
        )

        densities_np = np.asarray(densities, dtype=np.float64)

        if densities_np.size > 0 and 0.0 < poisson_density_quantile < 1.0:
            threshold = float(np.quantile(densities_np, poisson_density_quantile))
            vertices_to_keep = densities_np >= threshold
            mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
            print(
                f"Poisson: removed bottom {poisson_density_quantile:.3f} density "
                f"(threshold={threshold:.6g})"
            )

        return mesh

    median_spacing = estimate_knn_spacing(point_cloud, knn=1)
    if median_spacing <= 0.0:
        raise RuntimeError("Could not estimate spacing for BPA.")

    base_radius = float(bpa_radius_mult) * median_spacing
    radii_count = max(1, int(bpa_radii_count))
    radii = base_radius * np.linspace(1.0, float(radii_count), radii_count).astype(np.float64)

    print(
        f"BPA: median NN spacing={median_spacing:.6g}, "
        f"base_radius={base_radius:.6g}, radii={radii.tolist()}"
    )

    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud,
        o3d.utility.DoubleVector(radii.tolist()),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a mesh from surfel point cloud using Open3D.")
    parser.add_argument("--output-root", type=Path, default="../Assets/OptimizationOutput")
    parser.add_argument("--opacity-threshold", type=float, default=0.0)
    parser.add_argument("--area-threshold", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=0)

    parser.add_argument("--method", choices=["poisson", "bpa"], default="poisson")

    # Poisson
    parser.add_argument("--poisson-depth", type=int, default=13)
    parser.add_argument("--poisson-scale", type=float, default=1.00)
    parser.add_argument("--poisson-linear-fit", action="store_true", default=True)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.00,
                        help="Remove low-density vertices: drop bottom q (0..1). Typical 0.01..0.05")

    # BPA
    parser.add_argument("--bpa-radius-mult", type=float, default=2.5,
                        help="Ball pivot radius multiplier relative to median NN spacing.")
    parser.add_argument("--bpa-radii-count", type=int, default=3,
                        help="Number of radii levels: radii = base * linspace(1, count).")

    parser.add_argument("--split-components", action="store_true", default=False)
    parser.add_argument("--cluster-eps", type=float, default=0.0)
    parser.add_argument("--cluster-eps-mult", type=float, default=6.0)
    parser.add_argument("--min-component-points", type=int, default=100)

    # Cleanup
    parser.add_argument("--remove-degenerate-triangles", action="store_true")
    parser.add_argument("--remove-duplicated-triangles", action="store_true")
    parser.add_argument("--remove-duplicated-vertices", action="store_true")
    parser.add_argument("--remove-non-manifold-edges", action="store_true")

    parser.add_argument("--save-mesh", type=Path, default=None)
    parser.add_argument("--save-pointcloud", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    plyPath = find_latest_points_ply(args.output_root)

    positions, tangentU, tangentV, su, sv, colors, opacities = load_surfels_from_ply(
        plyPath,
        opacityThreshold=args.opacity_threshold,
    )

    ellipseArea = su * sv
    keepMask = ellipseArea >= float(args.area_threshold)


    scale_metric = np.sqrt(np.abs(su * sv))
    low = np.quantile(scale_metric, 0.01)
    high = np.quantile(scale_metric, 0.99)
    keepMask = keepMask & (scale_metric >= low) & (scale_metric <= high)


    positions = positions[keepMask]
    tangentU = tangentU[keepMask]
    tangentV = tangentV[keepMask]
    colors = colors[keepMask]

    if args.max_points and positions.shape[0] > args.max_points:
        positions = positions[: args.max_points]
        tangentU = tangentU[: args.max_points]
        tangentV = tangentV[: args.max_points]
        colors = colors[: args.max_points]

    normals = build_normals_from_tangents(tangentU, tangentV)

    pointCloud = o3d.geometry.PointCloud()
    pointCloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
    pointCloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pointCloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    # 1) Remove obvious outliers (often fixes the giant plane)
    pointCloud, inlier_indices = pointCloud.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0,
    )
    print(f"Outlier removal: kept {np.asarray(pointCloud.points).shape[0]} / {positions.shape[0]} points")

    # 2) Re-estimate normals from geometry (more stable than tangent cross products)
    pointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

    # 3) Make normals locally consistent (important for Poisson)
    pointCloud.orient_normals_consistent_tangent_plane(k=30)

    if args.split_components:
        point_clouds = split_point_cloud_into_components(
            point_cloud=pointCloud,
            cluster_eps=args.cluster_eps,
            cluster_eps_mult=args.cluster_eps_mult,
            min_component_points=args.min_component_points,
        )

        if not point_clouds:
            raise RuntimeError("No connected components survived clustering.")

    else:
        point_clouds = [pointCloud]

    meshes = []

    for component_index, component_point_cloud in enumerate(point_clouds):
        print()
        print(f"Reconstructing component {component_index}")

        component_point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.02,
                max_nn=30,
            )
        )
        component_point_cloud.orient_normals_consistent_tangent_plane(k=30)

        component_mesh = reconstruct_mesh_from_point_cloud(
            point_cloud=component_point_cloud,
            method=args.method,
            poisson_depth=args.poisson_depth,
            poisson_scale=args.poisson_scale,
            poisson_linear_fit=args.poisson_linear_fit,
            poisson_density_quantile=args.poisson_density_quantile,
            bpa_radius_mult=args.bpa_radius_mult,
            bpa_radii_count=args.bpa_radii_count,
        )

        component_mesh.compute_vertex_normals()
        meshes.append(component_mesh)

    mesh = o3d.geometry.TriangleMesh()

    for component_mesh in meshes:
        mesh += component_mesh

    if args.save_mesh is not None:
        args.save_mesh.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(args.save_mesh), mesh)
        print(f"Saved mesh: {args.save_mesh}")

    o3d.visualization.draw_geometries(
        [mesh],
        window_name=f"Open3D mesh ({args.method})",
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
