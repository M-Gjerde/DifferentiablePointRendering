from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

try:
    import trimesh
except ModuleNotFoundError:
    trimesh = None

try:
    from pytorch3d.loss import chamfer_distance
except ModuleNotFoundError as exception:
    raise ModuleNotFoundError(
        "PyTorch3D is required for this script because it uses "
        "pytorch3d.loss.chamfer_distance.\n\n"
        "Install with one of:\n"
        "  conda install pytorch3d -c pytorch3d\n"
        "  pip install \"git+https://github.com/facebookresearch/pytorch3d.git@stable\"\n"
    ) from exception


def validate_points(points: np.ndarray, mesh_path: Path) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points from {mesh_path}, got shape {points.shape}")

    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    if len(points) == 0:
        raise ValueError(f"No valid finite points found in: {mesh_path}")

    return np.ascontiguousarray(points, dtype=np.float32)


def triangulate_polygon_face(face: np.ndarray) -> list[list[int]]:
    if len(face) < 3:
        return []

    if len(face) == 3:
        return [[int(face[0]), int(face[1]), int(face[2])]]

    triangles: list[list[int]] = []

    for vertex_index in range(1, len(face) - 1):
        triangles.append([int(face[0]), int(face[vertex_index]), int(face[vertex_index + 1])])

    return triangles


def load_points_with_trimesh(mesh_path: Path, sample_count: int, use_vertices: bool) -> tuple[np.ndarray, str]:
    if trimesh is None:
        raise ModuleNotFoundError(
            "Open3D could not load this file as a usable triangle mesh, and trimesh is not installed.\n"
            "Install it with:\n"
            "  pip install trimesh"
        )

    loaded_object = trimesh.load(str(mesh_path), force="mesh", process=False)

    if isinstance(loaded_object, trimesh.Scene):
        geometry_list = [geometry for geometry in loaded_object.geometry.values()]

        if len(geometry_list) == 0:
            raise ValueError(f"Trimesh loaded an empty scene from: {mesh_path}")

        loaded_object = trimesh.util.concatenate(geometry_list)

    if not isinstance(loaded_object, trimesh.Trimesh):
        raise ValueError(f"Trimesh could not load a mesh from: {mesh_path}")

    vertices = np.asarray(loaded_object.vertices, dtype=np.float32)

    if len(vertices) == 0:
        raise ValueError(f"Trimesh loaded no vertices from: {mesh_path}")

    if use_vertices:
        return validate_points(vertices, mesh_path), "trimesh_vertices"

    faces_raw = loaded_object.faces

    if faces_raw is None or len(faces_raw) == 0:
        return validate_points(vertices, mesh_path), "trimesh_vertices_no_faces"

    triangle_faces: list[list[int]] = []

    for face in faces_raw:
        triangle_faces.extend(triangulate_polygon_face(np.asarray(face)))

    if len(triangle_faces) == 0:
        return validate_points(vertices, mesh_path), "trimesh_vertices_no_valid_faces"

    triangle_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(triangle_faces, dtype=np.int64),
        process=False,
    )

    sampled_points, _ = trimesh.sample.sample_surface(triangle_mesh, sample_count)
    return validate_points(np.asarray(sampled_points, dtype=np.float32), mesh_path), "trimesh_uniform_surface_sampling"


def load_points_from_mesh(mesh_path: Path, sample_count: int, use_vertices: bool) -> tuple[np.ndarray, str]:
    mesh_path = mesh_path.expanduser().resolve()

    if not mesh_path.exists():
        raise FileNotFoundError(f"File does not exist: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if not mesh.is_empty() and len(mesh.vertices) > 0:
        vertex_points = np.asarray(mesh.vertices, dtype=np.float32)
        triangle_count = len(mesh.triangles)

        if use_vertices:
            return validate_points(vertex_points, mesh_path), "open3d_vertices"

        if triangle_count > 0:
            sampled_point_cloud = mesh.sample_points_uniformly(number_of_points=sample_count)
            sampled_points = np.asarray(sampled_point_cloud.points, dtype=np.float32)
            return validate_points(sampled_points, mesh_path), "open3d_uniform_surface_sampling"

    try:
        return load_points_with_trimesh(
            mesh_path=mesh_path,
            sample_count=sample_count,
            use_vertices=use_vertices,
        )
    except Exception as trimesh_exception:
        point_cloud = o3d.io.read_point_cloud(str(mesh_path))

        if not point_cloud.is_empty() and len(point_cloud.points) > 0:
            points = np.asarray(point_cloud.points, dtype=np.float32)
            return validate_points(points, mesh_path), "open3d_point_cloud_fallback"

        raise RuntimeError(
            f"Could not load {mesh_path} as triangle mesh, polygon mesh, or point cloud.\n"
            f"Trimesh error was:\n{trimesh_exception}"
        ) from trimesh_exception


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        o3d.utility.random.seed(seed)
    except AttributeError:
        pass


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    return device


def tensor_from_points(points: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(points).to(device=device, dtype=torch.float32).unsqueeze(0).contiguous()


@torch.no_grad()
def compute_chamfer_metrics(
    reconstruction_points: np.ndarray,
    ground_truth_points: np.ndarray,
    device: torch.device,
    scale: float,
) -> dict[str, float]:
    reconstruction_tensor = tensor_from_points(reconstruction_points, device)
    ground_truth_tensor = tensor_from_points(ground_truth_points, device)

    directional_distances, _ = chamfer_distance(
        x=reconstruction_tensor,
        y=ground_truth_tensor,
        batch_reduction=None,
        point_reduction=None,
        norm=2,
        single_directional=False,
    )

    reconstruction_to_gt_squared, gt_to_reconstruction_squared = directional_distances

    reconstruction_to_gt = torch.sqrt(torch.clamp(reconstruction_to_gt_squared, min=0.0))
    gt_to_reconstruction = torch.sqrt(torch.clamp(gt_to_reconstruction_squared, min=0.0))

    accuracy = torch.mean(reconstruction_to_gt)
    completion = torch.mean(gt_to_reconstruction)
    chamfer_distance_value = 0.5 * (accuracy + completion)

    squared_loss, _ = chamfer_distance(
        x=reconstruction_tensor,
        y=ground_truth_tensor,
        batch_reduction="mean",
        point_reduction="mean",
        norm=2,
        single_directional=False,
    )

    return {
        "accuracy": float((accuracy * scale).item()),
        "completion": float((completion * scale).item()),
        "cd": float((chamfer_distance_value * scale).item()),
        "accuracy_raw": float(accuracy.item()),
        "completion_raw": float(completion.item()),
        "cd_raw": float(chamfer_distance_value.item()),
        "median_reconstruction_to_gt": float((torch.median(reconstruction_to_gt) * scale).item()),
        "median_gt_to_reconstruction": float((torch.median(gt_to_reconstruction) * scale).item()),
        "p95_reconstruction_to_gt": float((torch.quantile(reconstruction_to_gt, 0.95) * scale).item()),
        "p95_gt_to_reconstruction": float((torch.quantile(gt_to_reconstruction, 0.95) * scale).item()),
        "max_reconstruction_to_gt": float((torch.max(reconstruction_to_gt) * scale).item()),
        "max_gt_to_reconstruction": float((torch.max(gt_to_reconstruction) * scale).item()),
        "pytorch3d_squared_loss_raw": float(squared_loss.item()),
    }


def format_float(value: float) -> str:
    return f"{value:.12f}"


def evaluate_chamfer(args: argparse.Namespace) -> dict:
    ground_truth_path = args.ground_truth.expanduser().resolve()
    reconstruction_path = args.reconstruction.expanduser().resolve()
    device = resolve_device(args.device)

    set_random_seed(args.seed)

    reconstruction_points, reconstruction_sampling_mode = load_points_from_mesh(
        mesh_path=reconstruction_path,
        sample_count=args.samples,
        use_vertices=args.use_vertices,
    )

    ground_truth_points, ground_truth_sampling_mode = load_points_from_mesh(
        mesh_path=ground_truth_path,
        sample_count=args.samples,
        use_vertices=args.use_vertices,
    )

    metrics = compute_chamfer_metrics(
        reconstruction_points=reconstruction_points,
        ground_truth_points=ground_truth_points,
        device=device,
        scale=args.scale,
    )

    return {
        "ground_truth": str(ground_truth_path),
        "reconstruction": str(reconstruction_path),
        "reconstruction_points": len(reconstruction_points),
        "ground_truth_points": len(ground_truth_points),
        "reconstruction_sampling": reconstruction_sampling_mode,
        "ground_truth_sampling": ground_truth_sampling_mode,
        "device": str(device),
        "seed": args.seed,
        "samples": args.samples,
        "scale": args.scale,
        "label": args.label,
        "metrics": metrics,
    }


def print_report(result: dict) -> None:
    metrics = result["metrics"]

    print("Paper-ready Chamfer distance")
    print(f"  Reconstruction: {result['reconstruction']}")
    print(f"  Ground truth:   {result['ground_truth']}")
    print(f"  Reconstruction points: {result['reconstruction_points']}")
    print(f"  Ground-truth points:   {result['ground_truth_points']}")
    print(f"  Reconstruction sampling: {result['reconstruction_sampling']}")
    print(f"  Ground-truth sampling:   {result['ground_truth_sampling']}")
    print(f"  Device: {result['device']}")
    print("  Backend: pytorch3d.loss.chamfer_distance")
    print("  Alignment: none, assumes same reference frame")
    print(f"  Seed: {result['seed']}")
    print(f"  Report scale: {result['scale']:g} ({result['label']})")
    print()
    print("Paper metric")
    print(f"  Accuracy   Reconstruction -> GT: {format_float(metrics['accuracy'])}")
    print(f"  Completion GT -> Reconstruction: {format_float(metrics['completion'])}")
    print(f"  CD         0.5 * (Accuracy + Completion): {format_float(metrics['cd'])}")
    print()
    print("Raw unscaled metric")
    print(f"  Accuracy raw:   {format_float(metrics['accuracy_raw'])}")
    print(f"  Completion raw: {format_float(metrics['completion_raw'])}")
    print(f"  CD raw:         {format_float(metrics['cd_raw'])}")
    print()
    print("Diagnostics")
    print(f"  Median Reconstruction -> GT: {format_float(metrics['median_reconstruction_to_gt'])}")
    print(f"  Median GT -> Reconstruction: {format_float(metrics['median_gt_to_reconstruction'])}")
    print(f"  P95 Reconstruction -> GT:    {format_float(metrics['p95_reconstruction_to_gt'])}")
    print(f"  P95 GT -> Reconstruction:    {format_float(metrics['p95_gt_to_reconstruction'])}")
    print(f"  Max Reconstruction -> GT:    {format_float(metrics['max_reconstruction_to_gt'])}")
    print(f"  Max GT -> Reconstruction:    {format_float(metrics['max_gt_to_reconstruction'])}")
    print()
    print("Copy-paste table value")
    print(f"  CD ↓ = {format_float(metrics['cd'])}")
    print()
    print("PyTorch3D diagnostic")
    print(f"  Reduced squared-L2 chamfer_distance raw: {format_float(metrics['pytorch3d_squared_loss_raw'])}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute symmetric Chamfer distance between a ground-truth mesh and a SuGaR reconstructed OBJ mesh."
    )
    parser.add_argument("ground_truth", type=Path, help="Path to ground-truth mesh or point cloud, usually .ply.")
    parser.add_argument("reconstruction", type=Path, help="Path to SuGaR reconstructed mesh, usually .obj.")
    parser.add_argument("--samples", type=int, default=50_000, help="Number of sampled surface points per mesh.")
    parser.add_argument("--use-vertices", action="store_true", help="Use raw vertices instead of uniform mesh surface sampling.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible surface sampling.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to reported metrics. Use 1000 for CD x 10^3.")
    parser.add_argument("--label", type=str, default="scene units", help="Label printed next to the reported metric scale.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path for saving machine-readable JSON results.")
    parser.add_argument("--quiet", action="store_true", help="Only write JSON output, do not print the text report.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    result = evaluate_chamfer(args)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not args.quiet:
        print_report(result)


if __name__ == "__main__":
    main()
