from __future__ import annotations

import argparse
import re
from datetime import datetime
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


def parse_run_timestamp(path_name: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", path_name)

    if match is None:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def parse_latest_timestamp_from_path(path: Path) -> datetime | None:
    for path_part in reversed(path.parts):
        parsed_timestamp = parse_run_timestamp(path_part)

        if parsed_timestamp is not None:
            return parsed_timestamp

    return None


def validate_points(points: np.ndarray, model_path: Path) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points from {model_path}, got shape {points.shape}")

    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    if len(points) == 0:
        raise ValueError(f"No valid finite points found in: {model_path}")

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


def load_points_with_trimesh(model_path: Path, sample_count: int, use_vertices: bool) -> tuple[np.ndarray, str]:
    if trimesh is None:
        raise ModuleNotFoundError(
            "Open3D could not load this model as a triangle mesh, and trimesh is not installed.\n"
            "Install it with:\n"
            "  pip install trimesh"
        )

    loaded_object = trimesh.load(str(model_path), force="mesh", process=False)

    if isinstance(loaded_object, trimesh.Scene):
        geometry_list = [geometry for geometry in loaded_object.geometry.values()]

        if len(geometry_list) == 0:
            raise ValueError(f"Trimesh loaded an empty scene from: {model_path}")

        loaded_object = trimesh.util.concatenate(geometry_list)

    if not isinstance(loaded_object, trimesh.Trimesh):
        raise ValueError(f"Trimesh could not load a mesh from: {model_path}")

    vertices = np.asarray(loaded_object.vertices, dtype=np.float32)

    if len(vertices) == 0:
        raise ValueError(f"Trimesh loaded no vertices from: {model_path}")

    if use_vertices:
        return validate_points(vertices, model_path), "trimesh_vertices"

    faces_raw = loaded_object.faces

    if faces_raw is None or len(faces_raw) == 0:
        return validate_points(vertices, model_path), "trimesh_vertices_no_faces"

    triangle_faces: list[list[int]] = []

    for face in faces_raw:
        triangle_faces.extend(triangulate_polygon_face(np.asarray(face)))

    if len(triangle_faces) == 0:
        return validate_points(vertices, model_path), "trimesh_vertices_no_valid_faces"

    triangle_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(triangle_faces, dtype=np.int64),
        process=False,
    )

    sampled_points, _ = trimesh.sample.sample_surface(triangle_mesh, sample_count)
    return validate_points(np.asarray(sampled_points, dtype=np.float32), model_path), "trimesh_uniform_surface_sampling"


def load_points_from_model(model_path: Path, sample_count: int, use_vertices: bool) -> tuple[np.ndarray, str]:
    if not model_path.exists():
        raise FileNotFoundError(f"File does not exist: {model_path}")

    mesh = o3d.io.read_triangle_mesh(str(model_path))

    if not mesh.is_empty() and len(mesh.vertices) > 0:
        vertex_points = np.asarray(mesh.vertices, dtype=np.float32)
        triangle_count = len(mesh.triangles)

        if use_vertices:
            return validate_points(vertex_points, model_path), "open3d_vertices"

        if triangle_count > 0:
            sampled_point_cloud = mesh.sample_points_uniformly(number_of_points=sample_count)
            sampled_points = np.asarray(sampled_point_cloud.points, dtype=np.float32)
            return validate_points(sampled_points, model_path), "open3d_uniform_surface_sampling"

    try:
        return load_points_with_trimesh(
            model_path=model_path,
            sample_count=sample_count,
            use_vertices=use_vertices,
        )
    except Exception as trimesh_exception:
        point_cloud = o3d.io.read_point_cloud(str(model_path))

        if not point_cloud.is_empty() and len(point_cloud.points) > 0:
            points = np.asarray(point_cloud.points, dtype=np.float32)
            return validate_points(points, model_path), "open3d_point_cloud_fallback"

        raise RuntimeError(
            f"Could not load {model_path} as triangle mesh, polygon mesh, or point cloud.\n"
            f"Trimesh error was:\n{trimesh_exception}"
        ) from trimesh_exception


def find_reconstruction_path_by_index(
    search_root: Path,
    run_index: int,
    reconstruction_glob: str,
    name_contains: str | None,
) -> Path:
    if run_index < 0:
        raise ValueError(f"--index must be >= 0, got {run_index}")

    if not search_root.exists():
        raise FileNotFoundError(f"SuGaR output folder does not exist: {search_root}")

    candidate_paths = [path for path in search_root.glob(reconstruction_glob) if path.is_file()]

    if name_contains is not None:
        lowered_name_contains = name_contains.lower()
        candidate_paths = [path for path in candidate_paths if lowered_name_contains in path.name.lower()]

    candidate_reconstructions: list[dict] = []

    for reconstruction_path in candidate_paths:
        candidate_reconstructions.append(
            {
                "reconstruction_path": reconstruction_path,
                "parsed_timestamp": parse_latest_timestamp_from_path(reconstruction_path),
                "modified_time": reconstruction_path.stat().st_mtime,
            }
        )

    if not candidate_reconstructions:
        name_filter_text = "" if name_contains is None else f" containing '{name_contains}'"
        raise FileNotFoundError(
            f"No SuGaR reconstruction .obj files{name_filter_text} matching glob "
            f"'{reconstruction_glob}' found under: {search_root}"
        )

    candidate_reconstructions.sort(
        key=lambda item: (
            item["parsed_timestamp"] is not None,
            item["parsed_timestamp"] if item["parsed_timestamp"] is not None else datetime.min,
            item["modified_time"],
        ),
        reverse=True,
    )

    if run_index >= len(candidate_reconstructions):
        available_reconstructions = [
            f"[{candidate_index}] {candidate['reconstruction_path'].relative_to(search_root)}"
            for candidate_index, candidate in enumerate(candidate_reconstructions)
        ]

        raise IndexError(
            f"--index {run_index} is out of range. "
            f"Found {len(candidate_reconstructions)} matching SuGaR .obj reconstruction files.\n"
            "Available reconstructions:\n" + "\n".join(available_reconstructions)
        )

    selected_candidate = candidate_reconstructions[run_index]
    return selected_candidate["reconstruction_path"]


def resolve_reconstruction_path(args: argparse.Namespace) -> tuple[Path, Path | None, str]:
    if args.reconstruction is not None and args.run_dir is not None:
        raise ValueError("Use either positional reconstruction or --run-dir, not both.")

    if args.reconstruction is not None:
        return args.reconstruction.expanduser().resolve(), None, "explicit reconstruction"

    if args.run_dir is not None:
        search_root = args.run_dir.expanduser().resolve()
        reconstruction_path = find_reconstruction_path_by_index(
            search_root=search_root,
            run_index=args.index,
            reconstruction_glob=args.reconstruction_glob,
            name_contains=args.name_contains,
        )
        return reconstruction_path.resolve(), search_root, f"run dir search, index {args.index}"

    search_root = args.sugar_output_root.expanduser().resolve()
    reconstruction_path = find_reconstruction_path_by_index(
        search_root=search_root,
        run_index=args.index,
        reconstruction_glob=args.reconstruction_glob,
        name_contains=args.name_contains,
    )

    return reconstruction_path.resolve(), search_root, f"SuGaR output search, index {args.index}"


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
def compute_paper_ready_chamfer(
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
    cd = 0.5 * (accuracy + completion)

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
        "cd": float((cd * scale).item()),
        "accuracy_raw": float(accuracy.item()),
        "completion_raw": float(completion.item()),
        "cd_raw": float(cd.item()),
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paper-ready symmetric Chamfer distance for a SuGaR .obj reconstruction."
    )
    parser.add_argument("ground_truth", type=Path, help="Path to ground-truth .ply/.obj mesh or point cloud.")
    parser.add_argument(
        "reconstruction",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Optional path to reconstructed SuGaR .obj mesh. "
            "When omitted, the latest matching .obj is selected from --sugar-output-root."
        ),
    )
    parser.add_argument(
        "--sugar-output-root",
        type=Path,
        default=Path("../output"),
        help="Folder searched recursively for SuGaR .obj reconstructions when no reconstruction path is provided.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit SuGaR run/output folder. The script searches this folder recursively for matching .obj files.",
    )
    parser.add_argument("--index", type=int, default=0, help="0 = latest matching .obj, 1 = second latest, etc.")
    parser.add_argument(
        "--reconstruction-glob",
        type=str,
        default="**/*.obj",
        help="Glob used to find SuGaR .obj reconstructions relative to the search root.",
    )
    parser.add_argument(
        "--name-contains",
        type=str,
        default=None,
        help="Optional filename substring filter, useful if the SuGaR folder contains multiple .obj files.",
    )
    parser.add_argument("--samples", type=int, default=25_000, help="Number of sampled surface points per mesh.")
    parser.add_argument("--use_vertices", action="store_true", help="Use raw vertices instead of uniform mesh surface sampling.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible surface sampling.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to reported metrics. Use 1000 for CD x 10^3.")
    parser.add_argument("--label", type=str, default="scene units", help="Label printed next to the reported metric scale.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    set_random_seed(args.seed)
    device = resolve_device(args.device)

    reconstruction_path, search_root, reconstruction_source = resolve_reconstruction_path(args)
    ground_truth_path = args.ground_truth.expanduser().resolve()

    if reconstruction_path.suffix.lower() != ".obj":
        print(f"Warning: SuGaR reconstruction path does not end with .obj: {reconstruction_path}")

    reconstruction_points, reconstruction_sampling_mode = load_points_from_model(
        model_path=reconstruction_path,
        sample_count=args.samples,
        use_vertices=args.use_vertices,
    )

    ground_truth_points, ground_truth_sampling_mode = load_points_from_model(
        model_path=ground_truth_path,
        sample_count=args.samples,
        use_vertices=args.use_vertices,
    )

    results = compute_paper_ready_chamfer(
        reconstruction_points=reconstruction_points,
        ground_truth_points=ground_truth_points,
        device=device,
        scale=args.scale,
    )

    print("Paper-ready Chamfer distance")
    print(f"  Reconstruction: {reconstruction_path}")
    print(f"  Ground truth:   {ground_truth_path}")
    print(f"  Reconstruction source: {reconstruction_source}")

    if search_root is not None:
        print(f"  Search folder:  {search_root}")

    print(f"  Reconstruction points: {len(reconstruction_points)}")
    print(f"  Ground-truth points:   {len(ground_truth_points)}")
    print(f"  Reconstruction sampling: {reconstruction_sampling_mode}")
    print(f"  Ground-truth sampling:   {ground_truth_sampling_mode}")
    print(f"  Device: {device}")
    print(f"  Backend: pytorch3d.loss.chamfer_distance")
    print(f"  Alignment: none, assumes same reference frame")
    print(f"  Seed: {args.seed}")
    print(f"  Report scale: {args.scale:g} ({args.label})")
    print()
    print("Paper metric")
    print(f"  Accuracy   Reconstruction -> GT: {format_float(results['accuracy'])}")
    print(f"  Completion GT -> Reconstruction: {format_float(results['completion'])}")
    print(f"  CD         0.5 * (Accuracy + Completion): {format_float(results['cd'])}")
    print()
    print("Raw unscaled metric")
    print(f"  Accuracy raw:   {format_float(results['accuracy_raw'])}")
    print(f"  Completion raw: {format_float(results['completion_raw'])}")
    print(f"  CD raw:         {format_float(results['cd_raw'])}")
    print()
    print("Diagnostics")
    print(f"  Median Reconstruction -> GT: {format_float(results['median_reconstruction_to_gt'])}")
    print(f"  Median GT -> Reconstruction: {format_float(results['median_gt_to_reconstruction'])}")
    print(f"  P95 Reconstruction -> GT:    {format_float(results['p95_reconstruction_to_gt'])}")
    print(f"  P95 GT -> Reconstruction:    {format_float(results['p95_gt_to_reconstruction'])}")
    print(f"  Max Reconstruction -> GT:    {format_float(results['max_reconstruction_to_gt'])}")
    print(f"  Max GT -> Reconstruction:    {format_float(results['max_gt_to_reconstruction'])}")
    print()
    print("Copy-paste table value")
    print(f"  CD ↓ = {format_float(results['cd'])}")
    print()
    print("PyTorch3D diagnostic")
    print(f"  Reduced squared-L2 chamfer_distance raw: {format_float(results['pytorch3d_squared_loss_raw'])}")


if __name__ == "__main__":
    main()