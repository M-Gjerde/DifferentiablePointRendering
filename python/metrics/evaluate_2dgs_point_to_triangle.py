from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from chamfer_ours import (
    compute_paper_ready_point_to_triangle_distance,
    load_triangle_mesh_with_query_points,
    set_random_seed,
)


DEFAULT_OUTPUT_ROOT = Path("/home/magnus/projects/2D-GS-Viser-Viewer/output")
DEFAULT_GROUND_TRUTH_ROOT = Path("/home/magnus/phd/models")
DEFAULT_DATASETS = ("dragon", "horse", "lego", "plant", "teapot")


@dataclass(frozen=True)
class Dataset:
    name: str
    root: Path
    ground_truth: Path


@dataclass(frozen=True)
class Reconstruction:
    dataset: str
    method: str
    iteration: int
    path: Path


def require_file(path: Path, description: str) -> Path:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Could not find {description}: {resolved_path}")
    return resolved_path


def discover_reconstructions(
    dataset: str,
    dataset_root: Path,
    reconstruction_name: str,
) -> list[Reconstruction]:
    train_root = dataset_root.expanduser().resolve() / "train"
    if not train_root.is_dir():
        raise NotADirectoryError(f"Could not find 2DGS train directory: {train_root}")

    reconstructions: list[Reconstruction] = []
    for reconstruction_path in train_root.glob(f"ours_*/{reconstruction_name}"):
        method = reconstruction_path.parent.name
        match = re.fullmatch(r"ours_(\d+)", method)
        if match is None:
            continue
        reconstructions.append(
            Reconstruction(
                dataset=dataset,
                method=method,
                iteration=int(match.group(1)),
                path=reconstruction_path.resolve(),
            )
        )

    reconstructions.sort(key=lambda reconstruction: (reconstruction.iteration, reconstruction.method))
    if not reconstructions:
        raise FileNotFoundError(
            f"No reconstructions matched {train_root}/ours_*/{reconstruction_name}"
        )
    return reconstructions


def selected_dataset_names(dataset_filter: str | None) -> list[str]:
    if dataset_filter is None:
        return list(DEFAULT_DATASETS)

    selected = [name.strip() for name in dataset_filter.split(",") if name.strip()]
    if not selected:
        raise ValueError("--datasets must contain at least one dataset name")
    return selected


def resolve_datasets(args: argparse.Namespace) -> list[Dataset]:
    output_root = args.output_root.expanduser().resolve()
    ground_truth_root = args.ground_truth_root.expanduser().resolve()

    datasets: list[Dataset] = []
    for name in selected_dataset_names(args.datasets):
        view_count_suffix = f"_{args.view_count}" if args.view_count is not None else ""
        dataset_root = output_root / f"2dgs_{name}{view_count_suffix}"
        if not dataset_root.is_dir():
            raise NotADirectoryError(f"Could not find 2DGS dataset directory: {dataset_root}")
        datasets.append(
            Dataset(
                name=name,
                root=dataset_root,
                ground_truth=require_file(
                    ground_truth_root / f"{name}.ply",
                    f"ground-truth mesh for {name}",
                ),
            )
        )
    return datasets


def evaluate_reconstructions(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in resolve_datasets(args):
        reconstructions = discover_reconstructions(
            dataset=dataset.name,
            dataset_root=dataset.root,
            reconstruction_name=args.reconstruction_name,
        )
        set_random_seed(args.seed)
        ground_truth_mesh, ground_truth_points, ground_truth_sampling = (
            load_triangle_mesh_with_query_points(
                ply_path=dataset.ground_truth,
                sample_count=args.samples,
                use_vertices=args.use_vertices,
            )
        )

        for reconstruction in reconstructions:
            print(f"Evaluating {dataset.name} / {reconstruction.method}")
            print(f"  Reconstruction: {reconstruction.path}")
            print(f"  Ground truth:   {dataset.ground_truth}")

            set_random_seed(args.seed)
            reconstruction_mesh, reconstruction_points, reconstruction_sampling = (
                load_triangle_mesh_with_query_points(
                    ply_path=reconstruction.path,
                    sample_count=args.samples,
                    use_vertices=args.use_vertices,
                )
            )
            metrics = compute_paper_ready_point_to_triangle_distance(
                reconstruction_points=reconstruction_points,
                reconstruction_mesh=reconstruction_mesh,
                ground_truth_points=ground_truth_points,
                ground_truth_mesh=ground_truth_mesh,
                scale=args.scale,
            )

            rows.append(
                {
                    "dataset": reconstruction.dataset,
                    "method": reconstruction.method,
                    "iteration": reconstruction.iteration,
                    "cd": metrics["cd"],
                    "accuracy": metrics["accuracy"],
                    "completion": metrics["completion"],
                    "median_reconstruction_to_gt": metrics["median_reconstruction_to_gt"],
                    "median_gt_to_reconstruction": metrics["median_gt_to_reconstruction"],
                    "p95_reconstruction_to_gt": metrics["p95_reconstruction_to_gt"],
                    "p95_gt_to_reconstruction": metrics["p95_gt_to_reconstruction"],
                    "max_reconstruction_to_gt": metrics["max_reconstruction_to_gt"],
                    "max_gt_to_reconstruction": metrics["max_gt_to_reconstruction"],
                    "reconstruction_query_points": len(reconstruction_points),
                    "ground_truth_query_points": len(ground_truth_points),
                    "reconstruction_query_mode": reconstruction_sampling,
                    "ground_truth_query_mode": ground_truth_sampling,
                    "distance_mode": "symmetric_point_to_triangle",
                    "distance_backend": "open3d_raycasting_cpu",
                    "scale": args.scale,
                    "reconstruction": str(reconstruction.path),
                    "ground_truth": str(dataset.ground_truth),
                }
            )

    return rows


def write_csv(csv_path: Path, rows: list[dict[str, object]]) -> Path:
    resolved_path = csv_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with resolved_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return resolved_path


def print_markdown_table(rows: list[dict[str, object]], digits: int) -> None:
    print()
    print("| Dataset | Method | Iteration | CD ↓ | Accuracy ↓ | Completion ↓ |")
    print("|---|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['dataset']} "
            f"| {row['method']} "
            f"| {int(row['iteration'])} "
            f"| {float(row['cd']):.{digits}f} "
            f"| {float(row['accuracy']):.{digits}f} "
            f"| {float(row['completion']):.{digits}f} |"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 2DGS datasets and every train/ours_*/fuse_post.ply "
            "reconstruction using symmetric point-to-triangle mesh distance."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Folder containing 2dgs_<dataset>_<view-count> directories.",
    )
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_ROOT,
        help="Folder containing <dataset>.ply ground-truth meshes.",
    )
    parser.add_argument(
        "--view-count",
        type=int,
        default=None,
        help=(
            "Optional 2DGS dataset suffix. By default, evaluate 2dgs_<dataset>; "
            "pass N to evaluate 2dgs_<dataset>_N."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Optional comma-separated dataset names; defaults to "
            "dragon,horse,lego,plant,teapot."
        ),
    )
    parser.add_argument(
        "--reconstruction-name",
        type=str,
        default="fuse_post.ply",
        help="Mesh filename inside each train/ours_* directory.",
    )
    parser.add_argument(
        "--use-vertices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use raw mesh vertices as point-to-triangle queries (default). "
            "Pass --no-use-vertices for uniform surface query samples."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500_000,
        help="Uniform query samples per mesh when --no-use-vertices is selected.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--digits", type=int, default=5)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("2dgs_point_to_triangle_results.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    rows = evaluate_reconstructions(args)
    csv_path = write_csv(args.csv_output, rows)
    print_markdown_table(rows, args.digits)
    print()
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
