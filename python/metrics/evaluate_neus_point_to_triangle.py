from __future__ import annotations

import argparse
import csv
from pathlib import Path

from chamfer_neus_batch import (
    DatasetConfig,
    resolve_ground_truth_path,
    resolve_reconstruction_path,
    selected_datasets,
)
from chamfer_ours import (
    compute_paper_ready_point_to_triangle_distance,
    load_triangle_mesh_with_query_points,
    set_random_seed,
)


CSV_FIELDNAMES = [
    "dataset",
    "iteration",
    "cd",
    "accuracy",
    "completion",
    "median_reconstruction_to_gt",
    "median_gt_to_reconstruction",
    "p95_reconstruction_to_gt",
    "p95_gt_to_reconstruction",
    "max_reconstruction_to_gt",
    "max_gt_to_reconstruction",
    "reconstruction_query_points",
    "ground_truth_query_points",
    "reconstruction_query_mode",
    "ground_truth_query_mode",
    "distance_mode",
    "distance_backend",
    "scale",
    "reconstruction",
    "ground_truth",
]


def evaluate_dataset(args: argparse.Namespace, dataset: DatasetConfig) -> dict[str, object]:
    iteration = None if args.iteration < 0 else args.iteration
    ground_truth_path = resolve_ground_truth_path(args.ground_truth_root, dataset)
    reconstruction_candidate = resolve_reconstruction_path(
        output_root=args.output_root,
        dataset=dataset,
        scene_subdir=args.scene_subdir,
        mesh_subdir=args.mesh_subdir,
        iteration=iteration,
    )

    print(f"Evaluating {dataset.name}")
    print(f"  GT:        {ground_truth_path}")
    print(f"  Rec:       {reconstruction_candidate.reconstruction_path}")
    print(f"  Iteration: {reconstruction_candidate.iteration}")

    set_random_seed(args.seed)
    ground_truth_mesh, ground_truth_points, ground_truth_sampling = (
        load_triangle_mesh_with_query_points(
            ply_path=ground_truth_path,
            sample_count=args.samples,
            use_vertices=args.use_vertices,
        )
    )

    set_random_seed(args.seed)
    reconstruction_mesh, reconstruction_points, reconstruction_sampling = (
        load_triangle_mesh_with_query_points(
            ply_path=reconstruction_candidate.reconstruction_path,
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

    return {
        "dataset": dataset.name,
        "iteration": reconstruction_candidate.iteration,
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
        "reconstruction": str(reconstruction_candidate.reconstruction_path),
        "ground_truth": str(ground_truth_path),
    }


def write_csv(csv_path: Path, rows: list[dict[str, object]]) -> Path:
    resolved_path = csv_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return resolved_path


def format_metric(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def print_markdown_table(rows: list[dict[str, object]], digits: int) -> None:
    print()
    print("| Dataset | Iteration | CD ↓ | Accuracy ↓ | Completion ↓ |")
    print("|---|---:|---:|---:|---:|")

    for row in rows:
        print(
            f"| {row['dataset']} "
            f"| {row['iteration']} "
            f"| {format_metric(float(row['cd']), digits)} "
            f"| {format_metric(float(row['accuracy']), digits)} "
            f"| {format_metric(float(row['completion']), digits)} |"
        )

    if rows:
        mean_cd = sum(float(row["cd"]) for row in rows) / len(rows)
        mean_accuracy = sum(float(row["accuracy"]) for row in rows) / len(rows)
        mean_completion = sum(float(row["completion"]) for row in rows) / len(rows)

        print(
            f"| **Mean** "
            f"|  "
            f"| **{format_metric(mean_cd, digits)}** "
            f"| **{format_metric(mean_accuracy, digits)}** "
            f"| **{format_metric(mean_completion, digits)}** |"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run symmetric point-to-triangle mesh distance evaluation over "
            "multiple NeuS datasets."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/magnus/projects/NeuS/exp"),
        help="Root folder containing NeuS exp output folders.",
    )
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=Path("/home/magnus/phd/models"),
        help="Root folder containing ground-truth PLY meshes.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=300_000,
        help="NeuS PLY iteration to evaluate, e.g. 300000 for 00300000.ply. Use -1 to select latest *.ply.",
    )
    parser.add_argument(
        "--scene-subdir",
        type=str,
        default="womask_sphere",
        help="Scene output subfolder inside each NeuS dataset folder.",
    )
    parser.add_argument(
        "--mesh-subdir",
        type=str,
        default="meshes",
        help="Mesh folder inside each NeuS scene output folder.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated subset, e.g. dragon_10,horse_10,lego_10,plant_10,teapot_10.",
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible surface sampling.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to reported metrics.")
    parser.add_argument("--digits", type=int, default=5, help="Digits used in the printed markdown table.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing datasets instead of failing.")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("neus_point_to_triangle_results.csv"),
        help="Output CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    rows: list[dict[str, object]] = []

    for dataset in selected_datasets(args.datasets):
        try:
            rows.append(evaluate_dataset(args, dataset))
        except Exception as exception:
            if not args.allow_missing:
                raise

            print(f"Skipping {dataset.name}: {exception}")

    csv_path = write_csv(args.csv_output, rows)
    print_markdown_table(rows, args.digits)
    print()
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
