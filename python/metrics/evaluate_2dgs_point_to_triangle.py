from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .chamfer_ours import (
        compute_paper_ready_point_to_triangle_distance,
        load_triangle_mesh_with_query_points,
        set_random_seed,
    )
else:  # Support direct execution: python metrics/evaluate_2dgs_point_to_triangle.py
    from chamfer_ours import (
        compute_paper_ready_point_to_triangle_distance,
        load_triangle_mesh_with_query_points,
        set_random_seed,
    )


DEFAULT_OUTPUT_ROOT = Path("/home/magnus/projects/2D-GS-Viser-Viewer/output")
DEFAULT_GROUND_TRUTH_ROOT = Path("/home/magnus/phd/models")
DEFAULT_DATASETS = ("dragon", "horse", "lego", "plant", "teapot")
RUNTIME_DEFINITION = (
    "Cumulative wall-clock seconds from scene training setup through checkpoint "
    "saving, including evaluation and any GUI waits; 30000 includes 7000."
)


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


def safe_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_training_statistics(dataset_root: Path) -> dict[int, dict[str, float]]:
    stats_path = dataset_root / "training_stats.json"
    if not stats_path.is_file():
        return {}

    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise RuntimeError(f"Could not read 2DGS training statistics: {stats_path}") from exception

    raw_iterations = payload.get("iterations", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw_iterations, dict):
        raise RuntimeError(f"Expected an 'iterations' object in {stats_path}")

    statistics: dict[int, dict[str, float]] = {}
    for iteration_text, raw_stats in raw_iterations.items():
        if not isinstance(raw_stats, dict):
            continue
        try:
            iteration = int(iteration_text)
        except (TypeError, ValueError):
            continue
        runtime_seconds = safe_number(raw_stats.get("runtime_seconds"))
        point_count = safe_number(raw_stats.get("point_count"))
        statistics[iteration] = {
            key: value
            for key, value in (
                ("runtime_seconds", runtime_seconds),
                ("point_count", point_count),
            )
            if value is not None
        }
    return statistics


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


def dataset_names_from_run_summary(output_root: Path, view_count: int | None) -> list[str]:
    summary_path = output_root / "summary.json"
    if not summary_path.is_file():
        return []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scenes = payload.get("scenes", {}) if isinstance(payload, dict) else {}
    if not isinstance(scenes, dict):
        return []

    names: list[str] = []
    for scene in scenes.values():
        if not isinstance(scene, dict) or not isinstance(scene.get("dataset"), str):
            continue
        scene_views = scene.get("views")
        if view_count is not None and scene_views != view_count:
            continue
        names.append(scene["dataset"])
    return list(dict.fromkeys(names))


def selected_dataset_names(
    dataset_filter: str | None,
    output_root: Path,
    view_count: int | None,
) -> list[str]:
    if dataset_filter is None:
        summary_names = dataset_names_from_run_summary(output_root, view_count)
        return summary_names or list(DEFAULT_DATASETS)

    selected = [name.strip() for name in dataset_filter.split(",") if name.strip()]
    if not selected:
        raise ValueError("--datasets must contain at least one dataset name")
    return selected


def resolve_datasets(args: argparse.Namespace) -> list[Dataset]:
    output_root = args.output_root.expanduser().resolve()
    ground_truth_root = args.ground_truth_root.expanduser().resolve()

    datasets: list[Dataset] = []
    for name in selected_dataset_names(args.datasets, output_root, args.view_count):
        view_count_suffix = f"_{args.view_count}" if args.view_count is not None else ""
        dataset_root = output_root / "run1" / f"2dgs_{name}{view_count_suffix}"
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
        training_statistics = load_training_statistics(dataset.root)
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
            checkpoint_statistics = training_statistics.get(reconstruction.iteration, {})
            if not checkpoint_statistics:
                print(
                    "  Warning: no matching runtime/point-count entry in "
                    f"{dataset.root / 'training_stats.json'}"
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
                    "runtime_seconds": checkpoint_statistics.get("runtime_seconds", ""),
                    "point_count": checkpoint_statistics.get("point_count", ""),
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


def summarize_iterations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_iteration: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_iteration.setdefault(int(row["iteration"]), []).append(row)

    summaries: list[dict[str, object]] = []
    for iteration, iteration_rows in sorted(rows_by_iteration.items()):
        runtimes = [
            value
            for row in iteration_rows
            if (value := safe_number(row.get("runtime_seconds"))) is not None
        ]
        point_counts = [
            value
            for row in iteration_rows
            if (value := safe_number(row.get("point_count"))) is not None
        ]
        summaries.append(
            {
                "iteration": iteration,
                "scene_count": len(iteration_rows),
                "runtime_scene_count": len(runtimes),
                "point_count_scene_count": len(point_counts),
                "mean_cd": sum(float(row["cd"]) for row in iteration_rows) / len(iteration_rows),
                "mean_accuracy": (
                    sum(float(row["accuracy"]) for row in iteration_rows) / len(iteration_rows)
                ),
                "mean_completion": (
                    sum(float(row["completion"]) for row in iteration_rows) / len(iteration_rows)
                ),
                "total_runtime_seconds": sum(runtimes) if runtimes else "",
                "average_runtime_seconds": sum(runtimes) / len(runtimes) if runtimes else "",
                "average_point_count": sum(point_counts) / len(point_counts) if point_counts else "",
            }
        )
    return summaries


def write_summary_files(
    results_csv_path: Path,
    summaries: list[dict[str, object]],
) -> tuple[Path, Path]:
    summary_csv_path = results_csv_path.with_name(f"{results_csv_path.stem}_summary.csv")
    summary_json_path = results_csv_path.with_name(f"{results_csv_path.stem}_summary.json")

    if summaries:
        with summary_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    else:
        summary_csv_path.write_text("", encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(
            {
                "runtime_definition": RUNTIME_DEFINITION,
                "iterations": summaries,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return summary_csv_path, summary_json_path


def print_markdown_table(rows: list[dict[str, object]], digits: int) -> None:
    print()
    print("| Dataset | Method | Iteration | CD ↓ | Accuracy ↓ | Completion ↓ | Runtime (s) ↓ | Points ↓ |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        runtime = safe_number(row.get("runtime_seconds"))
        point_count = safe_number(row.get("point_count"))
        runtime_text = f"{runtime:.2f}" if runtime is not None else ""
        point_count_text = f"{point_count:.0f}" if point_count is not None else ""
        print(
            f"| {row['dataset']} "
            f"| {row['method']} "
            f"| {int(row['iteration'])} "
            f"| {float(row['cd']):.{digits}f} "
            f"| {float(row['accuracy']):.{digits}f} "
            f"| {float(row['completion']):.{digits}f} "
            f"| {runtime_text} "
            f"| {point_count_text} |"
        )

    for summary in summarize_iterations(rows):
        runtime = safe_number(summary["average_runtime_seconds"])
        point_count = safe_number(summary["average_point_count"])
        runtime_text = f"{runtime:.2f}" if runtime is not None else ""
        point_count_text = f"{point_count:.0f}" if point_count is not None else ""
        print(
            f"| **Mean @ {int(summary['iteration'])}** "
            f"|  | {int(summary['iteration'])} "
            f"| **{float(summary['mean_cd']):.{digits}f}** "
            f"| **{float(summary['mean_accuracy']):.{digits}f}** "
            f"| **{float(summary['mean_completion']):.{digits}f}** "
            f"| **{runtime_text}** "
            f"| **{point_count_text}** |"
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
        default=10,
        help=(
            "2DGS dataset suffix (default: 10), selecting "
            "2dgs_<dataset>_<view-count>."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Optional comma-separated dataset names. By default, use the scenes "
            "listed in <output-root>/summary.json, falling back to the standard suite."
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
        default=False,
        help=(
            "Use raw mesh vertices as point-to-triangle queries. By default, "
            "query points are sampled uniformly over both mesh surfaces so the "
            "metric is insensitive to tessellation density."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000_000,
        help="Uniform surface query samples per mesh (default: 500000).",
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
    summary_csv_path, summary_json_path = write_summary_files(csv_path, summarize_iterations(rows))
    print_markdown_table(rows, args.digits)
    print()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
