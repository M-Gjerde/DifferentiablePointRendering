#!/usr/bin/env python3
"""Evaluate PGSR meshes with the same symmetric point-to-triangle metric used for 2DGS."""
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
else:
    from chamfer_ours import (
        compute_paper_ready_point_to_triangle_distance,
        load_triangle_mesh_with_query_points,
        set_random_seed,
    )


DEFAULT_OUTPUT_ROOT = Path("/home/magnus/phd/pbdr/PGSR/output")
DEFAULT_GROUND_TRUTH_ROOT = Path("/home/magnus/phd/models")
RUNTIME_DEFINITION = "PGSR runtime is left blank unless supplied separately; point counts are read from checkpoint PLY headers."


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
    checkpoint_path: Path | None


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


def read_ply_vertex_count(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        with path.open("rb") as ply_file:
            for _ in range(10000):
                line = ply_file.readline()
                if not line:
                    break
                text = line.decode("ascii", errors="strict").strip()
                if text.startswith("element vertex "):
                    return int(text.split()[2])
                if text == "end_header":
                    break
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return None


def discover_reconstructions(dataset: str, dataset_root: Path, reconstruction_name: str) -> list[Reconstruction]:
    """Discover meshes produced by the current PGSR renderer.

    Expected layout:
        <dataset_root>/train/ours_<iteration>/<reconstruction_name>
    """
    train_root = dataset_root / "train"
    reconstructions: list[Reconstruction] = []

    if train_root.is_dir():
        for reconstruction_path in train_root.glob(f"ours_*/{reconstruction_name}"):
            match = re.fullmatch(r"ours_(\d+)", reconstruction_path.parent.name)
            if match is None or not reconstruction_path.is_file():
                continue

            iteration = int(match.group(1))
            checkpoint_path = dataset_root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
            reconstructions.append(
                Reconstruction(
                    dataset=dataset,
                    method=f"ours_{iteration}",
                    iteration=iteration,
                    path=reconstruction_path.resolve(),
                    checkpoint_path=checkpoint_path if checkpoint_path.is_file() else None,
                )
            )

    reconstructions.sort(key=lambda reconstruction: (reconstruction.iteration, reconstruction.method))
    if not reconstructions:
        expected = dataset_root / "train" / "ours_<iteration>" / reconstruction_name
        raise FileNotFoundError(f"No PGSR reconstructions found. Expected {expected}")
    return reconstructions


def parse_dataset_name(directory_name: str, view_count: int | None) -> str | None:
    if not directory_name.startswith("pgsr_"):
        return None
    name = directory_name[len("pgsr_"):]
    if view_count is not None:
        suffix = f"_{view_count}"
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name or None


def selected_dataset_names(dataset_filter: str | None, output_root: Path,
                           view_count: int | None) -> list[str]:
    if dataset_filter is not None:
        selected = [name.strip() for name in dataset_filter.split(",") if name.strip()]
        if not selected:
            raise ValueError("--datasets must contain at least one dataset name")
        return selected

    names: list[str] = []
    for path in sorted(output_root.expanduser().resolve().glob("pgsr_*")):
        if not path.is_dir():
            continue
        if view_count is not None and not path.name.endswith(f"_{view_count}"):
            # Also allow unsuffixed folders such as pgsr_restaurant.
            unsuffixed_name = parse_dataset_name(path.name, None)
            if unsuffixed_name is None or re.search(r"_\d+$", unsuffixed_name):
                continue
        name = parse_dataset_name(path.name, view_count)
        if name is not None:
            names.append(name)
    return list(dict.fromkeys(names))


def resolve_dataset_root(output_root: Path, name: str, view_count: int | None) -> Path:
    candidates: list[Path] = []
    if view_count is not None:
        candidates.append(output_root / f"pgsr_{name}_{view_count}")
    candidates.append(output_root / f"pgsr_{name}")
    candidates.append(output_root / name)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise NotADirectoryError(
        "Could not find PGSR dataset directory. Tried:\n" +
        "\n".join(f"  {candidate}" for candidate in candidates)
    )


def resolve_datasets(args: argparse.Namespace) -> list[Dataset]:
    output_root = args.output_root.expanduser().resolve()
    ground_truth_root = args.ground_truth_root.expanduser().resolve()
    names = selected_dataset_names(args.datasets, output_root, args.view_count)
    if not names:
        raise FileNotFoundError(f"No pgsr_* datasets found under {output_root}")

    datasets: list[Dataset] = []
    for name in names:
        dataset_root = resolve_dataset_root(output_root, name, args.view_count)
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
            point_count = read_ply_vertex_count(reconstruction.checkpoint_path)

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
                    "runtime_seconds": "",
                    "point_count": point_count if point_count is not None else "",
                    "reconstruction_query_points": len(reconstruction_points),
                    "ground_truth_query_points": len(ground_truth_points),
                    "reconstruction_query_mode": reconstruction_sampling,
                    "ground_truth_query_mode": ground_truth_sampling,
                    "distance_mode": "symmetric_point_to_triangle",
                    "distance_backend": "open3d_raycasting_cpu",
                    "scale": args.scale,
                    "reconstruction": str(reconstruction.path),
                    "ground_truth": str(dataset.ground_truth),
                    "checkpoint": str(reconstruction.checkpoint_path) if reconstruction.checkpoint_path else "",
                }
            )
    return rows


def write_csv(csv_path: Path, rows: list[dict[str, object]]) -> Path:
    if not rows:
        raise RuntimeError("No evaluation rows were produced")
    resolved_path = csv_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return resolved_path


def summarize_iterations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_iteration: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_iteration.setdefault(int(row["iteration"]), []).append(row)

    summaries: list[dict[str, object]] = []
    for iteration, iteration_rows in sorted(rows_by_iteration.items()):
        point_counts = [
            value
            for row in iteration_rows
            if (value := safe_number(row.get("point_count"))) is not None
        ]
        summaries.append(
            {
                "iteration": iteration,
                "scene_count": len(iteration_rows),
                "point_count_scene_count": len(point_counts),
                "mean_cd": sum(float(row["cd"]) for row in iteration_rows) / len(iteration_rows),
                "mean_accuracy": sum(float(row["accuracy"]) for row in iteration_rows) / len(iteration_rows),
                "mean_completion": sum(float(row["completion"]) for row in iteration_rows) / len(iteration_rows),
                "average_point_count": sum(point_counts) / len(point_counts) if point_counts else "",
            }
        )
    return summaries


def write_summary_files(results_csv_path: Path, summaries: list[dict[str, object]]) -> tuple[Path, Path]:
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
        json.dumps({"runtime_definition": RUNTIME_DEFINITION, "iterations": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_csv_path, summary_json_path


def print_markdown_table(rows: list[dict[str, object]], digits: int) -> None:
    print()
    print("| Dataset | Method | Iteration | CD ↓ | Accuracy ↓ | Completion ↓ | Points ↓ |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        point_count = safe_number(row.get("point_count"))
        point_count_text = f"{point_count:.0f}" if point_count is not None else ""
        print(
            f"| {row['dataset']} | {row['method']} | {int(row['iteration'])} "
            f"| {float(row['cd']):.{digits}f} | {float(row['accuracy']):.{digits}f} "
            f"| {float(row['completion']):.{digits}f} | {point_count_text} |"
        )

    for summary in summarize_iterations(rows):
        point_count = safe_number(summary["average_point_count"])
        point_count_text = f"{point_count:.0f}" if point_count is not None else ""
        print(
            f"| **Mean @ {int(summary['iteration'])}** |  | {int(summary['iteration'])} "
            f"| **{float(summary['mean_cd']):.{digits}f}** "
            f"| **{float(summary['mean_accuracy']):.{digits}f}** "
            f"| **{float(summary['mean_completion']):.{digits}f}** "
            f"| **{point_count_text}** |"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PGSR meshes using symmetric point-to-triangle mesh distance."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Folder containing pgsr_<dataset> directories.")
    parser.add_argument("--ground-truth-root", type=Path, default=DEFAULT_GROUND_TRUTH_ROOT,
                        help="Folder containing <dataset>.ply ground-truth meshes.")
    parser.add_argument("--view-count", type=int, default=None,
                        help="Optional suffix selecting pgsr_<dataset>_<view-count>; unsuffixed folders remain supported.")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Optional comma-separated dataset names. Default: discover pgsr_* folders.")
    parser.add_argument("--reconstruction-name", type=str, default="fuse_post_auto.ply",
                        help="Mesh filename inside train/ours_<iteration> directories.")
    parser.add_argument("--use-vertices", action=argparse.BooleanOptionalAction, default=False,
                        help="Use raw mesh vertices instead of uniform surface samples.")
    parser.add_argument("--samples", type=int, default=5_000_000,
                        help="Uniform surface query samples per mesh (default: 5,000,000).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--digits", type=int, default=5)
    parser.add_argument("--csv-output", type=Path, default=Path("pgsr_point_to_triangle_results.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    rows = evaluate_reconstructions(args)
    csv_path = write_csv(args.csv_output, rows)
    summaries = summarize_iterations(rows)
    summary_csv_path, summary_json_path = write_summary_files(csv_path, summaries)
    print_markdown_table(rows, args.digits)
    print()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
