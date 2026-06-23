from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    output_folder: str
    ground_truth_file: str


DEFAULT_DATASETS: tuple[DatasetConfig, ...] = (
    #DatasetConfig("dragon", "2dgs_dragon", "dragon.ply"),
    #DatasetConfig("horse", "2dgs_horse", "horse.ply"),
    DatasetConfig("lego", "2dgs_lego_15", "lego.ply"),
    #DatasetConfig("plant", "2dgs_plant", "plant.ply"),
    #DatasetConfig("teapot", "2dgs_teapot_10", "teapot.ply"),
)


def require_existing_path(path: Path, description: str) -> Path:
    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Could not find {description}: {resolved_path}")

    return resolved_path


def resolve_ground_truth_path(ground_truth_root: Path, dataset: DatasetConfig) -> Path:
    ground_truth_path = ground_truth_root.expanduser() / dataset.ground_truth_file
    return require_existing_path(ground_truth_path, f"ground truth for dataset '{dataset.name}'")


def resolve_reconstruction_path(output_root: Path, dataset: DatasetConfig, iteration: int | None) -> Path:
    dataset_root = output_root.expanduser() / dataset.output_folder

    if iteration is not None:
        reconstruction_path = dataset_root / "train" / f"ours_{iteration}" / "fuse_post.ply"
        return require_existing_path(
            reconstruction_path,
            f"2DGS reconstruction for dataset '{dataset.name}' at iteration {iteration}",
        )

    train_root = require_existing_path(dataset_root / "train", f"2DGS train folder for dataset '{dataset.name}'")

    reconstructions = sorted(
        train_root.glob("ours_*/fuse_post.ply"),
        key=lambda reconstruction_path: reconstruction_path.stat().st_mtime,
        reverse=True,
    )

    if not reconstructions:
        raise FileNotFoundError(
            f"Could not find latest 2DGS reconstruction for dataset '{dataset.name}'. "
            f"Expected files matching: {train_root}/ours_*/fuse_post.ply"
        )

    return reconstructions[0].expanduser().resolve()


def parse_dataset_filter(dataset_filter: str | None) -> set[str] | None:
    if dataset_filter is None:
        return None

    selected_names = {name.strip() for name in dataset_filter.split(",") if name.strip()}
    return selected_names if selected_names else None


def selected_datasets(dataset_filter: str | None) -> list[DatasetConfig]:
    selected_names = parse_dataset_filter(dataset_filter)

    if selected_names is None:
        return list(DEFAULT_DATASETS)

    known_names = {dataset.name for dataset in DEFAULT_DATASETS}
    unknown_names = selected_names - known_names

    if unknown_names:
        raise ValueError(
            "Unknown dataset names: "
            + ", ".join(sorted(unknown_names))
            + "\nKnown dataset names: "
            + ", ".join(sorted(known_names))
        )

    return [dataset for dataset in DEFAULT_DATASETS if dataset.name in selected_names]


def run_metric_script(
    metric_script: Path,
    ground_truth_path: Path,
    reconstruction_path: Path,
    samples: int,
    device: str,
    seed: int,
    scale: float,
    label: str,
    use_vertices: bool,
) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_file:
        json_output_path = Path(json_file.name)

    command = [
        sys.executable,
        str(metric_script),
        str(ground_truth_path),
        str(reconstruction_path),
        "--samples",
        str(samples),
        "--device",
        device,
        "--seed",
        str(seed),
        "--scale",
        str(scale),
        "--label",
        label,
        "--json-output",
        str(json_output_path),
        "--quiet",
    ]

    if use_vertices:
        command.append("--use-vertices")

    completed_process = subprocess.run(command, check=False, text=True, capture_output=True)

    if completed_process.returncode != 0:
        raise RuntimeError(
            "Metric script failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed_process.stdout}\n"
            f"stderr:\n{completed_process.stderr}"
        )

    try:
        return json.loads(json_output_path.read_text(encoding="utf-8"))
    finally:
        json_output_path.unlink(missing_ok=True)


def write_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "cd",
        "accuracy",
        "completion",
        "median_reconstruction_to_gt",
        "median_gt_to_reconstruction",
        "p95_reconstruction_to_gt",
        "p95_gt_to_reconstruction",
        "reconstruction_points",
        "ground_truth_points",
        "ground_truth",
        "reconstruction",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({fieldname: row[fieldname] for fieldname in fieldnames})


def print_markdown_table(rows: list[dict]) -> None:
    print()
    print("| Dataset | CD ↓ | Accuracy ↓ | Completion ↓ |")
    print("|---|---:|---:|---:|")

    for row in rows:
        print(
            f"| {row['dataset']} "
            f"| {row['cd']:.12f} "
            f"| {row['accuracy']:.12f} "
            f"| {row['completion']:.12f} |"
        )

    if rows:
        mean_cd = sum(row["cd"] for row in rows) / len(rows)
        mean_accuracy = sum(row["accuracy"] for row in rows) / len(rows)
        mean_completion = sum(row["completion"] for row in rows) / len(rows)

        print(
            f"| **Mean** "
            f"| **{mean_cd:.12f}** "
            f"| **{mean_accuracy:.12f}** "
            f"| **{mean_completion:.12f}** |"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chamfer evaluation over multiple 2DGS datasets.")

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/magnus/projects/2D-GS-Viser-Viewer/output"),
        help="Root folder containing 2DGS output folders.",
    )

    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=Path("/home/magnus/phd/models"),
        help="Root folder containing ground-truth PLY meshes.",
    )

    parser.add_argument(
        "--metric-script",
        type=Path,
        default=Path(__file__).with_name("chamfer_2dgs.py"),
        help="Path to chamfer_2dgs.py.",
    )

    parser.add_argument(
        "--iteration",
        type=int,
        default=7000,
        help="2DGS iteration folder to evaluate, e.g. 7000 or 30000. Use -1 to select latest ours_*/fuse_post.ply.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated subset, e.g. dragon,horse,lego,plant,run_30.",
    )

    parser.add_argument("--samples", type=int, default=50_000, help="Number of sampled surface points per mesh.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible surface sampling.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to reported metrics.")
    parser.add_argument("--label", type=str, default="scene units", help="Metric scale label.")
    parser.add_argument("--use-vertices", action="store_true", help="Use raw vertices instead of uniform mesh surface sampling.")
    parser.add_argument("--csv-output", type=Path, default=Path("2dgs_chamfer_results.csv"), help="Output CSV path.")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    metric_script = require_existing_path(args.metric_script, "metric script")
    iteration = None if args.iteration < 0 else args.iteration
    rows: list[dict] = []

    for dataset in selected_datasets(args.datasets):
        ground_truth_path = resolve_ground_truth_path(args.ground_truth_root, dataset)
        reconstruction_path = resolve_reconstruction_path(args.output_root, dataset, iteration)

        print(f"Evaluating {dataset.name}")
        print(f"  GT:  {ground_truth_path}")
        print(f"  Rec: {reconstruction_path}")

        result = run_metric_script(
            metric_script=metric_script,
            ground_truth_path=ground_truth_path,
            reconstruction_path=reconstruction_path,
            samples=args.samples,
            device=args.device,
            seed=args.seed,
            scale=args.scale,
            label=args.label,
            use_vertices=args.use_vertices,
        )

        metrics = result["metrics"]

        rows.append(
            {
                "dataset": dataset.name,
                "cd": metrics["cd"],
                "accuracy": metrics["accuracy"],
                "completion": metrics["completion"],
                "median_reconstruction_to_gt": metrics["median_reconstruction_to_gt"],
                "median_gt_to_reconstruction": metrics["median_gt_to_reconstruction"],
                "p95_reconstruction_to_gt": metrics["p95_reconstruction_to_gt"],
                "p95_gt_to_reconstruction": metrics["p95_gt_to_reconstruction"],
                "reconstruction_points": result["reconstruction_points"],
                "ground_truth_points": result["ground_truth_points"],
                "ground_truth": result["ground_truth"],
                "reconstruction": result["reconstruction"],
            }
        )

    csv_output_path = args.csv_output.expanduser().resolve()
    write_csv(csv_output_path, rows)

    print_markdown_table(rows)
    print()
    print(f"Saved CSV: {csv_output_path}")


if __name__ == "__main__":
    main()