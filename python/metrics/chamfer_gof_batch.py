from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    output_folder: str
    ground_truth_file: str


@dataclass(frozen=True)
class ReconstructionCandidate:
    reconstruction_path: Path
    iteration: int
    modified_time: float


DEFAULT_DATASETS: tuple[DatasetConfig, ...] = (
    #DatasetConfig("dragon_10", "gof_dragon_10", "dragon.ply"),
    DatasetConfig("horse_10", "gof_horse_10", "horse.ply"),
    #DatasetConfig("lego_10", "gof_lego_10", "lego.ply"),
    DatasetConfig("plant_10", "gof_plant_10", "plant.ply"),
    DatasetConfig("teapot_10", "gof_teapot_10", "teapot.ply"),
)


def require_existing_path(path: Path, description: str) -> Path:
    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Could not find {description}: {resolved_path}")

    return resolved_path


def resolve_ground_truth_path(ground_truth_root: Path, dataset: DatasetConfig) -> Path:
    ground_truth_path = ground_truth_root.expanduser() / dataset.ground_truth_file
    return require_existing_path(ground_truth_path, f"ground truth for dataset '{dataset.name}'")


def parse_ours_iteration(path: Path) -> int:
    match = re.match(r"^ours_(\d+)$", path.name)

    if match is None:
        return -1

    return int(match.group(1))


def latest_reconstruction_path(
    output_root: Path,
    dataset: DatasetConfig,
    split: str,
    fusion_subdir: str,
    reconstruction_name: str,
) -> ReconstructionCandidate:
    dataset_root = require_existing_path(output_root.expanduser() / dataset.output_folder, f"GOF output folder for dataset '{dataset.name}'")
    candidates: list[ReconstructionCandidate] = []

    for reconstruction_path in dataset_root.glob(f"{split}/ours_*/{fusion_subdir}/{reconstruction_name}"):
        resolved_reconstruction_path = reconstruction_path.expanduser().resolve()
        candidates.append(
            ReconstructionCandidate(
                reconstruction_path=resolved_reconstruction_path,
                iteration=parse_ours_iteration(reconstruction_path.parent.parent.name),
                modified_time=resolved_reconstruction_path.stat().st_mtime,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            f"Could not find latest GOF reconstruction for dataset '{dataset.name}'. "
            f"Expected files matching: {dataset_root}/{split}/ours_*/{fusion_subdir}/{reconstruction_name}"
        )

    candidates.sort(key=lambda candidate: (candidate.iteration, candidate.modified_time), reverse=True)
    return candidates[0]


def resolve_reconstruction_path(
    output_root: Path,
    dataset: DatasetConfig,
    split: str,
    fusion_subdir: str,
    reconstruction_name: str,
    iteration: int | None,
) -> ReconstructionCandidate:
    if iteration is None:
        return latest_reconstruction_path(
            output_root=output_root,
            dataset=dataset,
            split=split,
            fusion_subdir=fusion_subdir,
            reconstruction_name=reconstruction_name,
        )

    reconstruction_path = (
        output_root.expanduser()
        / dataset.output_folder
        / split
        / f"ours_{iteration}"
        / fusion_subdir
        / reconstruction_name
    )
    resolved_reconstruction_path = require_existing_path(
        reconstruction_path,
        f"GOF reconstruction for dataset '{dataset.name}' at iteration {iteration}",
    )

    return ReconstructionCandidate(
        reconstruction_path=resolved_reconstruction_path,
        iteration=iteration,
        modified_time=resolved_reconstruction_path.stat().st_mtime,
    )


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


def write_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "iteration",
        "reconstruction_name",
        "cd",
        "accuracy",
        "completion",
        "median_reconstruction_to_gt",
        "median_gt_to_reconstruction",
        "p95_reconstruction_to_gt",
        "p95_gt_to_reconstruction",
        "max_reconstruction_to_gt",
        "max_gt_to_reconstruction",
        "reconstruction_points",
        "ground_truth_points",
        "reconstruction_sampling",
        "ground_truth_sampling",
        "ground_truth",
        "reconstruction",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({fieldname: row[fieldname] for fieldname in fieldnames})


def format_metric(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def print_markdown_table(rows: list[dict], digits: int) -> None:
    print()
    print("| Dataset | CD ↓ | Accuracy ↓ | Completion ↓ |")
    print("|---|---:|---:|---:|")

    for row in rows:
        print(
            f"| {row['dataset']} "
            f"| {format_metric(row['cd'], digits)} "
            f"| {format_metric(row['accuracy'], digits)} "
            f"| {format_metric(row['completion'], digits)} |"
        )

    if rows:
        mean_cd = sum(row["cd"] for row in rows) / len(rows)
        mean_accuracy = sum(row["accuracy"] for row in rows) / len(rows)
        mean_completion = sum(row["completion"] for row in rows) / len(rows)

        print(
            f"| **Mean** "
            f"| **{format_metric(mean_cd, digits)}** "
            f"| **{format_metric(mean_accuracy, digits)}** "
            f"| **{format_metric(mean_completion, digits)}** |"
        )


def evaluate_dataset(args: argparse.Namespace, dataset: DatasetConfig, device) -> dict:
    from chamfer_ours import (
        compute_paper_ready_chamfer,
        load_points_from_ply,
        set_random_seed,
    )

    ground_truth_path = resolve_ground_truth_path(args.ground_truth_root, dataset)
    reconstruction_candidate = resolve_reconstruction_path(
        output_root=args.output_root,
        dataset=dataset,
        split=args.split,
        fusion_subdir=args.fusion_subdir,
        reconstruction_name=args.reconstruction_name,
        iteration=args.iteration,
    )

    print(f"Evaluating {dataset.name}")
    print(f"  GT:        {ground_truth_path}")
    print(f"  Rec:       {reconstruction_candidate.reconstruction_path}")
    print(f"  Iteration: {reconstruction_candidate.iteration}")

    set_random_seed(args.seed)

    reconstruction_points, reconstruction_sampling_mode = load_points_from_ply(
        ply_path=reconstruction_candidate.reconstruction_path,
        sample_count=args.samples,
        use_vertices=args.use_vertices,
    )

    ground_truth_points, ground_truth_sampling_mode = load_points_from_ply(
        ply_path=ground_truth_path,
        sample_count=args.samples,
        use_vertices=args.use_vertices,
    )

    metrics = compute_paper_ready_chamfer(
        reconstruction_points=reconstruction_points,
        ground_truth_points=ground_truth_points,
        device=device,
        scale=args.scale,
    )

    return {
        "dataset": dataset.name,
        "iteration": reconstruction_candidate.iteration,
        "reconstruction_name": args.reconstruction_name,
        "cd": metrics["cd"],
        "accuracy": metrics["accuracy"],
        "completion": metrics["completion"],
        "median_reconstruction_to_gt": metrics["median_reconstruction_to_gt"],
        "median_gt_to_reconstruction": metrics["median_gt_to_reconstruction"],
        "p95_reconstruction_to_gt": metrics["p95_reconstruction_to_gt"],
        "p95_gt_to_reconstruction": metrics["p95_gt_to_reconstruction"],
        "max_reconstruction_to_gt": metrics["max_reconstruction_to_gt"],
        "max_gt_to_reconstruction": metrics["max_gt_to_reconstruction"],
        "reconstruction_points": len(reconstruction_points),
        "ground_truth_points": len(ground_truth_points),
        "reconstruction_sampling": reconstruction_sampling_mode,
        "ground_truth_sampling": ground_truth_sampling_mode,
        "ground_truth": str(ground_truth_path),
        "reconstruction": str(reconstruction_candidate.reconstruction_path),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chamfer evaluation over multiple Gaussian Opacity Fields datasets.")

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/magnus/projects/gaussian-opacity-fields/output"),
        help="Root folder containing GOF output folders.",
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
        default=30000,
        help="GOF ours_<iteration> folder to evaluate. Use -1 to select the latest ours_* folder.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split folder inside each GOF output folder.",
    )

    parser.add_argument(
        "--fusion-subdir",
        type=str,
        default="fusion",
        help="Fusion folder inside each ours_<iteration> folder.",
    )

    parser.add_argument(
        "--reconstruction-name",
        type=str,
        default="mesh_binary_search_7.ply",
        help="GOF reconstructed mesh filename inside each fusion folder.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated subset, e.g. dragon_10,horse_10,lego_10,plant_10,teapot_10.",
    )

    parser.add_argument("--samples", type=int, default=50_000, help="Number of sampled surface points per mesh.")
    parser.add_argument("--use-vertices", action="store_true", help="Use raw vertices instead of uniform mesh surface sampling.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible surface sampling.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to reported metrics.")
    parser.add_argument("--label", type=str, default="scene units", help="Metric scale label.")
    parser.add_argument("--digits", type=int, default=5, help="Digits used in the printed markdown table.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing datasets instead of failing.")
    parser.add_argument("--csv-output", type=Path, default=Path("gof_chamfer_results.csv"), help="Output CSV path.")

    args = parser.parse_args()
    args.iteration = None if args.iteration < 0 else args.iteration
    return args


def main() -> None:
    args = parse_arguments()

    from chamfer_ours import resolve_device

    device = resolve_device(args.device)
    rows: list[dict] = []

    for dataset in selected_datasets(args.datasets):
        try:
            row = evaluate_dataset(args, dataset, device)
            rows.append(row)
        except Exception as exception:
            if not args.allow_missing:
                raise

            print(f"Skipping {dataset.name}: {exception}")

    csv_output_path = args.csv_output.expanduser().resolve()
    write_csv(csv_output_path, rows)

    print_markdown_table(rows, args.digits)
    print()
    print(f"Saved CSV: {csv_output_path}")


if __name__ == "__main__":
    main()
