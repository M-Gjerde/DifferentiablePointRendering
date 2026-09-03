from __future__ import annotations

import argparse
import csv
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
    DatasetConfig("dragon_10", "dragon_10_neus", "dragon.ply"),
    DatasetConfig("horse_10", "horse_10_neus", "horse.ply"),
    DatasetConfig("lego_10", "lego_10_neus", "lego.ply"),
    DatasetConfig("plant_10", "plant_10_neus", "plant.ply"),
    DatasetConfig("teapot_10", "teapot_10_neus", "teapot.ply"),
)


def require_existing_path(path: Path, description: str) -> Path:
    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Could not find {description}: {resolved_path}")

    return resolved_path


def resolve_ground_truth_path(ground_truth_root: Path, dataset: DatasetConfig) -> Path:
    ground_truth_path = ground_truth_root.expanduser() / dataset.ground_truth_file
    return require_existing_path(ground_truth_path, f"ground truth for dataset '{dataset.name}'")


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1


def mesh_root_for_dataset(output_root: Path, dataset: DatasetConfig, scene_subdir: str, mesh_subdir: str) -> Path:
    return output_root.expanduser() / dataset.output_folder / scene_subdir / mesh_subdir


def latest_reconstruction_path(
    output_root: Path,
    dataset: DatasetConfig,
    scene_subdir: str,
    mesh_subdir: str,
) -> ReconstructionCandidate:
    mesh_root = require_existing_path(
        mesh_root_for_dataset(
            output_root=output_root,
            dataset=dataset,
            scene_subdir=scene_subdir,
            mesh_subdir=mesh_subdir,
        ),
        f"NeuS mesh folder for dataset '{dataset.name}'",
    )

    candidates = [
        ReconstructionCandidate(
            reconstruction_path=path.expanduser().resolve(),
            iteration=numeric_stem(path),
            modified_time=path.stat().st_mtime,
        )
        for path in mesh_root.glob("*.ply")
        if numeric_stem(path) >= 0
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find latest NeuS reconstruction for dataset '{dataset.name}'. "
            f"Expected files matching: {mesh_root}/*.ply"
        )

    candidates.sort(key=lambda candidate: (candidate.iteration, candidate.modified_time), reverse=True)
    return candidates[0]


def resolve_reconstruction_path(
    output_root: Path,
    dataset: DatasetConfig,
    scene_subdir: str,
    mesh_subdir: str,
    iteration: int | None,
) -> ReconstructionCandidate:
    if iteration is None:
        return latest_reconstruction_path(
            output_root=output_root,
            dataset=dataset,
            scene_subdir=scene_subdir,
            mesh_subdir=mesh_subdir,
        )

    reconstruction_path = (
        mesh_root_for_dataset(
            output_root=output_root,
            dataset=dataset,
            scene_subdir=scene_subdir,
            mesh_subdir=mesh_subdir,
        )
        / f"{iteration:08d}.ply"
    )
    resolved_reconstruction_path = require_existing_path(
        reconstruction_path,
        f"NeuS reconstruction for dataset '{dataset.name}' at iteration {iteration}",
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
    print("| Dataset | Iteration | CD ↓ | Accuracy ↓ | Completion ↓ |")
    print("|---|---:|---:|---:|---:|")

    for row in rows:
        print(
            f"| {row['dataset']} "
            f"| {row['iteration']} "
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
            f"|  "
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
    parser = argparse.ArgumentParser(description="Run Chamfer evaluation over multiple NeuS datasets.")

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

    parser.add_argument("--samples", type=int, default=50_000, help="Number of sampled surface points per mesh.")
    parser.add_argument("--use-vertices", action="store_true", help="Use raw vertices instead of uniform mesh surface sampling.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible surface sampling.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to reported metrics.")
    parser.add_argument("--label", type=str, default="scene units", help="Metric scale label.")
    parser.add_argument("--digits", type=int, default=5, help="Digits used in the printed markdown table.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing datasets instead of failing.")
    parser.add_argument("--csv-output", type=Path, default=Path("neus_chamfer_results.csv"), help="Output CSV path.")

    return parser.parse_args()


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
