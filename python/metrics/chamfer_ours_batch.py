from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from chamfer_ours import (
    compute_paper_ready_chamfer,
    load_points_from_ply,
    resolve_device,
    set_random_seed,
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    ground_truth_file: str


@dataclass(frozen=True)
class ReconstructionCandidate:
    run_dir: Path
    reconstruction_path: Path
    parsed_timestamp: datetime | None
    modified_time: float


DEFAULT_DATASETS: tuple[DatasetConfig, ...] = (
    #DatasetConfig("dragon_30", "dragon.ply"),
    DatasetConfig("horse_30", "horse.ply"),
    #DatasetConfig("lego_30", "lego.ply"),
    #DatasetConfig("plant_30", "plant.ply"),
    #DatasetConfig("teapot_30", "teapot.ply"),
)


def default_optimization_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "OptimizationOutput"


def require_existing_path(path: Path, description: str) -> Path:
    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Could not find {description}: {resolved_path}")

    return resolved_path


def reconstruction_file_name(mode: str) -> str:
    if mode == "poisson":
        return "poisson_post.ply"

    if mode == "tsdf":
        return "fuse_post.ply"

    raise ValueError(f"Unknown reconstruction mode: {mode}")


def parse_run_timestamp(run_dir_name: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", run_dir_name)

    if match is None:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def run_directory_matches_dataset(run_dir_name: str, dataset_name: str) -> bool:
    normalized_run_dir_name = run_dir_name.lower()
    normalized_dataset_name = dataset_name.lower()

    pattern = rf"(^|[^a-z0-9]){re.escape(normalized_dataset_name)}([^a-z0-9]|$)"
    return re.search(pattern, normalized_run_dir_name) is not None


def run_dir_from_reconstruction_path(reconstruction_path: Path, mesh_subdir: str) -> Path:
    run_dir = reconstruction_path.parent

    for _ in Path(mesh_subdir).parts:
        run_dir = run_dir.parent

    return run_dir


def add_candidate_if_valid(
    candidates: list[ReconstructionCandidate],
    seen_reconstruction_paths: set[Path],
    reconstruction_path: Path,
    mesh_subdir: str,
) -> None:
    if not reconstruction_path.exists():
        return

    resolved_reconstruction_path = reconstruction_path.expanduser().resolve()

    if resolved_reconstruction_path in seen_reconstruction_paths:
        return

    seen_reconstruction_paths.add(resolved_reconstruction_path)

    run_dir = run_dir_from_reconstruction_path(
        reconstruction_path=resolved_reconstruction_path,
        mesh_subdir=mesh_subdir,
    )

    candidates.append(
        ReconstructionCandidate(
            run_dir=run_dir,
            reconstruction_path=resolved_reconstruction_path,
            parsed_timestamp=parse_run_timestamp(run_dir.name),
            modified_time=resolved_reconstruction_path.stat().st_mtime,
        )
    )


def find_reconstruction_candidates(
    optimization_output_root: Path,
    dataset: DatasetConfig,
    mesh_subdir: str,
    mode: str,
) -> list[ReconstructionCandidate]:
    output_root = require_existing_path(
        optimization_output_root,
        "optimization output root",
    )

    reconstruction_name = reconstruction_file_name(mode)
    candidates: list[ReconstructionCandidate] = []
    seen_reconstruction_paths: set[Path] = set()

    # New/clean layout:
    #   OptimizationOutput/horse/<run>/mesh/poisson_post.ply
    # or:
    #   OptimizationOutput/horse/mesh/poisson_post.ply
    dataset_root = output_root / dataset.name

    if dataset_root.exists():
        for reconstruction_path in dataset_root.rglob(f"{mesh_subdir}/{reconstruction_name}"):
            add_candidate_if_valid(
                candidates=candidates,
                seen_reconstruction_paths=seen_reconstruction_paths,
                reconstruction_path=reconstruction_path,
                mesh_subdir=mesh_subdir,
            )

    # Current flat layout:
    #   OptimizationOutput/2026-06-13_13-22-44_lr0.0005_it100000_horse_13/mesh/poisson_post.ply
    for child in output_root.iterdir():
        if not child.is_dir():
            continue

        if not run_directory_matches_dataset(child.name, dataset.name):
            continue

        reconstruction_path = child / mesh_subdir / reconstruction_name

        add_candidate_if_valid(
            candidates=candidates,
            seen_reconstruction_paths=seen_reconstruction_paths,
            reconstruction_path=reconstruction_path,
            mesh_subdir=mesh_subdir,
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.parsed_timestamp is not None,
            candidate.parsed_timestamp if candidate.parsed_timestamp is not None else datetime.min,
            candidate.modified_time,
        ),
        reverse=True,
    )

    return candidates


def resolve_reconstruction_path(
    optimization_output_root: Path,
    dataset: DatasetConfig,
    mesh_subdir: str,
    mode: str,
    index: int,
) -> ReconstructionCandidate:
    if index < 0:
        raise ValueError(f"--index must be >= 0, got {index}")

    candidates = find_reconstruction_candidates(
        optimization_output_root=optimization_output_root.expanduser().resolve(),
        dataset=dataset,
        mesh_subdir=mesh_subdir,
        mode=mode,
    )

    if not candidates:
        reconstruction_name = reconstruction_file_name(mode)

        raise FileNotFoundError(
            f"Could not find reconstruction for dataset '{dataset.name}'.\n"
            f"Searched for:\n"
            f"  <output-root>/{dataset.name}/**/{mesh_subdir}/{reconstruction_name}\n"
            f"  <output-root>/*{dataset.name}*/{mesh_subdir}/{reconstruction_name}"
        )

    if index >= len(candidates):
        available = [
            f"[{candidate_index}] {candidate.run_dir} -> {candidate.reconstruction_path}"
            for candidate_index, candidate in enumerate(candidates)
        ]

        raise IndexError(
            f"--index {index} is out of range for dataset '{dataset.name}'. "
            f"Found {len(candidates)} candidates.\n"
            + "\n".join(available)
        )

    return candidates[index]


def resolve_ground_truth_path(ground_truth_root: Path, dataset: DatasetConfig) -> Path:
    ground_truth_path = ground_truth_root.expanduser() / dataset.ground_truth_file
    return require_existing_path(ground_truth_path, f"ground truth for dataset '{dataset.name}'")


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
        "mode",
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
        "run_dir",
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
    ground_truth_path = resolve_ground_truth_path(args.ground_truth_root, dataset)

    reconstruction_candidate = resolve_reconstruction_path(
        optimization_output_root=args.optimization_output_root,
        dataset=dataset,
        mesh_subdir=args.mesh_subdir,
        mode=args.mode,
        index=args.index,
    )

    print(f"Evaluating {dataset.name}")
    print(f"  GT:      {ground_truth_path}")
    print(f"  Rec:     {reconstruction_candidate.reconstruction_path}")
    print(f"  Run dir: {reconstruction_candidate.run_dir}")

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
        "mode": args.mode,
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
        "run_dir": str(reconstruction_candidate.run_dir),
        "ground_truth": str(ground_truth_path),
        "reconstruction": str(reconstruction_candidate.reconstruction_path),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chamfer evaluation over multiple PALE / ours datasets.")

    parser.add_argument(
        "--optimization-output-root",
        type=Path,
        default=default_optimization_output_root(),
        help="Root folder containing PALE optimization output runs.",
    )

    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=Path("~/phd/models"),
        help="Root folder containing ground-truth PLY meshes.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated subset, e.g. dragon,horse,lego,plant,teapot.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="0 = latest matching run per dataset, 1 = second latest per dataset, etc.",
    )

    parser.add_argument(
        "--mesh-subdir",
        type=str,
        default="mesh",
        help="Mesh folder inside each run directory.",
    )

    parser.add_argument(
        "--mode",
        "--reconstruction-name",
        dest="mode",
        type=str,
        default="tsdf",
        choices=["poisson", "tsdf"],
        help="poisson uses poisson_post.ply, tsdf uses fuse_post.ply.",
    )

    parser.add_argument("--samples", type=int, default=50_000, help="Number of sampled surface points per mesh.")
    parser.add_argument("--use-vertices", action="store_true", help="Use raw vertices instead of uniform mesh surface sampling.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible surface sampling.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to reported metrics.")
    parser.add_argument("--label", type=str, default="scene units", help="Metric scale label.")
    parser.add_argument("--digits", type=int, default=5, help="Digits used in the printed markdown table.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing datasets instead of failing.")
    parser.add_argument("--csv-output", type=Path, default=Path("ours_chamfer_results.csv"), help="Output CSV path.")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

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