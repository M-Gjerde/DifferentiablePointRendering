#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "metrics"

RUN_CONFIG_PARAMETERS = [
    "iterations",
    "densification_interval",
    "densification_grad_abs_min",
    "normal_consistency_weight",
    "depth_distort_weight",
    "use_global_lr_schedule",
    "global_lr_scale_final",
    "global_lr_start_iteration",
    "global_lr_max_steps",
    "learning_rate_position",
    "learning_rate_rotation",
    "learning_rate_scale",
    "learning_rate_albedo",
    "learning_rate_opacity",
    "learning_rate_beta",
    "densify_bsdf_floor",
    "densify_bsdf_gamma",
    "mesh_extraction_iterations",
]

LOSS_COLUMNS = [
    ("loss_total_mean", "total"),
    ("loss_rgb_mean", "rgb"),
]

REGULARIZER_COLUMNS = [
    ("loss_depth_distortion_weighted_mean", "depth weighted"),
    ("loss_normal_consistency_weighted_mean", "normal weighted"),
    ("loss_visibility_weighted_opacity_weighted_mean", "visibility opacity weighted"),
]


@dataclass(frozen=True)
class MeshCheckpoint:
    iteration: int
    mesh_path: Path


@dataclass
class RunEvaluation:
    run_dir: Path
    evaluation_dir: Path
    loss_rows: list[dict[str, str]]
    geometry_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate optimization run folders. Produces loss plots from metrics.csv "
            "and optional Chamfer curves from mesh_checkpoints/iter_*/fuse_post.ply."
        )
    )
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Run directory to evaluate.")
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=[],
        help="Folder whose direct children are run directories.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When used with --run-root, recursively find folders containing metrics.csv.",
    )
    parser.add_argument(
        "--ground-truth", "--gt",
        type=Path,
        default=None,
        help="Optional GT PLY. Required for geometric Chamfer curves.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate every mesh checkpoint. By default, only the latest checkpoint is evaluated.",
    )
    parser.add_argument("--samples", type=int, default=50_000, help="Surface samples for Chamfer.")
    parser.add_argument("--device", type=str, default="auto", help="Chamfer device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for Chamfer.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to reported Chamfer metrics.")
    parser.add_argument("--use-vertices", action="store_true", help="Use mesh vertices instead of surface samples.")
    parser.add_argument(
        "--reconstruction-name",
        type=str,
        default="fuse_post.ply",
        help="Mesh filename inside each mesh checkpoint folder.",
    )
    parser.add_argument(
        "--complete-loss-only",
        action="store_true",
        help="Plot only complete-camera averaged rows from metrics.csv when available.",
    )
    parser.add_argument(
        "--linear-loss-y",
        action="store_true",
        help="Use linear y-axis for loss plots instead of log y-axis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Aggregate evaluation output directory. Defaults to <run-dir>/evaluation for one run, "
            "or <common-run-parent>/evaluation for multiple runs."
        ),
    )
    parser.add_argument("--max-summary-runs", type=int, default=20)
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute geometry CSVs instead of reusing cached results.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "nan"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    if not math.isfinite(number):
        return None
    return number


def csv_value_is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def format_number(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "" if value is None else str(value)
    return f"{number:.12g}"


def format_metric(value: Any, digits: int = 5) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def read_csv_dicts(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_dict_csv(csv_path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: set[str] = set()
        for row in rows:
            fields.update(row.keys())
        fieldnames = sorted(fields)

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_number(row.get(field, "")) for field in fieldnames})


def load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def optimization_config_from_run_config(run_config: dict[str, Any]) -> dict[str, Any]:
    optimization_config = run_config.get("optimization_config", {})
    return optimization_config if isinstance(optimization_config, dict) else {}


def find_run_dirs(run_dirs: list[Path], run_roots: list[Path], recursive: bool) -> list[Path]:
    discovered: list[Path] = []

    for run_dir in run_dirs:
        discovered.append(resolve_path(run_dir))

    for run_root in run_roots:
        root = resolve_path(run_root)
        if (root / "metrics.csv").is_file() or (root / "run_config.json").is_file():
            discovered.append(root)
            continue

        if recursive:
            children = [path.parent for path in root.rglob("metrics.csv")]
        else:
            children = [path for path in root.iterdir() if path.is_dir()]

        for child in children:
            if (child / "metrics.csv").is_file() or (child / "run_config.json").is_file():
                discovered.append(child.resolve())

    unique: dict[Path, Path] = {}
    for run_dir in discovered:
        unique[run_dir] = run_dir

    return sorted(unique.values(), key=lambda path: str(path))


def common_parent(paths: list[Path]) -> Path:
    if not paths:
        return PROJECT_ROOT
    return Path(os.path.commonpath([str(path) for path in paths]))


def find_mesh_checkpoints(run_dir: Path, reconstruction_name: str) -> list[MeshCheckpoint]:
    checkpoint_root = run_dir / "mesh_checkpoints"
    if not checkpoint_root.is_dir():
        return []

    checkpoints: list[MeshCheckpoint] = []
    for mesh_path in checkpoint_root.glob(f"iter_*/{reconstruction_name}"):
        match = re.search(r"iter_(\d+)", mesh_path.parent.name)
        if match is None:
            continue
        checkpoints.append(MeshCheckpoint(iteration=int(match.group(1)), mesh_path=mesh_path.resolve()))

    return sorted(checkpoints, key=lambda checkpoint: checkpoint.iteration)


def select_mesh_checkpoints(
    checkpoints: list[MeshCheckpoint],
    full: bool,
) -> list[MeshCheckpoint]:
    if full:
        return checkpoints
    if not checkpoints:
        return []
    return [checkpoints[-1]]


def filtered_loss_rows(rows: list[dict[str, str]], complete_only: bool) -> list[dict[str, str]]:
    if not complete_only:
        return rows
    if not rows or "loss_average_is_complete" not in rows[0]:
        return rows
    return [row for row in rows if csv_value_is_true(row.get("loss_average_is_complete"))]


def numeric_series(
    rows: list[dict[str, str]],
    x_column: str,
    y_column: str,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []

    for row in rows:
        x_value = safe_float(row.get(x_column))
        y_value = safe_float(row.get(y_column))
        if x_value is None or y_value is None:
            continue
        xs.append(x_value)
        ys.append(y_value)

    return xs, ys


def apply_log_scale_if_possible(axis, rows: list[dict[str, str]], columns: list[str], enabled: bool) -> None:
    if not enabled:
        return
    values: list[float] = []
    for row in rows:
        for column in columns:
            value = safe_float(row.get(column))
            if value is not None and value > 0.0:
                values.append(value)
    if values:
        axis.set_yscale("log")


def plot_loss_curve(
    run_dir: Path,
    rows: list[dict[str, str]],
    output_path: Path,
    complete_only: bool,
    log_y: bool,
) -> None:
    plot_rows = filtered_loss_rows(rows, complete_only)
    if not plot_rows:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), dpi=130, sharex=True)
    fig.suptitle(run_dir.name)

    for column, label in LOSS_COLUMNS:
        xs, ys = numeric_series(plot_rows, "iteration", column)
        if xs:
            axes[0].plot(xs, ys, linewidth=1.6, label=label)

    axes[0].set_ylabel("Optimization loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    apply_log_scale_if_possible(axes[0], plot_rows, [column for column, _ in LOSS_COLUMNS], log_y)

    for column, label in REGULARIZER_COLUMNS:
        xs, ys = numeric_series(plot_rows, "iteration", column)
        if xs and any(value != 0.0 for value in ys):
            axes[1].plot(xs, ys, linewidth=1.3, label=label)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Weighted regularizers")
    axes[1].grid(True, alpha=0.25)
    if axes[1].lines:
        axes[1].legend(loc="best")
    apply_log_scale_if_possible(axes[1], plot_rows, [column for column, _ in REGULARIZER_COLUMNS], log_y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def lazy_chamfer_imports():
    if str(METRICS_DIR) not in sys.path:
        sys.path.insert(0, str(METRICS_DIR))
    try:
        from chamfer_ours import (
            compute_paper_ready_chamfer,
            load_points_from_ply,
            resolve_device,
            set_random_seed,
        )
    except ModuleNotFoundError as exception:
        raise RuntimeError(
            "Geometry evaluation requires the dependencies used by metrics/chamfer_ours.py "
            "(notably open3d and pytorch3d). Loss-only evaluation works without them."
        ) from exception

    return compute_paper_ready_chamfer, load_points_from_ply, resolve_device, set_random_seed


def compute_geometry_rows(
    run_dir: Path,
    checkpoints: list[MeshCheckpoint],
    ground_truth_path: Path,
    samples: int,
    device_name: str,
    seed: int,
    scale: float,
    use_vertices: bool,
) -> list[dict[str, Any]]:
    if not checkpoints:
        return []

    (
        compute_paper_ready_chamfer,
        load_points_from_ply,
        resolve_device,
        set_random_seed,
    ) = lazy_chamfer_imports()

    device = resolve_device(device_name)
    set_random_seed(seed)
    ground_truth_points, ground_truth_sampling = load_points_from_ply(
        ply_path=ground_truth_path,
        sample_count=samples,
        use_vertices=use_vertices,
    )

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        print(f"Evaluating geometry: {run_dir.name} iter {checkpoint.iteration}")
        set_random_seed(seed)
        reconstruction_points, reconstruction_sampling = load_points_from_ply(
            ply_path=checkpoint.mesh_path,
            sample_count=samples,
            use_vertices=use_vertices,
        )
        metrics = compute_paper_ready_chamfer(
            reconstruction_points=reconstruction_points,
            ground_truth_points=ground_truth_points,
            device=device,
            scale=scale,
        )
        rows.append(
            {
                "run_name": run_dir.name,
                "iteration": checkpoint.iteration,
                "cd": metrics["cd"],
                "accuracy": metrics["accuracy"],
                "completion": metrics["completion"],
                "median_reconstruction_to_gt": metrics["median_reconstruction_to_gt"],
                "median_gt_to_reconstruction": metrics["median_gt_to_reconstruction"],
                "p95_reconstruction_to_gt": metrics["p95_reconstruction_to_gt"],
                "p95_gt_to_reconstruction": metrics["p95_gt_to_reconstruction"],
                "reconstruction_points": len(reconstruction_points),
                "ground_truth_points": len(ground_truth_points),
                "reconstruction_sampling": reconstruction_sampling,
                "ground_truth_sampling": ground_truth_sampling,
                "reconstruction": str(checkpoint.mesh_path),
                "ground_truth": str(ground_truth_path),
            }
        )

    return rows


def read_existing_geometry_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows = read_csv_dicts(csv_path)
    converted_rows: list[dict[str, Any]] = []
    for row in rows:
        converted = dict(row)
        for key in [
            "iteration",
            "cd",
            "accuracy",
            "completion",
            "median_reconstruction_to_gt",
            "median_gt_to_reconstruction",
            "p95_reconstruction_to_gt",
            "p95_gt_to_reconstruction",
            "reconstruction_points",
            "ground_truth_points",
        ]:
            value = safe_float(row.get(key))
            if value is not None:
                converted[key] = int(value) if key.endswith("points") or key == "iteration" else value
        converted_rows.append(converted)
    return converted_rows


def plot_geometry_curve(run_dir: Path, rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    sorted_rows = sorted(rows, key=lambda row: int(row["iteration"]))
    iterations = [int(row["iteration"]) for row in sorted_rows]

    fig, axis = plt.subplots(figsize=(9.0, 4.8), dpi=130)
    axis.plot(iterations, [float(row["cd"]) for row in sorted_rows], marker="o", linewidth=1.8, label="CD")
    axis.plot(
        iterations,
        [float(row["accuracy"]) for row in sorted_rows],
        marker="o",
        linewidth=1.2,
        label="accuracy",
    )
    axis.plot(
        iterations,
        [float(row["completion"]) for row in sorted_rows],
        marker="o",
        linewidth=1.2,
        label="completion",
    )
    axis.set_title(f"Mesh checkpoint geometry - {run_dir.name}")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Chamfer distance")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def last_numeric_value(rows: list[dict[str, str]], column: str) -> float | None:
    for row in reversed(rows):
        value = safe_float(row.get(column))
        if value is not None:
            return value
    return None


def make_summary(
    run_dir: Path,
    loss_rows: list[dict[str, str]],
    geometry_rows: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    optimization_config = optimization_config_from_run_config(run_config)

    summary: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "final_iteration": last_numeric_value(loss_rows, "iteration"),
        "final_loss_total": last_numeric_value(loss_rows, "loss_total_mean"),
        "final_loss_rgb": last_numeric_value(loss_rows, "loss_rgb_mean"),
        "final_num_points": last_numeric_value(loss_rows, "num_points"),
        "mesh_checkpoint_count": len(geometry_rows),
    }

    if geometry_rows:
        sorted_rows = sorted(geometry_rows, key=lambda row: int(row["iteration"]))
        best_row = min(sorted_rows, key=lambda row: float(row["cd"]))
        final_row = sorted_rows[-1]
        summary.update(
            {
                "best_cd": best_row["cd"],
                "best_cd_iteration": best_row["iteration"],
                "final_cd": final_row["cd"],
                "final_accuracy": final_row["accuracy"],
                "final_completion": final_row["completion"],
            }
        )

    for parameter_name in RUN_CONFIG_PARAMETERS:
        summary[parameter_name] = optimization_config.get(parameter_name, "")

    return summary


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> RunEvaluation:
    run_dir = resolve_path(run_dir)
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    loss_rows = read_csv_dicts(run_dir / "metrics.csv")
    plot_loss_curve(
        run_dir=run_dir,
        rows=loss_rows,
        output_path=evaluation_dir / "loss_curve.png",
        complete_only=args.complete_loss_only,
        log_y=not args.linear_loss_y,
    )

    geometry_rows: list[dict[str, Any]] = []
    geometry_mode = "full" if args.full else "latest"
    geometry_csv_path = evaluation_dir / f"mesh_checkpoint_metrics_{geometry_mode}.csv"
    geometry_plot_path = evaluation_dir / f"geometry_curve_{geometry_mode}.png"

    if args.ground_truth is not None:
        if geometry_csv_path.exists() and not args.force:
            geometry_rows = read_existing_geometry_rows(geometry_csv_path)
        else:
            checkpoints = select_mesh_checkpoints(
                find_mesh_checkpoints(run_dir, args.reconstruction_name),
                full=args.full,
            )
            geometry_rows = compute_geometry_rows(
                run_dir=run_dir,
                checkpoints=checkpoints,
                ground_truth_path=resolve_path(args.ground_truth),
                samples=args.samples,
                device_name=args.device,
                seed=args.seed,
                scale=args.scale,
                use_vertices=args.use_vertices,
            )
            write_dict_csv(geometry_csv_path, geometry_rows)

        plot_geometry_curve(run_dir, geometry_rows, geometry_plot_path)

    run_config = load_run_config(run_dir)
    summary = make_summary(run_dir, loss_rows, geometry_rows, run_config)
    with (evaluation_dir / "run_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    return RunEvaluation(
        run_dir=run_dir,
        evaluation_dir=evaluation_dir,
        loss_rows=loss_rows,
        geometry_rows=geometry_rows,
        summary=summary,
    )


def summary_sort_key(evaluation: RunEvaluation) -> tuple[float, float, str]:
    best_cd = safe_float(evaluation.summary.get("best_cd"))
    final_loss = safe_float(evaluation.summary.get("final_loss_total"))
    return (
        best_cd if best_cd is not None else math.inf,
        final_loss if final_loss is not None else math.inf,
        evaluation.run_dir.name,
    )


def plot_geometry_comparison(evaluations: list[RunEvaluation], output_path: Path, max_runs: int) -> None:
    selected = [evaluation for evaluation in sorted(evaluations, key=summary_sort_key) if evaluation.geometry_rows]
    selected = selected[:max_runs]
    if not selected:
        return

    fig, axis = plt.subplots(figsize=(10.5, 6.0), dpi=130)
    for evaluation in selected:
        rows = sorted(evaluation.geometry_rows, key=lambda row: int(row["iteration"]))
        iterations = [int(row["iteration"]) for row in rows]
        cds = [float(row["cd"]) for row in rows]
        label = f"{evaluation.run_dir.name} best={min(cds):.4g}"
        axis.plot(iterations, cds, marker="o", linewidth=1.4, label=label)

    axis.set_title("Mesh checkpoint Chamfer comparison")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("CD")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_loss_comparison(evaluations: list[RunEvaluation], output_path: Path, max_runs: int) -> None:
    selected = sorted(evaluations, key=summary_sort_key)[:max_runs]
    selected = [evaluation for evaluation in selected if evaluation.loss_rows]
    if not selected:
        return

    fig, axis = plt.subplots(figsize=(10.5, 6.0), dpi=130)
    for evaluation in selected:
        rows = filtered_loss_rows(evaluation.loss_rows, complete_only=False)
        xs, ys = numeric_series(rows, "iteration", "loss_total_mean")
        if not xs:
            continue
        axis.plot(xs, ys, linewidth=1.3, label=evaluation.run_dir.name)

    if not axis.lines:
        plt.close(fig)
        return

    axis.set_title("Optimization loss comparison")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("loss_total_mean")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def representative_geometry_row(evaluation: RunEvaluation) -> dict[str, Any] | None:
    if not evaluation.geometry_rows:
        return None
    rows = sorted(evaluation.geometry_rows, key=lambda row: int(row["iteration"]))
    return rows[-1]


def print_geometry_table(evaluations: list[RunEvaluation], full: bool) -> None:
    geometry_evaluations = [evaluation for evaluation in evaluations if evaluation.geometry_rows]
    if not geometry_evaluations:
        return

    print()
    print("| Run | Iteration | CD ↓ | Accuracy ↓ | Completion ↓ |")
    print("|---|---:|---:|---:|---:|")

    table_rows: list[tuple[str, dict[str, Any]]] = []

    if full:
        for evaluation in sorted(geometry_evaluations, key=summary_sort_key):
            for row in sorted(evaluation.geometry_rows, key=lambda item: int(item["iteration"])):
                table_rows.append((evaluation.run_dir.name, row))
    else:
        for evaluation in sorted(geometry_evaluations, key=summary_sort_key):
            row = representative_geometry_row(evaluation)
            if row is not None:
                table_rows.append((evaluation.run_dir.name, row))

    for run_name, row in table_rows:
        print(
            f"| {run_name} "
            f"| {int(row['iteration'])} "
            f"| {format_metric(row['cd'])} "
            f"| {format_metric(row['accuracy'])} "
            f"| {format_metric(row['completion'])} |"
        )

    if table_rows and not full:
        mean_cd = sum(float(row["cd"]) for _, row in table_rows) / len(table_rows)
        mean_accuracy = sum(float(row["accuracy"]) for _, row in table_rows) / len(table_rows)
        mean_completion = sum(float(row["completion"]) for _, row in table_rows) / len(table_rows)
        print(
            f"| **Mean** "
            f"|  "
            f"| **{format_metric(mean_cd)}** "
            f"| **{format_metric(mean_accuracy)}** "
            f"| **{format_metric(mean_completion)}** |"
        )

    best_evaluation = min(geometry_evaluations, key=summary_sort_key)
    best_cd = safe_float(best_evaluation.summary.get("best_cd"))
    best_iteration = best_evaluation.summary.get("best_cd_iteration", "")
    if best_cd is not None:
        print()
        print(
            "Best geometry: "
            f"{best_evaluation.run_dir.name} "
            f"iter={best_iteration} "
            f"CD={format_metric(best_cd)}"
        )


def main() -> None:
    args = parse_args()
    run_dirs = find_run_dirs(args.run_dir, args.run_root, args.recursive)
    if not run_dirs:
        raise SystemExit("No run directories found. Pass --run-dir or --run-root.")

    evaluations: list[RunEvaluation] = []
    for run_dir in run_dirs:
        print(f"Evaluating run: {run_dir}")
        evaluations.append(evaluate_run(run_dir, args))

    if args.output_dir is not None:
        aggregate_dir = resolve_path(args.output_dir)
    elif len(evaluations) == 1:
        aggregate_dir = evaluations[0].evaluation_dir
    else:
        aggregate_dir = common_parent(run_dirs) / "evaluation"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [evaluation.summary for evaluation in sorted(evaluations, key=summary_sort_key)]
    summary_fields = [
        "run_name",
        "best_cd",
        "best_cd_iteration",
        "final_cd",
        "final_accuracy",
        "final_completion",
        "final_loss_total",
        "final_loss_rgb",
        "final_num_points",
        "mesh_checkpoint_count",
        *RUN_CONFIG_PARAMETERS,
        "run_dir",
    ]
    write_dict_csv(aggregate_dir / "summary.csv", summary_rows, fieldnames=summary_fields)
    plot_geometry_comparison(evaluations, aggregate_dir / "geometry_comparison.png", args.max_summary_runs)
    plot_loss_comparison(evaluations, aggregate_dir / "loss_comparison.png", args.max_summary_runs)
    print_geometry_table(evaluations, full=args.full)

    print(f"Saved evaluation summary: {aggregate_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
