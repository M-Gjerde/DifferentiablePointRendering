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

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "metrics"
DEFAULT_RECONSTRUCTION_NAMES = ("fuse_post.ply", "poisson_post.ply")

RUN_CONFIG_PARAMETERS = [
    "iterations",
    "densification_interval",
    "densification_grad_abs_min",
    "densification_grad_abs_min_final",
    "densification_grad_abs_min_decay_start_iteration",
    "densification_grad_abs_min_decay_end_iteration",
    "densification_scale_min",
    "densification_split_offset_scale",
    "densification_split_scale_factor",
    "densification_exact_clone_percent_dense",
    "densification_scene_extent",
    "densification_max_new_fraction",
    "normal_consistency_weight",
    "depth_distort_weight",
    "opacity_prior_weight",
    "use_global_lr_decay",
    "global_lr_scale_init",
    "global_lr_scale_final",
    "use_position_lr_decay",
    "position_lr_scale_init",
    "position_lr_scale_final",
    "lr_decay_start_iteration",
    "lr_decay_max_steps",
    "learning_rate_position",
    "learning_rate_rotation",
    "learning_rate_scale",
    "learning_rate_albedo",
    "learning_rate_opacity",
    "learning_rate_beta",
    "densify_bsdf_floor",
    "densify_bsdf_gamma",
    "mesh_extraction_interval",
]

LOSS_COLUMNS = [
    ("loss_total_mean", "total"),
    ("loss_rgb_mean", "rgb"),
]

REGULARIZER_COLUMNS = [
    ("loss_depth_distortion_weighted_mean", "depth weighted"),
    ("loss_normal_consistency_weighted_mean", "normal weighted"),
    ("loss_opacity_prior_weighted_mean", "opacity prior weighted"),
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
    psnr_rows: list[dict[str, Any]]
    summary: dict[str, Any]


GROUND_TRUTH_SAMPLE_CACHE: dict[
    tuple[str, int, bool, int],
    tuple[Any, Any, str],
] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate optimization run folders. Produces loss plots from metrics.csv "
            "and optional Chamfer scores from final or checkpoint meshes."
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
        help="Evaluate every mesh checkpoint. By default, evaluate final meshes in mesh/.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500_000,
        help="Uniform surface query samples used with --no-use-vertices.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Retained for CLI compatibility; point-to-triangle evaluation uses Open3D on CPU.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for Chamfer.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to reported Chamfer metrics.")
    parser.add_argument(
        "--use-vertices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use raw mesh vertices as point-to-triangle queries (default). "
            "Pass --no-use-vertices to use uniform surface queries instead."
        ),
    )
    parser.add_argument(
        "--reconstruction-name",
        type=str,
        default=None,
        help=(
            "Evaluate only this mesh filename inside each mesh folder. "
            "By default, evaluate fuse_post.ply and poisson_post.ply when present."
        ),
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
    if number is None and value not in {None, ""}:
        try:
            fallback_number = float(value)
        except (TypeError, ValueError):
            fallback_number = math.nan
        if math.isinf(fallback_number):
            return "inf" if fallback_number > 0.0 else "-inf"
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


def default_run_root() -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from config import OptimizationConfig

    output_dir = Path(OptimizationConfig().output_dir).expanduser()
    if output_dir.is_absolute():
        return output_dir.resolve()
    return (PROJECT_ROOT / output_dir).resolve()


def load_rgb_image(path: Path) -> np.ndarray:
    image = iio.imread(path.as_posix())
    image = np.asarray(image, dtype=np.float32)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] > 3:
        image = image[..., :3]

    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"RGB image must be HxWx3, got shape {image.shape}: {path}")

    if image.size and float(image.max()) > 1.0:
        image = image / 255.0

    return np.ascontiguousarray(image, dtype=np.float32)


def compute_3dgs_psnr(render_image: np.ndarray, target_image: np.ndarray) -> tuple[float, float]:
    if render_image.shape != target_image.shape:
        raise RuntimeError(
            "Cannot compute PSNR for images with different shapes: "
            f"render={render_image.shape}, target={target_image.shape}"
        )

    difference = render_image.astype(np.float64) - target_image.astype(np.float64)
    mse = float(np.mean(np.square(difference)))
    if mse <= 0.0:
        return math.inf, mse
    return -10.0 * math.log10(mse), mse


def camera_name_from_final_render(render_path: Path) -> str | None:
    match = re.fullmatch(r"render_final_(.+)\.png", render_path.name)
    return match.group(1) if match is not None else None


def resolve_dataset_path(run_dir: Path, run_config: dict[str, Any]) -> Path | None:
    optimization_config = optimization_config_from_run_config(run_config)
    dataset_path_value = run_config.get("dataset_path", optimization_config.get("dataset_path"))
    if not dataset_path_value:
        return None

    dataset_path = Path(str(dataset_path_value)).expanduser()
    if dataset_path.is_absolute():
        return dataset_path

    project_relative = (PROJECT_ROOT / dataset_path).resolve()
    if project_relative.exists():
        return project_relative

    return (run_dir / dataset_path).resolve()


def find_target_image_for_camera(run_dir: Path, dataset_path: Path | None, camera_name: str) -> Path | None:
    saved_target_path = run_dir / f"render_target_{camera_name}.png"
    if saved_target_path.is_file():
        return saved_target_path

    if dataset_path is not None:
        dataset_target_path = dataset_path / "images" / f"{camera_name}.png"
        if dataset_target_path.is_file():
            return dataset_target_path

    return None


def compute_final_psnr_rows(run_dir: Path, run_config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_path = resolve_dataset_path(run_dir, run_config)
    rows: list[dict[str, Any]] = []

    for render_path in sorted(run_dir.glob("render_final_*.png")):
        camera_name = camera_name_from_final_render(render_path)
        if camera_name is None:
            continue

        target_path = find_target_image_for_camera(run_dir, dataset_path, camera_name)
        if target_path is None:
            print(f"Warning: no target image found for final render '{render_path.name}', skipping PSNR.")
            continue

        render_image = load_rgb_image(render_path)
        target_image = load_rgb_image(target_path)
        psnr_value, mse_value = compute_3dgs_psnr(render_image, target_image)

        rows.append(
            {
                "run_name": run_dir.name,
                "camera": camera_name,
                "psnr_3dgs": psnr_value,
                "mse": mse_value,
                "width": render_image.shape[1],
                "height": render_image.shape[0],
                "render": str(render_path.resolve()),
                "target": str(target_path.resolve()),
            }
        )

    return rows


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


def find_latest_run_dir_by_metrics_timestamp(run_root: Path) -> Path | None:
    if not run_root.is_dir():
        return None

    metrics_paths = [path for path in run_root.rglob("metrics.csv") if path.is_file()]
    if not metrics_paths:
        return None

    latest_metrics_path = max(metrics_paths, key=lambda path: (path.stat().st_mtime, str(path)))
    return latest_metrics_path.parent.resolve()


def common_parent(paths: list[Path]) -> Path:
    if not paths:
        return PROJECT_ROOT
    return Path(os.path.commonpath([str(path) for path in paths]))


def normalize_reconstruction_names(reconstruction_names: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(reconstruction_names, str):
        return (reconstruction_names,)
    return reconstruction_names


def find_mesh_checkpoints(
    run_dir: Path,
    reconstruction_name: str | tuple[str, ...],
) -> list[MeshCheckpoint]:
    checkpoint_root = run_dir / "mesh_checkpoints"
    if not checkpoint_root.is_dir():
        return []

    checkpoints: list[MeshCheckpoint] = []
    for candidate_name in normalize_reconstruction_names(reconstruction_name):
        for mesh_path in checkpoint_root.glob(f"iter_*/{candidate_name}"):
            match = re.search(r"iter_(\d+)", mesh_path.parent.name)
            if match is None:
                continue
            checkpoints.append(MeshCheckpoint(iteration=int(match.group(1)), mesh_path=mesh_path.resolve()))

    return sorted(checkpoints, key=lambda checkpoint: (checkpoint.iteration, checkpoint.mesh_path.name))


def final_iteration_from_rows(loss_rows: list[dict[str, str]], run_config: dict[str, Any]) -> int:
    final_iteration = last_numeric_value(loss_rows, "iteration")
    if final_iteration is not None:
        return int(final_iteration)

    optimization_config = optimization_config_from_run_config(run_config)
    configured_iterations = safe_float(optimization_config.get("iterations"))
    if configured_iterations is not None:
        return int(configured_iterations)

    return 0


def find_main_mesh(
    run_dir: Path,
    reconstruction_name: str | tuple[str, ...],
    iteration: int,
) -> list[MeshCheckpoint]:
    checkpoints: list[MeshCheckpoint] = []
    for candidate_name in normalize_reconstruction_names(reconstruction_name):
        mesh_path = run_dir / "mesh" / candidate_name
        if mesh_path.is_file():
            checkpoints.append(MeshCheckpoint(iteration=int(iteration), mesh_path=mesh_path.resolve()))
    return checkpoints


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
            compute_paper_ready_point_to_triangle_distance,
            load_triangle_mesh_with_query_points,
            set_random_seed,
        )
    except ModuleNotFoundError as exception:
        raise RuntimeError(
            "Geometry evaluation requires the dependencies used by metrics/chamfer_ours.py "
            "(notably Open3D). Loss-only evaluation works without them."
        ) from exception

    return (
        compute_paper_ready_point_to_triangle_distance,
        load_triangle_mesh_with_query_points,
        set_random_seed,
    )


def compute_geometry_rows(
    run_dir: Path,
    checkpoints: list[MeshCheckpoint],
    ground_truth_path: Path,
    samples: int,
    device_name: str,
    seed: int,
    scale: float,
    use_vertices: bool,
    print_each_score: bool,
) -> list[dict[str, Any]]:
    if not checkpoints:
        return []

    (
        compute_paper_ready_point_to_triangle_distance,
        load_triangle_mesh_with_query_points,
        set_random_seed,
    ) = lazy_chamfer_imports()

    ground_truth_cache_key = (
        str(ground_truth_path.resolve()),
        int(samples),
        bool(use_vertices),
        int(seed),
    )
    if ground_truth_cache_key in GROUND_TRUTH_SAMPLE_CACHE:
        (
            ground_truth_mesh,
            ground_truth_points,
            ground_truth_sampling,
        ) = GROUND_TRUTH_SAMPLE_CACHE[ground_truth_cache_key]
    else:
        set_random_seed(seed)
        (
            ground_truth_mesh,
            ground_truth_points,
            ground_truth_sampling,
        ) = load_triangle_mesh_with_query_points(
            ply_path=ground_truth_path,
            sample_count=samples,
            use_vertices=use_vertices,
        )
        GROUND_TRUTH_SAMPLE_CACHE[ground_truth_cache_key] = (
            ground_truth_mesh,
            ground_truth_points,
            ground_truth_sampling,
        )

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for checkpoint in checkpoints:
        print(
            f"Evaluating geometry: {run_dir.name} iter {checkpoint.iteration} | "
            f"reconstruction={checkpoint.mesh_path} | gt={ground_truth_path}",
            flush=True,
        )
        set_random_seed(seed)
        (
            reconstruction_mesh,
            reconstruction_points,
            reconstruction_sampling,
        ) = load_triangle_mesh_with_query_points(
            ply_path=checkpoint.mesh_path,
            sample_count=samples,
            use_vertices=use_vertices,
        )
        metrics = compute_paper_ready_point_to_triangle_distance(
            reconstruction_points=reconstruction_points,
            reconstruction_mesh=reconstruction_mesh,
            ground_truth_points=ground_truth_points,
            ground_truth_mesh=ground_truth_mesh,
            scale=scale,
        )
        row = {
            "run_name": run_dir.name,
            "iteration": checkpoint.iteration,
            "reconstruction_name": checkpoint.mesh_path.name,
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
            "distance_mode": "symmetric_point_to_triangle",
            "distance_backend": "open3d_raycasting_cpu",
            "reconstruction": str(checkpoint.mesh_path),
            "ground_truth": str(ground_truth_path),
        }
        rows.append(row)

        if best_row is None or float(row["cd"]) < float(best_row["cd"]):
            best_row = row

        if print_each_score:
            best_text = ""
            if best_row is not None:
                best_text = (
                    f" | best_so_far iter={int(best_row['iteration'])} "
                    f"CD={format_metric(best_row['cd'])}"
                )
            print(
                f"Geometry score: {run_dir.name} iter {checkpoint.iteration} "
                f"CD={format_metric(row['cd'])} "
                f"Accuracy={format_metric(row['accuracy'])} "
                f"Completion={format_metric(row['completion'])}"
                f"{best_text}",
                flush=True,
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


def reconstruction_name_from_row(row: dict[str, Any]) -> str:
    reconstruction_name = str(row.get("reconstruction_name", "")).strip()
    if reconstruction_name:
        return reconstruction_name
    reconstruction_path = str(row.get("reconstruction", "")).strip()
    return Path(reconstruction_path).name if reconstruction_path else "reconstruction"


def plot_geometry_curve(run_dir: Path, rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    fig, axis = plt.subplots(figsize=(9.0, 4.8), dpi=130)
    reconstruction_names = sorted({reconstruction_name_from_row(row) for row in rows})
    for reconstruction_name in reconstruction_names:
        reconstruction_rows = sorted(
            (row for row in rows if reconstruction_name_from_row(row) == reconstruction_name),
            key=lambda row: int(row["iteration"]),
        )
        iterations = [int(row["iteration"]) for row in reconstruction_rows]
        label_prefix = f"{reconstruction_name}: " if len(reconstruction_names) > 1 else ""
        axis.plot(
            iterations,
            [float(row["cd"]) for row in reconstruction_rows],
            marker="o",
            linewidth=1.8,
            label=f"{label_prefix}CD",
        )
        axis.plot(
            iterations,
            [float(row["accuracy"]) for row in reconstruction_rows],
            marker="o",
            linewidth=1.2,
            label=f"{label_prefix}accuracy",
        )
        axis.plot(
            iterations,
            [float(row["completion"]) for row in reconstruction_rows],
            marker="o",
            linewidth=1.2,
            label=f"{label_prefix}completion",
        )
    axis.set_title(f"Geometry - {run_dir.name}")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Chamfer distance")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_loss_geometry_curve(
    run_dir: Path,
    loss_rows: list[dict[str, str]],
    geometry_rows: list[dict[str, Any]],
    output_path: Path,
    complete_only: bool,
    log_loss_y: bool,
) -> None:
    plot_loss_rows = filtered_loss_rows(loss_rows, complete_only)
    if not plot_loss_rows or not geometry_rows:
        return

    rgb_xs, rgb_ys = numeric_series(plot_loss_rows, "iteration", "loss_rgb_mean")
    total_xs, total_ys = numeric_series(plot_loss_rows, "iteration", "loss_total_mean")
    if not rgb_xs and not total_xs:
        return

    fig, loss_axis = plt.subplots(figsize=(9.5, 5.4), dpi=130)
    geometry_axis = loss_axis.twinx()

    loss_lines = []
    if rgb_xs:
        loss_lines.extend(
            loss_axis.plot(
                rgb_xs,
                rgb_ys,
                color="tab:blue",
                linewidth=1.6,
                label="RGB image loss",
            )
        )
    if total_xs:
        loss_lines.extend(
            loss_axis.plot(
                total_xs,
                total_ys,
                color="tab:cyan",
                linewidth=1.2,
                alpha=0.65,
                label="total loss",
            )
        )

    geometry_lines = []
    reconstruction_names = sorted({reconstruction_name_from_row(row) for row in geometry_rows})
    for reconstruction_name in reconstruction_names:
        reconstruction_rows = sorted(
            (row for row in geometry_rows if reconstruction_name_from_row(row) == reconstruction_name),
            key=lambda row: int(row["iteration"]),
        )
        geometry_lines.extend(
            geometry_axis.plot(
                [int(row["iteration"]) for row in reconstruction_rows],
                [float(row["cd"]) for row in reconstruction_rows],
                marker="o",
                linewidth=1.5,
                label=f"CD ({reconstruction_name})" if len(reconstruction_names) > 1 else "CD",
            )
        )

    loss_axis.set_title(f"Image Loss vs Geometry - {run_dir.name}")
    loss_axis.set_xlabel("Iteration")
    loss_axis.set_ylabel("Image loss")
    geometry_axis.set_ylabel("Chamfer distance")
    loss_axis.grid(True, alpha=0.25)
    if log_loss_y:
        apply_log_scale_if_possible(loss_axis, plot_loss_rows, ["loss_rgb_mean", "loss_total_mean"], enabled=True)

    lines = loss_lines + geometry_lines
    labels = [line.get_label() for line in lines]
    loss_axis.legend(lines, labels, loc="best")

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
    psnr_rows: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    optimization_config = optimization_config_from_run_config(run_config)
    psnr_values = [
        float(row["psnr_3dgs"])
        for row in psnr_rows
        if row.get("psnr_3dgs") not in {None, ""}
    ]

    summary: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "final_iteration": last_numeric_value(loss_rows, "iteration"),
        "final_loss_total": last_numeric_value(loss_rows, "loss_total_mean"),
        "final_loss_rgb": last_numeric_value(loss_rows, "loss_rgb_mean"),
        "final_num_points": last_numeric_value(loss_rows, "num_points"),
        "mesh_checkpoint_count": len(geometry_rows),
        "final_psnr_3dgs": sum(psnr_values) / len(psnr_values) if psnr_values else "",
        "psnr_camera_count": len(psnr_values),
    }

    if geometry_rows:
        sorted_rows = sorted(geometry_rows, key=lambda row: int(row["iteration"]))
        best_row = min(sorted_rows, key=lambda row: float(row["cd"]))
        final_iteration = int(sorted_rows[-1]["iteration"])
        final_row = min(
            (row for row in sorted_rows if int(row["iteration"]) == final_iteration),
            key=lambda row: float(row["cd"]),
        )
        summary.update(
            {
                "best_cd": best_row["cd"],
                "best_cd_iteration": best_row["iteration"],
                "best_reconstruction": reconstruction_name_from_row(best_row),
                "final_cd": final_row["cd"],
                "final_accuracy": final_row["accuracy"],
                "final_completion": final_row["completion"],
                "final_reconstruction": reconstruction_name_from_row(final_row),
            }
        )

    for parameter_name in RUN_CONFIG_PARAMETERS:
        summary[parameter_name] = optimization_config.get(parameter_name, "")

    return summary


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> RunEvaluation:
    run_dir = resolve_path(run_dir)
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    run_config = load_run_config(run_dir)

    loss_rows = read_csv_dicts(run_dir / "metrics.csv")
    plot_loss_curve(
        run_dir=run_dir,
        rows=loss_rows,
        output_path=evaluation_dir / "loss_curve.png",
        complete_only=args.complete_loss_only,
        log_y=not args.linear_loss_y,
    )

    geometry_rows: list[dict[str, Any]] = []
    geometry_mode = "full" if args.full else "final"
    geometry_csv_path = evaluation_dir / f"mesh_checkpoint_metrics_{geometry_mode}.csv"
    geometry_plot_path = evaluation_dir / f"geometry_curve_{geometry_mode}.png"
    loss_geometry_plot_path = evaluation_dir / f"loss_geometry_curve_{geometry_mode}.png"

    if args.ground_truth is not None:
        if geometry_csv_path.exists() and not args.force:
            geometry_rows = read_existing_geometry_rows(geometry_csv_path)
        else:
            reconstruction_names = (
                (args.reconstruction_name,)
                if args.reconstruction_name is not None
                else DEFAULT_RECONSTRUCTION_NAMES
            )
            if args.full:
                checkpoints = select_mesh_checkpoints(
                    find_mesh_checkpoints(run_dir, reconstruction_names),
                    full=True,
                )
            else:
                checkpoints = find_main_mesh(
                    run_dir=run_dir,
                    reconstruction_name=reconstruction_names,
                    iteration=final_iteration_from_rows(loss_rows, run_config),
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
                print_each_score=args.full or len(checkpoints) > 1,
            )
            write_dict_csv(geometry_csv_path, geometry_rows)

        plot_geometry_curve(run_dir, geometry_rows, geometry_plot_path)
        plot_loss_geometry_curve(
            run_dir=run_dir,
            loss_rows=loss_rows,
            geometry_rows=geometry_rows,
            output_path=loss_geometry_plot_path,
            complete_only=args.complete_loss_only,
            log_loss_y=not args.linear_loss_y,
        )

    psnr_rows = compute_final_psnr_rows(run_dir, run_config)
    write_dict_csv(
        evaluation_dir / "final_image_psnr.csv",
        psnr_rows,
        fieldnames=["run_name", "camera", "psnr_3dgs", "mse", "width", "height", "render", "target"],
    )

    summary = make_summary(run_dir, loss_rows, geometry_rows, psnr_rows, run_config)
    with (evaluation_dir / "run_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    return RunEvaluation(
        run_dir=run_dir,
        evaluation_dir=evaluation_dir,
        loss_rows=loss_rows,
        geometry_rows=geometry_rows,
        psnr_rows=psnr_rows,
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
        reconstruction_names = sorted(
            {reconstruction_name_from_row(row) for row in evaluation.geometry_rows}
        )
        for reconstruction_name in reconstruction_names:
            rows = sorted(
                (
                    row
                    for row in evaluation.geometry_rows
                    if reconstruction_name_from_row(row) == reconstruction_name
                ),
                key=lambda row: int(row["iteration"]),
            )
            iterations = [int(row["iteration"]) for row in rows]
            cds = [float(row["cd"]) for row in rows]
            label = f"{evaluation.run_dir.name}/{reconstruction_name} best={min(cds):.4g}"
            axis.plot(iterations, cds, marker="o", linewidth=1.4, label=label)

    axis.set_title("Chamfer comparison")
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


def representative_geometry_rows(evaluation: RunEvaluation) -> list[dict[str, Any]]:
    if not evaluation.geometry_rows:
        return []
    rows = sorted(evaluation.geometry_rows, key=lambda row: int(row["iteration"]))
    final_iteration = int(rows[-1]["iteration"])
    return [row for row in rows if int(row["iteration"]) == final_iteration]


def print_geometry_table(evaluations: list[RunEvaluation], full: bool) -> None:
    geometry_evaluations = [evaluation for evaluation in evaluations if evaluation.geometry_rows]
    if not geometry_evaluations:
        return

    print()
    print("| Run | Reconstruction | Iteration | CD ↓ | Accuracy ↓ | Completion ↓ |")
    print("|---|---|---:|---:|---:|---:|")

    table_rows: list[tuple[str, dict[str, Any]]] = []

    if full:
        for evaluation in sorted(geometry_evaluations, key=summary_sort_key):
            for row in sorted(evaluation.geometry_rows, key=lambda item: int(item["iteration"])):
                table_rows.append((evaluation.run_dir.name, row))
    else:
        for evaluation in sorted(geometry_evaluations, key=summary_sort_key):
            for row in representative_geometry_rows(evaluation):
                table_rows.append((evaluation.run_dir.name, row))

    for run_name, row in table_rows:
        print(
            f"| {run_name} "
            f"| {reconstruction_name_from_row(row)} "
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
            f"reconstruction={best_evaluation.summary.get('best_reconstruction', '')} "
            f"CD={format_metric(best_cd)}"
        )


def print_psnr_table(evaluations: list[RunEvaluation]) -> None:
    psnr_evaluations = [evaluation for evaluation in evaluations if evaluation.psnr_rows]
    if not psnr_evaluations:
        return

    def psnr_sort_value(evaluation: RunEvaluation) -> float:
        try:
            value = float(evaluation.summary.get("final_psnr_3dgs", -math.inf))
        except (TypeError, ValueError):
            return -math.inf
        return value if not math.isnan(value) else -math.inf

    print()
    print("| Run | Final PSNR ↑ | Cameras |")
    print("|---|---:|---:|")

    table_rows = sorted(
        psnr_evaluations,
        key=lambda evaluation: (-psnr_sort_value(evaluation), evaluation.run_dir.name),
    )
    for evaluation in table_rows:
        print(
            f"| {evaluation.run_dir.name} "
            f"| {format_metric(evaluation.summary.get('final_psnr_3dgs'))} "
            f"| {int(evaluation.summary.get('psnr_camera_count', 0))} |"
        )

    psnr_values = [
        float(evaluation.summary["final_psnr_3dgs"])
        for evaluation in table_rows
        if evaluation.summary.get("final_psnr_3dgs") not in {None, ""}
    ]
    if psnr_values:
        print(f"| **Mean** | **{format_metric(sum(psnr_values) / len(psnr_values))}** |  |")


def main() -> None:
    args = parse_args()
    run_dirs = find_run_dirs(args.run_dir, args.run_root, args.recursive)
    if not run_dirs and not args.run_dir and not args.run_root:
        run_root = default_run_root()
        latest_run_dir = find_latest_run_dir_by_metrics_timestamp(run_root)
        if latest_run_dir is None:
            raise SystemExit(f"No run directories with metrics.csv found under default output dir: {run_root}")
        run_dirs = [latest_run_dir]
        print(f"Using latest run by metrics.csv timestamp: {latest_run_dir}")

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
        "best_reconstruction",
        "final_cd",
        "final_accuracy",
        "final_completion",
        "final_reconstruction",
        "final_psnr_3dgs",
        "psnr_camera_count",
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
    print_psnr_table(evaluations)

    print(f"Saved evaluation summary: {aggregate_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
