from __future__ import annotations

import argparse
import io
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import StrMethodFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGEND_KWARGS = {
    "fontsize": "small",
    "framealpha": 0.85,
}


@dataclass
class GeometryEvaluationState:
    run_dir: Path | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    evaluated_iterations: set[int] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live plot metrics.csv while an optimization run is writing to it. "
            "By default, the run with the newest metrics.csv under the configured output dir is used."
        )
    )
    parser.add_argument(
        "--optimization-output-root",
        type=Path,
        required=False,
        default=None,
        help="Path to the OptimizationOutput directory. Defaults to OptimizationConfig.output_dir.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory. If omitted, the latest run is used.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Use the N-th latest run when --run-dir is omitted. 0 means latest.",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default="metrics.csv",
        help="Metrics CSV filename inside the run directory.",
    )
    parser.add_argument(
        "--loss-column",
        type=str,
        default=None,
        help=(
            "Optional explicit loss column for the single-loss plot. "
            "If omitted, defaults to loss_total_mean, then loss_rgb_mean."
        ),
    )
    parser.add_argument(
        "--ground-truth",
        "--gt",
        type=Path,
        default=None,
        help="Optional GT PLY. When provided, live symmetric CD is evaluated from mesh checkpoints.",
    )
    parser.add_argument(
        "--geometry-every",
        type=int,
        default=500,
        help="Evaluate available mesh checkpoints whose iteration is a multiple of this value.",
    )
    parser.add_argument(
        "--geometry-samples",
        type=int,
        default=500_000,
        help="Uniform surface query samples used with --no-geometry-use-vertices.",
    )
    parser.add_argument(
        "--geometry-device",
        type=str,
        default="auto",
        help="Retained for compatibility; point-to-triangle evaluation uses Open3D on CPU.",
    )
    parser.add_argument(
        "--geometry-seed",
        type=int,
        default=0,
        help="Sampling seed for live Chamfer evaluation.",
    )
    parser.add_argument(
        "--geometry-scale",
        type=float,
        default=1.0,
        help="Scale applied to live Chamfer metrics.",
    )
    parser.add_argument(
        "--geometry-use-vertices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use raw mesh vertices as point-to-triangle queries (default). "
            "Pass --no-geometry-use-vertices to use uniform surface queries."
        ),
    )
    parser.add_argument(
        "--reconstruction-name",
        type=str,
        default="fuse_post.ply",
        help="Mesh filename inside each mesh checkpoint folder.",
    )
    parser.add_argument(
        "--plot-all-losses",
        dest="plot_all_losses",
        action="store_true",
        help="Show all supported panels.",
    )
    parser.add_argument(
        "--no-plot-all-losses",
        dest="plot_all_losses",
        action="store_false",
        help="Only show one selected loss column.",
    )
    parser.add_argument(
        "--iterations",
        "--it",
        type=int,
        default=None,
        help=(
            "Only plot the last N loss iterations. "
            "Point-count history remains full by default."
        ),
    )
    parser.add_argument(
        "--point-count-windowed",
        action="store_true",
        help=(
            "Also apply --iterations to the point-count panel. "
            "By default, point count always shows the full run history."
        ),
    )
    parser.add_argument(
        "--loss-y-scale",
        choices=("linear", "log", "symlog"),
        default="log",
        help=(
            "Y-axis scale for RGB, total-loss, and weighted-regularizer panels. "
            "Default: log."
        ),
    )
    parser.add_argument(
        "--skip",
        help=(
            "Skip the first 100 iterations after each opacity reset at every "
            "N iterations. For example, --skip 1000 removes 1000-1099, "
            "2000-2099, etc. from loss plots."
        ),
        type=int,
        default=0,
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=0.1,
        help="Polling interval for checking whether metrics.csv changed.",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Also write the current live plot to a PNG on every update.",
    )
    parser.add_argument(
        "--loss-output-name",
        type=str,
        default="loss_curve_live.png",
        help="Filename for the saved live plot image inside the run folder.",
    )
    parser.add_argument(
        "--watch-latest",
        action="store_true",
        default=True,
        help=(
            "Keep following the newest run folder. Useful if you start this script "
            "before launching a new optimization run."
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_iteration",
        type=int,
        default=None,
        help=(
            "Only plot iterations >= N. "
            "Example: --from 5000 plots from iteration 5000 onward."
        ),
    )

    parser.set_defaults(plot_all_losses=True)

    return parser.parse_args()


def parse_run_timestamp(run_dir_name: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", run_dir_name)
    if match is None:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def default_optimization_output_root() -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from config import OptimizationConfig

    output_dir = Path(OptimizationConfig().output_dir).expanduser()
    if output_dir.is_absolute():
        return output_dir.resolve()
    return (PROJECT_ROOT / output_dir).resolve()


def find_latest_run_dir(
        optimization_output_root: Path,
        metrics_name: str,
        index: int = 0,
) -> Path:
    if not optimization_output_root.exists():
        raise FileNotFoundError(
            f"OptimizationOutput folder does not exist: {optimization_output_root}"
        )

    if index < 0:
        raise ValueError(f"--index must be non-negative, got: {index}")

    candidate_run_dirs: list[dict[str, Any]] = []

    for metrics_csv_path in optimization_output_root.rglob(metrics_name):
        if not metrics_csv_path.is_file():
            continue

        candidate_run_dirs.append(
            {
                "run_dir": metrics_csv_path.parent,
                "metrics_modified_time_ns": metrics_csv_path.stat().st_mtime_ns,
            }
        )

    if not candidate_run_dirs:
        raise FileNotFoundError(
            f"No run folders with {metrics_name} found under: "
            f"{optimization_output_root}"
        )

    candidate_run_dirs.sort(
        key=lambda item: (item["metrics_modified_time_ns"], str(item["run_dir"])),
        reverse=True,
    )

    if index >= len(candidate_run_dirs):
        raise IndexError(
            f"Requested --index {index}, but only {len(candidate_run_dirs)} run(s) "
            f"with {metrics_name} were found under {optimization_output_root}."
        )

    return candidate_run_dirs[index]["run_dir"]


def lazy_evaluate_runs_imports():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from experiments.evaluate_runs import MeshCheckpoint, compute_geometry_rows
    except ModuleNotFoundError as exception:
        raise RuntimeError(
            "Live CD evaluation uses experiments/evaluate_runs.py and requires its "
            "geometry dependencies (notably Open3D). "
            "Alternatively, run without --gt to view loss curves only."
        ) from exception

    return MeshCheckpoint, compute_geometry_rows


def find_mesh_checkpoints(run_dir: Path, reconstruction_name: str) -> list[tuple[int, Path]]:
    checkpoint_root = run_dir / "mesh_checkpoints"
    if not checkpoint_root.is_dir():
        return []

    checkpoints: list[tuple[int, Path]] = []
    for mesh_path in checkpoint_root.glob(f"iter_*/{reconstruction_name}"):
        match = re.search(r"iter_(\d+)", mesh_path.parent.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), mesh_path.resolve()))

    return sorted(checkpoints, key=lambda item: item[0])


def update_geometry_evaluation_state(
        state: GeometryEvaluationState,
        run_dir: Path,
        latest_iteration: int,
        ground_truth_path: Path | None,
        geometry_every: int,
        reconstruction_name: str,
        samples: int,
        device_name: str,
        seed: int,
        scale: float,
        use_vertices: bool,
        metrics_dataframe: pd.DataFrame | None,
) -> bool:
    if state.run_dir != run_dir:
        state.run_dir = run_dir
        state.rows = []
        state.evaluated_iterations = set()

    if ground_truth_path is None:
        return False
    if geometry_every <= 0:
        raise ValueError(f"--geometry-every must be positive, got: {geometry_every}")

    eligible_checkpoints = [
        (iteration, mesh_path)
        for iteration, mesh_path in find_mesh_checkpoints(run_dir, reconstruction_name)
        if (
                iteration <= latest_iteration
                and iteration % geometry_every == 0
                and iteration not in state.evaluated_iterations
        )
    ]
    if not eligible_checkpoints:
        return False

    MeshCheckpoint, compute_geometry_rows = lazy_evaluate_runs_imports()
    new_rows = compute_geometry_rows(
        run_dir=run_dir,
        checkpoints=[
            MeshCheckpoint(iteration=iteration, mesh_path=mesh_path)
            for iteration, mesh_path in eligible_checkpoints
        ],
        ground_truth_path=ground_truth_path.expanduser().resolve(),
        samples=samples,
        device_name=device_name,
        seed=seed,
        scale=scale,
        use_vertices=use_vertices,
        print_each_score=False,
    )

    for metrics in new_rows:
        iteration = int(metrics["iteration"])
        state.rows.append(metrics)
        state.evaluated_iterations.add(iteration)

        # Print concise stats only at each CD update: loss, CD, completion, accuracy, points (split)
        loss_text = "n/a"
        points_text = "n/a"
        split_text = "n/a"
        try:
            if metrics_dataframe is not None and "iteration" in metrics_dataframe.columns:
                row_df = metrics_dataframe.loc[
                    metrics_dataframe["iteration"].astype(np.int64) == int(iteration)
                ]
                if not row_df.empty:
                    # Loss selection (prefer total mean, then RGB mean, else first available from selection logic)
                    try:
                        loss_column = select_loss_column(metrics_dataframe, explicit_loss_column=None)
                    except Exception:
                        loss_column = None
                    if loss_column is not None and loss_column in row_df.columns:
                        loss_val = pd.to_numeric(row_df[loss_column].iloc[-1], errors="coerce")
                        if np.isfinite(loss_val):
                            loss_text = f"{float(loss_val):.6g}"

                    # Point count (prefer num_points, then point_count)
                    for pc_name in ("num_points", "point_count"):
                        if pc_name in row_df.columns:
                            pc_val = pd.to_numeric(row_df[pc_name].iloc[-1], errors="coerce")
                            if np.isfinite(pc_val):
                                points_text = f"{int(pc_val)}"
                                break

                    # Split count (prefer total, then active, then per-iter events)
                    for sc in ("densification_split_points_total",
                               "densification_split_points_active",
                               "densification_split_points"):
                        if sc in row_df.columns:
                            sv = pd.to_numeric(row_df[sc].iloc[-1], errors="coerce")
                            if np.isfinite(sv):
                                try:
                                    split_text = f"{int(sv)}"
                                except Exception:
                                    split_text = f"{float(sv):.6g}"
                                break
        except Exception:
            pass

        print(
            f"loss={loss_text} | "
            f"CD={float(metrics['cd']):.6g} | "
            f"completion={float(metrics['completion']):.6g} | "
            f"accuracy={float(metrics['accuracy']):.6g} | "
            f"points={points_text} (split={split_text})",
            flush=True,
        )

    state.rows.sort(key=lambda row: int(row["iteration"]))
    return True


def filter_metrics_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    New metrics files write one loss-average row per iteration, even when one
    camera is optimized per iteration. Do not filter these rows.

    Older metrics files may contain one row per camera plus ALL_CAMERAS rows.
    For those, retain only ALL_CAMERAS rows when available.
    """
    averaged_loss_columns = (
        "loss_total_mean",
        "loss_rgb_mean",
        "loss_rgb_l2_mean",
        "loss_rgb_dssim_mean",
    )

    if any(column_name in dataframe.columns for column_name in averaged_loss_columns):
        return dataframe.copy()

    if "active_camera_name" in dataframe.columns:
        return dataframe.copy()

    if "camera_name" not in dataframe.columns:
        return dataframe.copy()

    all_cameras_mask = dataframe["camera_name"].astype(str) == "ALL_CAMERAS"
    if all_cameras_mask.any():
        return dataframe.loc[all_cameras_mask].copy()

    return dataframe.copy()


def prepare_metrics_dataframe(
        dataframe: pd.DataFrame,
        from_iteration: int | None,
        last_iterations: int | None,
        skip_opacity_reset_noise: int,
) -> pd.DataFrame:
    dataframe = filter_metrics_rows(dataframe)

    if "iteration" not in dataframe.columns:
        raise ValueError("metrics.csv does not contain an 'iteration' column")

    dataframe = dataframe.copy()
    dataframe["iteration"] = pd.to_numeric(dataframe["iteration"], errors="coerce")
    dataframe = dataframe.dropna(subset=["iteration"])
    dataframe["iteration"] = dataframe["iteration"].astype(np.int64)

    dataframe = dataframe.drop_duplicates(subset=["iteration"], keep="last")
    dataframe = dataframe.sort_values("iteration").reset_index(drop=True)

    if from_iteration is not None:
        if from_iteration < 0:
            raise ValueError(f"--from must be non-negative, got: {from_iteration}")

        dataframe = dataframe.loc[
            dataframe["iteration"] >= from_iteration
        ].reset_index(drop=True)

    if skip_opacity_reset_noise:
        opacity_reset_noise_mask = (
            (dataframe["iteration"] >= skip_opacity_reset_noise)
            & ((dataframe["iteration"] % skip_opacity_reset_noise) < 100)
        )
        dataframe = dataframe.loc[~opacity_reset_noise_mask].reset_index(drop=True)

    if last_iterations is not None:
        if last_iterations <= 0:
            raise ValueError(
                f"--iterations must be a positive integer, got: {last_iterations}"
            )

        dataframe = dataframe.tail(last_iterations).reset_index(drop=True)

    if "loss_rgb_dssim_mean" in dataframe.columns:
        dssim_values = pd.to_numeric(
            dataframe["loss_rgb_dssim_mean"], errors="coerce"
        )
        dataframe["rgb_ssim_mean"] = 1.0 - dssim_values

    return dataframe


def read_metrics_csv_safely(
        metrics_csv_path: Path,
        previous_dataframe: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """
    The optimizer can write while pandas is reading.
    Keep the previous valid dataframe instead of crashing on a partial row/write.
    """
    for _ in range(3):
        try:
            csv_bytes = metrics_csv_path.read_bytes()
            if not csv_bytes.strip():
                return previous_dataframe

            if not csv_bytes.endswith(b"\n"):
                last_newline_index = csv_bytes.rfind(b"\n")
                if last_newline_index < 0:
                    return previous_dataframe
                csv_bytes = csv_bytes[:last_newline_index + 1]

            dataframe = pd.read_csv(io.BytesIO(csv_bytes))
            if dataframe.empty:
                return previous_dataframe
            return dataframe
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
            time.sleep(0.05)

    return previous_dataframe


def get_available_columns(
        dataframe: pd.DataFrame,
        candidate_columns: list[str],
) -> list[str]:
    return [
        column_name
        for column_name in candidate_columns
        if column_name in dataframe.columns
    ]


def get_first_available_columns(
        dataframe: pd.DataFrame,
        candidate_groups: list[tuple[str, ...]],
) -> list[str]:
    selected_columns: list[str] = []

    for candidate_columns in candidate_groups:
        for column_name in candidate_columns:
            if column_name in dataframe.columns:
                selected_columns.append(column_name)
                break

    return selected_columns


def select_loss_column(
        dataframe: pd.DataFrame,
        explicit_loss_column: str | None,
) -> str:
    if explicit_loss_column is not None:
        if explicit_loss_column not in dataframe.columns:
            raise ValueError(
                f"Requested loss column '{explicit_loss_column}' was not found in "
                f"metrics.csv. Available columns: {list(dataframe.columns)}"
            )
        return explicit_loss_column

    preferred_columns = [
        "loss_total_mean",
        "loss_rgb_mean",
        "loss_rgb_l2_mean",
        "loss_rgb_dssim_mean",
        "rgb_ssim_mean",
        "loss_bsdf_decay_weighted_mean",
        "loss_opacity_prior_weighted_mean",
        "loss_intra_slab_depth_weighted_mean",
        "loss_curvature_scale_weighted_mean",
        "loss_normal_consistency_weighted_mean",
        "loss_depth_distortion_weighted_mean",
        "loss_total_sum",
        "loss_rgb_sum",
        "loss_bsdf_decay_weighted_sum",
        "loss_opacity_prior_weighted_sum",
        "loss_intra_slab_depth_weighted_sum",
        "loss_curvature_scale_weighted_sum",
        "loss_normal_consistency_weighted_sum",
        "loss_depth_distortion_weighted_sum",
    ]

    for column_name in preferred_columns:
        if column_name in dataframe.columns:
            return column_name

    raise ValueError(
        "Could not find a supported loss column in metrics.csv. "
        f"Available columns: {list(dataframe.columns)}"
    )


def dataframe_column_as_float_array(
        dataframe: pd.DataFrame,
        column_name: str,
) -> np.ndarray:
    return pd.to_numeric(
        dataframe[column_name],
        errors="coerce",
    ).to_numpy(dtype=np.float64)


def apply_loss_y_scale(axis, loss_y_scale: str) -> None:
    if loss_y_scale == "linear":
        return

    if loss_y_scale == "log":
        axis.set_yscale("log", nonpositive="mask")
        return

    axis.set_yscale("symlog", linthresh=1.0e-8)


def plot_linear_columns(
        axis,
        dataframe: pd.DataFrame,
        columns: list[str],
        style_map: dict[str, dict[str, Any]],
) -> None:
    for column_name in columns:
        values = dataframe_column_as_float_array(dataframe, column_name)

        axis.plot(
            dataframe["iteration"],
            values,
            label=column_name,
            **style_map.get(column_name, {}),
        )


def plot_top_loss_columns_with_rgb_axis(
        left_axis,
        dataframe: pd.DataFrame,
        loss_columns: list[str],
        dssim_columns: list[str],
        style_map: dict[str, dict[str, Any]],
):
    rgb_column_names = {
        "loss_rgb_mean",
        "loss_rgb_l2_mean",
        "loss_rgb_sum",
    }
    rgb_columns = [
        column_name
        for column_name in loss_columns
        if column_name in rgb_column_names
    ]
    left_loss_columns = [
        column_name
        for column_name in loss_columns
        if column_name not in rgb_column_names
    ]

    plot_linear_columns(
        left_axis,
        dataframe,
        left_loss_columns + dssim_columns,
        style_map,
    )

    right_axis = None

    if rgb_columns:
        right_axis = left_axis.twinx()
        plot_linear_columns(
            right_axis,
            dataframe,
            rgb_columns,
            style_map,
        )

    return right_axis


def place_legend_inside(
        axis,
        handles=None,
        labels=None,
        loc: str = "upper right",
) -> None:
    if handles is None or labels is None:
        handles, labels = axis.get_legend_handles_labels()

    if handles:
        axis.legend(handles, labels, loc=loc, **LEGEND_KWARGS)


def set_combined_legend(left_axis, right_axis=None) -> None:
    left_handles, left_labels = left_axis.get_legend_handles_labels()

    if right_axis is not None:
        right_handles, right_labels = right_axis.get_legend_handles_labels()
    else:
        right_handles, right_labels = [], []

    handles = left_handles + right_handles
    labels = left_labels + right_labels

    place_legend_inside(left_axis, handles, labels, loc="best")


def plot_positive_log_columns(
        axis,
        dataframe: pd.DataFrame,
        columns: list[str],
        style_map: dict[str, dict[str, Any]],
) -> bool:
    plotted_any_positive_values = False

    for column_name in columns:
        values = dataframe_column_as_float_array(dataframe, column_name)
        positive_values = np.where(values > 0.0, values, np.nan)

        if np.any(np.isfinite(positive_values)):
            plotted_any_positive_values = True

        axis.plot(
            dataframe["iteration"],
            positive_values,
            label=column_name,
            **style_map.get(column_name, {}),
        )

    return plotted_any_positive_values


def latest_numeric_value(dataframe: pd.DataFrame, column_name: str) -> float | None:
    if column_name not in dataframe.columns or dataframe.empty:
        return None

    values = pd.to_numeric(dataframe[column_name], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return None

    return float(values.iloc[-1])


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "unknown"

    total_seconds = int(round(max(0.0, seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds_part:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds_part:02d}s"
    return f"{seconds_part:d}s"


def optimization_status_text(dataframe: pd.DataFrame, latest_iteration: int) -> str:
    total_time_sec = latest_numeric_value(dataframe, "total_time_sec")
    iteration_time_sec = latest_numeric_value(dataframe, "iteration_time_sec")

    parts = [f"time={format_duration(total_time_sec)}"]

    average_iterations_per_second = None
    if total_time_sec is not None and total_time_sec > 0.0:
        run_start_iteration = None
        if "iteration" in dataframe.columns and not dataframe.empty:
            try:
                iters = pd.to_numeric(dataframe["iteration"], errors="coerce")
                iters = iters[np.isfinite(iters)]
                if not iters.empty:
                    run_start_iteration = int(np.min(iters))
            except Exception:
                run_start_iteration = None

        if run_start_iteration is not None:
            completed_iterations = max(0, int(latest_iteration) - int(run_start_iteration) + 1)
            average_iterations_per_second = completed_iterations / float(total_time_sec)
        else:
            # Fallback: behave as before if we cannot infer the start iteration
            average_iterations_per_second = float(latest_iteration) / float(total_time_sec)

    if average_iterations_per_second is not None and np.isfinite(average_iterations_per_second):
        parts.append(f"avg={average_iterations_per_second:.2f} it/s")

    if iteration_time_sec is not None and iteration_time_sec > 0.0:
        parts.append(f"last={1.0 / iteration_time_sec:.2f} it/s")

    return " | ".join(parts)


def compact_point_count_label(point_count_windowed: bool, point_count_row_count: int) -> str:
    if point_count_windowed:
        return "windowed"
    return f"aligned ({point_count_row_count} rows loaded)"


def format_compact_point_count(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"

    absolute_value = abs(float(value))
    if absolute_value < 1_000.0:
        return f"{int(round(value))}"

    value_in_thousands = float(value) / 1_000.0
    decimals = 1 if abs(value_in_thousands) < 100.0 else 0
    return f"{value_in_thousands:.{decimals}f}k"


def point_count_series_label(base_label: str, values: np.ndarray) -> str:
    finite_values = values[np.isfinite(values)]
    latest_value = float(finite_values[-1]) if finite_values.size > 0 else float("nan")
    return f"{base_label} ({format_compact_point_count(latest_value)})"


def shorten_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    if max_chars <= 3:
        return text[:max_chars]

    left_chars = (max_chars - 3) // 2
    right_chars = max_chars - 3 - left_chars
    return f"{text[:left_chars]}...{text[-right_chars:]}"


def dataframe_iteration_bounds(dataframe: pd.DataFrame) -> tuple[int, int]:
    return int(dataframe["iteration"].iloc[0]), int(dataframe["iteration"].iloc[-1])


def filter_geometry_rows_to_iteration_bounds(
        geometry_rows: list[dict[str, Any]],
        iteration_min: int,
        iteration_max: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in geometry_rows
        if iteration_min <= int(row["iteration"]) <= iteration_max
    ]


def set_iteration_x_limits(axes, iteration_min: int, iteration_max: int) -> None:
    if iteration_min == iteration_max:
        padding = max(1.0, abs(float(iteration_min)) * 0.01)
        x_min = float(iteration_min) - padding
        x_max = float(iteration_max) + padding
    else:
        x_min = float(iteration_min)
        x_max = float(iteration_max)

    for axis in np.atleast_1d(axes):
        if axis is not None:
            axis.set_xlim(x_min, x_max)


def plot_geometry_rows(axis, geometry_rows: list[dict[str, Any]]) -> None:
    sorted_rows = sorted(geometry_rows, key=lambda row: int(row["iteration"]))
    iterations = [int(row["iteration"]) for row in sorted_rows]
    cds = [float(row["cd"]) for row in sorted_rows]
    accuracies = [float(row["accuracy"]) for row in sorted_rows]
    completions = [float(row["completion"]) for row in sorted_rows]

    axis.plot(iterations, cds, marker="o", linewidth=1.8, label="symmetric CD")
    axis.plot(iterations, accuracies, marker="o", linewidth=1.1, alpha=0.8, label="accuracy")
    axis.plot(iterations, completions, marker="o", linewidth=1.1, alpha=0.8, label="completion")
    axis.set_ylabel("Chamfer")
    axis.grid(True)
    place_legend_inside(axis, loc="best")


def draw_metrics_figure(
        figure,
        dataframe: pd.DataFrame,
        metrics_csv_path: Path,
        explicit_loss_column: str | None,
        plot_all_losses: bool,
        from_iteration: int | None,
        last_iterations: int | None,
        skip_opacity_reset_noise: int,
        point_count_windowed: bool,
        loss_y_scale: str,
        geometry_rows: list[dict[str, Any]],
) -> str:
    loss_dataframe = prepare_metrics_dataframe(
        dataframe=dataframe,
        from_iteration=from_iteration,
        last_iterations=last_iterations,
        skip_opacity_reset_noise=skip_opacity_reset_noise,
    )

    point_count_dataframe = prepare_metrics_dataframe(
        dataframe=dataframe,
        from_iteration=from_iteration,
        last_iterations=last_iterations if point_count_windowed else None,
        skip_opacity_reset_noise=0,
    )

    if loss_dataframe.empty:
        raise ValueError("No valid loss rows to plot yet.")

    iteration_min, iteration_max = dataframe_iteration_bounds(loss_dataframe)
    visible_geometry_rows = filter_geometry_rows_to_iteration_bounds(
        geometry_rows=geometry_rows,
        iteration_min=iteration_min,
        iteration_max=iteration_max,
    )

    figure.clear()

    if not plot_all_losses:
        loss_column_name = select_loss_column(loss_dataframe, explicit_loss_column)
        include_geometry_panel = len(visible_geometry_rows) > 0
        axes = figure.subplots(
            2 if include_geometry_panel else 1,
            1,
            sharex=include_geometry_panel,
            gridspec_kw={"height_ratios": [1.4, 1.0]} if include_geometry_panel else None,
        )
        axes = np.atleast_1d(axes).tolist()
        axis = axes[0]

        axis.plot(
            loss_dataframe["iteration"],
            dataframe_column_as_float_array(loss_dataframe, loss_column_name),
            linewidth=2.2,
            color="tab:blue",
        )

        apply_loss_y_scale(axis, loss_y_scale)

        latest_iteration = int(loss_dataframe["iteration"].iloc[-1])
        title_run_name = shorten_middle(metrics_csv_path.parent.name, max_chars=80)
        axis.set_xlabel("Iteration")
        axis.set_ylabel(loss_column_name)
        axis.set_title(
            f"{loss_column_name} over iterations\n"
            f"{title_run_name}\n"
            f"iter={latest_iteration} | "
            f"{optimization_status_text(dataframe, latest_iteration)}"
        )
        axis.grid(True)

        if include_geometry_panel:
            axis.set_xlabel("")
            geometry_axis = axes[1]
            plot_geometry_rows(geometry_axis, visible_geometry_rows)
            geometry_axis.set_xlabel("Iteration")

        set_iteration_x_limits(axes, iteration_min, iteration_max)
        figure.tight_layout()
        return loss_column_name

    top_columns = get_first_available_columns(
        loss_dataframe,
        [
            ("loss_rgb_mean", "loss_rgb_sum"),
            ("loss_rgb_l2_mean",),
            ("loss_total_mean", "loss_total_sum"),
        ],
    )

    dssim_top_columns = get_available_columns(
        loss_dataframe,
        [
            "loss_rgb_dssim_mean",
        ],
    )

    weighted_regularizer_columns = get_first_available_columns(
        loss_dataframe,
        [
            (
                "loss_depth_distortion_weighted_mean",
                "loss_depth_distortion_weighted_sum",
            ),
            (
                "loss_normal_consistency_weighted_mean",
                "loss_normal_consistency_weighted_sum",
            ),
            (
                "loss_opacity_prior_weighted_mean",
                "loss_opacity_prior_weighted_sum",
            ),
            (
                "loss_intra_slab_depth_weighted_mean",
                "loss_intra_slab_depth_weighted_sum",
            ),
            (
                "loss_curvature_scale_weighted_mean",
                "loss_curvature_scale_weighted_sum",
            ),
            (
                "loss_bsdf_decay_weighted_mean",
                "loss_bsdf_decay_weighted_sum",
            ),
        ],
    )

    raw_diagnostic_columns: list[str] = []

    point_count_columns = get_available_columns(
        point_count_dataframe,
        [
            "num_points",
            "point_count",
        ],
    )
    point_clone_active_columns = get_available_columns(
        point_count_dataframe,
        [
            "densification_clone_points_active",
        ],
    )
    point_split_trigger_active_columns = get_available_columns(
        point_count_dataframe,
        [
            "densification_position_split_points_active",
            "densification_curvature_split_points_active",
        ],
    )
    point_growth_active_columns = point_clone_active_columns + (
        point_split_trigger_active_columns
        if point_split_trigger_active_columns
        else get_available_columns(
            point_count_dataframe,
            ["densification_split_points_active"],
        )
    )
    point_clone_total_columns = get_available_columns(
        point_count_dataframe,
        [
            "densification_clone_points_total",
        ],
    )
    point_split_trigger_total_columns = get_available_columns(
        point_count_dataframe,
        [
            "densification_position_split_points_total",
            "densification_curvature_split_points_total",
        ],
    )
    point_growth_total_columns = point_clone_total_columns + (
        point_split_trigger_total_columns
        if point_split_trigger_total_columns
        else get_available_columns(
            point_count_dataframe,
            ["densification_split_points_total"],
        )
    )
    point_clone_event_columns = get_available_columns(
        point_count_dataframe,
        [
            "densification_clone_points",
        ],
    )
    point_split_trigger_event_columns = get_available_columns(
        point_count_dataframe,
        [
            "densification_position_split_points",
            "densification_curvature_split_points",
        ],
    )
    point_growth_event_columns = point_clone_event_columns + (
        point_split_trigger_event_columns
        if point_split_trigger_event_columns
        else get_available_columns(
            point_count_dataframe,
            ["densification_split_points"],
        )
    )
    point_prune_event_columns = get_available_columns(
        point_count_dataframe,
        [
            "prune_scale_area_points",
            "prune_inactive_gradient_points",
        ],
    )

    if (
            not top_columns
            and not dssim_top_columns
            and not weighted_regularizer_columns
            and not raw_diagnostic_columns
            and not point_count_columns
            and not point_growth_active_columns
            and not point_growth_total_columns
            and not point_growth_event_columns
            and not point_prune_event_columns
    ):
        selected_loss_column = select_loss_column(
            loss_dataframe,
            explicit_loss_column,
        )
        top_columns = [selected_loss_column]

    style_map = {
        "loss_rgb_mean": dict(color="tab:blue", linewidth=2.5, alpha=1.0),
        "loss_rgb_l2_mean": dict(
            color="tab:cyan",
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
        ),
        "loss_rgb_dssim_mean": dict(
            color="tab:red",
            linewidth=2.2,
            alpha=1.0,
        ),
        "loss_total_mean": dict(color="tab:orange", linewidth=2.0, alpha=0.95),

        "loss_depth_distortion_weighted_mean": dict(
            color="tab:red",
            linewidth=1.8,
            alpha=0.95,
        ),
        "loss_normal_consistency_weighted_mean": dict(
            color="tab:green",
            linewidth=1.8,
            alpha=0.95,
        ),
        "loss_opacity_prior_weighted_mean": dict(
            color="#A020F0",
            linewidth=1.8,
            alpha=0.95,
        ),
        "loss_intra_slab_depth_weighted_mean": dict(
            color="tab:cyan",
            linewidth=1.8,
            alpha=0.95,
        ),
        "loss_curvature_scale_weighted_mean": dict(
            color="tab:pink",
            linewidth=1.8,
            alpha=0.95,
        ),
        "loss_bsdf_decay_weighted_mean": dict(
            color="tab:brown",
            linewidth=1.8,
            alpha=0.95,
        ),

        "loss_rgb_sum": dict(color="tab:blue", linewidth=2.5, alpha=1.0),
        "loss_total_sum": dict(color="tab:orange", linewidth=2.0, alpha=0.95),

        "loss_depth_distortion_weighted_sum": dict(
            color="tab:red",
            linewidth=1.8,
            alpha=0.95,
        ),
        "loss_normal_consistency_weighted_sum": dict(
            color="tab:green",
            linewidth=1.8,
            alpha=0.95,
        ),
        "loss_opacity_prior_weighted_sum": dict(
            color="#A020F0",
            linewidth=1.8,
            alpha=0.95,
        ),
        "loss_intra_slab_depth_weighted_sum": dict(
            color="tab:cyan",
            linewidth=1.8,
            alpha=0.95,
        ),
        "loss_curvature_scale_weighted_sum": dict(
            color="tab:pink",
            linewidth=1.8,
            alpha=0.95,
        ),
        "loss_bsdf_decay_weighted_sum": dict(
            color="tab:brown",
            linewidth=1.8,
            alpha=0.95,
        ),

        "num_points": dict(color="tab:brown", linewidth=2.0, alpha=0.95),
        "point_count": dict(color="tab:brown", linewidth=2.0, alpha=0.95),
        "densification_clone_points_active": dict(
            color="tab:green",
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
        ),
        "densification_split_points_active": dict(
            color="#A020F0",
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
        ),
        "densification_position_split_points_active": dict(
            color="tab:blue",
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
        ),
        "densification_curvature_split_points_active": dict(
            color="#A020F0",
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
        ),
        "densification_clone_points_total": dict(
            color="tab:green",
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
        ),
        "densification_split_points_total": dict(
            color="#A020F0",
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
        ),
        "densification_position_split_points_total": dict(
            color="tab:blue",
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
        ),
        "densification_curvature_split_points_total": dict(
            color="#A020F0",
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
        ),
        "prune_scale_area_points_total": dict(
            color="tab:red",
            linewidth=1.8,
            linestyle="-.",
            alpha=0.95,
        ),
        "prune_inactive_gradient_points_total": dict(
            color="#111111",
            linewidth=2.1,
            linestyle=":",
            alpha=1.0,
        ),
    }

    include_geometry_panel = len(visible_geometry_rows) > 0
    include_raw_panel = len(raw_diagnostic_columns) > 0
    include_point_count_panel = (
            len(point_count_columns) > 0
            or len(point_growth_active_columns) > 0
            or len(point_growth_total_columns) > 0
            or len(point_growth_event_columns) > 0
            or len(point_prune_event_columns) > 0
    )
    num_panels = (
        2
        + int(include_raw_panel)
        + int(include_geometry_panel)
        + int(include_point_count_panel)
    )

    height_ratios = [1.2, 1.0]
    if include_raw_panel:
        height_ratios.append(1.0)
    if include_geometry_panel:
        height_ratios.append(1.0)
    if include_point_count_panel:
        height_ratios.append(0.8)

    axes = figure.subplots(
        num_panels,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.atleast_1d(axes).tolist()

    ax_top = axes[0]
    next_axis_index = 1
    ax_weighted = axes[next_axis_index]
    next_axis_index += 1
    ax_raw = axes[next_axis_index] if include_raw_panel else None
    if include_raw_panel:
        next_axis_index += 1
    ax_geometry = axes[next_axis_index] if include_geometry_panel else None
    if include_geometry_panel:
        next_axis_index += 1
    ax_point_count = axes[next_axis_index] if include_point_count_panel else None

    ax_top_right = plot_top_loss_columns_with_rgb_axis(
        ax_top,
        loss_dataframe,
        top_columns,
        dssim_top_columns,
        style_map,
    )

    plot_linear_columns(
        ax_weighted,
        loss_dataframe,
        weighted_regularizer_columns,
        style_map,
    )

    raw_has_positive_values = False
    if ax_raw is not None:
        raw_has_positive_values = plot_positive_log_columns(
            ax_raw,
            loss_dataframe,
            raw_diagnostic_columns,
            style_map,
        )

    apply_loss_y_scale(ax_top, loss_y_scale)
    apply_loss_y_scale(ax_weighted, loss_y_scale)
    if ax_top_right is not None:
        apply_loss_y_scale(ax_top_right, loss_y_scale)

    has_total_loss = any(
            column_name in {"loss_total_mean", "loss_total_sum"}
            for column_name in top_columns
    )
    if has_total_loss and dssim_top_columns:
        ax_top.set_ylabel(f"Mean total / DSSIM loss ({loss_y_scale})")
    elif has_total_loss:
        ax_top.set_ylabel(f"Mean total loss ({loss_y_scale})")
    elif dssim_top_columns:
        ax_top.set_ylabel(f"Mean DSSIM loss ({loss_y_scale})")
    else:
        ax_top.set_ylabel(f"Mean image loss ({loss_y_scale})")

    if ax_top_right is not None:
        rgb_axis_color = style_map["loss_rgb_mean"]["color"]
        ax_top_right.set_ylabel(
            f"Mean RGB objective / half-MSE ({loss_y_scale})",
            color=rgb_axis_color,
        )
        ax_top_right.tick_params(axis="y", labelcolor=rgb_axis_color)

    latest_iteration = int(loss_dataframe["iteration"].iloc[-1])
    loss_row_count = len(loss_dataframe)
    point_count_row_count = len(point_count_dataframe)

    loss_average_text = ""

    if (
            "loss_average_camera_count" in loss_dataframe.columns
            and "loss_average_expected_camera_count" in loss_dataframe.columns
    ):
        latest_averaged_camera_count = pd.to_numeric(
            pd.Series([loss_dataframe["loss_average_camera_count"].iloc[-1]]),
            errors="coerce",
        ).iloc[0]

        latest_expected_camera_count = pd.to_numeric(
            pd.Series(
                [loss_dataframe["loss_average_expected_camera_count"].iloc[-1]]
            ),
            errors="coerce",
        ).iloc[0]

        if (
                np.isfinite(latest_averaged_camera_count)
                and np.isfinite(latest_expected_camera_count)
        ):
            loss_average_text = (
                f" | avg cams={int(latest_averaged_camera_count)}"
                f"/{int(latest_expected_camera_count)}"
            )

    point_count_history_label = compact_point_count_label(
        point_count_windowed=point_count_windowed,
        point_count_row_count=point_count_row_count,
    )

    ax_top.set_title(
        f"Live optimization metrics\n"
        f"{shorten_middle(metrics_csv_path.parent.name, max_chars=80)}\n"
        f"iter={latest_iteration} | loss rows={loss_row_count}{loss_average_text} | "
        f"{optimization_status_text(dataframe, latest_iteration)}"
        f"\npoints={point_count_history_label}"
    )

    if ax_top_right is not None:
        # Total loss and DSSIM own the left scale; RGB objective and half-MSE
        # use the independent right scale.
        ax_top.set_axisbelow(True)
        ax_top_right.set_axisbelow(True)
        ax_top_right.patch.set_visible(False)
        ax_top.grid(
            True,
            which="both",
            color="0.72",
            linewidth=0.8,
            alpha=0.55,
        )
        ax_top_right.grid(False)
    else:
        ax_top.set_axisbelow(True)
        ax_top.grid(
            True,
            which="both",
            color="0.72",
            linewidth=0.8,
            alpha=0.55,
        )
    set_combined_legend(ax_top, ax_top_right)

    ax_weighted.set_ylabel(f"Mean weighted regularizers ({loss_y_scale})")
    ax_weighted.grid(True)
    if weighted_regularizer_columns:
        place_legend_inside(ax_weighted, loc="best")

    if ax_raw is not None:
        ax_raw.set_ylabel("Mean raw diagnostics")
        if raw_has_positive_values:
            ax_raw.set_yscale("log")
        ax_raw.grid(True)
        place_legend_inside(ax_raw, loc="best")

    if ax_geometry is not None:
        plot_geometry_rows(ax_geometry, visible_geometry_rows)

    if ax_point_count is not None:
        if point_count_columns:
            point_count_column = point_count_columns[0]
            point_values = dataframe_column_as_float_array(
                point_count_dataframe,
                point_count_column,
            )

            ax_point_count.step(
                point_count_dataframe["iteration"],
                point_values,
                where="post",
                label=point_count_series_label("points", point_values),
                **style_map.get(point_count_column, {}),
            )

        if point_growth_active_columns:
            point_growth_columns = point_growth_active_columns
            point_growth_labels = {
                "densification_clone_points_active": "clone-created active",
                "densification_split_points_active": "split-created active",
                "densification_position_split_points_active": "position-split active",
                "densification_curvature_split_points_active": "curvature-split active",
            }
        else:
            point_growth_columns = point_growth_total_columns
            point_growth_labels = {
                "densification_clone_points_total": "clone additions total",
                "densification_split_points_total": "split additions total",
                "densification_position_split_points_total": "position-split additions total",
                "densification_curvature_split_points_total": "curvature-split additions total",
            }

        for column_name in point_growth_columns:
            values = dataframe_column_as_float_array(point_count_dataframe, column_name)
            label = point_count_series_label(
                point_growth_labels.get(column_name, column_name),
                values,
            )
            ax_point_count.step(
                point_count_dataframe["iteration"],
                values,
                where="post",
                label=label,
                **style_map.get(column_name, {}),
            )

        if not point_growth_active_columns and not point_growth_total_columns:
            for column_name in point_growth_event_columns:
                values = np.nancumsum(
                    np.nan_to_num(
                        dataframe_column_as_float_array(point_count_dataframe, column_name),
                        nan=0.0,
                    )
                )
                base_label = {
                    "densification_clone_points": "clone additions total",
                    "densification_split_points": "split additions total",
                    "densification_position_split_points": "position-split additions total",
                    "densification_curvature_split_points": "curvature-split additions total",
                }.get(column_name, f"{column_name} total")
                ax_point_count.step(
                    point_count_dataframe["iteration"],
                    values,
                    where="post",
                    label=point_count_series_label(base_label, values),
                    **style_map.get(f"{column_name}_total", {}),
                )

        point_prune_labels = {
            "prune_scale_area_points": "pruned: small area total",
            "prune_inactive_gradient_points": "pruned: inactive gradient total",
        }
        for column_name in point_prune_event_columns:
            values = np.nancumsum(
                np.nan_to_num(
                    dataframe_column_as_float_array(point_count_dataframe, column_name),
                    nan=0.0,
                )
            )
            ax_point_count.step(
                point_count_dataframe["iteration"],
                values,
                where="post",
                label=point_count_series_label(
                    point_prune_labels.get(column_name, f"{column_name} total"),
                    values,
                ),
                **style_map.get(f"{column_name}_total", {}),
            )

        ax_point_count.set_ylabel("Points")
        ax_point_count.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax_point_count.grid(True)
        place_legend_inside(ax_point_count, loc="best")

    for axis in (ax_top, ax_weighted, ax_raw, ax_geometry, ax_point_count):
        if axis is not None:
            axis.set_xlabel("")

    bottom_axis = ax_point_count or ax_geometry or ax_raw or ax_weighted
    bottom_axis.set_xlabel("Iteration")

    set_iteration_x_limits(axes, iteration_min, iteration_max)
    figure.tight_layout()

    plotted_columns = (
        top_columns
        + dssim_top_columns
        + weighted_regularizer_columns
        + raw_diagnostic_columns
        + point_count_columns
        + point_growth_active_columns
        + point_growth_total_columns
        + ([] if point_growth_active_columns or point_growth_total_columns else point_growth_event_columns)
        + point_prune_event_columns
    )

    return ", ".join(plotted_columns)


def get_file_state(path: Path) -> tuple[float, int] | None:
    try:
        file_stat = path.stat()
        return file_stat.st_mtime, file_stat.st_size
    except FileNotFoundError:
        return None


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
        return run_dir

    optimization_output_root = (
        args.optimization_output_root.expanduser().resolve()
        if args.optimization_output_root is not None
        else default_optimization_output_root()
    )

    return find_latest_run_dir(
        optimization_output_root=optimization_output_root,
        metrics_name=args.metrics_name,
        index=args.index,
    )


def main() -> None:
    args = parse_args()

    if args.refresh_seconds <= 0.0:
        raise ValueError(
            f"--refresh-seconds must be positive, got: {args.refresh_seconds}"
        )
    if args.geometry_every <= 0:
        raise ValueError(f"--geometry-every must be positive, got: {args.geometry_every}")
    if args.ground_truth is not None and not args.ground_truth.expanduser().is_file():
        raise FileNotFoundError(f"--gt does not exist: {args.ground_truth.expanduser()}")

    plt.ion()
    plt.rcParams["figure.raise_window"] = False

    figure = plt.figure(figsize=(12, 9))
    previous_dataframe: pd.DataFrame | None = None
    previous_file_state: tuple[float, int] | None = None
    previous_run_dir: Path | None = None
    last_plot_warning: str | None = None
    geometry_state = GeometryEvaluationState()

    print("Starting live metrics viewer. Press Ctrl+C in the terminal to stop.")
    print(f"Refresh interval   : {args.refresh_seconds:.3f}s")
    print(f"Watch latest       : {args.watch_latest}")
    print(f"Save plot          : {args.save_plot}")
    print(f"Skip reset noise   : {args.skip}")
    print(f"Loss y-scale       : {args.loss_y_scale}")
    print(f"Point count window : {args.point_count_windowed}")
    if args.ground_truth is not None:
        print(f"Live CD GT         : {args.ground_truth.expanduser().resolve()}")
        print(f"Live CD interval   : {args.geometry_every} iterations")

    try:
        while plt.fignum_exists(figure.number):
            run_dir = resolve_run_dir(args)
            metrics_csv_path = run_dir / args.metrics_name

            if previous_run_dir != run_dir:
                previous_run_dir = run_dir
                previous_file_state = None
                previous_dataframe = None
                last_plot_warning = None

                print()
                print(f"Watching run       : {run_dir}")
                print(f"Metrics file       : {metrics_csv_path}")

            file_state = get_file_state(metrics_csv_path)

            if file_state is None:
                time.sleep(args.refresh_seconds)
                continue

            file_changed = file_state != previous_file_state
            geometry_changed = False

            if file_changed:
                dataframe = read_metrics_csv_safely(
                    metrics_csv_path=metrics_csv_path,
                    previous_dataframe=previous_dataframe,
                )

                if dataframe is not None:
                    previous_dataframe = dataframe
                    previous_file_state = file_state

            if previous_dataframe is not None:
                try:
                    geometry_dataframe = prepare_metrics_dataframe(
                        dataframe=previous_dataframe,
                        from_iteration=None,
                        last_iterations=None,
                        skip_opacity_reset_noise=0,
                    )
                    if not geometry_dataframe.empty:
                        geometry_latest_iteration = int(geometry_dataframe["iteration"].iloc[-1])
                        geometry_changed = update_geometry_evaluation_state(
                            state=geometry_state,
                            run_dir=run_dir,
                            latest_iteration=geometry_latest_iteration,
                            ground_truth_path=args.ground_truth,
                            geometry_every=args.geometry_every,
                            reconstruction_name=args.reconstruction_name,
                            samples=args.geometry_samples,
                            device_name=args.geometry_device,
                            seed=args.geometry_seed,
                            scale=args.geometry_scale,
                            use_vertices=args.geometry_use_vertices,
                            metrics_dataframe=geometry_dataframe,
                        )
                except ValueError:
                    pass

            if previous_dataframe is not None and (file_changed or geometry_changed):
                try:
                    plotted_columns = draw_metrics_figure(
                        figure=figure,
                        dataframe=previous_dataframe,
                        metrics_csv_path=metrics_csv_path,
                        explicit_loss_column=args.loss_column,
                        plot_all_losses=args.plot_all_losses,
                        from_iteration=args.from_iteration,
                        last_iterations=args.iterations,
                        skip_opacity_reset_noise=args.skip,
                        point_count_windowed=args.point_count_windowed,
                        loss_y_scale=args.loss_y_scale,
                        geometry_rows=geometry_state.rows,
                    )
                except ValueError as exception:
                    warning_text = str(exception)
                    if warning_text != last_plot_warning:
                        print(
                            f"Waiting for plottable metrics rows: {warning_text}",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_plot_warning = warning_text
                else:
                    last_plot_warning = None
                    figure.canvas.draw_idle()
                    figure.canvas.flush_events()

                    if args.save_plot:
                        output_png_path = run_dir / args.loss_output_name
                        figure.savefig(output_png_path, dpi=200)

            plt.pause(args.refresh_seconds)

            if not args.watch_latest and args.run_dir is None:
                args.run_dir = run_dir

    except KeyboardInterrupt:
        print()
        print("Stopped live metrics viewer.")


if __name__ == "__main__":
    main()
