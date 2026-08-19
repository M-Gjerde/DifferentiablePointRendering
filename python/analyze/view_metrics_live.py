from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import StrMethodFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live plot metrics.csv while an optimization run is writing to it. "
            "By default, the newest run folder under OptimizationOutput is used."
        )
    )
    parser.add_argument(
        "--optimization-output-root",
        type=Path,
        required=False,
        default=Path("OptimizationOutput"),
        help="Path to the OptimizationOutput directory.",
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

    for child in optimization_output_root.iterdir():
        if not child.is_dir():
            continue

        metrics_csv_path = child / metrics_name
        if not metrics_csv_path.exists():
            continue

        # Deliberately use run-folder age, not metrics.csv modification time.
        # Updating a file inside a directory does not normally update the
        # directory mtime on Linux.
        run_dir_modified_time_ns = child.stat().st_mtime_ns

        candidate_run_dirs.append(
            {
                "run_dir": child,
                "run_dir_modified_time_ns": run_dir_modified_time_ns,
            }
        )

    if not candidate_run_dirs:
        raise FileNotFoundError(
            f"No run folders with {metrics_name} found under: "
            f"{optimization_output_root}"
        )

    candidate_run_dirs.sort(
        key=lambda item: item["run_dir_modified_time_ns"],
        reverse=True,
    )

    if index >= len(candidate_run_dirs):
        raise IndexError(
            f"Requested --index {index}, but only {len(candidate_run_dirs)} run(s) "
            f"with {metrics_name} were found under {optimization_output_root}."
        )

    return candidate_run_dirs[index]["run_dir"]


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
            dataframe = pd.read_csv(metrics_csv_path)
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
        "loss_bsdf_decay_weighted_mean",
        "loss_bsdf_decay_raw_mean",
        "loss_normal_consistency_weighted_mean",
        "loss_depth_distortion_weighted_mean",
        "loss_normal_consistency_raw_mean",
        "loss_depth_distortion_raw_mean",
        "loss_total_sum",
        "loss_rgb_sum",
        "loss_bsdf_decay_weighted_sum",
        "loss_bsdf_decay_raw_sum",
        "loss_normal_consistency_weighted_sum",
        "loss_depth_distortion_weighted_sum",
        "loss_normal_consistency_raw_sum",
        "loss_depth_distortion_raw_sum",
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


def plot_top_loss_columns_with_dual_axis(
        left_axis,
        dataframe: pd.DataFrame,
        columns: list[str],
        style_map: dict[str, dict[str, Any]],
):
    rgb_column_names = {
        "loss_rgb_mean",
        "loss_rgb_sum",
    }
    total_column_names = {
        "loss_total_mean",
        "loss_total_sum",
    }

    rgb_columns = [
        column_name
        for column_name in columns
        if column_name in rgb_column_names
    ]
    total_columns = [
        column_name
        for column_name in columns
        if column_name in total_column_names
    ]
    extra_columns = [
        column_name
        for column_name in columns
        if column_name not in rgb_column_names | total_column_names
    ]

    plot_linear_columns(
        left_axis,
        dataframe,
        rgb_columns + extra_columns,
        style_map,
    )

    right_axis = None

    if total_columns:
        right_axis = left_axis.twinx()
        plot_linear_columns(
            right_axis,
            dataframe,
            total_columns,
            style_map,
        )

    return right_axis


def set_combined_legend(left_axis, right_axis=None) -> None:
    left_handles, left_labels = left_axis.get_legend_handles_labels()

    if right_axis is not None:
        right_handles, right_labels = right_axis.get_legend_handles_labels()
    else:
        right_handles, right_labels = [], []

    handles = left_handles + right_handles
    labels = left_labels + right_labels

    if handles:
        left_axis.legend(handles, labels, loc="lower left")


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

    figure.clear()

    if not plot_all_losses:
        loss_column_name = select_loss_column(loss_dataframe, explicit_loss_column)
        axis = figure.subplots(1, 1)

        axis.plot(
            loss_dataframe["iteration"],
            dataframe_column_as_float_array(loss_dataframe, loss_column_name),
            linewidth=2.2,
            color="tab:blue",
        )

        apply_loss_y_scale(axis, loss_y_scale)

        axis.set_xlabel("Iteration")
        axis.set_ylabel(loss_column_name)
        axis.set_title(
            f"{loss_column_name} over iterations\n{metrics_csv_path.parent.name}"
        )
        axis.grid(True)

        figure.tight_layout()
        return loss_column_name

    top_columns = get_first_available_columns(
        loss_dataframe,
        [
            ("loss_rgb_mean", "loss_rgb_sum"),
            ("loss_total_mean", "loss_total_sum"),
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
                "loss_bsdf_decay_weighted_mean",
                "loss_bsdf_decay_weighted_sum",
            ),
        ],
    )

    raw_diagnostic_columns = get_first_available_columns(
        loss_dataframe,
        [
            (
                "loss_depth_distortion_raw_mean",
                "loss_depth_distortion_raw_sum",
            ),
            (
                "loss_normal_consistency_raw_mean",
                "loss_normal_consistency_raw_sum",
            ),
            (
                "loss_bsdf_decay_raw_mean",
                "loss_bsdf_decay_raw_sum",
            ),
        ],
    )

    point_count_columns = get_available_columns(
        point_count_dataframe,
        [
            "num_points",
            "point_count",
        ],
    )

    if (
            not top_columns
            and not weighted_regularizer_columns
            and not raw_diagnostic_columns
            and not point_count_columns
    ):
        selected_loss_column = select_loss_column(
            loss_dataframe,
            explicit_loss_column,
        )
        top_columns = [selected_loss_column]

    style_map = {
        "loss_rgb_mean": dict(color="tab:blue", linewidth=2.5, alpha=1.0),
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
        "loss_bsdf_decay_weighted_mean": dict(
            color="tab:brown",
            linewidth=1.8,
            alpha=0.95,
        ),

        "loss_depth_distortion_raw_mean": dict(
            color="tab:red",
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
        ),
        "loss_normal_consistency_raw_mean": dict(
            color="tab:green",
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
        ),
        "loss_bsdf_decay_raw_mean": dict(
            color="tab:brown",
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
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
        "loss_bsdf_decay_weighted_sum": dict(
            color="tab:brown",
            linewidth=1.8,
            alpha=0.95,
        ),

        "loss_depth_distortion_raw_sum": dict(
            color="tab:red",
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
        ),
        "loss_normal_consistency_raw_sum": dict(
            color="tab:green",
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
        ),
        "loss_bsdf_decay_raw_sum": dict(
            color="tab:brown",
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
        ),

        "num_points": dict(color="tab:brown", linewidth=2.0, alpha=0.95),
        "point_count": dict(color="tab:brown", linewidth=2.0, alpha=0.95),
    }

    include_point_count_panel = len(point_count_columns) > 0
    num_panels = 3 + int(include_point_count_panel)

    height_ratios = [1.2, 1.0, 1.0]
    if include_point_count_panel:
        height_ratios.append(0.8)

    share_x_axis = not include_point_count_panel or point_count_windowed

    axes = figure.subplots(
        num_panels,
        1,
        sharex=share_x_axis,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.atleast_1d(axes).tolist()

    ax_top = axes[0]
    ax_weighted = axes[1]
    ax_raw = axes[2]
    ax_point_count = axes[3] if include_point_count_panel else None

    ax_top_right = plot_top_loss_columns_with_dual_axis(
        ax_top,
        loss_dataframe,
        top_columns,
        style_map,
    )

    plot_linear_columns(
        ax_weighted,
        loss_dataframe,
        weighted_regularizer_columns,
        style_map,
    )

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

    if any(
            column_name in {"loss_rgb_mean", "loss_rgb_sum"}
            for column_name in top_columns
    ):
        ax_top.set_ylabel(f"Mean RGB loss ({loss_y_scale})")
    else:
        ax_top.set_ylabel(f"Mean image loss ({loss_y_scale})")

    if ax_top_right is not None:
        ax_top_right.set_ylabel(f"Mean total loss ({loss_y_scale})")

    latest_iteration = int(loss_dataframe["iteration"].iloc[-1])
    loss_row_count = len(loss_dataframe)
    point_count_row_count = len(point_count_dataframe)

    loss_average_coverage = ""

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
            loss_average_coverage = (
                f" | loss average={int(latest_averaged_camera_count)}"
                f"/{int(latest_expected_camera_count)} cameras"
            )

    point_count_history_label = (
        "windowed"
        if point_count_windowed
        else f"full history ({point_count_row_count} rows)"
    )

    ax_top.set_title(
        f"Live optimization metrics\n"
        f"{metrics_csv_path.parent.name} | loss rows={loss_row_count} | "
        f"point count={point_count_history_label} | "
        f"latest iteration={latest_iteration}{loss_average_coverage}"
    )

    ax_top.grid(True)
    set_combined_legend(ax_top, ax_top_right)

    ax_weighted.set_ylabel(f"Mean weighted regularizers ({loss_y_scale})")
    ax_weighted.grid(True)
    if weighted_regularizer_columns:
        ax_weighted.legend(loc="upper left")

    ax_raw.set_ylabel("Mean raw diagnostics")
    if raw_has_positive_values:
        ax_raw.set_yscale("log")
    ax_raw.grid(True)
    if raw_diagnostic_columns:
        ax_raw.legend(loc="upper left")

    if ax_point_count is not None:
        point_count_column = point_count_columns[0]
        point_values = dataframe_column_as_float_array(
            point_count_dataframe,
            point_count_column,
        )

        ax_point_count.step(
            point_count_dataframe["iteration"],
            point_values,
            where="post",
            label=point_count_column,
            **style_map.get(point_count_column, {}),
        )

        ax_point_count.set_ylabel("Point count")
        ax_point_count.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax_point_count.grid(True)
        ax_point_count.legend(loc="upper left")

    if ax_point_count is None:
        ax_raw.set_xlabel("Iteration")
    elif point_count_windowed:
        ax_point_count.set_xlabel("Iteration")
    else:
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_weighted.tick_params(axis="x", labelbottom=False)
        ax_raw.set_xlabel("Iteration (loss window)")
        ax_point_count.set_xlabel("Iteration (full point-count history)")

    figure.tight_layout()

    plotted_columns = (
        top_columns
        + weighted_regularizer_columns
        + raw_diagnostic_columns
        + point_count_columns
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

    return find_latest_run_dir(
        optimization_output_root=args.optimization_output_root.resolve(),
        metrics_name=args.metrics_name,
        index=args.index,
    )


def main() -> None:
    args = parse_args()

    if args.refresh_seconds <= 0.0:
        raise ValueError(
            f"--refresh-seconds must be positive, got: {args.refresh_seconds}"
        )

    plt.ion()
    plt.rcParams["figure.raise_window"] = False

    figure = plt.figure(figsize=(12, 7))
    previous_dataframe: pd.DataFrame | None = None
    previous_file_state: tuple[float, int] | None = None
    previous_run_dir: Path | None = None

    print("Starting live metrics viewer. Press Ctrl+C in the terminal to stop.")
    print(f"Refresh interval   : {args.refresh_seconds:.3f}s")
    print(f"Watch latest       : {args.watch_latest}")
    print(f"Save plot          : {args.save_plot}")
    print(f"Skip reset noise   : {args.skip}")
    print(f"Loss y-scale       : {args.loss_y_scale}")
    print(f"Point count window : {args.point_count_windowed}")

    try:
        while plt.fignum_exists(figure.number):
            run_dir = resolve_run_dir(args)
            metrics_csv_path = run_dir / args.metrics_name

            if previous_run_dir != run_dir:
                previous_run_dir = run_dir
                previous_file_state = None
                previous_dataframe = None

                print()
                print(f"Watching run       : {run_dir}")
                print(f"Metrics file       : {metrics_csv_path}")

            file_state = get_file_state(metrics_csv_path)

            if file_state is None:
                time.sleep(args.refresh_seconds)
                continue

            file_changed = file_state != previous_file_state

            if file_changed:
                dataframe = read_metrics_csv_safely(
                    metrics_csv_path=metrics_csv_path,
                    previous_dataframe=previous_dataframe,
                )

                if dataframe is not None:
                    previous_dataframe = dataframe
                    previous_file_state = file_state

                    plotted_columns = draw_metrics_figure(
                        figure=figure,
                        dataframe=dataframe,
                        metrics_csv_path=metrics_csv_path,
                        explicit_loss_column=args.loss_column,
                        plot_all_losses=args.plot_all_losses,
                        from_iteration=args.from_iteration,
                        last_iterations=args.iterations,
                        skip_opacity_reset_noise=args.skip,
                        point_count_windowed=args.point_count_windowed,
                        loss_y_scale=args.loss_y_scale,
                    )

                    figure.canvas.draw_idle()
                    figure.canvas.flush_events()

                    if args.save_plot:
                        output_png_path = run_dir / args.loss_output_name
                        figure.savefig(output_png_path, dpi=200)

                    latest_iteration = "unknown"

                    try:
                        prepared_dataframe = prepare_metrics_dataframe(
                            dataframe=dataframe,
                            from_iteration=args.from_iteration,
                            last_iterations=args.iterations,
                            skip_opacity_reset_noise=args.skip,
                        )

                        if not prepared_dataframe.empty:
                            latest_iteration = str(
                                int(prepared_dataframe["iteration"].iloc[-1])
                            )
                    except Exception:
                        pass

                    print(
                        f"\rUpdated live plot | iteration={latest_iteration} | "
                        f"columns={plotted_columns}",
                        end="",
                        flush=True,
                    )

            plt.pause(args.refresh_seconds)

            if not args.watch_latest and args.run_dir is None:
                args.run_dir = run_dir

    except KeyboardInterrupt:
        print()
        print("Stopped live metrics viewer.")


if __name__ == "__main__":
    main()
