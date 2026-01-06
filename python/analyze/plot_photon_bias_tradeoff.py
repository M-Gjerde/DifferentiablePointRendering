#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


"""
RUN WITH:

python3 plot_photon_tradeoff.py \
  --root ~/CLionProjects/DifferentiablePointRendering/Assets/OptimizationOutput \
  --pattern "photon_map_teapot_*" \
  --metrics_name metrics.csv \
  --log_x \
  --output_png figures/photon_tradeoff.png \
  --output_summary_csv figures/photon_tradeoff_summary.csv


"""

@dataclass(frozen=True)
class RunRow:
    run_dir: Path
    photon_launch_count: int
    final_quality: float
    total_time_sec: float
    mean_iteration_time_sec: float
    num_iterations: int


PHOTON_COUNT_REGEX = re.compile(r"photon_map_teapot_(\d+(?:\.\d+)?e\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover photon_map_teapot_* runs, read metrics.csv, and plot final quality vs runtime."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing photon_map_teapot_* folders.",
    )
    parser.add_argument(
        "--pattern",
        default="photon_map_teapot_*",
        help="Glob for run directories.",
    )
    parser.add_argument(
        "--metrics_name",
        default="metrics.csv",
        help="Metrics filename inside each run directory.",
    )

    parser.add_argument(
        "--photon_count_column",
        default="photonLaunchCount",
        help="CSV column containing photonLaunchCount (N).",
    )
    parser.add_argument(
        "--quality_column",
        default="loss_l2_window_mean",
        help="CSV column to treat as quality metric (lower is better).",
    )
    parser.add_argument(
        "--time_column",
        default="total_time",
        help="CSV column for total wall time (seconds).",
    )
    parser.add_argument(
        "--iteration_time_column",
        default="iteration_time_sec",
        help="CSV column for per-iteration time (seconds).",
    )
    parser.add_argument(
        "--final_row",
        choices=["last", "best"],
        default="last",
        help="Pick last row quality or best (min) quality in the run.",
    )

    parser.add_argument(
        "--log_x",
        action="store_true",
        help="Use log x-axis (photonLaunchCount).",
    )
    parser.add_argument(
        "--log_quality",
        action="store_true",
        help="Use log y-axis for quality metric.",
    )

    parser.add_argument(
        "--output_png",
        type=Path,
        default=Path("photon_tradeoff.png"),
        help="Output plot path.",
    )
    parser.add_argument(
        "--output_summary_csv",
        type=Path,
        default=Path("photon_tradeoff_summary.csv"),
        help="Output summary CSV path.",
    )
    return parser.parse_args()


def parse_photon_count_from_dirname(run_dir: Path) -> Optional[int]:
    match = PHOTON_COUNT_REGEX.search(run_dir.name)
    if match is None:
        return None
    return int(float(match.group(1)))


def safe_last_numeric(df: pd.DataFrame, column_name: str) -> float:
    if column_name not in df.columns:
        raise KeyError(f"Missing column '{column_name}'. Available: {list(df.columns)}")
    series = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if series.empty:
        raise RuntimeError(f"Column '{column_name}' contains no numeric values.")
    return float(series.iloc[-1])


def mean_numeric_or_nan(df: pd.DataFrame, column_name: str) -> float:
    if column_name not in df.columns:
        return float("nan")
    series = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.mean())

def parse_photon_count_from_dirname(run_dir: Path) -> Optional[int]:
    match = PHOTON_COUNT_REGEX.search(run_dir.name)
    if match is None:
        return None
    return int(float(match.group(1)))  # "1e5" -> 100000

def pick_quality_value(df: pd.DataFrame, quality_column: str, final_row: str) -> float:
    if quality_column not in df.columns:
        raise KeyError(f"Missing column '{quality_column}'. Available: {list(df.columns)}")
    series = pd.to_numeric(df[quality_column], errors="coerce").dropna()
    if series.empty:
        raise RuntimeError(f"Column '{quality_column}' contains no numeric values.")
    if final_row == "last":
        return float(series.iloc[-1])
    return float(series.min())


def infer_num_iterations(df: pd.DataFrame) -> int:
    if "iteration" in df.columns:
        series = pd.to_numeric(df["iteration"], errors="coerce").dropna()
        if not series.empty:
            return int(series.max())
    return int(len(df))


def discover_runs(root: Path, pattern: str, metrics_name: str) -> List[Tuple[Path, Path]]:
    run_pairs: List[Tuple[Path, Path]] = []
    for run_dir in sorted(root.glob(pattern)):
        if not run_dir.is_dir():
            continue
        if parse_photon_count_from_dirname(run_dir) is None:
            continue
        metrics_path = run_dir / metrics_name
        if metrics_path.is_file():
            run_pairs.append((run_dir, metrics_path))
    return run_pairs


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    run_pairs = discover_runs(root, args.pattern, args.metrics_name)
    if not run_pairs:
        raise RuntimeError(
            f"No runs found under '{root}' matching '{args.pattern}' that contain '{args.metrics_name}'."
        )

    rows: List[RunRow] = []
    for run_dir, metrics_path in run_pairs:
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue

        photon_launch_count: Optional[int] = None

        # Try CSV column first (if it exists), otherwise parse from folder name.
        if args.photon_count_column in df.columns:
            photon_launch_count = int(safe_last_numeric(df, args.photon_count_column))

        if photon_launch_count is None:
            photon_launch_count = parse_photon_count_from_dirname(run_dir)

        if photon_launch_count is None:
            raise RuntimeError(
                f"Could not determine photon launch count for run '{run_dir.name}'. "
                f"Expected a CSV column '{args.photon_count_column}' or a dirname like 'photon_map_teapot_1e5_...'."
            )

        final_quality = pick_quality_value(df, args.quality_column, args.final_row)
        total_time_sec = safe_last_numeric(df, args.time_column)
        mean_iteration_time_sec = mean_numeric_or_nan(df, args.iteration_time_column)
        num_iterations = infer_num_iterations(df)

        rows.append(
            RunRow(
                run_dir=run_dir,
                photon_launch_count=photon_launch_count,
                final_quality=final_quality,
                total_time_sec=total_time_sec,
                mean_iteration_time_sec=mean_iteration_time_sec,
                num_iterations=num_iterations,
            )
        )

    if not rows:
        raise RuntimeError("Found candidate runs, but none produced usable rows (empty CSVs or missing columns).")

    summary_df = pd.DataFrame(
        [
            {
                "run_dir": str(r.run_dir),
                "photonLaunchCount": r.photon_launch_count,
                "final_quality": r.final_quality,
                "total_time_sec": r.total_time_sec,
                "mean_iteration_time_sec": r.mean_iteration_time_sec,
                "num_iterations": r.num_iterations,
            }
            for r in rows
        ]
    ).sort_values("photonLaunchCount", ascending=True)

    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_summary_csv, index=False)

    x = summary_df["photonLaunchCount"].to_numpy()
    quality = summary_df["final_quality"].to_numpy()
    total_time = summary_df["total_time_sec"].to_numpy()

    fig, ax_quality = plt.subplots()

    ax_quality.plot(x, quality, marker="o")
    ax_quality.set_xlabel("Photon launch count (N)")
    ax_quality.set_ylabel(f"Final {args.quality_column}  ↓")
    ax_quality.grid(True, which="both", linestyle="--", linewidth=0.5)

    if args.log_x:
        ax_quality.set_xscale("log")
    if args.log_quality:
        ax_quality.set_yscale("log")

    ax_time = ax_quality.twinx()
    ax_time.plot(x, total_time, marker="s")
    ax_time.set_ylabel(f"Total time (s) ({args.time_column})  ↑")

    ax_quality.legend(
        [ax_quality.lines[0], ax_time.lines[0]],
        [f"Final {args.quality_column}", f"{args.time_column}"],
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(args.output_png, dpi=200)

    print(f"Wrote plot:    {args.output_png}")
    print(f"Wrote summary: {args.output_summary_csv}")
    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df)


if __name__ == "__main__":
    main()
