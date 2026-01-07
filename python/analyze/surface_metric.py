#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class DistanceStats:
    count: int
    mean: float
    rms: float
    median: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float


def load_distances_csv(csv_path: Path) -> np.ndarray:
    """
    Loads a CSV created by:
      np.savetxt(..., delimiter=",", header="distance", comments="")
    into a 1D float array.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Robust to trailing spaces/newlines.
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    if data.size == 0:
        raise RuntimeError(f"No data rows in CSV: {csv_path}")

    # genfromtxt returns a structured array when names=True
    if "distance" not in data.dtype.names:
        raise RuntimeError(f"Expected column 'distance' in {csv_path}, got {data.dtype.names}")

    distances = np.asarray(data["distance"], dtype=np.float64).reshape(-1)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        raise RuntimeError(f"All distances are non-finite in CSV: {csv_path}")

    return distances


def compute_stats(distances: np.ndarray) -> DistanceStats:
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        raise RuntimeError("compute_stats(): empty / non-finite distances")

    count = int(distances.size)
    mean = float(np.mean(distances))
    rms = float(np.sqrt(np.mean(distances * distances)))
    median = float(np.median(distances))
    p90 = float(np.percentile(distances, 90))
    p95 = float(np.percentile(distances, 95))
    p99 = float(np.percentile(distances, 99))
    dmin = float(np.min(distances))
    dmax = float(np.max(distances))

    return DistanceStats(
        count=count,
        mean=mean,
        rms=rms,
        median=median,
        p90=p90,
        p95=p95,
        p99=p99,
        min=dmin,
        max=dmax,
    )


def format_float(value: float) -> str:
    return f"{value:.6e}"


def write_summary_table_csv(output_csv: Path, rows: List[Dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "count",
        "mean",
        "rms",
        "median",
        "p90",
        "p95",
        "p99",
        "min",
        "max",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_markdown_table(rows: List[Dict[str, str]]) -> None:
    headers = [
        "Method",
        "Count",
        "Mean",
        "RMS",
        "Median",
        "P90",
        "P95",
        "P99",
        "Min",
        "Max",
    ]
    keys = ["method", "count", "mean", "rms", "median", "p90", "p95", "p99", "min", "max"]

    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize two distance CSVs (ours vs 2DGS) into a single table."
    )
    parser.add_argument("--ours_csv", type=Path, required=True, help="CSV with header 'distance' for our method")
    parser.add_argument("--dgs_csv", type=Path, required=True, help="CSV with header 'distance' for 2DGS")
    parser.add_argument("--ours_name", type=str, default="Ours", help="Label for our method row")
    parser.add_argument("--dgs_name", type=str, default="2DGS", help="Label for 2DGS row")
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Optional: write the summary table as a CSV",
    )
    args = parser.parse_args()

    ours_distances = load_distances_csv(args.ours_csv)
    dgs_distances = load_distances_csv(args.dgs_csv)

    ours_stats = compute_stats(ours_distances)
    dgs_stats = compute_stats(dgs_distances)

    def stats_to_row(method_name: str, stats: DistanceStats) -> Dict[str, str]:
        return {
            "method": method_name,
            "count": str(stats.count),
            "mean": format_float(stats.mean),
            "rms": format_float(stats.rms),
            "median": format_float(stats.median),
            "p90": format_float(stats.p90),
            "p95": format_float(stats.p95),
            "p99": format_float(stats.p99),
            "min": format_float(stats.min),
            "max": format_float(stats.max),
        }

    rows = [
        stats_to_row(args.dgs_name, dgs_stats),
        stats_to_row(args.ours_name, ours_stats),
    ]

    print_markdown_table(rows)

    if args.out_csv is not None:
        write_summary_table_csv(args.out_csv, rows)
        print(f"\nWrote summary table CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
