#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
CONFIG_PATH = PROJECT_ROOT / "config.py"
DEFAULT_DATASETS = ("dragon", "horse", "lego", "plant", "teapot")
RUNTIME_DEFINITION = (
    "Complete main.py subprocess wall-clock seconds, including interpreter startup, "
    "renderer setup, optimization, configured evaluation, and final saving."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train this method sequentially on the five object-level datasets using "
            "the unchanged defaults in config.py."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("~/phd/datasets"),
        help="Folder containing <dataset>_<view-count>_pbdr directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("OptimizationOutput/object_suite/run1"),
        help="Suite output folder (default: OptimizationOutput/object_suite/run1).",
    )
    parser.add_argument("--view-count", type=int, default=10)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated object names.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing per-scene output directories to be replaced by main.py.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the remaining scenes if one training process fails.",
    )
    return parser.parse_args()


def selected_datasets(value: str) -> list[str]:
    datasets = list(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))
    if not datasets:
        raise ValueError("--datasets must contain at least one dataset name")
    return datasets


def config_digest() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def final_metrics_statistics(run_dir: Path) -> tuple[float | None, int | None]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        return None, None

    runtime_seconds: float | None = None
    point_count: int | None = None
    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        for row in csv.DictReader(metrics_file):
            row_runtime = finite_number(row.get("total_time_sec"))
            row_point_count = finite_number(row.get("num_points"))
            if row_runtime is not None:
                runtime_seconds = row_runtime
            if row_point_count is not None:
                point_count = int(row_point_count)
    return runtime_seconds, point_count


def point_count_from_ply(ply_path: Path) -> int | None:
    if not ply_path.is_file():
        return None
    with ply_path.open("rb") as ply_file:
        header = ply_file.read(64 * 1024).decode("ascii", errors="ignore")
    for line in header.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[:2] == ["element", "vertex"]:
            try:
                return int(fields[2])
            except ValueError:
                return None
        if line.strip() == "end_header":
            break
    return None


def read_final_point_count(run_dir: Path) -> int | None:
    _, point_count = final_metrics_statistics(run_dir)
    if point_count is not None:
        return point_count
    return point_count_from_ply(run_dir / "points_final.ply")


def suite_averages(scenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    completed = [scene for scene in scenes.values() if scene.get("status") == "completed"]
    runtimes = [
        value
        for scene in completed
        if (value := finite_number(scene.get("runtime_seconds"))) is not None
    ]
    point_counts = [
        value
        for scene in completed
        if (value := finite_number(scene.get("point_count"))) is not None
    ]
    return {
        "scene_count": len(completed),
        "runtime_scene_count": len(runtimes),
        "point_count_scene_count": len(point_counts),
        "total_runtime_seconds": sum(runtimes) if runtimes else None,
        "average_runtime_seconds": sum(runtimes) / len(runtimes) if runtimes else None,
        "average_point_count": sum(point_counts) / len(point_counts) if point_counts else None,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def write_summaries(
    output_root: Path,
    scenes: dict[str, dict[str, Any]],
    expected_config_digest: str,
) -> None:
    averages = suite_averages(scenes)
    write_json_atomic(
        output_root / "summary.json",
        {
            "method": "DifferentiablePointRendering",
            "runtime_definition": RUNTIME_DEFINITION,
            "config_sha256": expected_config_digest,
            "expected_scene_count": len(scenes),
            "scenes": scenes,
            "averages": averages,
        },
    )

    summary_path = output_root / "summary.csv"
    temporary_path = summary_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            ["scene", "dataset", "views", "status", "runtime_seconds", "point_count", "output_dir"]
        )
        for scene_name, scene in scenes.items():
            writer.writerow(
                [
                    scene_name,
                    scene["dataset"],
                    scene["views"],
                    scene["status"],
                    scene.get("runtime_seconds") if scene.get("runtime_seconds") is not None else "",
                    scene.get("point_count") if scene.get("point_count") is not None else "",
                    scene["output_dir"],
                ]
            )
        writer.writerow(
            [
                "AVERAGE",
                "",
                "",
                "aggregate",
                averages["average_runtime_seconds"] if averages["average_runtime_seconds"] is not None else "",
                averages["average_point_count"] if averages["average_point_count"] is not None else "",
                "",
            ]
        )
        writer.writerow(
            [
                "TOTAL",
                "",
                "",
                "aggregate",
                averages["total_runtime_seconds"] if averages["total_runtime_seconds"] is not None else "",
                "",
                "",
            ]
        )
    temporary_path.replace(summary_path)


def write_scene_statistics(scene: dict[str, Any]) -> None:
    run_dir = Path(scene["output_dir"])
    if not run_dir.is_dir():
        return
    write_json_atomic(
        run_dir / "training_stats.json",
        {
            "runtime_definition": RUNTIME_DEFINITION,
            "status": scene["status"],
            "runtime_seconds": scene.get("runtime_seconds"),
            "point_count": scene.get("point_count"),
        },
    )


def ensure_scene_output_is_safe(run_dir: Path, overwrite: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to replace existing scene output: {run_dir}. "
            "Choose another --output-root or pass --overwrite."
        )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = selected_datasets(args.datasets)
    expected_config_digest = config_digest()
    scenes: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        scene_name = f"{dataset}_{args.view_count}"
        dataset_path = dataset_root / f"{dataset}_{args.view_count}_pbdr"
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"Dataset directory does not exist: {dataset_path}")
        run_dir = output_root / scene_name
        ensure_scene_output_is_safe(run_dir, args.overwrite)
        scenes[scene_name] = {
            "dataset": dataset,
            "views": args.view_count,
            "status": "pending",
            "runtime_seconds": None,
            "point_count": None,
            "output_dir": str(run_dir),
        }

    config_snapshot_path = output_root / "config.snapshot.py"
    if config_snapshot_path.exists() and not args.overwrite:
        existing_snapshot_digest = hashlib.sha256(config_snapshot_path.read_bytes()).hexdigest()
        if existing_snapshot_digest != expected_config_digest:
            raise RuntimeError(
                f"Existing suite snapshot differs from config.py: {config_snapshot_path}. "
                "Choose another --output-root or pass --overwrite."
            )
    shutil.copy2(CONFIG_PATH, config_snapshot_path)
    write_summaries(output_root, scenes, expected_config_digest)

    for scene_name, scene in scenes.items():
        if config_digest() != expected_config_digest:
            raise RuntimeError(
                "config.py changed after this suite started; refusing to mix configurations across scenes."
            )

        dataset_path = dataset_root / f"{scene['dataset']}_{scene['views']}_pbdr"
        run_dir = Path(scene["output_dir"])
        command = [
            sys.executable,
            str(MAIN_SCRIPT),
            "-s",
            str(dataset_path),
            "-o",
            str(run_dir),
        ]
        print(f"\n{'=' * 80}")
        print(f"Training: {scene_name}")
        print("Command:", " ".join(command))
        print(f"{'=' * 80}\n", flush=True)

        scene["status"] = "running"
        write_summaries(output_root, scenes, expected_config_digest)
        started_at = time.perf_counter()
        return_code: int | None = None
        try:
            completed_process = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            return_code = completed_process.returncode
            scene["status"] = "completed" if return_code == 0 else "failed"
        except KeyboardInterrupt:
            scene["status"] = "interrupted"
            raise
        finally:
            scene["runtime_seconds"] = time.perf_counter() - started_at
            scene["point_count"] = read_final_point_count(run_dir)
            scene["return_code"] = return_code
            if scene["status"] == "completed" and scene["point_count"] is None:
                scene["status"] = "missing_statistics"
            write_scene_statistics(scene)
            write_summaries(output_root, scenes, expected_config_digest)

        if scene["status"] != "completed":
            message = (
                f"Training did not complete cleanly for {scene_name}: "
                f"status={scene['status']}, exit_code={return_code}."
            )
            if not args.continue_on_error:
                raise RuntimeError(message)
            print(message, flush=True)

    averages = suite_averages(scenes)
    print("\nObject-level suite complete.")
    print(f"Summary JSON: {output_root / 'summary.json'}")
    print(f"Summary CSV:  {output_root / 'summary.csv'}")
    if averages["average_runtime_seconds"] is not None:
        print(f"Average runtime: {averages['average_runtime_seconds']:.2f} s")
    if averages["average_point_count"] is not None:
        print(f"Average points:  {averages['average_point_count']:.0f}")


if __name__ == "__main__":
    main()
