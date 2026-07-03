#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = Path.home() / "phd/datasets/horse_10_pbdr"
OUTPUT_ROOT = PROJECT_ROOT / "OptimizationOutput/ablations/horse_10"

ITERATIONS = 60_000
DEPTH_DISTORT_WEIGHT = 100.0
NORMAL_CONSISTENCY_WEIGHT = 0.005
OPACITY_PRIOR_WEIGHT = 0.05


def run_experiment(
    run_name: str,
    depth_distort_weight: float,
    normal_consistency_weight: float,
    opacity_prior_weight: float,
) -> None:
    output_directory = OUTPUT_ROOT / run_name
    output_directory.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "-s",
        str(DATASET_PATH),
        "-o",
        str(output_directory),
        "--iterations",
        str(ITERATIONS),
        "--depth-distort-weight",
        str(depth_distort_weight),
        "--normal-consistency-weight",
        str(normal_consistency_weight),
        "--visibility-weighted-opacity-weight",
        str(opacity_prior_weight),
    ]

    print("\n" + "=" * 80)
    print(f"Running:  {run_name}")
    print(f"Output:   {output_directory}")
    print("Command:", " ".join(command))
    print("=" * 80)

    log_path = output_directory / "train.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"Run '{run_name}' failed with exit code {process.returncode}. "
            f"See: {log_path}"
        )


def main() -> None:
    experiments = [
        (
            "full",
            DEPTH_DISTORT_WEIGHT,
            NORMAL_CONSISTENCY_WEIGHT,
            OPACITY_PRIOR_WEIGHT,
        ),
        (
            "no_depth_distortion",
            0.0,
            NORMAL_CONSISTENCY_WEIGHT,
            OPACITY_PRIOR_WEIGHT,
        ),
        (
            "no_normal_consistency",
            DEPTH_DISTORT_WEIGHT,
            0.0,
            OPACITY_PRIOR_WEIGHT,
        ),
        (
            "no_opacity_prior",
            DEPTH_DISTORT_WEIGHT,
            NORMAL_CONSISTENCY_WEIGHT,
            0.0,
        ),
    ]

    for experiment in experiments:
        run_experiment(*experiment)

    print("\nAll Horse ablation runs completed successfully.")


if __name__ == "__main__":
    main()