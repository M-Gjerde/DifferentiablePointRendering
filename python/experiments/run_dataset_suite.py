#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = [
    "lego_10_pbdr",
    "lego_30_pbdr",
    "horse_10_pbdr",
    "horse_30_pbdr",
    "plant_10_pbdr",
    "plant_30_pbdr",
    "dragon_10_pbdr",
    "dragon_30_pbdr",
    "workbench_pbdr",
    "teapot_10_pbdr",
    "teapot_30_pbdr",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run main.py sequentially on the standard PBDR dataset suite."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("~/phd/datasets"),
        help="Root folder containing *_pbdr datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "OptimizationOutput" / "dataset_suite",
        help="Root folder where run folders are written.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=None,
        help=(
            "Dataset folder name under --dataset-root, or an explicit path. "
            "Repeatable. Defaults to the standard 10/30 view suite plus workbench."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun existing output folders. main.py will clear the folder first.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write each run output only to its log file instead of streaming it to the terminal.",
    )
    parser.add_argument(
        "main_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed through to main.py. Prefix them with --, e.g. -- --iterations 13000.",
    )
    args = parser.parse_args()

    main_args = list(args.main_args)
    if main_args and main_args[0] == "--":
        main_args = main_args[1:]

    return args, main_args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def run_name_from_dataset_path(dataset_path: Path) -> str:
    name = dataset_path.name
    return name[:-5] if name.endswith("_pbdr") else name


def resolve_dataset(dataset: str, dataset_root: Path) -> Path:
    dataset_path = Path(dataset).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = dataset_root / dataset_path
    return dataset_path.resolve()


def build_command(dataset_path: Path, output_dir: Path, main_args: list[str]) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "-s",
        str(dataset_path),
        "-o",
        str(output_dir),
        *main_args,
    ]


def run_command(command: list[str], log_path: Path, quiet: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        if quiet:
            process = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return int(process.returncode)

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=None,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return int(process.wait())


def main() -> None:
    args, main_args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    output_root = resolve_path(args.output_root)
    logs_dir = output_root / "_logs"

    dataset_names = args.datasets if args.datasets is not None else DEFAULT_DATASETS
    dataset_paths = [resolve_dataset(dataset, dataset_root) for dataset in dataset_names]
    jobs = [
        (run_name_from_dataset_path(dataset_path), dataset_path, output_root / run_name_from_dataset_path(dataset_path))
        for dataset_path in dataset_paths
    ]

    missing_datasets = [dataset_path for _, dataset_path, _ in jobs if not dataset_path.is_dir()]
    if missing_datasets:
        print("Missing dataset folders:")
        for dataset_path in missing_datasets:
            print(f"  {dataset_path}")
        raise SystemExit(1)

    failed_runs: list[str] = []

    for index, (run_name, dataset_path, output_dir) in enumerate(jobs, start=1):
        complete = (output_dir / "points_final.ply").is_file()
        if complete and not args.force:
            print(f"[skip] {run_name}: {output_dir / 'points_final.ply'} already exists")
            continue

        if output_dir.exists() and not args.force:
            raise RuntimeError(f"Output folder exists but is incomplete: {output_dir}. Use --force to rerun.")

        command = build_command(dataset_path, output_dir, main_args)
        log_path = logs_dir / f"{run_name}.log"

        print("\n" + "=" * 80)
        print(f"Run {index}/{len(jobs)}: {run_name}")
        print(f"Dataset : {dataset_path}")
        print(f"Output  : {output_dir}")
        print(f"Log     : {log_path}")
        print("Command : " + shlex.join(command))
        print("=" * 80)

        if args.dry_run:
            continue

        return_code = run_command(command, log_path=log_path, quiet=bool(args.quiet))
        if return_code != 0:
            failed_runs.append(run_name)
            print(f"[failed] {run_name}: exit code {return_code}. Log: {log_path}")
            if not args.continue_on_error:
                raise SystemExit(return_code)
        else:
            print(f"[done] {run_name}")

    if failed_runs:
        print("Failed runs:")
        for run_name in failed_runs:
            print(f"  {run_name}")
        raise SystemExit(1)

    print(f"Finished dataset suite. Output root: {output_root}")


if __name__ == "__main__":
    main()
