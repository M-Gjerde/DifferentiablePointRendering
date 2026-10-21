#!/usr/bin/env python3
"""Run scenes sequentially with current config.py defaults and standard outputs.

Examples:
    python python/train_all.py
    python python/train_all.py --scenes workbench horse
    python python/train_all.py --datasets horse,workbench --output-root OptimizationOutput/work_4
    python python/train_all.py --dry-run

Relative output roots are resolved against this script's directory. Each child
loads config.py when it starts; leave that file unchanged during a series.
Restaurant uses mesh extraction resolution 1024 instead of the config.py default.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

PYTHON_DIR = Path(__file__).resolve().parent
SCENES = ("horse", "workbench", "dragon", "teapot", "plant", "restaurant")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scenes", nargs="+", metavar="SCENE",
                           help="Space-separated scenes, in execution order (default: all scenes).")
    selection.add_argument("--datasets",
                           help="Comma-separated scenes, in execution order.")
    parser.add_argument("--dataset-root", type=Path, default=Path("~/phd/datasets"))
    parser.add_argument("--ground-truth-root", type=Path, default=Path("~/phd/models/gt"))
    parser.add_argument("--output-root", type=Path, default=Path("OptimizationOutput/work_3"))
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands only.")
    parser.add_argument("--overwrite", action="store_true", help="Allow main.py to replace existing scene outputs.")
    args = parser.parse_args()

    selected = args.scenes if args.scenes is not None else (
        args.datasets.split(",") if args.datasets is not None else SCENES
    )
    scenes = list(dict.fromkeys(s.strip() for s in selected if s.strip()))
    if not scenes or any(Path(s).name != s or s in {".", ".."} for s in scenes):
        parser.error("--scenes/--datasets must contain simple scene names")
    dataset_root = args.dataset_root.expanduser().resolve()
    gt_root = args.ground_truth_root.expanduser().resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = PYTHON_DIR / output_root
    output_root = output_root.resolve()
    for inputs in (dataset_root, gt_root):
        if output_root == inputs or inputs in output_root.parents or output_root in inputs.parents:
            parser.error("Output root must be separate from input roots")

    commands = []
    for scene in scenes:
        dataset = dataset_root / f"{scene}_pbdr"
        gt = gt_root / f"{scene}.ply"
        output = output_root / scene
        for path in (dataset / "scene.xml", dataset / "points.ply", gt):
            if not path.is_file() or path.stat().st_size == 0:
                parser.error(f"Missing or empty input: {path}")
        if not (dataset / "images").is_dir():
            parser.error(f"Missing images directory: {dataset / 'images'}")
        if output.is_symlink() or (output.exists() and not args.overwrite):
            parser.error(f"Output already exists: {output}; choose another root or use --overwrite")
        command = [sys.executable, "-u", str(PYTHON_DIR / "main.py"),
                   "-s", str(dataset), "-o", str(output), "--ground-truth", str(gt)]
        if scene == "restaurant":
            command.extend(["--mesh-extraction-mesh-res", "1024"])
        commands.append(command)

    for scene, command in zip(scenes, commands):
        print(f"\n{scene}: {shlex.join(command)}", flush=True)
        if not args.dry_run:
            # Inherit the terminal, native-module environment, and metrics-window settings.
            result = subprocess.run(command, cwd=PYTHON_DIR)
            if result.returncode:
                print(f"{scene} failed (exit {result.returncode}); stopping the series.", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
