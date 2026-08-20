#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_CLI_FLAGS: dict[str, str | tuple[str, str]] = {
    "checkpoint": "--checkpoint",
    "iterations": "--iterations",
    "log_interval": "--log-interval",
    "save_interval": "--save-interval",
    "save_ply_files_interval": "--save-ply-files-interval",
    "mesh_extraction_iterations": "--mesh-extraction-iterations",
    "mesh_extraction_depth_key": "--mesh-extraction-depth-key",
    "mesh_extraction_mesh_res": "--mesh-extraction-mesh-res",
    "mesh_extraction_num_cluster": "--mesh-extraction-num-cluster",
    "save_final_mesh": ("--save-final-mesh", "--no-save-final-mesh"),
    "learning_rate": "--lr",
    "learning_rate_position": "--lr-pos",
    "learning_rate_rotation": "--lr-rot",
    "learning_rate_scale": "--lr-scale",
    "learning_rate_albedo": "--lr-albedo",
    "learning_rate_opacity": "--lr-opacity",
    "learning_rate_beta": "--lr-beta",
    "use_global_lr_schedule": ("--global-lr-schedule", "--no-global-lr-schedule"),
    "global_lr_scale_init": "--global-lr-scale-init",
    "global_lr_scale_final": "--global-lr-scale-final",
    "global_lr_start_iteration": "--global-lr-start-iteration",
    "global_lr_max_steps": "--global-lr-max-steps",
    "normal_consistency_weight": "--normal-consistency-weight",
    "normal_from_depth_use_mean_depth": (
        "--normal-from-depth-use-mean-depth",
        "--no-normal-from-depth-use-mean-depth",
    ),
    "depth_distort_weight": "--depth-distort-weight",
    "depth_distort_start_iteration": "--depth-distort-start-iteration",
    "densification_interval": "--densification-interval",
    "prune_interval": "--prune-interval",
    "densify_after": "--densify-after",
    "prune_after": "--prune-after",
    "densification_verbose": ("--densification-verbose", "--no-densification-verbose"),
    "densification_grad_quantile": "--densification-grad-quantile",
    "densification_grad_abs_min": "--densification-grad-abs-min",
    "densification_grad_abs_min_final": "--densification-grad-abs-min-final",
    "densification_grad_abs_min_decay_start_iteration": "--densification-grad-abs-min-decay-start-iteration",
    "densification_grad_abs_min_decay_end_iteration": "--densification-grad-abs-min-decay-end-iteration",
    "densification_grad_abs_min_iter_start": "--densification-grad-abs-min-iter-start",
    "densification_grad_abs_min_iter_end": "--densification-grad-abs-min-iter-end",
    "densification_stats_skip_interval_start": (
        "--densification-stats-skip-interval-start",
        "--no-densification-stats-skip-interval-start",
    ),
    "densify_bsdf_floor": "--densify-bsdf-floor",
    "densify_bsdf_gamma": "--densify-bsdf-gamma",
    "opacity_prune_threshold": "--opacity-prune-threshold",
    "max_prune_fraction": "--max-prune-fraction",
    "reset_opacity_interval": "--reset-opacity-interval",
    "reset_opacity_value": "--reset-opacity-value",
    "rebuild_bvh_interval": "--rebuild-bvh-interval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweeps and evaluate the produced run directories."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=PROJECT_ROOT / "experiments/example_teapot_sweep.json",
        help="Sweep JSON spec.",
    )
    parser.add_argument("--search-mode", choices=["random", "grid"], default="random")
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional GT PLY override for offline geometry plots. Defaults to ground_truth in the sweep JSON.",
    )
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force", action="store_true", help="Allow main.py to clear and rerun existing trial dirs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_spec(spec_path: Path) -> dict[str, Any]:
    spec_path = resolve_path(spec_path)
    with spec_path.open("r", encoding="utf-8") as spec_file:
        return json.load(spec_file)


def normalize_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def grid_trials(search_space: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(search_space.keys())
    value_lists = [normalize_values(search_space[key]) for key in keys]
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def make_trials(
    base_args: dict[str, Any],
    search_space: dict[str, Any],
    explicit_trials: list[dict[str, Any]],
    search_mode: str,
    max_trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    if explicit_trials:
        trials = [dict(base_args, **trial) for trial in explicit_trials]
        return trials[:max_trials] if max_trials > 0 else trials

    candidates = [dict(base_args, **trial) for trial in grid_trials(search_space)]
    if search_mode == "grid":
        return candidates[:max_trials] if max_trials > 0 else candidates

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:max_trials] if max_trials > 0 else candidates


def parameter_digest(parameters: dict[str, Any]) -> str:
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:8]


def slug_value(value: Any) -> str:
    text = str(value).lower()
    text = text.replace("-", "m").replace(".", "p").replace("+", "")
    text = text.replace("e", "e")
    return "".join(char if char.isalnum() else "_" for char in text)


def trial_run_name(index: int, parameters: dict[str, Any]) -> str:
    parts = [
        f"t{index:03d}",
        f"di{slug_value(parameters.get('densification_interval', 'x'))}",
        f"thr{slug_value(parameters.get('densification_grad_abs_min', 'x'))}",
        f"n{slug_value(parameters.get('normal_consistency_weight', 'x'))}",
        f"d{slug_value(parameters.get('depth_distort_weight', 'x'))}",
        f"glr{slug_value(parameters.get('use_global_lr_schedule', 'default'))}",
        parameter_digest(parameters),
    ]
    return "_".join(parts)[:180]


def cli_args_for_parameter(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    if name not in CONFIG_CLI_FLAGS:
        raise KeyError(f"No CLI flag mapping for sweep parameter: {name}")

    flag = CONFIG_CLI_FLAGS[name]
    if isinstance(value, bool):
        if not isinstance(flag, tuple):
            raise TypeError(f"Boolean parameter {name} needs true/false CLI flags.")
        return [flag[0] if value else flag[1]]

    if isinstance(flag, tuple):
        raise TypeError(f"Parameter {name} uses boolean CLI flags but value is {value!r}.")

    if isinstance(value, (list, tuple)):
        return [flag, *[str(item) for item in value]]

    return [flag, str(value)]


def build_train_command(dataset_path: Path | None, output_dir: Path, parameters: dict[str, Any]) -> list[str]:
    command = [sys.executable, str(PROJECT_ROOT / "main.py")]
    if dataset_path is not None:
        command.extend(["-s", str(dataset_path)])
    command.extend(["-o", str(output_dir)])

    for name in sorted(parameters.keys()):
        command.extend(cli_args_for_parameter(name, parameters[name]))

    return command


def write_trial_config(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "trial_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=2)


def run_trial(
    index: int,
    run_name: str,
    run_dir: Path,
    command: list[str],
    parameters: dict[str, Any],
    logs_dir: Path,
    force: bool,
    dry_run: bool,
) -> bool:
    complete = (run_dir / "points_final.ply").is_file()
    if complete and not force:
        print(f"[skip] {run_name}: points_final.ply already exists")
        return True

    if run_dir.exists() and not force and not complete:
        raise RuntimeError(f"Run directory exists but is incomplete: {run_dir}. Use --force to rerun.")

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_name}.log"

    print("\n" + "=" * 80)
    print(f"Trial {index}: {run_name}")
    print(f"Output   : {run_dir}")
    print("Command  : " + " ".join(command))
    print("=" * 80)

    if dry_run:
        return True

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if run_dir.is_dir():
        shutil.copy2(log_path, run_dir / "train.log")
        write_trial_config(
            run_dir,
            {
                "index": index,
                "run_name": run_name,
                "parameters": parameters,
                "command": command,
                "log_path": str(log_path),
            },
        )

    if process.returncode != 0:
        print(f"[failed] {run_name}: exit code {process.returncode}. Log: {log_path}")
        return False

    print(f"[done] {run_name}")
    return True


def run_evaluation(output_root: Path, args: argparse.Namespace, ground_truth: Path | None) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/evaluate_runs.py"),
        "--run-root",
        str(output_root),
        "--output-dir",
        str(output_root / "evaluation"),
        "--samples",
        str(args.samples),
        "--device",
        args.device,
        "--full",
    ]
    if ground_truth is not None:
        command.extend(["--ground-truth", str(ground_truth)])

    print("\nRunning evaluation:")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    spec = load_spec(args.spec)

    dataset_path = spec.get("dataset_path")
    resolved_dataset_path = resolve_path(Path(dataset_path)) if dataset_path else None
    ground_truth_value = args.ground_truth if args.ground_truth is not None else spec.get("ground_truth")
    resolved_ground_truth = resolve_path(Path(ground_truth_value)) if ground_truth_value else None
    output_root = resolve_path(Path(spec["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)

    base_args = dict(spec.get("base_args", {}))
    search_space = dict(spec.get("search_space", {}))
    explicit_trials = list(spec.get("trials", []))

    trials = make_trials(
        base_args=base_args,
        search_space=search_space,
        explicit_trials=explicit_trials,
        search_mode=args.search_mode,
        max_trials=args.max_trials,
        seed=args.seed,
    )

    if not trials:
        raise SystemExit("Sweep spec produced zero trials.")

    logs_dir = output_root / "_logs"
    failed_runs: list[str] = []

    for index, parameters in enumerate(trials):
        run_name = trial_run_name(index, parameters)
        run_dir = output_root / run_name
        command = build_train_command(resolved_dataset_path, run_dir, parameters)

        try:
            succeeded = run_trial(
                index=index,
                run_name=run_name,
                run_dir=run_dir,
                command=command,
                parameters=parameters,
                logs_dir=logs_dir,
                force=args.force,
                dry_run=args.dry_run,
            )
        except Exception:
            if not args.continue_on_error:
                raise
            succeeded = False

        if not succeeded:
            failed_runs.append(run_name)
            if not args.continue_on_error:
                raise RuntimeError(f"Trial failed: {run_name}")

    if not args.dry_run and not args.skip_evaluation:
        run_evaluation(output_root, args, resolved_ground_truth)

    if failed_runs:
        print("Failed runs:")
        for run_name in failed_runs:
            print(f"  {run_name}")


if __name__ == "__main__":
    main()
