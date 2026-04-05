# batch_fd_check.py
#
# Batch finite-difference vs analytic gradient checker.
#
# Features:
# - Reads per-case sweep config from tests.json
# - Per-case required fields:
#     scene, camera, parameter, index, min, max,
#     forward_passes, bounces, adjoint_passes, adjoint_bounces
# - Excludes last CSV row from scoring by default
# - Keeps excluded rows visible in printed output
# - Ignores tiny gradients below --grad_floor
# - Pass condition per scored row:
#       rel_err <= rel_threshold
# - Overall case pass:
#       fail_frac <= fail_frac_threshold
#
# Example tests.json:
# {
#   "common_args": ["--iterations", "10", "--fd_epsilon", "5e-3", "--ply", "pointcloud", "--seed", "42"],
#   "cases": [
#     {
#       "scene": "transmit",
#       "camera": "camera1",
#       "parameter": "translation_z",
#       "index": 0,
#       "min": -0.05,
#       "max": 0.05,
#       "forward_passes": 75,
#       "bounces": 1,
#       "adjoint_passes": 64,
#       "adjoint_bounces": 2
#     }
#   ]
# }

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"


def color(text: str, ansi_color: str, enable: bool) -> str:
    return f"{ansi_color}{text}{ANSI_RESET}" if enable else text


def safe_rel_err(value_a: float, value_b: float, eps: float) -> float:
    denominator = max(eps, abs(value_a) + abs(value_b))
    return abs(value_a - value_b) / denominator


def load_csv(run_dir: Path, camera: str, parameter: str) -> pd.DataFrame:
    csv_path = run_dir / f"{camera}_{parameter}_sweep.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data_frame = pd.read_csv(csv_path)
    required_columns = {"iter", parameter, "analytic_grad", "fd_grad", "fd_kind"}
    if not required_columns.issubset(data_frame.columns):
        raise RuntimeError(
            f"CSV missing columns {required_columns}, has: {list(data_frame.columns)}"
        )
    return data_frame


def resolve_run_dir(workspace_dir: Path, scene: str, parameter: str, index: int) -> Path:
    return workspace_dir / "Output" / scene / parameter / str(index) / "0"


def run_render_target(
    python_exe: str,
    render_target_script: Path,
    scene: str,
    parameter: str,
) -> int:
    command = [
        python_exe,
        str(render_target_script),
        "--parameter",
        parameter,
        "--scene",
        scene,
    ]
    print(color("TARGET:", ANSI_BOLD, True), " ".join(command))
    process = subprocess.run(command)
    return int(process.returncode)


def run_one(
    python_exe: str,
    script_path: Path,
    scene: str,
    camera: str,
    parameter: str,
    point_index: int,
    sweep_min: float,
    sweep_max: float,
    forward_passes: int,
    bounces: int,
    adjoint_passes: int,
    adjoint_bounces: int,
    common_args: list[str],
    extra_args: list[str],
) -> int:
    command = [
        python_exe,
        str(script_path),
        "--parameter",
        parameter,
        "--scene",
        scene,
        "--camera",
        camera,
        "--index",
        str(point_index),
        "--min",
        str(sweep_min),
        "--max",
        str(sweep_max),
        "--forward_passes",
        str(forward_passes),
        "--bounces",
        str(bounces),
        "--adjoint_passes",
        str(adjoint_passes),
        "--adjoint_bounces",
        str(adjoint_bounces),
    ]
    command += common_args
    command += extra_args

    print(color("RUN:", ANSI_BOLD, True), " ".join(command))
    process = subprocess.run(command)
    return int(process.returncode)


def filter_rows(
    data_frame: pd.DataFrame,
    parameter: str,
    tail: int,
    ignore_boundaries: bool,
    exclude_last_row: bool,
) -> pd.DataFrame:
    filtered_data_frame = data_frame

    if exclude_last_row and len(filtered_data_frame) >= 2:
        filtered_data_frame = filtered_data_frame.iloc[:-1]

    filtered_data_frame = filtered_data_frame.tail(tail) if tail > 0 else filtered_data_frame

    if ignore_boundaries and parameter == "opacity":
        parameter_values = filtered_data_frame[parameter].to_numpy(dtype=np.float64)
        valid_mask = (parameter_values > 1e-6) & (parameter_values < 1.0 - 1e-6)
        filtered_data_frame = filtered_data_frame.loc[valid_mask]

    return filtered_data_frame


def print_all_rows(
    data_frame: pd.DataFrame,
    parameter: str,
    scored_data_frame: pd.DataFrame,
    rel_eps: float,
    rel_threshold: float,
    grad_floor: float,
    enable_color: bool,
) -> None:
    scored_iterations = set(scored_data_frame["iter"].astype(int).tolist())

    print("\nAll rows (scored + excluded):")
    for _, row in data_frame.iterrows():
        iteration = int(row["iter"])
        parameter_value = float(row[parameter])
        analytic_grad = float(row["analytic_grad"])
        fd_grad = float(row["fd_grad"])
        fd_kind = int(row["fd_kind"])

        rel_err = safe_rel_err(analytic_grad, fd_grad, rel_eps)
        max_grad_magnitude = max(abs(analytic_grad), abs(fd_grad))
        is_tiny = max_grad_magnitude < grad_floor
        is_scored = iteration in scored_iterations

        if is_tiny:
            status = color("SKIP", ANSI_YELLOW, enable_color)
        elif is_scored and rel_err <= rel_threshold:
            status = color("PASS", ANSI_GREEN, enable_color)
        elif is_scored:
            status = color("FAIL", ANSI_RED, enable_color)
        else:
            status = color("SKIP", ANSI_YELLOW, enable_color)

        print(
            f" iter={iteration:3d}  param={parameter_value:.6g}  "
            f"AN={analytic_grad:+.6e}  FD={fd_grad:+.6e}  "
            f"rel={rel_err:.3e}  fd_kind={fd_kind}  {status}"
        )


def compute_check(
    data_frame: pd.DataFrame,
    parameter: str,
    tail: int,
    rel_eps: float,
    rel_threshold: float,
    ignore_boundaries: bool,
    exclude_last_row: bool,
    grad_floor: float,
) -> dict[str, Any]:
    scored_data_frame = filter_rows(
        data_frame=data_frame,
        parameter=parameter,
        tail=tail,
        ignore_boundaries=ignore_boundaries,
        exclude_last_row=exclude_last_row,
    )

    analytic_all = scored_data_frame["analytic_grad"].to_numpy(dtype=np.float64)
    fd_all = scored_data_frame["fd_grad"].to_numpy(dtype=np.float64)

    active_mask = np.maximum(np.abs(analytic_all), np.abs(fd_all)) >= grad_floor
    skipped_mask = ~active_mask

    active_data_frame = scored_data_frame.loc[active_mask].copy()

    analytic = active_data_frame["analytic_grad"].to_numpy(dtype=np.float64)
    fd = active_data_frame["fd_grad"].to_numpy(dtype=np.float64)

    rel_err = np.array(
        [safe_rel_err(float(a), float(b), rel_eps) for a, b in zip(analytic, fd)],
        dtype=np.float64,
    )

    row_pass = rel_err <= rel_threshold
    row_fail = ~row_pass
    fail_frac = float(np.mean(row_fail)) if len(row_fail) else 0.0

    if len(active_data_frame):
        if np.any(row_fail):
            worst_idx = int(np.argmax(np.where(row_fail, rel_err, -1.0)))
        else:
            worst_idx = int(np.argmax(rel_err))
    else:
        worst_idx = -1

    metrics: dict[str, Any] = {
        "rows_used": int(len(active_data_frame)),
        "rows_skipped_small_grad": int(np.sum(skipped_mask)),
        "fail_frac": fail_frac,
        "rel_mean": float(np.mean(rel_err)) if len(rel_err) else float("nan"),
        "rel_median": float(np.median(rel_err)) if len(rel_err) else float("nan"),
        "rel_max": float(np.max(rel_err)) if len(rel_err) else float("nan"),
    }

    if worst_idx >= 0 and len(active_data_frame):
        worst_row = active_data_frame.iloc[worst_idx]
        metrics["worst_iter"] = int(worst_row["iter"])
        metrics["worst_param"] = float(worst_row[parameter])
        metrics["worst_an"] = float(worst_row["analytic_grad"])
        metrics["worst_fd"] = float(worst_row["fd_grad"])
        metrics["worst_rel"] = float(
            safe_rel_err(metrics["worst_an"], metrics["worst_fd"], rel_eps)
        )
        metrics["worst_fd_kind"] = int(worst_row["fd_kind"])
        metrics["worst_row_pass"] = bool(metrics["worst_rel"] <= rel_threshold)
    else:
        metrics["worst_iter"] = None

    last_row = data_frame.iloc[-1]
    metrics["last_iter"] = int(last_row["iter"])
    metrics["last_param"] = float(last_row[parameter])
    metrics["last_an"] = float(last_row["analytic_grad"])
    metrics["last_fd"] = float(last_row["fd_grad"])
    metrics["last_rel"] = float(
        safe_rel_err(metrics["last_an"], metrics["last_fd"], rel_eps)
    )
    metrics["last_fd_kind"] = int(last_row["fd_kind"])

    metrics["_scored_df"] = active_data_frame
    return metrics


def validate_common_args(common_args: list[str]) -> None:
    forbidden_common_flags = {
        "--index",
        "--min",
        "--max",
        "--forward_passes",
        "--bounces",
        "--adjoint_passes",
        "--adjoint_bounces",
    }

    for token in common_args:
        if token in forbidden_common_flags:
            raise RuntimeError(
                f"{token} must not appear in common_args; it must be specified per case."
            )


def validate_case(case: dict[str, Any]) -> None:
    required_case_fields = [
        "scene",
        "camera",
        "parameter",
        "index",
        "min",
        "max",
        "forward_passes",
        "bounces",
        "adjoint_passes",
        "adjoint_bounces",
    ]

    for field_name in required_case_fields:
        if field_name not in case:
            raise RuntimeError(f"Case missing required field '{field_name}': {case}")

    sweep_min = float(case["min"])
    sweep_max = float(case["max"])
    if not sweep_min < sweep_max:
        raise RuntimeError(
            f"Case has invalid sweep range: min must be < max, got min={sweep_min}, max={sweep_max}. Case: {case}"
        )

    point_index = int(case["index"])
    if point_index < 0:
        raise RuntimeError(f"Case index must be >= 0, got {point_index}. Case: {case}")

    if int(case["forward_passes"]) <= 0:
        raise RuntimeError(f"forward_passes must be > 0. Case: {case}")
    if int(case["bounces"]) < 0:
        raise RuntimeError(f"bounces must be >= 0. Case: {case}")
    if int(case["adjoint_passes"]) <= 0:
        raise RuntimeError(f"adjoint_passes must be > 0. Case: {case}")
    if int(case["adjoint_bounces"]) < 0:
        raise RuntimeError(f"adjoint_bounces must be >= 0. Case: {case}")


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        "Batch FD vs analytic gradient checker"
    )
    argument_parser.add_argument("--tests", type=str, required=True)
    argument_parser.add_argument("--script", type=str, default="./finite_difference/fd_test.py")
    argument_parser.add_argument("--workspace", type=str, default="./finite_difference/")
    argument_parser.add_argument("--python", type=str, default=sys.executable)
    argument_parser.add_argument("--grad_floor", type=float, default=1e-5)
    argument_parser.add_argument(
        "--render_target_script",
        type=str,
        default="finite_difference/render_target.py",
        help="Path to render_target.py",
    )
    argument_parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Use last N iterations after optionally dropping last row. 0 means use all.",
    )
    argument_parser.add_argument("--rel_eps", type=float, default=1e-12)
    argument_parser.add_argument("--rel_threshold", type=float, default=0.05)
    argument_parser.add_argument(
        "--fail_frac_threshold",
        type=float,
        default=0.5,
        help="Allow this fraction of scored rows to fail.",
    )
    argument_parser.add_argument(
        "--ignore_boundaries",
        action="store_true",
        help="Ignore opacity near 0 and 1 in scoring.",
    )
    argument_parser.add_argument(
        "--exclude_last_row",
        dest="exclude_last_row",
        action="store_true",
        default=True,
        help="Exclude the last CSV row from scoring.",
    )
    argument_parser.add_argument(
        "--include_last_row",
        dest="exclude_last_row",
        action="store_false",
        help="Include the last CSV row in scoring.",
    )
    argument_parser.add_argument("--no_color", action="store_true")
    argument_parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[])
    args = argument_parser.parse_args()

    enable_color = not args.no_color and sys.stdout.isatty()

    tests_path = Path(args.tests).resolve()
    config = json.loads(tests_path.read_text())

    cases = config.get("cases", [])
    common_args = [str(token) for token in config.get("common_args", [])]

    if not cases:
        raise RuntimeError("tests.json: no cases provided")

    validate_common_args(common_args)
    for case in cases:
        validate_case(case)

    workspace_dir = Path(args.workspace).resolve()
    script_path = Path(args.script).resolve()
    render_target_script = Path(args.render_target_script).resolve()

    failures = 0
    results: list[dict[str, Any]] = []

    for case_number, case in enumerate(cases, start=1):
        scene = str(case["scene"])
        camera = str(case["camera"])
        parameter = str(case["parameter"])
        point_index = int(case["index"])
        sweep_min = float(case["min"])
        sweep_max = float(case["max"])
        forward_passes = int(case["forward_passes"])
        bounces = int(case["bounces"])
        adjoint_passes = int(case["adjoint_passes"])
        adjoint_bounces = int(case["adjoint_bounces"])

        print("\n" + color(f"=== Case {case_number}/{len(cases)} ===", ANSI_BOLD, enable_color))
        print(
            f"scene={scene} camera={camera} parameter={parameter} "
            f"index={point_index} min={sweep_min} max={sweep_max} "
            f"forward_passes={forward_passes} bounces={bounces} "
            f"adjoint_passes={adjoint_passes} adjoint_bounces={adjoint_bounces}"
        )

        render_target_return_code = run_render_target(
            python_exe=args.python,
            render_target_script=render_target_script,
            scene=scene,
            parameter=parameter,
        )
        if render_target_return_code != 0:
            failures += 1
            print(
                color(
                    f"TARGET FAILED (exit {render_target_return_code})",
                    ANSI_RED,
                    enable_color,
                )
            )
            results.append(
                {
                    "scene": scene,
                    "camera": camera,
                    "parameter": parameter,
                    "index": point_index,
                    "min": sweep_min,
                    "max": sweep_max,
                    "forward_passes": forward_passes,
                    "bounces": bounces,
                    "adjoint_passes": adjoint_passes,
                    "adjoint_bounces": adjoint_bounces,
                    "status": "target_failed",
                }
            )
            continue

        run_return_code = run_one(
            python_exe=args.python,
            script_path=script_path,
            scene=scene,
            camera=camera,
            parameter=parameter,
            point_index=point_index,
            sweep_min=sweep_min,
            sweep_max=sweep_max,
            forward_passes=forward_passes,
            bounces=bounces,
            adjoint_passes=adjoint_passes,
            adjoint_bounces=adjoint_bounces,
            common_args=common_args,
            extra_args=[str(token) for token in args.extra_args],
        )
        if run_return_code != 0:
            failures += 1
            print(color(f"RUN FAILED (exit {run_return_code})", ANSI_RED, enable_color))
            results.append(
                {
                    "scene": scene,
                    "camera": camera,
                    "parameter": parameter,
                    "index": point_index,
                    "min": sweep_min,
                    "max": sweep_max,
                    "forward_passes": forward_passes,
                    "bounces": bounces,
                    "adjoint_passes": adjoint_passes,
                    "adjoint_bounces": adjoint_bounces,
                    "status": "run_failed",
                }
            )
            continue

        run_dir = resolve_run_dir(workspace_dir, scene, parameter, point_index)

        try:
            data_frame = load_csv(run_dir, camera, parameter)
        except Exception as exception:
            failures += 1
            print(color(f"CSV READ FAILED: {exception}", ANSI_RED, enable_color))
            results.append(
                {
                    "scene": scene,
                    "camera": camera,
                    "parameter": parameter,
                    "index": point_index,
                    "min": sweep_min,
                    "max": sweep_max,
                    "forward_passes": forward_passes,
                    "bounces": bounces,
                    "adjoint_passes": adjoint_passes,
                    "adjoint_bounces": adjoint_bounces,
                    "status": "csv_failed",
                }
            )
            continue

        metrics = compute_check(
            data_frame=data_frame,
            parameter=parameter,
            tail=args.tail,
            rel_eps=args.rel_eps,
            rel_threshold=args.rel_threshold,
            ignore_boundaries=args.ignore_boundaries,
            exclude_last_row=args.exclude_last_row,
            grad_floor=args.grad_floor,
        )

        case_passed = metrics["fail_frac"] <= args.fail_frac_threshold
        status_text = "PASS" if case_passed else "FAIL"

        print(color(status_text, ANSI_GREEN if case_passed else ANSI_RED, enable_color))
        print(f"run_dir: {run_dir}")
        print(
            f"rows_used: {metrics['rows_used']}  tail={args.tail}  "
            f"exclude_last_row={args.exclude_last_row}  ignore_boundaries={args.ignore_boundaries}"
        )
        print(
            f"thresholds: rel<={args.rel_threshold}; "
            f"allow_fail_frac={args.fail_frac_threshold}"
        )
        print(f"fail_frac: {metrics['fail_frac']:.3f}")
        print(
            f"rel_err (mean/median/max): "
            f"{metrics['rel_mean']:.6g} / {metrics['rel_median']:.6g} / {metrics['rel_max']:.6g}"
        )

        if metrics.get("worst_iter") is not None:
            worst_status_text = "pass" if metrics["worst_row_pass"] else "fail"
            worst_status_color = ANSI_GREEN if metrics["worst_row_pass"] else ANSI_RED
            print(
                "worst(scored): "
                f"iter={metrics['worst_iter']} param={metrics['worst_param']:.6g} "
                f"AN={metrics['worst_an']:.6g} FD={metrics['worst_fd']:.6g} "
                f"fd_kind={metrics['worst_fd_kind']} "
                f"[{color(worst_status_text, worst_status_color, enable_color)}]"
            )

        print_all_rows(
            data_frame=data_frame,
            parameter=parameter,
            scored_data_frame=metrics["_scored_df"],
            rel_eps=args.rel_eps,
            rel_threshold=args.rel_threshold,
            grad_floor=args.grad_floor,
            enable_color=enable_color,
        )

        if not case_passed:
            failures += 1

        results.append(
            {
                "scene": scene,
                "camera": camera,
                "parameter": parameter,
                "index": point_index,
                "min": sweep_min,
                "max": sweep_max,
                "forward_passes": forward_passes,
                "bounces": bounces,
                "adjoint_passes": adjoint_passes,
                "adjoint_bounces": adjoint_bounces,
                "status": "pass" if case_passed else "fail",
                **metrics,
            }
        )

    print("\n" + color("=== Summary ===", ANSI_BOLD, enable_color))
    passed_count = sum(1 for result in results if result["status"] == "pass")
    failed_count = len(results) - passed_count
    print(f"Total: {len(results)}  Passed: {passed_count}  Failed: {failed_count}")

    if failed_count:
        print(color("Failed cases:", ANSI_RED, enable_color))
        for result in results:
            if result["status"] != "pass":
                print(
                    f"- scene={result['scene']} camera={result['camera']} "
                    f"parameter={result['parameter']} index={result.get('index', 'n/a')} "
                    f"min={result.get('min', 'n/a')} max={result.get('max', 'n/a')} "
                    f"forward_passes={result.get('forward_passes', 'n/a')} "
                    f"bounces={result.get('bounces', 'n/a')} "
                    f"adjoint_passes={result.get('adjoint_passes', 'n/a')} "
                    f"adjoint_bounces={result.get('adjoint_bounces', 'n/a')} "
                    f"fail_frac={result.get('fail_frac', 'n/a')}"
                )

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()