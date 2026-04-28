# batch_fd_check.py
#
# Batch finite-difference vs analytic gradient checker.
#
# Features:
# - Reads per-case sweep config from test_Y_empty.json
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
# Example test_Y_empty.json:
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
import shutil

import numpy as np
import pandas as pd


ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[96m"
ANSI_BOLD_CYAN = "\033[1;96m"

def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Batch finite-difference versus analytic gradient checker."
    )
    argument_parser.add_argument(
        "--tests",
        type=str,
        required=True,
        help="Path to the test_Y_empty.json file containing common_args and per-case configurations.",
    )
    argument_parser.add_argument(
        "--script",
        type=str,
        default="./finite_difference/fd_test.py",
        help="Path to the finite-difference test driver script.",
    )
    argument_parser.add_argument(
        "--workspace",
        type=str,
        default="./finite_difference/",
        help="Workspace root directory containing the Output folder with generated sweep CSV files.",
    )
    argument_parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch the helper scripts.",
    )
    argument_parser.add_argument(
        "--grad_floor",
        type=float,
        default=1e-4,
        help="Minimum gradient magnitude required for a row to be scored. Rows below this threshold are skipped.",
    )
    argument_parser.add_argument(
        "--render_target_script",
        type=str,
        default="finite_difference/render_target.py",
        help="Path to the target-render generation script.",
    )
    argument_parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Use only the last N rows after optional exclusion of the final row. Use 0 to score all available rows.",
    )
    argument_parser.add_argument(
        "--rel_eps",
        type=float,
        default=1e-12,
        help="Small epsilon used in the relative error denominator for numerical stability.",
    )
    argument_parser.add_argument(
        "--rel_threshold",
        type=float,
        default=0.05,
        help="Maximum allowed relative error for an individual scored row to pass.",
    )
    argument_parser.add_argument(
        "--fail_frac_threshold",
        type=float,
        default=0.3,
        help="Maximum allowed fraction of failing scored rows for an entire case to pass.",
    )
    argument_parser.add_argument(
        "--ignore_boundaries",
        action="store_true",
        help="Ignore opacity rows near 0 and 1 when scoring.",
    )
    argument_parser.add_argument(
        "--exclude_last_row",
        dest="exclude_last_row",
        action="store_true",
        help="Exclude the final CSV row from scoring.",
    )
    argument_parser.add_argument(
        "--no_color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    argument_parser.add_argument(
        "--case_index",
        type=int,
        default=None,
        help="Run only one case by zero-based index from test_Y_empty.json.",
    )
    argument_parser.add_argument(
        "--skip_first_cases",
        type=int,
        default=0,
        help="Skip the first N test cases from test_Y_empty.json before running the remaining cases.",
    )
    argument_parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded verbatim to the finite-difference test driver script.",
    )

    return argument_parser.parse_args()

def color(text: str, ansi_color: str, enable: bool) -> str:
    return f"{ansi_color}{text}{ANSI_RESET}" if enable else text

def full_width_headline(text: str, ansi_color: str, enable_color: bool, fill_char: str = "=") -> str:
    terminal_width = shutil.get_terminal_size(fallback=(100, 20)).columns
    plain_text = f" {text} "
    headline_text = plain_text.center(terminal_width, fill_char)
    return color(headline_text, ansi_color, enable_color)

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
    forward_passes: int,
    bounces: int,
) -> int:
    command = [
        python_exe,
        str(render_target_script),
        "--parameter",
        parameter,
        "--scene",
        scene,
        "--forward_passes",
        str(forward_passes),
        "--bounces",
        str(bounces),
    ]
    print(color("RENDER TARGET:", ANSI_BOLD_CYAN, True), " ".join(command))
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
    fd_epsilon: float,
    forward_passes: int,
    bounces: int,
    adjoint_passes: int,
    adjoint_bounces: int,
    enable_adjoint_shadow_rays: bool,
    target_mode: str,
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
        "--fd_epsilon",
        str(fd_epsilon),
        "--forward_passes",
        str(forward_passes),
        "--bounces",
        str(bounces),
        "--adjoint_passes",
        str(adjoint_passes),
        "--adjoint_bounces",
        str(adjoint_bounces),
        "--enable_adjoint_shadow_rays",
        "true" if enable_adjoint_shadow_rays else "false",
        "--target_mode",
        target_mode,
    ]
    command += common_args
    command += extra_args

    print(color("FD_TEST:", ANSI_BOLD_CYAN, True), " ".join(command))
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

        relative_error_percent = 100.0 * rel_err

        print(
            f" iter={iteration:3d}  param={parameter_value:.6f}  "
            f"AN={analytic_grad:+.6f}  FD={fd_grad:+.6f}  "
            f"rel={relative_error_percent:.2f}%  fd_kind={fd_kind}  {status}"
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

    active_mask = np.abs(analytic_all) >= grad_floor
    active_mask &= np.abs(fd_all) >= grad_floor

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

    rel_sum = float(np.sum(rel_err)) if len(rel_err) else float("nan")
    rel_mean = float(np.mean(rel_err)) if len(rel_err) else float("nan")
    rel_median = float(np.median(rel_err)) if len(rel_err) else float("nan")
    rel_max = float(np.max(rel_err)) if len(rel_err) else float("nan")
    rel_rms = float(np.sqrt(np.mean(np.square(rel_err)))) if len(rel_err) else float("nan")

    if len(rel_err) and rel_threshold > 0.0:
        normalized_score = max(0.0, 100.0 * (1.0 - rel_mean / rel_threshold))
    else:
        normalized_score = float("nan")

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
        "rel_sum": rel_sum,
        "rel_mean": rel_mean,
        "rel_median": rel_median,
        "rel_max": rel_max,
        "rel_rms": rel_rms,
        "score": normalized_score,
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
        "--fd_epsilon",
        "--forward_passes",
        "--bounces",
        "--adjoint_passes",
        "--adjoint_bounces",
        "--enable_adjoint_shadow_rays",
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
        "fd_epsilon",
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

    fd_epsilon = float(case["fd_epsilon"])
    if fd_epsilon <= 0.0:
        raise RuntimeError(f"fd_epsilon must be > 0. Case: {case}")

    if int(case["forward_passes"]) <= 0:
        raise RuntimeError(f"forward_passes must be > 0. Case: {case}")
    if int(case["bounces"]) < 0:
        raise RuntimeError(f"bounces must be >= 0. Case: {case}")
    if int(case["adjoint_passes"]) <= 0:
        raise RuntimeError(f"adjoint_passes must be > 0. Case: {case}")
    if int(case["adjoint_bounces"]) < 0:
        raise RuntimeError(f"adjoint_bounces must be >= 0. Case: {case}")

    target_mode = str(case.get("target_mode", "original"))
    valid_target_modes = {"original", "filled", "random"}
    if target_mode not in valid_target_modes:
        raise RuntimeError(
            f"Invalid target_mode '{target_mode}'. "
            f"Expected one of {sorted(valid_target_modes)}. Case: {case}"
        )

    enable_adjoint_shadow_rays = case.get("enable_adjoint_shadow_rays", False)
    if not isinstance(enable_adjoint_shadow_rays, bool):
        raise RuntimeError(
            f"enable_adjoint_shadow_rays must be a JSON boolean true/false. Case: {case}"
        )

def main() -> None:
    args = parse_args()

    enable_color = not args.no_color and sys.stdout.isatty()

    tests_path = Path(args.tests).resolve()
    config = json.loads(tests_path.read_text())

    cases = config.get("cases", [])
    common_args = [str(token) for token in config.get("common_args", [])]

    if not cases:
        raise RuntimeError("test_Y_empty.json: no cases provided")

    if args.skip_first_cases < 0:
        raise RuntimeError(
            f"--skip_first_cases must be >= 0, got {args.skip_first_cases}"
        )

    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(cases):
            raise RuntimeError(
                f"--case_index out of range: got {args.case_index}, "
                f"but test_Y_empty.json has {len(cases)} cases."
            )
        cases = [cases[args.case_index]]
    else:
        if args.skip_first_cases >= len(cases):
            raise RuntimeError(
                f"--skip_first_cases={args.skip_first_cases} skips all available cases. "
                f"test_Y_empty.json has only {len(cases)} cases."
            )
        cases = cases[args.skip_first_cases:]

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
        fd_epsilon = float(case["fd_epsilon"])
        forward_passes = int(case["forward_passes"])
        bounces = int(case["bounces"])
        adjoint_passes = int(case["adjoint_passes"])
        adjoint_bounces = int(case["adjoint_bounces"])
        target_mode = str(case.get("target_mode", "original"))
        enable_adjoint_shadow_rays = bool(case.get("enable_adjoint_shadow_rays", False))

        print()
        print(
            full_width_headline(
                f"Case {case_number}/{len(cases)} | {scene} | {camera} | {parameter}",
                ANSI_BOLD_CYAN,
                enable_color,
            )
        )
        print(
            f"scene={scene} camera={camera} parameter={parameter} "
            f"index={point_index} min={sweep_min} max={sweep_max} "
            f"fd_epsilon={fd_epsilon} "
            f"forward_passes={forward_passes} bounces={bounces} "
            f"adjoint_passes={adjoint_passes} adjoint_bounces={adjoint_bounces} "
            f"enable_adjoint_shadow_rays={enable_adjoint_shadow_rays} "
            f"target_mode={target_mode}"
        )

        render_target_return_code = run_render_target(
            python_exe=args.python,
            render_target_script=render_target_script,
            scene=scene,
            parameter=parameter,
            forward_passes=forward_passes,
            bounces=bounces,
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
                    "enable_adjoint_shadow_rays": enable_adjoint_shadow_rays,
                    "target_mode": target_mode,
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
            fd_epsilon=fd_epsilon,
            forward_passes=forward_passes,
            bounces=bounces,
            adjoint_passes=adjoint_passes,
            adjoint_bounces=adjoint_bounces,
            enable_adjoint_shadow_rays=enable_adjoint_shadow_rays,
            common_args=common_args,
            target_mode=target_mode,
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
        status_color = ANSI_GREEN if case_passed else ANSI_RED

        print(color(status_text, status_color, enable_color))
        print(f"run_dir: {run_dir}")
        print(
            f"rows_used: {metrics['rows_used']}  "
            f"rows_skipped_small_grad={metrics['rows_skipped_small_grad']}  "
            f"tail={args.tail}  "
            f"exclude_last_row={args.exclude_last_row}  "
            f"ignore_boundaries={args.ignore_boundaries}"
        )
        print(
            f"thresholds: rel<={100.0 * args.rel_threshold:.2f}%; "
            f"allow_fail_frac={100.0 * args.fail_frac_threshold:.2f}%"
        )

        print(f"fail_frac: {100.0 * metrics['fail_frac']:.2f}%")

        score_value = metrics["score"]
        if np.isnan(score_value):
            score_color = ANSI_YELLOW
            score_text = "SCORE: n/a"
        else:
            if score_value >= 90.0:
                score_color = ANSI_CYAN
            elif score_value >= 70.0:
                score_color = ANSI_GREEN
            elif score_value >= 50.0:
                score_color = ANSI_YELLOW
            else:
                score_color = ANSI_RED
            score_text = f"SCORE: {score_value:.2f}/100"

        print(
            f"rel_err: "
            f"sum={100.0 * metrics['rel_sum']:.2f}%  "
            f"mean={100.0 * metrics['rel_mean']:.2f}%  "
            f"median={100.0 * metrics['rel_median']:.2f}%  "
            f"rms={100.0 * metrics['rel_rms']:.2f}%  "
            f"max={100.0 * metrics['rel_max']:.2f}%"
        )

        print(
            color(
                f"{ANSI_BOLD}=== {score_text} ===",
                score_color,
                enable_color,
            )
        )

        if metrics.get("worst_iter") is not None:
            worst_status_text = "pass" if metrics["worst_row_pass"] else "fail"
            worst_status_color = ANSI_GREEN if metrics["worst_row_pass"] else ANSI_RED
            print(
                "worst(scored): "
                f"iter={metrics['worst_iter']}  "
                f"param={metrics['worst_param']:.6f}  "
                f"AN={metrics['worst_an']:+.6f}  "
                f"FD={metrics['worst_fd']:+.6f}  "
                f"rel={100.0 * metrics['worst_rel']:.2f}%  "
                f"fd_kind={metrics['worst_fd_kind']}  "
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

    if results:
        sorted_results = sorted(
            results,
            key=lambda result: (
                1 if result["status"] == "pass" else 0,
                result.get("score", float("-inf"))
                if isinstance(result.get("score", None), (int, float))
                and not np.isnan(result.get("score", float("nan")))
                else float("-inf"),
            ),
        )

        print("\nCase scores:")
        for result in sorted_results:
            result_score = result.get("score", float("nan"))
            if isinstance(result_score, (int, float)) and not np.isnan(result_score):
                result_score_text = f"{result_score:.2f}/100"
            else:
                result_score_text = "n/a"

            if result["status"] == "pass":
                result_status_color = ANSI_GREEN
            elif result["status"] in {"fail", "target_failed", "run_failed", "csv_failed"}:
                result_status_color = ANSI_RED
            else:
                result_status_color = ANSI_YELLOW

            print(
                f"- [{color(result['status'], result_status_color, enable_color)}] "
                f"scene={result['scene']} camera={result['camera']} "
                f"parameter={result['parameter']} index={result.get('index', 'n/a')} "
                f"score={result_score_text} "
                f"fail_frac="
                f"{100.0 * result.get('fail_frac', float('nan')):.2f}%"
                if isinstance(result.get("fail_frac", None), (int, float))
                and not np.isnan(result.get("fail_frac", float("nan")))
                else
                f"- [{color(result['status'], result_status_color, enable_color)}] "
                f"scene={result['scene']} camera={result['camera']} "
                f"parameter={result['parameter']} index={result.get('index', 'n/a')} "
                f"score={result_score_text} fail_frac=n/a"
            )

    if failed_count:
        print(color("Failed cases:", ANSI_RED, enable_color))
        for result in results:
            if result["status"] != "pass":
                fail_frac_value = result.get("fail_frac", float("nan"))
                fail_frac_text = (
                    f"{100.0 * fail_frac_value:.2f}%"
                    if isinstance(fail_frac_value, (int, float)) and not np.isnan(fail_frac_value)
                    else "n/a"
                )
                score_value = result.get("score", float("nan"))
                score_text = (
                    f"{score_value:.2f}/100"
                    if isinstance(score_value, (int, float)) and not np.isnan(score_value)
                    else "n/a"
                )

                print(
                    f"- scene={result['scene']} camera={result['camera']} "
                    f"parameter={result['parameter']} index={result.get('index', 'n/a')} "
                    f"min={result.get('min', 'n/a')} max={result.get('max', 'n/a')} "
                    f"forward_passes={result.get('forward_passes', 'n/a')} "
                    f"bounces={result.get('bounces', 'n/a')} "
                    f"adjoint_passes={result.get('adjoint_passes', 'n/a')} "
                    f"adjoint_bounces={result.get('adjoint_bounces', 'n/a')} "
                    f"score={score_text} fail_frac={fail_frac_text}"
                )

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()