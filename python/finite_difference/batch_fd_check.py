# batch_fd_check.py
#
# Changes vs your pasted version:
# 1) Exclude the *last* CSV row from scoring (default ON via --exclude_last_row).
# 3) Keep last row only for *printing visibility*, not for scoring.
#
# Pass condition per scored row:
#   abs_err <= abs_threshold  OR  rel_err <= rel_threshold
#
# Overall test pass:
#   fail_frac <= fail_frac_threshold
#
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


def color(s: str, c: str, enable: bool) -> str:
    return f"{c}{s}{ANSI_RESET}" if enable else s


def safe_rel_err(a: float, b: float, eps: float) -> float:
    denom = max(eps, abs(a) + abs(b))
    return abs(a - b) / denom


def load_csv(run_dir: Path, camera: str, parameter: str) -> pd.DataFrame:
    csv_path = run_dir / f"{camera}_{parameter}_sweep.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    needed = {"iter", parameter, "analytic_grad", "fd_grad", "fd_kind"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"CSV missing columns {needed}, has: {list(df.columns)}")
    return df


def resolve_run_dir(workspace_dir: Path, scene: str, parameter: str) -> Path:
    scene_path = Path(scene)
    return workspace_dir / "Output" / scene_path / parameter / "0"


def run_render_target(
    python_exe: str,
    render_target_script: Path,
    scene: str,
    parameter: str,
) -> int:
    cmd = [python_exe, str(render_target_script), "--parameter", parameter, "--scene", scene]
    print(color("TARGET:", ANSI_BOLD, True), " ".join(cmd))
    p = subprocess.run(cmd)
    return int(p.returncode)


def run_one(
    python_exe: str,
    script_path: Path,
    scene: str,
    camera: str,
    parameter: str,
    common_args: list[str],
    extra_args: list[str],
) -> int:
    cmd = [python_exe, str(script_path), "--parameter", parameter, "--scene", scene, "--camera", camera]
    cmd += common_args
    cmd += extra_args
    print(color("RUN:", ANSI_BOLD, True), " ".join(cmd))
    p = subprocess.run(cmd)
    return int(p.returncode)


def filter_rows(
    df: pd.DataFrame,
    parameter: str,
    tail: int,
    ignore_boundaries: bool,
    exclude_last_row: bool,
) -> pd.DataFrame:
    dft = df

    # Drop last row BEFORE tailing (this is what you want: last row often dominated by MC noise)
    if exclude_last_row and len(dft) >= 2:
        dft = dft.iloc[:-1]

    # Now tail
    dft = dft.tail(tail) if tail > 0 else dft

    if ignore_boundaries and parameter == "opacity":
        v = dft[parameter].to_numpy(dtype=np.float64)
        mask = (v > 1e-6) & (v < 1.0 - 1e-6)
        dft = dft.loc[mask]

    return dft


def print_all_rows(
    df: pd.DataFrame,
    parameter: str,
    scored_df: pd.DataFrame,
    rel_eps: float,
    rel_threshold: float,
    abs_threshold: float,
    enable_color: bool,
):
    scored_iters = set(scored_df["iter"].astype(int).tolist())

    print("\nAll rows (scored + excluded):")
    for _, r in df.iterrows():
        it = int(r["iter"])
        param = float(r[parameter])
        an = float(r["analytic_grad"])
        fd = float(r["fd_grad"])
        fd_kind = int(r["fd_kind"])

        abs_err = abs(an - fd)
        rel_err = safe_rel_err(an, fd, rel_eps)

        passes = (abs_err <= abs_threshold) or (rel_err <= rel_threshold)
        is_scored = it in scored_iters

        if is_scored and passes:
            status = color("PASS", ANSI_GREEN, enable_color)
        elif is_scored and not passes:
            status = color("FAIL", ANSI_RED, enable_color)
        else:
            status = color("SKIP", ANSI_YELLOW, enable_color)

        print(
            f" iter={it:3d}  param={param:.6g}  "
            f"AN={an:+.6e}  FD={fd:+.6e}  "
            f"abs={abs_err:.3e}  rel={rel_err:.3e}  "
            f"fd_kind={fd_kind}  {status}"
        )

def compute_check(
    df: pd.DataFrame,
    parameter: str,
    tail: int,
    rel_eps: float,
    rel_threshold: float,
    abs_threshold: float,
    ignore_boundaries: bool,
    exclude_last_row: bool,
) -> dict[str, Any]:
    scored = filter_rows(df, parameter, tail, ignore_boundaries, exclude_last_row)

    an = scored["analytic_grad"].to_numpy(dtype=np.float64)
    fd = scored["fd_grad"].to_numpy(dtype=np.float64)

    abs_err = np.abs(an - fd)
    rel_err = np.array([safe_rel_err(float(a), float(b), rel_eps) for a, b in zip(an, fd)], dtype=np.float64)

    row_pass = (abs_err <= abs_threshold) | (rel_err <= rel_threshold)
    row_fail = ~row_pass

    fail_frac = float(np.mean(row_fail)) if len(row_fail) else 0.0

    if len(row_fail) and len(scored):
        worst_idx = int(np.argmax(np.where(row_fail, rel_err, -1.0))) if np.any(row_fail) else int(np.argmax(rel_err))
    else:
        worst_idx = -1

    out: dict[str, Any] = {
        "rows_used": int(len(scored)),
        "fail_frac": fail_frac,
        "rel_mean": float(np.mean(rel_err)) if len(rel_err) else float("nan"),
        "rel_median": float(np.median(rel_err)) if len(rel_err) else float("nan"),
        "rel_max": float(np.max(rel_err)) if len(rel_err) else float("nan"),
        "abs_mean": float(np.mean(abs_err)) if len(abs_err) else float("nan"),
        "abs_max": float(np.max(abs_err)) if len(abs_err) else float("nan"),
    }

    if worst_idx >= 0 and len(scored):
        r = scored.iloc[worst_idx]
        out["worst_iter"] = int(r["iter"])
        out["worst_param"] = float(r[parameter])
        out["worst_an"] = float(r["analytic_grad"])
        out["worst_fd"] = float(r["fd_grad"])
        out["worst_abs"] = float(abs(out["worst_an"] - out["worst_fd"]))
        out["worst_rel"] = float(safe_rel_err(out["worst_an"], out["worst_fd"], rel_eps))
        out["worst_fd_kind"] = int(r["fd_kind"])
        out["worst_row_pass"] = bool(
            (out["worst_abs"] <= abs_threshold) or (out["worst_rel"] <= rel_threshold)
        )
    else:
        out["worst_iter"] = None

    # Keep last row for visibility ONLY (not used in scoring)
    last = df.iloc[-1]
    out["last_iter"] = int(last["iter"])
    out["last_param"] = float(last[parameter])
    out["last_an"] = float(last["analytic_grad"])
    out["last_fd"] = float(last["fd_grad"])
    out["last_abs"] = float(abs(out["last_an"] - out["last_fd"]))
    out["last_rel"] = float(safe_rel_err(out["last_an"], out["last_fd"], rel_eps))
    out["last_fd_kind"] = int(last["fd_kind"])
    out["_scored_df"] = scored
    return out


def main() -> None:
    ap = argparse.ArgumentParser("Batch FD vs analytic gradient checker (robust)")
    ap.add_argument("--tests", type=str, required=True)
    ap.add_argument("--script", type=str, default="./finite_difference/finite_difference_test.py")
    ap.add_argument("--workspace", type=str, default="./finite_difference/")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument(
        "--render_target_script",
        type=str,
        default="finite_difference/render_target.py",
        help="Path to render_target.py",
    )

    ap.add_argument("--tail", type=int, default=0, help="Use last N iterations AFTER dropping last row (0=all)")
    ap.add_argument("--rel_eps", type=float, default=1e-12)

    ap.add_argument("--rel_threshold", type=float, default=0.05)
    ap.add_argument("--abs_threshold", type=float, default=1e-4)
    ap.add_argument("--fail_frac_threshold", type=float, default=0.0, help="Allow this fraction of rows to fail")

    ap.add_argument("--ignore_boundaries", action="store_true", help="Ignore opacity near 0 and 1 in scoring")
    ap.add_argument(
        "--exclude_last_row",
        action="store_true",
        help="Exclude the last CSV row from scoring (recommended; often pure MC noise)",
        default=True
    )
    ap.add_argument("--no_color", action="store_true")
    ap.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    enable_color = not args.no_color and sys.stdout.isatty()

    cfg = json.loads(Path(args.tests).read_text())
    cases = cfg.get("cases", [])
    common_args = [str(x) for x in cfg.get("common_args", [])]

    if not cases:
        raise RuntimeError("tests.json: no cases provided")

    workspace_dir = Path(args.workspace).resolve()
    script_path = Path(args.script).resolve()
    render_target_script = Path(args.render_target_script).resolve()

    failures = 0
    results: list[dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        scene = case["scene"]
        camera = case["camera"]
        parameter = case["parameter"]

        print("\n" + color(f"=== Case {i}/{len(cases)} ===", ANSI_BOLD, enable_color))
        print(f"scene={scene} camera={camera} parameter={parameter}")

        # Render target first
        rc = run_render_target(
            python_exe=args.python,
            render_target_script=render_target_script,
            scene=scene,
            parameter=parameter,
        )
        if rc != 0:
            failures += 1
            print(color(f"TARGET FAILED (exit {rc})", ANSI_RED, enable_color))
            results.append({"scene": scene, "camera": camera, "parameter": parameter, "status": "target_failed"})
            continue

        # Run FD test
        rc = run_one(
            python_exe=args.python,
            script_path=script_path,
            scene=scene,
            camera=camera,
            parameter=parameter,
            common_args=common_args,
            extra_args=[str(x) for x in args.extra_args],
        )
        if rc != 0:
            failures += 1
            print(color(f"RUN FAILED (exit {rc})", ANSI_RED, enable_color))
            results.append({"scene": scene, "camera": camera, "parameter": parameter, "status": "run_failed"})
            continue

        run_dir = resolve_run_dir(workspace_dir, scene, parameter)
        try:
            df = load_csv(run_dir, camera, parameter)
        except Exception as e:
            failures += 1
            print(color(f"CSV READ FAILED: {e}", ANSI_RED, enable_color))
            results.append({"scene": scene, "camera": camera, "parameter": parameter, "status": "csv_failed"})
            continue

        m = compute_check(
            df=df,
            parameter=parameter,
            tail=args.tail,
            rel_eps=args.rel_eps,
            rel_threshold=args.rel_threshold,
            abs_threshold=args.abs_threshold,
            ignore_boundaries=args.ignore_boundaries,
            exclude_last_row=args.exclude_last_row,
        )

        ok = (m["fail_frac"] <= args.fail_frac_threshold)

        status = "PASS" if ok else "FAIL"
        print(color(status, ANSI_GREEN if ok else ANSI_RED, enable_color))
        print(f"run_dir: {run_dir}")
        print(
            f"rows_used: {m['rows_used']}  tail={args.tail}  "
            f"exclude_last_row={args.exclude_last_row}  ignore_boundaries={args.ignore_boundaries}"
        )
        print(
            f"thresholds: rel<={args.rel_threshold} OR abs<={args.abs_threshold} ; "
            f"allow_fail_frac={args.fail_frac_threshold}"
        )
        print(f"fail_frac: {m['fail_frac']:.3f}")
        print(f"rel_err (mean/median/max): {m['rel_mean']:.6g} / {m['rel_median']:.6g} / {m['rel_max']:.6g}")
        print(f"abs_err (mean/max):       {m['abs_mean']:.6g} / {m['abs_max']:.6g}")

        if m.get("worst_iter") is not None:
            wp = "pass" if m["worst_row_pass"] else "fail"
            wp_col = ANSI_GREEN if m["worst_row_pass"] else ANSI_RED
            print(
                "worst(scored): "
                f"iter={m['worst_iter']} param={m['worst_param']:.6g} "
                f"AN={m['worst_an']:.6g} FD={m['worst_fd']:.6g} "
                f"abs={m['worst_abs']:.6g} rel={m['worst_rel']:.6g} "
                f"fd_kind={m['worst_fd_kind']} "
                f"[{color(wp, wp_col, enable_color)}]"
            )

        # Last row visibility only
        print_all_rows(
            df=df,
            parameter=parameter,
            scored_df=m["_scored_df"],
            rel_eps=args.rel_eps,
            rel_threshold=args.rel_threshold,
            abs_threshold=args.abs_threshold,
            enable_color=enable_color,
        )

        if not ok:
            failures += 1

        results.append({"scene": scene, "camera": camera, "parameter": parameter, "status": "pass" if ok else "fail", **m})

    print("\n" + color("=== Summary ===", ANSI_BOLD, enable_color))
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = len(results) - passed
    print(f"Total: {len(results)}  Passed: {passed}  Failed: {failed}")
    if failed:
        print(color("Failed cases:", ANSI_RED, enable_color))
        for r in results:
            if r["status"] != "pass":
                print(
                    f"- scene={r['scene']} camera={r['camera']} parameter={r['parameter']} "
                    f"fail_frac={r.get('fail_frac', 'n/a')}"
                )

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()