#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Data structure for a single gradient entry
# ============================================================

@dataclass
class GradientEntry:
    parameter_name: str
    axis_name: str
    camera_name: str
    eps_sign: str
    scene_name: str
    gaussian_index: int
    loss_value: float
    fd_value: float
    an_value: float

    @property
    def error(self) -> float:
        return self.fd_value - self.an_value

    @property
    def abs_error(self) -> float:
        return abs(self.error)

    @property
    def rel_error(self) -> Optional[float]:
        if abs(self.an_value) == 0:
            return None
        return self.abs_error / abs(self.an_value)


# ============================================================
# Supported parameter types
# ============================================================

def all_parameters() -> List[str]:
    return ["translation", "rotation", "scale", "opacity", "albedo", "beta"]


# ============================================================
# Argument Parsing
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FD sweeping tool. Runs cost_finite_difference.py for one or all parameters."
    )

    parser.add_argument(
        "--script-path",
        type=Path,
        default=Path("finite_difference") / "cost_finite_difference.py",
        help="Path to FD script."
    )

    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name (e.g., target)"
    )

    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Gaussian index"
    )

    parser.add_argument(
        "--param",
        type=str,
        choices=all_parameters(),
        required=False,
        help="If set: only run this parameter. If omitted: run all."
    )

    parser.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help="Camera list"
    )

    parser.add_argument(
        "--eps-signs",
        nargs="+",
        default=["neg", "pos"],
        choices=["neg", "pos"],
        help="Which eps directions to test"
    )

    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python interpreter to use"
    )

    return parser.parse_args()


# ============================================================
# Path helpers
# ============================================================

def history_filename_for_param(param: str) -> str:
    return f"{param}_gradients_history.csv"


def build_base_output_dir(script_path: Path, scene: str, camera: str) -> Path:
    return script_path.parent / "finite_diff" / "beta_kernel" / scene / camera


# ============================================================
# FD Script Runner
# ============================================================

def run_fd_script(
        python_exec: str,
        script_path: Path,
        scene: str,
        index: int,
        param: str,
        camera: str,
        eps_sign: str,
        axis: str
) -> None:
    args = [
        python_exec,
        str(script_path),
        "--scene", scene,
        "--index", str(index),
        "--param", param,
        "--axis", axis,
        "--camera", camera,
        "--eps", eps_sign
    ]

    completed = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if completed.returncode != 0:
        print(f"[WARN] FD script failed for {param}/{camera}/{eps_sign}")
        print(completed.stderr)


# ============================================================
# CSV Reading
# ============================================================

def read_last_csv_row(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
        return rows[-1] if rows else None


def parse_float(v: str) -> Optional[float]:
    if v is None:
        return None
    v = v.strip().lower()
    if v == "" or v == "nan":
        return None
    try:
        return float(v)
    except:
        return None

def extract_entries(
    param: str,
    camera: str,
    eps: str,
    row: Dict[str, str],
    axis_filter: Optional[str] = None,
) -> List[GradientEntry]:
    out: List[GradientEntry] = []

    scene = row["scene"]
    idx = int(row["index"])
    loss = float(row["loss"])

    # Only consider the axis you actually measured, if provided
    axes = ["x", "y", "z"]
    if axis_filter is not None:
        axes = [axis_filter]

    for axis in axes:
        fd = parse_float(row.get(f"fd_{axis}", ""))
        an = parse_float(row.get(f"an_{axis}", ""))

        if fd is None or an is None:
            continue

        out.append(
            GradientEntry(
                parameter_name=param,
                axis_name=axis,
                camera_name=camera,
                eps_sign=eps,
                scene_name=scene,
                gaussian_index=idx,
                loss_value=loss,
                fd_value=fd,
                an_value=an,
            )
        )

    return out



# ============================================================
# Printing
# ============================================================

def fmt(v: Optional[float]) -> str:
    if v is None:
        return "None"
    return f"{v: .6e}"


def print_summary(entries: List[GradientEntry]) -> None:
    if not entries:
        print("No entries collected.")
        return

    entries.sort(key=lambda e: (e.parameter_name, e.camera_name, e.eps_sign, e.axis_name))

    current_group = None

    for e in entries:
        group = (e.parameter_name, e.camera_name, e.eps_sign)

        if group != current_group:
            current_group = group
            print()
            print(f"=== {e.parameter_name.upper()} | {e.camera_name} | eps={e.eps_sign} | loss={e.loss_value: .6e} ===")
            print(" axis |          FD            AN          |     abs_err        rel_err")
            print("------+-------------------------------------+-----------------------------")

        print(
            f"  {e.axis_name}   |  {fmt(e.fd_value)}  {fmt(e.an_value)}  |  "
            f"{fmt(e.abs_error)}  {fmt(e.rel_error)}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_arguments()

    # Choose parameters to run
    if args.param is None:
        parameters = all_parameters()
        print("[INFO] No --param specified → running ALL parameters.")
    else:
        parameters = [args.param]
        print(f"[INFO] Running only parameter: {args.param}")

    collected: List[GradientEntry] = []

    for param in parameters:
        for cam in args.cameras:
            for eps in args.eps_signs:
                for axis in ["x", "y", "z"]:
                    if param == "scale" and axis == "z":
                        continue

                    if param == "opacity" and axis != "x":
                        continue
                    if param == "beta" and axis != "x":
                        continue

                    print(f"\n[RUN] param={param} cam={cam}  axis={axis} eps={eps}")
                    run_fd_script(
                        python_exec=args.python_executable,
                        script_path=args.script_path,
                        scene=args.scene,
                        index=args.index,
                        param=param,
                        camera=cam,
                        eps_sign=eps,
                        axis=axis,
                    )

                    base = build_base_output_dir(args.script_path, args.scene, cam)
                    hist = base / param / history_filename_for_param(param)

                    row = read_last_csv_row(hist)
                    if row is None:
                        print(f"[WARN] No history CSV at {hist}")
                        continue

                    entries = extract_entries(param, cam, eps, row, axis_filter=axis)
                    collected.extend(entries)
    print_summary(collected)


if __name__ == "__main__":
    main()
