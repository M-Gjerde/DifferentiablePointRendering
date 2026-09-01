from __future__ import annotations

import argparse
import fnmatch
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finite_difference.fd_common import REPO_ROOT, format_result, load_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a rigorous schema-v2 Pale finite-difference suite in isolated subprocesses."
    )
    parser.add_argument(
        "--tests",
        type=Path,
        default=Path(__file__).with_name("tests_direct.json"),
        help="Schema-v2 suite JSON (default: tests_direct.json).",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case-name glob. Repeat to select several patterns.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Stage-name glob. Repeat to select several stages.",
    )
    parser.add_argument("--case-index", type=int, help="Run one zero-based case index after loading the suite.")
    parser.add_argument("--list", action="store_true", help="List selected cases without running them.")
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--python", default=sys.executable, help="Python executable for isolated cases.")
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("fd_test.py"),
        help="Single-case runner.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("Output") / "direct_suite",
        help="Suite output directory.",
    )
    parser.add_argument(
        "--pale-module-dir",
        type=Path,
        help="Optional directory containing pale.so; prepended to PYTHONPATH for child processes.",
    )
    return parser.parse_args()


def _matches(value: str, patterns: list[str]) -> bool:
    return not patterns or any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def _selected_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(cases):
            raise ValueError(f"--case-index={args.case_index} outside [0, {len(cases)})")
        cases = [cases[args.case_index]]
    selected = [
        case
        for case in cases
        if _matches(str(case["name"]), args.case) and _matches(str(case["stage"]), args.stage)
    ]
    if not selected:
        raise ValueError("Case filters selected zero tests")
    return selected


def _git_revision() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else "unknown"


def main() -> int:
    args = parse_args()
    suite_path = args.tests.resolve()
    _, all_cases = load_suite(suite_path)
    cases = _selected_cases(all_cases, args)

    if args.list:
        for index, case in enumerate(cases):
            settings = case["settings"]
            print(
                f"{index:02d} {case['stage']:<18} {case['name']:<42} "
                f"{case['parameter']}[{case['index']}] "
                f"shared={settings['share_local_layer_direct_lighting']} "
                f"lowpass={settings['minimum_projected_footprint']} "
                f"batch={settings.get('point_hit_batch_size', 'default')}"
            )
        return 0

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    python_paths = [str(REPO_ROOT / "python")]
    if args.pale_module_dir is not None:
        python_paths.insert(0, str(args.pale_module_dir.resolve()))
    existing_python_path = environment.get("PYTHONPATH", "")
    if existing_python_path:
        python_paths.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    results: list[dict[str, Any]] = []
    infrastructure_failures: list[dict[str, Any]] = []
    for number, case in enumerate(cases, start=1):
        name = str(case["name"])
        case_output = output_root / name
        case_output.mkdir(parents=True, exist_ok=True)
        for stale_artifact in (case_output / "result.json", case_output / "samples.csv"):
            stale_artifact.unlink(missing_ok=True)
        command = [
            str(args.python),
            str(args.script.resolve()),
            "--suite",
            str(suite_path),
            "--case",
            name,
            "--output",
            str(case_output),
        ]
        print(f"\n=== {number}/{len(cases)} {case['stage']}/{name} ===", flush=True)
        process = subprocess.run(command, cwd=REPO_ROOT / "python", env=environment, check=False)
        result_path = case_output / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            results.append(result)
            if process.returncode == 0 and not result["pass"]:
                infrastructure_failures.append(
                    {"name": name, "error": "runner exited zero for a failing result"}
                )
            if process.returncode != 0 and result["pass"]:
                infrastructure_failures.append(
                    {"name": name, "error": f"runner exited {process.returncode} for a passing result"}
                )
        else:
            infrastructure_failures.append(
                {"name": name, "error": f"runner exited {process.returncode} without result.json"}
            )
        if process.returncode != 0 and not args.keep_going:
            break

    passed = sum(bool(result["pass"]) for result in results)
    failed = len(results) - passed
    manifest = {
        "schema_version": 2,
        "suite": str(suite_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "selected_case_count": len(cases),
        "completed_case_count": len(results),
        "passed": passed,
        "failed": failed,
        "infrastructure_failures": infrastructure_failures,
        "results": results,
    }
    (output_root / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== Finite-difference summary ===")
    for result in results:
        print(format_result(result))
    for failure in infrastructure_failures:
        print(f"[ERROR] {failure['name']}: {failure['error']}")
    print(
        f"completed={len(results)}/{len(cases)} passed={passed} failed={failed} "
        f"infrastructure_errors={len(infrastructure_failures)}"
    )
    print(f"summary={output_root / 'summary.json'}")
    return 0 if len(results) == len(cases) and failed == 0 and not infrastructure_failures else 1


if __name__ == "__main__":
    sys.exit(main())
