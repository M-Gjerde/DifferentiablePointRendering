from __future__ import annotations

import argparse
import sys
from pathlib import Path

from finite_difference.fd_common import format_result, load_suite, run_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one schema-v2 Pale finite-difference case. The test uses a "
            "central stencil at multiple epsilon values and an arbitrary image VJP."
        )
    )
    parser.add_argument("--suite", type=Path, required=True, help="Schema-v2 JSON suite.")
    parser.add_argument("--case", required=True, help="Exact case name from the suite.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for result.json and samples.csv.")
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit zero after writing the result (useful for exploratory runs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, cases = load_suite(args.suite)
    matches = [case for case in cases if case["name"] == args.case]
    if len(matches) != 1:
        available = ", ".join(str(case["name"]) for case in cases)
        raise ValueError(f"Expected exactly one case named '{args.case}'. Available: {available}")

    result = run_case(matches[0], args.output.resolve())
    print(format_result(result), flush=True)
    return 0 if result["pass"] or args.no_fail else 1


if __name__ == "__main__":
    sys.exit(main())
