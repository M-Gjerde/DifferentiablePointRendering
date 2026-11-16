"""Shared output paths and timestamp parsing for training analysis tools."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def default_optimization_output_root() -> Path:
    return Path(__file__).resolve().parent / "OptimizationOutput"


def parse_run_timestamp(run_dir_name: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", run_dir_name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None
