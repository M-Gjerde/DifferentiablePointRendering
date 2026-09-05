#!/usr/bin/env python3
"""Run a resumable, single-worker, Chamfer-guided hyperparameter study."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import inspect
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.evaluate_runs import (
    MeshCheckpoint,
    compute_geometry_rows,
    read_csv_dicts,
    safe_float,
    write_dict_csv,
)
from experiments.search_common import (
    CONFIG_CLI_FLAGS,
    build_train_command,
    parameter_digest,
)


SEARCH_TYPES = {"float", "int", "categorical"}
FAILED_POINT_CAP = "FAILED_POINT_CAP"
FAILED_POINT_UNSTABLE = "FAILED_POINT_UNSTABLE"
FAILED_PROCESS = "FAILED_PROCESS"
COMPLETED = "COMPLETED"
PRUNED_CHAMFER = "PRUNED_CHAMFER"


class TrialRunError(RuntimeError):
    """A trial failed for a reason that should not stop the overall study."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially optimize main.py hyperparameters using checkpoint "
            "Chamfer distance and a persistent Optuna study."
        )
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=PROJECT_ROOT / "experiments/teapot_adaptive_search.json",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Total terminal trials desired in the study; defaults to the spec.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="How often to inspect metrics.csv while main.py is running.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--timeout-hours", type=float, default=None,
        help="Search budget from its first launch, including time between restarts. Overrides the spec.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one representative command without creating a study or run.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_spec(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved_path = resolve_path(path)
    with resolved_path.open("r", encoding="utf-8") as spec_file:
        payload = json.load(spec_file)
    if not isinstance(payload, dict):
        raise TypeError("Adaptive-search spec must contain a JSON object.")
    return resolved_path, payload


def require_optuna() -> Any:
    try:
        import optuna
    except ModuleNotFoundError as exception:
        raise SystemExit(
            "Adaptive search requires Optuna. Install it in the configured "
            "environment (for this project: conda install -c conda-forge optuna)."
        ) from exception
    return optuna


def validate_dimension(name: str, dimension: Any) -> None:
    if name not in CONFIG_CLI_FLAGS:
        raise KeyError(f"No main.py CLI mapping for search parameter: {name}")
    if not isinstance(dimension, dict):
        raise TypeError(f"Search dimension '{name}' must be an object.")
    dimension_type = dimension.get("type")
    if dimension_type not in SEARCH_TYPES:
        raise ValueError(
            f"Search dimension '{name}' has unsupported type {dimension_type!r}; "
            f"expected one of {sorted(SEARCH_TYPES)}."
        )
    if dimension_type == "categorical":
        choices = dimension.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Categorical dimension '{name}' needs non-empty choices.")
        return
    low = dimension.get("low")
    high = dimension.get("high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise TypeError(f"Dimension '{name}' needs numeric low/high bounds.")
    if low > high:
        raise ValueError(f"Dimension '{name}' has low > high.")
    if dimension.get("log", False) and low <= 0:
        raise ValueError(f"Log-scaled dimension '{name}' must have low > 0.")


def dimension_contains(dimension: dict[str, Any], value: Any) -> bool:
    if dimension["type"] == "categorical":
        return value in dimension["choices"]
    if not isinstance(value, (int, float)):
        return False
    if not float(dimension["low"]) <= float(value) <= float(dimension["high"]):
        return False
    if dimension["type"] == "int" and int(value) != value:
        return False
    return True


def validate_spec(spec: dict[str, Any], check_paths: bool = True) -> None:
    required = ["dataset_path", "ground_truth", "output_root", "base_args", "search_space"]
    missing = [name for name in required if name not in spec]
    if missing:
        raise KeyError(f"Adaptive-search spec is missing: {', '.join(missing)}")

    for name in ("timeout_hours", "trial_timeout_minutes", "trial_no_progress_minutes",
                 "min_free_disk_gib"):
        value = spec.get(name)
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError(f"{name} must be finite and positive.")
    percentile = float(spec.get("pruner_percentile", 50.0))
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("pruner_percentile must be in [0, 100].")

    if check_paths:
        dataset_path = resolve_path(spec["dataset_path"])
        ground_truth_path = resolve_path(spec["ground_truth"])
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"Dataset directory not found: {dataset_path}")
        if not ground_truth_path.is_file():
            raise FileNotFoundError(f"Ground-truth mesh not found: {ground_truth_path}")

    base_args = spec["base_args"]
    search_space = spec["search_space"]
    if not isinstance(base_args, dict) or not isinstance(search_space, dict) or not search_space:
        raise TypeError("base_args and a non-empty search_space must be JSON objects.")
    for name in base_args:
        if name not in CONFIG_CLI_FLAGS:
            raise KeyError(f"No main.py CLI mapping for base parameter: {name}")
    for name, dimension in search_space.items():
        if name in base_args:
            raise ValueError(f"Parameter '{name}' occurs in both base_args and search_space.")
        validate_dimension(name, dimension)

    iterations = int(base_args.get("iterations", 0))
    mesh_interval = int(base_args.get("mesh_extraction_interval", 0))
    rungs = [int(value) for value in spec.get("evaluation_iterations", [])]
    if iterations <= 0 or mesh_interval <= 0:
        raise ValueError("iterations and mesh_extraction_interval must be positive.")
    if not rungs or rungs != sorted(set(rungs)) or rungs[-1] != iterations:
        raise ValueError(
            "evaluation_iterations must be unique, sorted, and end at base_args.iterations."
        )
    if any(iteration <= 0 or iteration % mesh_interval != 0 for iteration in rungs):
        raise ValueError("Every evaluation iteration must be a positive mesh interval multiple.")

    guardrails = spec.get("guardrails", {})
    if int(guardrails.get("max_points", 0)) <= 0:
        raise ValueError("guardrails.max_points must be positive.")
    stability = guardrails.get("point_stability", {})
    if int(stability.get("window_iterations", 0)) <= 0:
        raise ValueError("point_stability.window_iterations must be positive.")
    if float(stability.get("max_relative_growth", -1.0)) < 0.0:
        raise ValueError("point_stability.max_relative_growth must be non-negative.")
    if int(stability.get("max_absolute_growth", -1)) < 0:
        raise ValueError("point_stability.max_absolute_growth must be non-negative.")
    if "max_new_points" in stability and int(stability["max_new_points"]) < 0:
        raise ValueError("point_stability.max_new_points must be non-negative.")
    enforce_fraction = float(stability.get("enforce_above_fraction", 0.9))
    if not 0.0 < enforce_fraction <= 1.0:
        raise ValueError("point_stability.enforce_above_fraction must be in (0, 1].")

    linked_parameters = spec.get("linked_parameters", {})
    if not isinstance(linked_parameters, dict):
        raise TypeError("linked_parameters must be an object.")
    for target, source in linked_parameters.items():
        if target not in CONFIG_CLI_FLAGS:
            raise KeyError(f"No main.py CLI mapping for linked target: {target}")
        if source not in search_space and source not in base_args:
            raise KeyError(f"Linked source '{source}' is not a searched or base parameter.")

    derived_parameters = spec.get("derived_parameters", {})
    if not isinstance(derived_parameters, dict):
        raise TypeError("derived_parameters must be an object.")
    for target, rule in derived_parameters.items():
        if target not in CONFIG_CLI_FLAGS:
            raise KeyError(f"No main.py CLI mapping for derived target: {target}")
        if target in search_space or target in base_args or target in linked_parameters:
            raise ValueError(
                f"Derived target '{target}' must not also be a base, searched, or linked parameter."
            )
        if not isinstance(rule, dict):
            raise TypeError(f"Derived parameter '{target}' must be an object.")
        source = rule.get("source")
        if source not in search_space and source not in base_args:
            raise KeyError(
                f"Derived source '{source}' for '{target}' is not a searched or base parameter."
            )
        for coefficient_name, default in (("scale", 1.0), ("offset", 0.0)):
            coefficient = rule.get(coefficient_name, default)
            if (
                not isinstance(coefficient, (int, float))
                or isinstance(coefficient, bool)
                or not math.isfinite(float(coefficient))
            ):
                raise ValueError(
                    f"Derived parameter '{target}' has invalid {coefficient_name}: {coefficient!r}"
                )

    initial_trials = spec.get("initial_trials", [])
    initial_trial_overrides = spec.get("initial_trial_overrides", [])
    if not isinstance(initial_trial_overrides, list):
        raise TypeError("initial_trial_overrides must be an array.")
    if len(initial_trial_overrides) > len(initial_trials):
        raise ValueError(
            "initial_trial_overrides cannot contain more entries than initial_trials."
        )
    override_targets = set(linked_parameters) | set(derived_parameters)
    for index, overrides in enumerate(initial_trial_overrides):
        if not isinstance(overrides, dict):
            raise TypeError(f"initial_trial_overrides[{index}] must be an object.")
        unknown = set(overrides) - override_targets
        if unknown:
            raise ValueError(
                f"initial_trial_overrides[{index}] may only override linked or derived "
                f"targets; unknown={sorted(unknown)}"
            )

    for index, initial in enumerate(initial_trials):
        if not isinstance(initial, dict):
            raise TypeError(f"initial_trials[{index}] must be an object.")
        unknown = set(initial) - set(search_space)
        missing_initial = set(search_space) - set(initial)
        if unknown or missing_initial:
            raise ValueError(
                f"initial_trials[{index}] must specify every search dimension; "
                f"missing={sorted(missing_initial)}, unknown={sorted(unknown)}"
            )
        out_of_range = [
            name
            for name, value in initial.items()
            if not dimension_contains(search_space[name], value)
        ]
        if out_of_range:
            raise ValueError(
                f"initial_trials[{index}] contains out-of-range values for: "
                f"{', '.join(sorted(out_of_range))}"
            )


def representative_value(dimension: dict[str, Any]) -> Any:
    dimension_type = dimension["type"]
    if dimension_type == "categorical":
        return dimension["choices"][0]
    low = dimension["low"]
    high = dimension["high"]
    if dimension.get("log", False):
        value = math.sqrt(float(low) * float(high))
    else:
        value = (float(low) + float(high)) / 2.0
    return int(round(value)) if dimension_type == "int" else value


def suggest_parameters(trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for name, dimension in search_space.items():
        dimension_type = dimension["type"]
        if dimension_type == "categorical":
            value = trial.suggest_categorical(name, dimension["choices"])
        elif dimension_type == "int":
            value = trial.suggest_int(
                name,
                int(dimension["low"]),
                int(dimension["high"]),
                step=int(dimension.get("step", 1)),
                log=bool(dimension.get("log", False)),
            )
        else:
            value = trial.suggest_float(
                name,
                float(dimension["low"]),
                float(dimension["high"]),
                step=dimension.get("step"),
                log=bool(dimension.get("log", False)),
            )
        parameters[name] = value
    return parameters


def apply_linked_parameters(parameters: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    result = dict(parameters)
    for target, source in spec.get("linked_parameters", {}).items():
        result[target] = result[source]
    for target, rule in spec.get("derived_parameters", {}).items():
        source = str(rule["source"])
        source_value = result[source]
        if (
            not isinstance(source_value, (int, float))
            or isinstance(source_value, bool)
            or not math.isfinite(float(source_value))
        ):
            raise ValueError(
                f"Derived source '{source}' for '{target}' must be finite and numeric, "
                f"got {source_value!r}."
            )
        result[target] = (
            float(source_value) * float(rule.get("scale", 1.0))
            + float(rule.get("offset", 0.0))
        )
    return result


def apply_initial_trial_overrides(
    parameters: dict[str, Any], trial: Any
) -> dict[str, Any]:
    result = dict(parameters)
    overrides = trial.user_attrs.get("initial_parameter_overrides", {})
    if overrides:
        result.update(overrides)
    return result


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    temporary_path.replace(path)


def spec_digest(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    return payload if isinstance(payload, dict) else None


def verify_config_snapshot(spec: dict[str, Any]) -> None:
    expected = spec.get("config_sha256")
    if expected and hashlib.sha256((PROJECT_ROOT / "config.py").read_bytes()).hexdigest() != expected:
        raise RuntimeError("config.py changed since this search was prepared. Generate a new study spec.")


def require_free_disk(output_root: Path, spec: dict[str, Any]) -> None:
    minimum = float(spec.get("min_free_disk_gib", 0.0))
    if minimum and shutil.disk_usage(output_root).free < minimum * 1024 ** 3:
        raise RuntimeError(f"Study stopped: less than {minimum:g} GiB free on {output_root}.")


def trial_timeout_reason(spec: dict[str, Any], elapsed: float, idle: float) -> str | None:
    for key, seconds, label in (
        ("trial_timeout_minutes", elapsed, "total trial time"),
        ("trial_no_progress_minutes", idle, "time without iteration progress"),
    ):
        limit = spec.get(key)
        if limit is not None and seconds >= float(limit) * 60.0:
            return f"Exceeded {float(limit):g} minutes of {label}."
    return None


def remaining_search_seconds(study_dir: Path, timeout_hours: float | None) -> float | None:
    if timeout_hours is None:
        return None
    budget_path = study_dir / "wall_clock_budget.json"
    budget = read_json_object(budget_path)
    if budget is None:
        budget = {"started_at": time.time()}
        atomic_write_json(budget_path, budget)
    return max(0.0, float(budget["started_at"]) + timeout_hours * 3600.0 - time.time())


def enqueue_initial_trials(study: Any, spec: dict[str, Any]) -> None:
    overrides = spec.get("initial_trial_overrides", [])
    repeated = bool(spec.get("allow_repeated_initial_trials", False))
    existing_indices = {t.user_attrs.get("initial_design_index") for t in study.trials}
    for index, parameters in enumerate(spec.get("initial_trials", [])):
        if repeated and index in existing_indices:
            continue
        attrs: dict[str, Any] = {"initial_design": True, "initial_design_index": index}
        if index < len(overrides) and overrides[index]:
            attrs["initial_parameter_overrides"] = dict(overrides[index])
        study.enqueue_trial(dict(parameters), user_attrs=attrs, skip_if_exists=not repeated)


def make_confirmation_spec(study: Any, spec: dict[str, Any]) -> dict[str, Any] | None:
    """Re-run feasible finalists plus the baseline in fresh processes, without pruning."""
    options = spec.get("confirmation", {})
    top_k, repeats = int(options.get("top_k", 0)), int(options.get("repeats", 3))
    if top_k <= 0 or repeats <= 0:
        return None
    eligible = [t for t in study.trials if t.user_attrs.get("outcome") == COMPLETED
                and t.value is not None and math.isfinite(t.value)
                and all(float(v) <= 0 for v in t.user_attrs.get("constraint_values", [0, 0]))]
    eligible.sort(key=lambda t: t.value)
    candidates: list[dict[str, Any]] = []
    for trial in eligible:
        params = dict(trial.params)
        if params not in candidates:
            candidates.append(params)
        if len(candidates) >= top_k:
            break
    if not candidates:
        return None
    baseline = dict(spec["initial_trials"][0])
    if baseline not in candidates:
        candidates.append(baseline)
    result = json.loads(json.dumps(spec))
    result.pop("confirmation", None)
    result.pop("initial_trial_overrides", None)
    result["study_name"] = str(spec["study_name"]) + "_confirmation"
    result["output_root"] = str(resolve_path(spec["output_root"]) / "confirmation")
    result["timeout_hours"] = float(options.get("timeout_hours", 12.0))
    result["confirmation_candidates"] = candidates
    # Round-robin candidates so interruption does not give only the first setting repetitions.
    result["initial_trials"] = [dict(p) for _ in range(repeats) for p in candidates]
    result["max_trials"] = len(result["initial_trials"])
    result["allow_repeated_initial_trials"] = True
    result["pruner_warmup_iteration"] = int(result["base_args"]["iterations"])
    result["enqueue_repairs"] = False
    return result


def export_confirmation_summary(study: Any, spec: dict[str, Any], study_dir: Path) -> None:
    candidates = spec.get("confirmation_candidates", [])
    if not candidates:
        return
    rows = []
    for index, params in enumerate(candidates):
        trials = [t for t in study.trials if t.params == params]
        values = [float(t.value) for t in trials if t.user_attrs.get("outcome") == COMPLETED
                  and t.value is not None and math.isfinite(t.value)]
        expected = len(spec["initial_trials"]) // len(candidates)
        values.sort()
        middle = len(values) // 2
        median = (values[middle] if len(values) % 2 else
                  (values[middle - 1] + values[middle]) / 2) if values else None
        rows.append({"candidate": index, "parameters": params, "cd_values": values,
                     "median_cd": median, "complete_repetitions": len(values),
                     "expected_repetitions": expected,
                     "eligible": len(values) == expected and len(trials) == expected})
    rows.sort(key=lambda r: (not r["eligible"], r["median_cd"] if r["median_cd"] is not None else math.inf))
    atomic_write_json(study_dir / "confirmation_summary.json", {"candidates": rows})


@contextmanager
def exclusive_study_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exception:
            raise RuntimeError(f"Another adaptive-search worker holds {lock_path}") from exception
        lock_file.seek(0)
        lock_file.truncate()
        json.dump(
            {"pid": os.getpid(), "host": socket.gethostname(), "started": time.time()},
            lock_file,
        )
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def finite_metric_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in rows:
        iteration = safe_float(row.get("iteration"))
        if iteration is not None:
            result.append(row)
    return result


def metric_iteration(row: dict[str, str]) -> int:
    value = safe_float(row.get("iteration"))
    if value is None:
        raise ValueError("Metrics row has no finite iteration.")
    return int(value)


def first_point_cap_excess(
    rows: list[dict[str, str]],
    max_points: int,
) -> tuple[int, int] | None:
    for row in finite_metric_rows(rows):
        point_count = safe_float(row.get("num_points"))
        if point_count is not None and point_count > max_points:
            return metric_iteration(row), int(point_count)
    return None


def metrics_diagnostics(
    rows: list[dict[str, str]],
    window_iterations: int,
) -> dict[str, Any]:
    rows = finite_metric_rows(rows)
    if not rows:
        return {}
    latest = rows[-1]
    latest_iteration = metric_iteration(latest)
    target_iteration = max(0, latest_iteration - int(window_iterations))
    window_start = min(rows, key=lambda row: abs(metric_iteration(row) - target_iteration))
    latest_points = safe_float(latest.get("num_points"))
    start_points = safe_float(window_start.get("num_points"))
    latest_rgb = safe_float(latest.get("loss_rgb_mean"))
    start_rgb = safe_float(window_start.get("loss_rgb_mean"))
    diagnostics: dict[str, Any] = {
        "iteration": latest_iteration,
        "window_start_iteration": metric_iteration(window_start),
        "num_points": int(latest_points) if latest_points is not None else None,
        "loss_rgb_mean": latest_rgb,
    }
    if latest_points is not None and start_points is not None:
        growth = latest_points - start_points
        diagnostics["point_growth"] = int(growth)
        diagnostics["point_growth_fraction"] = growth / max(start_points, 1.0)
    if latest_rgb is not None and start_rgb is not None:
        diagnostics["rgb_relative_improvement"] = (
            start_rgb - latest_rgb
        ) / max(abs(start_rgb), 1.0e-30)
    latest_clone_total = safe_float(latest.get("densification_clone_points_total"))
    latest_split_total = safe_float(latest.get("densification_split_points_total"))
    start_clone_total = safe_float(window_start.get("densification_clone_points_total"))
    start_split_total = safe_float(window_start.get("densification_split_points_total"))
    if None not in {
        latest_clone_total,
        latest_split_total,
        start_clone_total,
        start_split_total,
    }:
        assert latest_clone_total is not None
        assert latest_split_total is not None
        assert start_clone_total is not None
        assert start_split_total is not None
        diagnostics["densification_new_points"] = int(
            max(
                0.0,
                (latest_clone_total + latest_split_total)
                - (start_clone_total + start_split_total),
            )
        )
    return diagnostics


def point_count_is_stable(diagnostics: dict[str, Any], stability: dict[str, Any]) -> bool:
    if not stability.get("enabled", True):
        return True
    relative_growth = safe_float(diagnostics.get("point_growth_fraction"))
    absolute_growth = safe_float(diagnostics.get("point_growth"))
    if relative_growth is None or absolute_growth is None:
        return False
    net_count_is_stable = (
        relative_growth <= float(stability.get("max_relative_growth", 0.05))
        or absolute_growth <= int(stability.get("max_absolute_growth", 250))
    )
    gross_new_points = safe_float(diagnostics.get("densification_new_points"))
    max_new_points = stability.get("max_new_points")
    gross_additions_are_stable = (
        max_new_points is None
        or gross_new_points is None
        or gross_new_points <= int(max_new_points)
    )
    return net_count_is_stable and gross_additions_are_stable


def point_stability_violation(
    diagnostics: dict[str, Any],
    stability: dict[str, Any],
) -> float:
    if point_count_is_stable(diagnostics, stability):
        return 0.0
    relative_growth = safe_float(diagnostics.get("point_growth_fraction"))
    absolute_growth = safe_float(diagnostics.get("point_growth"))
    if relative_growth is None or absolute_growth is None:
        return 1.0
    relative_limit = float(stability.get("max_relative_growth", 0.05))
    absolute_limit = int(stability.get("max_absolute_growth", 250))
    net_violation = min(
        max(0.0, relative_growth - relative_limit),
        max(0.0, (absolute_growth - absolute_limit) / max(absolute_limit, 1)),
    )
    gross_new_points = safe_float(diagnostics.get("densification_new_points"))
    max_new_points = stability.get("max_new_points")
    gross_violation = 0.0
    if max_new_points is not None and gross_new_points is not None:
        gross_violation = max(
            0.0,
            (gross_new_points - int(max_new_points)) / max(int(max_new_points), 1),
        )
    return max(net_violation, gross_violation, 1.0e-12)


def stop_trial_process(process: subprocess.Popen[str], timeout_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def next_choice(value: Any, choices: list[Any], increasing: bool) -> Any:
    ordered = sorted(choices)
    try:
        index = ordered.index(value)
    except ValueError:
        return value
    next_index = min(index + 1, len(ordered) - 1) if increasing else max(index - 1, 0)
    return ordered[next_index]


def clamp_dimension_value(value: float, dimension: dict[str, Any]) -> float | int:
    clamped = min(max(value, float(dimension["low"])), float(dimension["high"]))
    if dimension["type"] == "int":
        return int(round(clamped))
    return clamped


def repair_densification_parameters(
    sampled_parameters: dict[str, Any],
    search_space: dict[str, Any],
) -> dict[str, Any] | None:
    repaired = dict(sampled_parameters)
    changed = False

    for name in ["densification_grad_abs_min", "densification_grad_abs_min_final"]:
        if name not in repaired or name not in search_space:
            continue
        dimension = search_space[name]
        if dimension["type"] == "categorical":
            value = next_choice(repaired[name], dimension["choices"], increasing=True)
        else:
            value = clamp_dimension_value(float(repaired[name]) * 1.6, dimension)
        changed = changed or value != repaired[name]
        repaired[name] = value

    name = "densification_interval"
    if name in repaired and name in search_space:
        dimension = search_space[name]
        if dimension["type"] == "categorical":
            value = next_choice(repaired[name], dimension["choices"], increasing=True)
        else:
            value = clamp_dimension_value(float(repaired[name]) * 1.5, dimension)
        changed = changed or value != repaired[name]
        repaired[name] = value

    name = "densification_max_new_fraction"
    if name in repaired and name in search_space:
        dimension = search_space[name]
        if dimension["type"] == "categorical":
            value = next_choice(repaired[name], dimension["choices"], increasing=False)
        else:
            value = clamp_dimension_value(float(repaired[name]) * 0.5, dimension)
        changed = changed or value != repaired[name]
        repaired[name] = value

    name = "curvature_violation_threshold"
    if name in repaired and name in search_space and float(repaired[name]) > 0.0:
        dimension = search_space[name]
        if dimension["type"] == "categorical":
            value = next_choice(repaired[name], dimension["choices"], increasing=True)
        else:
            value = clamp_dimension_value(float(repaired[name]) * 1.5, dimension)
        changed = changed or value != repaired[name]
        repaired[name] = value

    return repaired if changed else None


def enqueue_repaired_trial(
    study: Any,
    trial: Any,
    sampled_parameters: dict[str, Any],
    search_space: dict[str, Any],
    reason: str,
    enabled: bool = True,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    repaired = repair_densification_parameters(sampled_parameters, search_space)
    if repaired is None:
        return None
    study.enqueue_trial(
        repaired,
        user_attrs={"repair_of_trial": int(trial.number), "repair_reason": reason},
        skip_if_exists=True,
    )
    return repaired


def trial_penalty(spec: dict[str, Any], max_observed_points: int) -> float:
    max_points = int(spec["guardrails"]["max_points"])
    base_penalty = float(spec.get("failure_cd_penalty", 1.0))
    return base_penalty + max(0, max_observed_points - max_points) / max_points


def export_study_summary(study: Any, output_path: Path, search_names: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for frozen_trial in study.trials:
        row: dict[str, Any] = {
            "trial": frozen_trial.number,
            "state": frozen_trial.state.name,
            "value": frozen_trial.value,
            "outcome": frozen_trial.user_attrs.get("outcome", ""),
            "run_dir": frozen_trial.user_attrs.get("run_dir", ""),
            "max_observed_points": frozen_trial.user_attrs.get("max_observed_points", ""),
            "failure_iteration": frozen_trial.user_attrs.get("failure_iteration", ""),
            "repair_of_trial": frozen_trial.user_attrs.get("repair_of_trial", ""),
            "repair_reason": frozen_trial.user_attrs.get("repair_reason", ""),
            "best_cd": frozen_trial.user_attrs.get("best_cd", ""),
            "final_cd": frozen_trial.user_attrs.get("final_cd", ""),
            "point_growth_fraction": frozen_trial.user_attrs.get("point_growth_fraction", ""),
            "densification_new_points": frozen_trial.user_attrs.get("densification_new_points", ""),
        }
        for name in search_names:
            row[name] = frozen_trial.params.get(name, "")
        rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else ["trial", "state", "value", "outcome"]

    def write_rows(path: Path, selected_rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_rows)
        temporary_path.replace(path)

    write_rows(output_path, rows)
    write_rows(
        output_path.with_name("failed_settings.csv"),
        [row for row in rows if str(row.get("outcome", "")).startswith("FAILED_")],
    )


def make_objective(
    optuna: Any,
    study: Any,
    spec: dict[str, Any],
    output_root: Path,
    dataset_path: Path,
    ground_truth_path: Path,
    poll_seconds: float,
) -> Callable[[Any], float]:
    search_space = dict(spec["search_space"])
    base_args = dict(spec["base_args"])
    evaluation_iterations = [int(value) for value in spec["evaluation_iterations"]]
    guardrails = dict(spec["guardrails"])
    max_points = int(guardrails["max_points"])
    stability = dict(guardrails["point_stability"])
    study_dir = output_root / "_study"

    def objective(trial: Any) -> float:
        verify_config_snapshot(spec)
        require_free_disk(output_root, spec)
        sampled_parameters = suggest_parameters(trial, search_space)
        parameters = apply_initial_trial_overrides(
            apply_linked_parameters(
                dict(base_args, **sampled_parameters),
                spec,
            ),
            trial,
        )
        digest = parameter_digest(parameters)
        run_name = f"trial_{trial.number:04d}_{digest}"
        run_dir = output_root / run_name
        if run_dir.exists():
            raise TrialRunError(
                f"Refusing to clear existing adaptive trial directory: {run_dir}"
            )

        command = build_train_command(dataset_path, run_dir, parameters)
        log_path = study_dir / "logs" / f"{run_name}.log"
        state_path = study_dir / "trials" / f"trial_{trial.number:04d}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("command", command)
        state: dict[str, Any] = {
            "trial": trial.number,
            "run_name": run_name,
            "status": "RUNNING",
            "parameters": parameters,
            "sampled_parameters": sampled_parameters,
            "command": command,
            "log_path": str(log_path),
            "checkpoint_metrics": [],
            "max_observed_points": 0,
            "started_at": time.time(),
        }
        atomic_write_json(state_path, state)
        print(f"\n[trial {trial.number}] {' '.join(command)}", flush=True)

        checkpoint_rows: list[dict[str, Any]] = []
        evaluated_iterations: set[int] = set()
        evaluation_failures: dict[int, int] = {}
        max_observed_points = 0
        latest_metrics_rows: list[dict[str, str]] = []
        started_monotonic = time.monotonic()
        last_progress_at = started_monotonic
        last_progress_iteration = 0

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            state["pid"] = process.pid
            atomic_write_json(state_path, state)
            try:
                while True:
                    metrics_path = run_dir / "metrics.csv"
                    latest_metrics_rows = read_csv_dicts(metrics_path)
                    numeric_rows = finite_metric_rows(latest_metrics_rows)
                    latest_iteration = metric_iteration(numeric_rows[-1]) if numeric_rows else 0
                    now = time.monotonic()
                    if latest_iteration > last_progress_iteration:
                        last_progress_iteration = latest_iteration
                        last_progress_at = now
                    timeout_reason = trial_timeout_reason(
                        spec, now - started_monotonic, now - last_progress_at,
                    )
                    if timeout_reason and process.poll() is None:
                        stop_trial_process(process)
                        trial.set_user_attr("outcome", "FAILED_TIMEOUT")
                        trial.set_user_attr("failure_iteration", latest_iteration)
                        state.update(status="FAILED_TIMEOUT", failure_reason=timeout_reason,
                                     failure_iteration=latest_iteration, ended_at=time.time())
                        atomic_write_json(state_path, state)
                        raise TrialRunError(timeout_reason)
                    require_free_disk(output_root, spec)

                    for row in numeric_rows:
                        point_count = safe_float(row.get("num_points"))
                        if point_count is not None:
                            max_observed_points = max(max_observed_points, int(point_count))
                    state["max_observed_points"] = max_observed_points

                    point_cap_excess = first_point_cap_excess(numeric_rows, max_points)
                    if point_cap_excess is not None:
                        failure_iteration, failure_point_count = point_cap_excess
                        stop_trial_process(process)
                        outcome = FAILED_POINT_CAP
                        repaired = enqueue_repaired_trial(
                            study,
                            trial,
                            sampled_parameters,
                            search_space,
                            outcome,
                            bool(spec.get("enqueue_repairs", True)),
                        )
                        penalty = trial_penalty(spec, max_observed_points)
                        trial.set_user_attr("outcome", outcome)
                        trial.set_user_attr("failure_iteration", failure_iteration)
                        trial.set_user_attr("max_observed_points", max_observed_points)
                        trial.set_user_attr(
                            "constraint_values",
                            [(max_observed_points - max_points) / max_points, 0.0],
                        )
                        state.update(
                            status=outcome,
                            failure_iteration=failure_iteration,
                            repaired_parameters=repaired,
                            objective=penalty,
                            ended_at=time.time(),
                        )
                        atomic_write_json(state_path, state)
                        print(
                            f"[trial {trial.number}] {outcome} at iteration "
                            f"{failure_iteration}: {failure_point_count} > {max_points}; "
                            f"queued repair={repaired is not None}",
                            flush=True,
                        )
                        return penalty

                    if numeric_rows:
                        live_diagnostics = metrics_diagnostics(
                            latest_metrics_rows,
                            window_iterations=int(stability["window_iterations"]),
                        )
                        latest_point_count = int(
                            safe_float(live_diagnostics.get("num_points")) or 0
                        )
                        soft_limit = int(
                            max_points * float(stability.get("enforce_above_fraction", 0.9))
                        )
                        stability_min_iteration = int(
                            stability.get("min_iteration", evaluation_iterations[0])
                        )
                        if (
                            latest_iteration >= stability_min_iteration
                            and latest_point_count >= soft_limit
                            and not point_count_is_stable(live_diagnostics, stability)
                        ):
                            stop_trial_process(process)
                            outcome = FAILED_POINT_UNSTABLE
                            repaired = enqueue_repaired_trial(
                                study,
                                trial,
                                sampled_parameters,
                                search_space,
                                outcome,
                                bool(spec.get("enqueue_repairs", True)),
                            )
                            penalty = trial_penalty(spec, max_observed_points)
                            relative_growth = float(
                                live_diagnostics.get("point_growth_fraction", 1.0)
                            )
                            trial.set_user_attr("outcome", outcome)
                            trial.set_user_attr("failure_iteration", latest_iteration)
                            trial.set_user_attr("max_observed_points", max_observed_points)
                            trial.set_user_attr(
                                "constraint_values",
                                [0.0, point_stability_violation(live_diagnostics, stability)],
                            )
                            state.update(
                                status=outcome,
                                failure_iteration=latest_iteration,
                                diagnostics=live_diagnostics,
                                repaired_parameters=repaired,
                                objective=penalty,
                                ended_at=time.time(),
                            )
                            atomic_write_json(state_path, state)
                            print(
                                f"[trial {trial.number}] {outcome} near the point cap at "
                                f"iteration {latest_iteration}: points={latest_point_count}, "
                                f"window growth={relative_growth:.2%}; "
                                f"queued repair={repaired is not None}",
                                flush=True,
                            )
                            return penalty

                    for row in numeric_rows[-10:]:
                        for column in ["loss_rgb_mean", "loss_total_mean"]:
                            value = row.get(column)
                            if value not in {None, ""} and safe_float(value) is None:
                                stop_trial_process(process)
                                trial.set_user_attr("outcome", FAILED_PROCESS)
                                trial.set_user_attr("failure_iteration", metric_iteration(row))
                                state.update(
                                    status=FAILED_PROCESS,
                                    failure_iteration=metric_iteration(row),
                                    failure_reason=f"non-finite {column}: {value}",
                                )
                                atomic_write_json(state_path, state)
                                raise TrialRunError(state["failure_reason"])

                    for iteration in evaluation_iterations:
                        if iteration in evaluated_iterations or latest_iteration < iteration:
                            continue
                        mesh_path = (
                            run_dir
                            / "mesh_checkpoints"
                            / f"iter_{iteration:05d}"
                            / str(spec.get("reconstruction_name", "fuse_post.ply"))
                        )
                        if not mesh_path.is_file():
                            if process.poll() is None:
                                continue
                            trial.set_user_attr("outcome", FAILED_PROCESS)
                            state.update(
                                status=FAILED_PROCESS,
                                failure_iteration=iteration,
                                failure_reason=f"missing checkpoint mesh: {mesh_path}",
                                ended_at=time.time(),
                            )
                            atomic_write_json(state_path, state)
                            raise TrialRunError(
                                f"Metrics reached {iteration}, but checkpoint mesh is missing: {mesh_path}"
                            )
                        try:
                            geometry_rows = compute_geometry_rows(
                                run_dir=run_dir,
                                checkpoints=[MeshCheckpoint(iteration, mesh_path.resolve())],
                                ground_truth_path=ground_truth_path,
                                samples=int(spec.get("samples", 500_000)),
                                device_name="cpu",
                                seed=int(spec.get("evaluation_seed", 0)),
                                scale=float(spec.get("evaluation_scale", 1.0)),
                                use_vertices=bool(spec.get("use_vertices", True)),
                                print_each_score=True,
                            )
                        except Exception as exception:
                            failures = evaluation_failures.get(iteration, 0) + 1
                            evaluation_failures[iteration] = failures
                            if process.poll() is None and failures < 3:
                                print(
                                    f"[trial {trial.number}] checkpoint {iteration} was not "
                                    "readable yet; retrying.",
                                    flush=True,
                                )
                                continue
                            trial.set_user_attr("outcome", FAILED_PROCESS)
                            state.update(
                                status=FAILED_PROCESS,
                                failure_iteration=iteration,
                                failure_reason=f"checkpoint evaluation failed: {exception}",
                                ended_at=time.time(),
                            )
                            atomic_write_json(state_path, state)
                            raise TrialRunError(
                                f"Checkpoint evaluation failed at iteration {iteration}: "
                                f"{exception}"
                            ) from exception
                        if len(geometry_rows) != 1:
                            trial.set_user_attr("outcome", FAILED_PROCESS)
                            state.update(
                                status=FAILED_PROCESS,
                                failure_iteration=iteration,
                                failure_reason="checkpoint evaluation produced no score",
                                ended_at=time.time(),
                            )
                            atomic_write_json(state_path, state)
                            raise TrialRunError(f"No geometry score produced at iteration {iteration}.")
                        geometry_row = geometry_rows[0]
                        checkpoint_rows.append(geometry_row)
                        evaluated_iterations.add(iteration)
                        write_dict_csv(
                            run_dir / "evaluation" / "adaptive_checkpoint_metrics.csv",
                            checkpoint_rows,
                        )
                        cd = float(geometry_row["cd"])
                        if not math.isfinite(cd):
                            trial.set_user_attr("outcome", FAILED_PROCESS)
                            state.update(status=FAILED_PROCESS, failure_reason="non-finite geometry score",
                                         ended_at=time.time())
                            atomic_write_json(state_path, state)
                            raise TrialRunError(f"Non-finite geometry score at iteration {iteration}.")
                        trial.report(cd, step=iteration)
                        trial.set_user_attr(f"cd_{iteration}", cd)
                        state["checkpoint_metrics"] = checkpoint_rows
                        atomic_write_json(state_path, state)
                        if iteration < evaluation_iterations[-1] and trial.should_prune():
                            stop_trial_process(process)
                            trial.set_user_attr("outcome", PRUNED_CHAMFER)
                            trial.set_user_attr("max_observed_points", max_observed_points)
                            state.update(status=PRUNED_CHAMFER, ended_at=time.time())
                            atomic_write_json(state_path, state)
                            raise optuna.TrialPruned(
                                f"CD {cd:.6g} at iteration {iteration} is not competitive."
                            )

                    return_code = process.poll()
                    if return_code is not None:
                        if return_code != 0:
                            trial.set_user_attr("outcome", FAILED_PROCESS)
                            state.update(
                                status=FAILED_PROCESS,
                                return_code=return_code,
                                ended_at=time.time(),
                            )
                            atomic_write_json(state_path, state)
                            raise TrialRunError(
                                f"main.py exited with {return_code}; see {log_path}"
                            )
                        break
                    time.sleep(max(0.1, poll_seconds))
            except BaseException as exception:
                if state["status"] == "RUNNING":
                    state.update(status="INTERRUPTED", failure_reason=str(exception), ended_at=time.time())
                    atomic_write_json(state_path, state)
                raise
            finally:
                if process.poll() is None:
                    stop_trial_process(process)

        if evaluation_iterations[-1] not in evaluated_iterations:
            trial.set_user_attr("outcome", FAILED_PROCESS)
            state.update(
                status=FAILED_PROCESS,
                failure_iteration=evaluation_iterations[-1],
                failure_reason="final checkpoint was not evaluated",
                ended_at=time.time(),
            )
            atomic_write_json(state_path, state)
            raise TrialRunError(
                f"Trial completed without a score at iteration {evaluation_iterations[-1]}."
            )

        diagnostics = metrics_diagnostics(
            latest_metrics_rows,
            window_iterations=int(stability["window_iterations"]),
        )
        final_cd = float(checkpoint_rows[-1]["cd"])
        best_cd = min(float(row["cd"]) for row in checkpoint_rows)
        trial.set_user_attr("max_observed_points", max_observed_points)
        trial.set_user_attr("best_cd", best_cd)
        trial.set_user_attr("final_cd", final_cd)
        if diagnostics.get("point_growth_fraction") is not None:
            trial.set_user_attr(
                "point_growth_fraction",
                diagnostics["point_growth_fraction"],
            )
        if diagnostics.get("densification_new_points") is not None:
            trial.set_user_attr(
                "densification_new_points",
                diagnostics["densification_new_points"],
            )

        if not point_count_is_stable(diagnostics, stability):
            outcome = FAILED_POINT_UNSTABLE
            repaired = enqueue_repaired_trial(
                study,
                trial,
                sampled_parameters,
                search_space,
                outcome,
                bool(spec.get("enqueue_repairs", True)),
            )
            penalty = trial_penalty(spec, max_observed_points)
            relative_growth = float(diagnostics.get("point_growth_fraction", math.inf))
            trial.set_user_attr("outcome", outcome)
            trial.set_user_attr(
                "constraint_values",
                [0.0, point_stability_violation(diagnostics, stability)],
            )
            state.update(
                status=outcome,
                diagnostics=diagnostics,
                repaired_parameters=repaired,
                objective=penalty,
                ended_at=time.time(),
            )
            atomic_write_json(state_path, state)
            print(
                f"[trial {trial.number}] {outcome}: final-window point growth "
                f"was {relative_growth:.2%}; queued repair={repaired is not None}",
                flush=True,
            )
            return penalty

        trial.set_user_attr("outcome", COMPLETED)
        trial.set_user_attr("constraint_values", [0.0, 0.0])
        state.update(
            status=COMPLETED,
            diagnostics=diagnostics,
            objective=final_cd,
            best_cd=best_cd,
            ended_at=time.time(),
        )
        atomic_write_json(state_path, state)
        if run_dir.is_dir():
            shutil.copy2(log_path, run_dir / "train.log")
        print(
            f"[trial {trial.number}] complete: final CD={final_cd:.6g}, "
            f"best CD={best_cd:.6g}, max points={max_observed_points}",
            flush=True,
        )
        return final_cd

    return objective


def build_sampler(optuna: Any, spec: dict[str, Any]) -> Any:
    sampler_args: dict[str, Any] = {
        "seed": int(spec.get("seed", 0)),
        "n_startup_trials": int(spec.get("sampler_startup_trials", 3)),
        "multivariate": True,
        "n_ei_candidates": int(spec.get("sampler_ei_candidates", 24)),
    }

    def constraints_func(frozen_trial: Any) -> tuple[float, ...]:
        values = frozen_trial.user_attrs.get("constraint_values", [0.0, 0.0])
        return tuple(float(value) for value in values)

    sampler_args["constraints_func"] = constraints_func
    return optuna.samplers.TPESampler(**sampler_args)


def optimization_config_from_run(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Baseline run has no run_config.json: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        run_config = json.load(config_file)
    optimization_config = run_config.get("optimization_config", run_config)
    if not isinstance(optimization_config, dict):
        raise TypeError(f"Invalid optimization config in {config_path}")
    return optimization_config


def configuration_values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1.0e-12, abs_tol=1.0e-15)
    return expected == actual


def optuna_distribution(optuna: Any, dimension: dict[str, Any]) -> Any:
    dimension_type = dimension["type"]
    if dimension_type == "categorical":
        return optuna.distributions.CategoricalDistribution(dimension["choices"])
    if dimension_type == "int":
        return optuna.distributions.IntDistribution(
            low=int(dimension["low"]),
            high=int(dimension["high"]),
            step=int(dimension.get("step", 1)),
            log=bool(dimension.get("log", False)),
        )
    return optuna.distributions.FloatDistribution(
        low=float(dimension["low"]),
        high=float(dimension["high"]),
        step=dimension.get("step"),
        log=bool(dimension.get("log", False)),
    )


def import_baseline_run(
    optuna: Any,
    study: Any,
    spec: dict[str, Any],
    output_root: Path,
    ground_truth_path: Path,
) -> None:
    baseline_value = spec.get("baseline_run")
    if not baseline_value or study.trials:
        return

    baseline_run = resolve_path(baseline_value)
    if bool(spec.get("require_baseline_complete", True)) and not (
        baseline_run / "points_final.ply"
    ).is_file():
        raise RuntimeError(
            f"Baseline run is still incomplete: {baseline_run}. Wait for points_final.ply "
            "before starting the single-worker adaptive study."
        )
    optimization_config = optimization_config_from_run(baseline_run)
    sampled_parameters: dict[str, Any] = {}
    for name, dimension in spec["search_space"].items():
        if name not in optimization_config:
            raise KeyError(f"Baseline run config is missing searched parameter '{name}'.")
        value = optimization_config[name]
        if not dimension_contains(dimension, value):
            raise ValueError(
                f"Baseline value {name}={value!r} is outside the adaptive search space."
            )
        sampled_parameters[name] = value

    expected_parameters = apply_linked_parameters(
        dict(spec["base_args"], **sampled_parameters),
        spec,
    )
    operational_parameters = {
        "iterations",
        "log_interval",
        "save_interval",
        "save_ply_files_interval",
        "save_final_mesh",
        "ground_truth",
        "geometry_samples",
        "geometry_seed",
        "geometry_scale",
        "geometry_use_vertices",
        "enable_metrics",
        "enable_image_preview",
    }
    mismatches = [
        name
        for name, expected in expected_parameters.items()
        if name not in operational_parameters
        and (
            name not in optimization_config
            or not configuration_values_match(expected, optimization_config[name])
        )
    ]
    if mismatches:
        details = ", ".join(
            f"{name}: expected={expected_parameters[name]!r}, "
            f"actual={optimization_config.get(name)!r}"
            for name in mismatches
        )
        raise ValueError(f"Baseline run is not study-compatible ({details}).")

    evaluation_iterations = [int(value) for value in spec["evaluation_iterations"]]
    metrics_rows = [
        row
        for row in finite_metric_rows(read_csv_dicts(baseline_run / "metrics.csv"))
        if metric_iteration(row) <= evaluation_iterations[-1]
    ]
    if not metrics_rows or metric_iteration(metrics_rows[-1]) < evaluation_iterations[-1]:
        raise RuntimeError(
            f"Baseline run has not reached iteration {evaluation_iterations[-1]}. "
            "Let it reach the final study iteration before starting adaptive search."
        )

    checkpoints: list[MeshCheckpoint] = []
    reconstruction_name = str(spec.get("reconstruction_name", "fuse_post.ply"))
    for iteration in evaluation_iterations:
        mesh_path = (
            baseline_run
            / "mesh_checkpoints"
            / f"iter_{iteration:05d}"
            / reconstruction_name
        )
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Baseline checkpoint mesh is missing: {mesh_path}")
        checkpoints.append(MeshCheckpoint(iteration, mesh_path.resolve()))

    geometry_rows = compute_geometry_rows(
        run_dir=baseline_run,
        checkpoints=checkpoints,
        ground_truth_path=ground_truth_path,
        samples=int(spec.get("samples", 500_000)),
        device_name="cpu",
        seed=int(spec.get("evaluation_seed", 0)),
        scale=float(spec.get("evaluation_scale", 1.0)),
        use_vertices=bool(spec.get("use_vertices", True)),
        print_each_score=True,
    )
    write_dict_csv(
        output_root / "_study" / "baseline_checkpoint_metrics.csv",
        geometry_rows,
    )
    intermediate_values = {
        int(row["iteration"]): float(row["cd"]) for row in geometry_rows
    }
    max_observed_points = max(
        int(safe_float(row.get("num_points")) or 0) for row in metrics_rows
    )
    diagnostics = metrics_diagnostics(
        metrics_rows,
        int(spec["guardrails"]["point_stability"]["window_iterations"]),
    )
    max_points = int(spec["guardrails"]["max_points"])
    stable = point_count_is_stable(
        diagnostics,
        spec["guardrails"]["point_stability"],
    )
    feasible = max_observed_points <= max_points and stable
    final_cd = float(geometry_rows[-1]["cd"])
    value = final_cd if feasible else trial_penalty(spec, max_observed_points)
    outcome = "IMPORTED_BASELINE" if feasible else (
        FAILED_POINT_CAP if max_observed_points > max_points else FAILED_POINT_UNSTABLE
    )
    relative_growth = float(diagnostics.get("point_growth_fraction", 1.0))
    user_attrs = {
        "outcome": outcome,
        "run_dir": str(baseline_run),
        "max_observed_points": max_observed_points,
        "best_cd": min(float(row["cd"]) for row in geometry_rows),
        "final_cd": final_cd,
        "point_growth_fraction": relative_growth,
        "densification_new_points": diagnostics.get("densification_new_points"),
        "constraint_values": [
            max(0.0, (max_observed_points - max_points) / max_points),
            point_stability_violation(
                diagnostics,
                spec["guardrails"]["point_stability"],
            ),
        ],
    }
    distributions = {
        name: optuna_distribution(optuna, dimension)
        for name, dimension in spec["search_space"].items()
    }
    study.add_trial(
        optuna.trial.create_trial(
            params=sampled_parameters,
            distributions=distributions,
            value=value,
            intermediate_values=intermediate_values,
            user_attrs=user_attrs,
        )
    )
    print(
        f"Imported baseline {baseline_run}: final CD={final_cd:.6g}, "
        f"max points={max_observed_points}, outcome={outcome}",
        flush=True,
    )


def run_study(spec_path: Path, spec: dict[str, Any], args: argparse.Namespace) -> None:
    study_dir = resolve_path(spec["output_root"]) / "_study"
    with exclusive_study_lock(study_dir / "worker.lock"):
        _run_study_locked(spec_path, spec, args)


def _run_study_locked(spec_path: Path, spec: dict[str, Any], args: argparse.Namespace) -> None:
    verify_config_snapshot(spec)
    optuna = require_optuna()
    output_root = resolve_path(spec["output_root"])
    dataset_path = resolve_path(spec["dataset_path"])
    ground_truth_path = resolve_path(spec["ground_truth"])
    study_dir = output_root / "_study"
    study_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = study_dir / "study_spec.snapshot.json"
    current_spec_digest = spec_digest(spec)
    previous_snapshot = read_json_object(snapshot_path)
    if (
        previous_snapshot is not None
        and previous_snapshot.get("sha256") != current_spec_digest
    ):
        raise RuntimeError(
            "The adaptive-search spec changed since this study was created. "
            f"Use a new output_root/study_name instead of mixing search spaces in {study_dir}."
        )
    atomic_write_json(
        snapshot_path,
        {"source": str(spec_path), "sha256": current_spec_digest, "spec": spec},
    )

    storage_path = study_dir / "study.sqlite3"
    storage_url = f"sqlite:///{storage_path}"
    storage_options: dict[str, Any] = dict(
        url=storage_url,
        heartbeat_interval=60,
        grace_period=180,
    )
    # The heartbeat callback names changed in Optuna 4.9.
    if "heartbeat_stale_trial_callback" in inspect.signature(optuna.storages.RDBStorage).parameters:
        storage_options["heartbeat_stale_trial_callback"] = (
            optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=1)
        )
    else:
        storage_options["failed_trial_callback"] = optuna.storages.RetryFailedTrialCallback(max_retry=1)
    storage = optuna.storages.RDBStorage(**storage_options)
    sampler = build_sampler(optuna, spec)
    pruner = optuna.pruners.PercentilePruner(
        percentile=float(spec.get("pruner_percentile", 50.0)),
        n_startup_trials=int(spec.get("pruner_startup_trials", 3)),
        n_warmup_steps=int(spec.get("pruner_warmup_iteration", 2_000)),
        interval_steps=int(spec.get("pruner_interval_iterations", 1_000)),
        n_min_trials=int(spec.get("pruner_min_trials", 2)),
    )
    study = optuna.create_study(
        study_name=str(spec.get("study_name", "adaptive_chamfer")),
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    import_baseline_run(
        optuna=optuna,
        study=study,
        spec=spec,
        output_root=output_root,
        ground_truth_path=ground_truth_path,
    )
    enqueue_initial_trials(study, spec)

    target_trials = int(args.max_trials or spec.get("max_trials", 18))
    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    terminal_count = sum(trial.state in terminal_states for trial in study.trials)
    remaining_trials = max(0, target_trials - terminal_count)
    summary_path = study_dir / "study_summary.csv"

    def export_callback(completed_study: Any, _trial: Any) -> None:
        export_study_summary(
            completed_study,
            summary_path,
            list(spec["search_space"].keys()),
        )
        export_confirmation_summary(completed_study, spec, study_dir)
        # Infrastructure failures should not consume the whole unattended budget.
        limit = int(spec.get("max_consecutive_process_failures", 0))
        trials = [t for t in completed_study.trials if t.state.is_finished()]
        if limit and len(trials) >= limit and all(
            t.state == optuna.trial.TrialState.FAIL for t in trials[-limit:]
        ):
            raise RuntimeError(f"Stopping study after {limit} consecutive failed processes; inspect trial logs.")

    timeout_hours = getattr(args, "timeout_hours", None)
    if timeout_hours is None:
        timeout_hours = spec.get("timeout_hours")
    remaining_seconds = remaining_search_seconds(study_dir, timeout_hours)
    if remaining_trials and (remaining_seconds is None or remaining_seconds > 0):
        objective = make_objective(
            optuna=optuna, study=study, spec=spec, output_root=output_root,
            dataset_path=dataset_path, ground_truth_path=ground_truth_path,
            poll_seconds=args.poll_seconds,
        )
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=remaining_seconds,
            catch=(TrialRunError,),
            callbacks=[export_callback],
            gc_after_trial=True,
        )
    export_callback(study, None)
    print(f"Study summary: {summary_path}")

    # Freeze the finalists once so a restart resumes the same confirmation study.
    confirmation_path = study_dir / "confirmation_spec.json"
    confirmation_spec = read_json_object(confirmation_path)
    if confirmation_spec is None:
        confirmation_spec = make_confirmation_spec(study, spec)
        if confirmation_spec is not None:
            atomic_write_json(confirmation_path, confirmation_spec)
    if confirmation_spec is not None:
        validate_spec(confirmation_spec)
        run_study(confirmation_path, confirmation_spec, argparse.Namespace(
            max_trials=None, timeout_hours=None, poll_seconds=args.poll_seconds,
        ))


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0.0:
        raise SystemExit("--poll-seconds must be positive.")
    if args.max_trials is not None and args.max_trials <= 0:
        raise SystemExit("--max-trials must be positive.")
    if args.timeout_hours is not None and (not math.isfinite(args.timeout_hours) or args.timeout_hours <= 0):
        raise SystemExit("--timeout-hours must be finite and positive.")
    spec_path, spec = load_spec(args.spec)
    validate_spec(spec)
    verify_config_snapshot(spec)
    print(f"Valid adaptive-search spec: {spec_path}")
    if args.validate_only:
        return
    if args.dry_run:
        sampled = {
            name: representative_value(dimension)
            for name, dimension in spec["search_space"].items()
        }
        parameters = apply_linked_parameters(dict(spec["base_args"], **sampled), spec)
        run_dir = resolve_path(spec["output_root"]) / f"dry_run_{parameter_digest(parameters)}"
        command = build_train_command(resolve_path(spec["dataset_path"]), run_dir, parameters)
        print(" ".join(command))
        return
    run_study(spec_path, spec, args)


if __name__ == "__main__":
    main()
