#!/usr/bin/env python3
"""Refresh Workbench accuracy, completion and CD curves for paper and Crowd #1/#3.

From the repository root, in the DifferentiablePointRendering conda environment:
    python python/compare_workbench_cd.py

Scans saved mesh checkpoints every time; no extraction or training is launched.
Defaults: every 1K iterations, 5M surface samples, seed 0, distances x100.
Writes cd_curves.png, cd_curves.pdf, results.json and results.csv beneath
output/workbench_cd_comparison/. Content-based caching handles replaced runs,
new checkpoints and edited GT meshes. Opens one Matplotlib window with three
plots. Use --no-show for a headless run or --refresh to bypass cached metrics.
Run-local geometry_metrics.csv files are ignored: their GT may be incorrect.
Cached evaluations are reused only for matching mesh, GT and metric settings.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
METRIC_FILE = ROOT / "metrics/chamfer_ours.py"
SCALE = 100.0


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cache_key(mesh_hash: str, gt_hash: str, samples: int, seed: int, metric_hash: str) -> str:
    protocol = [mesh_hash, gt_hash, samples, seed, SCALE, metric_hash]
    return hashlib.sha256(json.dumps(protocol).encode()).hexdigest()


def valid_metrics(values: dict) -> bool:
    try:
        a, c, cd = [float(values[k]) for k in ("accuracy", "completion", "cd")]
        return all(math.isfinite(v) and v >= 0 for v in (a, c, cd)) and math.isclose(
            cd, (a + c) / 2, rel_tol=1e-12, abs_tol=1e-12)
    except (KeyError, TypeError, ValueError):
        return False


def discover_checkpoints(run: Path, step: int, maximum: int | None) -> tuple[list[dict], list[dict]]:
    """Use extraction.json as the completion marker; ignore files still being written."""
    selected, skipped = [], []
    for folder in sorted((run / "mesh_checkpoints").glob("iter_*")):
        match = re.fullmatch(r"iter_(\d+)", folder.name)
        if not match:
            continue
        iteration = int(match[1])
        if iteration <= 0 or iteration % step or (maximum is not None and iteration > maximum):
            continue
        mesh, metadata = folder / "fuse_post.ply", folder / "extraction.json"
        try:
            extraction = json.loads(metadata.read_text())
            stat = mesh.stat()
            if stat.st_size == 0 or extraction.get("post_triangles", 0) <= 0:
                raise ValueError("empty or incomplete extraction")
            if stat.st_mtime_ns > metadata.stat().st_mtime_ns:
                raise ValueError("mesh is newer than its extraction completion marker")
            fingerprint = file_hash(mesh)
            after = mesh.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("mesh changed during inspection")
            selected.append(dict(iteration=iteration, reconstruction=str(mesh.resolve()),
                                 reconstruction_sha256=fingerprint, extraction=extraction))
        except (OSError, ValueError, TypeError) as error:
            skipped.append(dict(iteration=iteration, path=str(mesh), reason=str(error)))
    return selected, skipped


def legacy_metrics(metric_hash: str) -> dict[tuple, dict]:
    """Import earlier evaluations by mesh and GT hashes, with unchanged metric code."""
    audit = ROOT / "../output/workbench_gt_reevaluation_20261009/manifest.json"
    previous = ROOT / "../output/workbench_cd_curves_20261009/results.json"
    try:
        code = json.loads(audit.read_text())["code_sha256"]["metrics/chamfer_ours.py"]
        old = json.loads(previous.read_text())
        if code != metric_hash or old["output_scale"] != SCALE:
            return {}
        return {(r["reconstruction_sha256"], old["ground_truth_sha256"], old["samples_per_mesh"], old["seed"]):
                {k: r[k] for k in ("accuracy", "completion", "cd")}
                for r in old["results"] if valid_metrics(r)}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def plot(report: dict, output: Path, *, show: bool = True) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output / "cache/matplotlib"))
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True)
    fig.canvas.manager.set_window_title("Workbench: Accuracy, Completion and CD")
    fig.subplots_adjust(left=.055, right=.98, top=.73, bottom=.23, wspace=.25)
    fig.text(.055, .95, "Workbench: geometry accuracy during training", fontsize=18, weight="bold")
    fig.text(.055, .90, f"Saved checkpoints every {report['step']:,} iterations | Lower is better",
             fontsize=11, color="#526171")
    colors = {"paper": "#2563a6", "crowd1": "#d66b20", "latest": "#16836b"}
    metrics = [("accuracy", "Accuracy", "Reconstruction to GT"),
               ("completion", "Completion", "GT to reconstruction"),
               ("cd", "Chamfer distance (CD)", "Mean of accuracy and completion")]
    for run in report["runs"]:
        rows = sorted((r for r in report["results"] if r["method"] == run["method"]), key=lambda r: r["iteration"])
        if not rows:
            continue
        iterations = list(range(rows[0]["iteration"], rows[-1]["iteration"] + 1, report["step"]))
        for ax, (metric, _, _) in zip(axes, metrics):
            by_iteration = {r["iteration"]: r[metric] for r in rows}
            # Missing saved checkpoints break the line rather than inventing intermediate values.
            ax.plot([n / 1000 for n in iterations], [by_iteration.get(n, math.nan) for n in iterations],
                    color=colors[run["method"]], lw=2.2, marker="o", markersize=3.5,
                    label=f"{run['label']} (through {rows[-1]['iteration'] / 1000:g}K)")
    maximum = max(r["iteration"] for r in report["results"]) / 1000
    for ax, (_, title, direction) in zip(axes, metrics):
        ax.set_title(f"{title}\n{direction}", fontsize=12, pad=12)
        ax.set_xlim(.5, max(2, maximum + .5))
        ax.xaxis.set_major_locator(MultipleLocator(max(1, math.ceil(maximum / 10))))
        ax.set_xlabel("Training iteration (thousands)", labelpad=10)
        ax.set_ylabel("Distance ×100", labelpad=8)
        ax.grid(axis="y", color="#dce2e8", linewidth=.7)
        ax.grid(axis="x", color="#eef1f4", linewidth=.5)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .86),
               ncol=len(labels), frameon=False, fontsize=10)
    fig.text(.055, .12, f"{report['samples_per_mesh']:,} samples/mesh; seed {report['seed']}; unsquared point-to-triangle distances. CD = (Accuracy + Completion) / 2.",
             fontsize=9, color="#526171")
    fig.text(.055, .075, "Original saved mesh checkpoints; no alignment. GT SHA-256: " + report["ground_truth_sha256"][:16],
             fontsize=9, color="#526171")
    fig.text(.055, .035, "Updated: " + report["snapshot"] + " | " + report["ground_truth"], fontsize=8, color="#526171")
    for extension in ("png", "pdf"):
        target = output / f"cd_curves.{extension}"
        temporary = target.with_suffix(f".{extension}.tmp")
        fig.savefig(temporary, format=extension, dpi=180, facecolor="white")
        temporary.replace(target)
    if show:
        print("Saved plots. Opening the Matplotlib window; close it to finish.", flush=True)
        plt.show(block=True)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "OptimizationOutput"
    parser.add_argument("--paper-run", type=Path, default=base / "paper/workbench")
    parser.add_argument("--crowd1-run", type=Path, default=base / "work/workbench_crowd_penalty")
    parser.add_argument("--latest-run", type=Path, default=base / "work/workbench_crowd_penalty_8")
    parser.add_argument("--gt", type=Path, default=Path.home() / "phd/models/gt/workbench.ply")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "../output/workbench_cd_comparison")
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--max-iteration", type=int)
    parser.add_argument("--samples", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="Recompute all selected metrics, ignoring caches")
    parser.add_argument("--no-show", action="store_true", help="Save the three plots without opening a Matplotlib window")
    parser.add_argument("--list-only", action="store_true", help="Inspect available checkpoints without evaluating or writing files")
    args = parser.parse_args(argv)
    if args.step <= 0 or args.samples <= 0 or args.seed < 0 or (args.max_iteration is not None and args.max_iteration <= 0):
        parser.error("step, samples and max-iteration must be positive; seed must be nonnegative")

    runs, selected, skipped = [], [], []
    for key, label, path in [("paper", "Paper default", args.paper_run),
                             ("crowd1", "Crowd penalty #1", args.crowd1_run),
                             ("latest", "Crowd penalty latest", args.latest_run)]:
        path = path.expanduser().resolve()
        entries, ignored = discover_checkpoints(path, args.step, args.max_iteration)
        runs.append(dict(method=key, label=label, path=str(path)))
        selected.extend(dict(entry, method=key, label=label, source_run=str(path)) for entry in entries)
        skipped.extend(dict(entry, method=key) for entry in ignored)
        print(f"{label}: {len(entries)} checkpoints; iterations {[e['iteration'] for e in entries]}", flush=True)
        if not entries:
            skipped.append(dict(method=key, path=str(path), reason="No completed matching checkpoints"))
    if args.list_only:
        for entry in skipped:
            print("Skipped:", entry)
        return 0 if selected else 1
    if not selected:
        parser.error("No completed mesh checkpoints found")

    gt = args.gt.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    gt_hash, metric_hash = file_hash(gt), file_hash(METRIC_FILE)
    frozen_gt = output / "cache/ground_truth" / (gt_hash + ".ply")
    frozen_gt.parent.mkdir(parents=True, exist_ok=True)
    if not frozen_gt.exists() or file_hash(frozen_gt) != gt_hash:
        shutil.copy2(gt, frozen_gt)
    if file_hash(frozen_gt) != gt_hash or file_hash(gt) != gt_hash:
        raise RuntimeError("GT changed while taking its snapshot; rerun the script")
    report = dict(snapshot=datetime.now(ZoneInfo("Europe/Copenhagen")).isoformat(timespec="seconds"),
                  ground_truth=str(gt), ground_truth_snapshot=str(frozen_gt), ground_truth_sha256=gt_hash,
                  metric_code_sha256=metric_hash, samples_per_mesh=args.samples, seed=args.seed, output_scale=SCALE,
                  step=args.step, runs=runs, skipped=skipped, results=[])
    imported = {} if args.refresh else legacy_metrics(metric_hash)
    gt_mesh = gt_points = None
    computed = reused = 0
    sys.path.insert(0, str(METRIC_FILE.parent))
    for index, entry in enumerate(selected, 1):
        mesh = Path(entry["reconstruction"])
        key = cache_key(entry["reconstruction_sha256"], gt_hash, args.samples, args.seed, metric_hash)
        cached = output / "cache/metrics" / (key + ".json")
        values = None
        if not args.refresh and cached.exists():
            try:
                record = json.loads(cached.read_text())
                if record.get("cache_key") == key and valid_metrics(record):
                    values = {k: record[k] for k in ("accuracy", "completion", "cd")}
            except (OSError, ValueError):
                pass
        if values is None:
            values = imported.get((entry["reconstruction_sha256"], gt_hash, args.samples, args.seed))
        was_cached = values is not None
        if values is None:
            from chamfer_ours import (set_random_seed, load_triangle_mesh_with_query_points,
                                     compute_paper_ready_point_to_triangle_distance)
            if gt_mesh is None:
                set_random_seed(args.seed)
                gt_mesh, gt_points, _ = load_triangle_mesh_with_query_points(frozen_gt, args.samples, False)
            print(f"Evaluating {entry['label']} at {entry['iteration']:,}...", flush=True)
            set_random_seed(args.seed)
            reconstruction, points, _ = load_triangle_mesh_with_query_points(mesh, args.samples, False)
            measured = compute_paper_ready_point_to_triangle_distance(points, reconstruction, gt_points, gt_mesh, scale=SCALE)
            values = {k: float(measured[k]) for k in ("accuracy", "completion", "cd")}
            del reconstruction, points
            gc.collect()
        if file_hash(mesh) != entry["reconstruction_sha256"]:
            report["skipped"].append(dict(method=entry["method"], iteration=entry["iteration"], reason="Mesh changed during evaluation; rerun"))
            continue
        if not valid_metrics(values):
            raise RuntimeError(f"Invalid metrics for {mesh}")
        write_json(cached, dict(values, cache_key=key, ground_truth_sha256=gt_hash,
                               reconstruction_sha256=entry["reconstruction_sha256"], samples_per_mesh=args.samples,
                               seed=args.seed, output_scale=SCALE, metric_code_sha256=metric_hash))
        report["results"].append(dict(entry, **values, cache_hit=was_cached))
        reused += int(was_cached)
        computed += int(not was_cached)
        print(f"[{index}/{len(selected)}] {entry['label']} {entry['iteration']:,}: CD={values['cd']:.6f}"
              + (" (cached)" if was_cached else ""), flush=True)
    if file_hash(gt) != gt_hash:
        raise RuntimeError("GT changed during evaluation; caches are preserved. Rerun before publishing the graph")
    if not report["results"]:
        raise RuntimeError("No stable mesh checkpoints could be evaluated")
    report.update(computed=computed, reused=reused)
    write_json(output / "results.json", report)
    temporary_csv = output / "results.csv.tmp"
    with temporary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "iteration", "accuracy", "completion", "cd", "reconstruction"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])
    temporary_csv.replace(output / "results.csv")
    plot(report, output, show=not args.no_show)
    print(f"Done: {computed} evaluated, {reused} cached. Output: {output}")
    for entry in report["skipped"]:
        print("Skipped:", entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
