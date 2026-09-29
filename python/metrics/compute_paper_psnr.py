#!/usr/bin/env python3
"""Evaluate saved RGB renders on every camera and print paper-ready LaTeX.

CPU only; requires numpy and Pillow. See PAPER_PSNR.md for the evaluation protocol.
Training outputs are read-only. Missing/incomplete scenes get --, never a partial mean.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

SCENES = ("teapot", "horse", "plant", "dragon", "workbench", "restaurant")
METHODS = ("2dgs", "pgsr", "radiosity_gs", "ours")
LABELS = {"2dgs": "2DGS", "pgsr": "PGSR", "radiosity_gs": "RadiosityGS", "ours": r"\textbf{Ours}"}
DEFAULT_ROOTS = {
    "ours": Path(__file__).resolve().parents[1] / "OptimizationOutput/paper",
    "2dgs": Path.home() / "projects/2D-GS-Viser-Viewer/output/batch_2dgs",
    "pgsr": Path.home() / "phd/pbdr/PGSR/output/batch_pgsr",
    "radiosity_gs": Path.home() / "phd/pbdr/RadiosityGS/output/batch_radiosity_gs",
}


def read_json(path):
    return json.loads(path.read_text())


def read_config(path):
    """Parse Namespace(...) without executing the saved Python expression."""
    node = ast.parse(path.read_text().strip(), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Namespace" or node.args:
        raise ValueError(f"Expected Namespace keyword arguments: {path}")
    return {item.arg: ast.literal_eval(item.value) for item in node.keywords}


def load_rgb(path):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode not in ("RGB", "RGBA"):
            raise ValueError(f"Expected lossless RGB/RGBA PNG: {path} ({image.format}, {image.mode})")
        array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"Expected 8-bit image: {path}")
    if array.shape[2] == 4 and np.any(array[..., 3] != 255):
        raise ValueError(f"Non-opaque alpha requires an explicit compositing policy: {path}")
    return array[..., :3].astype(np.float64) / 255.0


def convert_color(image, source, destination):
    if source == destination:
        return image
    if source == "linear":
        return np.where(image <= 0.0031308, 12.92 * image, 1.055 * image ** (1 / 2.4) - 0.055)
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def psnr(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(f"Image dimensions differ: {prediction.shape} versus {target.shape}; no resizing is performed")
    mse = float(np.mean((prediction - target) ** 2))
    if not math.isfinite(mse):
        raise ValueError("Non-finite image values")
    return (math.inf if mse == 0 else -10 * math.log10(mse)), mse


def canonical_frames(dataset_root, scene):
    data = read_json(dataset_root / f"{scene}_pbdr/transforms.json")
    frames = data["frames"]
    if not frames or len({f["camera_name"] for f in frames}) != len(frames):
        raise ValueError("Empty or duplicate canonical camera names")
    return frames


def match_camera(c2w, fx, fy, width, height, frames, name=None):
    """Match poses, not truncated names (Restaurant contains Camera.001 etc.)."""
    matches = []
    for frame in frames:
        if name is not None and frame["camera_name"] != name:
            continue
        expected = np.array(frame["transform_matrix"], dtype=float)
        expected[:3, 1:3] *= -1  # Blender to COLMAP camera coordinates.
        if (np.allclose(c2w, expected, atol=1e-4, rtol=1e-5)
                and np.allclose([fx, fy], [frame["fl_x"], frame["fl_y"]], atol=1e-3, rtol=1e-5)
                and (width, height) == (frame["w"], frame["h"])):
            matches.append(frame)
    if len(matches) != 1:
        raise ValueError(f"Camera pose/intrinsics match {len(matches)} canonical views; expected exactly one")
    return matches[0]


def colmap_image_names(source):
    """Reproduce the local 2DGS loader's stable ordering, preserving dotted names."""
    sparse = source / "sparse/0"
    if (sparse / "images.bin").exists():
        raise ValueError("This adapter expects the paper datasets' COLMAP images.txt, not images.bin")
    names = []
    with (sparse / "images.txt").open() as stream:
        for line in stream:
            if line.strip() and not line.lstrip().startswith("#"):
                fields = line.split()
                if len(fields) < 10:
                    raise ValueError("Invalid COLMAP image record")
                names.append(Path(fields[9]).name)
                if stream.readline() == "":
                    raise ValueError("Missing COLMAP POINTS2D line")
    return sorted(names, key=lambda name: name.split(".")[0])


def require_complete(pairs, frames):
    names = [p["camera"] for p in pairs]
    expected = {f["camera_name"] for f in frames}
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError(f"Incomplete/duplicate camera coverage: {len(names)} renders, {len(expected)} expected; "
                         f"missing={sorted(expected - set(names))}, extra={sorted(set(names) - expected)}")
    for pair in pairs:
        for key in ("render", "native_target"):
            if not pair[key].is_file():
                raise FileNotFoundError(f"Missing {key}: {pair[key]}")
    return pairs


def discover(method, scene, roots, frames, iteration):
    suffix = "" if method == "ours" else "_pbdr" if method == "radiosity_gs" else "_2dgs"
    run = roots[method] / (scene + suffix)
    if method == "ours":
        if not (run / "points_final.ply").is_file():
            raise FileNotFoundError(f"Missing final checkpoint: {run / 'points_final.ply'}")
        pairs = [dict(camera=p.stem.removeprefix("render_final_"), render=p,
                      native_target=p.with_name(p.name.replace("render_final_", "render_target_", 1)))
                 for p in sorted((run / "renders").glob("render_final_*.png"))]
        return require_complete(pairs, frames), "final"

    cfg = read_config(run / "cfg_args")
    if cfg.get("eval", False):
        raise ValueError("Run has a train/test split; this all-camera evaluator requires eval=False")
    available = [int(p.name.removeprefix("iteration_")) for p in (run / "point_cloud").glob("iteration_*")
                 if p.name.removeprefix("iteration_").isdigit() and (p / "point_cloud.ply").is_file()]
    selected = iteration if iteration is not None else max(available, default=None)
    if selected not in available or selected is None:
        raise FileNotFoundError(f"No {'requested' if iteration else 'trained'} checkpoint in {run / 'point_cloud'}")
    directory = run / "train" / f"ours_{selected}"
    pairs = []
    if method == "pgsr":
        manifest = read_json(directory / "render_views.json")
        if manifest.get("pass") != "render" or manifest["expected_views"] != len(manifest["views"]):
            raise ValueError("Incomplete PGSR render manifest (or fusion pass instead of RGB render)")
        for view in manifest["views"]:
            intrinsic = view["intrinsic"]
            source = Path(view["source_image_path"])
            frame = match_camera(np.linalg.inv(np.array(view["world_to_camera"])), intrinsic[0][0], intrinsic[1][1],
                                 view["width"], view["height"], frames, source.stem)
            # Preserve full source filename: image_name is truncated at '.' by PGSR.
            if source.stem != frame["camera_name"]:
                raise ValueError(f"PGSR source image disagrees with pose: {source}")
            target = directory / view["files"]["ground_truth"] if "ground_truth" in view["files"] else source
            pairs.append(dict(camera=frame["camera_name"], render=directory / view["files"]["rgb"], native_target=target))
    else:
        cameras = read_json(run / "cameras.json")
        source = Path(cfg["source_path"])
        names = colmap_image_names(source) if method == "2dgs" else None
        if names is not None and len(names) != len(cameras):
            raise ValueError("COLMAP camera count differs from saved cameras.json")
        for index, camera in enumerate(cameras):
            c2w = np.eye(4)
            c2w[:3, :3] = camera["rotation"]
            c2w[:3, 3] = camera["position"]
            name = (Path(names[index]).stem if names is not None else
                    (source / "images" / (camera["img_name"] + ".exr")).resolve().stem)
            frame = match_camera(c2w, camera["fx"], camera["fy"], camera["width"], camera["height"], frames, name)
            if method == "2dgs":
                _, target_mse = psnr(load_rgb(directory / f"gt/{index:05d}.png"),
                                     load_rgb(source / cfg.get("images", "images") / names[index]))
                if target_mse > (1 / 255) ** 2:
                    raise ValueError(f"Saved GT disagrees with source image for {name}; check camera ordering")
            pairs.append(dict(camera=frame["camera_name"], render=directory / f"renders/{index:05d}.png",
                              native_target=directory / f"gt/{index:05d}.png"))
        for subdir in ("renders", "gt"):
            if {p.name for p in (directory / subdir).glob("*.png")} != {f"{i:05d}.png" for i in range(len(cameras))}:
                raise ValueError(f"Incomplete/unexpected files in {directory / subdir}")
    return require_complete(pairs, frames), selected


def evaluate(method, scene, pairs, iteration, ours_root, reference, color_space):
    rows = []
    source_space = "linear" if method == "radiosity_gs" else "srgb"
    for pair in pairs:
        prediction = convert_color(load_rgb(pair["render"]), source_space, color_space)
        native = convert_color(load_rgb(pair["native_target"]), source_space, color_space)
        common_path = ours_root / scene / "renders" / f"render_target_{pair['camera']}.png"
        common = convert_color(load_rgb(common_path), "srgb", color_space) if common_path.is_file() else None
        if reference == "ours" and common is None:
            raise FileNotFoundError(f"Missing shared target: {common_path}")
        target = common if reference == "ours" else native
        value, mse = psnr(prediction, target)
        target_psnr, target_mse = psnr(native, common) if common is not None else (None, None)
        rows.append(dict(method=method, scene=scene, iteration=iteration, camera=pair["camera"],
                         psnr=value, mse=mse, width=prediction.shape[1], height=prediction.shape[0],
                         render=str(pair["render"]), target=str(common_path if reference == "ours" else pair["native_target"]),
                         native_target=str(pair["native_target"]), native_color_space=source_space,
                         native_target_vs_ours_psnr=target_psnr, native_target_vs_ours_mse=target_mse))
    return rows


def latex_rows(summaries, scenes):
    lookup = {(r["method"], r["scene"]): r for r in summaries}
    lines = []
    for method in METHODS:
        cells = []
        for scene in scenes:
            value = lookup[method, scene]["mean_psnr"]
            cells.append("--" if value is None else r"$\infty$" if math.isinf(value) else f"{value:.2f}")
        lines.append(LABELS[method] + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text("")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for method, root in DEFAULT_ROOTS.items():
        parser.add_argument(f"--{method.replace('_', '-')}-root", type=Path, default=root)
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / "phd/datasets")
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--iteration", type=int, help="GS checkpoint (default: latest trained checkpoint per scene; Ours uses final)")
    parser.add_argument("--reference", choices=("ours", "native"), default="native",
                        help="Each method's own exported/training targets (default), or shared Ours targets")
    parser.add_argument("--color-space", choices=("srgb", "linear"), default="srgb")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "evaluation_results/paper_psnr")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any cell is unavailable; reports are still written")
    args = parser.parse_args(argv)
    if args.iteration is not None and args.iteration <= 0:
        parser.error("--iteration must be positive")
    roots = {m: getattr(args, m + "_root").expanduser().resolve() for m in METHODS}
    scenes = list(dict.fromkeys(args.scenes))
    summaries, per_camera = [], []
    for method in METHODS:
        for scene in scenes:
            summary = dict(method=method, scene=scene, iteration=None, status="unavailable", cameras=0,
                           mean_psnr=None, reference=args.reference, color_space=args.color_space,
                           native_target_vs_ours_mean_mse=None, error="")
            try:
                frames = canonical_frames(args.dataset_root.expanduser(), scene)
                pairs, iteration = discover(method, scene, roots, frames, args.iteration)
                rows = evaluate(method, scene, pairs, iteration, roots["ours"], args.reference, args.color_space)
                audits = [r["native_target_vs_ours_mse"] for r in rows if r["native_target_vs_ours_mse"] is not None]
                summary.update(iteration=iteration, status="complete", cameras=len(rows),
                               mean_psnr=float(np.mean([r["psnr"] for r in rows])),
                               native_target_vs_ours_mean_mse=float(np.mean(audits)) if audits else None)
                per_camera.extend(rows)
                print(f"{LABELS[method]} / {scene}: {summary['mean_psnr']:.4f} dB ({len(rows)} cameras, {iteration})", file=sys.stderr)
            except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
                summary["error"] = str(error)
                print(f"UNAVAILABLE {method} / {scene}: {error}", file=sys.stderr)
            summaries.append(summary)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_camera.csv", per_camera)
    write_csv(output / "summary.csv", summaries)
    rows_text = latex_rows(summaries, scenes)
    (output / "rows.tex").write_text(rows_text)
    caption = (f"\\textbf{{Image Reconstruction.}} Average {args.color_space} PSNR (dB) over all cameras; higher is better. "
               + ("All methods use the same reference images." if args.reference == "ours" else "Each method uses its own reference images; tone mapping differs between datasets."))
    table = ("\\begin{table}[t]\n\\centering\\small\n\\caption{" + caption + "}\n\\label{tab:views}\n"
             "\\begin{tabular}{l" + "r" * len(scenes) + "}\n\\toprule\nMethod & "
             + " & ".join(s.title() for s in scenes) + r" \\" + "\n\\midrule\n" + rows_text
             + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (output / "table.tex").write_text(table)
    notes = [f"Reference: {args.reference}; evaluation color space: {args.color_space}.",
             "PSNR = -10 log10(mean RGB squared error), peak=1, then arithmetic mean of per-camera dB.",
             "Full saved images; no resizing, fitted exposure, cropping, or additional masks.",
             "Saved 8-bit PNGs are quantized/clipped; these are not raw floating-point renderer metrics.",
             "RadiosityGS PNGs are linear RGB; other methods' PNGs are sRGB. Convert both prediction and reference before scoring.",
             "RadiosityGS's exporter already applies its GT alpha mask to renders and targets.",
             "Camera coverage is validated against dataset transforms; GS cameras are matched by pose and intrinsics.",
             "Target differences are recorded per camera and in summary.csv. Common targets do not remove training-data differences."]
    notes.append("The local 2DGS/PGSR datasets apply Blender AgX before sRGB encoding; Ours/RadiosityGS use scene-linear supervision. "
                 "sRGB decoding does not invert AgX. Neither reference option removes this training/evaluation mismatch.")
    for row in summaries:
        audit = row["native_target_vs_ours_mean_mse"]
        if audit is not None and audit > (1 / 255) ** 2:
            message = f"TARGET DIFFERENCE {row['method']}/{row['scene']}: native-vs-Ours target RMSE={math.sqrt(audit):.6f}."
            notes.append(message)
            print(message, file=sys.stderr)
        if row["error"]:
            notes.append(f"UNAVAILABLE {row['method']}/{row['scene']}: {row['error']}")
    (output / "protocol.txt").write_text("\n".join(notes) + "\n")
    print(rows_text, end="")
    print(f"Reports: {output}", file=sys.stderr)
    return int(args.strict and any(r["status"] != "complete" for r in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
