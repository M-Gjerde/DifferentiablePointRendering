"""Measure uncapped surfel overlap using center rays from saved scene cameras.

Uses the ellipse and beta profile in KernelHelpers.h, without BVH or hit-list
limits. Bounding rectangles only accelerate the exact ray/ellipse tests.
Example:
  python python/analyze/analyze_surfel_overlap.py --run-dir PATH --output-dir PATH
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def load_surfels(path):
    with path.open() as stream:
        names = []
        count = None
        for line in stream:
            if line.startswith("format ") and "ascii" not in line:
                raise ValueError("This diagnostic expects an ASCII surfel PLY")
            if line.startswith("element vertex "):
                count = int(line.split()[-1])
            if line.startswith("property "):
                names.append(line.split()[-1])
            if line.strip() == "end_header":
                break
        rows = np.loadtxt(stream, max_rows=count, ndmin=2)
    data = {name: rows[:, i] for i, name in enumerate(names)}
    keep = data["power"] <= 0
    p = np.column_stack([data[k][keep] for k in ("x", "y", "z")])
    q = np.column_stack([data[k][keep] for k in ("rot_w", "rot_x", "rot_y", "rot_z")])
    lengths = np.linalg.norm(q, axis=1)
    q[lengths <= 1e-10] = (1, 0, 0, 0)
    q /= np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    u = np.column_stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)))
    v = np.column_stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)))
    u /= np.linalg.norm(u, axis=1)[:, None]
    v -= np.sum(u*v, axis=1)[:, None]*u
    v /= np.linalg.norm(v, axis=1)[:, None]
    return dict(position=p, u=u, v=v, normal=np.cross(u, v),
                su=data["su"][keep], sv=data["sv"][keep],
                opacity=np.clip(data["opacity"][keep], 0, 1),
                exponent=4*np.exp(data["beta"][keep]),
                source_index=np.flatnonzero(keep), total_count=len(rows))


def load_cameras(path):
    result = []
    for sensor in ET.parse(path).getroot().findall("sensor"):
        values = {e.get("name"): float(e.get("value")) for e in sensor.findall("float")}
        look = sensor.find("transform/lookat")
        origin, target, up = [np.fromstring(look.get(k), sep=",") for k in ("origin", "target", "up")]
        forward = target-origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        film = {e.get("name"): int(e.get("value")) for e in sensor.findall("film/integer")}
        result.append(dict(name=sensor.get("id"), origin=origin, forward=forward,
                           right=right, up=up, **values, **film))
    return result


def rays(camera, x, y):
    d = (camera["forward"] + ((x+0.5-camera["cx"])/camera["fx"])[..., None]*camera["right"]
         + ((camera["cy"]-y-0.5)/camera["fy"])[..., None]*camera["up"])
    return d/np.linalg.norm(d, axis=-1)[..., None]


def collect_hits(s, camera):
    """Return pixel IDs, ray distances, effective alphas, and surfel row indices."""
    width, height = camera["width"], camera["height"]
    yy, xx = np.mgrid[:height, :width]
    directions = rays(camera, xx, yy)
    parts = [[], [], [], []]
    for i, center in enumerate(s["position"]):
        if s["su"][i] <= 0 or s["sv"][i] <= 0:
            continue
        # Perspective projection of a rectangle enclosing the ellipse is a
        # conservative image bound whenever all four corners are in front.
        corners = (center + np.array([-1, -1, 1, 1])[:, None]*s["u"][i]*s["su"][i]
                   + np.array([-1, 1, -1, 1])[:, None]*s["v"][i]*s["sv"][i])
        relative = corners-camera["origin"]
        z = relative@camera["forward"]
        if np.max(z) <= 0:
            continue
        if np.min(z) <= 0:
            x0, x1, y0, y1 = 0, width-1, 0, height-1
        else:
            px = camera["cx"] + camera["fx"]*(relative@camera["right"])/z
            py = camera["cy"] - camera["fy"]*(relative@camera["up"])/z
            x0, x1 = max(0, int(np.floor(px.min()))-1), min(width-1, int(np.ceil(px.max()))+1)
            y0, y1 = max(0, int(np.floor(py.min()))-1), min(height-1, int(np.ceil(py.max()))+1)
        if x0 > x1 or y0 > y1:
            continue
        d = directions[y0:y1+1, x0:x1+1]
        denom = d@s["normal"][i]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.dot(center-camera["origin"], s["normal"][i])/denom
            r = camera["origin"]+t[..., None]*d-center
            r2 = (r@s["u"][i]/s["su"][i])**2 + (r@s["v"][i]/s["sv"][i])**2
        hit = (abs(denom) > 1e-6) & (t > 1e-6) & (r2 < 1)
        iy, ix = np.nonzero(hit)
        if not len(ix):
            continue
        parts[0].append(((iy+y0)*width+(ix+x0)).astype(np.int32))
        parts[1].append(t[hit])
        parts[2].append(s["opacity"][i]*(1-r2[hit])**s["exponent"][i])
        parts[3].append(np.full(len(ix), i, dtype=np.int32))
    if not parts[0]:
        return (np.empty(0, np.int32), np.empty(0), np.empty(0), np.empty(0, np.int32))
    return tuple(np.concatenate(p) for p in parts)


def brute_force_check(s, camera, hits, sample_count=64):
    """Check the projection bounds against independent all-surfel ray queries."""
    pix, _, alpha, _ = hits
    size = camera["width"]*camera["height"]
    rng = np.random.default_rng(1729)
    choices = np.r_[rng.choice(size, sample_count, replace=False),
                    np.argsort(np.bincount(pix, minlength=size))[-16:]]
    for pixel in choices:
        d = rays(camera, np.array(pixel % camera["width"]), np.array(pixel // camera["width"]))
        delta = s["position"]-camera["origin"]
        denom = s["normal"]@d
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.sum(delta*s["normal"], axis=1)/denom
            r = t[:, None]*d-delta
            r2 = (np.sum(r*s["u"], axis=1)/s["su"])**2 + (np.sum(r*s["v"], axis=1)/s["sv"])**2
        hit = (abs(denom) > 1e-6) & (t > 1e-6) & (r2 < 1)
        a = s["opacity"][hit]*(1-r2[hit])**s["exponent"][hit]
        for threshold in (0, .01, .1):
            expected = np.count_nonzero(a > threshold)
            actual = np.count_nonzero((pix == pixel) & (alpha > threshold))
            if expected != actual:
                raise AssertionError((camera["name"], int(pixel), threshold, expected, actual))
    return len(choices)


def summarize(values):
    values = np.asarray(values)
    if not values.size:
        return {"pixels": 0}
    return dict(pixels=int(values.size), min=int(values.min()), max=int(values.max()),
                mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.percentile(values, 90)), p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)),
                pct_ge_5=float(100*np.mean(values >= 5)),
                pct_ge_10=float(100*np.mean(values >= 10)),
                pct_ge_15=float(100*np.mean(values >= 15)),
                histogram=np.bincount(values).tolist())


def analyze(s, camera, hits, epsilon):
    pix, depth, alpha, ids = hits
    size = camera["width"]*camera["height"]
    maps, stats, peaks = {}, {}, {}
    for threshold, label in ((0, "support"), (.01, "alpha_gt_0p01"), (.1, "alpha_gt_0p1")):
        active = alpha > threshold
        p, t, si = pix[active], depth[active], ids[active]
        first = np.full(size, np.inf)
        np.minimum.at(first, p, t)
        front = t <= first[p]+epsilon
        for kind, selected in (("all_depths", np.ones(len(p), bool)), ("front_layer", front)):
            name = f"{label}_{kind}"
            count = np.bincount(p[selected], minlength=size)
            maps[name] = count.reshape(camera["height"], camera["width"]).astype(np.uint16)
            stats[name] = dict(all_pixels=summarize(count), covered_pixels=summarize(count[count > 0]))
            peak = int(np.argmax(count))
            select_peak = (p == peak) & selected
            order = np.argsort(t[select_peak])
            peaks[name] = dict(x=peak % camera["width"], y=peak // camera["width"],
                               count=int(count[peak]),
                               source_indices=s["source_index"][si[select_peak]][order].tolist(),
                               ray_distances=t[select_peak][order].tolist(),
                               effective_alphas=alpha[active][select_peak][order].tolist())
    return maps, stats, peaks


def make_plots(run_dir, out_dir, cameras, all_maps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle
    from PIL import Image

    key = "support_all_depths"
    vmax = max(int(m[key].max()) for m in all_maps)
    fig, axs = plt.subplots(2, 5, figsize=(16, 7), layout="constrained")
    for camera, maps, ax in zip(cameras, all_maps, axs.flat):
        values = maps[key]
        im = ax.imshow(np.ma.masked_equal(values, 0), cmap="inferno", norm=PowerNorm(.5, vmin=1, vmax=vmax))
        ax.set_title(f'{camera["name"]}\nmean {values[values>0].mean():.2f} · max {values.max()}', fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axs, shrink=.8, label="Surfels per pixel center ray (all depths)")
    fig.suptitle("Latest plant · uncapped ellipse intersections · 10 cameras, 500 × 500", fontsize=16)
    fig.savefig(out_dir/"all_cameras_overlap.png", dpi=150)
    plt.close(fig)

    worst = max(range(len(all_maps)), key=lambda i: all_maps[i]["support_front_layer"].max())
    camera, maps = cameras[worst], all_maps[worst]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.8), layout="constrained")
    rgb = run_dir/"renders"/f'render_final_{camera["name"]}.png'
    axs[0].imshow(Image.open(rgb))
    axs[0].set_title("Saved final render")
    for ax, key, title in zip(axs[1:],
                            ("support_all_depths", "support_front_layer", "alpha_gt_0p01_front_layer"),
                            ("All depths", "Within 0.005 of frontmost hit", "Front layer, alpha > 0.01")):
        im = ax.imshow(np.ma.masked_equal(maps[key], 0), cmap="inferno", norm=PowerNorm(.5, vmin=1, vmax=vmax))
        ax.set_title(f'{title}\nmax {maps[key].max()}', fontsize=10)
    for ax in axs:
        ax.axis("off")
    fig.colorbar(im, ax=axs[1:], shrink=.8, label="Surfels per pixel")
    fig.suptitle(f'Plant overlap · {camera["name"]}', fontsize=16)
    fig.savefig(out_dir/"overlap_comparison.png", dpi=160)
    plt.close(fig)

    key = "support_front_layer"
    y, x = np.unravel_index(np.argmax(maps[key]), maps[key].shape)
    x0, x1 = max(0, x-30), min(camera["width"], x+31)
    y0, y1 = max(0, y-30), min(camera["height"], y+31)
    pixels = np.asarray(Image.open(rgb))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.8), layout="constrained")
    axs[0].imshow(pixels)
    axs[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#ffbb00", linewidth=2))
    axs[0].set_title("Location in final render")
    axs[1].imshow(pixels[y0:y1, x0:x1])
    axs[1].set_title("61 × 61 pixel crop")
    im = axs[2].imshow(maps[key][y0:y1, x0:x1], cmap="inferno", vmin=0, vmax=maps[key].max(), interpolation="nearest")
    axs[2].scatter([x-x0], [y-y0], marker="+", color="cyan", s=150)
    axs[2].set_title(f'Front layer: peak {maps[key][y,x]} surfels\npixel ({x}, {y}), zero-based')
    fig.colorbar(im, ax=axs[2], shrink=.75, label="Surfel count within 0.005 of first hit")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f'Dense cluster · {camera["name"]}', fontsize=16)
    fig.savefig(out_dir/"hotspot_detail.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--front-depth", type=float, default=.005)
    args = parser.parse_args()
    run_dir, out_dir = args.run_dir.resolve(), args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ply = run_dir/"points_final.ply"
    s = load_surfels(ply)
    cameras = load_cameras(run_dir/"scene.xml")
    report = dict(ply=str(ply), ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),
                  non_emissive_surfels=len(s["position"]), total_ply_rows=s["total_count"],
                  method="One pixel-center ray; uncapped ray/ellipse intersections; lights excluded; no occlusion termination. Front layer is within front_depth of the nearest hit passing each alpha threshold. All saved cameras, including floor surfaces.",
                  front_depth=args.front_depth, cameras=[])
    all_maps = []
    for camera in cameras:
        hits = collect_hits(s, camera)
        checked = brute_force_check(s, camera, hits)
        maps, stats, peaks = analyze(s, camera, hits, args.front_depth)
        all_maps.append(maps)
        np.savez_compressed(out_dir/f'{camera["name"]}_counts.npz', **maps)
        report["cameras"].append(dict(name=camera["name"], width=camera["width"], height=camera["height"],
                                       brute_force_rays_checked=checked, statistics=stats, peak_pixels=peaks))
        print(camera["name"], {key: {k: v for k, v in value["covered_pixels"].items() if k in ("min", "max", "mean", "median")}
                               for key, value in stats.items() if key.startswith("support_")}, flush=True)
    report["aggregate"] = {}
    for key in all_maps[0]:
        values = np.concatenate([maps[key].ravel() for maps in all_maps])
        report["aggregate"][key] = dict(all_pixels=summarize(values), covered_pixels=summarize(values[values > 0]))
    (out_dir/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    with (out_dir/"per_camera.csv").open("w", newline="") as stream:
        columns = ["camera", "metric", "population", "pixels", "min", "max", "mean", "median", "p90", "p95", "p99", "pct_ge_5", "pct_ge_10", "pct_ge_15"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for cam in report["cameras"] + [dict(name="ALL", statistics=report["aggregate"])]:
            for metric, populations in cam["statistics"].items():
                for population, stats in populations.items():
                    writer.writerow(dict(camera=cam["name"], metric=metric, population=population,
                                         **{k: v for k, v in stats.items() if k != "histogram"}))
    make_plots(run_dir, out_dir, cameras, all_maps)
    print("Saved", out_dir, flush=True)


if __name__ == "__main__":
    main()
