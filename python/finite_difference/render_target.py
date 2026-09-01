from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pale

from finite_difference.fd_common import ASSETS_ROOT, _set_parameter, load_suite, scene_paths
from finite_difference.finite_diff_helpers import save_rgb_preview_exr, save_rgb_preview_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the baseline image for one schema-v2 FD case.")
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path(__file__).with_name("tests_direct.json"),
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--value-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, cases = load_suite(args.suite)
    case = next((candidate for candidate in cases if candidate["name"] == args.case), None)
    if case is None:
        raise ValueError(f"Unknown case '{args.case}'")
    values = [float(value) for value in case["values"]]
    if args.value_index < 0 or args.value_index >= len(values):
        raise ValueError(f"--value-index outside [0, {len(values)})")

    scene_xml, pointcloud_ply = scene_paths(case)
    renderer = pale.Renderer(
        str(ASSETS_ROOT),
        str(scene_xml),
        str(pointcloud_ply),
        case["settings"],
    )
    _set_parameter(
        renderer,
        str(case["parameter"]),
        values[args.value_index],
        int(case["index"]),
    )
    images = renderer.render_forward()
    camera = str(case["camera"])
    raw = np.asarray(images[camera]["raw"], dtype=np.float32)[..., :3]
    args.output.mkdir(parents=True, exist_ok=True)
    save_rgb_preview_exr(raw, args.output / f"{case['name']}_raw.exr")
    save_rgb_preview_png(images[camera]["image"], args.output / f"{case['name']}.png")


if __name__ == "__main__":
    main()
