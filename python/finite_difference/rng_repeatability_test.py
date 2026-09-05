from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pale


def render_rgb(renderer: "pale.Renderer", camera: str) -> np.ndarray:
    images = renderer.render_forward()
    return np.ascontiguousarray(np.asarray(images[camera]["raw"], dtype=np.float32)[..., :3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assert deterministic repeated Pale forward renders.")
    parser.add_argument("--scene", default="fd_direct")
    parser.add_argument("--ply", default="single_surfel")
    parser.add_argument("--camera", default="camera")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--shared-light", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--new-instance", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 2 or args.tolerance < 0.0:
        raise ValueError("--repeats must be >=2 and --tolerance must be non-negative")

    assets_root = Path(__file__).resolve().parents[2] / "Assets"
    scene_directory = assets_root / "GradientTests" / args.scene
    scene_xml = scene_directory / f"{args.scene}.xml"
    pointcloud_ply = scene_directory / f"{args.ply}.ply"
    settings = {
        "photons": 1,
        "bounces": 1,
        "forward_passes": 1,
        "adjoint_bounces": 1,
        "adjoint_passes": 1,
        "logging": 4,
        "seed": args.seed,
        "share_local_layer_direct_lighting": args.shared_light,
    }

    renderer = pale.Renderer(str(assets_root), str(scene_xml), str(pointcloud_ply), settings)
    reference = render_rgb(renderer, args.camera)
    failures: list[str] = []
    for repeat_index in range(1, args.repeats):
        image = render_rgb(renderer, args.camera)
        maximum = float(np.max(np.abs(image.astype(np.float64) - reference.astype(np.float64))))
        print(f"same-instance repeat {repeat_index}: max_abs={maximum:.9e}")
        if not np.isfinite(maximum) or maximum > args.tolerance:
            failures.append(f"same-instance repeat {repeat_index}: {maximum}")

    if args.new_instance:
        second_renderer = pale.Renderer(str(assets_root), str(scene_xml), str(pointcloud_ply), settings)
        second = render_rgb(second_renderer, args.camera)
        maximum = float(np.max(np.abs(second.astype(np.float64) - reference.astype(np.float64))))
        print(f"new-instance repeat: max_abs={maximum:.9e}")
        if not np.isfinite(maximum) or maximum > args.tolerance:
            failures.append(f"new-instance repeat: {maximum}")

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
