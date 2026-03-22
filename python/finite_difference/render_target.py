# main.py
import time
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
import pale
from matplotlib import cm

from finite_difference.finite_diff_helpers import save_rgb_preview_png, save_rgb_preview_exr
from losses import compute_l2_grad, compute_l2_loss


def main(args) -> None:
    # --- settings ---
    renderer_settings = {
        "photons": 1e6,
        "bounces": 3,
        "forward_passes": 500,
        "gather_passes": 1,
        "adjoint_bounces": 0,
        "adjoint_passes": 0,
        "logging": 2,
        "seed": 42
    }

    assets_root = Path(__file__).resolve().parents[2] / "Assets"

    scene_path = Path(args.scene).parent

    scene_xml = assets_root / "GradientTests" / f"{args.scene}" / f"{args.scene}.xml"
    pointcloud_ply = assets_root / "GradientTests" / scene_path / f"{args.scene}" / f"{args.ply}.ply"

    print("Assets root:", assets_root)
    print("Scene:", args.scene)
    print("Ply:", args.ply)
    print("Index:", args.index)
    print("Parameter:", args.parameter)

    output_dir = Path(__file__).parent / "Output" / scene_path / f"{args.scene}" / args.parameter
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- init renderer ---
    renderer = pale.Renderer(
        str(assets_root),
        str(scene_xml),
        str(pointcloud_ply),
        renderer_settings
    )

    camera_names = renderer.get_camera_names()
    print("Cameras:", camera_names)

    rendered_images = renderer.render_forward()

    if args.camera is None:
        cameras_to_render = camera_names
        print("Rendering from all cameras")
    else:
        if args.camera not in camera_names:
            raise ValueError(
                f"Camera '{args.camera}' not found. Available cameras: {camera_names}"
            )
        cameras_to_render = [args.camera]
        print("Rendering from camera:", args.camera)

    for camera_name in cameras_to_render:
        raw_key = f"{camera_name}_raw"
        png_key = camera_name

        if raw_key in rendered_images:
            save_rgb_preview_exr(
                rendered_images[raw_key],
                output_dir / f"{camera_name}_raw_target.exr"
            )

        if png_key in rendered_images:
            print(rendered_images[png_key].shape)
            save_rgb_preview_png(
                rendered_images[png_key],
                output_dir / f"{camera_name}_target.png"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-difference gradient visualization for Pale renderer."
    )
    parser.add_argument(
        "--ply",
        type=str,
        default="pointcloud",
        help="Points (PLY without extension). Default: 'initial'.",
    )

    parser.add_argument(
        "--scene",
        type=str,
        default="empty",
        help="Which scene file to use (without extension). Default: empty",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Gaussian index to perturb (>=0 for single, -1 for all). Default: -1.",
    )

    parser.add_argument(
        "--parameter", "--param",
        type=str
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Where to output files",
        default="output"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Which camera (in the xml file) to render from. If omitted, render all cameras.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)