import argparse
from pathlib import Path

import pale

from finite_difference.finite_diff_helpers import save_rgb_preview_png, save_rgb_preview_exr


def main(args) -> None:
    renderer_settings = {
        "photons": 1e6,
        "bounces": args.bounces,
        "forward_passes": args.forward_passes,
        "primal_shadow_rays":  1,
        "adjoint_shadow_rays": 1,
        "gather_passes": 1,
        "logging": 4,
        "seed": 42,
    }

    assets_root = Path(__file__).resolve().parents[2] / "Assets"

    scene_path = Path(args.scene).parent

    scene_xml = assets_root / "GradientTests" / f"{args.scene}" / f"{args.scene}.xml"
    pointcloud_ply = assets_root / "GradientTests" / scene_path / f"{args.scene}" / f"{args.ply}.ply"

    if renderer_settings["logging"] < 4:
        print("Assets root:", assets_root)
        print("Scene:", args.scene)
        print("Ply:", args.ply)
        print("Index:", args.index)
        print("Parameter:", args.parameter)
        print("Forward passes:", args.forward_passes)
        print("Bounces:", args.bounces)

    output_dir = Path(__file__).parent / "Output" / scene_path / f"{args.scene}" / args.parameter
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = pale.Renderer(
        str(assets_root),
        str(scene_xml),
        str(pointcloud_ply),
        renderer_settings,
    )

    camera_names = renderer.get_camera_names()
    if renderer_settings["logging"] <= 4:
        print("Found following cameras in scene file:", camera_names)

    rendered_images = renderer.render_forward()

    if args.camera is None:
        cameras_to_render = camera_names
        if renderer_settings["logging"] < 4:
            print("Rendering from all cameras")
    else:
        if args.camera not in camera_names:
            raise ValueError(
                f"Camera '{args.camera}' not found. Available cameras: {camera_names}"
            )
        cameras_to_render = [args.camera]
        if renderer_settings["logging"] < 4:
            print("Rendering from camera:", args.camera)

    for camera_name in cameras_to_render:
        save_rgb_preview_exr(
            rendered_images[camera_name]["raw"],
            output_dir / f"{camera_name}_raw_target.exr",
        )

        save_rgb_preview_png(
            rendered_images[camera_name]["image"],
            output_dir / f"{camera_name}_target.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-difference gradient visualization for Pale renderer."
    )
    parser.add_argument(
        "--ply",
        type=str,
        default="pointcloud",
        help="Points (PLY without extension).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="empty",
        help="Which scene file to use (without extension).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Gaussian index to perturb (>=0 for single, -1 for all).",
    )
    parser.add_argument(
        "--parameter",
        "--param",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Where to output files.",
    )
    parser.add_argument(
        "--min",
        type=float,
    )
    parser.add_argument(
        "--max",
        type=float,
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Which camera to render from. If omitted, render all cameras.",
    )
    parser.add_argument(
        "--forward_passes",
        type=int,
        default=100,
        help="Number of forward passes for target rendering.",
    )
    parser.add_argument(
        "--bounces",
        type=int,
        default=2,
        help="Number of forward bounces for target rendering.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)