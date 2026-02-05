# main.py
import time
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
import pale
from matplotlib import cm

from finite_difference.finite_diff_helpers import save_rgb_preview_png, finite_difference_opacity, write_fd_images
from io_utils import load_target_image
from losses import compute_l2_grad, compute_l2_loss


def main(args) -> None:
    # --- settings ---
    renderer_settings = {
        "photons": 1e6,
        "bounces": 4,
        "forward_passes": 100,
        "gather_passes": 1,
        "adjoint_bounces": 1,
        "adjoint_passes": 100,
        "logging": 3
    }

    axis = args.axis  # x, y, or z

    assets_root = Path(__file__).parent.parent.parent / "Assets"
    scene_xml = args.scene + ".xml"
    pointcloud_ply = args.ply + ".ply"


    print("Assets root:", assets_root)
    print("Scene:", args.scene)
    print("Ply:", args.ply)
    print("Index:", args.index)
    print("Parameter:", args.param)

    output_dir = (
        Path(__file__).parent / "Output" / args.scene
        if args.output == "" or args.output is None
        else Path(__file__).parent / Path(args.output)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- init renderer ---
    renderer = pale.Renderer(str(assets_root), scene_xml, pointcloud_ply, renderer_settings)
    cameras = renderer.get_camera_names()
    camera = args.camera
    print("Cameras:", cameras)
    print("Rendering from camera:", camera)

    renderer.render_forward()
    rendered_image = renderer.render_forward()[camera]
    save_rgb_preview_png(rendered_image,  output_dir /  Path(camera + "_rendered.png"))

    target_image = load_target_image(output_dir /  Path(camera + "_target.png"))
    ## Analytical first:
    loss_grad = compute_l2_grad(
        rendered_image,
        target_image
    )
    loss_value = compute_l2_loss(
        rendered_image,
        target_image
    )

    loss_grad_images = dict()
    loss_grad_images[camera] = loss_grad

    gradients, adjoint_images = renderer.render_backward(loss_grad_images)

    analytical_gradients_opacity = np.asarray(
        gradients["opacity"],
        dtype=np.float32,
        order="C").squeeze()


    eps_opacity = 0.275
    rgb_minus, rgb_plus, grad_fd = finite_difference_opacity(renderer, 0, eps_opacity, camera)

    write_fd_images(
        grad_fd,
        rgb_minus,
        rgb_plus,
        output_dir,
        "opacity",
        "",
    )

    loss_negative = compute_l2_loss(rgb_minus, target_image)
    loss_positive = compute_l2_loss(rgb_plus, target_image)

    finite_gradients_opacity = (loss_positive - loss_negative) / (2.0 * eps_opacity)
    print("AN Gradient:", analytical_gradients_opacity)
    print("FD Gradient:", finite_gradients_opacity)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-difference gradient visualization for Pale renderer."
    )
    parser.add_argument(
        "--ply",
        type=str,
        default="initial",
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
        "--param",
        type=str,
        choices=["translation", "rotation", "scale", "translation_rotation", "opacity"],
        default="translation",
    )

    parser.add_argument(
        "--axis",
        type=str,
        choices=["x", "y", "z"],
        default="axis of choice",
        help="Which axis to finite-difference: 'translation', 'rotation', or 'scale'.",
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
        help="Which camera (In the xml file) to render from",
        default="camera1"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)