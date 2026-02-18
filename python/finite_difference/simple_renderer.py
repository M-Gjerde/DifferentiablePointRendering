# main.py
import time
from pathlib import Path
import argparse

import numpy as np
import pale


from finite_difference.finite_diff_helpers import save_rgb_preview_png, finite_difference_opacity, write_fd_images, \
    render_with_trs, set_point_opacity, save_rgb_preview_exr, save_seismic_signed
from io_utils import load_target_image, read_rgb_exr
from losses import compute_l2_grad, compute_l2_loss
import matplotlib.pyplot as plt
import csv


def create_incremental_run_dir(base_output_dir: Path) -> Path:
    """
    Creates a new subfolder under base_output_dir named by incrementing integers:
      base_output_dir/0, base_output_dir/1, ...
    Returns the created Path.
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)

    max_run_index = -1
    for child in base_output_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            run_index = int(child.name)
        except ValueError:
            continue
        max_run_index = max(max_run_index, run_index)

    new_run_index = max_run_index + 1
    run_dir = base_output_dir / str(new_run_index)
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Created run_dir: {run_dir}")
    return run_dir



def main(args) -> None:
    renderer_settings = {
        "photons": 3e6,
        "bounces": 4,
        "forward_passes": 50,
        "gather_passes": 1,
        "adjoint_bounces": 1,
        "adjoint_passes": 1,
        "logging": 3
    }

    assets_root = Path(__file__).parent.parent.parent / "Assets"
    scene_xml = args.scene + ".xml"
    pointcloud_ply = args.ply + ".ply"

    base_output_dir = Path(__file__).parent / "Output" / args.scene / args.parameter
    output_dir = create_incremental_run_dir(base_output_dir)

    renderer = pale.Renderer(str(assets_root), scene_xml, pointcloud_ply, renderer_settings)

    camera = args.camera
    target_image = read_rgb_exr(output_dir.parent / Path(camera + "_raw_target.exr"))
    print("Target image path:", output_dir.parent / Path(camera + "_raw_target.exr"))

    # Render ONCE (baseline)
    csv_path = output_dir / f"{camera}_{args.parameter}_sweep.csv"
    fieldnames = ["iter", args.parameter, "loss", "analytic_grad"]

    iterations = int(args.iterations)
    # Write header once (overwrite each run). Use "a" if you want to keep adding across runs.
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for iteration_index in range(iterations + 1):

            if args.parameter == "opacity":
                value = (iteration_index) / iterations  # 0.00, 0.02, ..., 0.10
                renderer.set_point_opacity(
                    opacity=value,
                    index=0
                )
            elif args.parameter == "beta":
                value = 6 - (iteration_index * 12) / iterations  # 0.00, 0.02, ..., 0.10
                renderer.set_point_beta(
                    beta=value,
                    index = 0
                )
            renderer.rebuild_bvh()

            images = renderer.render_forward()
            image = images[camera + "_raw"]
            rendered_image = np.asarray(image, dtype=np.float32)
            rendered_image = rendered_image[..., :3] # Drop Alpha
            loss_grad_image = compute_l2_grad(rendered_image, target_image)
            loss_value = float(compute_l2_loss(rendered_image, target_image))

            # If you want per-iteration previews, include iteration in filename.
            save_rgb_preview_png(images[camera],  output_dir / "rendered" / Path(camera + f"_{value}" + ".png"), exposure_stops=0.0)
            #save_rgb_preview_png(target_image,  output_dir / "rendered" / Path(camera + f"_{value}_target" + ".png"), exposure_stops=0.0)
            save_rgb_preview_exr(rendered_image,  output_dir / "rendered" / Path(camera + f"_{value}" + ".exr"), exposure_stops=0.0)
            save_rgb_preview_exr(target_image,  output_dir / "rendered" / Path(camera + f"_target" + ".exr"), exposure_stops=0.0)
            grad_vis = (loss_grad_image - loss_grad_image.min()) / (loss_grad_image.max() - loss_grad_image.min() + 1e-8)

            save_seismic_signed(loss_grad_image, output_dir / "grad" / Path(camera + f"_{value}" + ".png"), 0.99)

            gradients, _adjoint_images = renderer.render_backward({camera: loss_grad_image})
            analytic_grad = float(np.asarray(gradients[args.parameter], dtype=np.float32).squeeze())

            writer.writerow({
                "iter": iteration_index,
                args.parameter: value,
                "loss": loss_value,
                "analytic_grad": analytic_grad,
            })
            print(f"{iteration_index}/{iterations}, {args.parameter}: {value}, Loss: {loss_value}, AN: {analytic_grad}")
            f.flush()  # ensures you can plot while it’s running

    print(f"Saved to run_dir: {output_dir}")

    #run_opacity_fd_linear_sweep(
    #    renderer=renderer,
    #    camera=camera,
    #    target_image=target_image,
    #    analytic_grad_scalar=analytic_grad,
    #    index=(args.index if args.index >= 0 else 0),
    #    output_dir=output_dir
    #)'

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
        "--parameter",
        type=str,
        choices=["translation", "rotation", "scale", "opacity", "beta"],
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
        "--iterations",
        type=int,
        default=20,
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