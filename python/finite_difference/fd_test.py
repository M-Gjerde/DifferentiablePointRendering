# main.py
import argparse
import csv
from pathlib import Path

import numpy as np
import pale
import shutil
import uuid

from io_utils import read_rgb_exr
from losses import compute_l2_grad, compute_l2_loss
from finite_difference.finite_diff_helpers import (
    save_rgb_preview_png,
    save_rgb_preview_exr,
    save_seismic_signed,
)

def safe_rel_err(value_a: float, value_b: float, eps: float = 1e-12) -> float:
    denominator = max(eps, abs(value_a) + abs(value_b))
    return abs(value_a - value_b) / denominator

def create_latest_run_dir(base_output_dir: Path) -> Path:
    """
    Always writes the newest run to:  base_output_dir/0

    Before creating a new 0, it rotates old runs:
      0 -> 1, 1 -> 2, 2 -> 3, ...

    Important behavior:
    - Old run 0 is COPIED to 1 (not renamed away first), so the watched folder `0`
      gets explicit file removals/creation and file viewers refresh more reliably.

    Returns the (new) Path base_output_dir/0.
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)

    run_indices: list[int] = []
    for child in base_output_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            run_indices.append(int(child.name))
        except ValueError:
            continue

    if 0 in run_indices:
        run0 = base_output_dir / "0"

        for idx in sorted((i for i in run_indices if i != 0), reverse=True):
            src = base_output_dir / str(idx)
            dst = base_output_dir / str(idx + 1)
            if dst.exists():
                shutil.rmtree(dst)
            src.rename(dst)

        run1 = base_output_dir / "1"
        if run1.exists():
            shutil.rmtree(run1)
        shutil.copytree(run0, run1)

    run_dir = base_output_dir / "0"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def _set_parameter(renderer: "pale.Renderer", parameter: str, value: float, index: int) -> None:
    if parameter == "opacity":
        renderer.set_point_opacity(opacity=float(value), index=int(index))
    elif parameter == "beta":
        renderer.set_point_beta(beta=float(value), index=int(index))
    elif parameter == "translation_x":
        renderer.set_point_translation(translation=float(value), axis=0, index=int(index))
    elif parameter == "translation_y":
        renderer.set_point_translation(translation=float(value), axis=1, index=int(index))
    elif parameter == "translation_z":
        renderer.set_point_translation(translation=float(value), axis=2, index=int(index))
    elif parameter == "scale_u":
        renderer.set_point_scale(scale=float(value), axis=0, index=int(index))
    elif parameter == "scale_v":
        renderer.set_point_scale(scale=float(value), axis=1, index=int(index))
    else:
        raise RuntimeError(f"FD currently not implemented for '{parameter}'.")


def _render_loss(
    renderer: "pale.Renderer",
    camera: str,
    target_image: np.ndarray,
) -> tuple[float, np.ndarray, dict]:
    """
    Returns (loss_value, rendered_rgb, images_dict).
    rendered_rgb is float32 (H,W,3)
    """
    images = renderer.render_forward()
    image = images[camera + "_raw"]
    rendered = np.asarray(image, dtype=np.float32)[..., :3]
    loss_value = float(compute_l2_loss(rendered, target_image))
    return loss_value, rendered, images


def _finite_difference_loss(
    renderer: "pale.Renderer",
    parameter: str,
    base_value: float,
    eps: float,
    index: int,
    camera: str,
    target_image: np.ndarray,
    clamp_01: bool = True,
) -> tuple[float, float, float]:
    """
    Computes L(base), and a finite-difference derivative dL/dparam at base_value.

    Uses:
      - central difference if possible
      - otherwise one-sided difference.

    Returns (L0, fd_grad, fd_kind_code)
      fd_kind_code: 0=central, 1=forward, 2=backward
    """
    _set_parameter(renderer, parameter, base_value, index)
    renderer.rebuild_bvh()
    L0, _, _ = _render_loss(renderer, camera, target_image)

    if clamp_01 and parameter in {"opacity", "scale_u", "scale_v"}:
        lo = 0.0
        hi = 1.0
    else:
        lo = -np.inf
        hi = np.inf

    can_central = (base_value - eps) >= lo and (base_value + eps) <= hi

    if can_central:
        v_minus = base_value - eps
        v_plus = base_value + eps

        _set_parameter(renderer, parameter, v_plus, index)
        renderer.rebuild_bvh()
        Lp, _, _ = _render_loss(renderer, camera, target_image)

        _set_parameter(renderer, parameter, v_minus, index)
        renderer.rebuild_bvh()
        Lm, _, _ = _render_loss(renderer, camera, target_image)

        fd = (Lp - Lm) / (2.0 * eps)
        return L0, float(fd), 0.0

    if (base_value + eps) <= hi:
        v_plus = base_value + eps
        _set_parameter(renderer, parameter, v_plus, index)
        renderer.rebuild_bvh()
        Lp, _, _ = _render_loss(renderer, camera, target_image)
        fd = (Lp - L0) / eps
        return L0, float(fd), 1.0

    if (base_value - eps) >= lo:
        v_minus = base_value - eps
        _set_parameter(renderer, parameter, v_minus, index)
        renderer.rebuild_bvh()
        Lm, _, _ = _render_loss(renderer, camera, target_image)
        fd = (L0 - Lm) / eps
        return L0, float(fd), 2.0

    raise RuntimeError("Could not form any finite difference stencil.")


def main(args) -> None:
    renderer_settings = {
        "photons": 1e6,
        "bounces": args.bounces,
        "forward_passes": args.forward_passes,
        "gather_passes": 1,
        "adjoint_bounces": args.adjoint_bounces,
        "adjoint_passes": args.adjoint_passes,
        "logging": 4,
        "seed": args.seed,
    }
    assets_root = Path(__file__).resolve().parents[2] / "Assets"

    scene_xml = assets_root / "GradientTests" / args.scene / f"{args.scene}.xml"
    pointcloud_ply = assets_root / "GradientTests" / args.scene / f"{args.ply}.ply"

    if renderer_settings["logging"] < 4:
        print("Assets root:", assets_root)
        print("Scene:", scene_xml)
        print("Ply:", pointcloud_ply)
        print("Index:", args.index)
        print("Parameter:", args.parameter)

    fd_epsilon = args.fd_epsilon
    index = int(args.index)

    if renderer_settings["logging"] < 4:
        print("FD epsilon:", fd_epsilon)

    base_output_dir = (
        Path(__file__).parent
        / "Output"
        / args.scene
        / args.parameter
        / str(index)
    )

    output_dir = create_latest_run_dir(base_output_dir)

    rendered_dir = output_dir / "rendered"
    grad_dir = output_dir / "grad"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    grad_dir.mkdir(parents=True, exist_ok=True)

    renderer = pale.Renderer(str(assets_root), str(scene_xml), str(pointcloud_ply), renderer_settings)
    renderer_cameras = list(renderer.get_camera_names())
    camera = args.camera

    for camera_name in renderer_cameras:
        if camera_name == camera:
            continue
        (rendered_dir / camera_name).mkdir(parents=True, exist_ok=True)

    target_image = read_rgb_exr(output_dir.parent.parent / Path(camera + "_raw_target.exr"))

    if args.fill_target:
        #print(target_image.min(), target_image.max())
        target_image = np.ones_like(target_image, dtype=np.float32)

    if renderer_settings["logging"] < 4:
        print(f"Using run_dir: {output_dir}")
        print("Target image path:", output_dir.parent.parent / Path(camera + "_raw_target.exr"))
        print("Renderer cameras:", renderer_cameras)
        print("Fill target:", args.fill_target)

    csv_path = output_dir / f"{camera}_{args.parameter}_sweep.csv"
    fieldnames = [
        "iter",
        args.parameter,
        "loss",
        "analytic_grad",
        "fd_grad",
        "fd_kind",
        "fd_epsilon",
    ]

    iterations = int(args.iterations)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for iteration_index in range(iterations + 1):
            if iterations <= 0:
                raise RuntimeError("--iterations must be > 0.")
            t = iteration_index / iterations
            value = args.min + t * (args.max - args.min)

            loss_value, fd_grad, fd_kind = _finite_difference_loss(
                renderer=renderer,
                parameter=args.parameter,
                base_value=float(value),
                eps=float(fd_epsilon),
                index=index,
                camera=camera,
                target_image=target_image,
                clamp_01=True,
            )

            _set_parameter(renderer, args.parameter, float(value), index)
            renderer.rebuild_bvh()
            images = renderer.render_forward()
            rendered_image = np.asarray(images[camera + "_raw"], dtype=np.float32)[..., :3]

            loss_grad_image = compute_l2_grad(rendered_image, target_image)

            save_seismic_signed(
                loss_grad_image,
                grad_dir / f"{iteration_index}_{camera}.png",
                0.99,
            )

            if iterations <= 0:
                save_rgb_preview_exr(
                    target_image,
                    rendered_dir / f"{camera}_target.exr",
                    exposure_stops=0.0,
                )

            save_rgb_preview_png(
                images[camera],
                rendered_dir / f"{iteration_index}_{camera}.png",
                exposure_stops=0.0,
            )

            for camera_name in renderer_cameras:
                if camera_name not in images or args.camera == camera_name:
                    continue

                raw_key = f"{camera_name}_raw"
                if raw_key not in images:
                    print(f"Skipping EXR for missing camera key: {raw_key}")
                    continue

                camera_output_dir = rendered_dir / camera_name

                save_rgb_preview_png(
                    images[camera_name],
                    camera_output_dir / f"{iteration_index}_{camera_name}.png",
                    exposure_stops=0.0,
                )

            gradients, _adjoint_images = renderer.render_backward({camera: loss_grad_image})
            if args.parameter == "translation_x":
                param_gradients = gradients["position"]
                param_gradient = param_gradients[args.index][0]
            elif args.parameter == "translation_y":
                param_gradients = gradients["position"]
                param_gradient = param_gradients[args.index][1]
            elif args.parameter == "translation_z":
                param_gradients = gradients["position"]
                param_gradient = param_gradients[args.index][2]
            elif args.parameter == "scale_u":
                param_gradients = gradients["scale"]
                param_gradient = param_gradients[args.index][0]
            elif args.parameter == "scale_v":
                param_gradients = gradients["scale"]
                param_gradient = param_gradients[args.index][1]
            else:
                param_gradients = gradients[args.parameter]
                param_gradient = param_gradients[args.index]

            writer.writerow(
                {
                    "iter": iteration_index,
                    args.parameter: value,
                    "loss": float(loss_value),
                    "analytic_grad": param_gradient,
                    "fd_grad": float(fd_grad),
                    "fd_kind": int(fd_kind),
                    "fd_epsilon": float(fd_epsilon),
                }
            )

            relative_error = safe_rel_err(float(param_gradient), float(fd_grad))
            relative_error_percent = 100.0 * relative_error

            print(
                f"{iteration_index}/{iterations}, {args.parameter}: {value:.2f}, "
                f"Loss: {loss_value:.6f}, AN: {param_gradient:.6f}, FD: {fd_grad:.6f}, "
                f"RelErr: {relative_error_percent:.2f}% (kind={int(fd_kind)})"
            )
            f.flush()

    print(f"Saved to run_dir: {output_dir}")


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
        default=0,
        help="Point index to perturb. Default: 0.",
    )
    parser.add_argument(
        "--parameter",
        type=str,
        choices=[
            "translation_x",
            "translation_y",
            "translation_z",
            "scale_u",
            "scale_v",
            "opacity",
            "beta",
        ],
        default="opacity",
    )
    parser.add_argument(
        "--min",
        type=float,
        required=True,
        default=0.0,
        help="Minimum sweep value for the selected parameter.",
    )
    parser.add_argument(
        "--max",
        type=float,
        required=True,
        default=1.0,
        help="Maximum sweep value for the selected parameter.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Where to output files",
        default="output",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--camera",
        type=str,
        help="Which camera (in the xml file) to render from",
        default="camera1",
    )
    parser.add_argument(
        "--fd_epsilon",
        type=float,
        default=1e-2,
        help="Finite difference epsilon.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for renderer",
    )
    parser.add_argument(
        "--forward_passes",
        type=int,
        default=50,
        help="Number of forward passes.",
    )
    parser.add_argument(
        "--bounces",
        type=int,
        default=1,
        help="Number of forward bounces.",
    )
    parser.add_argument(
        "--adjoint_passes",
        type=int,
        default=64,
        help="Number of adjoint passes.",
    )
    parser.add_argument(
        "--adjoint_bounces",
        type=int,
        default=2,
        help="Number of adjoint bounces.",
    )
    parser.add_argument(
        "--fill_target",
        action="store_true",
        default=True,
        help="Replace the loaded target image with an all-ones image.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)