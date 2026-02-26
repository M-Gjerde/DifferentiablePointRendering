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


def create_latest_run_dir(base_output_dir: Path) -> Path:
    """
    Always writes the newest run to:  base_output_dir/0

    Before creating a new 0, it rotates old runs:
      0 -> 1, 1 -> 2, 2 -> 3, ...

    Returns the (new) Path base_output_dir/0.
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Collect existing integer-named run dirs
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

        # Move 0 aside to avoid collisions
        tmp = base_output_dir / f".tmp_run0_{uuid.uuid4().hex}"
        run0.rename(tmp)

        # Shift N -> N+1 (descending)
        for idx in sorted((i for i in run_indices if i != 0), reverse=True):
            src = base_output_dir / str(idx)
            dst = base_output_dir / str(idx + 1)
            # dst shouldn't exist if we shift descending, but be safe
            if dst.exists():
                shutil.rmtree(dst)
            src.rename(dst)

        # Put old 0 into 1
        (base_output_dir / "1").mkdir(parents=True, exist_ok=True)  # ensure parent exists
        tmp.rename(base_output_dir / "1")

    # Create a fresh 0
    run_dir = base_output_dir / "0"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"Using latest run_dir: {run_dir}")
    return run_dir


def _set_parameter(renderer: "pale.Renderer", parameter: str, value: float, index: int) -> None:
    # Extend this if you later want FD for translation/rotation/scale.
    if parameter == "opacity":
        renderer.set_point_opacity(opacity=float(value), index=int(index))
    elif parameter == "beta":
        renderer.set_point_beta(beta=float(value), index=int(index))
    else:
        raise RuntimeError(f"FD currently implemented only for opacity/beta, got '{parameter}'.")


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
    rendered = np.asarray(image, dtype=np.float32)[..., :3]  # drop alpha
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
      - central difference if possible (base-eps >= 0 and base+eps <= 1 for opacity, if clamp_01)
      - otherwise one-sided difference.

    Returns (L0, fd_grad, fd_kind_code)
      fd_kind_code: 0=central, 1=forward, 2=backward
    """
    # Base
    _set_parameter(renderer, parameter, base_value, index)
    renderer.rebuild_bvh()
    L0, _, _ = _render_loss(renderer, camera, target_image)

    # Decide stencil
    if clamp_01 and parameter == "opacity":
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

    # One-sided
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

    # Should never happen for opacity in [0,1] with eps>0
    raise RuntimeError("Could not form any finite difference stencil.")


def main(args) -> None:
    renderer_settings = {
        "photons": 1e6,
        "bounces": 4,
        "forward_passes": 10,
        "gather_passes": 1,
        "adjoint_bounces": 1,
        "adjoint_passes": 8,
        "logging": 5,
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
    print("FD epsilon:", args.fd_epsilon)

    output_dir = Path(__file__).parent / "Output" / scene_path / f"{args.scene}" / args.parameter
    output_dir = create_latest_run_dir(output_dir)

    # Create subfolders
    (output_dir / "rendered").mkdir(parents=True, exist_ok=True)
    (output_dir / "grad").mkdir(parents=True, exist_ok=True)

    renderer = pale.Renderer(str(assets_root), str(scene_xml), str(pointcloud_ply), renderer_settings)

    camera = args.camera
    target_image = read_rgb_exr(output_dir.parent / Path(camera + "_raw_target.exr"))
    print("Target image path:", output_dir.parent / Path(camera + "_raw_target.exr"))

    csv_path = output_dir / f"{camera}_{args.parameter}_sweep.csv"
    fieldnames = [
        "iter",
        args.parameter,
        "loss",
        "analytic_grad",
        "fd_grad",
        "fd_kind",  # 0=central, 1=forward, 2=backward
        "fd_epsilon",
    ]

    iterations = int(args.iterations)
    index = int(args.index if args.index >= 0 else 0)  # keep your prior behavior

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for iteration_index in range(iterations + 1):
            if args.parameter == "opacity":
                value = (iteration_index) / iterations  # 0..1
            elif args.parameter == "beta":
                value = 6 - (iteration_index * 12) / iterations
            else:
                raise RuntimeError("This script currently supports opacity/beta sweeps only.")

            # --- Finite difference derivative of LOSS at 'value' ---
            # Note: this renders multiple times per iteration (central = 3 total renders).
            loss_value, fd_grad, fd_kind = _finite_difference_loss(
                renderer=renderer,
                parameter=args.parameter,
                base_value=float(value),
                eps=float(args.fd_epsilon),
                index=index,
                camera=camera,
                target_image=target_image,
                clamp_01=True,
            )

            # Restore base state and render once more for:
            #  - saving previews
            #  - computing per-pixel dLoss/dI for adjoint
            _set_parameter(renderer, args.parameter, float(value), index)
            renderer.rebuild_bvh()
            images = renderer.render_forward()
            rendered_image = np.asarray(images[camera + "_raw"], dtype=np.float32)[..., :3]

            loss_grad_image = compute_l2_grad(rendered_image, target_image)

            # Save previews
            save_rgb_preview_png(
                images[camera],
                output_dir / "rendered" / Path(camera + f"_{value}" + ".png"),
                exposure_stops=0.0,
            )
            save_rgb_preview_exr(
                rendered_image,
                output_dir / "rendered" / Path(camera + f"_{value}" + ".exr"),
                exposure_stops=0.0,
            )
            save_rgb_preview_exr(
                target_image,
                output_dir / "rendered" / Path(camera + f"_target" + ".exr"),
                exposure_stops=0.0,
            )
            save_seismic_signed(
                loss_grad_image,
                output_dir / "grad" / Path(camera + f"_{value}" + ".png"),
                0.99,
            )

            # Adjoint / analytic gradient
            gradients, _adjoint_images = renderer.render_backward({camera: loss_grad_image})
            analytic_grad = float(np.asarray(gradients[args.parameter], dtype=np.float32).squeeze())

            writer.writerow(
                {
                    "iter": iteration_index,
                    args.parameter: value,
                    "loss": float(loss_value),
                    "analytic_grad": analytic_grad,
                    "fd_grad": float(fd_grad),
                    "fd_kind": int(fd_kind),
                    "fd_epsilon": float(args.fd_epsilon),
                }
            )

            print(
                f"{iteration_index}/{iterations}, {args.parameter}: {value:.2f}, "
                f"Loss: {loss_value:.5f}, AN: {analytic_grad:.5f}, FD: {fd_grad:.5f} (kind={int(fd_kind)})"
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
        default="opacity",
    )
    parser.add_argument(
        "--axis",
        type=str,
        choices=["x", "y", "z"],
        default="x",
        help="Which axis to finite-difference: 'translation', 'rotation', or 'scale'.",
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
        default=1e-3,
        help="Finite difference epsilon. Default 1e-3.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
