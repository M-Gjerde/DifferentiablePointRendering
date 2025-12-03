# 1. Analytical approach
# Render a target image

# Render an initial guess with some loss value

# Use adjoint to find the gradient of the cost function

# 2. Numerical gradients
# Same target image
# Same initial guess
# Render third image once per pertubed parameter
# Find numerical gradient on cost function

import argparse
from pathlib import Path
import pale
import numpy as np
from PIL import Image
import csv

from losses import (
    compute_l2_loss,
    compute_l2_grad
)
from render_hooks import (
    fetch_parameters,
)
from debug_init_utils import add_debug_noise_to_initial_parameters

from io_utils import (
    load_target_image,
    save_positions_numpy,
    save_render,
    save_gradient_sign_png_py,
    save_loss_image,
    save_gaussians_to_ply,
)

from finite_difference.finite_diff_helpers import (
    finite_difference_translation,
    finite_difference_rotation,
    finite_difference_scale,
)

def append_translation_run_to_history(
    history_path: Path,
    scene_name: str,
    gaussian_index: int,
    run_label: str,
    loss_value: float,
    grad_trans: dict[str, float],
    analytical_gradients_position: np.ndarray,
) -> None:
    """
    Append one row to the translation gradient history CSV.

    Columns:
        scene, index, label, loss,
        fd_x, fd_y, fd_z,
        an_x, an_y, an_z,
        err_x, err_y, err_z,
        abs_err_x, abs_err_y, abs_err_z
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = history_path.exists()

    fd_x = float(grad_trans["x"])
    fd_y = float(grad_trans["y"])
    fd_z = float(grad_trans["z"])

    an_x = float(analytical_gradients_position[0][0])
    an_y = float(analytical_gradients_position[0][1])
    an_z = float(analytical_gradients_position[0][2])

    err_x = fd_x - an_x
    err_y = fd_y - an_y
    err_z = fd_z - an_z

    abs_err_x = abs(err_x)
    abs_err_y = abs(err_y)
    abs_err_z = abs(err_z)

    with history_path.open("a", newline="") as file_handle:
        writer = csv.writer(file_handle)
        if not file_exists:
            writer.writerow(
                [
                    "scene",
                    "index",
                    "label",
                    "loss",
                    "fd_x",
                    "fd_y",
                    "fd_z",
                    "an_x",
                    "an_y",
                    "an_z",
                    "err_x",
                    "err_y",
                    "err_z",
                    "abs_err_x",
                    "abs_err_y",
                    "abs_err_z",
                ]
            )

        writer.writerow(
            [
                scene_name,
                gaussian_index,
                run_label,
                loss_value,
                fd_x,
                fd_y,
                fd_z,
                an_x,
                an_y,
                an_z,
                err_x,
                err_y,
                err_z,
                abs_err_x,
                abs_err_y,
                abs_err_z,
            ]
        )
def print_translation_history(history_path: Path) -> None:
    """
    Print all stored translation FD/AN runs so far.

    Shows FD, AN, and FD-AN error per axis to compare progress.
    """
    if not history_path.exists():
        print("\n[HIST] No history file found yet.")
        return

    with history_path.open("r", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        rows = list(reader)

    if not rows:
        print("\n[HIST] History file is empty.")
        return

    print("\n================ Translation Gradient History ================")
    for row in rows:
        scene_name = row["scene"]
        gaussian_index = int(row["index"])
        run_label = row["label"]
        loss_value = float(row["loss"])

        fd_x = float(row["fd_x"])
        fd_y = float(row["fd_y"])
        fd_z = float(row["fd_z"])

        an_x = float(row["an_x"])
        an_y = float(row["an_y"])
        an_z = float(row["an_z"])

        # Support both new and older files (if errors not present)
        if "err_x" in row and row["err_x"] != "":
            err_x = float(row["err_x"])
            err_y = float(row["err_y"])
            err_z = float(row["err_z"])
            abs_err_x = float(row["abs_err_x"])
            abs_err_y = float(row["abs_err_y"])
            abs_err_z = float(row["abs_err_z"])
        else:
            err_x = fd_x - an_x
            err_y = fd_y - an_y
            err_z = fd_z - an_z
            abs_err_x = abs(err_x)
            abs_err_y = abs(err_y)
            abs_err_z = abs(err_z)

        print(
            f"[RUN] label='{run_label}', scene='{scene_name}', index={gaussian_index}, "
            f"loss={loss_value: .8f}"
        )
        print(f"      FD : x={fd_x: .10f}, y={fd_y: .10f}, z={fd_z: .10f}")
        print(f"      AN : x={an_x: .10f}, y={an_y: .10f}, z={an_z: .10f}")
        print(
            "      ER : "
            f"FD-AN_x={err_x: .10f}, |FD-AN_x|={abs_err_x: .10f}; "
            f"FD-AN_y={err_y: .10f}, |FD-AN_y|={abs_err_y: .10f}; "
            f"FD-AN_z={err_z: .10f}, |FD-AN_z|={abs_err_z: .10f}"
        )
        print("------------------------------------------------------------")


def save_rgb_preview_png(
        img_f32: np.ndarray,
        out_path: Path,
        exposure_stops: float = 0.0,
        gamma: float = 1.0,
) -> None:
    """
    Save a linear RGB float32 image as an 8-bit PNG.

    img_f32:      HxWx3, linear RGB, usually HDR (0..+inf)
    exposure_stops: photographic EV; +1 doubles brightness
    gamma:        gamma for encoding (e.g. 2.2 for sRGB)
    """
    img = np.asarray(img_f32, dtype=np.float32)

    if exposure_stops != 0.0:
        img = img * (2.0 ** exposure_stops)

    img = np.clip(img, 0.0, None)

    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        img = np.power(img, inv_gamma, where=(img > 0.0), out=img)

    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    #print(f"Saving RGB preview to: {out_path.absolute()}")
    img = Image.fromarray(img_u8).convert("RGB")
    img.save(out_path)

def render_with_trs(
        renderer,
        translation3,
        rotation_quat4,
        scale3,
        color3,
        opacity,
        beta=0.0,
        index=-1,
) -> np.ndarray:
    """Apply a full TRS+color+opacity+beta and render."""
    renderer.set_gaussian_transform(
        translation3=translation3,
        rotation_quat4=rotation_quat4,
        scale3=scale3,
        color3=color3,
        opacity=opacity,
        beta=beta,
        index=index,
    )
    rgb = np.asarray(renderer.render_forward()["camera1"], dtype=np.float32)
    return rgb

# ---------- Main driver: compute FD for all parameters ----------
def main(args) -> None:

    adjoint_passes = 4

    renderer_settings = {
        "photons": 1e4,
        "bounces": 4,
        "forward_passes": 500,
        "gather_passes": 1,
        "adjoint_bounces": 1,
        "adjoint_passes": adjoint_passes,
        "logging": 3,
    }

    assets_root = Path(__file__).parent.parent.parent / "Assets"
    scene_xml = "cbox_custom.xml"
    pointcloud_ply = args.scene + ".ply"

    print("Assets root:", assets_root)
    print("Scene:", args.scene)
    print("Index:", args.index)

    base_output_dir = (
        Path(__file__).parent / "finite_diff" / "beta_kernel" / args.scene
        if args.output == "" or args.output is None
        else Path(__file__).parent / Path(args.output)
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    renderer = pale.Renderer(str(assets_root), scene_xml, pointcloud_ply, renderer_settings)
    camera = renderer.get_camera_names()[0]
    target_image = renderer.render_forward()[camera]
    save_rgb_preview_png(target_image, base_output_dir / "target_image.png")

    ## Apply some noise to our target positions
    # ------------------------------------------------------------------
    # Fetch target parameters from renderer
    # ------------------------------------------------------------------
    target_params = fetch_parameters(renderer)
    target_positions_np = target_params["position"]
    target_tangent_u_np = target_params["tangent_u"]
    target_tangent_v_np = target_params["tangent_v"]
    target_scale_np = target_params["scale"]
    target_color_np = target_params["color"]
    target_opacity_np = target_params["opacity"]
    target_beta_np = target_params["beta"]

    (
        noisy_positions_np,
        noisy_tangent_u_np,
        noisy_tangent_v_np,
        noisy_scales_np,
        noisy_colors_np,
        noisy_opacities_np,
        noisy_betas_np,
    ) = add_debug_noise_to_initial_parameters(
        target_positions_np,
        target_tangent_u_np,
        target_tangent_v_np,
        target_scale_np,
        target_color_np,
        target_opacity_np,
        target_beta_np,
    )
    print("Initial parameters perturbed by debug Gaussian noise.")

    renderer.apply_point_optimization(
        {
            "position": noisy_positions_np,
            "tangent_u": noisy_tangent_u_np,
            "tangent_v": noisy_tangent_v_np,
            "scale": noisy_scales_np,
            "color": noisy_colors_np,
            "opacity": noisy_opacities_np,
            "beta": noisy_betas_np
        }
    )
    renderer.rebuild_bvh()
    # Render initial guess image:
    initial_guess = renderer.render_forward()[camera]
    save_rgb_preview_png(initial_guess, base_output_dir / "initial_guess.png")

    ## Now we're ready to do both analytical and finite difference calculation of cost function gradients

    ## Analytical first:
    loss_grad = compute_l2_grad(
        initial_guess,
        target_image
    )
    loss_value = compute_l2_loss(
        initial_guess,
        target_image
    )

    loss_grad_images = dict()
    loss_grad_images[camera] = loss_grad

    gradients, adjoint_images = renderer.render_backward(loss_grad_images)
    analytical_gradients_position = np.asarray(gradients["position"], dtype=np.float32, order="C")

    # Per-camera adjoint/gradient visualization
    grad_image_numpy = np.asarray(
        adjoint_images[camera],
        dtype=np.float32,
        order="C",
    )
    grad_image_numpy = np.nan_to_num(
        grad_image_numpy,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    grad_path = (
            base_output_dir
            / "grad_099.png"
    )
    save_gradient_sign_png_py(
        grad_path,
        grad_image_numpy,
        adjoint_spp=adjoint_passes,
        abs_quantile=0.999,
        flip_y=False,
    )

    # Step sizes
    eps_translation = 0.01
    eps_rotation_deg = 0.5
    eps_scale = 0.05
    eps_opacity = 0.1
    eps_albedo = 0.1
    eps_beta = 0.5  # reasonable start for log-shape

    # --- Translation (x,y,z) ---
    grad_trans = dict()
    for axis in ["x", "y", "z"]:
        print(f"[FD] Computing Translation axis={axis}...")
        rgb_minus, rgb_plus, grad_fd = finite_difference_translation(
            renderer, args.index, axis, eps_translation
        )
        # calculate loss for each image
        # OUr L1 loss is the differentiation of our L2 cost
        loss_negative = compute_l2_loss(rgb_minus, target_image)
        loss_positive = compute_l2_loss(rgb_plus, target_image)
        grad_trans[axis] = (loss_positive - loss_negative) / (2.0 * eps_translation)
        save_rgb_preview_png(rgb_minus, base_output_dir / "translation" / f"{axis}_neg.png")
        save_rgb_preview_png(rgb_plus, base_output_dir / "translation" / f"{axis}_pos.png")

    # --- Rotation (x,y,z) ---
    grad_rot = dict()
    #for axis in ["x", "y", "z"]:
    #    print(f"[FD] Computing Rotation axis={axis}...")
    #    rgb_minus, rgb_plus, grad_fd = finite_difference_rotation(
    #        renderer, args.index, axis, eps_rotation_deg
    #    )
    #    # calculate loss for each image
    #    # OUr L1 loss is the differentiation of our L2 cost
    #    loss_negative = compute_l2_loss(rgb_minus, target_image)
    #    loss_positive = compute_l2_loss(rgb_plus, target_image)
    #    grad_rot[axis] = (loss_positive - loss_negative) / (2.0 * eps_rotation_deg)
    #    save_rgb_preview_png(rgb_minus, base_output_dir / "rotation" / f"{axis}_neg.png")
    #    save_rgb_preview_png(rgb_plus, base_output_dir / "rotation" / f"{axis}_pos.png")

    axes = ["x", "y", "z"]
    print()
    for axis_index, axis_name in enumerate(axes):
        fd_val = float(grad_trans[axis_name])
        an_val = float(analytical_gradients_position[0][axis_index])
        err = fd_val - an_val
        abs_err = abs(err)

        print(f"[FD] Translation axis={axis_name}, Grad={fd_val: .10f}")
        print(f"[AN] Translation axis={axis_name}, Grad={an_val: .10f}")
        print(
            f"[ER] Translation axis={axis_name}, "
            f"FD-AN={err: .10f}, |FD-AN|={abs_err: .10f}"
        )
        print()


    # ----------------------------------------------------------
    # Save this run to history and print all runs so far
    # ----------------------------------------------------------
    history_path = base_output_dir / "translation_gradients_history.csv"
    append_translation_run_to_history(
        history_path=history_path,
        scene_name=args.scene,
        gaussian_index=args.index,
        run_label=args.label,
        loss_value=loss_value,
        grad_trans=grad_trans,
        analytical_gradients_position=analytical_gradients_position,
    )
    print_translation_history(history_path)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-difference gradient visualization for Pale renderer (all parameters)."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="target",
        help="Scene base name (PLY without extension). Default: 'initial'.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Gaussian index to perturb. Default: -1.",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (relative to this script). Default: Output/<scene>/finite_diff",
    )

    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label for this FD/AN comparison run (e.g. 'step_0001', 'noisy_init').",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
