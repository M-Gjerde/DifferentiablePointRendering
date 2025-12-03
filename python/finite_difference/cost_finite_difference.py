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
from matplotlib import cm

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
    Image.fromarray(img_u8, mode="RGB").save(out_path)


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


# ---------- Finite difference core ----------
def finite_difference_translation(
        renderer, index: int, axis: str, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axis == "x":
        negative_vector = (-eps, 0.0, 0.0)
        positive_vector = (+eps, 0.0, 0.0)
    elif axis == "y":
        negative_vector = (0.0, -eps, 0.0)
        positive_vector = (0.0, +eps, 0.0)
    elif axis == "z":
        negative_vector = (0.0, 0.0, -eps)
        positive_vector = (0.0, 0.0, +eps)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    rgb_minus = render_with_trs(
        renderer,
        translation3=negative_vector,
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        color3=(0.0, 0.0, 0.0),
        opacity=0.0,
        index=index,
    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=positive_vector,
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        color3=(0.0, 0.0, 0.0),
        opacity=0.0,
        index=index,
    )
    grad = (rgb_plus - rgb_minus) / (2.0 * eps)
    return rgb_minus, rgb_plus, grad


# ---------- Main driver: compute FD for all parameters ----------
def main(args) -> None:

    adjoint_passes = 4

    renderer_settings = {
        "photons": 5e3,
        "bounces": 3,
        "forward_passes": 1000,
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
    eps_translation = 0.001
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
        grad_trans[axis] = (loss_positive - loss_negative) / eps_translation
        save_rgb_preview_png(rgb_minus, base_output_dir / "translation" / f"{axis}_neg.png")
        save_rgb_preview_png(rgb_plus, base_output_dir / "translation" / f"{axis}_pos.png")

    for axis in ["x", "y", "z"]:
        print(f"[FD] Translation axis={axis}, Grad={grad_trans[axis]: .10f}")

    print()
    axes = ["x", "y", "z"]
    for axis in [0, 1, 2]:
        print(f"[AN] Translation axis={axes[axis]}, Grad={analytical_gradients_position[0][axis]: .10f}")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
