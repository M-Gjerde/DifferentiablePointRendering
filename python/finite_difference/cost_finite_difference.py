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
)

from finite_difference.finite_diff_helpers import (
    finite_difference_translation,
    finite_difference_rotation,
    finite_difference_scale,
    finite_difference_opacity,
    finite_difference_albedo,
    finite_difference_beta,
    save_rgb_preview_png,
    write_fd_images,
    degrees_to_quaternion,
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

    an_x = float(analytical_gradients_position[0])
    an_y = float(analytical_gradients_position[1])
    an_z = float(analytical_gradients_position[2])

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


def append_scale_run_to_history(
        history_path: Path,
        scene_name: str,
        gaussian_index: int,
        run_label: str,
        loss_value: float,
        grad_scale: dict[str, float],
        analytical_gradients_scale: np.ndarray,
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

    fd_x = float(grad_scale["x"])
    fd_y = float(grad_scale["y"])

    an_x = float(analytical_gradients_scale[0])
    an_y = float(analytical_gradients_scale[1])

    err_x = fd_x - an_x
    err_y = fd_y - an_y

    abs_err_x = abs(err_x)
    abs_err_y = abs(err_y)

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
                np.nan,
                an_x,
                an_y,
                np.nan,
                err_x,
                err_y,
                np.nan,
                abs_err_x,
                abs_err_y,
                np.nan,
            ]
        )


def append_1d_run_to_history(
        history_path: Path,
        scene_name: str,
        gaussian_index: int,
        run_label: str,
        loss_value: float,
        fd_grad: float,
        an_grad: float,
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

    fd_x = float(fd_grad)
    an_x = float(an_grad)

    err_x = fd_x - an_x

    abs_err_x = abs(err_x)

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
                np.nan,
                np.nan,
                an_x,
                np.nan,
                np.nan,
                err_x,
                np.nan,
                np.nan,
                abs_err_x,
                np.nan,
                np.nan,
            ]
        )


def print_history_3axis(history_path: Path) -> None:
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

    print("\n================ 3Axis Gradient History ================")
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


def print_history_2axis(history_path: Path) -> None:
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

        an_x = float(row["an_x"])
        an_y = float(row["an_y"])

        # Support both new and older files (if errors not present)
        if "err_x" in row and row["err_x"] != "":
            err_x = float(row["err_x"])
            err_y = float(row["err_y"])
            err_z = float(row["err_z"])
            abs_err_x = float(row["abs_err_x"])
            abs_err_y = float(row["abs_err_y"])
        else:
            err_x = fd_x - an_x
            err_y = fd_y - an_y
            err_z = fd_z - an_z
            abs_err_x = abs(err_x)
            abs_err_y = abs(err_y)

        print(
            f"[RUN] label='{run_label}', scene='{scene_name}', index={gaussian_index}, "
            f"loss={loss_value: .8f}"
        )
        print(f"      FD : x={fd_x: .10f}, y={fd_y: .10f}")
        print(f"      AN : x={an_x: .10f}, y={an_y: .10f}")
        print(
            "      ER : "
            f"FD-AN_x={err_x: .10f}, |FD-AN_x|={abs_err_x: .10f}; "
            f"FD-AN_y={err_y: .10f}, |FD-AN_y|={abs_err_y: .10f}; "
        )
        print("------------------------------------------------------------")


def print_history_1axis(history_path: Path) -> None:
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

        an_x = float(row["an_x"])

        # Support both new and older files (if errors not present)
        if "err_x" in row and row["err_x"] != "":
            err_x = float(row["err_x"])
            err_y = float(row["err_y"])
            err_z = float(row["err_z"])
            abs_err_x = float(row["abs_err_x"])
        else:
            err_x = fd_x - an_x
            abs_err_x = abs(err_x)

        print(
            f"[RUN] label='{run_label}', scene='{scene_name}', index={gaussian_index}, "
            f"loss={loss_value: .8f}"
        )
        print(f"      FD : val={fd_x: .10f}")
        print(f"      AN : val={an_x: .10f}")
        print(
            "      ER : "
            f"FD-AN_x={err_x: .10f}, |FD-AN_x|={abs_err_x: .10f}; "
        )
        print("------------------------------------------------------------")


def append_rotation_run_to_history(
        history_path: Path,
        scene_name: str,
        gaussian_index: int,
        run_label: str,
        loss_value: float,
        grad_rot: dict[str, float],
        analytical_rot_grad: np.ndarray,
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

    fd_x = float(grad_rot["x"])
    fd_y = float(grad_rot["y"])
    fd_z = float(grad_rot["z"])

    an_x = float(analytical_rot_grad["x"])
    an_y = float(analytical_rot_grad["y"])
    an_z = float(analytical_rot_grad["z"])

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



def apply_quaternion_to_vector(quaternion_xyzw: np.ndarray,
                               vector_xyz: np.ndarray) -> np.ndarray:
    """
    Rotate `vector_xyz` by unit quaternion `quaternion_xyzw` (x, y, z, w).
    """
    qx, qy, qz, qw = quaternion_xyzw
    vector = np.asarray(vector_xyz, dtype=np.float32)

    # Quaternion-vector rotation using: v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
    q_vec = np.array([qx, qy, qz], dtype=np.float32)

    temp = 2.0 * np.cross(q_vec, vector)
    rotated_vector = vector + qw * temp + np.cross(q_vec, temp)
    return rotated_vector


def render_with_trs(
        renderer,
        translation3,
        rotation_quat4,
        scale3,
        albedo3,
        opacity,
        beta=0.0,
        index=-1,
) -> np.ndarray:
    """Apply a full TRS+albedo+opacity+beta and render."""
    renderer.set_gaussian_transform(
        translation3=translation3,
        rotation_quat4=rotation_quat4,
        scale3=scale3,
        albedo3=albedo3,
        opacity=opacity,
        beta=beta,
        index=index,
    )
    rgb = np.asarray(renderer.render_forward()["camera1"], dtype=np.float32)
    return rgb

def analytic_rotation_grad_for_axis_deg(
    axis_name: str,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    grad_tangent_u: np.ndarray,
    grad_tangent_v: np.ndarray,
) -> float:
    """
    Compute analytical dL/d(theta_deg) for rotation about axis_name ('x','y','z'),
    given tangents and their gradients.
    """
    if axis_name == "x":
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif axis_name == "y":
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elif axis_name == "z":
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        raise ValueError(f"Unknown axis '{axis_name}'")

    # d t / d theta_rad = a × t
    du_dtheta_rad = np.cross(axis, tangent_u)
    dv_dtheta_rad = np.cross(axis, tangent_v)

    # dL/dtheta_rad
    dL_dtheta_rad = float(
        np.dot(grad_tangent_u, du_dtheta_rad) +
        np.dot(grad_tangent_v, dv_dtheta_rad)
    )

    # convert to per-degree derivative
    dL_dtheta_deg = dL_dtheta_rad * (np.pi / 180.0)
    return dL_dtheta_deg


# ---------- Main driver: compute FD for all parameters ----------
def main(args) -> None:
    renderer_settings = {
        "photons": 1e3,
        "bounces": 4,
        "forward_passes": 100,
        "gather_passes": 1,
        "adjoint_bounces": 2,
        "adjoint_passes": 4,
        "logging": 3,
        "debug_images": True,
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
    target_scales_np = target_params["scale"]
    target_albedos_np = target_params["albedo"]
    target_opacities_np = target_params["opacity"]
    target_betas_np = target_params["beta"]

    apply_noise = False
    if apply_noise:
        (
            noisy_positions_np,
            noisy_tangent_u_np,
            noisy_tangent_v_np,
            noisy_scales_np,
            noisy_albedos_np,
            noisy_opacities_np,
            noisy_betas_np,
        ) = add_debug_noise_to_initial_parameters(
            target_positions_np,
            target_tangent_u_np,
            target_tangent_v_np,
            target_scale_np,
            target_albedo_np,
            target_opacity_np,
            target_beta_np,
        )
        print("Initial parameters perturbed by debug Gaussian noise.")

    if args.param == "translation":
        eps_trans = 0.1
        if args.axis == "x":
            noise = [eps_trans, 0.0, 0.0]
        elif args.axis == "y":
            noise = [0.0, eps_trans, 0.0]
        else:
            noise = [0.0, 0.0, eps_trans]

        target_positions_np[args.index] = target_positions_np[args.index] + np.array(noise,
                                                                                         dtype=np.float32)
    elif args.param == "rotation":
        # Example: small rotation (e.g. for FD) around given axis
        noise_deg = 10
        qx_minus, qy_minus, qz_minus, qw_minus = degrees_to_quaternion(noise_deg, args.axis)
        quaternion_minus = np.array([qx_minus, qy_minus, qz_minus, qw_minus],
                                    dtype=np.float32)

        # Rotate tangents at this index in-place
        target_tangent_u_np[args.index] = apply_quaternion_to_vector(
            quaternion_minus,
            target_tangent_u_np[args.index],
        )
        target_tangent_v_np[args.index] = apply_quaternion_to_vector(
            quaternion_minus,
            target_tangent_v_np[args.index],
        )

    elif args.param == "scale":
        scale_percent = 20
        eps_scale = 1.0 + scale_percent / 100
        if args.axis == "x":
            noise = [eps_scale, 1.0]
        else:
            noise = [1.0, eps_scale]

        target_scales_np[args.index] = target_scales_np[args.index] * np.array(noise,dtype=np.float32)

    elif args.param == "opacity":
        eps_opacity = 0.3
        target_opacities_np[args.index] = target_opacities_np[args.index] + np.array(eps_opacity, dtype=np.float32)


    if args.param == "albedo":
        eps_albedo = -0.3
        if args.axis == "x":
            noise = [eps_albedo, 0.0, 0.0]
        elif args.axis == "y":
            noise = [0.0, eps_albedo, 0.0]
        else:
            noise = [0.0, 0.0, eps_albedo]
        target_albedos_np[args.index] = target_albedos_np[args.index] + np.array(noise,
                                                                                     dtype=np.float32)

    elif args.param == "beta":
        eps_beta = -0.3
        target_betas_np[args.index] = target_betas_np[args.index] + np.array(eps_beta, dtype=np.float32)

    renderer.apply_point_optimization(
        {
            "position": target_positions_np,
            "tangent_u": target_tangent_u_np,
            "tangent_v": target_tangent_v_np,
            "scale": target_scales_np,
            "albedo": target_albedos_np,
            "opacity": target_opacities_np,
            "beta": target_betas_np
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

    analytical_gradients_position = np.asarray(
        gradients["position"],
        dtype=np.float32,
        order="C",
    )

    analytical_gradients_tangent_u = np.asarray(
        gradients["tangent_u"],
        dtype=np.float32,
        order="C",
    )

    analytical_gradients_tangent_v = np.asarray(
        gradients["tangent_v"],
        dtype=np.float32,
        order="C",
    )

    analytical_gradients_scale = np.asarray(
        gradients["scale"],
        dtype=np.float32,
        order="C",
    )

    analytical_gradients_opacity = np.asarray(
        gradients["opacity"],
        dtype=np.float32,
        order="C",
    )

    analytical_gradients_beta = np.asarray(
        gradients["beta"],
        dtype=np.float32,
        order="C",
    )

    analytical_gradients_albedo = np.asarray(
        gradients["albedo"],
        dtype=np.float32,
        order="C",
    )

    # Main adjoint source image per camera
    grad_image_numpy = np.asarray(
        adjoint_images["adjoint_source"][camera],
        dtype=np.float32,
        order="C",
    )
    grad_image_numpy = np.nan_to_num(
        grad_image_numpy,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    grad_path = base_output_dir / "grad_099.png"
    save_gradient_sign_png_py(
        grad_path,
        grad_image_numpy,
        adjoint_spp=1,
        abs_quantile=0.999,
        flip_y=False,
    )

    # Example: debug position gradient image
    if "debug" in adjoint_images and camera in adjoint_images["debug"]:
        for axis in ["x", "y", "z"]:
            if f"position_{axis}" in adjoint_images["debug"][camera]:
                debug_pos_img = np.asarray(
                    adjoint_images["debug"][camera][f"position_{axis}"],
                    dtype=np.float32,
                    order="C",
                )
                debug_pos_img = np.nan_to_num(
                    debug_pos_img,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                debug_pos_path = base_output_dir / f"{axis}_position_debug_099.png"
                save_gradient_sign_png_py(
                    debug_pos_path,
                    debug_pos_img,
                    adjoint_spp=1,
                    abs_quantile=0.999,
                    flip_y=False,
                )

    # Step sizes
    eps_translation = 0.01
    eps_rotation_deg = 0.5
    eps_scale = 0.05
    eps_opacity = 0.1
    eps_albedo = 0.1
    eps_beta = 0.1  # reasonable start for log-shape

    iterations = 6

    label = args.param + "_" + args.axis + "_" + str(args.index)

    if args.param == "translation":
        # --- Translation (x,y,z) ---
        grad_trans = {"x": 0.0, "y": 0.0, "z": 0.0}
        grad_trans_samples = {"x": [], "y": [], "z": []}

        for i in range(iterations):
            print(f"[FD] Computing Translation axis iteration={i + 1}/{iterations}...")
            for axis in ["x", "y", "z"]:
                rgb_minus, rgb_plus, grad_fd = finite_difference_translation(
                    renderer, args.index, axis, eps_translation
                )
                # calculate loss for each image
                loss_negative = compute_l2_loss(rgb_minus, target_image)
                loss_positive = compute_l2_loss(rgb_plus, target_image)

                grad_value = (loss_positive - loss_negative) / (2.0 * eps_translation)

                # accumulate for mean
                grad_trans[axis] += grad_value
                # store sample for std
                grad_trans_samples[axis].append(grad_value)

                if i == 0:
                    save_rgb_preview_png(
                        rgb_minus, base_output_dir / "translation" / f"{axis}_neg.png"
                    )
                    save_rgb_preview_png(
                        rgb_plus, base_output_dir / "translation" / f"{axis}_pos.png"
                    )
                    write_fd_images(
                        grad_fd,
                        rgb_minus,
                        rgb_plus,
                        base_output_dir / "translation",
                        "pos",
                        axis,
                    )

        # compute means
        for axis in ["x", "y", "z"]:
            grad_trans[axis] /= iterations

        # compute mean and std per axis from samples
        grad_trans_stats = {}
        for axis in ["x", "y", "z"]:
            samples = np.array(grad_trans_samples[axis], dtype=np.float64)
            mean_val = samples.mean()
            # ddof=1 for sample std; use ddof=0 if you want population std
            std_val = samples.std(ddof=1) if iterations > 1 else 0.0
            grad_trans_stats[axis] = {"mean": float(mean_val), "std": float(std_val)}

        axes = ["x", "y", "z"]
        print()
        for axis_index, axis_name in enumerate(axes):
            fd_val = float(grad_trans[axis_name])
            an_val = float(analytical_gradients_position[args.index][axis_index])
            err = fd_val - an_val
            abs_err = abs(err)

            mean_val = grad_trans_stats[axis_name]["mean"]
            std_val = grad_trans_stats[axis_name]["std"]
            print(
                f"[FD] Translation axis={axis_name}, "
                f"Mean Grad={mean_val: .10f}, Std={std_val: .10f}"
            )

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
            run_label=label,
            loss_value=loss_value,
            grad_trans=grad_trans,
            analytical_gradients_position=analytical_gradients_position[args.index],
        )
        print_history_3axis(history_path)


    elif args.param == "rotation":
        # --- Rotation (x,y,z) ---
        grad_rot = {"x": 0.0, "y": 0.0, "z": 0.0}
        grad_rot_samples = {"x": [], "y": [], "z": []}
        for i in range(iterations):
            print(f"[FD] Computing Rotation axis iteration={i + 1}/{iterations}...")
            for axis in ["x", "y", "z"]:
                rgb_minus, rgb_plus, grad_fd = finite_difference_rotation(
                    renderer, args.index, axis, eps_rotation_deg
                )
                loss_negative = compute_l2_loss(rgb_minus, target_image)
                loss_positive = compute_l2_loss(rgb_plus, target_image)
                grad_value = (loss_positive - loss_negative) / (2.0 * eps_rotation_deg)
                grad_rot[axis] += grad_value
                grad_rot_samples[axis].append(grad_value)

                if i == 0:
                    save_rgb_preview_png(
                        rgb_minus, base_output_dir / "rotation" / f"{axis}_neg.png"
                    )
                    save_rgb_preview_png(
                        rgb_plus, base_output_dir / "rotation" / f"{axis}_pos.png"
                    )
                    write_fd_images(
                        grad_fd,
                        rgb_minus,
                        rgb_plus,
                        base_output_dir / "rotation",
                        "rot",
                        axis,
                    )
        # compute means
        for axis in ["x", "y", "z"]:
            grad_rot[axis] /= iterations
        # compute mean/std
        grad_rot_stats = {}
        for axis in ["x", "y", "z"]:
            samples = np.array(grad_rot_samples[axis], dtype=np.float64)
            mean_val = samples.mean()
            std_val = samples.std(ddof=1) if iterations > 1 else 0.0
            grad_rot_stats[axis] = {"mean": float(mean_val), "std": float(std_val)}
        axes = ["x", "y", "z"]
        print()
        for axis_index, axis_name in enumerate(axes):
            fd_val = float(grad_rot[axis_name])
            # tangents and their grads at this Gaussian
            t_u = target_tangent_u_np[args.index]
            t_v = target_tangent_v_np[args.index]
            g_tu = analytical_gradients_tangent_u[args.index]
            g_tv = analytical_gradients_tangent_v[args.index]
            # analytical dL/d(theta_deg) for this axis
            an_val = analytic_rotation_grad_for_axis_deg(
                axis_name,
                t_u,
                t_v,
                g_tu,
                g_tv,

            )

            err = fd_val - an_val
            abs_err = abs(err)
            mean_val = grad_rot_stats[axis_name]["mean"]
            std_val = grad_rot_stats[axis_name]["std"]
            print(
                f"[FD] Rotation axis={axis_name}, "
                f"Mean Grad={mean_val: .10f}, Std={std_val: .10f}"
            )
            print(f"[FD] Rotation axis={axis_name}, Grad={fd_val: .10f}")
            print(f"[AN] Rotation axis={axis_name}, Grad={an_val: .10f}")
            print(
                f"[ER] Rotation axis={axis_name}, "
                f"FD-AN={err: .10f}, |FD-AN|={abs_err: .10f}"
            )
            print()
        # ----------------------------------------------------------
        # Save this run to history and print all runs so far
        # ----------------------------------------------------------
        analytical_rot_grad = {}
        for axis_name in ["x", "y", "z"]:
            analytical_rot_grad[axis_name] = analytic_rotation_grad_for_axis_deg(
                axis_name,
                target_tangent_u_np[args.index],
                target_tangent_v_np[args.index],
                analytical_gradients_tangent_u[args.index],
                analytical_gradients_tangent_v[args.index],
            )

        history_path = base_output_dir / "rotation_gradients_history.csv"
        append_rotation_run_to_history(
            history_path=history_path,
            scene_name=args.scene,
            gaussian_index=args.index,
            run_label=label,
            loss_value=loss_value,
            grad_rot=grad_rot,
            analytical_rot_grad=analytical_rot_grad,
        )

        print_history_3axis(history_path)

    if args.param == "scale":
        # --- Scale (x,y,z) ---
        grad_scale = {"x": 0.0, "y": 0.0}
        grad_scale_samples = {"x": [], "y": []}

        for i in range(iterations):
            print(f"[FD] Computing Scale axis iteration={i + 1}/{iterations}...")
            for axis in ["x", "y"]:
                rgb_minus, rgb_plus, grad_fd = finite_difference_scale(
                    renderer, args.index, axis, eps_scale
                )
                # calculate loss for each image
                loss_negative = compute_l2_loss(rgb_minus, target_image)
                loss_positive = compute_l2_loss(rgb_plus, target_image)

                grad_value = (loss_positive - loss_negative) / (2.0 * eps_scale)

                # accumulate for mean
                grad_scale[axis] += grad_value
                # store sample for std
                grad_scale_samples[axis].append(grad_value)

                if i == 0:
                    save_rgb_preview_png(
                        rgb_minus, base_output_dir / "scale" / f"{axis}_neg.png"
                    )
                    save_rgb_preview_png(
                        rgb_plus, base_output_dir / "scale" / f"{axis}_pos.png"
                    )
                    write_fd_images(
                        grad_fd,
                        rgb_minus,
                        rgb_plus,
                        base_output_dir / "scale",
                        "scale",
                        axis,
                    )

        # compute means
        for axis in ["x", "y"]:
            grad_scale[axis] /= iterations

        # compute mean and std per axis from samples
        grad_scale_stats = {}
        for axis in ["x", "y"]:
            samples = np.array(grad_scale_samples[axis], dtype=np.float64)
            mean_val = samples.mean()
            # ddof=1 for sample std; use ddof=0 if you want population std
            std_val = samples.std(ddof=1) if iterations > 1 else 0.0
            grad_scale_stats[axis] = {"mean": float(mean_val), "std": float(std_val)}

        axes = ["x", "y"]
        print()
        for axis_index, axis_name in enumerate(axes):
            fd_val = float(grad_scale[axis_name])
            an_val = float(analytical_gradients_position[args.index][axis_index])
            err = fd_val - an_val
            abs_err = abs(err)

            mean_val = grad_scale_stats[axis_name]["mean"]
            std_val = grad_scale_stats[axis_name]["std"]
            print(
                f"[FD] Scale axis={axis_name}, "
                f"Mean Grad={mean_val: .10f}, Std={std_val: .10f}"
            )

            print(f"[FD] Scale axis={axis_name}, Grad={fd_val: .10f}")
            print(f"[AN] Scale axis={axis_name}, Grad={an_val: .10f}")
            print(
                f"[ER] Scale axis={axis_name}, "
                f"FD-AN={err: .10f}, |FD-AN|={abs_err: .10f}"
            )
            print()

        # ----------------------------------------------------------
        # Save this run to history and print all runs so far
        # ----------------------------------------------------------
        history_path = base_output_dir / "scale_gradients_history.csv"
        append_scale_run_to_history(
            history_path=history_path,
            scene_name=args.scene,
            gaussian_index=args.index,
            run_label=label,
            loss_value=loss_value,
            grad_scale=grad_scale,
            analytical_gradients_scale=analytical_gradients_scale[args.index],
        )
        print_history_2axis(history_path)

    if args.param == "opacity":
        # --- Scale (x,y,z) ---
        grad_opacity = 0.0
        grad_opacity_samples = []

        for i in range(iterations):
            print(f"[FD] Computing Opacity iteration={i + 1}/{iterations}...")
            rgb_minus, rgb_plus, grad_fd = finite_difference_opacity(
                renderer, args.index, eps_opacity
            )
            # calculate loss for each image
            loss_negative = compute_l2_loss(rgb_minus, target_image)
            loss_positive = compute_l2_loss(rgb_plus, target_image)

            grad_value = (loss_positive - loss_negative) / (2.0 * eps_opacity)

            # accumulate for mean
            grad_opacity += grad_value
            # store sample for std
            grad_opacity_samples.append(grad_value)

            if i == 0:
                save_rgb_preview_png(
                    rgb_minus, base_output_dir / "opacity" / f"neg.png"
                )
                save_rgb_preview_png(
                    rgb_plus, base_output_dir / "opacity" / f"pos.png"
                )
                write_fd_images(
                    grad_fd,
                    rgb_minus,
                    rgb_plus,
                    base_output_dir / "opacity",
                    "opacity",
                    "",
                )


        grad_opacity /= iterations

        # compute mean and std per axis from samples
        samples = np.array(grad_opacity_samples, dtype=np.float64)
        mean_val = samples.mean()
        # ddof=1 for sample std; use ddof=0 if you want population std
        std_val = samples.std(ddof=1) if iterations > 1 else 0.0
        grad_opacity_stats = {"mean": float(mean_val), "std": float(std_val)}

        print()
        fd_val = float(grad_opacity)
        an_val = float(analytical_gradients_opacity[args.index])
        err = fd_val - an_val
        abs_err = abs(err)

        mean_val = grad_opacity_stats["mean"]
        std_val = grad_opacity_stats["std"]
        print(
            f"[FD] Mean Opacity Grad={mean_val: .10f}, Std={std_val: .10f}"
        )

        print(f"[FD] Opacity Grad={fd_val: .10f}")
        print(f"[AN] Opacity Grad={an_val: .10f}")
        print(
            f"[ER] FD-AN={err: .10f}, |FD-AN|={abs_err: .10f}"
        )
        print()

        # ----------------------------------------------------------
        # Save this run to history and print all runs so far
        # ----------------------------------------------------------
        history_path = base_output_dir / "opacity_gradients_history.csv"
        append_1d_run_to_history(
            history_path=history_path,
            scene_name=args.scene,
            gaussian_index=args.index,
            run_label=label,
            loss_value=loss_value,
            fd_grad=grad_opacity,
            an_grad=analytical_gradients_opacity[args.index],
        )
        print_history_1axis(history_path)


    if args.param == "beta":
        # --- Scale (x,y,z) ---
        grad_beta = 0.0
        grad_beta_samples = []

        for i in range(iterations):
            print(f"[FD] Computing Beta iteration={i + 1}/{iterations}...")
            rgb_minus, rgb_plus, grad_fd = finite_difference_beta(
                renderer, args.index, eps_beta
            )
            # calculate loss for each image
            loss_negative = compute_l2_loss(rgb_minus, target_image)
            loss_positive = compute_l2_loss(rgb_plus, target_image)

            grad_value = (loss_positive - loss_negative) / (2.0 * eps_beta)

            # accumulate for mean
            grad_beta += grad_value
            # store sample for std
            grad_beta_samples.append(grad_value)

            if i == 0:
                save_rgb_preview_png(
                    rgb_minus, base_output_dir / "beta" / f"neg.png"
                )
                save_rgb_preview_png(
                    rgb_plus, base_output_dir / "beta" / f"pos.png"
                )
                write_fd_images(
                    grad_fd,
                    rgb_minus,
                    rgb_plus,
                    base_output_dir / "beta",
                    "beta",
                    "",
                )


        grad_beta /= iterations

        # compute mean and std per axis from samples
        samples = np.array(grad_beta_samples, dtype=np.float64)
        mean_val = samples.mean()
        # ddof=1 for sample std; use ddof=0 if you want population std
        std_val = samples.std(ddof=1) if iterations > 1 else 0.0
        grad_beta_stats = {"mean": float(mean_val), "std": float(std_val)}

        print()
        fd_val = float(grad_beta)
        an_val = float(analytical_gradients_beta[args.index])
        err = fd_val - an_val
        abs_err = abs(err)

        mean_val = grad_beta_stats["mean"]
        std_val = grad_beta_stats["std"]
        print(
            f"[FD] Mean Opacity Grad={mean_val: .10f}, Std={std_val: .10f}"
        )

        print(f"[FD] Opacity Grad={fd_val: .10f}")
        print(f"[AN] Opacity Grad={an_val: .10f}")
        print(
            f"[ER] FD-AN={err: .10f}, |FD-AN|={abs_err: .10f}"
        )
        print()

        # ----------------------------------------------------------
        # Save this run to history and print all runs so far
        # ----------------------------------------------------------
        history_path = base_output_dir / "beta_gradients_history.csv"
        append_1d_run_to_history(
            history_path=history_path,
            scene_name=args.scene,
            gaussian_index=args.index,
            run_label=label,
            loss_value=loss_value,
            fd_grad=grad_beta,
            an_grad=analytical_gradients_beta[args.index],
        )
        print_history_1axis(history_path)

    if args.param == "albedo":
        # --- Albedo (x,y,z) ---
        grad_albedo = {"x": 0.0, "y": 0.0, "z": 0.0}
        grad_albedo_samples = {"x": [], "y": [], "z": []}

        for i in range(iterations):
            print(f"[FD] Computing Albedo axis iteration={i + 1}/{iterations}...")
            for axis in ["x", "y", "z"]:
                rgb_minus, rgb_plus, grad_fd = finite_difference_albedo(
                    renderer, args.index, axis, eps_albedo
                )
                # calculate loss for each image
                loss_negative = compute_l2_loss(rgb_minus, target_image)
                loss_positive = compute_l2_loss(rgb_plus, target_image)

                grad_value = (loss_positive - loss_negative) / (2.0 * eps_albedo)

                # accumulate for mean
                grad_albedo[axis] += grad_value
                # store sample for std
                grad_albedo_samples[axis].append(grad_value)

                if i == 0:
                    save_rgb_preview_png(
                        rgb_minus, base_output_dir / "albedo" / f"{axis}_neg.png"
                    )
                    save_rgb_preview_png(
                        rgb_plus, base_output_dir / "albedo" / f"{axis}_pos.png"
                    )
                    write_fd_images(
                        grad_fd,
                        rgb_minus,
                        rgb_plus,
                        base_output_dir / "albedo",
                        "albedo",
                        axis,
                    )

        # compute means
        for axis in ["x", "y", "z"]:
            grad_albedo[axis] /= iterations

        # compute mean and std per axis from samples
        grad_albedo_stats = {}
        for axis in ["x", "y", "z"]:
            samples = np.array(grad_albedo_samples[axis], dtype=np.float64)
            mean_val = samples.mean()
            # ddof=1 for sample std; use ddof=0 if you want population std
            std_val = samples.std(ddof=1) if iterations > 1 else 0.0
            grad_albedo_stats[axis] = {"mean": float(mean_val), "std": float(std_val)}

        axes = ["x", "y", "z"]
        print()
        for axis_index, axis_name in enumerate(axes):
            fd_val = float(grad_albedo[axis_name])
            an_val = float(analytical_gradients_albedo[args.index][axis_index])
            err = fd_val - an_val
            abs_err = abs(err)

            mean_val = grad_albedo_stats[axis_name]["mean"]
            std_val = grad_albedo_stats[axis_name]["std"]
            print(
                f"[FD] Translation axis={axis_name}, "
                f"Mean Grad={mean_val: .10f}, Std={std_val: .10f}"
            )

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
        history_path = base_output_dir / "albedo_gradients_history.csv"
        append_translation_run_to_history(
            history_path=history_path,
            scene_name=args.scene,
            gaussian_index=args.index,
            run_label=label,
            loss_value=loss_value,
            grad_trans=grad_albedo,
            analytical_gradients_position=analytical_gradients_albedo[args.index],
        )
        print_history_3axis(history_path)

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
        required=True,
        default=0,
        help="Gaussian index to perturb. Default: 0",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (relative to this script). Default: Output/<scene>/finite_diff",
    )

    parser.add_argument(
        "--param",
        type=str,
        required=True,
        default="translation",
        help="parameter",
    )
    parser.add_argument(
        "--axis",
        type=str,
        required=False,
        default="x",
        help="parameter axis to test",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
