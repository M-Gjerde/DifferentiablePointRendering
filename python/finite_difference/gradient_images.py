#!/usr/bin/env python3
import time
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
import pale
from matplotlib import cm


# ---------- I/O helpers ----------
def save_rgb_float32_npy(img_f32: np.ndarray, out_path: Path) -> None:
    np.save(out_path, img_f32.astype(np.float32))


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
    print(f"Saving RGB preview to: {out_path.absolute()}")
    Image.fromarray(img_u8, mode="RGB").save(out_path)


def save_color_luma_seismic_black(scalar: np.ndarray, out_png: Path) -> None:
    """Blue→Black→Red diverging map for signed scalars with zero at black."""
    from matplotlib.colors import LinearSegmentedColormap
    blue_black_red = LinearSegmentedColormap.from_list(
        "blue_black_red",
        [(0.0, (0.0, 0.0, 1.0)),
         (0.5, (0.0, 0.0, 0.0)),
         (1.0, (1.0, 0.0, 0.0))],
        N=256,
    )

    scalar = np.asarray(scalar, dtype=np.float32)
    finite_mask = np.isfinite(scalar)
    if not np.any(finite_mask):
        vis = np.zeros((*scalar.shape, 3), dtype=np.uint8)
    else:
        max_abs = np.max(np.abs(scalar[finite_mask]))
        max_abs = max(max_abs, 1e-12)
        norm_signed = np.zeros_like(scalar, dtype=np.float32)
        norm_signed[finite_mask] = scalar[finite_mask] / max_abs
        mapped_01 = (norm_signed + 1.0) * 0.5
        vis_float = blue_black_red(mapped_01)[..., :3]
        vis_float[~finite_mask] = (0.0, 0.0, 0.0)
        vis = (vis_float * 255.0).astype(np.uint8)

    Image.fromarray(vis.astype(np.uint8)).save(out_png)


def save_color_luma_seismic_white(scalar: np.ndarray, out_png: Path) -> None:
    """Blue→White→Red diverging map for signed scalars with zero at white."""
    from matplotlib.colors import LinearSegmentedColormap

    blue_white_red = LinearSegmentedColormap.from_list(
        "blue_white_red",
        [(0.0, (0.0, 0.0, 1.0)),
         (0.5, (1.0, 1.0, 1.0)),
         (1.0, (1.0, 0.0, 0.0))],
        N=256,
    )

    scalar = np.asarray(scalar, dtype=np.float32)
    finite_mask = np.isfinite(scalar)
    if not np.any(finite_mask):
        vis = np.ones((*scalar.shape, 3), dtype=np.uint8) * 255
    else:
        max_abs = np.max(np.abs(scalar[finite_mask]))
        max_abs = max(max_abs, 1e-12)
        norm_signed = np.zeros_like(scalar, dtype=np.float32)
        norm_signed[finite_mask] = scalar[finite_mask] / max_abs
        mapped_01 = (norm_signed + 1.0) * 0.5
        vis_float = blue_white_red(mapped_01)[..., :3]
        vis_float[~finite_mask] = (1.0, 1.0, 1.0)
        vis = (vis_float * 255.0).astype(np.uint8)

    Image.fromarray(vis.astype(np.uint8)).save(out_png)


# ---------- Robust signed normalization + visualizers ----------
def robustSignedNormalize(
    signed_array: np.ndarray,
    mode: str = "percentile",
    percentile: float = 0.995,
    mad_clip: float | None = None,
    global_scale: float | None = None,
    exclude_zero: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Normalize a signed array to [-1, 1] using a symmetric scale around zero.
    Returns (normalized, scale_used).
    """
    s = np.asarray(signed_array, dtype=np.float32)
    finite_mask = np.isfinite(s)
    if not np.any(finite_mask):
        return np.zeros_like(s, dtype=np.float32), 1.0

    work = s[finite_mask]
    if exclude_zero:
        nonzero = work != 0.0
        work = work[nonzero] if np.any(nonzero) else work

    if global_scale is not None and np.isfinite(global_scale) and global_scale > 0.0:
        scale = float(global_scale)
    elif mode == "percentile":
        q = float(np.clip(percentile, 0.5, 1.0))
        scale = float(np.quantile(np.abs(work), q))
    elif mode == "mad":
        med = float(np.median(work))
        mad = float(np.median(np.abs(work - med)))
        scale = 1.4826 * mad
        if mad_clip is not None and scale > 0:
            hi = med + mad_clip * scale
            lo = med - mad_clip * scale
            work = np.clip(work, lo, hi)
    elif mode == "std":
        scale = float(np.std(work))
    else:
        raise ValueError("mode must be 'percentile', 'mad', or 'std'")

    if not (np.isfinite(scale) and scale > 0.0):
        scale = 1.0

    norm = np.zeros_like(s, dtype=np.float32)
    norm[finite_mask] = np.clip(s[finite_mask] / scale, -1.0, 1.0)
    return norm, scale


def saveSeismicSignedRobust(
    signed_array: np.ndarray,
    out_png: Path,
    mode: str = "percentile",
    percentile: float = 0.995,
    global_scale: float | None = None,
) -> float:
    """Seismic colormap with symmetric robust normalization. Returns scale used."""
    norm, scale = robustSignedNormalize(
        signed_array, mode=mode, percentile=percentile, global_scale=global_scale
    )
    t = 0.5 * (norm + 1.0)
    rgba = cm.get_cmap("seismic")(t)
    rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgb).save(out_png)
    return scale


def saveSignedLogCompress(
    signed_array: np.ndarray,
    out_png: Path,
    epsilon: float = 1e-6,
    percentile: float = 0.995,
) -> float:
    """
    Log-compress the magnitude before coloring to expand tiny ranges.
    Returns the reference scale used for normalization.
    """
    s = np.asarray(signed_array, dtype=np.float32)
    finite = np.isfinite(s)
    mags = np.abs(s[finite])
    if mags.size:
        scale = float(np.quantile(mags, np.clip(percentile, 0.5, 1.0)))
    else:
        scale = 1.0
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0

    comp = np.zeros_like(s, dtype=np.float32)
    denom = np.log1p(scale / max(epsilon, 1e-12))
    denom = denom if denom > 0 else 1.0
    comp[finite] = (
        np.sign(s[finite])
        * np.log1p(np.abs(s[finite]) / max(epsilon, 1e-12))
        / denom
    )
    comp = np.clip(comp, -1.0, 1.0)

    t = 0.5 * (comp + 1.0)
    rgba = cm.get_cmap("seismic")(t)
    rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgb).save(out_png)
    return scale


# ---------- TRS helpers ----------
def degrees_to_quaternion(deg: float, axis: str):
    """Quaternion (x, y, z, w) for rotation of `deg` degrees around axis {'x','y','z'}."""
    rad = np.deg2rad(deg)
    half = rad * 0.5
    sin_h = np.sin(half)
    cos_h = np.cos(half)

    axis = axis.lower()
    if axis == "x":
        return (sin_h, 0.0, 0.0, cos_h)
    elif axis == "y":
        return (0.0, sin_h, 0.0, cos_h)
    elif axis == "z":
        return (0.0, 0.0, sin_h, cos_h)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


def render_with_trs(
    renderer,
    translation3,
    rotation_quat4,
    scale3,
    color3,
    opacity,
    beta = 0.0,
    index = -1,
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


def finite_difference_rotation(
    renderer, index: int, axis: str, eps_degrees: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qx_minus, qy_minus, qz_minus, qw_minus = degrees_to_quaternion(-eps_degrees, axis)
    qx_plus, qy_plus, qz_plus, qw_plus = degrees_to_quaternion(+eps_degrees, axis)

    rgb_minus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(qx_minus, qy_minus, qz_minus, qw_minus),
        scale3=(1.0, 1.0, 1.0),
        color3=(0.0, 0.0, 0.0),
        opacity=0.0,
        beta=0.0,
        index=index,
    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(qx_plus, qy_plus, qz_plus, qw_plus),
        scale3=(1.0, 1.0, 1.0),
        color3=(0.0, 0.0, 0.0),
        opacity=0.0,
        beta=0.0,
        index=index,
    )
    grad = (rgb_plus - rgb_minus) / (2.0 * eps_degrees)
    return rgb_minus, rgb_plus, grad


def finite_difference_scale(
    renderer, index: int, axis: str, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axis == "x":
        negative_scale = (1.0 - eps, 1.0, 1.0)
        positive_scale = (1.0 + eps, 1.0, 1.0)
    elif axis == "y":
        negative_scale = (1.0, 1.0 - eps, 1.0)
        positive_scale = (1.0, 1.0 + eps, 1.0)
    elif axis == "z":
        negative_scale = (1.0, 1.0, 1.0 - eps)
        positive_scale = (1.0, 1.0, 1.0 + eps)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    rgb_minus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=negative_scale,
        color3=(0.0, 0.0, 0.0),
        opacity=0.0,
        beta=0.0,
        index=index,
    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=positive_scale,
        color3=(0.0, 0.0, 0.0),
        opacity=0.0,
        beta=0.0,
        index=index,
    )
    grad = (rgb_plus - rgb_minus) / (2.0 * eps)
    return rgb_minus, rgb_plus, grad


def finite_difference_opacity(
    renderer, index: int, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    negative_opacity = -eps
    positive_opacity = +eps

    rgb_minus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        color3=(0.0, 0.0, 0.0),
        opacity=negative_opacity,
        beta=0.0,
        index=index,
    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        color3=(0.0, 0.0, 0.0),
        opacity=positive_opacity,
        beta=0.0,
        index=index,
    )
    grad = (rgb_plus - rgb_minus) / (2.0 * eps)
    return rgb_minus, rgb_plus, grad


def finite_difference_albedo(
    renderer, index: int, channel: str, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    channel in {'R','G','B'}: we perturb that color channel while keeping others at zero.
    Opacity is set to 1.0 so albedo changes have visible effect.
    """
    channel = channel.upper()
    if channel == "R":
        color_minus = (-eps, 0.0, 0.0)
        color_plus = (+eps, 0.0, 0.0)
    elif channel == "G":
        color_minus = (0.0, -eps, 0.0)
        color_plus = (0.0, +eps, 0.0)
    elif channel == "B":
        color_minus = (0.0, 0.0, -eps)
        color_plus = (0.0, 0.0, +eps)
    else:
        raise ValueError("channel must be 'R', 'G', or 'B'")

    rgb_minus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        color3=color_minus,
        opacity=1.0,
        beta=0.0,
        index=index,
    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        color3=color_plus,
        opacity=1.0,
        beta=0.0,
        index=index,
    )
    grad = (rgb_plus - rgb_minus) / (2.0 * eps)
    return rgb_minus, rgb_plus, grad

def finite_difference_beta(
    renderer, index: int, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Central finite differences for the beta parameter (log-shape).
    b = 4 * exp(beta); here we perturb beta by ±eps.
    """
    beta_minus = -eps
    beta_plus = +eps

    rgb_minus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        color3=(0.0, 0.0, 0.0),
        opacity=1.0,
        beta=beta_minus,
        index=index,
    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        color3=(0.0, 0.0, 0.0),
        opacity=1.0,
        beta=beta_plus,
        index=index,
    )
    grad = (rgb_plus - rgb_minus) / (2.0 * eps)
    return rgb_minus, rgb_plus, grad



# ---------- Visualization for one FD gradient tensor ----------
def write_fd_images(
    grad_disp_fd: np.ndarray,
    rgb_minus: np.ndarray,
    rgb_plus: np.ndarray,
    out_dir: Path,
    param_name: str,
    axis_or_channel: str,
) -> None:
    """
    grad_disp_fd: H x W x 3, finite-difference gradient in display-space RGB.
    Writes:
      - minus/plus previews
      - R/G/B seismic PNGs (full & q=0.99)
      - luminance seismic PNGs (full & q=0.99)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    channel_names = ["R", "G", "B"]

    # Previews
    save_rgb_preview_png(rgb_minus, out_dir / f"{param_name}_{axis_or_channel}_minus.png")
    save_rgb_preview_png(rgb_plus,  out_dir / f"{param_name}_{axis_or_channel}_plus.png")

    # Simple seismic helper (same as earlier script)
    def save_seismic_signed(scalar: np.ndarray, out_png: Path, abs_quantile: float = 1.0) -> None:
        scalar_array = np.asarray(scalar, dtype=np.float32)
        finite_mask = np.isfinite(scalar_array)
        if not np.any(finite_mask):
            Image.fromarray(np.zeros((*scalar_array.shape, 3), dtype=np.uint8)).save(out_png)
            return
        magnitudes = np.abs(scalar_array[finite_mask])
        q = np.clip(abs_quantile, 0.0, 1.0)
        if q < 1.0:
            scale_value = np.quantile(magnitudes, q)
        else:
            scale_value = magnitudes.max()
        if not (np.isfinite(scale_value) and scale_value > 0.0):
            scale_value = 1.0
        normalized = np.clip(scalar_array / scale_value, -1.0, 1.0)
        t = 0.5 * (normalized + 1.0)
        rgba = cm.get_cmap("seismic")(t)
        rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
        rgb[~finite_mask] = (255, 255, 255)
        Image.fromarray(rgb).save(out_png)

    # Per-channel outputs
    for c_idx, cname in enumerate(channel_names):
        grad_c = grad_disp_fd[..., c_idx]  # H x W

        save_seismic_signed(
            grad_c,
            out_dir / f"{param_name}_{axis_or_channel}_grad_{cname}_seismic.png",
            abs_quantile=1.0,
        )
        save_seismic_signed(
            grad_c,
            out_dir / f"{param_name}_{axis_or_channel}_grad_{cname}_seismic_q099.png",
            abs_quantile=0.99,
        )

    # Luminance
    luminance_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma_grad = np.tensordot(grad_disp_fd, luminance_weights, axes=([2], [0]))

    save_seismic_signed(
        luma_grad,
        out_dir / f"{param_name}_{axis_or_channel}_grad_L_seismic.png",
        abs_quantile=1.0,
    )
    save_seismic_signed(
        luma_grad,
        out_dir / f"{param_name}_{axis_or_channel}_grad_L_seismic_q099.png",
        abs_quantile=0.99,
    )


# ---------- Main driver: compute FD for all parameters ----------
def main(args) -> None:
    renderer_settings = {
        "photons": 1e4,
        "bounces": 4,
        "forward_passes": 1000,
        "gather_passes": 1,
        "adjoint_bounces": 0,
        "adjoint_passes": 0,
        "logging": 2,
    }

    assets_root = Path(__file__).parent.parent.parent / "Assets"
    scene_xml = "cbox_custom.xml"
    pointcloud_ply = args.scene + ".ply"

    print("Assets root:", assets_root)
    print("Scene:", args.scene)
    print("Index:", args.index)

    base_output_dir = (
        Path(__file__).parent / "Output" / "beta_kernel" / args.scene
        if args.output == "" or args.output is None
        else Path(__file__).parent / Path(args.output)
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    renderer = pale.Renderer(str(assets_root), scene_xml, pointcloud_ply, renderer_settings)

    # Step sizes
    eps_translation = 0.01
    eps_rotation_deg = 0.5
    eps_scale = 0.05
    eps_opacity = 0.1
    eps_albedo = 0.1
    eps_beta = 0.5  # reasonable start for log-shape

    # --- Translation (x,y,z) ---
    for axis in ["x", "y", "z"]:
        print(f"[FD] Translation axis={axis}")
        rgb_minus, rgb_plus, grad_fd = finite_difference_translation(
            renderer, args.index, axis, eps_translation
        )
        out_dir = base_output_dir / "translation" / axis
        write_fd_images(grad_fd, rgb_minus, rgb_plus, out_dir, "translation", axis)


    # --- Rotation (x,y,z) ---
    for axis in ["y", "x", "z"]:
        print(f"[FD] Rotation axis={axis}")
        rgb_minus, rgb_plus, grad_fd = finite_difference_rotation(
            renderer, args.index, axis, eps_rotation_deg
        )
        out_dir = base_output_dir / "rotation" / axis
        write_fd_images(grad_fd, rgb_minus, rgb_plus, out_dir, "rotation", axis)







    # --- Scale (x,y,z) ---
    for axis in ["x", "y", "z"]:
        print(f"[FD] Scale axis={axis}")
        rgb_minus, rgb_plus, grad_fd = finite_difference_scale(
            renderer, args.index, axis, eps_scale
        )
        out_dir = base_output_dir / "scale" / axis
        write_fd_images(grad_fd, rgb_minus, rgb_plus, out_dir, "scale", axis)

    # --- Opacity (scalar, no axis) ---
    print("[FD] Opacity")
    rgb_minus, rgb_plus, grad_fd = finite_difference_opacity(
        renderer, args.index, eps_opacity
    )
    out_dir = base_output_dir / "opacity"
    write_fd_images(grad_fd, rgb_minus, rgb_plus, out_dir, "opacity", "scalar")

    # --- Albedo (R,G,B channels as separate parameters) ---
    for channel in ["R", "G", "B"]:
        print("[FD] Beta parameter")
        rgb_minus, rgb_plus, grad_fd = finite_difference_beta(
            renderer, args.index, eps_beta
        )
        out_dir = base_output_dir / "beta"
        write_fd_images(grad_fd, rgb_minus, rgb_plus, out_dir, "beta", "scalar")


        print(f"[FD] Albedo channel={channel}")
        rgb_minus, rgb_plus, grad_fd = finite_difference_albedo(
            renderer, args.index, channel, eps_albedo
        )
        out_dir = base_output_dir / "albedo" / channel
        write_fd_images(grad_fd, rgb_minus, rgb_plus, out_dir, "albedo", channel)

    print("Finite-difference image generation complete.")
    time.sleep(1)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-difference gradient visualization for Pale renderer (all parameters)."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="initial",
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
