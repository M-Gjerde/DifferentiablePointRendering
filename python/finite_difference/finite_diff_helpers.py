import numpy as np
from matplotlib import cm

from PIL import Image
from pathlib import Path


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
    # print(f"Saving RGB preview to: {out_path.absolute()}")
    img = Image.fromarray(img_u8).convert("RGB")
    img.save(out_path)


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
        albedo3,
        opacity,
        beta=0.0,
        index=-1,
        camera_name="camera1"
) -> np.ndarray:
    """Apply a full TRS+color+opacity+beta and render."""
    renderer.set_gaussian_transform(
        translation3=translation3,
        rotation_quat4=rotation_quat4,
        scale3=scale3,
        albedo3=albedo3,
        opacity=opacity,
        beta=beta,
        index=index,
    )

    renderer.rebuild_bvh()

    image = renderer.render_forward()[camera_name]
    rgb = np.asarray(image, dtype=np.float32)
    return rgb


# ---------- Finite difference core ----------
def finite_difference_translation(
        renderer, index: int, axis: str, eps: float, camera_name: str
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
        albedo3=(0.0, 0.0, 0.0),
        opacity=0.0,
        index=index,
        camera_name=camera_name
    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=positive_vector,
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        albedo3=(0.0, 0.0, 0.0),
        opacity=0.0,
        index=index,
        camera_name=camera_name

    )
    grad = (rgb_plus - rgb_minus) / (2.0 * eps)
    return rgb_minus, rgb_plus, grad


def finite_difference_rotation(
        renderer, index: int, axis: str, eps_degrees: float, camera_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qx_minus, qy_minus, qz_minus, qw_minus = degrees_to_quaternion(-eps_degrees, axis)
    qx_plus, qy_plus, qz_plus, qw_plus = degrees_to_quaternion(+eps_degrees, axis)

    rgb_minus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(qx_minus, qy_minus, qz_minus, qw_minus),
        scale3=(1.0, 1.0, 1.0),
        albedo3=(0.0, 0.0, 0.0),
        opacity=0.0,
        beta=0.0,
        index=index,
        camera_name=camera_name

    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(qx_plus, qy_plus, qz_plus, qw_plus),
        scale3=(1.0, 1.0, 1.0),
        albedo3=(0.0, 0.0, 0.0),
        opacity=0.0,
        beta=0.0,
        index=index,
        camera_name=camera_name

    )
    grad = (rgb_plus - rgb_minus) / (2.0 * eps_degrees)
    return rgb_minus, rgb_plus, grad


def finite_difference_scale(
        renderer, index: int, axis: str, eps: float, camera_name: str
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
        albedo3=(0.0, 0.0, 0.0),
        opacity=0.0,
        beta=0.0,
        index=index,
        camera_name=camera_name

    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=positive_scale,
        albedo3=(0.0, 0.0, 0.0),
        opacity=0.0,
        beta=0.0,
        index=index,
        camera_name=camera_name

    )
    grad = (rgb_plus - rgb_minus) / (2.0 * eps)
    return rgb_minus, rgb_plus, grad


def finite_difference_opacity(
        renderer, index: int, eps: float, camera_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    negative_opacity = -eps
    positive_opacity = +eps

    rgb_minus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        albedo3=(0.0, 0.0, 0.0),
        opacity=negative_opacity,
        beta=0.0,
        index=index,
        camera_name=camera_name

    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        albedo3=(0.0, 0.0, 0.0),
        opacity=positive_opacity,
        beta=0.0,
        index=index,
        camera_name=camera_name

    )
    grad = (rgb_plus - rgb_minus) / (2.0 * eps)
    return rgb_minus, rgb_plus, grad


def finite_difference_albedo(
        renderer, index: int, channel: str, eps: float, camera_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    channel in {'R','G','B'}: we perturb that color channel while keeping others at zero.
    Opacity is set to 1.0 so albedo changes have visible effect.
    """
    channel = channel.lower()
    if channel == "x":
        color_minus = (-eps, 0.0, 0.0)
        color_plus = (+eps, 0.0, 0.0)
    elif channel == "y":
        color_minus = (0.0, -eps, 0.0)
        color_plus = (0.0, +eps, 0.0)
    elif channel == "z":
        color_minus = (0.0, 0.0, -eps)
        color_plus = (0.0, 0.0, +eps)
    else:
        raise ValueError("channel must be 'x', 'y', or 'z'")

    rgb_minus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        albedo3=color_minus,
        opacity=1.0,
        beta=0.0,
        index=index,
        camera_name=camera_name

    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        albedo3=color_plus,
        opacity=1.0,
        beta=0.0,
        index=index,
        camera_name=camera_name

    )
    grad = (rgb_plus - rgb_minus) / (2.0 * eps)
    return rgb_minus, rgb_plus, grad


def finite_difference_beta(
        renderer, index: int, eps: float, camera_name: str
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
        albedo3=(0.0, 0.0, 0.0),
        opacity=1.0,
        beta=beta_minus,
        index=index,
        camera_name=camera_name

    )
    rgb_plus = render_with_trs(
        renderer,
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(1.0, 1.0, 1.0),
        albedo3=(0.0, 0.0, 0.0),
        opacity=1.0,
        beta=beta_plus,
        index=index,
        camera_name=camera_name

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
