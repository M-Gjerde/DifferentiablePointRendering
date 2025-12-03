import numpy as np
import pale


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


