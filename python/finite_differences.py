# main.py
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


def save_rgb_preview_png(img_f32: np.ndarray, out_path: Path,
                         exposure_stops: float = 0, gamma: float = 1) -> None:
    img_u8 = (np.clip(img_f32, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    print("Saving RGB preview to: {}".format(out_path.absolute()))
    Image.fromarray(img_u8.astype(np.uint8)).save(out_path)


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
        vis_float[~finite_mask] = (1.0, 1.0, 1.0)  # NaNs/Infs to white
        vis = (vis_float * 255.0).astype(np.uint8)

    Image.fromarray(vis.astype(np.uint8)).save(out_png)


# ---------- Robust signed normalization + visualizers ----------
def robustSignedNormalize(signed_array: np.ndarray,
                          mode: str = "percentile",
                          percentile: float = 0.995,
                          mad_clip: float | None = None,
                          global_scale: float | None = None,
                          exclude_zero: bool = True) -> tuple[np.ndarray, float]:
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


def saveSeismicSignedRobust(signed_array: np.ndarray, out_png: Path,
                            mode: str = "percentile",
                            percentile: float = 0.995,
                            global_scale: float | None = None) -> float:
    """Seismic colormap with symmetric robust normalization. Returns scale used."""
    norm, scale = robustSignedNormalize(
        signed_array, mode=mode, percentile=percentile, global_scale=global_scale
    )
    t = 0.5 * (norm + 1.0)
    rgba = cm.get_cmap("seismic")(t)  # [0,1]
    rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgb).save(out_png)
    return scale


def saveSignedLogCompress(signed_array: np.ndarray, out_png: Path,
                          epsilon: float = 1e-6,
                          percentile: float = 0.995) -> float:
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
    comp[finite] = np.sign(s[finite]) * np.log1p(
        np.abs(s[finite]) / max(epsilon, 1e-12)
    ) / denom
    comp = np.clip(comp, -1.0, 1.0)

    t = 0.5 * (comp + 1.0)
    rgba = cm.get_cmap("seismic")(t)
    rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgb).save(out_png)
    return scale

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


def render_with_trs(renderer,
                    translation3,
                    rotation_quat4,
                    scale3,
                    i: int) -> np.ndarray:
    """Apply a full TRS (translation, rotation, scale) and render."""
    renderer.set_gaussian_transform(
        translation3=translation3,
        rotation_quat4=rotation_quat4,  # (x, y, z, w)
        scale3=scale3,
        index=i
    )
    rgb = np.asarray(renderer.render_forward(), dtype=np.float32)
    return rgb

# ---------- Render + central differences ----------
def render_with_translation(renderer,
                            tx: float,
                            ty: float,
                            tz: float,
                            i: int) -> np.ndarray:
    """Apply TRS with translation on 'Gaussian', render, return HxWx3 float32 linear RGB. Raw."""
    renderer.set_gaussian_transform(
        translation3=(tx, ty, tz),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),  # (x, y, z, w)
        scale3=(1.0, 1.0, 1.0),
        index=i
    )
    rgb = renderer.render_forward()  # C++ ignores args; returns linear float32
    rgb = np.asarray(rgb, dtype=np.float32)
    # C++ tonemapper (which flipped Y) is disabled, so flip here
    return rgb


def render_with_rotation(renderer,
                         degrees: float,
                         axis: str,
                         i: int) -> np.ndarray:
    """Apply TRS with a rotation specified by (degrees, axis)."""
    qx, qy, qz, qw = degrees_to_quaternion(degrees, axis)

    renderer.set_gaussian_transform(
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(qx, qy, qz, qw),
        scale3=(1.0, 1.0, 1.0),
        index=i
    )

    rgb = np.asarray(renderer.render_forward(), dtype=np.float32)
    return rgb



def render_with_scale(renderer,
                      sx: float,
                      sy: float,
                      sz: float,
                      i: int) -> np.ndarray:
    """Apply TRS with scale on 'Gaussian', render, return HxWx3 float32 linear RGB. Raw."""
    renderer.set_gaussian_transform(
        translation3=(0.0, 0.0, 0.0),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),
        scale3=(sx, sy, sz),
        index=i
    )
    rgb = renderer.render_forward()
    rgb = np.asarray(rgb, dtype=np.float32)
    return rgb



def main(args) -> None:
    # --- settings ---
    renderer_settings = {
        "photons": 1e4,
        "bounces": 4,
        "forward_passes": 50,
        "gather_passes": 16,
        "adjoint_bounces": 4,
        "adjoint_passes": 6,
    }

    # use positive epsilon for central differences
    axis = args.axis  # could also be an argparse argument if needed

    assets_root = Path(__file__).parent.parent / "Assets"
    scene_xml = "cbox_custom.xml"
    pointcloud_ply = args.scene + ".ply"

    print("Assets root:", assets_root)
    print("Scene:", args.scene)
    print("Index:", args.index)
    print("Parameter:", args.param)

    output_dir = Path(__file__).parent / "Output" / args.scene if args.output == "" else Path(__file__).parent / Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- init renderer ---
    renderer = pale.Renderer(str(assets_root), scene_xml, pointcloud_ply, renderer_settings)

    eps_translation = 0.01  # or 0.005
    eps_rotation_deg = 0.75  # degrees
    eps_scale = 0.05
    eps_color = 0.1


    # --- render: minus and plus depending on parameter type ---
    if args.param == "translation":
        # central differences over translation component: (f(+e) - f(-e)) / (2e)
        if axis == "x":
            negative_vector, positive_vector = (-eps_translation, 0.0, 0.0), (+eps_translation, 0.0, 0.0)
        elif axis == "y":
            negative_vector, positive_vector = (0.0, -eps_translation, 0.0), (0.0, +eps_translation, 0.0)
        elif axis == "z":
            negative_vector, positive_vector = (0.0, 0.0, -eps_translation), (0.0, 0.0, +eps_translation)
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        rgb_minus = render_with_translation(renderer, *negative_vector, args.index)
        rgb_plus = render_with_translation(renderer, *positive_vector, args.index)

    elif args.param == "rotation":
        # central differences over rotation angle (in degrees) around given axis
        # parameter = rotation angle; we perturb by ±eps degrees
        rgb_minus = render_with_rotation(renderer, -eps_rotation_deg, axis, args.index)
        rgb_plus = render_with_rotation(renderer, +eps_rotation_deg, axis, args.index)

    elif args.param == "scale":
        # central differences over scale along one axis; base scale = 1
        if axis == "x":
            negative_scale = (1.0 - eps_scale, 1.0, 1.0)
            positive_scale = (1.0 + eps_scale, 1.0, 1.0)
        elif axis == "y":
            negative_scale = (1.0, 1.0 - eps_scale, 1.0)
            positive_scale = (1.0, 1.0 + eps_scale, 1.0)
        elif axis == "z":
            negative_scale = (1.0, 1.0, 1.0 - eps_scale)
            positive_scale = (1.0, 1.0, 1.0 + eps_scale)
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        rgb_minus = render_with_scale(renderer, *negative_scale, args.index)
        rgb_plus = render_with_scale(renderer, *positive_scale, args.index)
    elif args.param == "translation_rotation":
        # One scalar parameter π moves you simultaneously in translation and rotation.
        # Here we use:
        #   translation = π * e_axis
        #   rotation angle (degrees) = π
        #
        # Finite differences: f(π + eps) - f(π - eps)
        # with base π = 0, so:
        #   minus: translation = -eps * e_axis, angle = -eps
        #   plus:  translation = +eps * e_axis, angle = +eps

        axis = args.axis.lower()
        if axis == "x":
            t_dir = (1.0, 0.0, 0.0)
        elif axis == "y":
            t_dir = (0.0, 1.0, 0.0)
        elif axis == "z":
            t_dir = (0.0, 0.0, 1.0)
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        # translation for ±eps
        negative_translation = tuple(-eps_translation * c for c in t_dir)
        positive_translation = tuple(+eps_translation * c for c in t_dir)

        # rotation for ±eps degrees around same axis
        qx_minus, qy_minus, qz_minus, qw_minus = degrees_to_quaternion(-eps_rotation_deg, axis)
        qx_plus,  qy_plus,  qz_plus,  qw_plus  = degrees_to_quaternion(+eps_rotation_deg, axis)

        rgb_minus = render_with_trs(
            renderer,
            translation3=negative_translation,
            rotation_quat4=(qx_minus, qy_minus, qz_minus, qw_minus),
            scale3=(1.0, 1.0, 1.0),
            i=args.index,
        )
        rgb_plus = render_with_trs(
            renderer,
            translation3=positive_translation,
            rotation_quat4=(qx_plus, qy_plus, qz_plus, qw_plus),
            scale3=(1.0, 1.0, 1.0),
            i=args.index,
        )

    else:
        raise ValueError("param must be 'translation', 'rotation', or 'scale'")

    # raw gradient
    grad_raw = (rgb_plus - rgb_minus) / (2.0 * eps_translation)

    # display-space gradient by finite differences
    grad_disp_fd = (rgb_plus - rgb_minus) / (2.0 * eps_translation)

    # display-space gradient via chain rule (predict from raw)
    dTdx_mid = 0.5 * (rgb_minus + rgb_plus)
    grad_disp_chain = dTdx_mid * grad_raw

    # previews
    save_rgb_preview_png(rgb_minus, output_dir / "initial.png")
    save_rgb_preview_png(rgb_plus, output_dir / "target.png")

    # luminance projection (equal weights)
    weights_luminance = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0
    luma_grad = np.tensordot(grad_disp_fd, weights_luminance, 1)

    # legacy visualizers
    save_color_luma_seismic_white(luma_grad, output_dir / "grad_disp_fd_luma_white.png")

    # robust percentile scaling (symmetric) and save scale for reproducibility
    scale_used = saveSeismicSignedRobust(
        luma_grad, output_dir / "grad_disp_fd_seismic_robust.png",
        mode="percentile", percentile=0.995
    )
    np.savetxt(output_dir / "grad_scale_used.txt", [scale_used])

    # log-compressed magnitude view for very small dynamic ranges
    _ = saveSignedLogCompress(
        luma_grad, output_dir / "grad_disp_fd_seismic_log.png",
        epsilon=1e-6, percentile=0.995
    )

    # direct percentiles for comparison with your previous function
    def save_seismic_signed(scalar: np.ndarray, out_png: Path, abs_quantile: float = 1.0) -> None:
        scalar_array = np.asarray(scalar, dtype=np.float32)
        finite_mask = np.isfinite(scalar_array)
        if not np.any(finite_mask):
            Image.fromarray(np.zeros((*scalar_array.shape, 3), dtype=np.uint8)).save(out_png)
            return
        magnitudes = np.abs(scalar_array[finite_mask])
        q = np.clip(abs_quantile, 0.0, 1.0)
        scale_value = (np.quantile(magnitudes, q) if q < 1.0 else magnitudes.max())
        if not (np.isfinite(scale_value) and scale_value > 0.0):
            scale_value = 1.0
        normalized = np.clip(scalar_array / scale_value, -1.0, 1.0)
        t = 0.5 * (normalized + 1.0)
        rgba = cm.get_cmap("seismic")(t)
        rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
        rgb[~finite_mask] = (255, 255, 255)
        Image.fromarray(rgb).save(out_png)

    # seismic, full range and 0.99 percentile for continuity with old outputs
    save_seismic_signed(luma_grad, output_dir / "grad_disp_fd_seismic.png", abs_quantile=1.0)
    save_seismic_signed(luma_grad, output_dir / "grad_disp_fd_seismic_q099.png", abs_quantile=0.99)

    print("Wrote:")
    print(f"  {(output_dir / 'initial.png').resolve()}   (minus, preview)")
    print(f"  {(output_dir / 'target.png').resolve()}    (plus,  preview)")
    print(f"  {(output_dir / 'grad_disp_fd_luma_white.png').resolve()}")
    print(f"  {(output_dir / 'grad_disp_fd_seismic_robust.png').resolve()}   (robust percentile)")
    print(f"  {(output_dir / 'grad_disp_fd_seismic_log.png').resolve()}      (log-compressed)")
    print(f"  {(output_dir / 'grad_disp_fd_seismic.png').resolve()}          (legacy full-range)")
    print(f"  {(output_dir / 'grad_disp_fd_seismic_q099.png').resolve()}     (legacy q=0.99)")
    print(f"  {(output_dir / 'grad_scale_used.txt').resolve()}               (scale used)")
    time.sleep(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-difference gradient visualization for Pale renderer."
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
        help="Gaussian index to perturb (>=0 for single, -1 for all). Default: -1.",
    )

    parser.add_argument(
        "--param",
        type=str,
        choices=["translation", "rotation", "scale", "translation_rotation"],
        default="translation",
        help=(
            "Which parameter to finite-difference: "
            "'translation', 'rotation', 'scale', or 'translation_rotation'."
        ),
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
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)