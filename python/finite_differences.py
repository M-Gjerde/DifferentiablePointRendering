# main.py
import time
from pathlib import Path

import numpy as np
from PIL import Image
import pale
from matplotlib import cm


# ---------- I/O helpers ----------
def save_rgb_float32_npy(img_f32: np.ndarray, out_path: Path) -> None:
    np.save(out_path, img_f32.astype(np.float32))


def tonemap_exposure_gamma(linear: np.ndarray, exposure_stops: float, gamma: float) -> np.ndarray:
    # T(x) = (1 - exp(-k x))^(1/gamma), k = 2**stops
    k = 2.0 ** exposure_stops
    y = 1.0 - np.exp(-k * np.clip(linear, 0.0, None))
    return np.power(np.clip(y, 0.0, 1.0), 1.0 / max(gamma, 1e-6))


def d_tonemap_dx(linear: np.ndarray, exposure_stops: float, gamma: float) -> np.ndarray:
    # dT/dx = (1/gamma) * (1 - exp(-k x))^(1/gamma - 1) * exp(-k x) * k
    k = 2.0 ** exposure_stops
    x = np.clip(linear, 0.0, None)
    e = np.exp(-k * x)
    y = 1.0 - e
    inv_gamma = 1.0 / max(gamma, 1e-6)
    y_pow = np.power(np.clip(y, 1e-20, 1.0), inv_gamma - 1.0)
    return inv_gamma * y_pow * e * k


def save_rgb_preview_png(img_f32: np.ndarray, out_path: Path,
                         exposure_stops: float = 5.8, gamma: float = 2.2) -> None:
    mapped = tonemap_exposure_gamma(img_f32, exposure_stops, gamma)
    img_u8 = (np.clip(mapped, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
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
    comp[finite] = np.sign(s[finite]) * np.log1p(np.abs(s[finite]) / max(epsilon, 1e-12)) / denom
    comp = np.clip(comp, -1.0, 1.0)

    t = 0.5 * (comp + 1.0)
    rgba = cm.get_cmap("seismic")(t)
    rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgb).save(out_png)
    return scale


# ---------- Render + central differences ----------
def render_with_translation(renderer, tx: float, ty: float, tz: float) -> np.ndarray:
    """Apply TRS on 'Gaussian', render, return HxWx3 float32 linear RGB. Raw."""
    renderer.set_gaussian_transform(
        translation3=(tx, ty, tz),
        rotation_quat4=(0.0, 0.0, 0.0, 1.0),  # (x,y,z,w)
        scale3=(1.0, 1.0, 1.0),
    )
    rgb = renderer.render_forward()  # C++ ignores args; returns linear float32
    rgb = np.asarray(rgb, dtype=np.float32)
    # C++ tonemapper (which flipped Y) is disabled, so flip here
    rgb = np.flipud(rgb)
    return rgb


def main():
    # --- settings ---
    renderer_settings = {
        "photons": 1e6,
        "bounces": 4,
        "forward_passes": 8,
        "gather_passes": 1024,
        "adjoint_bounces": 4,
        "adjoint_passes": 6,
    }
    # central differences over translation.x: (f(+e) - f(-e)) / (2e)
    eps = 0.01
    axis = "x"  # "y" or "z"

    assets_root = Path(__file__).parent.parent / "Assets"
    scene_xml = "cbox_custom.xml"

    scene = "initial"
    pointcloud_ply = scene + ".ply"

    print(assets_root)
    output_dir = Path(__file__).parent / "Output" / scene
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- init renderer ---
    renderer = pale.Renderer(str(assets_root), scene_xml, pointcloud_ply, renderer_settings)

    # --- choose perturbation vector ---
    if axis == "x":
        neg_vec, pos_vec = (0, 0.0, 0.0), (+eps, 0.0, 0.0)
    elif axis == "y":
        neg_vec, pos_vec = (0.0, -eps, 0.0), (0.0, +eps, 0.0)
    elif axis == "z":
        neg_vec, pos_vec = (0.0, 0.0, -eps), (0.0, 0.0, +eps)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    # --- render: minus and plus ---
    rgb_minus = render_with_translation(renderer, *neg_vec)
    rgb_plus = render_with_translation(renderer, *pos_vec)

    # raw gradient
    grad_raw = (rgb_plus - rgb_minus) / (2.0 * eps)
    # np.save(output_dir / "central_diff_gradient_raw_rgb_float32.npy", grad_raw.astype(np.float32))

    # display-space gradient by finite differences
    exposure_stops, gamma_val = 1.8, 2.0
    disp_minus = tonemap_exposure_gamma(rgb_minus, exposure_stops, gamma_val)
    disp_plus = tonemap_exposure_gamma(rgb_plus, exposure_stops, gamma_val)
    grad_disp_fd = (disp_plus - disp_minus) / (2.0 * eps)
    # np.save(output_dir / "central_diff_gradient_display_fd_rgb_float32.npy", grad_disp_fd.astype(np.float32))

    # display-space gradient via chain rule (predict from raw)
    dTdx_minus = d_tonemap_dx(rgb_minus, exposure_stops, gamma_val)
    dTdx_plus = d_tonemap_dx(rgb_plus, exposure_stops, gamma_val)
    dTdx_mid = 0.5 * (dTdx_minus + dTdx_plus)
    grad_disp_chain = dTdx_mid * grad_raw
    # np.save(output_dir / "central_diff_gradient_display_chain_rgb_float32.npy", grad_disp_chain.astype(np.float32))

    # previews
    save_rgb_preview_png(rgb_minus, output_dir / "initial.png", exposure_stops, gamma_val)
    save_rgb_preview_png(rgb_plus, output_dir / "target.png", exposure_stops, gamma_val)

    # luminance projection
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma_grad = np.tensordot(grad_disp_fd, w, 1)

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
        s = np.asarray(scalar, dtype=np.float32)
        finite = np.isfinite(s)
        if not np.any(finite):
            Image.fromarray(np.zeros((*s.shape, 3), dtype=np.uint8)).save(out_png)
            return
        mags = np.abs(s[finite])
        q = np.clip(abs_quantile, 0.0, 1.0)
        scale = (np.quantile(mags, q) if q < 1.0 else mags.max())
        if not (np.isfinite(scale) and scale > 0.0):
            scale = 1.0
        norm = np.clip(s / scale, -1.0, 1.0)
        t = 0.5 * (norm + 1.0)
        rgba = cm.get_cmap("seismic")(t)
        rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
        rgb[~finite] = (255, 255, 255)
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


if __name__ == "__main__":
    main()
