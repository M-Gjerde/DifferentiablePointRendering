# debug_gradients.py
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import matplotlib
from matplotlib import cm  # noqa: F401

import pale


def save_gradient_sign_png_py(
    file_path: Path,
    rgba32f: np.ndarray,            # (H,W,4) float32
    adjoint_spp: float = 32.0,
    abs_quantile: float = 0.99,
    flip_y: bool = True,
) -> bool:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(rgba32f, dtype=np.float32, order="C")
    if img.ndim != 3 or img.shape[2] < 3:
        return False

    # 1) scalar = mean(R,G,B) / SPP
    rgb = img[..., :3] / float(max(adjoint_spp, 1e-8))
    scalar = np.mean(rgb, axis=2)
    scalar[~np.isfinite(scalar)] = 0.0

    # 2) symmetric robust scale using |scalar| quantile
    finite_abs = np.abs(scalar[np.isfinite(scalar)])
    if finite_abs.size:
        q = np.clip(abs_quantile, 0.0, 1.0)
        scale_abs = np.quantile(finite_abs, q) if q < 1.0 else finite_abs.max()
        if not (np.isfinite(scale_abs) and scale_abs > 0.0):
            scale_abs = 1.0
    else:
        scale_abs = 1.0
    norm = np.clip(scalar / scale_abs, -1.0, 1.0)

    # 3) map [-1,1] -> [0,1], apply matplotlib seismic
    cmap = matplotlib.colormaps["seismic"]
    t = 0.5 * (norm + 1.0)                    # [0,1]
    rgba = cmap(t, bytes=True)                # uint8 RGBA
    out = rgba[..., :3]                       # drop alpha

    # 4) flip and save
    if flip_y:
        out = np.flipud(out)
    iio.imwrite(str(file_path), out)
    return True


def summarize_grad_field(name: str, arr: np.ndarray, max_to_print: int = 5) -> None:
    arr = np.asarray(arr, dtype=np.float32, order="C")
    print(f"\n=== Gradient '{name}' ===")
    print(f"shape: {arr.shape}, dtype: {arr.dtype}")

    # Treat last dimension as components, everything before as "per-point"
    if arr.ndim == 1:
        flat = arr.reshape(-1, 1)
    else:
        flat = arr.reshape(arr.shape[0], -1)

    norms = np.linalg.norm(flat, axis=1)
    print(f"  count      = {flat.shape[0]}")
    print(f"  min |g|    = {norms.min():.6e}")
    print(f"  max |g|    = {norms.max():.6e}")
    print(f"  mean |g|   = {norms.mean():.6e}")

    n_print = min(max_to_print, flat.shape[0])
    print(f"  first {n_print} entries:")
    for idx in range(n_print):
        components = " ".join(f"{c:+.6e}" for c in flat[idx])
        print(f"    idx {idx:4d}: {components}")


def main() -> None:
    out_dir = Path(__file__).parent / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer_settings = {
        "photons":         1e5,
        "bounces":         4,
        "forward_passes":  10,
        "gather_passes":   8,
        "adjoint_bounces": 1,
        "adjoint_passes":  8,
    }

    assets_root = Path(__file__).parent.parent / "Assets"
    scene_xml = "cbox_custom.xml"
    pointcloud_ply = "initial.ply"

    renderer = pale.Renderer(
        str(assets_root),
        scene_xml,
        pointcloud_ply,
        renderer_settings,
    )

    # ---------- Forward target ----------
    rgb = renderer.render_forward()  # (H,W,3)
    target = np.asarray(rgb, dtype=np.float32, order="C")

    # ---------- Backward ----------
    # gradients is a dict, grad_img is (H,W,4)
    gradients, grad_img = renderer.render_backward(target)

    np.set_printoptions(precision=6, suppress=True, linewidth=140)

    # ---- Extract and save individual gradient arrays ----
    # Each entry is a NumPy array already backed by your C++ buffers.
    grad_position   = np.asarray(gradients["position"],   dtype=np.float32, order="C")  # (N,3)
    grad_tangent_u  = np.asarray(gradients["tangent_u"],  dtype=np.float32, order="C")  # (N,3)
    grad_tangent_v  = np.asarray(gradients["tangent_v"],  dtype=np.float32, order="C")  # (N,3)
    grad_scale      = np.asarray(gradients["scale"],      dtype=np.float32, order="C")  # (N,2)
    grad_color      = np.asarray(gradients["color"],      dtype=np.float32, order="C")  # (N,3)
    grad_opacity    = np.asarray(gradients["opacity"],    dtype=np.float32, order="C")  # (N,)
    grad_beta       = np.asarray(gradients["beta"],       dtype=np.float32, order="C")  # (N,)
    grad_shape      = np.asarray(gradients["shape"],      dtype=np.float32, order="C")  # (N,)

    # Save as npy for external inspection
    np.save(out_dir / "grad_position.npy",  grad_position)
    np.save(out_dir / "grad_tangent_u.npy", grad_tangent_u)
    np.save(out_dir / "grad_tangent_v.npy", grad_tangent_v)
    np.save(out_dir / "grad_scale.npy",     grad_scale)
    np.save(out_dir / "grad_color.npy",     grad_color)
    np.save(out_dir / "grad_opacity.npy",   grad_opacity)
    np.save(out_dir / "grad_beta.npy",      grad_beta)
    np.save(out_dir / "grad_shape.npy",     grad_shape)

    # ---- Print overall info ----
    num_points = grad_position.shape[0]
    print(f"Number of points: {num_points}")
    print("Gradient fields:", list(gradients.keys()))

    # Summaries per field
    summarize_grad_field("position",   grad_position, max_to_print=10)
    summarize_grad_field("tangent_u",  grad_tangent_u)
    summarize_grad_field("tangent_v",  grad_tangent_v)
    summarize_grad_field("scale",      grad_scale)
    summarize_grad_field("color",      grad_color)
    summarize_grad_field("opacity",    grad_opacity)
    summarize_grad_field("beta",       grad_beta)
    summarize_grad_field("shape",      grad_shape)

    # If you specifically want the first few position gradients, matching your C++ access:
    print("\nFirst 10 position gradients as (dx, dy, dz):")
    max_pos_print = min(10, num_points)
    for primitive_index in range(max_pos_print):
        gx, gy, gz = grad_position[primitive_index]
        print(
            f"  idx {primitive_index:4d}: "
            f"gradPosition.x = {gx:+.6e}, "
            f"gradPosition.y = {gy:+.6e}, "
            f"gradPosition.z = {gz:+.6e}"
        )

    # ---------- Gradient image visualization ----------
    img = np.asarray(grad_img, dtype=np.float32, order="C")  # (H,W,4)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    save_gradient_sign_png_py(
        out_dir / "adjoint_grad_vis.png",
        img,
        adjoint_spp=renderer_settings.get("adjoint_passes", 32),
        abs_quantile=1.0,
        flip_y=True,
    )

    save_gradient_sign_png_py(
        out_dir / "adjoint_grad_vis_0_999.png",
        img,
        adjoint_spp=renderer_settings.get("adjoint_passes", 32),
        abs_quantile=0.999,
        flip_y=True,
    )

    print("\nSaved files:")
    for name in [
        "grad_position.npy",
        "grad_tangent_u.npy",
        "grad_tangent_v.npy",
        "grad_scale.npy",
        "grad_color.npy",
        "grad_opacity.npy",
        "grad_beta.npy",
        "grad_shape.npy",
        "adjoint_grad_vis.png",
        "adjoint_grad_vis_0_999.png",
    ]:
        print(" ", (out_dir / name).resolve())


if __name__ == "__main__":
    main()
