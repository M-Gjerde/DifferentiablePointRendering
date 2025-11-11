from pathlib import Path

import matplotlib
import numpy as np
import imageio.v3 as iio
import pale

from pathlib import Path
import numpy as np
import imageio.v3 as iio

from pathlib import Path
import numpy as np
import imageio.v3 as iio
from matplotlib import cm

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


out_dir = Path(__file__).parent / "Output"
out_dir.mkdir(parents=True, exist_ok=True)

rendererSettings = {
    "photons": 1e6,
    "bounces": 6,
    "forward_passes": 6,
    "gather_passes": 6,
    "adjoint_bounces": 1,
    "adjoint_passes": 6,
}

assets_root = Path(__file__).parent.parent / "Assets"
scene_xml = "cbox_custom.xml"
pointcloud_ply = "initial.ply"

renderer = pale.Renderer(str(assets_root), scene_xml, pointcloud_ply, rendererSettings)

# Forward target
rgb = renderer.render_forward()  # HxWx3 float32
target = np.asarray(rgb, dtype=np.float32, order="C")

# Backward: returns (N,3) and (H,W,4) float32
grad_vecs, grad_img = renderer.render_backward(target)  # flipY uses C++ default

# ---- Save point gradients ----
gv = np.asarray(grad_vecs, dtype=np.float32, order="C")
np.save(out_dir / "point_gradients.npy", gv)

# ---- Prepare gradient image visualization ----
img = np.asarray(grad_img, dtype=np.float32, order="C")  # HxWx4
img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

# Use R,G,B channels. Robust symmetric scaling by percentile.
rgb_img = img[..., :3]
abs_vals = np.abs(rgb_img.reshape(-1, 3))
# fallback if mostly zeros
scale = np.percentile(abs_vals, 99.0, axis=0)
scale = np.where(scale < 1e-8, 1.0, scale)

H, W = grad_img.shape[:2]
save_gradient_sign_png_py(
    out_dir / "adjoint_grad_vis.png",
    grad_img,
    adjoint_spp=rendererSettings.get("adjoint_passes", 32),
    abs_quantile=1.0,
    flip_y=True,
)                # [0,1]

save_gradient_sign_png_py(
    out_dir / "adjoint_grad_vis_0_999.png",
    grad_img,
    adjoint_spp=rendererSettings.get("adjoint_passes", 32),
    abs_quantile=0.999,
    flip_y=True,
)                # [0,1]


print("Saved:",
      (out_dir / "point_gradients.npy").resolve(),
      (out_dir / "adjoint_grad_vis.png").resolve(), sep="\n")
