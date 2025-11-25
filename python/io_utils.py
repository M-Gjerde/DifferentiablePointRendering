from __future__ import annotations

import os
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np


def save_gradient_sign_png_py(
        file_path: Path,
        rgba32f: np.ndarray,  # (H,W,4) float32
        adjoint_spp: float = 32.0,
        abs_quantile: float = 0.99,
        flip_y: bool = True,
) -> bool:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(rgba32f, dtype=np.float32, order="C")
    if img.ndim != 3 or img.shape[2] < 3:
        return False

    rgb = img[..., :3] / float(max(adjoint_spp, 1e-8))
    scalar = np.mean(rgb, axis=2)
    scalar[~np.isfinite(scalar)] = 0.0

    finite_abs = np.abs(scalar[np.isfinite(scalar)])
    if finite_abs.size:
        q = np.clip(abs_quantile, 0.0, 1.0)
        scale_abs = np.quantile(finite_abs, q) if q < 1.0 else finite_abs.max()
        if not (np.isfinite(scale_abs) and scale_abs > 0.0):
            scale_abs = 1.0
    else:
        scale_abs = 1.0
    norm = np.clip(scalar / scale_abs, -1.0, 1.0)

    cmap = matplotlib.colormaps["seismic"]
    t = 0.5 * (norm + 1.0)
    rgba = cmap(t, bytes=True)
    out = rgba[..., :3]

    if flip_y:
        out = np.flipud(out)
    iio.imwrite(str(file_path), out)
    return True


def load_target_image(path: Path) -> np.ndarray:
    """
    Load a target RGB image as float32 array (H,W,3).
    """
    img = iio.imread(path.as_posix())
    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]

    if img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError(f"Target image must be HxWx3, got shape {img.shape}")

    if img.max() > 1.0 + 1e-4:
        img = img / 255.0

    return np.ascontiguousarray(img)


def save_positions_numpy(path: Path, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(positions, dtype=np.float32, order="C"))


def save_render(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(rgb, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    iio.imwrite(path.as_posix(), img_u8)


def save_loss_image(
        output_dir: Path,
        loss_image: np.ndarray,
        iteration: int,
) -> None:
    loss_image_vis = np.clip(
        loss_image / np.percentile(loss_image, 99.0), 0.0, 1.0
    )
    loss_image_u8 = (loss_image_vis * 255).astype(np.uint8)
    os.makedirs(output_dir / "loss", exist_ok=True)
    iio.imwrite(
        (output_dir / "loss" / f"loss_image_iter_{iteration:04d}.png").as_posix(),
        loss_image_u8,
    )


def save_gaussians_to_ply(
        file_path: Path,
        positions: torch.Tensor,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        beta_default: float = 0.0,
        shape_default: float = 0.0,
) -> None:
    """
    Save current point parameters to an ASCII PLY file with layout:

    ply
    format ascii 1.0
    comment 2D Gaussian splats: pk, tu, tv, scales, diffuse albedo, opacity
    element vertex N
    property float x
    property float y
    property float z
    property float tu_x
    property float tu_y
    property float tu_z
    property float tv_x
    property float tv_y
    property float tv_z
    property float su
    property float sv
    property float albedo_r
    property float albedo_g
    property float albedo_b
    property float opacity
    property float beta
    property float shape
    end_header
    ...

    For non-optimized parameters, we use:
        opacity = opacity_default
        beta    = beta_default
        shape   = shape_default
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    pos = positions.detach().cpu().numpy()
    tu = tangent_u.detach().cpu().numpy()
    tv = tangent_v.detach().cpu().numpy()
    sc = scales.detach().cpu().numpy()
    col = colors.detach().cpu().numpy()
    opa = opacities.detach().cpu().numpy()

    num_points = pos.shape[0]

    if not (tu.shape[0] == tv.shape[0] == sc.shape[0] == col.shape[0] == opa.shape[0] == num_points):
        raise ValueError(
            "Inconsistent point counts between positions/tangent_u/tangent_v/scales/colors/opacities"
        )

    if sc.ndim != 2 or sc.shape[1] < 2:
        raise ValueError(
            f"Expected scales to have at least 2 components per point (su, sv), got shape {sc.shape}"
        )

    su = sc[:, 0]
    sv = sc[:, 1]

    with file_path.open("w", encoding="ascii") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(
            "comment 2D Gaussian splats: pk, tu, tv, scales, diffuse albedo, opacity\n"
        )
        f.write(f"element vertex {num_points}\n")
        f.write("property float x          # pk.x\n")
        f.write("property float y          # pk.y\n")
        f.write("property float z          # pk.z\n")
        f.write("property float tu_x       # tangential axis u (unit)\n")
        f.write("property float tu_y\n")
        f.write("property float tu_z\n")
        f.write("property float tv_x       # tangential axis v (unit, orthonormal to tu)\n")
        f.write("property float tv_y\n")
        f.write("property float tv_z\n")
        f.write("property float su         # scale along tu\n")
        f.write("property float sv         # scale along tv\n")
        f.write("property float albedo_r   # diffuse BRDF albedo\n")
        f.write("property float albedo_g\n")
        f.write("property float albedo_b\n")
        f.write("property float opacity\n")
        f.write("property float beta\n")
        f.write("property float shape\n")
        f.write("end_header\n")

        # Data
        for i in range(num_points):
            x, y, z = pos[i]
            tu_x, tu_y, tu_z = tu[i]
            tv_x, tv_y, tv_z = tv[i]
            su_i = su[i]
            sv_i = sv[i]
            opa_i = opa[i]
            albedo_r, albedo_g, albedo_b = col[i]

            # Use general-format with enough precision, but still readable
            line = (
                f"{x:.9g} {y:.9g} {z:.9g}  "
                f"{tu_x:.9g} {tu_y:.9g} {tu_z:.9g}  "
                f"{tv_x:.9g} {tv_y:.9g} {tv_z:.9g}  "
                f"{su_i:.9g} {sv_i:.9g}  "
                f"{albedo_r:.9g} {albedo_g:.9g} {albedo_b:.9g}  "
                f"{opa_i:.9g} {beta_default:.9g} {shape_default:.9g}\n"
            )
            f.write(line)
