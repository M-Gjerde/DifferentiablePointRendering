from __future__ import annotations

import os
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
import torch
import OpenEXR
import Imath
import json
import math
from dataclasses import asdict, is_dataclass

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

    if img.max() > 1.0:
        img = img / 255.0

    return np.ascontiguousarray(img)

def read_rgb_exr(
        exr_path: Path,
        apply_exposure_stops: float | None = None,
) -> np.ndarray:
    """
    Read a linear RGB EXR into a float32 HxWx3 array.

    exr_path: path to .exr file
    apply_exposure_stops:
        - None: return raw linear values
        - float: multiply image by 2**stops (optional convenience)

    Returns:
        img_f32: HxWx3, dtype float32, linear RGB
    """

    exr_path = Path(exr_path)
    exr = OpenEXR.InputFile(str(exr_path))
    header = exr.header()

    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Determine pixel type (FLOAT or HALF)
    channel_info = header["channels"]["R"]
    pixel_type = channel_info.type

    # Read planar channels
    r_bytes = exr.channel("R", pixel_type)
    g_bytes = exr.channel("G", pixel_type)
    b_bytes = exr.channel("B", pixel_type)

    if pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
        np_type = np.float32
    elif pixel_type == Imath.PixelType(Imath.PixelType.HALF):
        np_type = np.float16
    else:
        raise ValueError(f"Unsupported EXR pixel type: {pixel_type}")

    # Convert to numpy and reshape
    r = np.frombuffer(r_bytes, dtype=np_type).reshape(height, width)
    g = np.frombuffer(g_bytes, dtype=np_type).reshape(height, width)
    b = np.frombuffer(b_bytes, dtype=np_type).reshape(height, width)

    # Stack to HxWx3 and promote to float32
    img = np.stack([r, g, b], axis=-1).astype(np.float32)

    if apply_exposure_stops is not None:
        img = img * (2.0 ** float(apply_exposure_stops))

    return img

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
    loss_image = np.asarray(loss_image, dtype=np.float32)

    # If a multi-channel residual image is passed, reduce it to one value per pixel
    # for visualization.
    if loss_image.ndim == 3:
        if loss_image.shape[2] == 1:
            loss_image = loss_image[..., 0]
        else:
            loss_image = np.mean(loss_image, axis=2)

    finite_mask = np.isfinite(loss_image)

    # Default to white image
    height, width = loss_image.shape
    loss_image_rgb = np.ones((height, width, 3), dtype=np.float32)

    if np.any(finite_mask):
        scale = np.percentile(np.abs(loss_image[finite_mask]), 99.0)

        if scale > 1.0e-12:
            normalized_loss = np.clip(loss_image / scale, -1.0, 1.0)

            positive_mask = normalized_loss > 0.0
            negative_mask = normalized_loss < 0.0

            positive_strength = normalized_loss[positive_mask]          # 0 .. 1
            negative_strength = -normalized_loss[negative_mask]         # 0 .. 1

            # Positive residuals: white -> red
            # [1, 1, 1] -> [1, 0, 0]
            loss_image_rgb[positive_mask, 1] = 1.0 - positive_strength
            loss_image_rgb[positive_mask, 2] = 1.0 - positive_strength

            # Negative residuals: white -> blue
            # [1, 1, 1] -> [0, 0, 1]
            loss_image_rgb[negative_mask, 0] = 1.0 - negative_strength
            loss_image_rgb[negative_mask, 1] = 1.0 - negative_strength

    loss_image_u8 = (np.clip(loss_image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    os.makedirs(output_dir / "loss", exist_ok=True)
    iio.imwrite(
        (output_dir / "loss" / f"loss_image_iter_{iteration:04d}.png").as_posix(),
        loss_image_u8,
    )

def _normalize_np(v: np.ndarray, fallback: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        return np.asarray(fallback, dtype=np.float64)
    return v / n

def _quat_from_rotation_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n <= 1.0e-12:
        return 1.0, 0.0, 0.0, 0.0
    q /= n
    if q[0] < 0.0:
        q = -q
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])

def _quat_from_tangents(tangent_u: np.ndarray, tangent_v: np.ndarray) -> tuple[float, float, float, float]:
    u = _normalize_np(tangent_u, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    v_raw = np.asarray(tangent_v, dtype=np.float64)
    v = v_raw - float(np.dot(v_raw, u)) * u
    if float(np.linalg.norm(v)) <= 1.0e-12:
        aux = np.array([0.0, 1.0, 0.0], dtype=np.float64) if abs(float(u[1])) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v = aux - float(np.dot(aux, u)) * u
    v = _normalize_np(v, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    w = _normalize_np(np.cross(u, v), np.array([0.0, 0.0, 1.0], dtype=np.float64))
    R = np.column_stack((u, v, w))
    return _quat_from_rotation_matrix(R)
def _normalize_quaternions_np(q: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32, order="C")

    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected rotations to have shape (N,4), got {q.shape}")

    norms = np.linalg.norm(q, axis=1, keepdims=True)
    finite = np.isfinite(q).all(axis=1, keepdims=True)
    valid = finite & (norms > eps)

    fallback = np.zeros_like(q, dtype=np.float32)
    fallback[:, 0] = 1.0

    q_normalized = q / np.maximum(norms, eps)
    q_normalized = np.where(valid, q_normalized, fallback)

    # Canonicalize sign. q and -q represent the same rotation.
    q_normalized = np.where(q_normalized[:, 0:1] < 0.0, -q_normalized, q_normalized)

    return q_normalized.astype(np.float32, copy=False)


def save_gaussians_to_ply(
        file_path: Path,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        betas: torch.Tensor,
        powers: torch.Tensor,
        shape_default: float = 0.0,
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    pos = np.asarray(positions.detach().cpu().numpy(), dtype=np.float32, order="C")
    rot = _normalize_quaternions_np(rotations.detach().cpu().numpy())
    sc = np.asarray(scales.detach().cpu().numpy(), dtype=np.float32, order="C")
    col = np.asarray(colors.detach().cpu().numpy(), dtype=np.float32, order="C")
    opa = np.asarray(opacities.detach().cpu().numpy(), dtype=np.float32, order="C").reshape(-1)
    beta_values = np.asarray(betas.detach().cpu().numpy(), dtype=np.float32, order="C").reshape(-1)
    power_values = np.asarray(powers.detach().cpu().numpy(), dtype=np.float32, order="C").reshape(-1)

    num_points = pos.shape[0]

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected positions to have shape (N,3), got {pos.shape}")

    if rot.ndim != 2 or rot.shape[1] != 4:
        raise ValueError(f"Expected rotations to have shape (N,4), got {rot.shape}")

    if sc.ndim != 2 or sc.shape[1] < 2:
        raise ValueError(f"Expected scales to have at least 2 components per point, got {sc.shape}")

    if col.ndim != 2 or col.shape[1] != 3:
        raise ValueError(f"Expected colors to have shape (N,3), got {col.shape}")

    if not (
        rot.shape[0]
        == sc.shape[0]
        == col.shape[0]
        == opa.shape[0]
        == beta_values.shape[0]
        == power_values.shape[0]
        == num_points
    ):
        raise ValueError(
            "Inconsistent point counts between "
            "positions/rotations/scales/colors/opacities/betas/powers"
        )

    with file_path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Quaternion surfels: position, rotation quaternion, scales, diffuse albedo, opacity\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float rot_w\n")
        f.write("property float rot_x\n")
        f.write("property float rot_y\n")
        f.write("property float rot_z\n")
        f.write("property float su\n")
        f.write("property float sv\n")
        f.write("property float albedo_r\n")
        f.write("property float albedo_g\n")
        f.write("property float albedo_b\n")
        f.write("property float opacity\n")
        f.write("property float beta\n")
        f.write("property float shape\n")
        f.write("property float power\n")
        f.write("end_header\n")

        for i in range(num_points):
            x, y, z = pos[i]
            qw, qx, qy, qz = rot[i]
            su_i, sv_i = sc[i, 0], sc[i, 1]
            albedo_r, albedo_g, albedo_b = col[i]

            f.write(
                f"{x:.9g} {y:.9g} {z:.9g}  "
                f"{qw:.9g} {qx:.9g} {qy:.9g} {qz:.9g}  "
                f"{su_i:.9g} {sv_i:.9g}  "
                f"{albedo_r:.9g} {albedo_g:.9g} {albedo_b:.9g}  "
                f"{opa[i]:.9g} {beta_values[i]:.9g} {shape_default:.9g} {power_values[i]:.9g}\n"
            )

def _jsonify_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonify_value(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_value(item) for item in value]
    return value


def resolve_existing_path(path: Path, description: str) -> Path:
    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Could not find {description}: {resolved_path}")

    return resolved_path


def resolve_dataset_file(dataset_folder: Path, filename: str, description: str) -> Path:
    file_path = dataset_folder / filename

    if not file_path.is_file():
        raise FileNotFoundError(f"Could not find {description}: {file_path}")

    return file_path.resolve()


def configure_paths_from_dataset_folder(config) -> None:
    dataset_folder = resolve_existing_path(Path(config.dataset_path), "dataset folder")

    if not dataset_folder.is_dir():
        raise RuntimeError(f"--dataset-path must point to a dataset folder: {dataset_folder}")

    images_folder = dataset_folder / "images"
    if not images_folder.is_dir():
        raise RuntimeError(f"Dataset folder is missing images folder: {images_folder}")

    if not config.scene_xml_is_explicit:
        config.scene_xml = str(resolve_dataset_file(dataset_folder, "scene.xml", "scene XML"))

    if not config.pointcloud_ply_is_explicit:
        config.pointcloud_ply = str(resolve_dataset_file(dataset_folder, "points.ply", "pointcloud PLY"))

    config.dataset_path = dataset_folder

def get_python_project_dir() -> Path:
    return Path(__file__).resolve().parent


def get_python_project_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_output_dir(output_dir: Path, output_dir_is_explicit: bool) -> Path:
    python_project_dir = get_python_project_dir()
    output_dir = Path(output_dir).expanduser()

    if output_dir.is_absolute():
        return output_dir.resolve()

    default_output_root = python_project_dir / "OptimizationOutput"

    if not output_dir_is_explicit:
        return default_output_root.resolve()

    if output_dir.parts and output_dir.parts[0] == "OptimizationOutput":
        return (python_project_dir / output_dir).resolve()

    return (default_output_root / output_dir).resolve()

def save_run_config(
    output_dir: Path,
    config,
    renderer_settings,
    run_folder_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    optimization_config_dict = asdict(config) if is_dataclass(config) else dict(config)
    renderer_settings_dict = renderer_settings.as_dict(config)

    run_config = {
        "run_folder_name": run_folder_name,
        "assets_root": str(Path(config.assets_root).resolve()),
        "scene_xml": config.scene_xml,
        "initial_pointcloud_ply": config.pointcloud_ply,
        "dataset_path": str(config.dataset_path),
        "output_dir": str(Path(config.output_dir).resolve()),
        "optimization_config": _jsonify_value(optimization_config_dict),
        "renderer_settings": _jsonify_value(renderer_settings_dict),
    }

    run_config_path = output_dir / "run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as json_file:
        json.dump(run_config, json_file, indent=2)

    print(f"Saved run config: {run_config_path}")