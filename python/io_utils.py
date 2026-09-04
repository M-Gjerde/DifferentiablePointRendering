from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
import torch
import OpenEXR
import Imath
from PIL import Image, ImageCms
import json
from dataclasses import asdict, is_dataclass


SUPPORTED_TARGET_IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".webp",
    ".exr",
    ".hdr",
)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode continuous sRGB values into linear-light sRGB/Rec.709."""
    encoded = np.asarray(rgb, dtype=np.float32)
    power_base = np.maximum((encoded + 0.055) / 1.055, 0.0)
    linear = np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power(power_base, 2.4),
    )
    return linear.astype(np.float32, copy=False)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Encode linear-light sRGB/Rec.709 values with the sRGB transfer function."""
    linear = np.asarray(rgb, dtype=np.float32)
    non_negative = np.maximum(linear, 0.0)
    encoded = np.where(
        non_negative <= 0.0031308,
        12.92 * non_negative,
        1.055 * np.power(non_negative, 1.0 / 2.4) - 0.055,
    )
    return encoded.astype(np.float32, copy=False)


def _normalize_image_samples(image: np.ndarray) -> np.ndarray:
    samples = np.asarray(image)
    if np.issubdtype(samples.dtype, np.bool_):
        return samples.astype(np.float32)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        if info.min < 0 and samples.size and np.min(samples) >= 0:
            observed_max = int(np.max(samples))
            scale = 255.0 if observed_max <= 255 else 65535.0
        else:
            scale = float(info.max)
        return samples.astype(np.float32) / scale
    return samples.astype(np.float32)


def _as_rgb_image(image: np.ndarray, path: Path) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] > 3:
        image = image[..., :3]

    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"Target image '{path}' must be HxWx3, got shape {image.shape}")
    return image


def _load_icc_converted_srgb(path: Path) -> np.ndarray | None:
    """Return ICC-managed sRGB samples, or None when the image has no profile."""
    try:
        with Image.open(path) as source_image:
            icc_bytes = source_image.info.get("icc_profile")
            if not icc_bytes:
                return None
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
            srgb_profile = ImageCms.createProfile("sRGB")
            converted = ImageCms.profileToProfile(
                source_image.convert("RGB"),
                source_profile,
                srgb_profile,
                outputMode="RGB",
            )
            return _normalize_image_samples(np.asarray(converted))
    except (OSError, ImageCms.PyCMSError) as exception:
        print(
            f"Warning: could not apply embedded ICC profile for '{path}': {exception}. "
            "Falling back to untagged sRGB interpretation."
        )
        return None


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


def load_target_image(path: Path, color_space: str = "auto") -> np.ndarray:
    """Load a target into the canonical linear-light sRGB/Rec.709 working space.

    ``auto`` honors embedded ICC profiles for ordinary images, treats untagged
    integer images as sRGB, and treats floating-point/EXR/HDR images as linear.
    ``srgb`` and ``linear`` explicitly override that inference.
    """
    path = Path(path)
    requested_color_space = str(color_space).strip().lower()
    if requested_color_space not in {"auto", "srgb", "linear"}:
        raise ValueError(
            f"Unsupported target color space '{color_space}'; expected auto, srgb, or linear"
        )

    suffix = path.suffix.lower()
    icc_srgb = None
    if requested_color_space == "auto" and suffix not in {".exr", ".hdr"}:
        icc_srgb = _load_icc_converted_srgb(path)

    if icc_srgb is not None:
        image = _as_rgb_image(icc_srgb, path)
        interpreted_color_space = "srgb"
        interpretation = "embedded ICC profile converted to sRGB"
    else:
        if suffix == ".exr":
            raw_image = read_rgb_exr(path)
        else:
            raw_image = iio.imread(path.as_posix())
        raw_image = _as_rgb_image(np.asarray(raw_image), path)
        raw_is_floating_point = np.issubdtype(raw_image.dtype, np.floating)
        image = _normalize_image_samples(raw_image)

        if requested_color_space == "auto":
            interpreted_color_space = (
                "linear" if suffix in {".exr", ".hdr"} or raw_is_floating_point else "srgb"
            )
            interpretation = (
                "floating-point/HDR input"
                if interpreted_color_space == "linear"
                else "untagged integer input"
            )
        else:
            interpreted_color_space = requested_color_space
            interpretation = "explicit override"

    if interpreted_color_space == "srgb":
        image = srgb_to_linear(image)

    print(
        f"    color: {interpretation} interpreted as {interpreted_color_space}; "
        "training copy converted to linear sRGB"
    )
    return np.ascontiguousarray(image, dtype=np.float32)


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


def save_render(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(rgb, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    iio.imwrite(path.as_posix(), img_u8)


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
        densification_origins: np.ndarray | None = None,
        primitive_ages: np.ndarray | None = None,
        densification_position_signals: np.ndarray | None = None,
        densification_position_sample_counts: np.ndarray | None = None,
        densification_position_threshold: float | None = None,
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
    origin_values: np.ndarray | None = None
    if densification_origins is not None:
        origin_values = np.asarray(densification_origins, dtype=np.uint8).reshape(-1)
        if origin_values.shape[0] != num_points:
            raise ValueError(
                "Expected densification_origins to have one value per point, got "
                f"{origin_values.shape[0]} for {num_points} points"
            )
    age_values: np.ndarray | None = None
    if primitive_ages is not None:
        age_values = np.asarray(primitive_ages, dtype=np.uint32).reshape(-1)
        if age_values.shape[0] != num_points:
            raise ValueError(
                "Expected primitive_ages to have one value per point, got "
                f"{age_values.shape[0]} for {num_points} points"
            )
    position_signal_values: np.ndarray | None = None
    position_sample_count_values: np.ndarray | None = None
    position_threshold_value: float | None = None
    position_metadata = (
        densification_position_signals is not None
        or densification_position_sample_counts is not None
        or densification_position_threshold is not None
    )
    if position_metadata:
        if (
            densification_position_signals is None
            or densification_position_sample_counts is None
            or densification_position_threshold is None
        ):
            raise ValueError(
                "Position densification metadata requires signals, sample counts, and threshold"
            )
        position_signal_values = np.asarray(
            densification_position_signals, dtype=np.float32,
        ).reshape(-1)
        position_sample_count_values = np.asarray(
            densification_position_sample_counts, dtype=np.uint32,
        ).reshape(-1)
        if position_signal_values.shape[0] != num_points:
            raise ValueError(
                "Expected densification_position_signals to have one value per point, got "
                f"{position_signal_values.shape[0]} for {num_points} points"
            )
        if position_sample_count_values.shape[0] != num_points:
            raise ValueError(
                "Expected densification_position_sample_counts to have one value per point, got "
                f"{position_sample_count_values.shape[0]} for {num_points} points"
            )
        position_threshold_value = float(densification_position_threshold)
        if not np.isfinite(position_threshold_value) or position_threshold_value <= 0.0:
            raise ValueError(
                "densification_position_threshold must be finite and positive, got "
                f"{position_threshold_value}"
            )

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

    temporary_file_path: Path | None = None
    try:
        # Publish snapshots only after the complete file has been flushed and closed.
        # The temporary file lives beside the destination so os.replace is atomic.
        with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="ascii",
                dir=file_path.parent,
                prefix=f".{file_path.name}.",
                suffix=".tmp",
                delete=False,
        ) as f:
            temporary_file_path = Path(f.name)
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
            if origin_values is not None:
                # Stored as float for compatibility with the existing PLY float loader.
                f.write("property float densification_origin\n")
            if age_values is not None:
                # Stored as float for compatibility with the existing PLY scalar loader.
                f.write("property float primitive_age\n")
            if position_signal_values is not None:
                f.write("property float densification_position_signal\n")
                f.write("property float densification_position_sample_count\n")
                f.write("property float densification_position_threshold\n")
            f.write("end_header\n")

            for i in range(num_points):
                x, y, z = pos[i]
                qw, qx, qy, qz = rot[i]
                su_i, sv_i = sc[i, 0], sc[i, 1]
                albedo_r, albedo_g, albedo_b = col[i]

                origin_suffix = (
                    f" {float(origin_values[i]):.1f}"
                    if origin_values is not None
                    else ""
                )
                age_suffix = (
                    f" {float(age_values[i]):.1f}"
                    if age_values is not None
                    else ""
                )
                position_metadata_suffix = (
                    f" {position_signal_values[i]:.9g}"
                    f" {float(position_sample_count_values[i]):.9g}"
                    f" {position_threshold_value:.9g}"
                    if position_signal_values is not None
                    and position_sample_count_values is not None
                    and position_threshold_value is not None
                    else ""
                )
                f.write(
                    f"{x:.9g} {y:.9g} {z:.9g}  "
                    f"{qw:.9g} {qx:.9g} {qy:.9g} {qz:.9g}  "
                    f"{su_i:.9g} {sv_i:.9g}  "
                    f"{albedo_r:.9g} {albedo_g:.9g} {albedo_b:.9g}  "
                    f"{opa[i]:.9g} {beta_values[i]:.9g} {shape_default:.9g} {power_values[i]:.9g}"
                    f"{origin_suffix}{age_suffix}{position_metadata_suffix}\n"
                )

            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary_file_path, file_path)
        temporary_file_path = None
    finally:
        if temporary_file_path is not None:
            temporary_file_path.unlink(missing_ok=True)

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

def resolve_output_dir(output_dir: Path, output_dir_is_explicit: bool) -> Path:
    python_project_dir = Path(__file__).resolve().parent
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
