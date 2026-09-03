from pathlib import Path

import OpenEXR
import Imath
import numpy as np
from PIL import Image

def save_rgb_preview_exr(
        img_f32: np.ndarray,
        out_path: Path,
        exposure_stops: float = 0.0,
) -> None:
    """
    Save a linear RGB/RGBA float32 image as a 16-bit (HALF) or 32-bit (FLOAT) EXR.

    img_f32:       HxWx3 (RGB) or HxWx4 (RGBA), linear, usually HDR (0..+inf)
    exposure_stops: photographic EV; +1 doubles brightness

    Notes:
    - No gamma is applied. EXR remains linear.
    - Channels are written as R, G, B.
    - If RGBA is provided, alpha is discarded.
    """

    img = np.asarray(img_f32, dtype=np.float32)

    # --- Handle RGB vs RGBA explicitly ---
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4 image, got {img.shape}")

    if img.shape[2] == 4:
        img = img[..., :3]  # drop alpha deterministically

    if exposure_stops != 0.0:
        img = img * (2.0 ** exposure_stops)

    img = np.clip(img, 0.0, None)

    height, width, _ = img.shape

    # Split channels (EXR expects planar layout)
    r = img[..., 0].astype(np.float32).tobytes()
    g = img[..., 1].astype(np.float32).tobytes()
    b = img[..., 2].astype(np.float32).tobytes()

    header = OpenEXR.Header(width, height)
    header["channels"] = {
        "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exr = OpenEXR.OutputFile(str(out_path), header)
    exr.writePixels({"R": r, "G": g, "B": b})
    exr.close()


def save_rgb_preview_png(
        img_f32: np.ndarray,
        out_path: Path,
        exposure_stops: float = 0.0,
        gamma: float = 1.0,
) -> None:
    """
    Save a linear RGB/RGBA float32 image as an 8-bit PNG.

    img_f32:       HxWx3 (RGB) or HxWx4 (RGBA), linear, usually HDR (0..+inf)
    exposure_stops: photographic EV; +1 doubles brightness
    gamma:         gamma for encoding (e.g. 2.2 for sRGB)

    If RGBA is provided, the alpha channel is discarded.
    """

    img = np.asarray(img_f32, dtype=np.float32)

    # --- Handle RGB vs RGBA explicitly ---
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4 image, got {img.shape}")

    if img.shape[2] == 4:
        img = img[..., :3]  # drop alpha deterministically

    # --- Same processing as before ---
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

    Image.fromarray(img_u8, mode="RGB").save(out_path)
