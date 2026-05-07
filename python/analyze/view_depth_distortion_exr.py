from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Try imageio first
try:
    import imageio.v3 as iio

    def read_exr(path: str) -> np.ndarray:
        img = iio.imread(path)
        img = np.asarray(img, dtype=np.float32)
        return img
except Exception:
    # Fallback: OpenCV (may require OPENCV_IO_ENABLE_OPENEXR=1)
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2

    def read_exr(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read EXR: {path}")
        # cv2 returns BGR(A)
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 3 and img.shape[2] >= 3:
            img = img[..., ::-1]  # BGR -> RGB (or BGRA -> ARGB-ish, first 3 channels enough)
        return img


def visualize_scalar_exr(path: str, use_log: bool = True) -> None:
    img = read_exr(path)

    # You saved the same scalar into R,G,B, so take channel 0 if multi-channel
    if img.ndim == 3:
        scalar = img[..., 0]
    else:
        scalar = img

    scalar = np.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"shape={scalar.shape}")
    print(f"min={scalar.min():.6e}, max={scalar.max():.6e}, mean={scalar.mean():.6e}")
    print(f"p50={np.percentile(scalar, 50):.6e}")
    print(f"p90={np.percentile(scalar, 90):.6e}")
    print(f"p99={np.percentile(scalar, 99):.6e}")
    print(f"p99.9={np.percentile(scalar, 99.9):.6e}")

    # Robust clipping for visualization
    lo = np.percentile(scalar, 1.0)
    hi = np.percentile(scalar, 99.0)
    vis = np.clip(scalar, lo, hi)

    if use_log:
        vis = np.log1p(np.maximum(vis, 0.0))
        title = f"{Path(path).name} (log1p, clipped 1-99%)"
    else:
        title = f"{Path(path).name} (linear, clipped 1-99%)"

    plt.figure(figsize=(8, 6))
    im = plt.imshow(vis, cmap="magma")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_scalar_exr("/home/magnus-desktop/CLionProjects/DifferentiablePointRendering/Assets/Output/OptimizerTests/teapot/images/DatasetCam_000depth_distortion.exr", use_log=True)