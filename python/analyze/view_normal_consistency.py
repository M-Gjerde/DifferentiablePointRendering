import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2


def load_exr_rgba(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Failed to load EXR: {path}")

    arr = arr.astype(np.float32)

    if arr.ndim == 2:
        arr = arr[..., None]

    # OpenCV returns BGR/BGRA
    if arr.shape[2] == 4:
        arr = arr[..., [2, 1, 0, 3]]
    elif arr.shape[2] == 3:
        arr = arr[..., [2, 1, 0]]
        alpha = np.ones((*arr.shape[:2], 1), dtype=np.float32)
        arr = np.concatenate([arr, alpha], axis=2)
    elif arr.shape[2] == 1:
        rgb = np.repeat(arr, 3, axis=2)
        alpha = np.ones((*arr.shape[:2], 1), dtype=np.float32)
        arr = np.concatenate([rgb, alpha], axis=2)
    else:
        raise RuntimeError(f"Unsupported EXR channel count {arr.shape[2]} in {path}")

    return arr


def normalize_vectors(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > eps, v / n, 0.0)


def remap_normal_for_display(n: np.ndarray) -> np.ndarray:
    n = normalize_vectors(n)
    return np.clip(0.5 * (n + 1.0), 0.0, 1.0)


def robust_depth_display(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(depth, dtype=np.float32)
    vals = depth[valid]
    if vals.size == 0:
        return out

    lo = np.percentile(vals, 1.0)
    hi = np.percentile(vals, 99.0)
    if hi <= lo:
        hi = lo + 1e-6

    out[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def print_stats(name: str, arr: np.ndarray, valid: np.ndarray | None = None) -> None:
    if valid is None:
        vals = arr[np.isfinite(arr)]
    else:
        vals = arr[valid & np.isfinite(arr)]

    if vals.size == 0:
        print(f"{name}: no valid values")
        return

    print(
        f"{name}: min={vals.min():.6f}  max={vals.max():.6f}  "
        f"mean={vals.mean():.6f}  std={vals.std():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the EXRs")
    parser.add_argument("--camera", type=str, default="camera1", help="Filename prefix before _median_depth.exr etc.")
    args = parser.parse_args()

    base = Path(args.dir)
    prefix = args.camera

    depth_path = base / f"{prefix}_median_depth.exr"
    world_path = base / f"{prefix}_median_world_position.exr"
    visn_path = base / f"{prefix}_visible_normal.exr"
    dnorm_path = base / f"{prefix}_normal_from_depth.exr"

    depth_rgba = load_exr_rgba(depth_path)
    world_rgba = load_exr_rgba(world_path)
    visn_rgba = load_exr_rgba(visn_path)
    dnorm_rgba = load_exr_rgba(dnorm_path)

    depth = depth_rgba[..., 0]
    depth_valid = depth_rgba[..., 3] > 0.5

    world = world_rgba[..., :3]
    world_valid = world_rgba[..., 3] > 0.5
    if not np.any(world_valid):
        world_valid = depth_valid.copy()

    visible_normal = normalize_vectors(visn_rgba[..., :3])
    visible_normal_valid = visn_rgba[..., 3] > 0.5
    if not np.any(visible_normal_valid):
        visible_normal_valid = depth_valid.copy()

    depth_normal = normalize_vectors(dnorm_rgba[..., :3])
    depth_normal_valid = dnorm_rgba[..., 3] > 0.5
    if not np.any(depth_normal_valid):
        depth_normal_valid = depth_valid.copy()

    joint_valid = depth_valid & world_valid & visible_normal_valid & depth_normal_valid

    depth_disp = robust_depth_display(depth, depth_valid)
    visn_disp = remap_normal_for_display(visible_normal)
    dnorm_disp = remap_normal_for_display(depth_normal)

    dot_map = np.sum(visible_normal * depth_normal, axis=-1)
    dot_map = np.clip(dot_map, -1.0, 1.0)

    angle_deg = np.degrees(np.arccos(np.clip(dot_map, -1.0, 1.0)))
    angle_deg[~joint_valid] = 0.0

    print_stats("median depth", depth, depth_valid)
    print_stats("world x", world[..., 0], world_valid)
    print_stats("world y", world[..., 1], world_valid)
    print_stats("world z", world[..., 2], world_valid)
    print_stats("visible normal x", visible_normal[..., 0], visible_normal_valid)
    print_stats("visible normal y", visible_normal[..., 1], visible_normal_valid)
    print_stats("visible normal z", visible_normal[..., 2], visible_normal_valid)
    print_stats("depth normal x", depth_normal[..., 0], depth_normal_valid)
    print_stats("depth normal y", depth_normal[..., 1], depth_normal_valid)
    print_stats("depth normal z", depth_normal[..., 2], depth_normal_valid)
    print_stats("normal dot", dot_map, joint_valid)
    print_stats("normal angular error (deg)", angle_deg, joint_valid)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    ax = axes.ravel()

    im0 = ax[0].imshow(depth_disp, cmap="viridis")
    ax[0].set_title("Median depth")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    ax[1].imshow(depth_valid, cmap="gray")
    ax[1].set_title("Depth valid mask")

    ax[2].imshow(visn_disp)
    ax[2].set_title("Visible normal")

    ax[3].imshow(dnorm_disp)
    ax[3].set_title("Normal from depth")

    x_img = world[..., 0].copy()
    y_img = world[..., 1].copy()
    z_img = world[..., 2].copy()
    for img in (x_img, y_img, z_img):
        img[~world_valid] = 0.0

    im4 = ax[4].imshow(x_img, cmap="coolwarm")
    ax[4].set_title("World X")
    plt.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)

    im5 = ax[5].imshow(y_img, cmap="coolwarm")
    ax[5].set_title("World Y")
    plt.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.04)

    im6 = ax[6].imshow(z_img, cmap="coolwarm")
    ax[6].set_title("World Z")
    plt.colorbar(im6, ax=ax[6], fraction=0.046, pad=0.04)

    dot_disp = dot_map.copy()
    dot_disp[~joint_valid] = 0.0
    im7 = ax[7].imshow(dot_disp, cmap="magma", vmin=-1.0, vmax=1.0)
    ax[7].set_title("Normal dot(visible, depth)")
    plt.colorbar(im7, ax=ax[7], fraction=0.046, pad=0.04)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
    im_angle = ax2.imshow(angle_deg, cmap="inferno", vmin=0.0, vmax=90.0)
    ax2.set_title("Angular error (deg)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(im_angle, ax=ax2, fraction=0.046, pad=0.04)

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if x < 0 or y < 0 or x >= depth.shape[1] or y >= depth.shape[0]:
            return

        print("\n--- Pixel inspect ---")
        print(f"pixel = ({x}, {y})")
        print(f"depth valid = {depth_valid[y, x]}")
        print(f"median depth = {depth[y, x]:.6f}")
        print(f"world pos = {world[y, x]}")
        print(f"visible normal = {visible_normal[y, x]}")
        print(f"depth normal = {depth_normal[y, x]}")
        print(f"normal dot = {dot_map[y, x]:.6f}")
        print(f"angular error deg = {angle_deg[y, x]:.6f}")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig2.canvas.mpl_connect("button_press_event", on_click)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()