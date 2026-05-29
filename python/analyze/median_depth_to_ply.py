#!/usr/bin/env python3
"""
Backproject a median-depth EXR into a point cloud.

Assumptions:
- Depth image stores forward-axis camera depth:
      z = dot(P_world - camera_pos, camera_forward)
  which matches your renderer.
- Camera coordinates:
      +X right, +Y up, +Z forward
- Pixel coordinates:
      u right, v down
- Depth EXR may be RGBA with depth replicated in RGB and alpha as validity.
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import math
import struct
from pathlib import Path

import numpy as np


def read_image(path: Path) -> np.ndarray:
    """Read PNG/EXR/etc. Returns float32 or uint8 numpy image."""
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    try:
        import imageio.v3 as iio
        return iio.imread(path)
    except Exception as e:
        raise RuntimeError(f"Could not read image: {path}") from e


def extract_depth_and_validity(depth_img: np.ndarray, min_alpha: float):
    """
    Depth EXR from your code is RGBA:
        R = G = B = median depth
        A = validity
    Invalid pixels appear as all zeros, including alpha = 0.
    """
    depth_img = np.asarray(depth_img)

    if depth_img.ndim == 2:
        depth = depth_img.astype(np.float32)
        alpha_valid = np.ones_like(depth, dtype=bool)
    elif depth_img.ndim == 3:
        depth = depth_img[..., 0].astype(np.float32)

        if depth_img.shape[2] >= 4:
            alpha = depth_img[..., 3].astype(np.float32)
            alpha_valid = alpha > min_alpha
        else:
            alpha_valid = np.ones_like(depth, dtype=bool)
    else:
        raise ValueError(f"Unsupported depth image shape: {depth_img.shape}")

    return depth, alpha_valid


def extract_rgb(rgb_img: np.ndarray, expected_hw):
    """
    Optional color image. OpenCV loads PNG as BGR/BGRA, but EXR/PNG
    channel ordering is not critical for point geometry. For PNG from cv2,
    this swaps BGR -> RGB.
    """
    rgb_img = np.asarray(rgb_img)

    if rgb_img.shape[:2] != expected_hw:
        raise ValueError(
            f"RGB image resolution {rgb_img.shape[:2]} does not match depth {expected_hw}"
        )

    if rgb_img.ndim == 2:
        rgb = np.repeat(rgb_img[..., None], 3, axis=2)
    else:
        rgb = rgb_img[..., :3]

    # Assume cv2-style BGR for common 8-bit images.
    if rgb.dtype == np.uint8:
        rgb = rgb[..., ::-1]

    rgb = rgb.astype(np.float32)

    if rgb.max() <= 1.0:
        rgb = rgb * 255.0

    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return rgb


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray | None):
    path.parent.mkdir(parents=True, exist_ok=True)

    has_color = colors is not None
    n = points.shape[0]

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]

    header += ["end_header\n"]

    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))

        if has_color:
            for p, c in zip(points, colors):
                f.write(struct.pack(
                    "<fffBBB",
                    float(p[0]), float(p[1]), float(p[2]),
                    int(c[0]), int(c[1]), int(c[2])
                ))
        else:
            for p in points:
                f.write(struct.pack(
                    "<fff",
                    float(p[0]), float(p[1]), float(p[2])
                ))


def backproject_depth(
    depth: np.ndarray,
    valid: np.ndarray,
    fov_y_deg: float,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    stride: int,
):
    h, w = depth.shape

    if fy is None:
        fov_y = math.radians(fov_y_deg)
        fy = 0.5 * h / math.tan(0.5 * fov_y)

    if fx is None:
        fx = fy

    if cx is None:
        cx = 0.5 * w

    if cy is None:
        cy = 0.5 * h

    ys, xs = np.mgrid[0:h:stride, 0:w:stride]

    z = depth[0:h:stride, 0:w:stride].astype(np.float32) * depth_scale
    mask = valid[0:h:stride, 0:w:stride]

    mask &= np.isfinite(z)
    mask &= z > min_depth
    mask &= z < max_depth

    xs = xs.astype(np.float32) + 0.5
    ys = ys.astype(np.float32) + 0.5

    # Since z is forward-axis camera depth, not ray length:
    #
    #   X = (u - cx) / fx * z
    #   Y = -(v - cy) / fy * z
    #   Z = z
    #
    # The negative sign makes image-up correspond to +Y.
    x_cam = (xs - cx) * z / fx
    y_cam = -(ys - cy) * z / fy
    z_cam = z

    points = np.stack([x_cam, y_cam, z_cam], axis=-1)
    points = points[mask]

    return points, mask, (fx, fy, cx, cy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("depth_exr", type=Path)
    parser.add_argument("output_ply", type=Path)

    parser.add_argument("--rgb", type=Path, default=None,
                        help="Optional rendered RGB/RGBA image for point colors.")

    parser.add_argument("--fov-y-deg", type=float, default=30.0,
                        help="Vertical field of view used to synthesize pinhole intrinsics.")

    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)

    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=1e8)
    parser.add_argument("--min-alpha", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()

    depth_img = read_image(args.depth_exr)
    depth, valid = extract_depth_and_validity(depth_img, args.min_alpha)

    points, mask, intrinsics = backproject_depth(
        depth=depth,
        valid=valid,
        fov_y_deg=args.fov_y_deg,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        stride=args.stride,
    )

    colors = None
    if args.rgb is not None:
        rgb_img = read_image(args.rgb)
        rgb = extract_rgb(rgb_img, depth.shape)
        rgb_sub = rgb[0:depth.shape[0]:args.stride, 0:depth.shape[1]:args.stride]
        colors = rgb_sub[mask]

    write_binary_ply(args.output_ply, points, colors)

    fx, fy, cx, cy = intrinsics
    print(f"Wrote {points.shape[0]} points to {args.output_ply}")
    print(f"Intrinsics: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")


if __name__ == "__main__":
    main()