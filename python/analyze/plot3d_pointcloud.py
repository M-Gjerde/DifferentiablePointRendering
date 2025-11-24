#!/usr/bin/env python3
"""
Render a 3D point cloud from a PLY file (Gaussian splat format) using matplotlib.

Usage:
    python view_ply_points.py --input initial.ply
"""

import argparse
from pathlib import Path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize point positions (x, y, z) from a PLY file.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the PLY file.")
    return parser.parse_args()


def load_positions_and_scales_from_ply(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads (x,y,z) and (su,sv) from the ASCII PLY file.
    Returns:
        positions: (N, 3)
        scales:    (N, 2)  # (su, sv)
    """
    x_vals, y_vals, z_vals = [], [], []
    su_vals, sv_vals = [], []

    with ply_path.open("r", encoding="utf-8") as file:
        header_finished = False

        for line in file:
            if not header_finished:
                if line.strip() == "end_header":
                    header_finished = True
                continue

            parts = line.strip().split()
            if len(parts) < 11:
                continue

            # pk.x pk.y pk.z tu_x tu_y tu_z tv_x tv_y tv_z su sv ...
            x_vals.append(float(parts[0]))
            y_vals.append(float(parts[1]))
            z_vals.append(float(parts[2]))

            su_vals.append(float(parts[9]))
            sv_vals.append(float(parts[10]))

    positions = np.stack([x_vals, y_vals, z_vals], axis=1)
    scales = np.stack([su_vals, sv_vals], axis=1)
    return positions, scales


def plot_point_cloud(positions: np.ndarray) -> None:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    xs = positions[:, 0]
    ys = positions[:, 1]
    zs = positions[:, 2]

    ax.scatter(xs, ys, zs, s=100, c="blue", depthshade=True)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Filtered Gaussian Splat Point Cloud")

    # Axis equalization
    max_range = np.array([
        xs.max() - xs.min(),
        ys.max() - ys.min(),
        zs.max() - zs.min()
    ]).max() / 2.0

    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def main() -> None:
    args = parse_arguments()

    positions, scales = load_positions_and_scales_from_ply(args.input)

    su = scales[:, 0]
    sv = scales[:, 1]

    # ---- Custom area threshold ----
    area = su * sv
    AREA_THRESHOLD = 0.01  # change this to whatever you want1

    mask = area >= AREA_THRESHOLD
    filtered_positions = positions[mask]

    print(f"Loaded {len(positions)} points, showing {len(filtered_positions)} after filtering.")

    plot_point_cloud(filtered_positions)


if __name__ == "__main__":
    main()
