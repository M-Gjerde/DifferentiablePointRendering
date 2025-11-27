#!/usr/bin/env python3
"""
Visualize Gaussian splats as oriented colored ellipses in 3D from an ASCII PLY file.

Assumes per-vertex layout:
    pk.x pk.y pk.z
    tu_x tu_y tu_z
    tv_x tv_y tv_z
    su sv
    f_dc_0 f_dc_1 f_dc_2   <-- COLOR
    ...

Usage:
    python view_ply_ellipses_color.py --input my_points.ply
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Gaussian splats as colored 3D ellipses from a PLY file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the ASCII PLY file.",
    )
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=0.0,
        help="Minimum ellipse area su*sv to display.",
    )
    parser.add_argument(
        "--max-ellipses",
        type=int,
        default=None,
        help="Cap number of ellipses displayed.",
    )
    parser.add_argument(
        "--samples-per-ellipse",
        type=int,
        default=64,
        help="Circle sampling resolution.",
    )
    return parser.parse_args()


# ---------------------------------------------------------
# LOAD SURFELS (INCLUDING COLOR)
# ---------------------------------------------------------
def load_surfels_from_ply(
    ply_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads:
        pk      -> (x,y,z)
        t_u     -> (3,)
        t_v     -> (3,)
        su, sv  -> float
        color   -> (3,)  (f_dc_0, f_dc_1, f_dc_2)

    Returns:
        positions: (N,3)
        tangent_u: (N,3)
        tangent_v: (N,3)
        su:        (N,)
        sv:        (N,)
        colors:    (N,3)
    """

    pk_x, pk_y, pk_z = [], [], []
    tu_x, tu_y, tu_z = [], [], []
    tv_x, tv_y, tv_z = [], [], []
    su_vals, sv_vals = [], []
    c0, c1, c2 = [], [], []

    with ply_path.open("r", encoding="utf-8") as file:
        header_finished = False

        for line in file:
            if not header_finished:
                if line.strip() == "end_header":
                    header_finished = True
                continue

            parts = line.strip().split()

            # Require enough attributes
            if len(parts) < 14:
                continue

            # pk
            pk_x.append(float(parts[0]))
            pk_y.append(float(parts[1]))
            pk_z.append(float(parts[2]))

            # t_u
            tu_x.append(float(parts[3]))
            tu_y.append(float(parts[4]))
            tu_z.append(float(parts[5]))

            # t_v
            tv_x.append(float(parts[6]))
            tv_y.append(float(parts[7]))
            tv_z.append(float(parts[8]))

            # su, sv
            su_vals.append(float(parts[9]))
            sv_vals.append(float(parts[10]))

            # COLOR (f_dc_0, f_dc_1, f_dc_2)
            c0.append(float(parts[11]))
            c1.append(float(parts[12]))
            c2.append(float(parts[13]))

    positions = np.stack([pk_x, pk_y, pk_z], axis=1)
    tangent_u = np.stack([tu_x, tu_y, tu_z], axis=1)
    tangent_v = np.stack([tv_x, tv_y, tv_z], axis=1)
    su = np.array(su_vals)
    sv = np.array(sv_vals)
    colors = np.stack([c0, c1, c2], axis=1)

    return positions, tangent_u, tangent_v, su, sv, colors


# ---------------------------------------------------------
# DRAW ELLIPSES
# ---------------------------------------------------------
def plot_ellipses_3d(
    positions,
    tangent_u,
    tangent_v,
    su,
    sv,
    colors,
    area_threshold,
    max_ellipses,
    samples_per_ellipse,
):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    area = su * sv
    mask = area >= area_threshold

    positions = positions[mask]
    tangent_u = tangent_u[mask]
    tangent_v = tangent_v[mask]
    su = su[mask]
    sv = sv[mask]
    colors = colors[mask]

    if max_ellipses is not None:
        positions = positions[:max_ellipses]
        tangent_u = tangent_u[:max_ellipses]
        tangent_v = tangent_v[:max_ellipses]
        su = su[:max_ellipses]
        sv = sv[:max_ellipses]
        colors = colors[:max_ellipses]

    theta = np.linspace(0, 2 * np.pi, samples_per_ellipse)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    all_x, all_y, all_z = [], [], []

    for pk, tu, tv, su_i, sv_i, col in zip(
        positions, tangent_u, tangent_v, su, sv, colors
    ):
        tu_n = tu / (np.linalg.norm(tu) + 1e-12)
        tv_n = tv / (np.linalg.norm(tv) + 1e-12)

        ellipse_pts = (
            pk[None, :]
            + su_i * cos_t[:, None] * tu_n[None, :]
            + sv_i * sin_t[:, None] * tv_n[None, :]
        )

        # color stays constant per-ellipse
        ax.plot(
            ellipse_pts[:, 0],
            ellipse_pts[:, 1],
            ellipse_pts[:, 2],
            color=col.clip(0, 1),
            linewidth=1.0,
        )

        all_x.append(ellipse_pts[:, 0])
        all_y.append(ellipse_pts[:, 1])
        all_z.append(ellipse_pts[:, 2])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Gaussian Splat Ellipses (Colored)")

    max_range = np.array(
        [all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()]
    ).max() / 2

    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
def main():
    args = parse_arguments()

    positions, tu, tv, su, sv, colors = load_surfels_from_ply(args.input)

    plot_ellipses_3d(
        positions=positions,
        tangent_u=tu,
        tangent_v=tv,
        su=su,
        sv=sv,
        colors=colors,
        area_threshold=args.area_threshold,
        max_ellipses=args.max_ellipses,
        samples_per_ellipse=args.samples_per_ellipse,
    )


if __name__ == "__main__":
    main()
