#!/usr/bin/env python3
"""
Visualize Gaussian splats as oriented colored ellipses in 3D from the latest run directory.

Expected directory structure:

    OUTPUT_ROOT/
        2025-11-28_14-19-18_lr5e+04_it5000_cbox_custom_color/
            points_final.ply
            ...
        2025-11-28_15-03-12_lr1e+04_it5000_cbox_custom_color/
            points_final.ply
            ...

Usage:
    python view_ply_ellipses_color.py --output-root /path/to/OptimizationOutput
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Gaussian splats as colored 3D ellipses from points_final.ply\n"
            "in the latest optimization run under --output-root."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Path to either:\n"
            "  (a) a single run directory containing points_final.ply, or\n"
            "  (b) a root directory containing multiple run subdirectories."
        ),
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


def find_latest_points_ply(outputRootPath: Path) -> Path:
    """
    If outputRootPath/points_final.ply exists, return it.
    Otherwise, search immediate subdirectories for points_final.ply and
    return the one whose points_final.ply has the latest modification time.
    """
    if not outputRootPath.exists():
        raise FileNotFoundError(f"Output root '{outputRootPath}' does not exist.")

    # Case we gave them the file
    pointsInRoot = outputRootPath
    if pointsInRoot.is_file():
        print(f"Using point_ploud.ply in run directory: {outputRootPath}")
        return pointsInRoot

    # Case 1: the provided path is already a run directory
    pointsInRoot = outputRootPath / "points_final.ply"
    if pointsInRoot.is_file():
        print(f"Using points_final.ply in run directory: {outputRootPath}")
        return pointsInRoot

    # Case 2: find latest subdirectory with points_final.ply
    candidateRunDirs: List[Path] = []
    for childPath in outputRootPath.iterdir():
        if not childPath.is_dir():
            continue
        candidatePlyPath = childPath / "points_final.ply"
        if candidatePlyPath.is_file():
            candidateRunDirs.append(childPath)

    if not candidateRunDirs:
        raise FileNotFoundError(
            f"No subdirectories with points_final.ply found under '{outputRootPath}'."
        )

    # Select run directory whose points_final.ply has latest mtime
    def ply_mtime(runPath: Path) -> float:
        plyPath = runPath / "points_final.ply"
        return plyPath.stat().st_mtime

    latestRunDir = max(candidateRunDirs, key=ply_mtime)
    latestPlyPath = latestRunDir / "points_final.ply"
    print(f"Using latest run directory: {latestRunDir}")
    print(f"points_final.ply: {latestPlyPath}")

    return latestPlyPath


# ---------------------------------------------------------
# LOAD SURFELS (INCLUDING COLOR)
# ---------------------------------------------------------
def load_surfels_from_ply(
    plyPath: Path,
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

    pkX, pkY, pkZ = [], [], []
    tuX, tuY, tuZ = [], [], []
    tvX, tvY, tvZ = [], [], []
    suValues, svValues = [], []
    color0, color1, color2 = [], [], []

    with plyPath.open("r", encoding="utf-8") as fileHandle:
        headerFinished = False

        for line in fileHandle:
            if not headerFinished:
                if line.strip() == "end_header":
                    headerFinished = True
                continue

            parts = line.strip().split()

            # Require enough attributes
            if len(parts) < 14:
                continue

            #opacity
            opacity = (float(parts[13]))

            if opacity < 0.5:
                continue
            # pk
            pkX.append(float(parts[0]))
            pkY.append(float(parts[1]))
            pkZ.append(float(parts[2]))

            # t_u
            tuX.append(float(parts[3]))
            tuY.append(float(parts[4]))
            tuZ.append(float(parts[5]))

            # t_v
            tvX.append(float(parts[6]))
            tvY.append(float(parts[7]))
            tvZ.append(float(parts[8]))

            # su, sv
            suValues.append(float(parts[9]))
            svValues.append(float(parts[10]))

            # COLOR (f_dc_0, f_dc_1, f_dc_2)
            color0.append(float(parts[11]))
            color1.append(float(parts[12]))
            color2.append(float(parts[13]))


    positions = np.stack([pkX, pkY, pkZ], axis=1)
    tangentU = np.stack([tuX, tuY, tuZ], axis=1)
    tangentV = np.stack([tvX, tvY, tvZ], axis=1)

    su = np.array(suValues)
    sv = np.array(svValues)
    colors = np.stack([color0, color1, color2], axis=1)

    print(f"Found {len(colors)} points")

    return positions, tangentU, tangentV, su, sv, colors


# ---------------------------------------------------------
# DRAW ELLIPSES (FILLED)
# ---------------------------------------------------------
def plot_ellipses_3d(
    positions: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    su: np.ndarray,
    sv: np.ndarray,
    colors: np.ndarray,
    area_threshold: float,
    max_ellipses: Optional[int],
    samples_per_ellipse: int,
) -> None:
    figureHandle = plt.figure(figsize=(8, 8))
    axisHandle = figureHandle.add_subplot(111, projection="3d")

    ellipseArea = su * sv
    ellipseMask = ellipseArea >= area_threshold

    positions = positions[ellipseMask]
    tangent_u = tangent_u[ellipseMask]
    tangent_v = tangent_v[ellipseMask]
    su = su[ellipseMask]
    sv = sv[ellipseMask]
    colors = colors[ellipseMask]

    if max_ellipses is not None:
        positions = positions[:max_ellipses]
        tangent_u = tangent_u[:max_ellipses]
        tangent_v = tangent_v[:max_ellipses]
        su = su[:max_ellipses]
        sv = sv[:max_ellipses]
        colors = colors[:max_ellipses]

    theta = np.linspace(0.0, 2.0 * np.pi, samples_per_ellipse)
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    allX, allY, allZ = [], [], []

    for centerPosition, tangentUVector, tangentVVector, suValue, svValue, colorVector in zip(
        positions, tangent_u, tangent_v, su, sv, colors
    ):
        tangentUNormalized = tangentUVector / (np.linalg.norm(tangentUVector) + 1e-12)
        tangentVNormalized = tangentVVector / (np.linalg.norm(tangentVVector) + 1e-12)

        ellipsePoints = (
            centerPosition[None, :]
            + suValue * cosTheta[:, None] * tangentUNormalized[None, :]
            + svValue * sinTheta[:, None] * tangentVNormalized[None, :]
        )

        faceColor = tuple(colorVector.clip(0.0, 1.0))

        poly = Poly3DCollection(
            [ellipsePoints],
            facecolors=[faceColor],
            edgecolors="black",
            linewidths=1.5,
            alpha=0.9,
        )
        axisHandle.add_collection3d(poly)

        allX.append(ellipsePoints[:, 0])
        allY.append(ellipsePoints[:, 1])
        allZ.append(ellipsePoints[:, 2])

    allX = np.concatenate(allX)
    allY = np.concatenate(allY)
    allZ = np.concatenate(allZ)

    axisHandle.set_xlabel("X")
    axisHandle.set_ylabel("Y")
    axisHandle.set_zlabel("Z")
    axisHandle.set_title("Gaussian Splat Ellipses (Colored, Filled)")

    maxRange = np.array(
        [allX.max() - allX.min(), allY.max() - allY.min(), allZ.max() - allZ.min()]
    ).max() / 2.0

    midX = (allX.max() + allX.min()) * 0.5
    midY = (allY.max() + allY.min()) * 0.5
    midZ = (allZ.max() + allZ.min()) * 0.5

    axisHandle.set_xlim(midX - maxRange, midX + maxRange)
    axisHandle.set_ylim(midY - maxRange, midY + maxRange)
    axisHandle.set_zlim(midZ - maxRange, midZ + maxRange)

    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_arguments()

    plyPath = find_latest_points_ply(args.output_root)

    positions, tangentU, tangentV, su, sv, colors = load_surfels_from_ply(plyPath)

    plot_ellipses_3d(
        positions=positions,
        tangent_u=tangentU,
        tangent_v=tangentV,
        su=su,
        sv=sv,
        colors=colors,
        area_threshold=args.area_threshold,
        max_ellipses=args.max_ellipses,
        samples_per_ellipse=args.samples_per_ellipse,
    )


if __name__ == "__main__":
    main()
