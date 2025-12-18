#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


def normalize3(x: float, y: float, z: float) -> tuple[float, float, float]:
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 0.0:
        return 1.0, 0.0, 0.0
    inv = 1.0 / length
    return x * inv, y * inv, z * inv


def orthonormalize_tangents(
    tu: tuple[float, float, float],
    tv: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    tu_x, tu_y, tu_z = normalize3(*tu)
    dot = tv[0] * tu_x + tv[1] * tu_y + tv[2] * tu_z
    tv_x = tv[0] - dot * tu_x
    tv_y = tv[1] - dot * tu_y
    tv_z = tv[2] - dot * tu_z
    tv_x, tv_y, tv_z = normalize3(tv_x, tv_y, tv_z)
    return (tu_x, tu_y, tu_z), (tv_x, tv_y, tv_z)


def rotate_tangent_frame_with_noise(
    tangentNoiseStd: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    tu = (1.0, 0.0, 0.0)
    tv = (0.0, 1.0, 0.0)

    if tangentNoiseStd <= 0.0:
        return tu, tv

    tangentNoiseStdRadians = math.radians(tangentNoiseStd)

    angle = random.gauss(0.0, tangentNoiseStdRadians)
    axis_x = random.gauss(0.0, 1.0)
    axis_y = random.gauss(0.0, 1.0)
    axis_z = random.gauss(0.0, 1.0)
    axis_x, axis_y, axis_z = normalize3(axis_x, axis_y, axis_z)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    def rotate(v):
        vx, vy, vz = v
        rx = (
            vx * cos_a
            + (axis_y * vz - axis_z * vy) * sin_a
            + axis_x * (axis_x * vx + axis_y * vy + axis_z * vz) * (1.0 - cos_a)
        )
        ry = (
            vy * cos_a
            + (axis_z * vx - axis_x * vz) * sin_a
            + axis_y * (axis_x * vx + axis_y * vy + axis_z * vz) * (1.0 - cos_a)
        )
        rz = (
            vz * cos_a
            + (axis_x * vy - axis_y * vx) * sin_a
            + axis_z * (axis_x * vx + axis_y * vy + axis_z * vz) * (1.0 - cos_a)
        )
        return rx, ry, rz

    tu_rot = rotate(tu)
    tv_rot = rotate(tv)
    return orthonormalize_tangents(tu_rot, tv_rot)


def compute_grid_dimensions_for_volume(
    targetPointCount: int,
    extentX: float,
    extentY: float,
    extentZ: float,
) -> tuple[int, int, int]:
    targetPointCount = max(1, int(targetPointCount))
    extentX = max(1e-12, extentX)
    extentY = max(1e-12, extentY)
    extentZ = max(1e-12, extentZ)

    volume = extentX * extentY * extentZ
    idealCellVolume = volume / float(targetPointCount)
    idealSpacing = idealCellVolume ** (1.0 / 3.0)

    nx = max(1, int(round(extentX / idealSpacing)))
    ny = max(1, int(round(extentY / idealSpacing)))
    nz = max(1, int(round(extentZ / idealSpacing)))

    best = (nx, ny, nz)
    bestError = abs(nx * ny * nz - targetPointCount)

    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for dz in range(-2, 3):
                cx = max(1, nx + dx)
                cy = max(1, ny + dy)
                cz = max(1, nz + dz)
                err = abs(cx * cy * cz - targetPointCount)
                if err < bestError:
                    bestError = err
                    best = (cx, cy, cz)

    return best


def generate_volume_ply(
    outputPath: Path,
    minX: float,
    maxX: float,
    minY: float,
    maxY: float,
    minZ: float,
    maxZ: float,
    pointCount: int,
    scaleValue: float,
    positionNoiseStd: float,
    tangentNoiseStd: float,
    seed: int | None,
) -> None:
    if seed is not None:
        random.seed(seed)

    minX, maxX = sorted((minX, maxX))
    minY, maxY = sorted((minY, maxY))
    minZ, maxZ = sorted((minZ, maxZ))

    extentX = maxX - minX
    extentY = maxY - minY
    extentZ = maxZ - minZ

    gridX, gridY, gridZ = compute_grid_dimensions_for_volume(
        pointCount, extentX, extentY, extentZ
    )
    actualPointCount = gridX * gridY * gridZ

    stepX = extentX / (gridX - 1) if gridX > 1 else 0.0
    stepY = extentY / (gridY - 1) if gridY > 1 else 0.0
    stepZ = extentZ / (gridZ - 1) if gridZ > 1 else 0.0

    defaultOpacity = 0.4
    defaultBeta = 0.0
    defaultShape = 0.0
    defaultRGB = [0.8, 0.9, 0.9]

    lines: list[str] = []
    lines.extend(
        [
            "ply",
            "format ascii 1.0",
            "comment Volume-initialized Gaussian surfels",
            f"element vertex {actualPointCount}",
            "property float x",
            "property float y",
            "property float z",
            "property float tu_x",
            "property float tu_y",
            "property float tu_z",
            "property float tv_x",
            "property float tv_y",
            "property float tv_z",
            "property float su",
            "property float sv",
            "property float albedo_r",
            "property float albedo_g",
            "property float albedo_b",
            "property float opacity",
            "property float beta",
            "property float shape",
            "end_header",
        ]
    )

    for kz in range(gridZ):
        z0 = minZ + kz * stepZ if gridZ > 1 else 0.5 * (minZ + maxZ)
        for jy in range(gridY):
            y0 = minY + jy * stepY if gridY > 1 else 0.5 * (minY + maxY)
            for ix in range(gridX):
                x0 = minX + ix * stepX if gridX > 1 else 0.5 * (minX + maxX)

                x = x0 + random.gauss(0.0, positionNoiseStd)
                y = y0 + random.gauss(0.0, positionNoiseStd)
                z = z0 + random.gauss(0.0, positionNoiseStd)

                (tu_x, tu_y, tu_z), (tv_x, tv_y, tv_z) = rotate_tangent_frame_with_noise(
                    tangentNoiseStd
                )

                lines.append(
                    f"{x:.6f} {y:.6f} {z:.6f} "
                    f"{tu_x:.6f} {tu_y:.6f} {tu_z:.6f} "
                    f"{tv_x:.6f} {tv_y:.6f} {tv_z:.6f} "
                    f"{scaleValue:.6f} {scaleValue:.6f} "
                    f"{defaultRGB[0]:.6f} {defaultRGB[1]:.6f} {defaultRGB[2]:.6f} "
                    f"{defaultOpacity:.6f} {defaultBeta:.6f} {defaultShape:.6f}"
                )

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"Written {actualPointCount} points (requested {pointCount})\n"
        f"Grid: {gridX} x {gridY} x {gridZ}\n"
        f"AABB: x[{minX}, {maxX}] y[{minY}, {maxY}] z[{minZ}, {maxZ}]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill an axis-aligned volume with default-initialized Gaussian surfel points."
    )

    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)

    # ✅ DEFAULTS REQUESTED
    parser.add_argument("--min-x", type=float, default=-0.55)
    parser.add_argument("--max-x", type=float, default=0.5)
    parser.add_argument("--min-y", type=float, default=-0.3)
    parser.add_argument("--max-y", type=float, default=0.3)
    parser.add_argument("--min-z", type=float, default=0.01)
    parser.add_argument("--max-z", type=float, default=0.55)

    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--position-noise-std", type=float, default=0.0)
    parser.add_argument("--tangent-noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_volume_ply(
        outputPath=args.out,
        minX=args.min_x,
        maxX=args.max_x,
        minY=args.min_y,
        maxY=args.max_y,
        minZ=args.min_z,
        maxZ=args.max_z,
        pointCount=args.count,
        scaleValue=args.scale,
        positionNoiseStd=args.position_noise_std,
        tangentNoiseStd=args.tangent_noise_std,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
