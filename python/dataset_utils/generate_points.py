#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Any


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

def normalize_vector(vector):
    x, y, z = vector
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-8:
        return None
    return [x / length, y / length, z / length]


def cross_product(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return [
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ]


def compute_tangent_basis_from_normal(nx, ny, nz,
                                      tu_fallback=(1.0, 0.0, 0.0),
                                      tv_fallback=(0.0, 1.0, 0.0)):
    """
    Given a normal (nx, ny, nz), compute an orthonormal tangent basis (tu, tv).
    Returns (tu_x, tu_y, tu_z, tv_x, tv_y, tv_z).
    If the normal is degenerate, fall back to defaults.
    """
    normal = normalize_vector([nx, ny, nz])
    if normal is None:
        return (*tu_fallback, *tv_fallback)

    nx, ny, nz = normal

    # Choose helper vector that is not parallel to normal
    if abs(nz) < 0.999:
        helper = [0.0, 0.0, 1.0]
    else:
        helper = [0.0, 1.0, 0.0]

    # tu = normalize(n × helper)
    tu = cross_product(normal, helper)
    tu = normalize_vector(tu)
    if tu is None:
        # Fallback if cross-product degenerates
        return (*tu_fallback, *tv_fallback)

    # tv = n × tu
    tv = cross_product(normal, tu)
    tv = normalize_vector(tv)
    if tv is None:
        return (*tu_fallback, *tv_fallback)

    return tu[0], tu[1], tu[2], tv[0], tv[1], tv[2]



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
    opacity: float,
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
    generatedPointCount = gridX * gridY * gridZ
    lightPointCount = 3
    totalPointCount = generatedPointCount + lightPointCount

    stepX = extentX / (gridX - 1) if gridX > 1 else 0.0
    stepY = extentY / (gridY - 1) if gridY > 1 else 0.0
    stepZ = extentZ / (gridZ - 1) if gridZ > 1 else 0.0

    defaultOpacity = opacity
    defaultBeta = -0.0
    defaultShape = 0.0
    defaultRGB = [0.3, 0.3, 0.3]
    color_noise = 0.05

    lines: list[str] = []
    lines.extend(
        [
            "ply",
            "format ascii 1.0",
            "comment Volume-initialized Gaussian surfels",
            "comment Includes one emissive point at (0, 0, 2.2)",
            f"element vertex {totalPointCount}",
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
            "property float power",
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

                r = defaultRGB[0] + random.gauss(0.0, color_noise)
                g = defaultRGB[1] + random.gauss(0.0, color_noise)
                b = defaultRGB[2] + random.gauss(0.0, color_noise)

                defaultPower = 0.0

                lines.append(
                    f"{x:.6f} {y:.6f} {z:.6f} "
                    f"{tu_x:.6f} {tu_y:.6f} {tu_z:.6f} "
                    f"{tv_x:.6f} {tv_y:.6f} {tv_z:.6f} "
                    f"{scaleValue:.6f} {scaleValue:.6f} "
                    f"{r:.6f} {g:.6f} {b:.6f} "
                    f"{defaultOpacity:.6f} {defaultBeta:.6f} {defaultShape:.6f} "
                    f"{defaultPower:.6f}"
                )


    light_power = 1200.0

    light_nx = 0.0
    light_ny = 0.0
    light_nz = -1.0

    light_tu_x, light_tu_y, light_tu_z, light_tv_x, light_tv_y, light_tv_z = (
        compute_tangent_basis_from_normal(light_nx, light_ny, light_nz)
    )

    light_x = 3
    light_y = -0.8
    light_z = 2.2
    light_su = 0.001
    light_sv = 0.001
    light_albedo_r = 1.0
    light_albedo_g = 1.0
    light_albedo_b = 1.0
    light_opacity = 1.0
    light_beta = -1000.0
    light_shape = 0.0

    light_line = (
        f"{light_x:.7f} {light_y:.7f} {light_z:.7f} "
        f"{light_tu_x:.7f} {light_tu_y:.7f} {light_tu_z:.7f} "
        f"{light_tv_x:.7f} {light_tv_y:.7f} {light_tv_z:.7f} "
        f"{light_su:.7f} {light_sv:.7f} "
        f"{light_albedo_r:.7f} {light_albedo_g:.7f} {light_albedo_b:.7f} "
        f"{light_opacity:.7f} {light_beta:.7f} {light_shape:.7f} {light_power:.7f}"
    )
    lines.append(light_line)

    light_x = -3
    light_y = 0.8
    light_z = 2.2
    light_albedo_r = 1.0
    light_albedo_g = 1.0
    light_albedo_b = 1.0
    light_opacity = 1.0
    light_beta = -100.0
    light_shape = 0.0
    light_power = 800.0

    light_line = (
        f"{light_x:.7f} {light_y:.7f} {light_z:.7f} "
        f"{light_tu_x:.7f} {light_tu_y:.7f} {light_tu_z:.7f} "
        f"{light_tv_x:.7f} {light_tv_y:.7f} {light_tv_z:.7f} "
        f"{light_su:.7f} {light_sv:.7f} "
        f"{light_albedo_r:.7f} {light_albedo_g:.7f} {light_albedo_b:.7f} "
        f"{light_opacity:.7f} {light_beta:.7f} {light_shape:.7f} {light_power:.7f}"
    )
    lines.append(light_line)
    light_x = 0.0
    light_y = 0.0
    light_z = 3.0
    light_albedo_r = 1.0
    light_albedo_g = 1.0
    light_albedo_b = 1.0
    light_opacity = 1.0
    light_beta = -1000.0
    light_shape = 0.0

    light_tu_x, light_tu_y, light_tu_z, light_tv_x, light_tv_y, light_tv_z = (
        compute_tangent_basis_from_normal(light_nx, light_ny, light_nz)
    )

    light_line = (
        f"{light_x:.7f} {light_y:.7f} {light_z:.7f} "
        f"{light_tu_x:.7f} {light_tu_y:.7f} {light_tu_z:.7f} "
        f"{light_tv_x:.7f} {light_tv_y:.7f} {light_tv_z:.7f} "
        f"{light_su:.7f} {light_sv:.7f} "
        f"{light_albedo_r:.7f} {light_albedo_g:.7f} {light_albedo_b:.7f} "
        f"{light_opacity:.7f} {light_beta:.7f} {light_shape:.7f} {light_power:.7f}"
    )
    lines.append(light_line)


    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"Written {totalPointCount} points "
        f"({generatedPointCount} grid points + 1 light point, requested {pointCount} grid points)\n"
        f"Grid: {gridX} x {gridY} x {gridZ}\n"
        f"AABB: x[{minX}, {maxX}] y[{minY}, {maxY}] z[{minZ}, {maxZ}]\n"
        f"Light point: position=(0.0, 0.0, 2.2), power"
    )



PRESETS: Dict[str, Dict[str, Any]] = {
    "teapot": {
        "min_x": -0.7,
        "max_x": 0.55,
        "min_y": -0.5,
        "max_y": 0.5,
        "min_z": -0.01,
        "max_z": 0.55,
        "scale": 0.025,
        "position_noise_std": 0.05,
        "tangent_noise_std": 45.0,
    },
    "plant": {
        "min_x": -0.45,
        "max_x": 0.45,
        "min_y": -0.45,
        "max_y": 0.45,
        "min_z": -0.01,
        "max_z": 0.6,
        "scale": 0.025,
        "position_noise_std": 0.05,
        "tangent_noise_std": 45.0,
    },
    "teapot_plane": {
        "min_x": -1.7,
        "max_x": 1.55,
        "min_y": -1.5,
        "max_y": 1.5,
        "min_z": -0.01,
        "max_z": 0.55,
        "scale": 0.025,
        "position_noise_std": 0.05,
        "tangent_noise_std": 5.0,
    },
    "bunny": {
        "min_x": -1,
        "max_x": 1,
        "min_y": -1,
        "max_y": 1,
        "min_z": 0.0,
        "max_z": 0.6,
        "scale": 0.02,
        "position_noise_std": 0.02,
        "tangent_noise_std": 5.0,
    },
    "plane": {
        "min_x": -0.25,
        "max_x": 0.25,
        "min_y": -0.25,
        "max_y": 0.25,
        "min_z": -0.01,
        "max_z": 0.01,
        "scale": 0.05,
        "position_noise_std": 0.02,
        "tangent_noise_std": 5.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill an axis-aligned volume with default-initialized Gaussian surfel points."
    )

    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)

    parser.add_argument(
        "--preset",
        type=str,
        choices=PRESETS.keys(),
        default="teapot",
        help="Preset that defines default volume and noise parameters",
    )

    parser.add_argument("--min-x", type=float)
    parser.add_argument("--max-x", type=float)
    parser.add_argument("--min-y", type=float)
    parser.add_argument("--max-y", type=float)
    parser.add_argument("--min-z", type=float)
    parser.add_argument("--max-z", type=float)

    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--opacity", type=float, default=0.5)
    parser.add_argument("--position-noise-std", type=float)
    parser.add_argument("--tangent-noise-std", type=float)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    apply_preset_defaults(args)
    return args


def apply_preset_defaults(args: argparse.Namespace) -> None:
    preset_values = PRESETS[args.preset]

    for key, value in preset_values.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


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
        opacity=args.opacity,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
