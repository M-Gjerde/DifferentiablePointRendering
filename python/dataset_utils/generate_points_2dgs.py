#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def normalize3(x: float, y: float, z: float) -> Tuple[float, float, float]:
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 0.0:
        return 1.0, 0.0, 0.0
    inv = 1.0 / length
    return x * inv, y * inv, z * inv


def rotate_vector_axis_angle(
    v: Tuple[float, float, float],
    axis: Tuple[float, float, float],
    angle_radians: float,
) -> Tuple[float, float, float]:
    ax, ay, az = normalize3(*axis)
    vx, vy, vz = v
    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)

    rx = (
        vx * cos_a
        + (ay * vz - az * vy) * sin_a
        + ax * (ax * vx + ay * vy + az * vz) * (1.0 - cos_a)
    )
    ry = (
        vy * cos_a
        + (az * vx - ax * vz) * sin_a
        + ay * (ax * vx + ay * vy + az * vz) * (1.0 - cos_a)
    )
    rz = (
        vz * cos_a
        + (ax * vy - ay * vx) * sin_a
        + az * (ax * vx + ay * vy + az * vz) * (1.0 - cos_a)
    )
    return rx, ry, rz


def sample_random_unit_vector() -> Tuple[float, float, float]:
    return normalize3(
        random.gauss(0.0, 1.0),
        random.gauss(0.0, 1.0),
        random.gauss(0.0, 1.0),
    )


def compute_grid_dimensions_for_volume(
    target_point_count: int,
    extent_x: float,
    extent_y: float,
    extent_z: float,
) -> Tuple[int, int, int]:
    target_point_count = max(1, int(target_point_count))
    extent_x = max(1e-12, float(extent_x))
    extent_y = max(1e-12, float(extent_y))
    extent_z = max(1e-12, float(extent_z))

    volume = extent_x * extent_y * extent_z
    ideal_cell_volume = volume / float(target_point_count)
    ideal_spacing = ideal_cell_volume ** (1.0 / 3.0)

    nx = max(1, int(round(extent_x / ideal_spacing)))
    ny = max(1, int(round(extent_y / ideal_spacing)))
    nz = max(1, int(round(extent_z / ideal_spacing)))

    best = (nx, ny, nz)
    best_error = abs(nx * ny * nz - target_point_count)

    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for dz in range(-2, 3):
                cx = max(1, nx + dx)
                cy = max(1, ny + dy)
                cz = max(1, nz + dz)
                err = abs(cx * cy * cz - target_point_count)
                if err < best_error:
                    best_error = err
                    best = (cx, cy, cz)

    return best


def iter_volume_grid_points(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    grid_x: int,
    grid_y: int,
    grid_z: int,
) -> Iterable[Tuple[float, float, float]]:
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    extent_z = max_z - min_z

    step_x = extent_x / (grid_x - 1) if grid_x > 1 else 0.0
    step_y = extent_y / (grid_y - 1) if grid_y > 1 else 0.0
    step_z = extent_z / (grid_z - 1) if grid_z > 1 else 0.0

    for kz in range(grid_z):
        z0 = min_z + kz * step_z if grid_z > 1 else 0.5 * (min_z + max_z)
        for jy in range(grid_y):
            y0 = min_y + jy * step_y if grid_y > 1 else 0.5 * (min_y + max_y)
            for ix in range(grid_x):
                x0 = min_x + ix * step_x if grid_x > 1 else 0.5 * (min_x + max_x)
                yield x0, y0, z0


def build_simple_ply_header_ascii(vertex_count: int) -> str:
    # Minimal PLY: position + normal + color
    # Use uchar RGB because many PLY loaders assume this.
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {vertex_count}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    return "\n".join(lines) + "\n"


def generate_points_ply_minimal(
    output_path: Path,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    point_count: int,
    position_noise_std: float,
    normal_noise_deg: float,
    seed: int | None,
    base_rgb: Tuple[float, float, float],
    color_noise_std: float,
    normal_mode: str,
) -> None:
    if seed is not None:
        random.seed(seed)

    min_x, max_x = sorted((float(min_x), float(max_x)))
    min_y, max_y = sorted((float(min_y), float(max_y)))
    min_z, max_z = sorted((float(min_z), float(max_z)))

    extent_x = max_x - min_x
    extent_y = max_y - min_y
    extent_z = max_z - min_z

    grid_x, grid_y, grid_z = compute_grid_dimensions_for_volume(
        point_count, extent_x, extent_y, extent_z
    )
    actual_point_count = grid_x * grid_y * grid_z

    header = build_simple_ply_header_ascii(actual_point_count)
    normal_noise_rad = math.radians(max(0.0, float(normal_noise_deg)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(header)

        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        cz = 0.5 * (min_z + max_z)

        for (x0, y0, z0) in iter_volume_grid_points(
            min_x, max_x, min_y, max_y, min_z, max_z, grid_x, grid_y, grid_z
        ):
            x = x0 + random.gauss(0.0, position_noise_std)
            y = y0 + random.gauss(0.0, position_noise_std)
            z = z0 + random.gauss(0.0, position_noise_std)

            if normal_mode == "fixed":
                nx, ny, nz = 0.0, 1.0, 0.0
            elif normal_mode == "random":
                nx, ny, nz = sample_random_unit_vector()
            elif normal_mode == "outward":
                nx, ny, nz = normalize3(x - cx, y - cy, z - cz)
            else:
                raise ValueError(f"Unknown normal_mode: {normal_mode}")

            if normal_noise_rad > 0.0:
                axis = sample_random_unit_vector()
                nx, ny, nz = rotate_vector_axis_angle(
                    (nx, ny, nz),
                    axis,
                    random.gauss(0.0, normal_noise_rad),
                )
                nx, ny, nz = normalize3(nx, ny, nz)

            r = base_rgb[0] + random.gauss(0.0, color_noise_std)
            g = base_rgb[1] + random.gauss(0.0, color_noise_std)
            b = base_rgb[2] + random.gauss(0.0, color_noise_std)

            # map to 0..255
            r_u8 = int(round(max(0.0, min(1.0, r)) * 255.0))
            g_u8 = int(round(max(0.0, min(1.0, g)) * 255.0))
            b_u8 = int(round(max(0.0, min(1.0, b)) * 255.0))

            f.write(
                f"{x:.6f} {y:.6f} {z:.6f} "
                f"{nx:.6f} {ny:.6f} {nz:.6f} "
                f"{r_u8:d} {g_u8:d} {b_u8:d}\n"
            )

    print(
        f"Wrote ASCII PLY: {output_path}\n"
        f"Points: {actual_point_count} (requested {point_count})\n"
        f"Grid: {grid_x} x {grid_y} x {grid_z}\n"
        f"AABB: x[{min_x}, {max_x}] y[{min_y}, {max_y}] z[{min_z}, {max_z}]"
    )


PRESETS: Dict[str, Dict[str, Any]] = {
    "teapot": {
        "min_x": -0.55,
        "max_x": 0.55,
        "min_y": -0.40,
        "max_y": 0.40,
        "min_z": 0.10,
        "max_z": 0.55,
        "position_noise_std": 0.05,
        "normal_noise_deg": 45.0,
        "base_r": 0.7,
        "base_g": 0.7,
        "base_b": 0.7,
        "color_noise_std": 0.2,
    },
    "bunny": {
        "min_x": -0.40,
        "max_x": 0.40,
        "min_y": -0.35,
        "max_y": 0.35,
        "min_z": 0.30,
        "max_z": 1.20,
        "position_noise_std": 0.02,
        "normal_noise_deg": 30.0,
        "base_r": 0.7,
        "base_g": 0.7,
        "base_b": 0.7,
        "color_noise_std": 0.15,
    },
}


def apply_preset_defaults(args: argparse.Namespace) -> None:
    preset_values = PRESETS[args.preset]
    for key, value in preset_values.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a minimal ASCII PLY: position + normals + RGB. No 2DGS attributes."
    )

    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)

    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESETS.keys()),
        default="teapot",
    )

    parser.add_argument("--min-x", type=float)
    parser.add_argument("--max-x", type=float)
    parser.add_argument("--min-y", type=float)
    parser.add_argument("--max-y", type=float)
    parser.add_argument("--min-z", type=float)
    parser.add_argument("--max-z", type=float)

    parser.add_argument("--position-noise-std", type=float)
    parser.add_argument("--normal-noise-deg", type=float)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--base-r", type=float)
    parser.add_argument("--base-g", type=float)
    parser.add_argument("--base-b", type=float)
    parser.add_argument("--color-noise-std", type=float)

    parser.add_argument(
        "--normal-mode",
        type=str,
        choices=["fixed", "random", "outward"],
        default="outward",
    )

    args = parser.parse_args()
    apply_preset_defaults(args)
    return args


def main() -> None:
    args = parse_args()

    generate_points_ply_minimal(
        output_path=args.out,
        min_x=args.min_x,
        max_x=args.max_x,
        min_y=args.min_y,
        max_y=args.max_y,
        min_z=args.min_z,
        max_z=args.max_z,
        point_count=args.count,
        position_noise_std=args.position_noise_std,
        normal_noise_deg=args.normal_noise_deg,
        seed=args.seed,
        base_rgb=(args.base_r, args.base_g, args.base_b),
        color_noise_std=args.color_noise_std,
        normal_mode=args.normal_mode,
    )


if __name__ == "__main__":
    main()
