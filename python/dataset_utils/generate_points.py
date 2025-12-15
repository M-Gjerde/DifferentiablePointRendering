#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


def compute_square_grid_side(target_primitive_count: int) -> int:
    """
    Compute side length for a square grid that approximates target_primitive_count.
    We round sqrt(N) to get the nearest square.
    """
    target_primitive_count = max(1, target_primitive_count)
    side_float = math.sqrt(target_primitive_count)
    side = max(1, int(round(side_float)))
    return side


def generate_ply(
    output_path: Path,
    target_primitive_count_per_slice: int,
    slice_count: int,
    slice_vertical_offset: float,
    scale_value: float,
    in_plane_noise_std: float,
    seed: int | None = None,
) -> None:
    """
    Generate a PLY file with Gaussian surfels arranged in a square grid per slice.

    - Plane extents in XY are [-2, 2] × [-2, 2].
    - Number of primitives per slice is approximated to a square grid:
        side = round(sqrt(target_primitive_count_per_slice))
        actual_per_slice = side * side
    - Tangents:
        tu = (1, 0, 0)
        tv = (0, 1, 0)
      → normal = tu × tv = (0, 0, 1) (z-up)
    - Colors are randomized in [0, 1].
    - Additional slices are stacked in +Z, with vertical spacing slice_vertical_offset,
      and XY jitter (Gaussian noise) for slices above the base.
    """
    if seed is not None:
        random.seed(seed)

    # Decide square grid side and actual primitive count per slice
    grid_side = compute_square_grid_side(target_primitive_count_per_slice)
    grid_width = grid_side
    grid_height = grid_side
    actual_primitive_count_per_slice = grid_width * grid_height

    # Extents: [-2, 2] in both axes
    min_coord = -0.45
    max_coord = 0.45

    if grid_width > 1:
        grid_spacing_x = (max_coord - min_coord) / float(grid_width - 1)
    else:
        grid_spacing_x = 0.0

    if grid_height > 1:
        grid_spacing_y = (max_coord - min_coord) / float(grid_height - 1)
    else:
        grid_spacing_y = 0.0

    # Tangents (z-up normal)
    tu_x, tu_y, tu_z = 1.0, 0.0, 0.0
    tv_x, tv_y, tv_z = 0.0, 1.0, 0.0

    su = scale_value
    sv = scale_value

    default_opacity = 0.8
    default_beta = -0.0
    default_shape = 0.0

    vertex_count = actual_primitive_count_per_slice * slice_count

    lines: list[str] = []
    lines.append("ply")
    lines.append("format ascii 1.0")
    lines.append("comment 2D Gaussian splats: pk, tu, tv, scales, diffuse albedo, opacity")
    lines.append(f"element vertex {vertex_count}")
    lines.append("property float x          # pk.x")
    lines.append("property float y          # pk.y")
    lines.append("property float z          # pk.z")
    lines.append("property float tu_x       # tangential axis u (unit)")
    lines.append("property float tu_y")
    lines.append("property float tu_z")
    lines.append("property float tv_x       # tangential axis v (unit, orthonormal to tu)")
    lines.append("property float tv_y")
    lines.append("property float tv_z")
    lines.append("property float su         # scale along tu")
    lines.append("property float sv         # scale along tv")
    lines.append("property float albedo_r   # diffuse BRDF albedo")
    lines.append("property float albedo_g")
    lines.append("property float albedo_b")
    lines.append("property float opacity")
    lines.append("property float beta")
    lines.append("property float shape")
    lines.append("end_header")

    for slice_index in range(slice_count):
        z = slice_index * slice_vertical_offset

        for j in range(grid_height):
            # y in [-2, 2]
            if grid_height > 1:
                base_y = min_coord + j * grid_spacing_y
            else:
                base_y = 0.0

            for i in range(grid_width):
                # x in [-2, 2]
                if grid_width > 1:
                    base_x = min_coord + i * grid_spacing_x
                else:
                    base_x = 0.0

                if slice_index == 0:
                    jitter_x = 0.0
                    jitter_y = 0.0
                else:
                    jitter_x = random.gauss(0.0, in_plane_noise_std)
                    jitter_y = random.gauss(0.0, in_plane_noise_std)

                x = base_x + jitter_x
                y = base_y + jitter_y

                albedo_r = random.random()
                albedo_g = random.random()
                albedo_b = random.random()

                lines.append(
                    f"{x:.6f} {y:.6f} {z:.6f} "
                    f"{tu_x:.6f} {tu_y:.6f} {tu_z:.6f} "
                    f"{tv_x:.6f} {tv_y:.6f} {tv_z:.6f} "
                    f"{su:.6f} {sv:.6f} "
                    f"{albedo_r:.6f} {albedo_g:.6f} {albedo_b:.6f} "
                    f"{default_opacity:.6f} {default_beta:.6f} {default_shape:.6f}"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"Written {vertex_count} vertices "
        f"({actual_primitive_count_per_slice} per slice, {slice_count} slices) "
        f"to {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PLY file with Gaussian surfels in a square grid per slice.\n"
            "The base plane is in [-2, 2] x [-2, 2]. "
            "Number of primitives per slice is rounded to a square."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .ply file path.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help=(
            "Desired number of primitives per slice. "
            "Will be rounded to side^2 for a square grid."
        ),
    )
    parser.add_argument(
        "--slice-count",
        type=int,
        default=1,
        help="Number of stacked slices in Z.",
    )
    parser.add_argument(
        "--slice-vertical-offset",
        type=float,
        default=0.06,
        help="Vertical spacing between consecutive slices in world units.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.012,
        help="Default scale value for su and sv.",
    )
    parser.add_argument(
        "--in-plane-noise-std",
        type=float,
        default=0.01,
        help="Standard deviation of XY jitter for slices above the base.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible colors and jitter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_ply(
        output_path=args.out,
        target_primitive_count_per_slice=args.count,
        slice_count=args.slice_count,
        slice_vertical_offset=args.slice_vertical_offset,
        scale_value=args.scale,
        in_plane_noise_std=args.in_plane_noise_std,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
