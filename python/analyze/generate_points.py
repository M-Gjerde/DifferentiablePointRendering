#!/usr/bin/env python3
"""
Generate a PLY file with 2D Gaussian splats laid out on a uniform 3D grid.

- num_vertices points in total (default: 100)
- Positions are sampled uniformly on a regular grid within [min, max] per axis
- All other attributes use fixed default values

Example:
    python generate_gaussians_ply.py \
        --output initial.ply \
        --x-min -0.6 --x-max 0.6 \
        --y-min -0.6 --y-max 0.6 \
        --z-min  0.3 --z-max 1.3 \
        --num-vertices 100
"""

import argparse
from pathlib import Path
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2D Gaussian splat PLY file with a uniform grid.")

    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        default=Path("./initial.ply"),
        help="Path to output PLY file.",
    )
    parser.add_argument(
        "--num-vertices",
        type=int,
        default=50,
        help="Number of vertices to generate (default: 100).",
    )

    # Position bounds
    parser.add_argument("--x-min", type=float, default=-0.4)
    parser.add_argument("--x-max", type=float, default=0.4)
    parser.add_argument("--y-min", type=float, default=-0.4)
    parser.add_argument("--y-max", type=float, default=0.4)
    parser.add_argument("--z-min", type=float, default=0.6)
    parser.add_argument("--z-max", type=float, default=1.0)

    return parser.parse_args()


def make_grid_positions(
    num_vertices: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """
    Create a regular 3D grid of positions within the given bounds.

    Returns an array of shape (num_vertices, 3).
    """
    # Try to make the grid as "cube-like" as possible
    approx = int(round(num_vertices ** (1.0 / 3.0)))
    approx = max(1, approx)

    grid_x_count = approx
    grid_y_count = approx
    grid_z_count = int(np.ceil(num_vertices / (grid_x_count * grid_y_count)))

    xs = np.linspace(x_min, x_max, grid_x_count, dtype=np.float32)
    ys = np.linspace(y_min, y_max, grid_y_count, dtype=np.float32)
    zs = np.linspace(z_min, z_max, grid_z_count, dtype=np.float32)

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="xy")
    positions = np.stack(
        [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()],
        axis=-1,
    )

    # Keep only the requested number of vertices
    positions = positions[:num_vertices, :]
    assert positions.shape == (num_vertices, 3)
    return positions


def generate_vertices(args: argparse.Namespace) -> np.ndarray:
    """
    Generate an array of shape (N, 17) with the vertex attributes in correct order:
    x, y, z,
    tu_x, tu_y, tu_z,
    tv_x, tv_y, tv_z,
    su, sv,
    albedo_r, albedo_g, albedo_b,
    opacity,
    beta,
    shape
    """
    num_vertices = args.num_vertices

    positions = make_grid_positions(
        num_vertices=num_vertices,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
    )

    # Default tangent axes (orthonormal)
    tu = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tv = np.array([0.0, 1.0, -0.3], dtype=np.float32)

    tu_array = np.tile(tu[None, :], (num_vertices, 1))
    tv_array = np.tile(tv[None, :], (num_vertices, 1))

    # Default scales, albedo, opacity, beta, shape
    su = np.full((num_vertices, 1), 0.08, dtype=np.float32)
    sv = np.full((num_vertices, 1), 0.08, dtype=np.float32)

    albedo = np.full((num_vertices, 3), 0.9, dtype=np.float32)
    opacity = np.full((num_vertices, 1), 1.0, dtype=np.float32)
    beta = np.full((num_vertices, 1), 0.0, dtype=np.float32)
    shape = np.full((num_vertices, 1), 0.0, dtype=np.float32)

    vertices = np.concatenate(
        [
            positions,
            tu_array,
            tv_array,
            su,
            sv,
            albedo,
            opacity,
            beta,
            shape,
        ],
        axis=1,
    )

    assert vertices.shape == (num_vertices, 17)
    return vertices


def write_ply(output_path: Path, vertices: np.ndarray) -> None:
    num_vertices = vertices.shape[0]

    header_lines = [
        "ply",
        "format ascii 1.0",
        "comment 2D Gaussian splats: pk, tu, tv, scales, diffuse albedo, opacity",
        f"element vertex {num_vertices}",
        "property float x          # pk.x",
        "property float y          # pk.y",
        "property float z          # pk.z",
        "property float tu_x       # tangential axis u (unit)",
        "property float tu_y",
        "property float tu_z",
        "property float tv_x       # tangential axis v (unit, orthonormal to tu)",
        "property float tv_y",
        "property float tv_z",
        "property float su         # scale along tu",
        "property float sv         # scale along tv",
        "property float albedo_r   # diffuse BRDF albedo",
        "property float albedo_g",
        "property float albedo_b",
        "property float opacity",
        "property float beta",
        "property float shape",
        "end_header",
    ]

    with output_path.open("w", encoding="utf-8") as file:
        for line in header_lines:
            file.write(line + "\n")

        for vertex in vertices:
            file.write(" ".join(f"{value:.6f}" for value in vertex) + "\n")


def main() -> None:
    args = parse_arguments()
    vertices = generate_vertices(args)
    write_ply(args.output, vertices)


if __name__ == "__main__":
    main()
