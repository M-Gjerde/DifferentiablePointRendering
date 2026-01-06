#!/usr/bin/env python3
import argparse
import math
from pathlib import Path


def parse_ply_header(lines):
    """
    Parse a simple ASCII PLY header and return:
    - header_end_index (index of the line after 'end_header')
    - vertex_count (int)
    """
    vertex_count = None
    header_end_index = None

    for line_index, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith("element vertex"):
            parts = stripped_line.split()
            if len(parts) >= 3:
                vertex_count = int(parts[2])
        if stripped_line == "end_header":
            header_end_index = line_index + 1
            break

    if vertex_count is None:
        raise RuntimeError("Could not find 'element vertex <N>' in PLY header.")
    if header_end_index is None:
        raise RuntimeError("Could not find 'end_header' in PLY file.")

    return header_end_index, vertex_count


def read_colmap_ply(input_path: Path):
    """
    Read a COLMAP-style ASCII PLY with:
    x y z nx ny nz r g b
    Returns a list of dicts.
    """
    lines = input_path.read_text().splitlines()
    header_end_index, vertex_count = parse_ply_header(lines)

    if len(lines) < header_end_index + vertex_count:
        raise RuntimeError(
            f"File ended early: expected {vertex_count} vertices "
            f"but only {len(lines) - header_end_index} lines after header."
        )

    vertices = []
    for vertex_index in range(vertex_count):
        line = lines[header_end_index + vertex_index].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 9:
            raise RuntimeError(
                f"Vertex line {vertex_index} does not have 9 components: '{line}'"
            )

        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
        nx = float(parts[3])
        ny = float(parts[4])
        nz = float(parts[5])
        red = int(parts[6])
        green = int(parts[7])
        blue = int(parts[8])

        vertices.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "r": red,
                "g": green,
                "b": blue,
            }
        )

    return vertices


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


import numpy as np


def write_gaussian_ply(
    vertices,
    output_path,
    args,
    opacity_default=1.0,
    beta_default=-5.0,
    shape_default=0.0,

    # -----------------------------------------------------------
    # Noise parameters (set these yourself)
    # -----------------------------------------------------------
    noise_sigma_translation=0.00,
    noise_sigma_rotation=0.00,
    noise_sigma_albedo=0.0,
    noise_sigma_opacity=0.0,
    noise_sigma_beta=0.0,
    noise_sigma_shape=0.0,
):
    """
    Write 2D Gaussian splats to PLY with optional Gaussian noise added to:
        - translation (x,y,z)
        - rotation / tangent frame (tu,tv)   [renormalized + orthogonalized]
        - albedo (r,g,b)  in [0,1]
        - opacity
        - beta
        - shape
    Scale (su,sv) is NOT perturbed.
    """

    # ------------------------------
    # Small utility functions
    # ------------------------------
    def add_noise(value, sigma):
        if sigma == 0.0:
            return value
        return value + np.random.normal(0.0, sigma)

    def renormalize(v):
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm

    def orthonormalize(tu, tv, normal):
        tu = renormalize(tu)
        tv = tv - np.dot(tv, tu) * tu
        tv = renormalize(tv)
        # Ensure right-handed frame
        if np.dot(np.cross(tu, tv), normal) < 0:
            tv = -tv
        return tu, tv

    # ------------------------------
    # Build header
    # ------------------------------
    vertex_count = len(vertices)

    lines = [
        "ply",
        "format ascii 1.0",
        "comment 2D Gaussian splats: pk, tu, tv, scales, diffuse albedo, opacity",
        f"element vertex {vertex_count}",
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

    # ------------------------------
    # Vertex loop
    # ------------------------------
    for vertex in vertices:
        # Base position
        x = float(vertex["x"])
        y = float(vertex["y"])
        z = float(vertex["z"])

        nx = float(vertex["nx"])
        ny = float(vertex["ny"])
        nz = float(vertex["nz"])

        # Compute tangent basis from normal
        tu_x, tu_y, tu_z, tv_x, tv_y, tv_z = compute_tangent_basis_from_normal(nx, ny, nz)
        tu = np.array([tu_x, tu_y, tu_z], dtype=float)
        tv = np.array([tv_x, tv_y, tv_z], dtype=float)
        normal = np.array([nx, ny, nz], dtype=float)

        # Scales (unchanged)
        su = float(args.scale)
        sv = float(args.scale)

        # Albedo 0..1
        albedo = np.array([
            vertex["r"] / 255.0,
            vertex["g"] / 255.0,
            vertex["b"] / 255.0
        ], dtype=float)

        opacity = float(opacity_default)
        beta = float(beta_default)
        shape = float(shape_default)

        # -------------------------------------------------------
        # Noise injection
        # -------------------------------------------------------

        # Translation
        if noise_sigma_translation > 0.0:
            x = add_noise(x, noise_sigma_translation)
            y = add_noise(y, noise_sigma_translation)
            z = add_noise(z, noise_sigma_translation)

        # Rotation
        if noise_sigma_rotation > 0.0:
            tu = tu + np.random.normal(0.0, noise_sigma_rotation, size=3)
            tv = tv + np.random.normal(0.0, noise_sigma_rotation, size=3)
            tu, tv = orthonormalize(tu, tv, normal)

        # Albedo
        if noise_sigma_albedo > 0.0:
            albedo = albedo + np.random.normal(0.0, noise_sigma_albedo, size=3)
            albedo = np.clip(albedo, 0.0, 1.0)

        # Opacity
        if noise_sigma_opacity > 0.0:
            opacity = add_noise(opacity, noise_sigma_opacity)
            opacity = float(np.clip(opacity, 0.0, 1.0))

        # Beta
        if noise_sigma_beta > 0.0:
            beta = add_noise(beta, noise_sigma_beta)

        # Shape
        if noise_sigma_shape > 0.0:
            shape = add_noise(shape, noise_sigma_shape)

        # ------------------------------
        # Emit line
        # ------------------------------
        line = (
            f"{x:.7f} {y:.7f} {z:.7f} "
            f"{tu[0]:.7f} {tu[1]:.7f} {tu[2]:.7f} "
            f"{tv[0]:.7f} {tv[1]:.7f} {tv[2]:.7f} "
            f"{su:.7f} {sv:.7f} "
            f"{albedo[0]:.7f} {albedo[1]:.7f} {albedo[2]:.7f} "
            f"{opacity:.7f} {beta:.7f} {shape:.7f}"
        )
        lines.append(line)

    # Save file
    output_path.write_text("\n".join(lines) + "\n")



def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP PLY (xyz,n,uchar rgb) to 2D Gaussian splat PLY with tangents from normals."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input COLMAP-style ASCII PLY file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Gaussian splat PLY file.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=False,
        default=0.01,
        help="Default scale for su and sv parameters",
    )
    args = parser.parse_args()

    vertices = read_colmap_ply(args.input)
    write_gaussian_ply(vertices, Path(args.output), args)


if __name__ == "__main__":
    main()
