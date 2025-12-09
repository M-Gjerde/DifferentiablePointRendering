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


def write_gaussian_ply(vertices, output_path: Path,
                       su_default=0.1,
                       sv_default=0.1,
                       opacity_default=0.9,
                       beta_default=-0.25,
                       shape_default=0.0):
    """
    Write out vertices in 2D Gaussian splat PLY format,
    with tu/tv computed from normals.
    """
    vertex_count = len(vertices)

    lines = []
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

    for vertex in vertices:
        x = vertex["x"]
        y = vertex["y"]
        z = vertex["z"]

        nx = vertex["nx"]
        ny = vertex["ny"]
        nz = vertex["nz"]

        tu_x, tu_y, tu_z, tv_x, tv_y, tv_z = compute_tangent_basis_from_normal(nx, ny, nz)

        su = su_default
        sv = sv_default

        albedo_r = vertex["r"] / 255.0
        albedo_g = vertex["g"] / 255.0
        albedo_b = vertex["b"] / 255.0

        opacity = opacity_default
        beta = beta_default
        shape = shape_default

        line = (
            f"{x:.7f} {y:.7f} {z:.7f} "
            f"{tu_x:.7f} {tu_y:.7f} {tu_z:.7f} "
            f"{tv_x:.7f} {tv_y:.7f} {tv_z:.7f} "
            f"{su:.7f} {sv:.7f} "
            f"{albedo_r:.7f} {albedo_g:.7f} {albedo_b:.7f} "
            f"{opacity:.7f} {beta:.7f} {shape:.7f}"
        )
        lines.append(line)

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
    args = parser.parse_args()

    vertices = read_colmap_ply(args.input)
    write_gaussian_ply(vertices, args.output)


if __name__ == "__main__":
    main()
