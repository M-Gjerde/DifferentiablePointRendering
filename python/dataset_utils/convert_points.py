#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def parse_ply_header(lines: list[str]) -> tuple[int, int]:
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


def read_colmap_ply(input_path: Path) -> list[dict[str, float | int]]:
    """
    Read a COLMAP-style ASCII PLY containing:

        x y z nx ny nz r g b
    """
    lines = input_path.read_text().splitlines()
    header_end_index, vertex_count = parse_ply_header(lines)

    if len(lines) < header_end_index + vertex_count:
        raise RuntimeError(
            f"File ended early: expected {vertex_count} vertices but only "
            f"{len(lines) - header_end_index} lines after header."
        )

    vertices: list[dict[str, float | int]] = []

    for vertex_index in range(vertex_count):
        line = lines[header_end_index + vertex_index].strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 9:
            raise RuntimeError(
                f"Vertex line {vertex_index} does not have 9 components: '{line}'"
            )

        vertices.append(
            {
                "x": float(parts[0]),
                "y": float(parts[1]),
                "z": float(parts[2]),
                "nx": float(parts[3]),
                "ny": float(parts[4]),
                "nz": float(parts[5]),
                "r": int(parts[6]),
                "g": int(parts[7]),
                "b": int(parts[8]),
            }
        )

    return vertices


def normalize_vector(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    length = float(np.linalg.norm(vector))

    if not np.isfinite(length) or length < 1.0e-12:
        if fallback is None:
            raise RuntimeError("Cannot normalize a degenerate vector.")

        return fallback.copy()

    return vector / length


def compute_orthonormal_frame_from_normal(
    nx: float,
    ny: float,
    nz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a right-handed local frame:

        local +X -> tangent_u
        local +Y -> tangent_v
        local +Z -> normal

    Therefore:
        cross(tangent_u, tangent_v) == normal
    """
    normal = normalize_vector(
        np.array([nx, ny, nz], dtype=np.float64),
        fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )

    helper = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(normal[2]) < 0.999
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )

    tangent_u = normalize_vector(np.cross(normal, helper))
    tangent_v = normalize_vector(np.cross(normal, tangent_u))

    return tangent_u, tangent_v, normal


def quaternion_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to quaternion [w, x, y, z].

    The matrix columns are the world-space directions of local X, Y, Z.
    """
    m00 = float(rotation_matrix[0, 0])
    m01 = float(rotation_matrix[0, 1])
    m02 = float(rotation_matrix[0, 2])
    m10 = float(rotation_matrix[1, 0])
    m11 = float(rotation_matrix[1, 1])
    m12 = float(rotation_matrix[1, 2])
    m20 = float(rotation_matrix[2, 0])
    m21 = float(rotation_matrix[2, 1])
    m22 = float(rotation_matrix[2, 2])

    trace = m00 + m11 + m22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(max(0.0, 1.0 + m00 - m11 - m22)) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(max(0.0, 1.0 + m11 - m00 - m22)) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(max(0.0, 1.0 + m22 - m00 - m11)) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale

    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    quaternion = normalize_vector(
        quaternion,
        fallback=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )

    # Match PLYPointLoader canonicalization.
    if quaternion[0] < 0.0:
        quaternion *= -1.0

    return quaternion


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs

    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quaternion_from_rotation_vector(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))

    if angle < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    axis = rotation_vector / angle
    half_angle = 0.5 * angle
    sin_half_angle = math.sin(half_angle)

    return np.array(
        [
            math.cos(half_angle),
            axis[0] * sin_half_angle,
            axis[1] * sin_half_angle,
            axis[2] * sin_half_angle,
        ],
        dtype=np.float64,
    )


def quaternion_from_normal(nx: float, ny: float, nz: float) -> np.ndarray:
    tangent_u, tangent_v, normal = compute_orthonormal_frame_from_normal(nx, ny, nz)

    rotation_matrix = np.column_stack((tangent_u, tangent_v, normal))
    return quaternion_from_rotation_matrix(rotation_matrix)


def perturb_quaternion(
    quaternion: np.ndarray,
    rotation_noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if rotation_noise_sigma <= 0.0:
        return quaternion

    rotation_vector = rng.normal(0.0, rotation_noise_sigma, size=3)
    noise_quaternion = quaternion_from_rotation_vector(rotation_vector)

    perturbed_quaternion = quaternion_multiply(noise_quaternion, quaternion)
    perturbed_quaternion = normalize_vector(
        perturbed_quaternion,
        fallback=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )

    if perturbed_quaternion[0] < 0.0:
        perturbed_quaternion *= -1.0

    return perturbed_quaternion


def write_quaternion_surfel_ply(
    vertices: list[dict[str, float | int]],
    output_path: Path,
    args: argparse.Namespace,
    opacity_default: float = 1.0,
    beta_default: float = 0.0,
    shape_default: float = 0.0,
    noise_sigma_translation: float = 0.0,
    noise_sigma_rotation: float = 0.0,
    noise_sigma_albedo: float = 0.0,
    noise_sigma_opacity: float = 0.0,
    noise_sigma_beta: float = 0.0,
    noise_sigma_shape: float = 0.0,
) -> None:
    """
    Write quaternion surfels in the schema required by PLYPointLoader:

        x y z
        rot_w rot_x rot_y rot_z
        su sv
        albedo_r albedo_g albedo_b
        opacity beta shape power
    """
    rng = np.random.default_rng(args.seed)

    light_definitions = [
        {
            "position": (3.0, -0.8, 2.2),
            "power": 300.0,
        },
        {
            "position": (-3.0, 0.8, 2.2),
            "power": 200.0,
        },
        {
            "position": (0.0, 0.8, 3.0),
            "power": 200.0,
        },
    ]

    total_vertex_count = len(vertices) + len(light_definitions)

    lines = [
        "ply",
        "format ascii 1.0",
        "comment Quaternion surfels: local X=tangent_u, local Y=tangent_v, local Z=normal",
        f"element vertex {total_vertex_count}",
        "property float x",
        "property float y",
        "property float z",
        "property float rot_w",
        "property float rot_x",
        "property float rot_y",
        "property float rot_z",
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

    for vertex in vertices:
        position = np.array(
            [
                float(vertex["x"]),
                float(vertex["y"]),
                float(vertex["z"]),
            ],
            dtype=np.float64,
        )

        normal = np.array(
            [
                float(vertex["nx"]),
                float(vertex["ny"]),
                float(vertex["nz"]),
            ],
            dtype=np.float64,
        )

        quaternion = quaternion_from_normal(normal[0], normal[1], normal[2])

        if noise_sigma_translation > 0.0:
            position += rng.normal(0.0, noise_sigma_translation, size=3)

        quaternion = perturb_quaternion(
            quaternion,
            noise_sigma_rotation,
            rng,
        )

        albedo = np.array([0.30, 0.30, 0.30], dtype=np.float64)

        if noise_sigma_albedo > 0.0:
            albedo += rng.normal(0.0, noise_sigma_albedo, size=3)
            albedo = np.clip(albedo, 0.0, 1.0)

        opacity = float(opacity_default)
        beta = float(beta_default)
        shape = float(shape_default)

        if noise_sigma_opacity > 0.0:
            opacity += float(rng.normal(0.0, noise_sigma_opacity))
            opacity = float(np.clip(opacity, 0.0, 1.0))

        if noise_sigma_beta > 0.0:
            beta += float(rng.normal(0.0, noise_sigma_beta))

        if noise_sigma_shape > 0.0:
            shape += float(rng.normal(0.0, noise_sigma_shape))

        lines.append(
            f"{position[0]:.7f} {position[1]:.7f} {position[2]:.7f} "
            f"{quaternion[0]:.7f} {quaternion[1]:.7f} "
            f"{quaternion[2]:.7f} {quaternion[3]:.7f} "
            f"{args.scale:.7f} {args.scale:.7f} "
            f"{albedo[0]:.7f} {albedo[1]:.7f} {albedo[2]:.7f} "
            f"{opacity:.7f} {beta:.7f} {shape:.7f} 0.0000000"
        )

    light_quaternion = quaternion_from_normal(0.0, 0.0, -1.0)

    for light in light_definitions:
        light_x, light_y, light_z = light["position"]

        lines.append(
            f"{light_x:.7f} {light_y:.7f} {light_z:.7f} "
            f"{light_quaternion[0]:.7f} {light_quaternion[1]:.7f} "
            f"{light_quaternion[2]:.7f} {light_quaternion[3]:.7f} "
            f"0.0010000 0.0010000 "
            f"1.0000000 1.0000000 1.0000000 "
            f"1.0000000 -100.0000000 0.0000000 {light['power']:.7f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert COLMAP XYZ-normal-RGB ASCII PLY to quaternion surfel PLY."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input ASCII COLMAP PLY file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output quaternion surfel PLY file.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.015,
        help="Initial surfel scale for both su and sv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization noise.",
    )
    args = parser.parse_args()

    vertices = read_colmap_ply(args.input)
    write_quaternion_surfel_ply(vertices, args.output, args)


if __name__ == "__main__":
    main()