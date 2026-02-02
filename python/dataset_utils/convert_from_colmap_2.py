#!/usr/bin/env python3
"""
Convert a COLMAP/NeRF-style transforms.json (Blender export) into a scene.xml.

This XML contains one <sensor> per frame with:
- Pin-hole intrinsics: fx, fy, cx, cy  (pixels)
- Film width/height
- to_world as <lookat> derived from camera-to-world (c2w)
- near/far/focus_distance
- Also includes fov_axis/fov (legacy) computed from intrinsics

Assumptions:
- frame["transform_matrix"] is Blender cam.matrix_world (camera-to-world, c2w)
- transform_matrix is row-major flat 16-list or nested 4x4
- intrinsics are provided as: fl_x, fl_y, cx, cy
- global resolution is data["w"], data["h"]
"""

import argparse
import json
import math
import xml.etree.ElementTree as et
from pathlib import Path
from typing import Any, Dict, List, Tuple


def compute_fov_deg_from_focal(focal_length_pixels: float, sensor_extent_pixels: int) -> float:
    """
    fov = 2 * atan((extent/2) / f)
    extent is width or height in pixels depending on axis.
    """
    if focal_length_pixels <= 0.0:
        raise ValueError(f"focal_length_pixels must be > 0, got {focal_length_pixels}")
    return 2.0 * math.degrees(math.atan(0.5 * float(sensor_extent_pixels) / float(focal_length_pixels)))


def flatten_4x4_matrix_row_major(transform_matrix: Any) -> List[float]:
    """
    Accept either:
      - flat list length 16 (row-major)
      - nested list 4x4
    Return: flat list length 16 (row-major)
    """
    if isinstance(transform_matrix, list) and len(transform_matrix) == 16 and not isinstance(transform_matrix[0], list):
        return [float(v) for v in transform_matrix]

    if isinstance(transform_matrix, list) and len(transform_matrix) == 4 and isinstance(transform_matrix[0], list):
        flat: List[float] = []
        for row in transform_matrix:
            if len(row) != 4:
                raise ValueError("Nested transform_matrix must be 4x4.")
            flat.extend([float(v) for v in row])
        if len(flat) != 16:
            raise ValueError("Flattening nested 4x4 matrix did not produce 16 values.")
        return flat

    raise ValueError("transform_matrix must be a flat 16-list or a nested 4x4 list.")


def matrix_list_to_rows_row_major(m16: List[float]) -> List[List[float]]:
    if len(m16) != 16:
        raise ValueError(f"transform_matrix must have 16 elements, got {len(m16)}")
    return [
        [m16[0],  m16[1],  m16[2],  m16[3]],
        [m16[4],  m16[5],  m16[6],  m16[7]],
        [m16[8],  m16[9],  m16[10], m16[11]],
        [m16[12], m16[13], m16[14], m16[15]],
    ]


def normalize_3(v: List[float]) -> List[float]:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < 1e-12:
        return [0.0, 0.0, 1.0]
    return [v[0] / length, v[1] / length, v[2] / length]


def extract_lookat_from_c2w_blender(c2w_row_major_4x4: List[List[float]]) -> Tuple[List[float], List[float], List[float]]:
    """
    Blender camera convention:
      - forward is -Z in camera local space
      - up is +Y in camera local space
    c2w transforms camera-space vectors to world space.

    Returns origin, target, up for <lookat>.
    """
    origin = [c2w_row_major_4x4[0][3], c2w_row_major_4x4[1][3], c2w_row_major_4x4[2][3]]

    def transform_dir(dx: float, dy: float, dz: float) -> List[float]:
        return [
            c2w_row_major_4x4[0][0] * dx + c2w_row_major_4x4[0][1] * dy + c2w_row_major_4x4[0][2] * dz,
            c2w_row_major_4x4[1][0] * dx + c2w_row_major_4x4[1][1] * dy + c2w_row_major_4x4[1][2] * dz,
            c2w_row_major_4x4[2][0] * dx + c2w_row_major_4x4[2][1] * dy + c2w_row_major_4x4[2][2] * dz,
        ]

    forward_world = normalize_3(transform_dir(0.0, 0.0, -1.0))
    up_world = normalize_3(transform_dir(0.0, 1.0, 0.0))

    target = [origin[0] + forward_world[0], origin[1] + forward_world[1], origin[2] + forward_world[2]]
    return origin, target, up_world


def format_vec3(v: List[float]) -> str:
    return f"{v[0]},{v[1]},{v[2]}"


def add_sensor_element(
    parent: et.Element,
    camera_id: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    transform_matrix_flat_row_major: List[float],
    near_clip: float,
    far_clip: float,
    focus_distance: float,
    fov_axis: str,
) -> None:
    """
    Always emits pinhole intrinsics (fx,fy,cx,cy).
    Also emits (fov_axis,fov) computed from intrinsics on the chosen axis.
    """

    if fov_axis == "x":
        fov_deg = compute_fov_deg_from_focal(fx, width)
    elif fov_axis == "y":
        fov_deg = compute_fov_deg_from_focal(fy, height)
    else:
        raise ValueError("fov_axis must be 'x' or 'y'")

    c2w_rows = matrix_list_to_rows_row_major(transform_matrix_flat_row_major)
    origin, target, up = extract_lookat_from_c2w_blender(c2w_rows)

    sensor = et.SubElement(parent, "sensor", {"type": "perspective", "id": camera_id})

    # Mark this as intrinsics-driven (your renderer can key off this).
    et.SubElement(sensor, "string", {"name": "camera_model", "value": "pinhole_intrinsics"})

    # Core clipping / focus
    et.SubElement(sensor, "float", {"name": "near_clip", "value": f"{near_clip}"})
    et.SubElement(sensor, "float", {"name": "far_clip", "value": f"{far_clip}"})
    et.SubElement(sensor, "float", {"name": "focus_distance", "value": f"{focus_distance}"})

    # Legacy FOV fields (still useful for tools / back-compat)
    et.SubElement(sensor, "string", {"name": "fov_axis", "value": fov_axis})
    et.SubElement(sensor, "float", {"name": "fov", "value": f"{fov_deg}"})

    # Pinhole parameters (the actual goal)
    et.SubElement(sensor, "float", {"name": "fx", "value": f"{fx}"})
    et.SubElement(sensor, "float", {"name": "fy", "value": f"{fy}"})
    et.SubElement(sensor, "float", {"name": "cx", "value": f"{cx}"})
    et.SubElement(sensor, "float", {"name": "cy", "value": f"{cy}"})

    # Extrinsics
    transform_elem = et.SubElement(sensor, "transform", {"name": "to_world"})
    et.SubElement(
        transform_elem,
        "lookat",
        {
            "origin": format_vec3(origin),
            "target": format_vec3(target),
            "up": format_vec3(up),
        },
    )

    # Film
    film = et.SubElement(sensor, "film", {"type": "hdrfilm"})
    et.SubElement(film, "integer", {"name": "width", "value": str(width)})
    et.SubElement(film, "integer", {"name": "height", "value": str(height)})
    et.SubElement(film, "rfilter", {"type": "tent"})
    et.SubElement(film, "string", {"name": "pixel_format", "value": "rgb"})


def build_scene_xml_from_transforms(
    transforms_path: Path,
    near_clip: float,
    far_clip: float,
    focus_distance: float,
    fov_axis: str,
) -> et.Element:
    with transforms_path.open("r") as f:
        data: Dict[str, Any] = json.load(f)

    width = int(data["w"])
    height = int(data["h"])
    frames = data["frames"]

    scene = et.Element("scene", {"version": "3.0.0"})

    # Keep an integrator node if your loader expects/ignores it
    integrator = et.SubElement(scene, "integrator", {"type": "path"})
    et.SubElement(integrator, "integer", {"name": "max_depth", "value": "6"})

    default_fx = float(data.get("fl_x", 0.0))
    default_fy = float(data.get("fl_y", 0.0))
    default_cx = float(data.get("cx", width * 0.5))
    default_cy = float(data.get("cy", height * 0.5))

    for frame_index, frame in enumerate(frames):
        camera_name = frame.get("camera_name", f"camera_{frame_index:03d}")

        fx = float(frame.get("fl_x", default_fx))
        fy = float(frame.get("fl_y", default_fy))
        cx = float(frame.get("cx", default_cx))
        cy = float(frame.get("cy", default_cy))

        if fx <= 0.0 or fy <= 0.0:
            raise ValueError(f"Missing/invalid intrinsics for frame '{camera_name}': fx={fx}, fy={fy}")

        transform_matrix_flat = flatten_4x4_matrix_row_major(frame["transform_matrix"])

        add_sensor_element(
            parent=scene,
            camera_id=camera_name,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            transform_matrix_flat_row_major=transform_matrix_flat,
            near_clip=near_clip,
            far_clip=far_clip,
            focus_distance=focus_distance,
            fov_axis=fov_axis,
        )

    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert transforms.json to scene.xml with pinhole intrinsics.")
    parser.add_argument("--transforms", type=Path, required=True, help="Path to transforms.json")
    parser.add_argument("--output", type=Path, required=True, help="Output scene.xml path")
    parser.add_argument("--near", type=float, default=0.01, help="Near clip (default: 0.01)")
    parser.add_argument("--far", type=float, default=100.0, help="Far clip (default: 100.0)")
    parser.add_argument("--focus", type=float, default=1000.0, help="Focus distance (default: 1000.0)")
    parser.add_argument(
        "--fov_axis",
        type=str,
        default="x",
        choices=["x", "y"],
        help="Which axis to use for the legacy fov field (default: x).",
    )
    args = parser.parse_args()

    scene = build_scene_xml_from_transforms(
        transforms_path=args.transforms,
        near_clip=args.near,
        far_clip=args.far,
        focus_distance=args.focus,
        fov_axis=args.fov_axis,
    )

    try:
        et.indent(scene, space="    ")
    except AttributeError:
        pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    et.ElementTree(scene).write(args.output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
