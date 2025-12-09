#!/usr/bin/env python3
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


import math

def compute_vertical_fov_deg(fl_y, image_height):
    """Compute vertical FOV (Y) in degrees from fl_y and image height."""
    return 2.0 * math.degrees(math.atan(0.5 * float(image_height) / float(fl_y)))



def matrix_list_to_rows(m16):
    """Convert flat 16-element list (row-major) to 4x4 row-major matrix."""
    if len(m16) != 16:
        raise ValueError(f"transform_matrix must have 16 elements, got {len(m16)}")
    return [
        [m16[0],  m16[1],  m16[2],  m16[3]],
        [m16[4],  m16[5],  m16[6],  m16[7]],
        [m16[8],  m16[9],  m16[10], m16[11]],
        [m16[12], m16[13], m16[14], m16[15]],
    ]


def extract_origin_target_up_from_c2w(c2w):
    """
    Interpret c2w as camera-to-world (OpenGL convention: camera looks along -Z, up is +Y).
    Return (origin, target, up) for Mitsuba <lookat>.
    """
    # Origin is translation part
    origin = [
        c2w[0][3],
        c2w[1][3],
        c2w[2][3],
    ]

    # Transform a direction (no translation)
    def transform_dir(dx, dy, dz):
        return [
            c2w[0][0] * dx + c2w[0][1] * dy + c2w[0][2] * dz,
            c2w[1][0] * dx + c2w[1][1] * dy + c2w[1][2] * dz,
            c2w[2][0] * dx + c2w[2][1] * dy + c2w[2][2] * dz,
        ]

    # Forward: camera looks along -Z in camera space
    forward = transform_dir(0.0, 0.0, -1.0)
    # Up: +Y in camera space
    up = transform_dir(0.0, 1.0, 0.0)

    # Normalize helper
    def normalize(v):
        length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if length < 1e-8:
            return [0.0, 0.0, 1.0]
        return [v[i] / length for i in range(3)]

    forward = normalize(forward)
    up = normalize(up)

    target = [
        origin[0] + forward[0],
        origin[1] + forward[1],
        origin[2] + forward[2],
    ]

    return origin, target, up
def add_sensor_element(
    parent,
    camera_id,
    width,
    height,
    fl_x,
    fl_y,
    cx,
    cy,
    transform_matrix_flat,
    sample_count=128,
    near_clip=0.01,
    far_clip=100.0,
    focus_distance=1000.0,
):
    # --- FOV: use vertical FOV directly, axis = 'y' ---
    fov_y_deg = compute_vertical_fov_deg(fl_y, height)

    c2w = matrix_list_to_rows(transform_matrix_flat)
    origin, target, up = extract_origin_target_up_from_c2w(c2w)  # or the non-zup version

    sensor = ET.SubElement(parent, "sensor", {"type": "perspective", "id": camera_id})

    # Important: axis is 'y', not 'smaller'
    ET.SubElement(sensor, "string", {"name": "fov_axis", "value": "y"})
    ET.SubElement(sensor, "float",  {"name": "near_clip", "value": f"{near_clip}"})
    ET.SubElement(sensor, "float",  {"name": "far_clip", "value": f"{far_clip}"})
    ET.SubElement(sensor, "float",  {"name": "focus_distance", "value": f"{focus_distance}"})
    ET.SubElement(sensor, "float",  {"name": "fov", "value": f"{fov_y_deg}"})

    transform_elem = ET.SubElement(sensor, "transform", {"name": "to_world"})
    lookat_attrib = {
        "origin": f"{origin[0]},{origin[1]},{origin[2]}",
        "target": f"{target[0]},{target[1]},{target[2]}",
        "up":     f"{up[0]},{up[1]},{up[2]}",
    }
    ET.SubElement(transform_elem, "lookat", lookat_attrib)

    sampler = ET.SubElement(sensor, "sampler", {"type": "independent"})
    ET.SubElement(sampler, "integer", {
        "name": "sample_count",
        "value": f"{sample_count}",
    })

    film = ET.SubElement(sensor, "film", {"type": "hdrfilm"})
    ET.SubElement(film, "integer", {"name": "width",  "value": f"{width}"})
    ET.SubElement(film, "integer", {"name": "height", "value": f"{height}"})
    ET.SubElement(film, "rfilter", {"type": "tent"})
    ET.SubElement(film, "string", {"name": "pixel_format",      "value": "rgb"})
    ET.SubElement(film, "string", {"name": "component_format",  "value": "float32"})

def build_scene_from_transforms(
    transforms_path: Path,
    sample_count: int = 128,
    near_clip: float = 0.01,
    far_clip: float = 100.0,
    focus_distance: float = 1000.0,
):
    """
    Build a Mitsuba <scene> Element from a NeRF-style transforms.json file.
    Only cameras are added; geometry/lights should be added separately.
    """
    with transforms_path.open("r") as f:
        data = json.load(f)

    width = int(data["w"])
    height = int(data["h"])
    frames = data["frames"]

    scene = ET.Element("scene", {"version": "3.0.0"})

    # Simple integrator
    integrator = ET.SubElement(scene, "integrator", {"type": "path"})
    ET.SubElement(integrator, "integer", {"name": "max_depth", "value": "6"})

    for idx, frame in enumerate(frames):
        camera_name = frame.get("camera_name", f"camera_{idx:03d}")
        fl_x = float(frame.get("fl_x", data.get("fl_x", 1.0)))
        fl_y = float(frame.get("fl_y", data.get("fl_y", 1.0)))
        cx = float(frame.get("cx", data.get("cx", width / 2.0)))
        cy = float(frame.get("cy", data.get("cy", height / 2.0)))

        transform_matrix_flat = frame["transform_matrix"]
        # Some JSON exporters store matrix as nested list; flatten if needed
        if isinstance(transform_matrix_flat[0], list):
            flat = []
            for row in transform_matrix_flat:
                flat.extend(row)
            transform_matrix_flat = flat

        add_sensor_element(
            scene,
            camera_id=camera_name,
            width=width,
            height=height,
            fl_x=fl_x,
            fl_y=fl_y,
            cx=cx,
            cy=cy,
            transform_matrix_flat=transform_matrix_flat,
            sample_count=sample_count,
            near_clip=near_clip,
            far_clip=far_clip,
            focus_distance=focus_distance,
        )

    return scene


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeRF-style transforms.json to Mitsuba scene XML with cameras."
    )
    parser.add_argument(
        "--transforms",
        type=Path,
        required=True,
        help="Path to transforms.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output XML path",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Sample count per pixel for Mitsuba sampler (default: 128)",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.01,
        help="Near clip distance (default: 0.01)",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=100.0,
        help="Far clip distance (default: 100.0)",
    )
    parser.add_argument(
        "--focus",
        type=float,
        default=1000.0,
        help="Focus distance (default: 1000.0)",
    )

    args = parser.parse_args()

    scene = build_scene_from_transforms(
        transforms_path=args.transforms,
        sample_count=args.samples,
        near_clip=args.near,
        far_clip=args.far,
        focus_distance=args.focus,
    )

    # Pretty-print if available (Python 3.9+)
    try:
        ET.indent(scene, space="    ")
    except AttributeError:
        # For older Python versions, you can omit indentation or implement it manually.
        pass

    tree = ET.ElementTree(scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
