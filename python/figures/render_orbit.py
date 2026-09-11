#!/usr/bin/env python3
"""Render a reproducible 360-degree camera orbit around a trained surfel model.

The generated JSON pose file is intentionally renderer-independent. Pass that
same file to the companion 2DGS ``render_orbit.py`` to compare both methods
from identical world-space camera poses and intrinsics.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image

if __package__:
    from .orbit_maps import depth_display_range, geometry_video_path, save_geometry_frame, write_geometry_metadata
else:
    from orbit_maps import depth_display_range, geometry_video_path, save_geometry_frame, write_geometry_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSE_SCHEMA = "dpr-orbit-poses-v1"


def normalize(vector: np.ndarray, *, epsilon: float = 1.0e-12) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= epsilon:
        raise ValueError(f"Cannot normalize vector {value.tolist()}")
    return value / norm


def compute_non_emissive_bounds(
    positions: np.ndarray,
    powers: np.ndarray,
    *,
    emissive_threshold: float = 0.0,
) -> tuple[np.ndarray, float, int]:
    """Return AABB center and bounding radius after removing emissive surfels."""
    xyz = np.asarray(positions, dtype=np.float64)
    power = np.asarray(powers, dtype=np.float64).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected positions with shape (N, 3), got {xyz.shape}")
    if power.shape[0] != xyz.shape[0]:
        raise ValueError("Positions and powers must contain the same number of surfels")

    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(power)
    non_emissive = valid & (power <= float(emissive_threshold))
    selected = xyz[non_emissive]
    if selected.size == 0:
        raise ValueError(
            "No finite non-emissive surfels remain after applying "
            f"power <= {emissive_threshold:g}"
        )

    center = 0.5 * (selected.min(axis=0) + selected.max(axis=0))
    extent = float(np.linalg.norm(selected - center, axis=1).max())
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError(f"Non-emissive model extent is invalid: {extent}")
    return center, extent, int(selected.shape[0])


def read_surfel_positions_and_powers(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the ASCII PLY fields written by this project without loading Pale."""
    property_names: list[str] = []
    vertex_count: int | None = None
    in_vertex_element = False

    with ply_path.open("r", encoding="ascii") as handle:
        if handle.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {ply_path}")
        format_line = handle.readline().strip()
        if format_line != "format ascii 1.0":
            raise ValueError(
                f"Orbit center extraction expects this project's ASCII PLY format; "
                f"got '{format_line}' in {ply_path}"
            )

        for line in handle:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "element":
                in_vertex_element = len(tokens) >= 3 and tokens[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(tokens[2])
                    property_names = []
            elif tokens[0] == "property" and in_vertex_element:
                if len(tokens) != 3 or tokens[1] == "list":
                    raise ValueError("List-valued vertex properties are unsupported")
                property_names.append(tokens[2])
            elif tokens[0] == "end_header":
                break
        else:
            raise ValueError(f"PLY header has no end_header: {ply_path}")

        if vertex_count is None:
            raise ValueError(f"PLY has no vertex element: {ply_path}")
        required = ("x", "y", "z", "power")
        missing = [name for name in required if name not in property_names]
        if missing:
            raise ValueError(f"PLY is missing fields {missing}: {ply_path}")
        indices = [property_names.index(name) for name in required]
        values = np.loadtxt(handle, dtype=np.float64, max_rows=vertex_count, usecols=indices)

    values = np.atleast_2d(values)
    if values.shape != (vertex_count, 4):
        raise ValueError(
            f"Expected {vertex_count} PLY vertices with x/y/z/power, got {values.shape}"
        )
    return values[:, :3], values[:, 3]


def make_orbit_pose_document(
    *,
    center: np.ndarray,
    model_extent: float,
    non_emissive_surfel_count: int,
    frame_count: int,
    width: int,
    height: int,
    vertical_fov_degrees: float,
    radius: float | None,
    radius_margin: float,
    elevation_degrees: float,
    start_angle_degrees: float,
    clockwise: bool,
    emissive_threshold: float,
    source_points: Path,
) -> dict[str, Any]:
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("Frame count, width, and height must be positive")
    if not 0.0 < vertical_fov_degrees < 179.0:
        raise ValueError("--fov-degrees must be in (0, 179)")
    if radius_margin <= 1.0:
        raise ValueError("--radius-margin must be greater than 1")
    if abs(elevation_degrees) >= 89.0:
        raise ValueError("--elevation-degrees must be between -89 and 89")

    center = np.asarray(center, dtype=np.float64)
    fovy = math.radians(vertical_fov_degrees)
    fovx = 2.0 * math.atan(math.tan(0.5 * fovy) * width / height)
    limiting_half_fov = 0.5 * min(fovx, fovy)
    camera_distance = (
        float(radius)
        if radius is not None
        else radius_margin * float(model_extent) / math.sin(limiting_half_fov)
    )
    if not math.isfinite(camera_distance) or camera_distance <= model_extent:
        raise ValueError(
            f"Orbit radius must be finite and greater than model extent "
            f"({model_extent:.6g}); got {camera_distance}"
        )

    elevation = math.radians(elevation_degrees)
    horizontal_radius = camera_distance * math.cos(elevation)
    height_offset = camera_distance * math.sin(elevation)
    direction = -1.0 if clockwise else 1.0
    frames: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        angle_degrees = start_angle_degrees + direction * 360.0 * frame_index / frame_count
        angle = math.radians(angle_degrees)
        position = center + np.array(
            [horizontal_radius * math.cos(angle),
             horizontal_radius * math.sin(angle),
             height_offset],
            dtype=np.float64,
        )
        frames.append({
            "index": frame_index,
            "angle_degrees": angle_degrees,
            "position": position.tolist(),
            "target": center.tolist(),
            "up": [0.0, 0.0, 1.0],
        })

    focal_y = height / (2.0 * math.tan(0.5 * fovy))
    focal_x = width / (2.0 * math.tan(0.5 * fovx))
    return {
        "schema": POSE_SCHEMA,
        "coordinate_system": "shared_world_space",
        "center_method": "non_emissive_axis_aligned_bounds",
        "center": center.tolist(),
        "model_extent": float(model_extent),
        "non_emissive_surfel_count": int(non_emissive_surfel_count),
        "emissive_power_threshold": float(emissive_threshold),
        "source_points": str(source_points.resolve()),
        "camera_distance": camera_distance,
        "elevation_degrees": float(elevation_degrees),
        "intrinsics": {
            "width": int(width),
            "height": int(height),
            "fx": focal_x,
            "fy": focal_y,
            "cx": width / 2.0,
            "cy": height / 2.0,
            "fov_x_radians": fovx,
            "fov_y_radians": fovy,
            "near_clip": 0.01,
            "far_clip": max(100.0, camera_distance + 4.0 * model_extent),
        },
        "frames": frames,
    }


def validate_pose_document(document: dict[str, Any]) -> None:
    if document.get("schema") != POSE_SCHEMA:
        raise ValueError(f"Expected pose schema '{POSE_SCHEMA}'")
    intrinsics = document.get("intrinsics")
    frames = document.get("frames")
    if not isinstance(intrinsics, dict) or not isinstance(frames, list) or not frames:
        raise ValueError("Pose file requires intrinsics and at least one frame")
    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        if key not in intrinsics or not math.isfinite(float(intrinsics[key])):
            raise ValueError(f"Invalid or missing intrinsic '{key}'")
    first_index = int(frames[0].get("index", -1))
    if first_index < 0:
        raise ValueError("Pose frame indices must be non-negative")
    for frame_offset, frame in enumerate(frames):
        expected_index = first_index + frame_offset
        if int(frame.get("index", -1)) != expected_index:
            raise ValueError("Pose frame indices must be contiguous")
        for key in ("position", "target", "up"):
            value = np.asarray(frame.get(key), dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"Frame {expected_index} has invalid '{key}'")
        normalize(np.asarray(frame["target"]) - np.asarray(frame["position"]))
        normalize(np.asarray(frame["up"]))


def write_orbit_scene(document: dict[str, Any], output_path: Path) -> list[str]:
    validate_pose_document(document)
    intrinsics = document["intrinsics"]
    root = ET.Element("scene", {"version": "3.0.0"})
    camera_names: list[str] = []
    for frame in document["frames"]:
        camera_name = f"OrbitCam_{int(frame['index']):05d}"
        camera_names.append(camera_name)
        sensor = ET.SubElement(root, "sensor", {"type": "perspective", "id": camera_name})
        ET.SubElement(sensor, "string", {"name": "camera_model", "value": "pinhole_intrinsics"})
        for name in ("near_clip", "far_clip", "fx", "fy", "cx", "cy"):
            ET.SubElement(sensor, "float", {"name": name, "value": f"{float(intrinsics[name]):.17g}"})
        ET.SubElement(sensor, "float", {"name": "focus_distance", "value": "1000"})
        transform = ET.SubElement(sensor, "transform", {"name": "to_world"})
        ET.SubElement(transform, "lookat", {
            "origin": ",".join(f"{float(value):.17g}" for value in frame["position"]),
            "target": ",".join(f"{float(value):.17g}" for value in frame["target"]),
            "up": ",".join(f"{float(value):.17g}" for value in frame["up"]),
        })
        film = ET.SubElement(sensor, "film", {"type": "hdrfilm"})
        ET.SubElement(film, "integer", {"name": "width", "value": str(int(intrinsics["width"]))})
        ET.SubElement(film, "integer", {"name": "height", "value": str(int(intrinsics["height"]))})
        ET.SubElement(film, "rfilter", {"type": "tent"})
        ET.SubElement(film, "string", {"name": "pixel_format", "value": "rgb"})

    ET.indent(root, space="    ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=False)
    return camera_names


def rendered_rgb(rendered_images: dict[str, Any], camera_name: str) -> np.ndarray:
    value = rendered_images[camera_name]
    if isinstance(value, dict):
        value = value["image"]
    image = np.asarray(value, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Unexpected render shape for {camera_name}: {image.shape}")
    return np.clip(np.nan_to_num(image[..., :3]), 0.0, 1.0)


def camera_batch_ranges(frame_count: int, batch_size: int) -> list[tuple[int, int]]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if batch_size <= 0:
        raise ValueError("camera batch size must be positive")
    return [
        (start, min(start + batch_size, frame_count))
        for start in range(0, frame_count, batch_size)
    ]


def render_pose_batch(
    *,
    run_config: dict[str, Any],
    points_path: Path,
    document: dict[str, Any],
    output_dir: Path,
    start: int,
    end: int,
    depth_range: tuple[float, float] | None = None,
    save_raw_maps: bool = False,
) -> None:
    if not 0 <= start < end <= len(document["frames"]):
        raise ValueError(f"Invalid camera batch [{start}, {end})")

    batch_document = dict(document)
    batch_document["frames"] = document["frames"][start:end]
    scene_path = output_dir / "batch_scenes" / f"orbit_{start:05d}_{end:05d}.xml"
    camera_names = write_orbit_scene(batch_document, scene_path)

    # Pale allocates per-camera gradient buffers even for forward-only rendering.
    # Each worker therefore loads only a small camera batch, and process exit
    # guarantees that the SYCL/CUDA allocator releases every batch allocation.
    import pale

    assets_root = resolve_run_path(Path(run_config["assets_root"]))
    renderer = pale.Renderer(
        str(assets_root),
        str(scene_path),
        str(points_path),
        dict(run_config["renderer_settings"]),
    )
    available = set(renderer.get_camera_names())
    missing = [name for name in camera_names if name not in available]
    if missing:
        raise RuntimeError(f"Pale did not load orbit cameras: {missing[:3]}")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    depth_range = depth_display_range(document, depth_range)
    for frame, camera_name in zip(batch_document["frames"], camera_names, strict=True):
        frame_index = int(frame["index"])
        rendered = renderer.render_forward(camera_name)
        rgb = rendered_rgb(rendered, camera_name)
        camera_output = rendered[camera_name]
        save_geometry_frame(output_dir, frame_index, camera_output["median_depth"],
                            camera_output["visible_normal"], depth_range,
                            save_raw=save_raw_maps)
        frame_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(frame_u8, mode="RGB").save(
            frames_dir / f"frame_{frame_index:05d}.png"
        )
        print(f"[{frame_index + 1:04d}/{len(document['frames']):04d}] {camera_name}")


def encode_video(
    *,
    frames_dir: Path,
    frame_count: int,
    video_path: Path,
    fps: float,
) -> None:
    writer = imageio.get_writer(
        video_path,
        fps=float(fps),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame_index in range(frame_count):
            frame_path = frames_dir / f"frame_{frame_index:05d}.png"
            if not frame_path.is_file():
                raise FileNotFoundError(f"Missing rendered orbit frame: {frame_path}")
            writer.append_data(imageio.imread(frame_path))
    finally:
        writer.close()


def resolve_run_path(raw_path: Path, *, base: Path = PROJECT_ROOT) -> Path:
    expanded = raw_path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed run containing run_config.json and points_final.ply")
    parser.add_argument("--points", type=Path, default=None, help="Surfel PLY override")
    parser.add_argument("--poses", type=Path, default=None, help="Consume an existing shared orbit pose JSON instead of generating one")
    parser.add_argument("--poses-output", type=Path, default=None, help="Generated pose JSON path (default: OUTPUT/orbit_poses.json)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: RUN_DIR/orbit")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fov-degrees", type=float, default=45.0, help="Vertical field of view")
    parser.add_argument("--radius", type=float, default=None, help="Camera distance from look-at center; default auto-fits model")
    parser.add_argument("--radius-margin", type=float, default=1.15)
    parser.add_argument("--elevation-degrees", type=float, default=-10.0)
    parser.add_argument("--start-angle-degrees", type=float, default=0.0)
    parser.add_argument("--clockwise", action="store_true")
    parser.add_argument("--emissive-threshold", type=float, default=0.0, help="Only power <= threshold contributes to model center")
    parser.add_argument("--poses-only", action="store_true", help="Write poses and orbit XML without rendering")
    parser.add_argument(
        "--camera-batch-size", "--worker",
        type=int,
        default=4,
        help=(
            "Cameras loaded by each isolated Pale worker (default: 4; --worker is an alias). "
            "Lower to 1 if another GPU job is using substantial memory."
        ),
    )
    parser.add_argument("--no-video", action="store_true", help="Save PNG frames without encoding MP4")
    parser.add_argument("--video-name", default="orbit.mp4")
    parser.add_argument("--depth-range", type=float, nargs=2, metavar=("NEAR", "FAR"),
                        help="Fixed depth visualization range in scene units; default uses orbit bounds")
    parser.add_argument("--save-raw-maps", action="store_true",
                        help="Also save float depth and world normals in per-frame NPZ files")
    parser.add_argument("--worker-start", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-end", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_path(args.run_dir)
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))

    output_dir = resolve_run_path(args.output_dir) if args.output_dir else run_dir / "orbit"
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = resolve_run_path(args.points) if args.points else run_dir / "points_final.ply"
    if not points_path.is_file():
        raise FileNotFoundError(f"Missing surfel model: {points_path}")

    if args.poses is not None:
        poses_path = resolve_run_path(args.poses)
        document = json.loads(poses_path.read_text(encoding="utf-8"))
        validate_pose_document(document)
    else:
        positions, powers = read_surfel_positions_and_powers(points_path)
        center, extent, count = compute_non_emissive_bounds(
            positions,
            powers,
            emissive_threshold=args.emissive_threshold,
        )
        document = make_orbit_pose_document(
            center=center,
            model_extent=extent,
            non_emissive_surfel_count=count,
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            vertical_fov_degrees=args.fov_degrees,
            radius=args.radius,
            radius_margin=args.radius_margin,
            elevation_degrees=args.elevation_degrees,
            start_angle_degrees=args.start_angle_degrees,
            clockwise=args.clockwise,
            emissive_threshold=args.emissive_threshold,
            source_points=points_path,
        )
        poses_path = (
            resolve_run_path(args.poses_output)
            if args.poses_output is not None
            else output_dir / "orbit_poses.json"
        )
        poses_path.parent.mkdir(parents=True, exist_ok=True)
        poses_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    depth_range = depth_display_range(document, args.depth_range)
    worker_requested = args.worker_start is not None or args.worker_end is not None
    if worker_requested:
        if args.worker_start is None or args.worker_end is None:
            raise ValueError("Internal worker mode requires both batch bounds")
        render_pose_batch(
            run_config=run_config,
            points_path=points_path,
            document=document,
            output_dir=output_dir,
            start=args.worker_start,
            end=args.worker_end,
            depth_range=depth_range,
            save_raw_maps=args.save_raw_maps,
        )
        return

    scene_path = output_dir / "orbit_scene.xml"
    write_orbit_scene(document, scene_path)
    print(f"Look-at center: {document['center']}")
    print(f"Shared poses  : {poses_path}")
    print(f"Orbit scene   : {scene_path}")
    if args.poses_only:
        return

    write_geometry_metadata(output_dir, depth_range, save_raw=args.save_raw_maps)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.camera_batch_size <= 0:
        raise ValueError("--camera-batch-size must be positive")

    def run_worker(start: int, end: int) -> None:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-dir", str(run_dir),
            "--points", str(points_path),
            "--poses", str(poses_path),
            "--output-dir", str(output_dir),
            "--worker-start", str(start),
            "--worker-end", str(end),
            "--no-video",
            "--depth-range", str(depth_range[0]), str(depth_range[1]),
        ]
        if args.save_raw_maps:
            command.append("--save-raw-maps")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            if end - start <= 1:
                raise
            midpoint = start + (end - start) // 2
            print(
                f"Camera batch [{start}, {end}) failed; retrying as "
                f"[{start}, {midpoint}) and [{midpoint}, {end}).",
                flush=True,
            )
            run_worker(start, midpoint)
            run_worker(midpoint, end)

    for batch_start, batch_end in camera_batch_ranges(
        len(document["frames"]),
        args.camera_batch_size,
    ):
        run_worker(batch_start, batch_end)

    video_path = output_dir / args.video_name
    if not args.no_video:
        encode_video(
            frames_dir=frames_dir,
            frame_count=len(document["frames"]),
            video_path=video_path,
            fps=args.fps,
        )
        for name in ("depth", "normal"):
            path = geometry_video_path(video_path, name)
            encode_video(frames_dir=output_dir / f"{name}_frames",
                         frame_count=len(document["frames"]), video_path=path, fps=args.fps)
            print(f"{name.title()} video   : {path}")

    print(f"Frames        : {frames_dir}")
    if not args.no_video:
        print(f"Video         : {video_path}")


if __name__ == "__main__":
    main()
