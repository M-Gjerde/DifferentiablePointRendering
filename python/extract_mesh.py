from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import imageio.v3 as iio

from poisson_reconstruction import (
    PoissonReconstructionSettings,
    PoissonSamplingSettings,
    load_quaternion_surfel_cloud,
    orient_surfel_normals,
    quaternion_tangent_frames_wxyz,
    reconstruct_screened_poisson,
    sample_opacity_weighted_surfel_surfaces,
)

def parse_vector3(value: str) -> np.ndarray:
    values = [float(part.strip()) for part in value.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {value}")
    return np.asarray(values, dtype=np.float64)


def normalize_vector(vector: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1.0e-12:
        raise ValueError(f"Cannot normalize near-zero vector: {name}")
    return vector / norm


def get_named_child(parent: ET.Element, tag: str, name: str) -> ET.Element:
    child = parent.find(f"{tag}[@name='{name}']")
    if child is None:
        raise ValueError(f"Missing <{tag} name=\"{name}\"> in XML element {parent.tag}")
    return child


def get_named_float(parent: ET.Element, name: str) -> float:
    return float(get_named_child(parent, "float", name).attrib["value"])


def get_named_integer(parent: ET.Element, name: str) -> int:
    return int(get_named_child(parent, "integer", name).attrib["value"])


def lookat_to_opencv_world_to_camera(origin: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    # XML/Mitsuba-style lookat (Z-Up world):
    #   forward = target - origin
    #   up      = world-space camera up
    #
    # Open3D/OpenCV camera convention:
    #   +X right
    #   +Y down
    #   +Z forward
    forward = normalize_vector(target - origin, "forward")
    world_up = normalize_vector(up, "up")

    camera_right = normalize_vector(np.cross(forward, world_up), "camera_right")
    camera_down = normalize_vector(np.cross(forward, camera_right), "camera_down")

    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 0] = camera_right
    camera_to_world[:3, 1] = camera_down
    camera_to_world[:3, 2] = forward
    camera_to_world[:3, 3] = origin

    return np.linalg.inv(camera_to_world)


def load_scene_xml_cameras(scene_xml: Path) -> dict[str, Open3DCamera]:
    scene_xml = scene_xml.expanduser().resolve()
    root = ET.parse(scene_xml).getroot()

    cameras: dict[str, Open3DCamera] = {}

    for sensor in root.findall("sensor[@type='perspective']"):
        camera_name = sensor.attrib.get("id")
        if camera_name is None:
            raise ValueError("Found perspective sensor without id attribute.")

        film = sensor.find("film")
        if film is None:
            raise ValueError(f"Camera {camera_name} is missing <film>.")

        width = get_named_integer(film, "width")
        height = get_named_integer(film, "height")

        fx = get_named_float(sensor, "fx")
        fy = get_named_float(sensor, "fy")
        cx = get_named_float(sensor, "cx")
        cy = get_named_float(sensor, "cy")

        lookat = sensor.find("transform[@name='to_world']/lookat")
        if lookat is None:
            raise ValueError(f"Camera {camera_name} is missing transform/to_world/lookat.")

        origin = parse_vector3(lookat.attrib["origin"])
        target = parse_vector3(lookat.attrib["target"])
        up = parse_vector3(lookat.attrib["up"])

        world_to_camera = lookat_to_opencv_world_to_camera(
            origin=origin,
            target=target,
            up=up,
        )

        cameras[camera_name] = Open3DCamera(
            name=camera_name,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            world_to_camera=world_to_camera,
        )

    if not cameras:
        raise ValueError(f"No perspective cameras found in {scene_xml}")

    return cameras

def infer_cameras_xml(args: argparse.Namespace, run_config: dict) -> Path:
    assets_root = Path(run_config["assets_root"]).expanduser().resolve()
    scene_xml = Path(run_config["scene_xml"]).expanduser()

    if not scene_xml.is_absolute():
        scene_xml = assets_root / scene_xml

    if not scene_xml.is_file():
        raise FileNotFoundError(f"Could not find scene XML: {scene_xml}")

    return scene_xml.resolve()

@dataclass(frozen=True)
class Open3DCamera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray

    def intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            self.width,
            self.height,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        )


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_run_timestamp(path_name: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", path_name)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")


def find_run(output_root: Path, run_index: int) -> tuple[Path, Path]:
    output_root = output_root.resolve()

    if output_root.is_file():
        if output_root.name != "points_final.ply":
            raise ValueError(f"Expected points_final.ply, got {output_root}")
        return output_root.parent, output_root

    if (output_root / "points_final.ply").is_file():
        return output_root, output_root / "points_final.ply"

    candidates = []
    for run_dir in output_root.iterdir():
        points_path = run_dir / "points_final.ply"
        if not run_dir.is_dir() or not points_path.is_file():
            continue

        timestamp = parse_run_timestamp(run_dir.name)
        candidates.append((timestamp is not None, timestamp or datetime.min, points_path.stat().st_mtime, run_dir))

    if not candidates:
        raise FileNotFoundError(f"No run folders with points_final.ply found under {output_root}")

    candidates.sort(reverse=True)

    if run_index < 0 or run_index >= len(candidates):
        available_runs = "\n".join(f"[{i}] {item[3].name}" for i, item in enumerate(candidates))
        raise IndexError(f"--index {run_index} is out of range.\nAvailable runs:\n{available_runs}")

    run_dir = candidates[run_index][3]
    return run_dir, run_dir / "points_final.ply"


def find_run_dir_for_ply(points_path: Path) -> Path | None:
    points_path = points_path.expanduser().resolve()

    for parent in points_path.parents:
        if (parent / "run_config.json").is_file():
            return parent

    return None


def find_run_and_points(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.ply is None:
        return find_run(args.output_root, args.index)

    points_path = args.ply.expanduser().resolve()
    if not points_path.is_file():
        raise FileNotFoundError(f"PLY file does not exist: {points_path}")

    run_dir = find_run_dir_for_ply(points_path)
    if run_dir is None:
        run_dir, _ = find_run(args.output_root, args.index)

    return run_dir, points_path


def path_relative_to_assets(path: Path, assets_root: Path) -> str:
    path = path.resolve()
    assets_root = assets_root.resolve()

    try:
        return str(path.relative_to(assets_root))
    except ValueError:
        return str(path)


def format_ply_override_value(value: float) -> str:
    return f"{value:.9g}"


def format_suffix_value(value: float) -> str:
    return f"{value:.9g}".replace("-", "m").replace(".", "p")


def write_points_with_property_overrides(points_path: Path, output_path: Path, property_values: dict[str, float]) -> Path:
    points_path = points_path.resolve()
    output_path = output_path.resolve()

    if points_path == output_path:
        raise ValueError("Refusing to overwrite the input point cloud while applying property overrides.")

    lines = points_path.read_text(encoding="utf-8").splitlines(keepends=True)

    header_end_index = None
    for line_index, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end_index = line_index
            break

    if header_end_index is None:
        raise ValueError(f"{points_path} does not contain a valid PLY end_header line.")

    header_lines = lines[:header_end_index + 1]
    body_lines = lines[header_end_index + 1:]

    vertex_count = None
    vertex_property_names: list[str] = []
    current_element = None

    for header_line in header_lines:
        parts = header_line.strip().split()
        if not parts:
            continue

        if parts[0] == "format" and (len(parts) < 2 or parts[1] != "ascii"):
            raise ValueError("Applying property overrides currently supports ASCII PLY files only.")

        if parts[0] == "element":
            if len(parts) < 3:
                raise ValueError(f"Malformed PLY element line: {header_line.strip()}")
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
            continue

        if parts[0] == "property" and current_element == "vertex":
            if len(parts) < 3:
                raise ValueError(f"Malformed PLY property line: {header_line.strip()}")
            vertex_property_names.append(parts[-1])

    if vertex_count is None:
        raise ValueError(f"{points_path} does not contain a vertex element.")

    if len(body_lines) < vertex_count:
        raise ValueError(f"{points_path} has fewer vertex rows than declared in the header.")

    missing_property_names = [property_name for property_name in property_values if property_name not in vertex_property_names]
    if missing_property_names:
        raise ValueError(f"{points_path} does not contain vertex properties: {', '.join(missing_property_names)}")

    property_column_indices = {
        property_name: vertex_property_names.index(property_name)
        for property_name in property_values
    }
    formatted_property_values = {
        property_name: format_ply_override_value(property_value)
        for property_name, property_value in property_values.items()
    }

    modified_vertex_lines: list[str] = []

    for vertex_line_index, vertex_line in enumerate(body_lines[:vertex_count]):
        values = vertex_line.strip().split()
        if len(values) < len(vertex_property_names):
            raise ValueError(
                f"Vertex row {vertex_line_index} has {len(values)} values, "
                f"but the header declares {len(vertex_property_names)} vertex properties."
            )

        for property_name, column_index in property_column_indices.items():
            values[column_index] = formatted_property_values[property_name]

        if vertex_line.endswith("\r\n"):
            newline = "\r\n"
        elif vertex_line.endswith("\n"):
            newline = "\n"
        else:
            newline = ""

        modified_vertex_lines.append(" ".join(values) + newline)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(header_lines + modified_vertex_lines + body_lines[vertex_count:]),
        encoding="utf-8",
    )

    return output_path


def write_points_with_forced_opacity(points_path: Path, output_path: Path) -> Path:
    return write_points_with_property_overrides(
        points_path=points_path,
        output_path=output_path,
        property_values={"opacity": 1.0},
    )



def validate_quaternion_surfel_ply(points_path: Path) -> None:
    points_path = points_path.resolve()

    required_vertex_properties = {
        "x", "y", "z",
        "rot_w", "rot_x", "rot_y", "rot_z",
        "su", "sv",
        "albedo_r", "albedo_g", "albedo_b",
        "opacity", "beta", "shape", "power",
    }

    current_element = None
    vertex_property_names: list[str] = []
    ply_format: str | None = None

    with points_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            parts = stripped.split()

            if not parts:
                continue

            if parts[0] == "format":
                if len(parts) < 2:
                    raise ValueError(f"Malformed PLY format line in {points_path}")
                ply_format = parts[1]
                continue

            if parts[0] == "element":
                if len(parts) < 3:
                    raise ValueError(f"Malformed PLY element line in {points_path}: {stripped}")
                current_element = parts[1]
                continue

            if parts[0] == "property" and current_element == "vertex":
                if len(parts) < 3:
                    raise ValueError(f"Malformed PLY property line in {points_path}: {stripped}")
                vertex_property_names.append(parts[-1])
                continue

            if stripped == "end_header":
                break

    if ply_format != "ascii":
        raise ValueError(
            f"{points_path} is {ply_format!r}. The current quaternion surfel loader expects ASCII PLY."
        )

    available = set(vertex_property_names)
    missing = sorted(required_vertex_properties - available)
    if missing:
        raise ValueError(
            f"{points_path} is not in the quaternion surfel format. Missing vertex properties: "
            + ", ".join(missing)
            + ". Expected x y z rot_w rot_x rot_y rot_z su sv albedo_r albedo_g albedo_b opacity beta shape power."
        )

def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_renderer(run_dir: Path, points_path: Path):
    import pale

    run_config = load_run_config(run_dir)

    assets_root = Path(run_config["assets_root"]).resolve()
    scene_xml = run_config["scene_xml"]
    renderer_settings = run_config["renderer_settings"]
    pointcloud_path = path_relative_to_assets(points_path, assets_root)

    return pale.Renderer(
        str(assets_root),
        scene_xml,
        pointcloud_path,
        renderer_settings,
    ), run_config


def camera_world_positions(cameras: dict[str, Open3DCamera]) -> np.ndarray:
    positions = []
    for camera in cameras.values():
        camera_to_world = np.linalg.inv(camera.world_to_camera)
        positions.append(camera_to_world[:3, 3])
    if not positions:
        raise ValueError("Cannot orient Poisson normals without any cameras")
    return np.ascontiguousarray(np.stack(positions, axis=0), dtype=np.float64)


def infer_cameras_json(args: argparse.Namespace, run_config: dict) -> Path:
    if args.cameras_json is not None:
        return args.cameras_json.resolve()

    if "cameras_json" in run_config:
        return Path(run_config["cameras_json"]).resolve()

    assets_root = Path(run_config["assets_root"]).resolve()
    candidates = [
        assets_root / "transforms_train.json",
        assets_root / "transforms.json",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not infer cameras JSON. Pass --cameras_json /path/to/transforms.json"
    )


def load_nerf_cameras(cameras_json: Path) -> dict[str, Open3DCamera]:
    with cameras_json.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "frames" not in data:
        raise ValueError("This simplified extractor expects a NeRF/2DGS-style transforms JSON with frames[].")

    top_width = data.get("w", data.get("width"))
    top_height = data.get("h", data.get("height"))
    top_fx = data.get("fl_x", data.get("fx"))
    top_fy = data.get("fl_y", data.get("fy"))
    top_cx = data.get("cx")
    top_cy = data.get("cy")
    top_angle_x = data.get("camera_angle_x")

    cameras: dict[str, Open3DCamera] = {}

    for frame_index, frame in enumerate(data["frames"]):
        file_path = frame.get("file_path", f"{frame_index:04d}")
        camera_name = frame.get("camera_name", frame.get("name", Path(file_path).stem))

        width = int(frame.get("w", frame.get("width", top_width)))
        height = int(frame.get("h", frame.get("height", top_height)))

        fx = frame.get("fl_x", frame.get("fx", top_fx))
        fy = frame.get("fl_y", frame.get("fy", top_fy))

        if fx is None:
            angle_x = frame.get("camera_angle_x", top_angle_x)
            if angle_x is None:
                raise ValueError(f"Camera {camera_name} is missing fl_x/fx/camera_angle_x")
            fx = 0.5 * width / np.tan(0.5 * float(angle_x))

        if fy is None:
            fy = fx

        cx = float(frame.get("cx", top_cx if top_cx is not None else 0.5 * (width - 1)))
        cy = float(frame.get("cy", top_cy if top_cy is not None else 0.5 * (height - 1)))

        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64).reshape(4, 4)

        # NeRF/Blender/OpenGL c2w:
        #   +X right, +Y up, -Z forward.
        # Open3D/OpenCV depth convention:
        #   +X right, +Y down, +Z forward.
        opengl_to_opencv = np.eye(4, dtype=np.float64)
        opengl_to_opencv[1, 1] = -1.0
        opengl_to_opencv[2, 2] = -1.0

        opencv_c2w = c2w @ opengl_to_opencv
        world_to_camera = np.linalg.inv(opencv_c2w)

        cameras[camera_name] = Open3DCamera(
            name=camera_name,
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=cx,
            cy=cy,
            world_to_camera=world_to_camera,
        )

    return cameras


def load_point_radius(points_path: Path) -> float:
    point_cloud = o3d.io.read_point_cloud(str(points_path))
    points = np.asarray(point_cloud.points)

    if points.size == 0:
        raise RuntimeError(f"No points found in {points_path}")

    center = np.mean(points, axis=0)
    return float(np.linalg.norm(points - center, axis=1).max())


def get_camera_names(renderer, args: argparse.Namespace) -> list[str]:
    from render_hooks import get_training_camera_names

    if args.camera_names is not None:
        return [name.strip() for name in args.camera_names.split(",") if name.strip()]

    camera_names = get_training_camera_names(renderer)
    if isinstance(camera_names, dict):
        return list(camera_names.keys())

    return list(camera_names)


def sanitize_color(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    if image.max(initial=0.0) <= 1.5:
        image = 255.0 * np.clip(image, 0.0, 1.0)
    else:
        image = np.clip(image, 0.0, 255.0)

    return np.ascontiguousarray(image.astype(np.uint8))


def depth_to_visualization(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    depth_u8 = np.zeros(depth.shape, dtype=np.uint8)

    if not np.any(valid):
        return depth_u8

    valid_depth = depth[valid]
    lower = float(valid_depth.min(initial=0.0))
    upper = float(valid_depth.max(initial=lower))

    if upper <= lower:
        depth_u8[valid] = 255
        return depth_u8

    normalized = np.clip((depth - lower) / (upper - lower), 0.0, 1.0)
    depth_u8[valid] = (255.0 * normalized[valid]).astype(np.uint8)
    return depth_u8


def save_extraction_render_images(
    output_dir: Path,
    camera_name: str,
    color: np.ndarray,
    depth: np.ndarray,
    depth_key: str,
    name_suffix: str,
) -> None:
    suffix = f"{name_suffix}_" if name_suffix else ""
    color_path = output_dir / f"render_{suffix}{camera_name}.png"
    depth_path = output_dir / f"{depth_key}_{suffix}{camera_name}.png"

    output_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(color_path.as_posix(), color)
    iio.imwrite(depth_path.as_posix(), depth_to_visualization(depth))


def post_process_mesh(mesh: o3d.geometry.TriangleMesh, cluster_to_keep: int) -> o3d.geometry.TriangleMesh:
    if cluster_to_keep < 0:
        raise ValueError(f"cluster_to_keep must be non-negative, got {cluster_to_keep}")

    mesh = o3d.geometry.TriangleMesh(mesh)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    if len(mesh.triangles) == 0:
        return mesh

    if cluster_to_keep > 0:
        triangle_clusters, cluster_triangle_counts, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_triangle_counts = np.asarray(cluster_triangle_counts)

        keep_count = min(cluster_to_keep, len(cluster_triangle_counts))
        keep_clusters = np.argsort(cluster_triangle_counts)[-keep_count:]

        remove_mask = ~np.isin(triangle_clusters, keep_clusters)
        mesh.remove_triangles_by_mask(remove_mask.tolist())
    mesh.remove_unreferenced_vertices()

    # Edge cleanup can leave a few bow-tie vertices. Removing their incident
    # triangles is deterministic, terminates because every pass removes at
    # least one vertex, and produces a genuinely manifold supported mesh.
    while len(mesh.vertices) > 0:
        non_manifold_vertices = np.asarray(
            mesh.get_non_manifold_vertices(), dtype=np.int64
        )
        if non_manifold_vertices.size == 0:
            break
        vertex_mask = np.zeros(len(mesh.vertices), dtype=bool)
        vertex_mask[non_manifold_vertices] = True
        mesh.remove_vertices_by_mask(vertex_mask.tolist())
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    return mesh


class PaleExtractor:
    def __init__(
        self,
        renderer,
        cameras: dict[str, Open3DCamera],
        camera_names: list[str],
        radius: float,
        depth_key: str,
    ):
        self.renderer = renderer
        self.cameras = cameras
        self.camera_names = camera_names
        self.radius = radius
        self.depth_key = depth_key
        self.forward_out = None

    def reconstruction(self) -> None:
        self.forward_out = self.renderer.render_forward()

    def infer_depth_trunc(self, margin: float = 1.05) -> float:
        if self.forward_out is None:
            raise RuntimeError("Call reconstruction() before infer_depth_trunc().")

        max_depth = 0.0

        for camera_name in self.camera_names:
            camera_output = self.forward_out.get(camera_name)
            if camera_output is None or self.depth_key not in camera_output:
                continue

            depth = as_numpy(camera_output[self.depth_key])
            valid_depth = depth[np.isfinite(depth) & (depth > 0.0)]
            if valid_depth.size == 0:
                continue

            max_depth = max(max_depth, float(valid_depth.max(initial=max_depth)))

        if max_depth <= 0.0:
            raise RuntimeError(f"Could not infer depth_trunc: no positive finite {self.depth_key} values.")

        return margin * max_depth

    def extract_mesh_bounded(
        self,
        voxel_size: float,
        sdf_trunc: float,
        depth_trunc: float,
        render_output_dir: Path | None = None,
        render_name_suffix: str = "",
    ) -> o3d.geometry.TriangleMesh:
        if self.forward_out is None:
            raise RuntimeError("Call reconstruction() before extract_mesh_bounded().")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        integrated_count = 0

        for camera_name in self.camera_names:
            if camera_name not in self.forward_out:
                print(f"Skipping {camera_name}: missing renderer output")
                continue

            if camera_name not in self.cameras:
                print(f"Skipping {camera_name}: missing camera metadata")
                continue

            camera = self.cameras[camera_name]
            camera_output = self.forward_out[camera_name]

            color = as_numpy(camera_output["image"])[..., :3]
            depth = as_numpy(camera_output[self.depth_key])

            if depth.shape != (camera.height, camera.width):
                raise RuntimeError(
                    f"Depth/camera size mismatch for {camera_name}: "
                    f"depth={depth.shape}, camera={(camera.height, camera.width)}"
                )

            color = sanitize_color(color)
            depth = np.ascontiguousarray(depth, dtype=np.float32)

            if render_output_dir is not None:
                save_extraction_render_images(
                    output_dir=render_output_dir / "renders",
                    camera_name=camera_name,
                    color=color,
                    depth=depth,
                    depth_key=self.depth_key,
                    name_suffix=render_name_suffix,
                )

            if np.count_nonzero(depth) == 0:
                print(f"Skipping {camera_name}: no valid depth")
                continue

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color),
                o3d.geometry.Image(depth),
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )

            volume.integrate(
                rgbd,
                camera.intrinsic(),
                camera.world_to_camera,
            )

            integrated_count += 1

        if integrated_count == 0:
            raise RuntimeError("TSDF integration used zero cameras.")

        print(f"Integrated {integrated_count} RGB-D frames")

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PALE surfel mesh extraction")

    parser.add_argument("--output-root", "-o", type=Path, default=Path("OptimizationOutput"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
        help=(
            "Use this PLY instead of the selected run's points_final.ply. "
            "If the PLY is inside a run folder, that run_config.json is used; "
            "otherwise --output-root/--index provide the run context."
        ),
    )
    parser.add_argument("--camera-names", type=str, default=None)

    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--mesh-output-subdir",
        type=Path,
        default=Path("mesh"),
        help=(
            "Mesh output folder. Relative paths are interpreted inside the run directory. "
            "Use this for iteration checkpoints so they do not overwrite mesh/fuse_post.ply."
        ),
    )

    parser.add_argument("--voxel-size", default=-1.0, type=float)
    parser.add_argument("--depth-trunc", default=-1.0, type=float)
    parser.add_argument("--sdf-trunc", default=-1.0, type=float)
    parser.add_argument("--num-cluster", default=50, type=int)
    parser.add_argument("--mesh-res", default=2048, type=int)
    parser.add_argument(
        "--method",
        "--reconstruction-method",
        choices=["tsdf", "poisson"],
        default="tsdf",
        help="Use camera-depth TSDF fusion or opacity-weighted surfel Screened Poisson reconstruction.",
    )

    parser.add_argument("--poisson-samples", type=int, default=500_000)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--poisson-scale", type=float, default=1.1)
    parser.add_argument(
        "--poisson-linear-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use linear interpolation for Poisson iso-vertex positions.",
    )
    parser.add_argument("--poisson-threads", type=int, default=-1)
    parser.add_argument("--poisson-seed", type=int, default=0)
    parser.add_argument("--poisson-opacity-threshold", type=float, default=1.0e-3)
    parser.add_argument("--poisson-emitter-power-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--poisson-min-samples-per-surfel", type=int, default=0)
    parser.add_argument(
        "--poisson-beta-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample the full beta opacity profile instead of uniformly sampling each ellipse.",
    )
    parser.add_argument(
        "--poisson-normal-orientation",
        choices=["surfel", "camera", "consistent", "consistent-camera"],
        default="consistent-camera",
        help="How to make two-sided surfel normals globally usable by Poisson.",
    )
    parser.add_argument("--poisson-orientation-neighbors", type=int, default=20)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.01)
    parser.add_argument("--poisson-coverage-trim-cells", type=float, default=4.0)
    parser.add_argument("--poisson-crop-padding-cells", type=float, default=4.0)
    parser.add_argument(
        "--poisson-save-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the oriented opacity-weighted Poisson input samples for inspection.",
    )

    parser.add_argument("--depth-key", type=str, default="median_depth", choices=["median_depth", "mean_depth"])
    parser.add_argument(
        "--force-opacity-one",
        "--force_opacity_1",
        action="store_true",
        help="Write a derived point cloud with opacity=1.0 for all surfels before rendering/extracting the mesh.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Write a derived point cloud with this beta value for all surfels before rendering/extracting the mesh.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_dir, points_path = find_run_and_points(args)

    mesh_dir = args.mesh_output_subdir
    if not mesh_dir.is_absolute():
        mesh_dir = run_dir / mesh_dir
    os.makedirs(mesh_dir, exist_ok=True)

    point_property_overrides: dict[str, float] = {}
    mesh_name_suffix_parts: list[str] = []

    if args.force_opacity_one:
        point_property_overrides["opacity"] = 1.0
        mesh_name_suffix_parts.append("opacity_1")

    if args.beta is not None:
        point_property_overrides["beta"] = args.beta
        mesh_name_suffix_parts.append(f"beta_{format_suffix_value(args.beta)}")

    mesh_name_suffix = f"_{'_'.join(mesh_name_suffix_parts)}" if mesh_name_suffix_parts else ""

    if point_property_overrides:
        points_path = write_points_with_property_overrides(
            points_path=points_path,
            output_path=mesh_dir / f"points{mesh_name_suffix}.ply",
            property_values=point_property_overrides,
        )
        print(f"Using property-overridden point cloud {points_path}")

    validate_quaternion_surfel_ply(points_path)

    run_config = load_run_config(run_dir)
    cameras_path = infer_cameras_xml(args, run_config)
    cameras = load_scene_xml_cameras(cameras_path)

    print(f"Using cameras {cameras_path}")
    print(f"Extracting mesh from {run_dir}")
    print(f"Using point cloud {points_path}")

    if not args.skip_mesh:
        if args.method == "poisson":
            print("Sampling opacity-weighted surfel surfaces for Screened Poisson reconstruction ...")
            cloud, load_report = load_quaternion_surfel_cloud(
                points_path,
                opacity_threshold=args.poisson_opacity_threshold,
                emitter_power_epsilon=args.poisson_emitter_power_epsilon,
            )
            _, _, raw_normals = quaternion_tangent_frames_wxyz(cloud.quaternions_wxyz)

            orientation_cameras = cameras
            if args.camera_names is not None:
                requested_camera_names = [
                    name.strip() for name in args.camera_names.split(",") if name.strip()
                ]
                missing_camera_names = [
                    name for name in requested_camera_names if name not in cameras
                ]
                if missing_camera_names:
                    raise ValueError(
                        "Unknown Poisson orientation cameras: "
                        + ", ".join(missing_camera_names)
                    )
                orientation_cameras = {
                    name: cameras[name] for name in requested_camera_names
                }

            oriented_normals, orientation_report = orient_surfel_normals(
                positions=cloud.positions,
                normals=raw_normals,
                mode=args.poisson_normal_orientation,
                consistent_neighbor_count=args.poisson_orientation_neighbors,
                camera_positions=(
                    camera_world_positions(orientation_cameras)
                    if args.poisson_normal_orientation in {"camera", "consistent-camera"}
                    else None
                ),
            )
            sampling_settings = PoissonSamplingSettings(
                sample_count=args.poisson_samples,
                seed=args.poisson_seed,
                minimum_samples_per_surfel=args.poisson_min_samples_per_surfel,
                use_beta_profile=args.poisson_beta_profile,
            )
            poisson_samples, sampling_report = sample_opacity_weighted_surfel_surfaces(
                cloud=cloud,
                oriented_normals=oriented_normals,
                settings=sampling_settings,
            )

            if args.poisson_save_samples:
                samples_path = mesh_dir / f"poisson_samples{mesh_name_suffix}.ply"
                if not o3d.io.write_point_cloud(str(samples_path), poisson_samples):
                    raise RuntimeError(f"Failed to save Poisson samples to {samples_path}")
                print(f"Poisson samples saved at {samples_path}")

            reconstruction_settings = PoissonReconstructionSettings(
                depth=args.poisson_depth,
                scale=args.poisson_scale,
                linear_fit=args.poisson_linear_fit,
                n_threads=args.poisson_threads,
                density_quantile=args.poisson_density_quantile,
                coverage_trim_cells=args.poisson_coverage_trim_cells,
                crop_padding_cells=args.poisson_crop_padding_cells,
            )
            print(
                f"Poisson settings: samples={args.poisson_samples}, depth={args.poisson_depth}, "
                f"scale={args.poisson_scale:.4g}, density_quantile={args.poisson_density_quantile:.4g}, "
                f"coverage_cells={args.poisson_coverage_trim_cells:.4g}"
            )
            poisson_raw, poisson_trimmed, reconstruction_report = reconstruct_screened_poisson(
                samples=poisson_samples,
                settings=reconstruction_settings,
            )

            mesh_path = mesh_dir / f"poisson{mesh_name_suffix}.ply"
            if not o3d.io.write_triangle_mesh(str(mesh_path), poisson_raw):
                raise RuntimeError(f"Failed to save raw Poisson mesh to {mesh_path}")
            print(f"raw Poisson mesh saved at {mesh_path}")

            poisson_post = post_process_mesh(
                poisson_trimmed,
                cluster_to_keep=args.num_cluster,
            )
            mesh_post_path = mesh_dir / f"poisson_post{mesh_name_suffix}.ply"
            if not o3d.io.write_triangle_mesh(str(mesh_post_path), poisson_post):
                raise RuntimeError(f"Failed to save post-processed Poisson mesh to {mesh_post_path}")
            print(f"post-processed Poisson mesh saved at {mesh_post_path}")

            report = {
                "method": "screened_poisson",
                "source_point_cloud": str(points_path),
                "camera_file": str(cameras_path),
                "orientation_camera_count": len(orientation_cameras),
                "load": load_report,
                "normal_orientation": orientation_report,
                "sampling": sampling_report,
                "reconstruction": reconstruction_report,
                "post_process": {
                    "clusters_to_keep": int(args.num_cluster),
                    "vertices": int(len(poisson_post.vertices)),
                    "triangles": int(len(poisson_post.triangles)),
                    "connected_components": int(
                        len(poisson_post.cluster_connected_triangles()[1])
                        if len(poisson_post.triangles) > 0
                        else 0
                    ),
                    "edge_manifold_allow_boundary": bool(
                        poisson_post.is_edge_manifold(allow_boundary_edges=True)
                    ),
                    "vertex_manifold": bool(poisson_post.is_vertex_manifold()),
                },
            }
            report_path = mesh_dir / f"poisson_report{mesh_name_suffix}.json"
            with report_path.open("w", encoding="utf-8") as report_file:
                json.dump(report, report_file, indent=2)
            print(f"Poisson extraction report saved at {report_path}")
        else:
            renderer, _ = load_renderer(run_dir, points_path)
            camera_names = get_camera_names(renderer, args)
            print(f"Rendering {len(camera_names)} cameras")

            radius = load_point_radius(points_path)
            pale_extractor = PaleExtractor(
                renderer=renderer,
                cameras=cameras,
                camera_names=camera_names,
                radius=radius,
                depth_key=args.depth_key,
            )

            print("Exporting TSDF mesh ...")
            pale_extractor.reconstruction()

            depth_trunc = pale_extractor.infer_depth_trunc() if args.depth_trunc < 0 else args.depth_trunc
            voxel_size = (2.0 * pale_extractor.radius) / args.mesh_res if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 10.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            print(
                f"TSDF settings: depth_trunc={depth_trunc:.6g}, "
                f"voxel_size={voxel_size:.6g}, sdf_trunc={sdf_trunc:.6g}"
            )

            mesh = pale_extractor.extract_mesh_bounded(
                voxel_size=voxel_size,
                sdf_trunc=sdf_trunc,
                depth_trunc=depth_trunc,
                render_output_dir=mesh_dir,
                render_name_suffix=mesh_name_suffix.lstrip("_"),
            )

            mesh_path = mesh_dir / f"fuse{mesh_name_suffix}.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), mesh)
            print(f"mesh saved at {mesh_path}")

            mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)

            mesh_post_path = mesh_dir / f"fuse_post{mesh_name_suffix}.ply"
            o3d.io.write_triangle_mesh(str(mesh_post_path), mesh_post)
            print(f"mesh post processed saved at {mesh_post_path}")
