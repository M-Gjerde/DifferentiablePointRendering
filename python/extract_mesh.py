from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

import pale
from render_hooks import get_training_camera_names


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


def path_relative_to_assets(path: Path, assets_root: Path) -> str:
    path = path.resolve()
    assets_root = assets_root.resolve()

    try:
        return str(path.relative_to(assets_root))
    except ValueError:
        return str(path)


def load_renderer(run_dir: Path, points_path: Path):
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        run_config = json.load(file)

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


def sanitize_depth(depth: np.ndarray, depth_trunc: float) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid = (depth > 1.0e-6) & (depth < depth_trunc)
    depth[~valid] = 0.0
    return np.ascontiguousarray(depth)


def post_process_mesh(mesh: o3d.geometry.TriangleMesh, cluster_to_keep: int) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(mesh)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    if len(mesh.triangles) == 0:
        return mesh

    triangle_clusters, cluster_triangle_counts, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_triangle_counts = np.asarray(cluster_triangle_counts)

    keep_count = min(cluster_to_keep, len(cluster_triangle_counts))
    keep_clusters = np.argsort(cluster_triangle_counts)[-keep_count:]

    remove_mask = ~np.isin(triangle_clusters, keep_clusters)
    mesh.remove_triangles_by_mask(remove_mask.tolist())
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

    def extract_mesh_bounded(
        self,
        voxel_size: float,
        sdf_trunc: float,
        depth_trunc: float,
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
            depth = sanitize_depth(depth, depth_trunc)

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
    parser = argparse.ArgumentParser(description="PALE 2DGS-style mesh extraction")

    parser.add_argument("--output_root", type=Path, default=Path("../Assets/OptimizationOutput"))
    parser.add_argument("--index", type=int, default=0)

    parser.add_argument("--cameras_json", type=Path, default="/home/magnus/phd/models/teapot_pbdr/transforms.json")
    parser.add_argument("--camera_names", type=str, default=None)

    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--voxel_size", default=-1.0, type=float)
    parser.add_argument("--depth_trunc", default=-1.0, type=float)
    parser.add_argument("--sdf_trunc", default=-1.0, type=float)
    parser.add_argument("--num_cluster", default=50, type=int)
    parser.add_argument("--mesh_res", default=2048, type=int)

    parser.add_argument("--depth_key", type=str, default="mean_depth", choices=["median_depth", "mean_depth"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_dir, points_path = find_run(args.output_root, args.index)
    renderer, run_config = load_renderer(run_dir, points_path)

    cameras_json = infer_cameras_json(args, run_config)
    cameras = load_nerf_cameras(cameras_json)
    camera_names = get_camera_names(renderer, args)

    print(f"Extracting mesh from {run_dir}")
    print(f"Using point cloud {points_path}")
    print(f"Using cameras {cameras_json}")
    print(f"Rendering {len(camera_names)} cameras")

    radius = load_point_radius(points_path)

    pale_extractor = PaleExtractor(
        renderer=renderer,
        cameras=cameras,
        camera_names=camera_names,
        radius=radius,
        depth_key=args.depth_key,
    )

    mesh_dir = run_dir / "mesh"
    os.makedirs(mesh_dir, exist_ok=True)

    if not args.skip_mesh:
        print("export mesh ...")

        pale_extractor.reconstruction()

        depth_trunc = pale_extractor.radius * 2.0 if args.depth_trunc < 0 else args.depth_trunc
        voxel_size = depth_trunc / args.mesh_res if args.voxel_size < 0 else args.voxel_size
        sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc

        mesh = pale_extractor.extract_mesh_bounded(
            voxel_size=voxel_size,
            sdf_trunc=sdf_trunc,
            depth_trunc=depth_trunc,
        )

        mesh_path = mesh_dir / "fuse.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"mesh saved at {mesh_path}")

        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)

        mesh_post_path = mesh_dir / "fuse_post.ply"
        o3d.io.write_triangle_mesh(str(mesh_post_path), mesh_post)
        print(f"mesh post processed saved at {mesh_post_path}")