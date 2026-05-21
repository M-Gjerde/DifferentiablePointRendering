from __future__ import annotations

"""
Standalone 2DGS-style mesh extraction for your PALE renderer.

This script is meant to be run after optimization has finished.
It does NOT reuse saved depth PNGs/NPYs. Instead it:

  1. Finds the latest run folder using points_final.ply.
  2. Reads <run>/run_config.json saved by main.py.
  3. Creates a fresh pale.Renderer exactly like main.py, but with
     points_final.ply as the point cloud.
  4. Calls renderer.render_forward().
  5. Uses the fresh floating-point median_depth maps for Open3D TSDF fusion.
  6. Writes <run>/mesh/fuse.ply and <run>/mesh/fuse_post.ply.

Typical usage:

    python extract_mesh.py --output-root ../Assets/OptimizationOutput

If camera introspection from the renderer is not available, provide camera
metadata explicitly:

    python extract_mesh.py \
        --output-root ../Assets/OptimizationOutput \
        --cameras-json /path/to/transforms_train.json

Important Open3D convention:
    TSDF integration expects camera z-depth and a world-to-camera matrix T_cw
    satisfying x_camera = T_cw @ x_world.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

import pale
from render_hooks import get_training_camera_names

try:
    from render_hooks import get_all_camera_names
except Exception:  # pragma: no cover
    get_all_camera_names = None


# -----------------------------------------------------------------------------
# Latest-run lookup, reused from your existing standalone script
# -----------------------------------------------------------------------------


def find_latest_points_ply(outputRootPath: Path) -> Path:
    if not outputRootPath.exists():
        raise FileNotFoundError(f"Output root '{outputRootPath}' does not exist.")

    if outputRootPath.is_file():
        print(f"Using PLY file: {outputRootPath}")
        return outputRootPath

    pointsInRoot = outputRootPath / "points_final.ply"
    if pointsInRoot.is_file():
        print(f"Using points_final.ply in run directory: {outputRootPath}")
        return pointsInRoot

    candidateRunDirs: List[Path] = []
    for childPath in outputRootPath.iterdir():
        if childPath.is_dir() and (childPath / "points_final.ply").is_file():
            candidateRunDirs.append(childPath)

    if not candidateRunDirs:
        raise FileNotFoundError(f"No subdirectories with points_final.ply found under '{outputRootPath}'.")

    latestRunDir = max(candidateRunDirs, key=lambda runPath: (runPath / "points_final.ply").stat().st_mtime)
    latestPlyPath = latestRunDir / "points_final.ply"
    print(f"Using latest run directory: {latestRunDir}")
    print(f"points_final.ply: {latestPlyPath}")
    return latestPlyPath


# -----------------------------------------------------------------------------
# Camera representation for Open3D
# -----------------------------------------------------------------------------


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
            int(self.width),
            int(self.height),
            float(self.fx),
            float(self.fy),
            float(self.cx),
            float(self.cy),
        )


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, order="C")


def get_optional(dictionary: dict, *keys, default=None):
    for key in keys:
        if key in dictionary:
            return dictionary[key]
    return default


def get_attr_or_key(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            value = getattr(obj, name)
            return value() if callable(value) and name.startswith("get_") else value
    return default


def matrix4(value, field_name: str) -> np.ndarray:
    mat = as_numpy(value).astype(np.float64, copy=False)
    if mat.size == 16:
        mat = mat.reshape(4, 4)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected {field_name} to be 4x4, got shape {mat.shape}")
    return np.array(mat, dtype=np.float64, copy=True, order="C")


def matrix3(value, field_name: str) -> np.ndarray:
    mat = as_numpy(value).astype(np.float64, copy=False)
    if mat.size == 9:
        mat = mat.reshape(3, 3)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected {field_name} to be 3x3, got shape {mat.shape}")
    return np.array(mat, dtype=np.float64, copy=True, order="C")


def path_is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def renderer_path_argument(path: Path, assets_root: Path, mode: str) -> str:
    """
    main.py passes config.pointcloud_ply to pale.Renderer together with assets_root.
    If the C++ side expects paths relative to assets_root, use
    --pointcloud-path-mode relative-to-assets. The default mode converts to a
    relative path when possible and otherwise falls back to absolute.
    """
    path = Path(path)
    assets_root = Path(assets_root)

    if mode == "absolute":
        return str(path.resolve())
    if mode == "relative-to-assets":
        return str(path.resolve().relative_to(assets_root.resolve()))
    if mode == "auto":
        if path_is_relative_to(path, assets_root):
            return str(path.resolve().relative_to(assets_root.resolve()))
        return str(path.resolve())
    raise ValueError(f"Unsupported --pointcloud-path-mode: {mode}")


# -----------------------------------------------------------------------------
# Run config and renderer creation
# -----------------------------------------------------------------------------


def load_run_config(run_dir: Path) -> dict:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(
            f"Missing {run_config_path}. This extractor expects the run_config.json "
            "written by main.py."
        )
    with run_config_path.open("r", encoding="utf-8") as json_file:
        run_config = json.load(json_file)
    print(f"Loaded run config: {run_config_path}")
    return run_config


def create_renderer_from_run_config(
    run_config: dict,
    points_final_ply: Path,
    pointcloud_path_mode: str,
):
    assets_root = Path(run_config["assets_root"]).resolve()
    scene_xml = run_config["scene_xml"]
    renderer_settings = run_config["renderer_settings"]
    points_arg = renderer_path_argument(points_final_ply, assets_root, pointcloud_path_mode)

    print("Creating renderer:")
    print(f"  assets_root : {assets_root}")
    print(f"  scene_xml   : {scene_xml}")
    print(f"  pointcloud  : {points_arg}")

    return pale.Renderer(
        str(assets_root),
        scene_xml,
        points_arg,
        renderer_settings,
    )


def normalize_camera_names(camera_names) -> List[str]:
    """
    Accept raw C++ binding/wrapper outputs:
      - list/tuple[str]
      - dict[name, ...], where keys are camera names
      - dict-like object exposing keys()
    """
    if camera_names is None:
        return []
    if isinstance(camera_names, dict):
        return [str(name) for name in camera_names.keys()]
    if hasattr(camera_names, "keys"):
        return [str(name) for name in camera_names.keys()]
    return [str(name) for name in list(camera_names)]


# -----------------------------------------------------------------------------
# Forward-output access
# -----------------------------------------------------------------------------


def get_forward_rgba(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    rgba = as_numpy(forward_out[camera_name]["image"]).astype(np.float32, copy=False)
    if rgba.ndim != 3 or rgba.shape[-1] < 3:
        raise RuntimeError(f"Camera '{camera_name}' image has invalid shape {rgba.shape}")
    return np.nan_to_num(rgba, nan=0.0, posinf=0.0, neginf=0.0)


def get_forward_rgb(forward_out: Dict[str, dict], camera_name: str) -> np.ndarray:
    return get_forward_rgba(forward_out, camera_name)[..., :3]


def get_forward_depth(
    forward_out: Dict[str, dict],
    camera_name: str,
    depth_key: str,
) -> np.ndarray:
    camera_out = forward_out[camera_name]
    if depth_key not in camera_out:
        keys = sorted(camera_out.keys())
        raise RuntimeError(
            f"Camera '{camera_name}' has no depth key '{depth_key}'. Available keys: {keys}"
        )
    depth = as_numpy(camera_out[depth_key]).astype(np.float32, copy=False)
    if depth.ndim != 2:
        raise RuntimeError(
            f"Camera '{camera_name}' depth '{depth_key}' has shape {depth.shape}; expected HxW"
        )
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


# -----------------------------------------------------------------------------
# Camera loading / introspection
# -----------------------------------------------------------------------------


def camera_name_from_path(path_like: str) -> str:
    return Path(path_like).stem


def maybe_opengl_c2w_to_opencv_c2w(c2w: np.ndarray, convention: str) -> np.ndarray:
    if convention == "opencv":
        return c2w
    if convention != "opengl":
        raise ValueError(f"Unsupported camera convention: {convention}")

    # NeRF/Blender/OpenGL camera-to-world matrices use camera axes
    #   +X right, +Y up, -Z forward.
    # OpenCV/Open3D image-depth convention uses
    #   +X right, +Y down, +Z forward.
    # This changes only the camera basis, not the camera center.
    flip = np.eye(4, dtype=np.float64)
    flip[1, 1] = -1.0
    flip[2, 2] = -1.0
    return c2w @ flip


def resolve_c2w_convention(camera_convention: str, source_format: str) -> str:
    if camera_convention != "auto":
        return camera_convention
    if source_format == "nerf_transforms":
        return "opengl"
    return "opencv"


def load_cameras_from_json(cameras_json: Path, camera_convention: str) -> Dict[str, Open3DCamera]:
    """
    Supported formats:
      - NeRF/Blender transforms JSON with frames[].transform_matrix
      - Open3D camera trajectory JSON with parameters[].intrinsic/extrinsic
      - Custom {cameras:[{name,width,height,fx,fy,cx,cy,world_to_camera}]}

    With --camera-convention auto:
      - NeRF-style frames[].transform_matrix is treated as OpenGL/Blender c2w.
      - Explicit world_to_camera/w2c/T_cw matrices are used as-is.
      - Custom camera_to_world/c2w/T_wc matrices are treated as OpenCV c2w.
    """
    cameras_json = Path(cameras_json)
    with cameras_json.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    cameras: Dict[str, Open3DCamera] = {}

    # Open3D camera trajectory.
    if isinstance(data, dict) and isinstance(data.get("parameters"), list):
        for index, parameter in enumerate(data["parameters"]):
            intrinsic = parameter["intrinsic"]
            width = int(intrinsic["width"])
            height = int(intrinsic["height"])
            K = matrix3(intrinsic["intrinsic_matrix"], "intrinsic_matrix")
            name = str(parameter.get("name", parameter.get("camera_name", f"{index:04d}")))
            world_to_camera = matrix4(parameter["extrinsic"], "extrinsic")
            cameras[name] = Open3DCamera(
                name=name,
                width=width,
                height=height,
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                world_to_camera=world_to_camera,
            )
        return cameras

    # NeRF / Blender transforms.
    if isinstance(data, dict) and isinstance(data.get("frames"), list):
        top_w = get_optional(data, "w", "width", default=None)
        top_h = get_optional(data, "h", "height", default=None)
        top_fx = get_optional(data, "fl_x", "fx", default=None)
        top_fy = get_optional(data, "fl_y", "fy", default=None)
        top_cx = get_optional(data, "cx", default=None)
        top_cy = get_optional(data, "cy", default=None)
        top_angle_x = get_optional(data, "camera_angle_x", "fov_x", default=None)
        top_angle_y = get_optional(data, "camera_angle_y", "fov_y", default=None)

        for frame_index, frame in enumerate(data["frames"]):
            file_path = str(get_optional(frame, "file_path", "image_path", "name", default=f"{frame_index:04d}"))
            name = str(get_optional(frame, "camera_name", "name", default=camera_name_from_path(file_path)))
            width = int(get_optional(frame, "w", "width", default=top_w))
            height = int(get_optional(frame, "h", "height", default=top_h))

            fx = get_optional(frame, "fl_x", "fx", default=top_fx)
            fy = get_optional(frame, "fl_y", "fy", default=top_fy)
            cx = get_optional(frame, "cx", default=top_cx)
            cy = get_optional(frame, "cy", default=top_cy)
            angle_x = get_optional(frame, "camera_angle_x", "fov_x", default=top_angle_x)
            angle_y = get_optional(frame, "camera_angle_y", "fov_y", default=top_angle_y)

            if fx is None and angle_x is not None:
                fx = 0.5 * width / np.tan(0.5 * float(angle_x))
            if fy is None and angle_y is not None:
                fy = 0.5 * height / np.tan(0.5 * float(angle_y))
            if fy is None and fx is not None:
                fy = fx
            if fx is None and fy is not None:
                fx = fy
            if cx is None:
                cx = 0.5 * (width - 1)
            if cy is None:
                cy = 0.5 * (height - 1)
            if fx is None or fy is None:
                raise ValueError(f"Camera '{name}' is missing focal length in {cameras_json}")

            if "world_to_camera" in frame:
                world_to_camera = matrix4(frame["world_to_camera"], "world_to_camera")
            elif "w2c" in frame:
                world_to_camera = matrix4(frame["w2c"], "w2c")
            elif "T_cw" in frame:
                world_to_camera = matrix4(frame["T_cw"], "T_cw")
            else:
                c2w_raw = get_optional(frame, "transform_matrix", "camera_to_world", "c2w", "T_wc", default=None)
                if c2w_raw is None:
                    raise ValueError(f"Camera '{name}' has no pose matrix in {cameras_json}")
                c2w_convention = resolve_c2w_convention(camera_convention, "nerf_transforms")
                c2w = maybe_opengl_c2w_to_opencv_c2w(
                    matrix4(c2w_raw, "camera_to_world"),
                    c2w_convention,
                )
                world_to_camera = np.linalg.inv(c2w)

            cameras[name] = Open3DCamera(
                name=name,
                width=width,
                height=height,
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                world_to_camera=world_to_camera,
            )
        return cameras

    # Custom list or dict format.
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and isinstance(data.get("cameras"), list):
        entries = data["cameras"]
    elif isinstance(data, dict):
        entries = []
        for name, entry in data.items():
            if isinstance(entry, dict):
                copied = dict(entry)
                copied.setdefault("name", name)
                entries.append(copied)
    else:
        raise ValueError(f"Unsupported camera JSON format: {cameras_json}")

    for index, entry in enumerate(entries):
        name = str(get_optional(entry, "name", "camera_name", "id", default=f"{index:04d}"))
        width = int(get_optional(entry, "width", "w"))
        height = int(get_optional(entry, "height", "h"))
        fx = get_optional(entry, "fx", "fl_x", default=None)
        fy = get_optional(entry, "fy", "fl_y", default=None)
        cx = get_optional(entry, "cx", default=None)
        cy = get_optional(entry, "cy", default=None)

        K = get_optional(entry, "K", "intrinsic", "intrinsics", "intrinsic_matrix", default=None)
        if K is not None and (fx is None or fy is None or cx is None or cy is None):
            K = matrix3(K, "K")
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        if cx is None:
            cx = 0.5 * (width - 1)
        if cy is None:
            cy = 0.5 * (height - 1)
        if fx is None or fy is None:
            raise ValueError(f"Camera '{name}' is missing fx/fy")

        w2c_raw = get_optional(entry, "world_to_camera", "w2c", "T_cw", "extrinsic", default=None)
        if w2c_raw is not None:
            world_to_camera = matrix4(w2c_raw, "world_to_camera")
        else:
            c2w_raw = get_optional(entry, "camera_to_world", "c2w", "T_wc", "pose", default=None)
            if c2w_raw is None:
                raise ValueError(f"Camera '{name}' is missing world_to_camera/camera_to_world")
            c2w_convention = resolve_c2w_convention(camera_convention, "custom")
            c2w = maybe_opengl_c2w_to_opencv_c2w(
                matrix4(c2w_raw, "camera_to_world"),
                c2w_convention,
            )
            world_to_camera = np.linalg.inv(c2w)

        cameras[name] = Open3DCamera(
            name=name,
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            world_to_camera=world_to_camera,
        )

    return cameras


def camera_to_open3d_from_renderer(
    renderer,
    camera_name: str,
    image_shape_hw: Tuple[int, int],
) -> Open3DCamera:
    """
    Best-effort camera introspection. If your C++ renderer does not expose these
    fields to Python, pass --cameras-json instead.
    """
    h, w = int(image_shape_hw[0]), int(image_shape_hw[1])

    cam = None
    cameras = get_attr_or_key(renderer, "cameras", "camera_map", "_cameras", default=None)
    if isinstance(cameras, dict) and camera_name in cameras:
        cam = cameras[camera_name]
    elif hasattr(renderer, "get_camera"):
        cam = renderer.get_camera(camera_name)
    elif hasattr(renderer, "camera"):
        cam = renderer.camera

    if cam is None:
        raise RuntimeError(
            "Could not introspect camera from renderer. Pass --cameras-json."
        )

    width = int(get_attr_or_key(cam, "width", "image_width", "W", default=w))
    height = int(get_attr_or_key(cam, "height", "image_height", "H", default=h))

    fx = get_attr_or_key(cam, "fx", "focal_x", default=None)
    fy = get_attr_or_key(cam, "fy", "focal_y", default=None)
    cx = get_attr_or_key(cam, "cx", "principal_x", default=None)
    cy = get_attr_or_key(cam, "cy", "principal_y", default=None)

    K = get_attr_or_key(cam, "K", "intrinsic", "intrinsics", "camera_matrix", default=None)
    if K is not None and (fx is None or fy is None or cx is None or cy is None):
        K = matrix3(K, "K")
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    if fx is None or fy is None:
        fov_x = get_attr_or_key(cam, "fov_x", "FoVx", "fovx", default=None)
        fov_y = get_attr_or_key(cam, "fov_y", "FoVy", "fovy", default=None)
        if fov_x is not None:
            fx = 0.5 * width / np.tan(0.5 * float(fov_x))
        if fov_y is not None:
            fy = 0.5 * height / np.tan(0.5 * float(fov_y))

    if fy is None and fx is not None:
        fy = fx
    if fx is None and fy is not None:
        fx = fy
    if cx is None:
        cx = 0.5 * (width - 1)
    if cy is None:
        cy = 0.5 * (height - 1)
    if fx is None or fy is None:
        raise RuntimeError(f"Could not infer intrinsics for camera '{camera_name}'. Pass --cameras-json.")

    w2c = get_attr_or_key(
        cam,
        "world_to_camera",
        "world_view_transform",
        "w2c",
        "T_cw",
        "extrinsic",
        "view_matrix",
        default=None,
    )
    if w2c is not None:
        world_to_camera = matrix4(w2c, "world_to_camera")
    else:
        c2w = get_attr_or_key(cam, "camera_to_world", "c2w", "T_wc", "pose", default=None)
        if c2w is None:
            raise RuntimeError(f"Could not infer extrinsics for camera '{camera_name}'. Pass --cameras-json.")
        world_to_camera = np.linalg.inv(matrix4(c2w, "camera_to_world"))

    return Open3DCamera(
        name=camera_name,
        width=width,
        height=height,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        world_to_camera=world_to_camera,
    )


# -----------------------------------------------------------------------------
# Open3D TSDF fusion
# -----------------------------------------------------------------------------


def sanitize_color(rgb_float: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_float, dtype=np.float32, order="C")
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)

    if rgb.max(initial=0.0) <= 1.5:
        rgb = 255.0 * np.clip(rgb, 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 255.0)

    return np.ascontiguousarray(rgb.astype(np.uint8, copy=False))


def sanitize_depth(depth: np.ndarray, min_depth: float, depth_trunc: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32, order="C").copy()
    valid = np.isfinite(depth) & (depth > float(min_depth)) & (depth < float(depth_trunc))
    depth[~valid] = 0.0
    return np.ascontiguousarray(depth)


def convert_ray_length_to_z_depth(depth: np.ndarray, camera: Open3DCamera) -> np.ndarray:
    """Use only if median_depth is Euclidean ray distance instead of camera z-depth."""
    h, w = depth.shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    x = (xx - float(camera.cx)) / float(camera.fx)
    y = (yy - float(camera.cy)) / float(camera.fy)
    dir_z = 1.0 / np.sqrt(x * x + y * y + 1.0)
    return np.ascontiguousarray(depth * dir_z.astype(np.float32))


def estimate_depth_trunc(
    forward_out: Dict[str, dict],
    camera_names: Sequence[str],
    depth_key: str,
    quantile: float,
    scale: float,
) -> float:
    estimates: List[float] = []
    for camera_name in camera_names:
        depth = get_forward_depth(forward_out, camera_name, depth_key=depth_key)
        valid = depth[np.isfinite(depth) & (depth > 0.0)]
        if valid.size:
            estimates.append(float(np.quantile(valid, quantile)))

    if not estimates:
        raise RuntimeError("Could not auto-estimate depth_trunc: no positive finite rendered depths.")

    return float(scale * max(estimates))


def post_process_mesh(
    mesh: o3d.geometry.TriangleMesh,
    cluster_to_keep: int,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh

    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if cluster_n_triangles.size == 0:
        return mesh

    keep_count = min(int(cluster_to_keep), int(cluster_n_triangles.size))
    keep_clusters = np.argsort(cluster_n_triangles)[-keep_count:]
    remove_mask = ~np.isin(triangle_clusters, keep_clusters)

    mesh.remove_triangles_by_mask(remove_mask.tolist())
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


def fuse_rendered_depths(
    renderer,
    forward_out: Dict[str, dict],
    camera_names: Sequence[str],
    *,
    cameras_json: Optional[Path],
    camera_convention: str,
    open3d_extrinsic: str,
    depth_key: str,
    depth_trunc: float,
    voxel_size: float,
    sdf_trunc: float,
    min_depth: float,
    depth_is_ray_length: bool,
    save_rerendered_frames: bool,
    frames_dir: Path,
) -> o3d.geometry.TriangleMesh:
    json_cameras: Optional[Dict[str, Open3DCamera]] = None
    if cameras_json is not None:
        json_cameras = load_cameras_from_json(cameras_json, camera_convention=camera_convention)
        print(f"Loaded {len(json_cameras)} Open3D cameras from {cameras_json}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    if save_rerendered_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    used = 0
    for camera_name in camera_names:
        if camera_name not in forward_out:
            print(f"Skipping camera '{camera_name}': missing from renderer output")
            continue

        depth = get_forward_depth(forward_out, camera_name, depth_key=depth_key)
        color = sanitize_color(get_forward_rgb(forward_out, camera_name))

        if json_cameras is not None:
            if camera_name not in json_cameras:
                print(f"Skipping camera '{camera_name}': no matching camera in --cameras-json")
                continue
            camera = json_cameras[camera_name]
        else:
            camera = camera_to_open3d_from_renderer(renderer, camera_name, depth.shape)

        if depth.shape != (camera.height, camera.width):
            raise RuntimeError(
                f"Camera/depth size mismatch for '{camera_name}': "
                f"depth={depth.shape}, camera={(camera.height, camera.width)}"
            )

        if depth_is_ray_length:
            depth = convert_ray_length_to_z_depth(depth, camera)

        depth = sanitize_depth(depth, min_depth=min_depth, depth_trunc=depth_trunc)
        if np.count_nonzero(depth) == 0:
            print(f"Skipping camera '{camera_name}': no valid depth samples after truncation")
            continue

        if save_rerendered_frames:
            np.save(frames_dir / f"median_depth_rerendered_{camera_name}.npy", depth)
            o3d.io.write_image(
                str(frames_dir / f"render_rerendered_{camera_name}.png"),
                o3d.geometry.Image(color),
            )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth.astype(np.float32, copy=False)),
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )

        if open3d_extrinsic == "world-to-camera":
            extrinsic = camera.world_to_camera
        elif open3d_extrinsic == "camera-to-world":
            extrinsic = np.linalg.inv(camera.world_to_camera)
        else:
            raise ValueError(f"Unsupported --open3d-extrinsic: {open3d_extrinsic}")

        volume.integrate(
            rgbd,
            camera.intrinsic(),
            extrinsic.astype(np.float64, copy=False),
        )
        used += 1
        print(f"Integrated freshly rendered camera '{camera_name}'")

    if used == 0:
        raise RuntimeError("TSDF integration used zero cameras.")

    print(f"Integrated {used} freshly rendered RGB-D frames")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-render points_final.ply with pale.Renderer and extract an Open3D TSDF mesh."
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../Assets/OptimizationOutput"),
        help="Root containing run folders, a specific run folder, or a points_final.ply file.",
    )
    parser.add_argument(
        "--pointcloud-path-mode",
        choices=["auto", "absolute", "relative-to-assets"],
        default="auto",
        help="How to pass points_final.ply to pale.Renderer. Default: relative to assets_root if possible.",
    )

    parser.add_argument(
        "--camera-names",
        type=str,
        default=None,
        help="Comma-separated camera names. Default: get_training_camera_names(renderer).",
    )
    parser.add_argument(
        "--use-all-cameras",
        action="store_true",
        help="Use get_all_camera_names(renderer) if available instead of training cameras.",
    )
    parser.add_argument(
        "--cameras-json",
        type=Path,
        default=None,
        help="Optional camera metadata JSON for Open3D intrinsics/extrinsics if renderer introspection fails.",
    )
    parser.add_argument(
        "--camera-convention",
        choices=["auto", "opencv", "opengl"],
        default="auto",
        help="Convention used by camera_to_world matrices in --cameras-json. auto treats NeRF transforms.json as OpenGL/Blender c2w. world_to_camera matrices are used as-is.",
    )
    parser.add_argument(
        "--open3d-extrinsic",
        choices=["world-to-camera", "camera-to-world"],
        default="world-to-camera",
        help="Matrix convention passed to Open3D volume.integrate. Open3D examples pass inverse camera pose, i.e. world-to-camera.",
    )

    parser.add_argument("--depth-key", type=str, default="median_depth")
    parser.add_argument(
        "--depth-is-ray-length",
        action="store_true",
        help="Convert Euclidean ray distance to camera z-depth before TSDF fusion.",
    )
    parser.add_argument("--min-depth", type=float, default=1.0e-6)
    parser.add_argument("--depth-trunc", type=float, default=-1.0)
    parser.add_argument("--depth-trunc-quantile", type=float, default=0.995)
    parser.add_argument("--depth-trunc-scale", type=float, default=1.10)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.002,
        help="TSDF voxel size. 2DGS uses 0.004.",
    )
    parser.add_argument(
        "--mesh-res",
        type=int,
        default=1024,
        help="Only used when --voxel-size <= 0; then voxel_size = depth_trunc / mesh_res.",
    )
    parser.add_argument(
        "--sdf-trunc",
        type=float,
        default=0.02,
        help="TSDF truncation threshold. 2DGS uses 0.02.",
    )
    parser.add_argument("--num-cluster", type=int, default=50)

    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--raw-name", type=str, default="fuse.ply")
    parser.add_argument("--post-name", type=str, default="fuse_post.ply")
    parser.add_argument("--save-rerendered-frames", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    points_final_ply = find_latest_points_ply(args.output_root)
    run_dir = points_final_ply.parent
    run_config = load_run_config(run_dir)

    renderer = create_renderer_from_run_config(
        run_config,
        points_final_ply=points_final_ply,
        pointcloud_path_mode=str(args.pointcloud_path_mode),
    )

    if args.camera_names is not None:
        camera_names = [name.strip() for name in args.camera_names.split(",") if name.strip()]
    elif args.use_all_cameras and get_all_camera_names is not None:
        camera_names = normalize_camera_names(get_all_camera_names(renderer))
    else:
        camera_names = normalize_camera_names(get_training_camera_names(renderer))

    if len(camera_names) == 0:
        raise RuntimeError("No cameras selected for rendering.")

    print(f"Rendering {len(camera_names)} cameras: {camera_names}")
    forward_out = renderer.render_forward()

    if args.depth_trunc > 0.0:
        depth_trunc = float(args.depth_trunc)
    else:
        depth_trunc = estimate_depth_trunc(
            forward_out,
            camera_names,
            depth_key=str(args.depth_key),
            quantile=float(args.depth_trunc_quantile),
            scale=float(args.depth_trunc_scale),
        )

    if args.voxel_size > 0.0:
        voxel_size = float(args.voxel_size)
    else:
        if args.mesh_res <= 0:
            raise ValueError("--mesh-res must be positive when --voxel-size is not set")
        voxel_size = float(depth_trunc) / float(args.mesh_res)

    if args.sdf_trunc > 0.0:
        sdf_trunc = float(args.sdf_trunc)
    else:
        sdf_trunc = 5.0 * voxel_size

    print(
        "TSDF settings: "
        f"depth_key={args.depth_key}, "
        f"depth_trunc={depth_trunc:.6g}, "
        f"voxel_size={voxel_size:.6g}, "
        f"sdf_trunc={sdf_trunc:.6g}, "
        f"camera_convention={args.camera_convention}, "
        f"open3d_extrinsic={args.open3d_extrinsic}"
    )

    mesh_dir = args.mesh_dir if args.mesh_dir is not None else run_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    mesh = fuse_rendered_depths(
        renderer,
        forward_out,
        camera_names,
        cameras_json=args.cameras_json,
        camera_convention=str(args.camera_convention),
        open3d_extrinsic=str(args.open3d_extrinsic),
        depth_key=str(args.depth_key),
        depth_trunc=float(depth_trunc),
        voxel_size=float(voxel_size),
        sdf_trunc=float(sdf_trunc),
        min_depth=float(args.min_depth),
        depth_is_ray_length=bool(args.depth_is_ray_length),
        save_rerendered_frames=bool(args.save_rerendered_frames),
        frames_dir=mesh_dir / "rerendered_frames",
    )

    raw_path = mesh_dir / args.raw_name
    o3d.io.write_triangle_mesh(str(raw_path), mesh)
    print(f"mesh saved at {raw_path}")

    mesh_post = post_process_mesh(mesh, cluster_to_keep=int(args.num_cluster))
    post_path = mesh_dir / args.post_name
    o3d.io.write_triangle_mesh(str(post_path), mesh_post)
    print(f"post-processed mesh saved at {post_path}")


if __name__ == "__main__":
    main()
