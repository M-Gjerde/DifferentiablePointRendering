from __future__ import annotations

import argparse
import colorsys
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image

import pale


def parse_run_timestamp(run_dir_name: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", run_dir_name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def find_run_dir_by_index(optimization_output_root: Path, run_index: int) -> Path:
    if run_index < 0:
        raise ValueError(f"--index must be >= 0, got {run_index}")

    if not optimization_output_root.exists():
        raise FileNotFoundError(f"OptimizationOutput folder does not exist: {optimization_output_root}")

    candidate_run_dirs: list[dict] = []

    for child in optimization_output_root.iterdir():
        if not child.is_dir():
            continue

        metrics_csv_path = child / "metrics.csv"
        if not metrics_csv_path.exists():
            continue

        candidate_run_dirs.append(
            {
                "run_dir": child,
                "parsed_timestamp": parse_run_timestamp(child.name),
                "modified_time": metrics_csv_path.stat().st_mtime,
            }
        )

    if not candidate_run_dirs:
        raise FileNotFoundError(
            f"No run folders with metrics.csv found under: {optimization_output_root}"
        )

    candidate_run_dirs.sort(
        key=lambda item: (
            item["parsed_timestamp"] is not None,
            item["parsed_timestamp"] if item["parsed_timestamp"] is not None else datetime.min,
            item["modified_time"],
        ),
        reverse=True,
    )

    if run_index >= len(candidate_run_dirs):
        available_runs = [
            f"[{candidate_index}] {candidate['run_dir'].name}"
            for candidate_index, candidate in enumerate(candidate_run_dirs)
        ]

        raise IndexError(
            f"--index {run_index} is out of range. "
            f"Found {len(candidate_run_dirs)} run folders with metrics.csv.\n"
            "Available runs:\n" + "\n".join(available_runs)
        )

    return candidate_run_dirs[run_index]["run_dir"]


def load_run_config(run_config_path: Path) -> dict:
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json: {run_config_path}")
    with open(run_config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps or not np.isfinite(n):
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def normalize_quaternions_wxyz(q: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32, order="C")

    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quaternion array with shape (N,4), got {q.shape}")

    norms = np.linalg.norm(q, axis=1, keepdims=True)
    finite = np.isfinite(q).all(axis=1, keepdims=True)
    valid = finite & (norms > eps)

    fallback = np.zeros_like(q, dtype=np.float32)
    fallback[:, 0] = 1.0

    qn = q / np.maximum(norms, eps)
    qn = np.where(valid, qn, fallback)

    # Canonical sign: q and -q are the same rotation.
    qn = np.where(qn[:, 0:1] < 0.0, -qn, qn)

    return qn.astype(np.float32, copy=False)


def quaternion_from_rotation_matrix_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    R = np.asarray(rotation_matrix, dtype=np.float64)

    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    q = np.array([[qw, qx, qy, qz]], dtype=np.float32)
    return normalize_quaternions_wxyz(q)[0]


def quaternion_from_tangent_frame_wxyz(tangent_u: np.ndarray, tangent_v: np.ndarray) -> np.ndarray:
    u = normalize(np.asarray(tangent_u, dtype=np.float32))
    v_raw = np.asarray(tangent_v, dtype=np.float32)

    v = v_raw - float(np.dot(v_raw, u)) * u
    if float(np.linalg.norm(v)) < 1.0e-8:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(float(u[1])) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v = helper - float(np.dot(helper, u)) * u

    v = normalize(v)
    w = normalize(np.cross(u, v))

    # Columns are the local frame axes: R = [tu tv tw].
    R = np.column_stack((u, v, w))
    return quaternion_from_rotation_matrix_wxyz(R)


def orthonormal_frame_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normalize(normal)

    if abs(float(n[2])) < 0.9:
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    tu = np.cross(helper, n)
    if np.linalg.norm(tu) < 1e-8:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tu = np.cross(helper, n)

    tu = normalize(tu)
    tv = normalize(np.cross(n, tu))

    # Ensure tanU x tanV points along n.
    if float(np.dot(np.cross(tu, tv), n)) < 0.0:
        tv = -tv

    return tu.astype(np.float32), tv.astype(np.float32)


def quaternion_from_normal_wxyz(normal: np.ndarray) -> np.ndarray:
    tu, tv = orthonormal_frame_from_normal(normal)
    return quaternion_from_tangent_frame_wxyz(tu, tv)


def orbit_position_on_yz_arc(
    t: float,
    radius: float,
    orbit_degrees: float = 180.0,
) -> np.ndarray:
    theta = math.radians(orbit_degrees) * t
    y = radius * math.cos(theta)
    z = radius * math.sin(theta)
    return np.array([0.0, y, z], dtype=np.float32)


def orbit_position_on_xz_arc(
    t: float,
    radius: float,
    orbit_degrees: float = 180.0,
) -> np.ndarray:
    theta = math.radians(orbit_degrees) * t
    x = radius * math.cos(theta)
    z = radius * math.sin(theta)
    return np.array([-x, 0.0, z], dtype=np.float32)


def color_ramp_rgb(
    t: float,
    variation: float = 0.0,
    saturation: float = 0.95,
    value: float = 1.0,
    base_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    variation = max(0.0, float(variation))

    if variation <= 1e-8:
        return np.array(base_color, dtype=np.float32)

    hue_center = 0.08
    hue_span = 0.86 * variation
    hue = (hue_center + hue_span * t) % 1.0

    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    rgb = np.array(rgb, dtype=np.float32)

    blend = min(variation, 1.0)
    base = np.array(base_color, dtype=np.float32)
    out = (1.0 - blend) * base + blend * rgb

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def save_rgb_png(output_path: Path, rgb: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(image_uint8, mode="RGB").save(output_path)


def collect_existing_frame_paths(
    frames_dir: Path,
    requested_frame_count: int | None,
) -> list[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")

    frame_paths = sorted(frames_dir.glob("frame_*.png"))

    if not frame_paths:
        raise FileNotFoundError(f"No existing frame_*.png files found in: {frames_dir}")

    if requested_frame_count is not None:
        if requested_frame_count <= 0:
            raise ValueError("--frames must be positive when provided.")

        if len(frame_paths) < requested_frame_count:
            raise FileNotFoundError(
                f"--frames requested {requested_frame_count} frames, "
                f"but only found {len(frame_paths)} existing frames in: {frames_dir}"
            )

        frame_paths = frame_paths[:requested_frame_count]

    return frame_paths


def build_gif_from_pngs(frame_paths: Sequence[Path], output_gif_path: Path, fps: float) -> None:
    duration_sec = 1.0 / max(fps, 1e-6)
    with imageio.get_writer(output_gif_path, mode="I", duration=duration_sec, loop=0) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))


def get_forward_rgb(rendered_images: dict, camera_name: str) -> np.ndarray:
    camera_output = rendered_images[camera_name]

    if isinstance(camera_output, dict):
        if "image" not in camera_output:
            raise KeyError(
                f"Rendered output for camera '{camera_name}' is a dict without 'image'. "
                f"Keys: {list(camera_output.keys())}"
            )
        image_numpy = np.asarray(camera_output["image"], dtype=np.float32)
    else:
        image_numpy = np.asarray(camera_output, dtype=np.float32)

    if image_numpy.ndim != 3:
        raise RuntimeError(f"Unexpected image shape for camera '{camera_name}': {image_numpy.shape}")

    return np.clip(image_numpy[..., :3], 0.0, 1.0)


def get_camera_names_from_renderer(renderer: pale.Renderer) -> list[str]:
    names = list(renderer.get_camera_names())
    if not names:
        raise RuntimeError("Renderer returned no camera names.")
    return names


def fetch_parameters(renderer: pale.Renderer) -> dict[str, np.ndarray]:
    params = renderer.get_point_parameters()
    return {
        "position": np.asarray(params["position"], dtype=np.float32, order="C"),
        "rotation": normalize_quaternions_wxyz(np.asarray(params["rotation"], dtype=np.float32, order="C")),
        "scale": np.asarray(params["scale"], dtype=np.float32, order="C"),
        "albedo": np.asarray(params["albedo"], dtype=np.float32, order="C"),
        "opacity": np.asarray(params["opacity"], dtype=np.float32, order="C"),
        "beta": np.asarray(params["beta"], dtype=np.float32, order="C"),
        "power": np.asarray(params["power"], dtype=np.float32, order="C"),
    }


def apply_point_parameters(renderer: pale.Renderer, params: dict[str, np.ndarray]) -> None:
    renderer.apply_point_optimization(
        {
            "position": np.asarray(params["position"], dtype=np.float32, order="C"),
            "rotation": normalize_quaternions_wxyz(np.asarray(params["rotation"], dtype=np.float32, order="C")),
            "scale": np.asarray(params["scale"], dtype=np.float32, order="C"),
            "albedo": np.asarray(params["albedo"], dtype=np.float32, order="C"),
            "opacity": np.asarray(params["opacity"], dtype=np.float32, order="C"),
            "beta": np.asarray(params["beta"], dtype=np.float32, order="C"),
            "power": np.asarray(params["power"], dtype=np.float32, order="C"),
        }
    )


def add_light_point(
    renderer: pale.Renderer,
    *,
    scale_u: float,
    scale_v: float,
    opacity: float,
    beta: float,
) -> int:
    animated_position = np.array([[0.0, 5.0, 0.0]], dtype=np.float32)
    animated_normal = normalize(-animated_position[0])
    animated_rotation = quaternion_from_normal_wxyz(animated_normal).reshape(1, 4).astype(np.float32)

    static_position = np.array([[0.0, 0.0, 3.0]], dtype=np.float32)
    static_tu = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    static_tv = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    static_rotation = quaternion_from_tangent_frame_wxyz(static_tu, static_tv).reshape(1, 4).astype(np.float32)

    renderer.add_points(
        {
            "new": {
                "position": animated_position,
                "rotation": animated_rotation,
                "scale": np.array([[scale_u, scale_v]], dtype=np.float32),
                "albedo": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
                "opacity": np.array([opacity], dtype=np.float32),
                "beta": np.array([beta], dtype=np.float32),
                "power": np.array([0.0], dtype=np.float32),
            }
        }
    )

    renderer.add_points(
        {
            "new": {
                "position": static_position,
                "rotation": static_rotation,
                "scale": np.array([[scale_u, scale_v]], dtype=np.float32),
                "albedo": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
                "opacity": np.array([opacity], dtype=np.float32),
                "beta": np.array([beta], dtype=np.float32),
                "power": np.array([100.0], dtype=np.float32),
            }
        }
    )

    renderer.rebuild_bvh()

    params = fetch_parameters(renderer)
    return int(params["position"].shape[0] - 2)


def zero_existing_lights(renderer: pale.Renderer) -> None:
    params = fetch_parameters(renderer)
    params["power"][:] = 0.0
    apply_point_parameters(renderer, params)
    renderer.rebuild_bvh()


def update_light_point_in_params(
    params: dict[str, np.ndarray],
    light_index: int,
    position: np.ndarray,
    color_rgb: np.ndarray,
    power: float,
    scale_u: float,
    scale_v: float,
    opacity: float,
    beta: float,
) -> None:
    normal = normalize(-position)
    rotation = quaternion_from_normal_wxyz(normal)

    params["position"][light_index] = position.astype(np.float32)
    params["rotation"][light_index] = rotation.astype(np.float32)
    params["scale"][light_index, 0] = scale_u
    params["scale"][light_index, 1] = scale_v
    params["albedo"][light_index] = color_rgb.astype(np.float32)
    params["opacity"][light_index] = opacity
    params["beta"][light_index] = beta
    params["power"][light_index] = power


def get_camera_output_paths(
    animation_output_dir: Path,
    camera_name: str,
    gif_name: str,
    render_all_cameras: bool,
) -> tuple[Path, Path]:
    if render_all_cameras:
        return (
            animation_output_dir / "frames" / camera_name,
            animation_output_dir / f"{camera_name}.gif",
        )

    camera_output_dir = animation_output_dir / camera_name
    return camera_output_dir / "frames", camera_output_dir / gif_name


def find_existing_camera_output_dirs(animation_output_dir: Path) -> list[Path]:
    if not animation_output_dir.exists():
        raise FileNotFoundError(
            f"Animation output directory does not exist: {animation_output_dir}"
        )

    camera_output_dirs = sorted(
        child
        for child in animation_output_dir.iterdir()
        if child.is_dir() and (child / "frames").is_dir()
    )

    if not camera_output_dirs:
        raise FileNotFoundError(
            "No per-camera frame folders found. Expected folders such as "
            f"'{animation_output_dir / 'view_camera_00' / 'frames'}'."
        )

    return camera_output_dirs


def select_camera_names(
    available_camera_names: Sequence[str],
    requested_camera_name: str,
    render_all_cameras: bool,
) -> list[str]:
    if render_all_cameras:
        return list(available_camera_names)

    if requested_camera_name not in available_camera_names:
        raise ValueError(
            f"Unknown camera name '{requested_camera_name}'. "
            f"Available cameras: {list(available_camera_names)}"
        )

    return [requested_camera_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a moving emissive surfel orbit and save PNG frames + GIF."
    )

    parser.add_argument("--optimization-output-root", type=Path, default=Path("OptimizationOutput"))
    parser.add_argument("--run-dir", type=Path, default=None)

    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help=(
            "Number of frames to render. If omitted, render mode uses 60 frames. "
            "With --skip-render, omitted means use all existing frame_*.png files."
        ),
    )

    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip renderer setup and frame rendering. Rebuild the GIF from existing PNG frames only.",
    )

    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--power", type=float, default=100.0)
    parser.add_argument("--orbit", type=str, default="y", choices=["x", "y"])
    parser.add_argument("--scale-u", type=float, default=0.05)
    parser.add_argument("--scale-v", type=float, default=0.05)
    parser.add_argument("--opacity", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=-100.0)
    parser.add_argument("--output-subdir", type=str, default="orbit_light_animation")
    parser.add_argument("--gif-name", type=str, default="orbit_light.gif")
    parser.add_argument("--renderer-forward-passes", type=int, default=1)
    parser.add_argument("--renderer-primal-shadow-rays", type=int, default=1)

    parser.add_argument(
        "--color-variation",
        type=float,
        default=0.0,
        help="How aggressively the light color changes over time. 0 = constant white, 1 = full sweep.",
    )
    parser.add_argument(
        "--color-saturation",
        type=float,
        default=0.95,
        help="HSV saturation used for animated light colors.",
    )
    parser.add_argument(
        "--color-value",
        type=float,
        default=1.0,
        help="HSV value used for animated light colors.",
    )
    parser.add_argument(
        "--scene-xml",
        type=str,
        default="../Assets/view_cameras.xml",
        help=(
            "Optional scene XML override. If omitted, uses scene_xml from run_config.json. "
            "Example: cbox_custom_alt_views.xml"
        ),
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="view_camera",
        help="Single camera name to render when --all is not used.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Render every camera in the scene. Each camera gets its own frames/ folder "
            "and GIF under --output-subdir."
        ),
    )
    parser.add_argument(
        "--rebuild-every-frame",
        action="store_true",
        default=True,
        help="Call rebuild_bvh() after moving the light each frame. Safer if BVH depends on point positions.",
    )
    parser.add_argument(
        "--orbit-degrees",
        type=float,
        default=180.0,
        help="Angular sweep of the light path in degrees. 180 = dome arc, 360 = full circle.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help=(
            "Zero-based index of the run to use when --run-dir is omitted. "
            "0 = latest, 1 = second latest, 2 = third latest, ..."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
    else:
        run_dir = find_run_dir_by_index(
            optimization_output_root=args.optimization_output_root.resolve(),
            run_index=args.index,
        )

    animation_output_dir = run_dir / args.output_subdir

    if args.skip_render:
        if args.all:
            camera_output_dirs = find_existing_camera_output_dirs(animation_output_dir)
            completed_camera_names: list[str] = []

            for camera_output_dir in camera_output_dirs:
                camera_name = camera_output_dir.name
                frames_dir = camera_output_dir / "frames"
                output_gif_path = camera_output_dir / args.gif_name
                frame_paths = collect_existing_frame_paths(
                    frames_dir=frames_dir,
                    requested_frame_count=args.frames,
                )

                build_gif_from_pngs(frame_paths, output_gif_path, fps=args.fps)
                completed_camera_names.append(camera_name)

                print(
                    f"[GIF] camera={camera_name} "
                    f"frames={len(frame_paths)} -> {output_gif_path}"
                )

            print()
            print("Done.")
            print("Skipped rendering and rebuilt GIFs from existing PNG frames.")
            print(f"Run folder : {run_dir}")
            print(f"Camera count: {len(completed_camera_names)}")
            print(f"Cameras    : {', '.join(completed_camera_names)}")
            print(f"FPS        : {args.fps}")
            print(f"Output root: {animation_output_dir}")
            return

        frames_dir, output_gif_path = get_camera_output_paths(
            animation_output_dir=animation_output_dir,
            camera_name=args.camera_name,
            gif_name=args.gif_name,
        )

        # Backward compatibility with the old single-camera output layout.
        if not frames_dir.exists() and (animation_output_dir / "frames").exists():
            frames_dir = animation_output_dir / "frames"
            output_gif_path = animation_output_dir / args.gif_name
            print(
                "Using legacy single-camera frame layout: "
                f"{frames_dir}"
            )

        frame_paths = collect_existing_frame_paths(
            frames_dir=frames_dir,
            requested_frame_count=args.frames,
        )

        output_gif_path.parent.mkdir(parents=True, exist_ok=True)
        build_gif_from_pngs(frame_paths, output_gif_path, fps=args.fps)

        print()
        print("Done.")
        print("Skipped rendering.")
        print(f"Run folder : {run_dir}")
        print(f"Camera     : {args.camera_name}")
        print(f"Frames     : {frames_dir}")
        print(f"Frame count: {len(frame_paths)}")
        print(f"FPS        : {args.fps}")
        print(f"GIF        : {output_gif_path}")
        return

    frame_count = 30 if args.frames is None else args.frames
    if frame_count <= 0:
        raise ValueError("--frames must be positive.")

    run_config = load_run_config(run_dir / "run_config.json")

    renderer_settings = dict(run_config["renderer_settings"])
    # renderer_settings["forward_passes"] = int(args.renderer_forward_passes)
    # renderer_settings["primal_shadow_rays"] = int(args.renderer_primal_shadow_rays)

    assets_root = Path(run_config["assets_root"])
    scene_xml = args.scene_xml if args.scene_xml is not None else run_config["scene_xml"]
    points_final_ply_path = run_dir / "points_final.ply"

    if not points_final_ply_path.exists():
        raise FileNotFoundError(f"Missing points_final.ply: {points_final_ply_path}")

    renderer = pale.Renderer(
        str(assets_root),
        str(scene_xml),
        str(points_final_ply_path),
        renderer_settings,
    )

    available_camera_names = get_camera_names_from_renderer(renderer)
    camera_names = select_camera_names(
        available_camera_names=available_camera_names,
        requested_camera_name=args.camera_name,
        render_all_cameras=args.all,
    )

    if args.all:
        print(
            f"Rendering {len(camera_names)} cameras: "
            f"{', '.join(camera_names)}"
        )
    else:
        print(f"Rendering camera: {camera_names[0]}")

    frames_dirs: dict[str, Path] = {}
    gif_paths: dict[str, Path] = {}
    frame_paths_by_camera: dict[str, list[Path]] = {}

    for camera_name in camera_names:
        frames_dir, output_gif_path = get_camera_output_paths(
            animation_output_dir=animation_output_dir,
            camera_name=camera_name,
            gif_name=args.gif_name,
            render_all_cameras=args.all,
        )
        frames_dir.mkdir(parents=True, exist_ok=True)

        frames_dirs[camera_name] = frames_dir
        gif_paths[camera_name] = output_gif_path
        frame_paths_by_camera[camera_name] = []

    zero_existing_lights(renderer)

    light_index = add_light_point(
        renderer,
        scale_u=args.scale_u,
        scale_v=args.scale_v,
        opacity=args.opacity,
        beta=args.beta,
    )

    params = fetch_parameters(renderer)

    # Outer loop is the light animation. This updates the point parameters and BVH once
    # per light position, then renders that identical lighting state from every camera.
    for frame_index in range(frame_count):
        t = 0.0 if frame_count <= 1 else frame_index / float(frame_count - 1)

        if args.orbit == "y":
            light_position = orbit_position_on_yz_arc(
                t=t,
                radius=args.radius,
                orbit_degrees=args.orbit_degrees,
            )
        elif args.orbit == "x":
            light_position = orbit_position_on_xz_arc(
                t=t,
                radius=args.radius,
                orbit_degrees=args.orbit_degrees,
            )
        else:
            raise ValueError(f"Unknown orbit axis: {args.orbit}")

        light_color = color_ramp_rgb(
            t,
            variation=args.color_variation,
            saturation=args.color_saturation,
            value=args.color_value,
        )

        update_light_point_in_params(
            params=params,
            light_index=light_index,
            position=light_position,
            color_rgb=light_color,
            power=args.power,
            scale_u=args.scale_u,
            scale_v=args.scale_v,
            opacity=args.opacity,
            beta=args.beta,
        )

        apply_point_parameters(renderer, params)

        if args.rebuild_every_frame:
            renderer.rebuild_bvh()

        for camera_name in camera_names:
            rendered_images = renderer.render_forward(camera_name)
            rgb = get_forward_rgb(rendered_images, camera_name)

            frame_png_path = frames_dirs[camera_name] / f"frame_{frame_index:04d}.png"
            save_rgb_png(frame_png_path, rgb)
            frame_paths_by_camera[camera_name].append(frame_png_path)

            print(
                f"[{frame_index + 1:03d}/{frame_count:03d}] "
                f"camera={camera_name} "
                f"pos=({light_position[0]:.3f}, {light_position[1]:.3f}, {light_position[2]:.3f}) "
                f"color=({light_color[0]:.3f}, {light_color[1]:.3f}, {light_color[2]:.3f}) "
                f"-> {frame_png_path.name}"
            )

    for camera_name in camera_names:
        build_gif_from_pngs(
            frame_paths_by_camera[camera_name],
            gif_paths[camera_name],
            fps=args.fps,
        )

    print()
    print("Done.")
    print(f"Run folder : {run_dir}")
    print(f"Run index  : {args.index if args.run_dir is None else 'explicit --run-dir'}")
    print(f"Camera count: {len(camera_names)}")
    print(f"Frame count: {frame_count}")
    print(f"FPS        : {args.fps}")
    print(f"Output root: {animation_output_dir}")

    for camera_name in camera_names:
        print(f"  {camera_name}: {gif_paths[camera_name]}")


if __name__ == "__main__":
    main()
