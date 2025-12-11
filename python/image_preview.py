#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live preview for optimization run.\n"
            "Expected structure in --output-path:\n"
            "  run_dir/\n"
            "    camera1/\n"
            "      render/0001_render.png, ...   (or PNGs directly in camera1/)\n"
            "    camera2/\n"
            "      render/0001_render.png, ...\n"
            "    render_target_camera1.png\n"
            "    render_target_camera2.png\n"
        )
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to the optimization output run directory.",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=200,
        help="Refresh interval in milliseconds (default: 200).",
    )
    return parser.parse_args()


def load_image_rgb(image_path: Path, wait_time: float = 0.05) -> Optional[np.ndarray]:
    """
    Load an RGB image safely, skipping if the file is still being written.
    """
    if not image_path.exists():
        return None

    size1 = image_path.stat().st_size
    time.sleep(wait_time)
    if not image_path.exists():
        return None
    size2 = image_path.stat().st_size

    if size1 != size2:
        # File is still being written; skip this frame
        return None

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def discover_camera_dirs(run_dir: Path) -> List[Path]:
    """
    Discover camera directories under run_dir.
    Any immediate subdirectory is treated as a camera folder.
    """
    if not run_dir.is_dir():
        raise RuntimeError(f"output-path '{run_dir}' must be a directory.")

    camera_dirs: List[Path] = []
    for entry in sorted(run_dir.iterdir()):
        if entry.is_dir():
            camera_dirs.append(entry)
    return camera_dirs


def get_latest_render_image_path(render_directory: Path) -> Optional[Path]:
    """
    Returns the latest PNG found in render_directory, or None if none exist.
    """
    if not render_directory.exists():
        return None

    render_files: List[Path] = sorted(render_directory.glob("*.png"))
    if not render_files:
        return None
    return render_files[-1]


def main() -> None:
    args = parse_arguments()

    run_dir: Path = args.output_path
    refresh_ms: int = args.refresh_ms

    if not run_dir.exists():
        raise RuntimeError(f"Output path '{run_dir}' does not exist.")

    # Layout configuration
    WINDOW_WIDTH = 500
    WINDOW_HEIGHT = 500
    WINDOWS_PER_ROW = 6
    ROW_OFFSET_PIXELS = WINDOW_HEIGHT + 40  # gap between rows

    # Stable camera index mapping for layout
    camera_index_map: Dict[str, int] = {}
    next_camera_index: int = 0

    # Per-camera render window state
    windows_render: Dict[str, str] = {}
    last_render_paths: Dict[str, Optional[Path]] = {}

    # Per-target window state
    target_windows: Dict[str, str] = {}
    last_target_paths: Dict[str, Optional[Path]] = {}


    try:
        while True:
            # --------------------------------------------------
            # 1. Discover camera folders dynamically
            # --------------------------------------------------
            camera_dirs = discover_camera_dirs(run_dir)
            for camera_dir in camera_dirs:
                camera_name = camera_dir.name

                # Assign stable index for layout
                if camera_name not in camera_index_map:
                    camera_index_map[camera_name] = next_camera_index
                    next_camera_index += 1

                if camera_name not in windows_render:
                    render_window_title = f"Render ({camera_name})"
                    windows_render[camera_name] = render_window_title
                    last_render_paths[camera_name] = None

                    cv2.namedWindow(render_window_title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(render_window_title, WINDOW_WIDTH, WINDOW_HEIGHT)

                    camera_index = camera_index_map[camera_name]
                    col_index = camera_index % WINDOWS_PER_ROW
                    row_index = camera_index // WINDOWS_PER_ROW

                    position_x = col_index * WINDOW_WIDTH
                    position_y = row_index * ROW_OFFSET_PIXELS  # row 0 for renders

                    cv2.moveWindow(render_window_title, position_x, position_y)

            # --------------------------------------------------
            # 2. Discover and show target images per camera
            #    Files must be named: render_target_<camera>.png
            # --------------------------------------------------
            target_candidates = sorted(run_dir.glob("render_target_*.png"))

            for target_path in target_candidates:
                stem = target_path.stem  # e.g. "render_target_camera1"
                prefix = "render_target_"
                if not stem.startswith(prefix):
                    continue
                camera_name = stem[len(prefix) :]

                # Create window if new
                if camera_name not in target_windows:
                    target_window_title = f"Target ({camera_name})"
                    target_windows[camera_name] = target_window_title
                    last_target_paths[camera_name] = None

                    cv2.namedWindow(target_window_title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(target_window_title, WINDOW_WIDTH, WINDOW_HEIGHT)

                    # Same column as render window, but on row 1 below it
                    camera_index = camera_index_map.get(camera_name, 0)
                    col_index = camera_index % WINDOWS_PER_ROW
                    row_index = camera_index // WINDOWS_PER_ROW

                    position_x = col_index * WINDOW_WIDTH
                    position_y = row_index * ROW_OFFSET_PIXELS + ROW_OFFSET_PIXELS  # second row

                    cv2.moveWindow(target_window_title, position_x, position_y)

                prev_target_path = last_target_paths.get(camera_name)
                if target_path != prev_target_path:
                    target_image_rgb = load_image_rgb(target_path)
                    if target_image_rgb is not None:
                        cv2.imshow(
                            target_windows[camera_name],
                            cv2.cvtColor(target_image_rgb, cv2.COLOR_RGB2BGR),
                        )
                        last_target_paths[camera_name] = target_path

            # --------------------------------------------------
            # 3. For each camera, display latest render
            # --------------------------------------------------
            for camera_name, render_window_title in windows_render.items():
                camera_dir = run_dir / camera_name

                # Prefer camera_dir/render/ if it exists, else camera_dir itself
                render_dir = camera_dir / "render"
                if not render_dir.exists():
                    render_dir = camera_dir

                latest_render_path = get_latest_render_image_path(render_dir)
                prev_render_path = last_render_paths.get(camera_name)

                if latest_render_path is not None and latest_render_path != prev_render_path:
                    render_image_rgb = load_image_rgb(latest_render_path)
                    if render_image_rgb is not None:
                        cv2.imshow(
                            render_window_title,
                            cv2.cvtColor(render_image_rgb, cv2.COLOR_RGB2BGR),
                        )
                        last_render_paths[camera_name] = latest_render_path

            # --------------------------------------------------
            # 4. Handle keyboard input
            # --------------------------------------------------
            key_code = cv2.waitKey(refresh_ms) & 0xFF
            if key_code == ord("q") or key_code == 27:  # 'q' or ESC
                break

    except KeyboardInterrupt:
        print("Preview interrupted by user.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
