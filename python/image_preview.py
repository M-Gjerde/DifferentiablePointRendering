#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class TileSelection:
    cameraName: str
    viewMode: str  # "render" or "target"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-window mosaic preview (NxM).")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--refresh-ms", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=500)
    parser.add_argument("--tile-height", type=int, default=500)
    parser.add_argument("--rows", type=int, default=2, help="Number of tile rows.")
    parser.add_argument("--cols", type=int, default=4, help="Number of tile columns.")
    parser.add_argument("--pad", type=int, default=6, help="Padding between tiles (pixels).")
    parser.add_argument("--window-scale", type=float, default=1.0, help="Resize window to scale*mosaic size.")
    return parser.parse_args()


def load_image_bgr_safely(image_path: Path, wait_time: float = 0.03) -> Optional[np.ndarray]:
    if not image_path.exists():
        return None

    try:
        size1 = image_path.stat().st_size
        time.sleep(wait_time)
        if not image_path.exists():
            return None
        size2 = image_path.stat().st_size
    except OSError:
        return None

    if size1 != size2:
        return None

    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def discover_camera_dirs(run_dir: Path) -> List[Path]:
    if not run_dir.is_dir():
        raise RuntimeError(f"output-path '{run_dir}' must be a directory.")
    return [p for p in sorted(run_dir.iterdir()) if p.is_dir()]


def get_latest_png(render_directory: Path) -> Optional[Path]:
    if not render_directory.exists():
        return None

    candidates = sorted(render_directory.glob("*.png"))
    if not candidates:
        return None

    # Prefer 1 step behind newest to avoid reading a file that's still being written.
    if len(candidates) >= 2:
        return candidates[-2]

    return candidates[-1]



def get_latest_render_path(run_dir: Path, camera_name: str) -> Optional[Path]:
    camera_dir = run_dir / camera_name
    render_dir = camera_dir / "render"
    if not render_dir.exists():
        render_dir = camera_dir
    return get_latest_png(render_dir)


def get_target_path(run_dir: Path, camera_name: str) -> Optional[Path]:
    candidate = run_dir / f"render_target_{camera_name}.png"
    return candidate if candidate.exists() else None


def resize_and_letterbox(image_bgr: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    scale = min(tile_w / w, tile_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    x0 = (tile_w - new_w) // 2
    y0 = (tile_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def draw_label(tile_bgr: np.ndarray, text: str, is_selected: bool) -> None:
    cv2.rectangle(tile_bgr, (0, 0), (tile_bgr.shape[1], 26), (0, 0, 0), thickness=-1)
    cv2.putText(
        tile_bgr,
        text,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if is_selected:
        cv2.rectangle(
            tile_bgr,
            (0, 0),
            (tile_bgr.shape[1] - 1, tile_bgr.shape[0] - 1),
            (255, 255, 255),
            2,
        )


def build_mosaic(
    tiles: List[np.ndarray],
    rows: int,
    cols: int,
    tile_w: int,
    tile_h: int,
    pad: int,
) -> np.ndarray:
    mosaic_h = rows * tile_h + (rows + 1) * pad
    mosaic_w = cols * tile_w + (cols + 1) * pad
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    for idx, tile in enumerate(tiles[: rows * cols]):
        r = idx // cols
        c = idx % cols
        y0 = pad + r * (tile_h + pad)
        x0 = pad + c * (tile_w + pad)
        mosaic[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

    return mosaic


def compute_default_view_mode(tile_index: int, cols: int, rows: int) -> str:
    row_index = tile_index // cols
    is_bottom_row = (row_index == rows - 1)
    return "target" if is_bottom_row else "render"



def main() -> None:
    args = parse_arguments()
    run_dir: Path = args.output_path
    refresh_ms: int = args.refresh_ms
    tile_w: int = args.tile_width
    tile_h: int = args.tile_height
    rows: int = max(1, int(args.rows))
    cols: int = max(1, int(args.cols))
    pad: int = max(0, int(args.pad))
    window_scale: float = float(args.window_scale)

    if not run_dir.exists():
        raise RuntimeError(f"Output path '{run_dir}' does not exist.")

    tile_count = rows * cols
    active_tile_index = 0
    selections: List[Optional[TileSelection]] = [None] * tile_count

    window_title = (
        f"Optimization Preview ({rows}x{cols}) — "
        "1-9 tile, arrows move, r/t toggle, [ ] camera, n/p page cameras, q quit"
    )
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    mosaic_w = cols * tile_w + (cols + 1) * pad
    mosaic_h = rows * tile_h + (rows + 1) * pad
    cv2.resizeWindow(window_title, max(200, int(mosaic_w * window_scale)), max(200, int(mosaic_h * window_scale)))
    camera_offset = 0  # index into camera_names

    time.sleep(3)
    try:
        while True:
            camera_dirs = discover_camera_dirs(run_dir)
            camera_names = [p.name for p in camera_dirs]

            if not camera_names:
                blank_tiles = [np.zeros((tile_h, tile_w, 3), dtype=np.uint8) for _ in range(tile_count)]
                mosaic = build_mosaic(blank_tiles, rows, cols, tile_w, tile_h, pad)
                cv2.imshow(window_title, mosaic)
                key = cv2.waitKey(refresh_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue


            cameras = []
            for i in range(cols):
                cameras.append(camera_names[(camera_offset + i) % len(camera_names)])

            # Initialize any unset tiles with defaults
            for i in range(tile_count):
                if selections[i] is None:
                    default_camera = cameras[(i) % len(cameras)]
                    default_mode = "target" if (i // cols) == (rows - 1) else "render"
                    selections[i] = TileSelection(
                        cameraName=default_camera,
                        viewMode=default_mode
                    )

            tiles: List[np.ndarray] = []
            for i in range(tile_count):
                tile_selection = selections[i]
                assert tile_selection is not None

                if tile_selection.viewMode == "render":
                    image_path = get_latest_render_path(run_dir, tile_selection.cameraName)
                else:
                    image_path = get_target_path(run_dir, tile_selection.cameraName)

                image_bgr: Optional[np.ndarray] = None
                if image_path is not None:
                    image_bgr = load_image_bgr_safely(image_path)

                if image_bgr is None:
                    tile_bgr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                else:
                    tile_bgr = resize_and_letterbox(image_bgr, tile_w, tile_h)

                label = f"[{i+1}] {tile_selection.cameraName} | {tile_selection.viewMode}"
                draw_label(tile_bgr, label, is_selected=(i == active_tile_index))
                tiles.append(tile_bgr)

            mosaic = build_mosaic(tiles, rows, cols, tile_w, tile_h, pad)
            cv2.imshow(window_title, mosaic)

            key = cv2.waitKey(refresh_ms) & 0xFF
            if key in (ord("q"), 27):
                break

            # Tile selection:
            # - 1..9 selects tiles 0..8
            # - 0 selects tile 9 (10th tile)
            # For >10 tiles, use arrow keys to move selection.
            if ord("1") <= key <= ord("9"):
                requested_index = int(chr(key)) - 1
                if requested_index < tile_count:
                    active_tile_index = requested_index
                continue
            if key == ord("0"):
                if tile_count >= 10:
                    active_tile_index = 9
                continue

            # Page cameras if more cameras than tiles
            if key == ord("n") and len(camera_names) > tile_count:
                camera_offset = (camera_offset - tile_count) % len(camera_names)
                selections = [None] * tile_count  # reinitialize tiles
                continue

            if key == ord("p") and len(camera_names) > tile_count:
                camera_offset = (camera_offset + tile_count) % len(camera_names)
                selections = [None] * tile_count
                continue

            # Arrow-key navigation (works for any tile_count)
            if key in (81, 82, 83, 84):  # left, up, right, down in OpenCV waitKey
                r = active_tile_index // cols
                c = active_tile_index % cols
                if key == 81:  # left
                    c = max(0, c - 1)
                elif key == 83:  # right
                    c = min(cols - 1, c + 1)
                elif key == 82:  # up
                    r = max(0, r - 1)
                elif key == 84:  # down
                    r = min(rows - 1, r + 1)
                active_tile_index = r * cols + c
                continue

            # Toggle render/target on active tile
            if key == ord("r"):
                selections[active_tile_index].viewMode = "render"
                continue
            if key == ord("t"):
                selections[active_tile_index].viewMode = "target"
                continue

            # Cycle camera on active tile
            if key in (ord("["), ord("]")):
                current_camera = selections[active_tile_index].cameraName
                if current_camera not in camera_names:
                    selections[active_tile_index].cameraName = camera_names[0]
                    continue

                current_index = camera_names.index(current_camera)
                delta = -1 if key == ord("[") else 1
                new_index = (current_index + delta) % len(camera_names)
                selections[active_tile_index].cameraName = camera_names[new_index]
                continue

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
