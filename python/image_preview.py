#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live preview of target and latest rendered image.")
    parser.add_argument(
        "--target-image",
        type=Path,
        required=True,
        help="Path to the target image file.",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        required=True,
        help="Directory where rendered *_render.png images are written.",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=1,
        help="Refresh interval in milliseconds (default: 500).",
    )
    return parser.parse_args()


def load_image_rgb(image_path: Path) -> Optional[np.ndarray]:
    if not image_path.exists():
        return None
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def get_latest_render_image_path(render_directory: Path) -> Optional[Path]:
    if not render_directory.exists():
        return None
    render_files: List[Path] = sorted(render_directory.glob("*_render.png"))
    if not render_files:
        return None
    return render_files[-1]


def main() -> None:
    args = parse_arguments()

    target_image_rgb = load_image_rgb(args.target_image)
    if target_image_rgb is not None:
        cv2.namedWindow("Target Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Target Image", 600, 600)
        cv2.imshow("Target Image", cv2.cvtColor(target_image_rgb, cv2.COLOR_RGB2BGR))

    # Current render window
    cv2.namedWindow("Current Render", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Current Render", 600, 600)

    last_render_path: Optional[Path] = None

    try:
        while True:
            latest_render_path = get_latest_render_image_path(args.render_dir)

            if latest_render_path is not None:
                if last_render_path is None or latest_render_path != last_render_path:
                    render_image_rgb = load_image_rgb(latest_render_path)
                    if render_image_rgb is not None:
                        cv2.imshow("Current Render", cv2.cvtColor(render_image_rgb, cv2.COLOR_RGB2BGR))
                        last_render_path = latest_render_path

            # Keep showing target image (if loaded)
            if target_image_rgb is not None:
                cv2.imshow("Target Image", cv2.cvtColor(target_image_rgb, cv2.COLOR_RGB2BGR))

            key_code = cv2.waitKey(args.refresh_ms) & 0xFF
            if key_code == ord("q") or key_code == 27:  # 'q' or ESC
                break

    except KeyboardInterrupt:
        print("Preview interrupted by user.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
