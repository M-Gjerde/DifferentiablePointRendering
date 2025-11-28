from __future__ import annotations

import time
import sys
import subprocess
from pathlib import Path

import pale
from config import RendererSettingsConfig, parse_args
from training import run_optimization
from render_hooks import get_camera_names


def main() -> None:
    config = parse_args()
    renderer_settings = RendererSettingsConfig()

    base_output_dir: Path = config.output_dir

    # Human-readable timestamp
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Short scene name
    scene_short = Path(config.scene_xml).stem

    # Build readable base
    run_folder_name = (
        f"{timestamp}_"
        f"lr{config.learning_rate_position:.3g}_"
        f"it{config.iterations}_"
        f"{scene_short}"
    )

    # Optional custom suffix
    if config.personal_suffix:
        safe_suffix = config.personal_suffix.strip().replace(" ", "_")
        run_folder_name += f"_{safe_suffix}"

    run_output_dir = base_output_dir / run_folder_name

    # Override config.output_dir
    config.output_dir = config.assets_root / run_output_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------
    # 1. Initialize renderer once
    # ------------------------------------------------------------------
    renderer = pale.Renderer(
        str(config.assets_root),
        config.scene_xml,
        config.pointcloud_ply,
        renderer_settings.as_dict(),
    )

    # Camera IDs from C++
    camera_ids = get_camera_names(renderer)
    if len(camera_ids) == 0:
        raise RuntimeError("No cameras found in scene.")
    main_camera = camera_ids[0]

    print("Starting optimization with configuration:")
    print(f"  assets_root          : {config.assets_root}")
    print(f"  scene_xml            : {config.scene_xml}")
    print(f"  pointcloud           : {config.pointcloud_ply}")
    print(f"  dataset_path         : {config.dataset_path}")
    print(f"  iterations           : {config.iterations}")
    print(f"  lr_position          : {config.learning_rate_position}")
    print(f"  lr_tangent           : {config.learning_rate_tangent}")
    print(f"  lr_scale             : {config.learning_rate_scale}")
    print(f"  lr_color             : {config.learning_rate_albedo}")
    print(f"  lr_opacity           : {config.learning_rate_opacity}")
    print(f"  lr_beta              : {config.learning_rate_beta}")
    print(f"  optimizer            : {config.optimizer_type}")
    print(f"  suffix               : '{config.personal_suffix}'")
    print(f"  run_output_dir       : {config.output_dir}")
    print(f"  cameras              : {camera_ids}")
    print(f"  main camera          : {main_camera}")

    # ------------------------------------------------------------------
    # 2. Launch external image preview script (non-blocking)
    # ------------------------------------------------------------------
    image_preview_script = Path(__file__).parent / "image_preview.py"
    image_preview_process = None

    # dataset_path is usually relative to assets_root (directory or file)
    dataset_path_full = (config.assets_root / config.dataset_path).resolve()

    if image_preview_script.exists():
        preview_args = [
            sys.executable,
            str(image_preview_script),
            "--output-path",
            str(config.output_dir.resolve()),
            "--refresh-ms",
            "200",                # adjust as you like
        ]
        try:
            image_preview_process = subprocess.Popen(preview_args)
            print(f"Started image preview : {image_preview_script}")
            print(f"  dataset-path        : {dataset_path_full}")
            print(f"  output-path         : {config.output_dir}")
            print(f"  camera-name         : {main_camera}")
        except Exception as exception:
            print(f"Warning: could not start image preview ({exception}). Continuing without preview.")
    else:
        print(f"Warning: image_preview.py not found at {image_preview_script}. No live preview will be shown.")

    # ------------------------------------------------------------------
    # 3. Run optimization (reusing the same renderer)
    # ------------------------------------------------------------------
    try:
        # Make sure run_optimization signature is:
        #   def run_optimization(renderer, config, renderer_settings) -> None:
        run_optimization(renderer, config, renderer_settings)
    finally:
        if image_preview_process is not None:
            try:
                image_preview_process.wait(timeout=5)
            except Exception:
                pass


if __name__ == "__main__":
    main()
