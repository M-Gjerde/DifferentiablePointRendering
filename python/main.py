from __future__ import annotations

import time
import sys
import subprocess
from pathlib import Path

from config import RendererSettingsConfig, parse_args
from training import run_optimization


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
        # ensure clean formatting: no spaces, only safe chars
        safe_suffix = config.personal_suffix.strip().replace(" ", "_")
        run_folder_name += f"_{safe_suffix}"

    run_output_dir = base_output_dir / run_folder_name

    # Override config.output_dir
    config.output_dir = run_output_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure render subfolder exists (viewer will look here)
    render_output_dir = config.output_dir / "render"
    render_output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting optimization with configuration:")
    print(f"  assets_root          : {config.assets_root}")
    print(f"  scene_xml            : {config.scene_xml}")
    print(f"  pointcloud           : {config.pointcloud_ply}")
    print(f"  target_image         : {config.target_image_path}")
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

    # ------------------------------------------------------------------
    # Launch external image preview script (non-blocking)
    # ------------------------------------------------------------------
    image_preview_script = Path(__file__).parent / "image_preview.py"
    image_preview_process = None

    if image_preview_script.exists():
        preview_args = [
            sys.executable,
            str(image_preview_script),
            "--target-image",
            str(config.assets_root / config.target_image_path),
            "--render-dir",
            str(config.assets_root / render_output_dir),
            "--refresh-ms",
            "1",  # adjust refresh rate as desired
        ]
        try:
            image_preview_process = subprocess.Popen(preview_args)
            print(f"Started image preview: {image_preview_script}")
        except Exception as exception:
            print(f"Warning: could not start image preview ({exception}). Continuing without preview.")
    else:
        print(f"Warning: image_preview.py not found at {image_preview_script}. No live preview will be shown.")

    # ------------------------------------------------------------------
    # Run optimization
    # ------------------------------------------------------------------
    try:
        run_optimization(config, renderer_settings)
    finally:
        # Optionally wait for the preview process at the end,
        # so it can be closed cleanly by the user.
        if image_preview_process is not None:
            try:
                image_preview_process.wait(timeout=5)
            except Exception:
                pass


if __name__ == "__main__":
    main()
