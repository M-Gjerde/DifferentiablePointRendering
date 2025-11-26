from __future__ import annotations

import time
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
    print(f"  lr_opacity             : {config.learning_rate_opacity}")
    print(f"  optimizer            : {config.optimizer_type}")
    print(f"  suffix               : '{config.personal_suffix}'")
    print(f"  run_output_dir       : {config.output_dir}")

    run_optimization(config, renderer_settings)


if __name__ == "__main__":
    main()
