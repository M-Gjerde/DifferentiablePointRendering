from __future__ import annotations

from config import RendererSettingsConfig, parse_args
from training import run_optimization


def main() -> None:
    config = parse_args()
    renderer_settings = RendererSettingsConfig()
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
    print(f"  lr_color             : {config.learning_rate_color}")
    print(f"  optimizer            : {config.optimizer_type}")
    print(f"  output_dir           : {config.output_dir}")
    print(f"  device               : {config.device}")

    run_optimization(config, renderer_settings)


if __name__ == "__main__":
    main()
