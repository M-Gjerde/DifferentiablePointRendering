from __future__ import annotations

import pale
from config import RendererSettingsConfig, parse_args
from render_hooks import get_training_camera_names
from run_setup import prepare_run, print_run_configuration, start_companions, stop_companions
from training import run_optimization


def main() -> None:
    config = parse_args()
    renderer_settings = RendererSettingsConfig()
    prepare_run(config, renderer_settings)

    renderer = pale.Renderer(
        str(config.assets_root),
        config.scene_xml,
        config.pointcloud_ply,
        renderer_settings.as_dict(config),
    )

    camera_ids = get_training_camera_names(renderer)
    if len(camera_ids) == 0:
        raise RuntimeError("No cameras found in scene.")

    print_run_configuration(config, camera_ids, camera_ids[0])
    companions = start_companions(config)

    try:
        run_optimization(renderer, config, renderer_settings)
    finally:
        stop_companions(companions)


if __name__ == "__main__":
    main()
