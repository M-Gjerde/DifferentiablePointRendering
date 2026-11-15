from __future__ import annotations

from pathlib import Path

import pale
from config import OptimizationConfig, RendererSettingsConfig, parse_args
from render_hooks import get_training_camera_names
from run_setup import prepare_run, print_run_configuration, start_companions, stop_companions
from training import run_optimization
from viewer_session import ViewerSession, materialize_workspace


def main(config: OptimizationConfig | None = None,
         renderer_settings: RendererSettingsConfig | None = None) -> None:
    """Train with CLI defaults or supplied, already-resolved configuration objects."""
    if config is None:
        config = parse_args()
    if renderer_settings is None:
        renderer_settings = RendererSettingsConfig()
    if config.adjoint_spp is not None:
        renderer_settings.adjoint_passes = config.adjoint_spp
    session = ViewerSession(config.viewer_session, config.viewer_live_interval) if config.viewer_session else None
    try:
        if config.viewer_workspace:
            if session is None:
                raise ValueError("Viewer workspace initialization requires a viewer session")
            state = materialize_workspace(config.viewer_workspace, session.directory / "initial_scene.xml")
            config.scene_xml = str(session.directory / "initial_scene.xml")
            config.scene_xml_is_explicit = True
            config.pointcloud_ply = state["pointcloud"]
            config.pointcloud_ply_is_explicit = True
            config.assets_root = Path(state["asset_root"])
        prepare_run(config, renderer_settings)

        if session:
            session.configure(config)

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
            run_optimization(renderer, config, renderer_settings, viewer_session=session)
        finally:
            stop_companions(companions)

    except BaseException as error:
        if session:
            session.finish(error)
        raise
    else:
        if session:
            session.finish()


if __name__ == "__main__":
    main()
