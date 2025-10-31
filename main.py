import time

import pale
from pathlib import Path

rendererSettings = dict({
    "photons": 1e6,
    "bounces": 6,
    "forward_passes": 6,
    "gather_passes": 6,
    "adjoint_bounces": 1,
    "adjoint_passes": 6
})

assets_root = Path(__file__).parent / "Assets"
scene_xml = "cbox_custom.xml"
pointcloud_ply = "initial.ply"

renderer = pale.Renderer(str(assets_root), scene_xml, pointcloud_ply, rendererSettings)


pkg = renderer.render_forward()


time.sleep(2)

#renderer.render_backward()




