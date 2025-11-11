# main.py
import time
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import pale


def tone_map_exposure_gamma(rgb_linear: np.ndarray, exposure_stops: float, gamma: float) -> np.ndarray:
    """Simple exposure + gamma tonemap. Input HxWx3 float32 in linear radiance."""
    # Exposure in stops -> multiply by 2**stops
    scaled = rgb_linear * (2.0 ** exposure_stops)
    # Photographic curve approximation
    mapped = 1.0 - np.exp(-scaled)
    # Gamma encode
    mapped = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / max(gamma, 1e-6))
    return mapped


rendererSettings = {
    "photons": 1e6,
    "bounces": 6,
    "forward_passes": 6,
    "gather_passes": 6,
    "adjoint_bounces": 1,
    "adjoint_passes": 6,
}

assets_root = Path(__file__).parent / "Assets"
scene_xml = "cbox_custom.xml"
pointcloud_ply = "initial.ply"

renderer = pale.Renderer(str(assets_root), scene_xml, pointcloud_ply, rendererSettings)

# Forward render: returns HxWx3 float32 in linear space
# Note: current C++ path copies RGB without tone-mapping or flipping.
exposure_stops = 5.8
gamma_value = 2.8

# update Gaussian transform and re-render
renderer.set_gaussian_transform(
    translation3=(-0.01, 0.0, 0.0),
    rotation_quat4=(0.0, 0.0, 0.0, 1.0),  # (x,y,z,w)
    scale3=(1.0, 1.0, 1.0),
)

rgb_linear = renderer.render_forward()  # args accepted but not applied in C++ right now
rgb_linear = np.asarray(rgb_linear, dtype=np.float32)
rgb_linear = np.flipud(rgb_linear) # Flip vertically because the C++ tone-mapper (which handled flipping) is commented out
# Apply tone mapping in Python and save
rgb_tonemapped = tone_map_exposure_gamma(rgb_linear, exposure_stops, gamma_value)
rgb_uint8 = (np.clip(rgb_tonemapped, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)



output_dir = Path(__file__).parent / "Output"
output_dir.mkdir(parents=True, exist_ok=True)
output_png = output_dir / "target.png"

iio.imwrite(output_png, rgb_uint8)
print(f"Saved: {output_png.resolve()}")


# update Gaussian transform and re-render
renderer.set_gaussian_transform(
    translation3=(0.01, 0.0, 0.0),
    rotation_quat4=(0.0, 0.0, 0.0, 1.0),  # (x,y,z,w)
    scale3=(1.0, 1.0, 1.0),
)

rgb_linear = renderer.render_forward(5.8, 2.8, True)
rgb_linear = np.asarray(rgb_linear, dtype=np.float32)
rgb_linear = np.flipud(rgb_linear) # Flip vertically because the C++ tone-mapper (which handled flipping) is commented out
# Apply tone mapping in Python and save
rgb_tonemapped = tone_map_exposure_gamma(rgb_linear, exposure_stops, gamma_value)
rgb_uint8 = (np.clip(rgb_tonemapped, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

output_png = output_dir / "initial.png"
iio.imwrite(output_png, rgb_uint8)
print(f"Saved: {output_png.resolve()}")
time.sleep(5)

# renderer.render_backward(...)
