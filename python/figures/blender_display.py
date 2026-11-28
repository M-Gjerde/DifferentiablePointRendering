"""Apply the open Blender scene's render view to top-down linear PALE RGB arrays."""
import argparse
from pathlib import Path
import sys

import bpy
import numpy as np


def save_display(scene, source, destination, exposure):
    linear = np.load(source, allow_pickle=False)
    if linear.ndim != 3 or linear.shape[2] != 3 or not np.isfinite(linear).all():
        raise ValueError(f'Expected a finite H x W x 3 linear radiance image: {source}')
    height, width = linear.shape[:2]
    image = bpy.data.images.new('PALE linear radiance', width=width, height=height,
                                alpha=True, float_buffer=True)
    try:
        # Blender 5.3's scene-linear role is Linear Rec.709; float images use it by default.
        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[..., :3] = linear[::-1] * exposure
        image.pixels.foreach_set(rgba.ravel())
        image.update()
        image.save_render(str(destination.resolve()), scene=scene)
    finally:
        bpy.data.images.remove(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, nargs=2, action='append', required=True,
                        metavar=('LINEAR_NPY', 'DISPLAY_PNG'))
    parser.add_argument('--exposure', type=float, default=1)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if not np.isfinite(args.exposure) or args.exposure <= 0:
        parser.error('Exposure must be positive and finite')
    scene = bpy.context.scene
    settings = scene.render.image_settings
    settings.file_format = 'PNG'
    settings.color_mode = 'RGB'
    settings.color_depth = '8'
    settings.color_management = 'FOLLOW_SCENE'
    for source, destination in args.image:
        save_display(scene, source, destination, args.exposure)
