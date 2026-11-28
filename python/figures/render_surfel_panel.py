"""Render one PALE GGX panel from an exported pinhole camera and surfel PLY.

Use --hits 1 for depth ordering or --hits 8 for slab compositing. --grid 4
integrates 16 primary rays per output pixel before applying display exposure.
Run with the native pale Python module on PYTHONPATH.
"""
import argparse
import copy
import math
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pale


def tile_camera(source, sensor_id, grid, x, y, width, height, path):
    tree = copy.deepcopy(source)
    sensor = tree.find('sensor')
    sensor.set('id', sensor_id)
    for name in ('fx', 'fy', 'cx', 'cy'):
        node = sensor.find(f"float[@name='{name}']")
        value = float(node.attrib['value'])
        if name == 'cx':
            value -= x
        if name == 'cy':
            value -= y
        # PALE casts primary rays at integer coordinates. Center the subpixel
        # lattice on the exact ray used by the full-resolution one-sample panel.
        offset = (grid - 1) / 2 if name in ('cx', 'cy') else 0
        node.set('value', str(value * grid + offset))
    for name, extent in [('width', width), ('height', height)]:
        sensor.find(f"film/integer[@name='{name}']").set('value', str(extent * grid))
    tree.write(path, encoding='utf-8', xml_declaration=True)


def render_panel(dataset, grid, hits):
    points = dataset / 'points.ply'
    scene = ET.parse(dataset / 'scene.xml')
    sensors = scene.findall('sensor')
    if len(sensors) != 1 or sensors[0].find("string[@name='camera_model'][@value='pinhole_intrinsics']") is None:
        raise ValueError('Expected one calibrated pinhole camera from the surfel exporter')
    sensor = sensors[0]
    width, height = [int(sensor.find(f"film/integer[@name='{axis}']").attrib['value'])
                     for axis in ('width', 'height')]
    camera_name = sensor.attrib['id']
    settings = dict(surfel_bsdf='ggx', seed=42, forward_passes=1, bounces=1, photons=1,
                    max_local_surfel_hits=hits, local_layer_depth_epsilon=0.0002,
                    local_layer_normal_cosine_threshold=0.9, share_local_layer_direct_lighting=False)
    panel = np.empty((height, width, 3), dtype=np.float32)
    # Limit the native forward and diagnostic buffers while rendering large figures.
    tile_side = max(1, 2048 // grid)
    previous_dir = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='pale-surfel-panel-') as tmp:
        try:
            for y in range(0, height, tile_side):
                for x in range(0, width, tile_side):
                    w, h = min(tile_side, width - x), min(tile_side, height - y)
                    scene_path = Path(tmp) / f'camera_{x}_{y}.xml'
                    tile_camera(scene, camera_name, grid, x, y, w, h, scene_path)
                    print(f'Rendering tile ({x}, {y}), {w} x {h}, {grid**2} rays/pixel', flush=True)
                    renderer = pale.Renderer(tmp, str(scene_path), str(points), settings)
                    raw = renderer.render_forward(camera_name)[camera_name]['raw']
                    pixels = np.asarray(raw)[..., :3].reshape(h, grid, w, grid, 3)
                    panel[y:y+h, x:x+w] = pixels.mean(axis=(1, 3), dtype=np.float64)
                    del pixels, raw, renderer
        finally:
            # The PALE constructor changes the process's working directory.
            os.chdir(previous_dir)
    if not np.isfinite(panel).all():
        raise ValueError('Non-finite linear radiance in rendered panel')
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Directory containing points.ply and scene.xml')
    parser.add_argument('output', type=Path, help='Display PNG at the original camera resolution')
    parser.add_argument('--hits', type=int, choices=(1, 8), default=8)
    parser.add_argument('--grid', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--exposure', type=float, default=1.0, help='Linear multiplier as in PALE Appearance')
    parser.add_argument('--linear-out', type=Path, help='Optional unexposed linear RGB .npy')
    parser.add_argument('--blender-scene', type=Path, help="Use this .blend's display, view, look and color management")
    parser.add_argument('--blender', default='blender', help='Blender executable for --blender-scene')
    args = parser.parse_args()
    if not math.isfinite(args.exposure) or args.exposure <= 0:
        parser.error('Exposure must be positive and finite')
    rgb = render_panel(args.dataset.resolve(), args.grid, args.hits)
    if args.linear_out:
        np.save(args.linear_out, rgb)
    if args.blender_scene:
        worker = Path(__file__).with_name('blender_display.py').resolve()
        with tempfile.TemporaryDirectory(prefix='pale-blender-display-') as tmp:
            linear_file = Path(tmp) / 'linear.npy'
            np.save(linear_file, rgb)
            subprocess.run([args.blender, '-b', '-noaudio', str(args.blender_scene.resolve()),
                            '--python-exit-code', '1', '-P', str(worker), '--', '--image', str(linear_file),
                            str(args.output.resolve()), '--exposure', str(args.exposure)], check=True)
        return
    linear = np.clip(rgb * args.exposure, 0, 1)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear**(1 / 2.4) - 0.055)
    Image.fromarray(np.rint(255 * srgb).astype(np.uint8)).save(args.output, format='PNG')


if __name__ == '__main__':
    main()
