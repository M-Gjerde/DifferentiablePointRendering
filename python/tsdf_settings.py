"""Data-driven TSDF settings shared by the PALE and top-level 2DGS extractors.

Keep this NumPy-only module identical in both projects; neither project needs
the other on its import path. Distances are in scene units, never assumed metres.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass
class DepthFrame:
    name: str
    depth: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray


def add_tsdf_arguments(parser):
    parser.add_argument('--depth-trunc', '--depth_trunc', type=float, default=-1.0,
                        help='Maximum camera-forward depth in scene units; negative: infer from depth maps.')
    parser.add_argument('--voxel-size', '--voxel_size', type=float, default=-1.0,
                        help='Voxel edge in scene units; negative: infer from visible surface extent and pixel spacing.')
    parser.add_argument('--sdf-trunc', '--sdf_trunc', type=float, default=-1.0,
                        help='TSDF band in scene units; negative: 4 times the voxel size.')
    parser.add_argument('--mesh-res', '--mesh_res', type=int, default=512,
                        help='Target voxels along the longest robust surface extent (default 512).')
    parser.add_argument('--depth-percentile', '--depth_percentile', type=float, default=99.0,
                        help='Per-view valid-depth percentile for automatic far cutoff (default 99).')
    parser.add_argument('--depth-margin', '--depth_margin', type=float, default=1.1,
                        help='Multiply the largest per-view depth percentile by this margin (default 1.1).')
    parser.add_argument('--voxel-pixels', '--voxel_pixels', type=float, default=1.0,
                        help='Automatic voxel size is at least this many median pixel footprints (default 1).')
    parser.add_argument('--num-cluster', '--num_cluster', type=int, default=0,
                        help='Keep largest N connected components; 0 keeps all (default).')
    parser.add_argument('--min-cluster-triangles', '--min_cluster_triangles', type=int, default=50,
                        help='Minimum triangles per component (default 50); 0 disables small-component filtering.')
    parser.add_argument('--tsdf-settings', '--tsdf_settings', type=Path,
                        help='Reuse depth/voxel/SDF values from an extraction JSON; explicit numeric flags win.')


def clean_depth(depth, depth_trunc=None):
    """Open3D uses zero for invalid depth. Never alter the renderer's buffer."""
    result = np.array(depth, dtype=np.float32, order='C', copy=True)
    invalid = ~np.isfinite(result) | (result <= 0)
    if depth_trunc is not None:
        invalid |= result >= depth_trunc
    result[invalid] = 0
    return result


def validate_options(args):
    if args.mesh_res <= 0:
        raise ValueError('mesh_res must be positive')
    if not np.isfinite(args.depth_percentile) or not 0 < args.depth_percentile <= 100:
        raise ValueError('depth_percentile must be in (0, 100]')
    if not np.isfinite(args.depth_margin) or args.depth_margin <= 1:
        raise ValueError('depth_margin must be greater than 1 (Open3D excludes depths at the cutoff)')
    if not np.isfinite(args.voxel_pixels) or args.voxel_pixels <= 0:
        raise ValueError('voxel_pixels must be finite and positive')
    if args.num_cluster < 0 or args.min_cluster_triangles < 0:
        raise ValueError('Component limits must be nonnegative; 0 disables filtering')
    for key in ('depth_trunc', 'voxel_size', 'sdf_trunc'):
        value = getattr(args, key)
        if not np.isfinite(value) or value == 0:
            raise ValueError(f'{key} must be positive or negative for automatic selection')


def resolve_tsdf_settings(frames, args):
    """Estimate geometry scale from depth, independent of camera arrangement.

    Far cutoff: margin * max(per-view depth percentile). Voxel edge: max of
    robust back-projected world extent / mesh_res and median depth/focal length.
    Bounds use pooled 0.5/99.5 percentiles; they set resolution, not a crop box.
    Up to 8192 valid pixels per frame are sampled deterministically for bounds.
    """
    validate_options(args)
    frames = list(frames)
    if not frames:
        raise ValueError('No RGB-D frames available for TSDF integration')
    reference = {}
    if args.tsdf_settings is not None:
        document = json.loads(Path(args.tsdf_settings).read_text())
        reference = document.get('tsdf', document)
        for key in ('depth_trunc', 'voxel_size', 'sdf_trunc'):
            value = reference.get(key)
            if not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid {key} in {args.tsdf_settings}')
    requested = {key: (getattr(args, key) if getattr(args, key) > 0 else reference.get(key))
                 for key in ('depth_trunc', 'voxel_size', 'sdf_trunc')}
    depths = []
    per_view = []
    for frame in frames:
        depth = np.asarray(frame.depth)
        if depth.ndim != 2:
            raise ValueError(f'{frame.name}: expected HxW depth, got {depth.shape}')
        if not all(np.isfinite(v) for v in (frame.fx, frame.fy, frame.cx, frame.cy)) or min(frame.fx, frame.fy) <= 0:
            raise ValueError(f'{frame.name}: invalid camera intrinsics')
        pose = np.asarray(frame.world_to_camera)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f'{frame.name}: invalid camera extrinsics')
        valid = np.isfinite(depth) & (depth > 0)
        values = depth[valid]
        depths.append((depth, valid))
        per_view.append({'name': frame.name, 'pixels': int(depth.size), 'valid_pixels': int(values.size),
                         'depth_percentile': float(np.percentile(values, args.depth_percentile)) if values.size else None})
    percentiles = [v['depth_percentile'] for v in per_view if v['depth_percentile'] is not None]
    if not percentiles:
        raise ValueError('No positive finite depth pixels; cannot extract a mesh')
    cutoff = requested['depth_trunc'] or max(percentiles) * args.depth_margin
    rng = np.random.default_rng(0)
    world_samples, pixel_samples = [], []
    for frame, (depth, valid), stats in zip(frames, depths, per_view):
        retained = valid & (depth < cutoff)
        stats['retained_pixels'] = int(retained.sum())
        indices = np.flatnonzero(retained)
        if indices.size == 0:
            continue
        if indices.size > 8192:
            indices = rng.choice(indices, 8192, replace=False)
        y, x = np.unravel_index(indices, depth.shape)
        z = depth[y, x].astype(np.float64)
        camera_points = np.stack(((x-frame.cx)*z/frame.fx, (y-frame.cy)*z/frame.fy, z), axis=1)
        camera_to_world = np.linalg.inv(frame.world_to_camera)
        world_samples.append(camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3])
        pixel_samples.append(z / np.sqrt(frame.fx * frame.fy))
    if not world_samples:
        raise ValueError(f'depth_trunc={cutoff:g} rejects every valid depth pixel')
    points = np.concatenate(world_samples)
    bounds = np.percentile(points, [0.5, 99.5], axis=0)
    extent = float(np.max(bounds[1] - bounds[0]))
    footprint = float(np.median(np.concatenate(pixel_samples)))
    voxel = requested['voxel_size'] or max(extent / args.mesh_res, footprint * args.voxel_pixels)
    band = requested['sdf_trunc'] or 4 * voxel
    valid_count = sum(v['valid_pixels'] for v in per_view)
    retained_count = sum(v['retained_pixels'] for v in per_view)
    result = {'policy': 'rendered-depth-v1', 'depth_trunc': float(cutoff),
              'voxel_size': float(voxel), 'sdf_trunc': float(band),
              'mesh_res': args.mesh_res, 'depth_percentile': args.depth_percentile,
              'depth_margin': args.depth_margin, 'voxel_pixels': args.voxel_pixels,
              'robust_world_bounds': bounds.tolist(), 'robust_extent': extent,
              'median_pixel_footprint': footprint, 'valid_pixels': valid_count,
              'retained_pixels': retained_count,
              'rejected_depth_fraction': 1-retained_count/valid_count,
              'source': {key: ('cli' if getattr(args,key)>0 else 'settings_file' if key in reference else 'automatic') for key in requested},
              'views': per_view}
    for key in ('depth_trunc', 'voxel_size', 'sdf_trunc'):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'Invalid resolved {key}: {result[key]}')
    return result


def print_tsdf_settings(settings):
    print('TSDF: depth_trunc={depth_trunc:.6g}, voxel_size={voxel_size:.6g}, sdf_trunc={sdf_trunc:.6g}'.format(**settings))
    print(f"Depth range retains {settings['retained_pixels']:,}/{settings['valid_pixels']:,} valid pixels "
          f"({100*(1-settings['rejected_depth_fraction']):.2f}%)")
    if settings['rejected_depth_fraction'] > 0.05:
        print('WARNING: more than 5% of valid depth pixels are outside the fusion range; inspect the depth cutoff.')


def post_process_mesh(mesh, cluster_to_keep=0, min_cluster_triangles=0):
    """Preserve disconnected geometry unless the caller requests filtering."""
    import copy
    if cluster_to_keep < 0 or min_cluster_triangles < 0:
        raise ValueError('Component limits must be nonnegative')
    result = copy.deepcopy(mesh)
    result.remove_duplicated_vertices()
    result.remove_duplicated_triangles()
    result.remove_degenerate_triangles()
    if len(result.triangles) and (cluster_to_keep or min_cluster_triangles):
        labels, counts, _ = result.cluster_connected_triangles()
        labels, counts = np.asarray(labels), np.asarray(counts)
        keep = counts >= min_cluster_triangles
        if cluster_to_keep:
            largest = np.argsort(-counts, kind='stable')[:cluster_to_keep]
            keep &= np.isin(np.arange(len(counts)), largest)
        result.remove_triangles_by_mask(~keep[labels])
    result.remove_unreferenced_vertices()
    result.compute_vertex_normals()
    return result
