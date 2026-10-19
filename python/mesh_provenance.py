"""Trace mesh accuracy error to the exact surfels supplying extraction depths.

Run with --run RUN --out OUT. No training state or saved mesh is modified.
Enable OptimizationConfig.densification_diagnostics during training to record
stable IDs and split/prune events. Existing runs still support depth attribution
and ablation, but cannot supply unrecorded historical ancestry. For a resumed
run, ancestry stops at that run's initial checkpoint.

Optional: extract_mesh.py --save-depth-provenance saves pixel identities during
extraction. This analyzer replays the frozen mesh input and validates identities.
Outputs include surfel_accuracy_attribution.csv, mesh_samples.npz, depth-source
maps, a colored mesh, and (with logged ancestry) mesh_error_lineage.csv plus per-
surfel JSON split histories and saved optimization trajectories.
Pixel-to-surfel identity is exact in the supported renderer path. Mesh-to-pixel
association is TSDF-band support, not an exact derivative of marching cubes:
each uniformly sampled mesh point shares its error equally among supporting
views. Optional fixed-TSDF removal experiments test the causal interpretation.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

import numpy as np
import open3d as o3d
from extract_mesh import PaleExtractor, load_scene_xml_cameras
from tsdf_settings import clean_depth, post_process_mesh


def distances(points, mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy()


def sample(mesh, count):
    o3d.utility.random.seed(0)
    return np.asarray(mesh.sample_points_uniformly(count).points, dtype=np.float32)


def depth_support(points, camera, depth, primitive, band):
    """Nearest-pixel lookup and ray-distance TSDF band, as used by fusion."""
    cp = points @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
    z = cp[:, 2]
    safe_z = np.where(z > 0, z, 1.)
    u = np.floor(camera.fx*cp[:, 0]/safe_z + camera.cx + .5).astype(np.int64)
    v = np.floor(camera.fy*cp[:, 1]/safe_z + camera.cy + .5).astype(np.int64)
    inside = (z > 0) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    rows = np.flatnonzero(inside)
    d = depth[v[rows], u[rows]]
    ids = primitive[v[rows], u[rows]]
    ray_scale = np.sqrt(1+((u[rows]-camera.cx)/camera.fx)**2+((v[rows]-camera.cy)/camera.fy)**2)
    valid = (d > 0) & (ids != np.iinfo(np.uint32).max) & (np.abs((d-z[rows])*ray_scale) <= band)
    result = np.full(len(points), -1, dtype=np.int32)
    result[rows[valid]] = ids[valid]
    return result


def attribute_errors(support, errors, point_count):
    valid = support >= 0
    counts = valid.sum(axis=1)
    shares = 1 / np.maximum(counts, 1)
    area = np.zeros((point_count, support.shape[1]))
    error = np.zeros_like(area)
    for c in range(support.shape[1]):
        mask = valid[:, c]
        area[:, c] = np.bincount(support[mask, c], weights=shares[mask], minlength=point_count) / len(errors)
        error[:, c] = np.bincount(support[mask, c], weights=shares[mask]*errors[mask], minlength=point_count) / len(errors)
    # Ray queries return float32; accumulate the reference in float64 like bincount.
    expected = errors[counts > 0].sum(dtype=np.float64) / len(errors)
    np.testing.assert_allclose(error.sum(), expected, rtol=1e-6)
    return area, error, counts


def trace_lineage(run, stable_id, through_iteration=None):
    """Return actual topology history; retained sources and appended children both count."""
    folder = run/'densification_diagnostics'
    events = []
    current = int(stable_id)
    for path in reversed(sorted((folder/'lineage').glob('event_*.npz'))):
        iteration = int(path.stem.split('_')[1])
        if through_iteration is not None and iteration > through_iteration:
            continue
        with np.load(path) as data:
            child = np.flatnonzero(data['child_ids'] == current)
            retained = np.flatnonzero(data['parent_ids'] == current)
            if not len(child) and not len(retained):
                continue
            k = int(child[0] if len(child) else retained[0])
            source = int(data['source_index'][k])
            event = dict(iteration=iteration, branch='appended_child' if len(child) else 'retained_source',
                         descendant_id=current, parent_id=int(data['parent_ids'][k]), source_row=source)
            decision_path = folder/f'iter_{iteration:05d}.npz'
            with np.load(decision_path) as decision:
                if int(decision['surfel_ids'][source]) != event['parent_id']:
                    raise RuntimeError(f'{path}: decision and topology identities disagree')
                count = float(decision['camera_count'][source].sum())
                event.update(score=float(decision['score'][source]),
                             positive_observations=count,
                             camera_contributions=dict(zip(decision['camera_names'].tolist(),
                                 (decision['camera_sum'][source]/max(count, 1)).tolist())),
                             before_position=decision['position'][source].tolist(),
                             before_scale=decision['scale'][source].tolist(),
                             before_albedo=decision['albedo'][source].tolist())
                if decision['vector_sum'].size:
                    event['vector_sum'] = decision['vector_sum'][source].tolist()
                for key in ('rotation', 'opacity', 'beta'):
                    if key in decision:
                        event['before_'+key] = np.asarray(decision[key][source]).tolist()
                if decision['threshold'].size:
                    event['threshold'] = float(decision['threshold'])
            for key in ('position', 'scale', 'opacity'):
                prefix = 'child_' if len(child) else 'source_after_'
                if prefix+key in data:
                    event['after_'+key] = np.asarray(data[prefix+key][k]).tolist()
            events.append(event)
            current = event['parent_id']
    return dict(final_id=int(stable_id), initial_id=current, splits=list(reversed(events)))


def lineage_roots(run, through_iteration):
    folder = run/'densification_diagnostics/lineage'
    initial = list(folder.glob('ids_*_initial.npy'))
    if len(initial) != 1:
        raise RuntimeError('Expected exactly one initial identity namespace')
    roots = {int(i): int(i) for i in np.load(initial[0])}
    for path in sorted(folder.glob('event_*.npz')):
        if int(path.stem.split('_')[1]) > through_iteration:
            break
        with np.load(path) as event:
            for parent, child in zip(event['parent_ids'], event['child_ids']):
                if int(child) in roots:
                    raise RuntimeError(f'{path}: stable ID was reused')
                roots[int(child)] = roots[int(parent)]
    return roots


def motion_breakdown(trace, initial_position, final_position):
    """Separate explicit split offsets from motion between recorded split events."""
    previous = np.asarray(initial_position, dtype=float)
    split = np.zeros(3)
    optimization = np.zeros(3)
    for event in trace['splits']:
        before = np.asarray(event['before_position'])
        after = np.asarray(event['after_position'])
        event['motion_since_previous_split'] = (before-previous).tolist()
        event['split_offset'] = (after-before).tolist()
        optimization += before-previous
        split += after-before
        previous = after
    optimization += np.asarray(final_position)-previous
    return dict(initial_position=np.asarray(initial_position).tolist(),
                final_position=np.asarray(final_position).tolist(),
                net_split_displacement=split.tolist(),
                net_between_split_displacement=optimization.tolist())


def read_training_snapshot(path):
    with path.open() as stream:
        names = []
        for line in stream:
            if line.startswith('format ') and 'ascii' not in line:
                raise ValueError('Trajectory reader expects training ASCII PLY snapshots')
            if line.startswith('property '): names.append(line.split()[-1])
            if line.strip() == 'end_header': break
        values = np.loadtxt(stream, ndmin=2)
    return values, names


def trace_trajectories(run, stable_ids, through_iteration=None):
    """Attach saved optimizer trajectories without inferring identity from proximity."""
    trajectories = {int(i): [] for i in stable_ids}
    for path in sorted((run/'densification_diagnostics/lineage').glob('ids_*_points.npy')):
        iteration = int(path.stem.split('_')[1])
        if through_iteration is not None and iteration > through_iteration:
            continue
        ids = np.load(path)
        wanted = [(int(i), int(np.flatnonzero(ids == i)[0])) for i in stable_ids if np.any(ids == i)]
        if not wanted:
            continue
        ply = run/f'points/iter_{iteration:05d}_points.ply'
        values, names = read_training_snapshot(ply)
        if len(values) != len(ids):
            raise RuntimeError(f'{ply}: snapshot identity count mismatch')
        for stable_id, row in wanted:
            trajectories[stable_id].append(dict(iteration=iteration, row=row,
                **{key: float(values[row, names.index(key)]) for key in
                   ('x', 'y', 'z', 'su', 'sv', 'albedo_r', 'albedo_g', 'albedo_b', 'opacity')}))
    return trajectories


def main():
    import pale
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=20000)
    parser.add_argument('--samples', type=int, default=1_000_000)
    parser.add_argument('--bad-distance', type=float, default=.02)
    parser.add_argument('--ablate-top', type=int, nargs='*', default=[])
    parser.add_argument('--ablate-indices', type=int, nargs='*', default=[])
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((run/'run_config.json').read_text())
    checkpoint = run/f'mesh_checkpoints/iter_{args.iteration:05d}'
    metadata = json.loads((checkpoint/'extraction.json').read_text())
    if metadata['depth_key'] != 'median_depth':
        raise ValueError('Exact primitive identity requires median_depth extraction')
    points_path = run/f'mesh_inputs/iter_{args.iteration:05d}_before_topology_update.ply'
    settings = config['renderer_settings'] | {'median_depth_retain_last_hit': metadata['median_depth_retain_last_hit']}
    assets = out/'renderer_assets'; assets.mkdir(exist_ok=True)
    os.chdir(out)
    renderer = pale.Renderer(str(assets), str(run/'scene.xml'), str(points_path), settings)
    params = {k: np.array(v, copy=True) for k, v in renderer.get_point_parameters().items()}
    cameras = load_scene_xml_cameras(run/'scene.xml')
    names = [v['name'] for v in metadata['tsdf']['views']]
    print('Rendering exact median-depth identities', flush=True)
    frames = renderer.render_forward(depth_provenance=True)
    depth_frames = {}
    # Renderer IDs must index the canonical saved rows, verified geometrically.
    w, x, y, z = params['rotation'].T
    normals = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
    checks = {}
    for name in names:
        f = frames[name]
        ids = np.asarray(f['median_primitive_index'])
        depth = clean_depth(f['median_depth'], metadata['tsdf']['depth_trunc'])
        valid = depth > 0
        if np.any(ids[valid] >= len(params['position'])):
            raise RuntimeError(f'{name}: unsupported depth identities')
        indices = ids[valid]
        world = np.asarray(f['median_world_position'])[valid, :3]
        residual = np.abs(np.sum((world-params['position'][indices])*normals[indices], axis=1))
        if residual.max(initial=0) > 2e-5:
            raise RuntimeError(f'{name}: native IDs fail canonical surfel plane validation')
        checks[name] = dict(valid_pixels=int(valid.sum()), max_plane_residual=float(residual.max(initial=0)))
        depth_frames[name] = (depth, ids)
        np.savez_compressed(out/f'depth_sources_{name}.npz', depth=depth, primitive_index=ids)
    mesh = o3d.io.read_triangle_mesh(str(checkpoint/'fuse_post.ply'))
    gt_path = Path(config['optimization_config']['ground_truth'])
    gt = o3d.io.read_triangle_mesh(str(gt_path))
    queries = sample(mesh, args.samples)
    errors = distances(queries, gt)
    support = np.column_stack([depth_support(queries, cameras[n], *depth_frames[n], metadata['tsdf']['sdf_trunc']) for n in names])
    area, error, support_count = attribute_errors(support, errors, len(params['position']))
    _, bad_error, _ = attribute_errors(support, errors*(errors > args.bad_distance), len(params['position']))
    ranking = np.argsort(-error.sum(axis=1))
    ids_path = run/f'densification_diagnostics/lineage/ids_{args.iteration:05d}_mesh_input.npy'
    stable_ids = np.load(ids_path) if ids_path.exists() else None
    # Training exports this mesh input before this iteration's topology mutation.
    roots = lineage_roots(run, args.iteration-1) if stable_ids is not None else None
    if stable_ids is not None and len(stable_ids) != len(params['position']):
        raise RuntimeError('Mesh input and stable identity count mismatch')
    rows = []
    for i in ranking:
        total_area = area[i].sum()
        row = dict(surfel_row=int(i), stable_id=int(stable_ids[i]) if stable_ids is not None else '',
                   initial_ancestor=roots[int(stable_ids[i])] if roots is not None else '',
                   attributed_accuracy=float(error[i].sum()), surface_area_fraction=float(total_area),
                   mean_supported_error=float(error[i].sum()/total_area) if total_area else 0,
                   bad_error_contribution=float(bad_error[i].sum()))
        for key in ('position', 'scale', 'albedo'):
            for k, value in enumerate(params[key][i]): row[f'{key}_{k}'] = float(value)
        for c, name in enumerate(names): row[name+'_accuracy_contribution'] = float(error[i, c])
        rows.append(row)
    with (out/'surfel_accuracy_attribution.csv').open('w') as stream:
        writer = csv.DictWriter(stream, rows[0]); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out/'mesh_samples.npz', points=queries, error=errors, support=support,
                        camera_names=names, stable_ids=stable_ids if stable_ids is not None else [])
    vertex_error = distances(np.asarray(mesh.vertices), gt)
    from matplotlib import colormaps
    mesh.vertex_colors = o3d.utility.Vector3dVector(colormaps['inferno'](np.clip(vertex_error/.1, 0, 1))[:, :3])
    o3d.io.write_triangle_mesh(str(out/'mesh_accuracy_heatmap.ply'), mesh)
    gt_queries = sample(gt, args.samples)
    completion = float(distances(gt_queries, mesh).mean())
    summary = dict(run=str(run), iteration=args.iteration, samples=args.samples, seed=0,
        sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                (Path(pale.__file__), points_path, checkpoint/'fuse_post.ply', gt_path)},
        tsdf=metadata['tsdf'],
        attribution='Exact depth-hit identities; equal-share TSDF-band support attribution for sampled mesh points (not exact fusion derivatives).',
        provenance_checks=checks, accuracy=float(errors.mean()), completion=completion,
        cd=(float(errors.mean())+completion)/2, surface_area=float(mesh.get_surface_area()),
        support_fraction=float(np.mean(support_count > 0)),
        attributed_error_fraction=float(error.sum()/errors.sum()*len(errors)),
        bad_distance=args.bad_distance, bad_area_fraction=float(np.mean(errors > args.bad_distance)),
        top_surfels=rows[:30], ablations={})
    if roots is not None:
        summary['initial_ancestor_attribution'] = {
            str(root): float(sum(row['attributed_accuracy'] for row in rows if row['initial_ancestor'] == root))
            for root in sorted(set(roots.values()))}
    if stable_ids is not None:
        initial_values = initial_names = None
        if not config['optimization_config'].get('checkpoint'):
            initial_values, initial_names = read_training_snapshot(run/'initial_points.ply')
        same_step_event = run/f'densification_diagnostics/lineage/event_{args.iteration:05d}.npz'
        trajectory_end = args.iteration-1 if same_step_event.exists() else args.iteration
        trajectories = trace_trajectories(run, stable_ids[ranking[:20]], trajectory_end)
        lineage_rows = []
        for i in ranking[:20]:
            trace = trace_lineage(run, stable_ids[i], args.iteration-1)
            if initial_values is not None:
                initial_position = initial_values[trace['initial_id'], [initial_names.index(k) for k in ('x', 'y', 'z')]]
                trace['motion_breakdown'] = motion_breakdown(trace, initial_position, params['position'][i])
            trace['trajectory'] = trajectories[int(stable_ids[i])]
            (out/f'lineage_row_{i}_id_{stable_ids[i]}.json').write_text(json.dumps(trace, indent=2))
            for event in trace['splits']:
                row = dict(final_row=int(i), final_id=int(stable_ids[i]),
                           final_attributed_accuracy=float(error[i].sum()),
                           **{k: event[k] for k in ('iteration', 'branch', 'descendant_id', 'parent_id', 'source_row', 'score', 'positive_observations')},
                           **event['camera_contributions'])
                lineage_rows.append(row)
        if lineage_rows:
            with (out/'mesh_error_lineage.csv').open('w') as stream:
                writer = csv.DictWriter(stream, lineage_rows[0]); writer.writeheader(); writer.writerows(lineage_rows)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    ablations = {f'top_{n}': ranking[:n] for n in args.ablate_top}
    if args.ablate_indices: ablations['specified'] = np.asarray(args.ablate_indices)
    for name, indices in ablations.items():
        print('Ablating', name, indices.tolist(), flush=True)
        changed = {k: v.copy() for k, v in params.items()}
        changed['opacity'][indices] = 0
        renderer.apply_point_optimization(changed); renderer.rebuild_bvh()
        extractor = PaleExtractor(renderer, cameras, names, 'median_depth')
        extractor.reconstruction()
        reconstructed = extractor.extract_mesh_bounded(**{k: metadata['tsdf'][k] for k in ('voxel_size', 'sdf_trunc', 'depth_trunc')})
        reconstructed = post_process_mesh(reconstructed, metadata['num_cluster'], metadata['min_cluster_triangles'])
        o3d.io.write_triangle_mesh(str(out/f'mesh_without_{name}.ply'), reconstructed)
        accuracy = float(distances(sample(reconstructed, args.samples), gt).mean())
        completion = float(distances(gt_queries, reconstructed).mean())
        summary['ablations'][name] = dict(indices=indices.tolist(), accuracy=accuracy, completion=completion,
                                          cd=(accuracy+completion)/2, area=float(reconstructed.get_surface_area()))
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('top_surfels','provenance_checks')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
