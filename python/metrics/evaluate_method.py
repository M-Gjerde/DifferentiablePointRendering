"""Batch geometry evaluation for the comparison runners; no model/data writes."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics
import math

DEFAULTS = {
    'ours': (str(Path(__file__).resolve().parents[1] / 'OptimizationOutput' / 'paper'), '', 'fuse_post.ply'),
    'gof': ('/home/magnus/phd/pbdr/GOF/output/batch_gof', '_2dgs', 'mesh.ply'),
    'pgsr': ('/home/magnus/phd/pbdr/PGSR/output/batch_pgsr', '_2dgs', 'fuse_post_auto.ply'),
    '2dgs': ('/home/magnus/projects/2D-GS-Viser-Viewer/output/batch_2dgs', '_2dgs', 'fuse_post_auto.ply'),
    'radiosity_gs': ('/home/magnus/phd/pbdr/RadiosityGS/output/batch_radiosity_gs', '_pbdr', 'fuse_post.ply'),
    'neus': ('/home/magnus/phd/pbdr/NeuS/output/batch_neus', '_2dgs', 'mesh.ply'),
    'gaussian_wrapping': ('/home/magnus/phd/pbdr/GaussianWrapping/output/batch_gaussian_wrapping', '_2dgs', 'mesh.ply'),
}
SCENES = ('horse', 'teapot', 'dragon', 'plant', 'workbench', 'restaurant')


def point_count(path):
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError(f'Invalid PLY: {path}')
        count = None
        for line in stream:
            fields = line.split()
            if fields[:2] == [b'element', b'vertex']:
                count = int(fields[2])
            if fields == [b'end_header']:
                if count is None or count < 0:
                    raise ValueError(f'Missing/invalid vertex count: {path}')
                return count
        raise ValueError(f'Incomplete PLY: {path}')


def training_time(model, iteration):
    """Prefer checkpoint timing; label whole-run timings as totals."""
    def read(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
    def valid(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    stats = read(model / 'training_stats.json').get('iterations', {}).get(str(iteration), {})
    seconds = stats.get('runtime_seconds')
    if valid(seconds):
        return float(seconds), 'checkpoint'
    batch = read(model / 'batch_train.json')
    if batch:
        seconds = batch.get('training_seconds')
        if batch.get('status') == 'complete' and valid(seconds):
            return float(seconds), 'total'
        return None, 'unavailable'
    latest = None
    try:
        with (model.parent / 'train_runs.jsonl').open() as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get('model_path') == str(model):
                    latest = record
    except OSError:
        pass
    if latest is not None:
        seconds = latest.get('elapsed_seconds')
        if latest.get('status') == 'complete' and valid(seconds):
            return float(seconds), 'total'
        return None, 'unavailable'
    native = read(model / 'training_time.json')
    start, stop = native.get('start_time'), native.get('stop_time')
    if valid(start) and valid(stop) and stop >= start:
        return float(stop - start), 'total'
    return None, 'unavailable'


def format_training_time(seconds):
    if seconds is None:
        return 'N/A'
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def checkpoints(model, method=None):
    if method == 'neus':
        return sorted(int(p.stem[5:]) for p in (model / 'checkpoints').glob('ckpt_*.pth')
                      if p.stem[5:].isdigit() and p.is_file())
    return sorted(int(p.name.split('_')[-1]) for p in (model / 'point_cloud').glob('iteration_*')
                  if p.name.split('_')[-1].isdigit() and (p/'point_cloud.ply').is_file())


def input_paths(model, method, iteration, mesh_name):
    if method == 'ours':
        return model / 'points_final.ply', model / 'mesh' / mesh_name
    if method == 'neus':
        checkpoint = model / 'checkpoints' / (f'ckpt_{iteration:06d}.pth' if iteration is not None else 'ckpt_missing.pth')
    else:
        checkpoint = model / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
    if method in ('neus', 'gaussian_wrapping'):
        mesh = model / 'meshes' / f'iteration_{iteration}' / mesh_name
    else:
        mesh = model / 'train' / f'ours_{iteration}' / mesh_name
    return checkpoint, mesh


def mean(values):
    numbers = [v for v in values if v is not None]
    return statistics.mean(numbers) if numbers else None


def summaries(rows):
    # Exactly one row per scene/checkpoint: mesh variants never inflate counts.
    by_scene = []
    for scene in sorted({r['scene'] for r in rows}):
        group = [r for r in rows if r['scene'] == scene]
        by_scene.append(dict(scene=scene, measured_checkpoints=sum(r['point_count'] is not None for r in group),
                             average_point_count=mean(r['point_count'] for r in group),
                             evaluated_meshes=sum(r['status']=='complete' for r in group),
                             mean_cd=mean(r.get('cd') for r in group),
                             mean_cd_bbox_percent=mean(r.get('cd_bbox_percent') for r in group)))
    by_iteration = []
    for iteration in sorted({r['iteration'] for r in rows if r['iteration'] is not None}):
        group = [r for r in rows if r['iteration']==iteration]
        by_iteration.append(dict(iteration=iteration, requested_scenes=len(group),
                                 evaluated_scenes=sum(r['status']=='complete' for r in group),
                                 point_count_scenes=sum(r['point_count'] is not None for r in group),
                                 average_point_count=mean(r['point_count'] for r in group),
                                 mean_cd=mean(r.get('cd') for r in group),
                                 mean_cd_bbox_percent=mean(r.get('cd_bbox_percent') for r in group)))
    return dict(per_scene=by_scene, per_iteration=by_iteration,
                average_points_per_scene=mean(r['average_point_count'] for r in by_scene),
                point_count_scene_count=sum(r['average_point_count'] is not None for r in by_scene))


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(method, argv=None):
    default_root, suffix, mesh_name = DEFAULTS[method]
    p=argparse.ArgumentParser(description=f'Evaluate all {method} batch meshes and saved surface-point counts.')
    p.add_argument('--output-root',type=Path,default=Path(default_root),help='Training output root containing scene directories (with method-specific suffixes)')
    p.add_argument('--ground-truth-root',type=Path,default=Path('/home/magnus/phd/models'))
    p.add_argument('--scenes','--scene','--datasets',nargs='+',default=list(SCENES))
    p.add_argument('--iterations',type=int,nargs='+',help='Default: 7000 30000 for 2DGS; latest saved checkpoint otherwise')
    p.add_argument('--mesh-name',default=mesh_name,help='Exact filename in the method\'s extracted-mesh directory (no fallback)')
    p.add_argument('--samples',type=int,default=5_000_000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'evaluation_results'/method)
    p.add_argument('--list-only',action='store_true',help='List selected input paths without computing or writing results')
    a=p.parse_args(argv)
    if method == 'ours' and a.iterations is not None:
        p.error('ours evaluates points_final.ply and mesh/<mesh-name>; --iterations does not apply')
    if a.samples<=0 or (a.iterations is not None and any(i<=0 for i in a.iterations)):
        p.error('samples and iterations must be positive')
    if Path(a.mesh_name).name != a.mesh_name:
        p.error('mesh-name must be a filename')
    names=[]
    for name in a.scenes:
        name=name.removesuffix('_pbdr').removesuffix('_2dgs')
        name='workbench' if name=='workshop' else name
        if '/' in name or '\\' in name or name in ('.','..',''):
            p.error('Invalid scene name')
        if name not in names:names.append(name)
    output=a.output_root.expanduser().resolve()
    gt_root=a.ground_truth_root.expanduser().resolve()
    rows=[]
    if not a.list_only:
        if __package__:
            from .chamfer_ours import load_triangle_mesh_with_query_points, compute_paper_ready_point_to_triangle_distance, set_random_seed
        else:
            from chamfer_ours import load_triangle_mesh_with_query_points, compute_paper_ready_point_to_triangle_distance, set_random_seed
        import numpy as np
    for name in names:
        model=output/(name+suffix)
        available=checkpoints(model, method) if method != 'ours' else []
        iterations=([None] if method == 'ours' else
                    a.iterations or ([7000,30000] if method=='2dgs' else [available[-1] if available else None]))
        gt_path=gt_root/(name+'.ply')
        gt=None
        for iteration in dict.fromkeys(iterations):
            checkpoint, mesh = input_paths(model, method, iteration, a.mesh_name)
            row=dict(method=method,scene=name,iteration=iteration,status='failed',point_count=None,
                     checkpoint=str(checkpoint),reconstruction=str(mesh),ground_truth=str(gt_path))
            if a.list_only:
                print(f'{name} [{iteration}]: mesh={mesh} | checkpoint={checkpoint} | GT={gt_path}')
                continue
            seconds, scope = training_time(model, iteration)
            row.update(training_seconds=seconds, training_time=format_training_time(seconds), training_time_scope=scope)
            time_label = 'Training(total)' if scope == 'total' else 'Training'
            try:
                if iteration is None and method != 'ours':
                    raise FileNotFoundError(f'No saved checkpoints under {model}')
                if not checkpoint.is_file():raise FileNotFoundError(f'Missing checkpoint: {checkpoint}')
                if method != 'neus':
                    row['point_count']=point_count(checkpoint)
                else:
                    row['point_count_note']='Not applicable: implicit SDF network, not optimized surface points'
                if not mesh.is_file():
                    if method == 'neus':
                        preview = model / 'meshes' / f'{iteration:08d}.ply'
                        detail = (' The numbered mesh exists, but may be a resolution-64 training preview in normalized coordinates; it is not used as a fallback.'
                                  if preview.is_file() else '')
                        raise FileNotFoundError(
                            f'Missing evaluation mesh: {mesh}.{detail} '
                            f'Run NeuS extract_mesh_all.py --scenes {name} --output-root {output} first.')
                    raise FileNotFoundError(f'Missing reconstruction: {mesh}')
                extraction_record = mesh.parent / 'batch_mesh.json'
                if method in ('neus', 'gaussian_wrapping') and extraction_record.exists():
                    extraction = json.loads(extraction_record.read_text())
                    if extraction.get('status') != 'complete':
                        raise ValueError(f'Extraction is not marked complete: {extraction_record}')
                if gt is None:
                    set_random_seed(a.seed)
                    gt=load_triangle_mesh_with_query_points(gt_path,a.samples,False)
                gt_mesh,gt_points,_=gt
                diagonal=float(np.linalg.norm(np.asarray(gt_mesh.get_max_bound())-np.asarray(gt_mesh.get_min_bound())))
                if not np.isfinite(diagonal) or diagonal<=0:raise ValueError('Invalid GT bounding-box diagonal')
                set_random_seed(a.seed)
                recon,recon_points,_=load_triangle_mesh_with_query_points(mesh,a.samples,False)
                values=compute_paper_ready_point_to_triangle_distance(recon_points,recon,gt_points,gt_mesh,scale=1.0)
                row.update(cd=values['cd'],accuracy=values['accuracy'],completion=values['completion'],
                           cd_bbox_percent=100*values['cd']/diagonal,gt_bbox_diagonal=diagonal,status='complete')
                count_label = f"{row['point_count']:,}" if row['point_count'] is not None else 'N/A (implicit SDF)'
                print(f"{name} [{iteration}]: CD={row['cd']:.6g}, Accuracy={row['accuracy']:.6g}, Completion={row['completion']:.6g}, Points={count_label}, {time_label}={row['training_time']}",flush=True)
            except Exception as error:
                row['error']=f'{type(error).__name__}: {error}'
                print(f"{name} [{iteration}]: {row['error']}",flush=True)
            rows.append(row)
        del gt
    if a.list_only:return 0
    result=a.results_dir.expanduser().resolve();result.mkdir(parents=True,exist_ok=True)
    summary=summaries(rows)
    payload=dict(method=method,metric='Symmetric unsquared point-to-triangle: (accuracy + completion)/2; uniform surface samples; no alignment or rescaling',
                 samples_per_mesh=a.samples,seed=a.seed,rows=rows,summary=summary)
    (result/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    write_csv(result/'per_scene.csv',rows)
    write_csv(result/'scene_averages.csv',summary['per_scene'])
    write_csv(result/'iteration_averages.csv',summary['per_iteration'])
    if method == 'neus':
        print(f"Average points per scene: N/A (implicit SDF).\nResults: {result}")
    else:
        print(f"Average points per measured scene: {summary['average_points_per_scene']} ({summary['point_count_scene_count']} scenes).\nResults: {result}")
    return int(any(row['status']!='complete' for row in rows) or not rows)
