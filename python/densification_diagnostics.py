"""Optional observation of native density windows, without extra renders.

Each one-camera step adds its actual device accumulator delta to that camera.
Summing the camera columns therefore recovers the training decision, including
its positive-score-only denominator. Geometry groups are diagnostics, not priors.
"""
from pathlib import Path
import json

import numpy as np


class DensificationDiagnostics:
    def __init__(self, output_dir, camera_names, point_count, start_iteration=0):
        self.output_dir = Path(output_dir) / 'densification_diagnostics'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_names = list(camera_names)
        self.sums = np.zeros((point_count, len(camera_names)), dtype=np.float64)
        self.counts = np.zeros_like(self.sums)
        self.ids = np.arange(point_count, dtype=np.int64)
        self.next_id = point_count
        self.lineage_dir = self.output_dir / 'lineage'
        self.lineage_dir.mkdir(exist_ok=True)
        if any(self.lineage_dir.iterdir()):
            raise FileExistsError('Refusing to mix an existing lineage with new run-local IDs; use a fresh output directory')
        self.snapshot_ids(start_iteration, 'initial')
        self.groups = ('all', 'dark', 'selected')
        self.columns = ['density_diag_window_iteration']
        for group in self.groups:
            self.columns += [f'density_diag_{group}_{key}' for key in
                             ('count', 'score_median', 'score_p90', 'score_max')]
            for camera in camera_names:
                self.columns += [f'density_diag_{group}_{camera}_{key}' for key in
                                 ('score_sum', 'dominant_count')]
        self.values = [''] * len(self.columns)
        (self.output_dir / 'schema.json').write_text(json.dumps({
            'cameras': self.camera_names,
            'dark': 'max linear albedo < 0.1',
            'score_sum': 'Sum over group of camera window sum / total positive observation count',
            'dominant_count': 'Number of positive-score surfels whose largest contribution is this camera',
            'timing': 'Populated only at densification decisions, before topology mutation; blanks otherwise',
            'snapshots': 'Canonical pre-topology row indices, exact selected parent indices, per-camera sums/counts',
            'lineage': 'Stable run-local IDs; retained split source keeps its ID, appended child receives a new ID. Events record both branches. Resumed runs start a new ID namespace.',
        }, indent=2))

    def begin_iteration(self):
        self.values = [''] * len(self.columns)

    def observe(self, camera, before, after):
        slot = self.camera_names.index(camera)
        for key, dest in [('position_sum', self.sums), ('position_count', self.counts)]:
            delta = np.asarray(after[key], dtype=np.float64).reshape(-1) - np.asarray(before[key], dtype=np.float64).reshape(-1)
            if delta.shape[0] != dest.shape[0] or np.any(delta < -1e-9):
                raise RuntimeError('Density diagnostic accumulator reset or changed topology during a step')
            dest[:, slot] += delta

    def snapshot_ids(self, iteration, phase):
        np.save(self.lineage_dir / f'ids_{iteration:05d}_{phase}.npy', self.ids)

    def remap(self, keep, new_count, *, iteration, result=None, prune_reasons=None):
        sources = np.asarray(result['source_index'] if result else [], dtype=np.int64)
        if len(sources) != new_count:
            raise RuntimeError('Missing ordered child-to-parent mapping')
        children = np.arange(self.next_id, self.next_id + new_count, dtype=np.int64)
        event = dict(ids_before=self.ids, keep=np.asarray(keep), source_index=sources,
                     parent_ids=self.ids[sources], child_ids=children)
        if result:
            for key, value in result.get('update_source', {}).items():
                event['source_after_' + key] = value
            for key, value in result.get('new', {}).items():
                event['child_' + key] = value
        for reason, indices in (prune_reasons or {}).items():
            event['pruned_' + reason + '_ids'] = self.ids[np.asarray(indices, dtype=np.int64)]
        self.ids = np.concatenate((self.ids[keep], children))
        self.next_id += new_count
        np.savez_compressed(self.lineage_dir / f'event_{iteration:05d}.npz',
                            **event, ids_after=self.ids)
        for name in ('sums', 'counts'):
            value = getattr(self, name)[keep]
            setattr(self, name, np.concatenate((value, np.zeros((new_count, value.shape[1])))))

    def reset(self):
        self.sums.fill(0)
        self.counts.fill(0)

    def decision(self, iteration, parameters, result, device_sum, device_count,
                 vector_sum=None, threshold=None):
        # Check attribution against the actual selection inputs every time.
        np.testing.assert_allclose(self.sums.sum(axis=1), np.asarray(device_sum).reshape(-1), rtol=2e-6, atol=1e-9)
        np.testing.assert_array_equal(self.counts.sum(axis=1), np.asarray(device_count).reshape(-1))
        p = {k: np.asarray(v.detach().cpu() if hasattr(v, 'detach') else v)
             for k, v in parameters.items()}
        count = self.counts.sum(axis=1)
        contribution = self.sums / np.maximum(count[:, None], 1)
        score = contribution.sum(axis=1)
        selected = np.zeros(len(score), dtype=bool)
        if result is not None:
            selected[np.asarray(result.get('source_index', []), dtype=np.int64)] = True
        surface = p['power'].reshape(-1) == 0
        masks = {'all': surface,
                 'dark': surface & (np.max(p['albedo'], axis=1) < .1),
                 'selected': selected}
        dominant = np.argmax(contribution, axis=1)
        values = [iteration]
        for group in self.groups:
            mask = masks[group]
            quantiles = np.quantile(score[mask], [.5, .9, 1]) if mask.any() else [0., 0., 0.]
            values += [int(mask.sum()), *quantiles]
            for c in range(len(self.camera_names)):
                values += [float(contribution[mask, c].sum()), int(np.sum(mask & (score > 0) & (dominant == c)))]
        self.values = values
        np.savez_compressed(self.output_dir / f'iter_{iteration:05d}.npz',
                            **p, camera_sum=self.sums, camera_count=self.counts,
                            score=score, selected=selected,
                            surfel_ids=self.ids,
                            source_index=np.asarray(result['source_index'] if result else [], dtype=np.int64),
                            vector_sum=np.asarray(vector_sum if vector_sum is not None else []),
                            threshold=np.asarray(threshold if threshold is not None else []),
                            camera_names=np.asarray(self.camera_names))

