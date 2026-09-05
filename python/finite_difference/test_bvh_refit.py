"""Serial/parallel refit equivalence after real updates and topology changes."""

import unittest
import pale
import numpy as np
from test_renderer_correctness import renderer, light, surfel, rgb

PARAMETERS = ('position', 'rotation', 'scale', 'albedo', 'opacity', 'beta')
UPDATE = dict(zip(('learning_rate_' + p for p in PARAMETERS),
                  (.015, .02, .02, .003, .005, .01)))


def points_for_tree():
    points = [light(.65, .3)]
    for y in range(-2, 3):
        for x in range(-2, 3):
            p = surfel(x=x*.23+.01, y=y*.21, z=.01*x+.013*y,
                       su=.19, sv=.145, opacity=.65, beta=-1.25)
            angle = .03*(x+y+1)
            p[3:7] = [np.cos(angle/2), np.sin(angle/2), 0, 0]
            points.append(p)
    points.append(surfel(x=.16, y=.08, z=.45, su=.11, sv=.08, opacity=.6))
    return points


class BvhRefitTests(unittest.TestCase):
    def assert_arrays_close(self, actual, expected, *, atol=2e-7, rtol=3e-5):
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            if isinstance(actual[key], np.ndarray):
                with self.subTest(array=key):
                    self.assertTrue(np.isfinite(actual[key]).all())
                    # World points are reconstructed by subtracting a ray depth
                    # near two metres; allow a few float32 ULPs of that scale.
                    array_atol = 1e-6 if key == 'median_world_position' else atol
                    np.testing.assert_allclose(actual[key], expected[key], rtol=rtol, atol=array_atol)

    def test_parallel_matches_serial_after_updates_pruning_and_addition(self):
        for shared in (False, True):
            snapshots = []
            for parallel in (False, True):
                sequence = []
                with renderer(points_for_tree(), width=32, height=32, fx=32, fy=32, cx=16, cy=16,
                              bounces=0, parallel_bvh_refit=parallel,
                              share_local_layer_direct_lighting=shared) as instance:
                    initial = instance.get_point_parameters()
                    for iteration in range(5):
                        # Feed identical parameter changes to both refits. Two
                        # independent Adam trajectories can diverge at support
                        # boundaries due to atomic gradient summation order.
                        params = {k: np.ascontiguousarray(v) for k, v in
                                  instance.get_point_parameters().items()}
                        params['position'][1:] += np.array([.013, -.007, .011], np.float32)
                        angle = .07*(iteration+1)
                        params['rotation'][1:] = [np.cos(angle/2), np.sin(angle/2), 0, 0]
                        params['scale'][1:] *= np.array([1.07, .96], np.float32)
                        params['albedo'][1:] *= .98
                        params['opacity'][1:] += .01
                        params['beta'][1:] -= .02
                        instance.apply_point_optimization(params)
                        # Host parameter upload intentionally does not refit.
                        # Use the production optimizer/refit path with zero LR
                        # so it retains these prescribed geometry changes.
                        instance.apply_device_training_step({})
                        frame = instance.render_forward()['camera']
                        source = np.full_like(frame['raw'][..., :3], 1/(32*32*3))
                        gradients, _ = instance.render_backward({'camera': np.ascontiguousarray(source)})
                        sequence.append((frame, gradients, instance.get_point_parameters(),
                                         instance.capture_device_adam_state()))
                        if iteration == 1:
                            # Reorder/reduce a tree after its cached schedule has been used twice.
                            instance.remove_points({'indices': np.array([2, 7, 12, 17], np.int32)})
                        if iteration == 2:
                            params = instance.get_point_parameters()
                            added = {k: np.ascontiguousarray(params[k][1:3]) for k in PARAMETERS}
                            added['position'][:, 0] += .035
                            added['position'][:, 2] += .04
                            instance.add_points({'new': added})
                    for key in PARAMETERS:
                        self.assertGreater(np.max(np.abs(sequence[0][2][key]-initial[key])), 1e-7,
                                           msg=f'{key} must actually change in this regression')
                snapshots.append(sequence)
            with self.subTest(shared=shared):
                for serial, parallel in zip(*snapshots):
                    for expected, actual in zip(serial, parallel):
                        self.assert_arrays_close(actual, expected)

    def test_refit_matches_fresh_rebuild_for_moving_single_leaf_and_tree(self):
        for points in ([light(.6, .3), surfel(beta=-1.25)], points_for_tree()):
            with self.subTest(points=len(points)):
                with renderer(points, width=32, height=32, fx=32, fy=32, cx=16, cy=16,
                              bounces=0, parallel_bvh_refit=True,
                              share_local_layer_direct_lighting=False) as instance:
                    initial = instance.get_point_parameters()
                    instance.upload_training_targets({'camera': np.ascontiguousarray(rgb(instance)*.7)})
                    for iteration in range(3):
                        instance.render_rgb_training_step(['camera'], UPDATE)
                    updated = instance.get_point_parameters()
                    for key in PARAMETERS:
                        self.assertGreater(np.max(np.abs(updated[key]-initial[key])), 1e-7)
                    before = instance.render_forward()['camera']
                    source = np.full_like(before['raw'][..., :3], 1/(32*32*3))
                    gradients_before, _ = instance.render_backward({'camera': np.ascontiguousarray(source)})
                    instance.rebuild_bvh()
                    after = instance.render_forward()['camera']
                    gradients_after, _ = instance.render_backward({'camera': np.ascontiguousarray(source)})
                    self.assert_arrays_close(after, before)
                    self.assert_arrays_close(gradients_after, gradients_before)


if __name__ == '__main__':
    unittest.main()
