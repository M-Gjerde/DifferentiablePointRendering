"""Demand-driven curvature preserves RGB and every active regularizer."""

import math
import unittest

import numpy as np

from test_renderer_correctness import light, renderer, surfel


def curved_points():
    points = [light(x=.5)]
    for x, angle in ((-.22, -.45), (0., 0.), (.22, .45)):
        point = surfel(x=x, su=.8, sv=.7, opacity=.95, beta=-1.5)
        point[3:7] = [math.cos(angle / 2), 0, math.sin(angle / 2), 0]
        points.append(point)
    return points


OPTIONS = {
    'use_depth_distortion': True, 'depth_distortion_weight': 10.,
    'use_normal_consistency': True, 'normal_consistency_weight': .005,
    'use_intra_slab_depth': True, 'intra_slab_depth_weight': 1e-5,
}


class CurvatureWorkTests(unittest.TestCase):
    def test_no_consumers_preserves_rgb_and_active_gradients(self):
        captures = []
        for diagnostics in (False, True):
            with renderer(curved_points(), width=24, height=24,
                          fx=24, fy=24, cx=12, cy=12,
                          compute_curvature_diagnostics=diagnostics,
                          enable_curvature_densification=False) as instance:
                frame = instance.render_forward()['camera']
                seed = np.full_like(frame['raw'][..., :3], 1 / (24 * 24 * 3))
                photo = instance.render_backward({'camera': seed})[0]
                instance.upload_training_targets({'camera': np.zeros_like(seed)})
                losses = instance.render_forward_surface_regularizer_loss_and_adjoint(
                    ['camera'], OPTIONS)
                gradients = instance.render_surface_regularizers_backward_from_current_adjoint(
                    ['camera'], True)
                captures.append((frame, photo, losses, gradients))
        off, on = captures
        for name in off[0]:
            if name in ('curvature_scale', 'curvature_scale_active_slab_count'):
                self.assertEqual(np.count_nonzero(off[0][name]), 0)
                self.assertGreater(np.count_nonzero(on[0][name]), 0)
            else:
                np.testing.assert_allclose(off[0][name], on[0][name], rtol=2e-5, atol=2e-7)
        for name in off[1]:
            np.testing.assert_allclose(off[1][name], on[1][name], rtol=3e-5, atol=2e-7)
        for term in off[3]:
            for name in off[3][term]:
                np.testing.assert_allclose(off[3][term][name], on[3][term][name],
                                           rtol=3e-5, atol=2e-7)

    def test_densification_without_regularizer_keeps_curvature(self):
        with renderer(curved_points(), width=24, height=24,
                      fx=24, fy=24, cx=12, cy=12,
                      curvature_scale_weight=0.,
                      enable_curvature_densification=True) as instance:
            frame = instance.render_forward()['camera']
            stats = instance.get_curvature_densification_stats()
            self.assertGreater(np.count_nonzero(frame['curvature_scale']), 0)
            self.assertGreater(int(stats['violation_count'].sum()), 0)
            self.assertGreater(float(stats['violation_sum'].sum()), 0)

    def test_regularizer_alone_and_disabling_clear_outputs(self):
        with renderer(curved_points(), width=24, height=24,
                      fx=24, fy=24, cx=12, cy=12,
                      enable_curvature_densification=False) as instance:
            instance.upload_training_targets({'camera': np.zeros((24, 24, 3), np.float32)})
            active_options = OPTIONS | {'use_curvature_scale': True, 'curvature_scale_weight': 1e-6}
            active = instance.render_forward_surface_regularizer_loss_and_adjoint(
                ['camera'], active_options)
            self.assertGreater(active['total_curvature_scale_loss_weighted'], 0)
            active_gradients = instance.render_surface_regularizers_backward_from_current_adjoint(
                ['camera'], True)
            self.assertGreater(np.count_nonzero(active_gradients['curvature_scale']['scale']), 0)
            inactive = instance.render_forward_surface_regularizer_loss_and_adjoint(
                ['camera'], OPTIONS)
            self.assertEqual(inactive['total_curvature_scale_loss_raw'], 0)
            inactive_gradients = instance.render_surface_regularizers_backward_from_current_adjoint(
                ['camera'], True)
            for name, values in inactive_gradients['curvature_scale'].items():
                self.assertEqual(np.count_nonzero(values), 0, name)
            frame = instance.render_forward()['camera']
            self.assertEqual(np.count_nonzero(frame['curvature_scale']), 0)
            self.assertEqual(np.count_nonzero(frame['curvature_scale_active_slab_count']), 0)


if __name__ == '__main__':
    unittest.main()
