"""World-space distortion: distance invariance and forward/adjoint agreement."""

import unittest
from unittest import mock

import pale  # Initialize the native runtime before other numerical libraries.
import numpy as np

from config import OptimizationConfig, RendererSettingsConfig, parse_args
from test_renderer_correctness import renderer, surfel


OPTIONS = {'use_depth_distortion': True, 'depth_distortion_weight': 1.0}


def layers(depth=2.0):
    # Camera is at z=2 and looks along -z. The center ray hits both disk centers.
    return [surfel(z=2-depth, su=3, sv=3, opacity=.4),
            surfel(z=2-depth-.25, su=3, sv=3, opacity=.5)]


def one_pixel(points, **settings):
    return renderer(points, width=1, height=1, fx=1, fy=1, cx=0, cy=0,
                    bounces=0, depth_distort_weight=1., **settings)


class WorldSpaceDistortionTests(unittest.TestCase):
    def test_config_and_cli(self):
        self.assertTrue(OptimizationConfig().depth_distort_world_space)
        for flag, expected in (('--depth-distort-world-space', True),
                               ('--no-depth-distort-world-space', False)):
            with mock.patch('sys.argv', ['optimize', flag]):
                config = parse_args()
            self.assertEqual(config.depth_distort_world_space, expected)
            self.assertEqual(RendererSettingsConfig().as_dict(config)['depth_distort_world_space'],
                             expected)

    def test_pair_loss_is_independent_of_camera_distance_only_in_world_space(self):
        captures = {}
        # None omits the setting and verifies backward-compatible default behavior.
        for world_space in (None, False, True):
            values = []
            for depth in (2., 8.):
                settings = {} if world_space is None else {'depth_distort_world_space': world_space}
                with one_pixel(layers(depth), **settings) as instance:
                    frame = instance.render_forward()['camera']
                    value = float(frame['depth_distortion'][0, 0])
                    delta = .25 if world_space else (1000*.2/(1000-.2)) * .25/(depth*(depth+.25))
                    expected = .4 * ((1-.4)*.5) * delta**2
                    np.testing.assert_allclose(value, expected, rtol=2e-4, atol=1e-10)
                    instance.upload_training_targets({'camera': np.zeros((1, 1, 3), np.float32)})
                    loss = instance.render_forward_surface_regularizer_loss_and_adjoint(['camera'], OPTIONS)
                    np.testing.assert_allclose(loss['total_depth_distortion_loss_raw'], value,
                                               rtol=2e-5, atol=1e-10)
                    values.append(value)
            captures[world_space] = values
        np.testing.assert_array_equal(captures[None], captures[False])
        np.testing.assert_allclose(captures[True][0], captures[True][1], rtol=2e-5)
        self.assertLess(captures[False][1], captures[False][0] / 100)

    def test_position_and_opacity_gradients_match_finite_differences(self):
        for world_space in (False, True):
            # Keep geometry inside the optimizer's [-5, 5] position bounds;
            # the zero-rate device step used for refitting still enforces them.
            for depth in (2., 6.):
                with self.subTest(world_space=world_space, depth=depth):
                    with one_pixel(layers(depth), depth_distort_world_space=world_space) as instance:
                        instance.upload_training_targets({'camera': np.zeros((1, 1, 3), np.float32)})
                        instance.render_forward_surface_regularizer_loss_and_adjoint(['camera'], OPTIONS)
                        gradients = instance.render_surface_regularizers_backward_from_current_adjoint(
                            ['camera'], True)['depth_distortion']
                        original = instance.get_point_parameters()
                        for key, index in (('position', (0, 2)), ('position', (1, 2)),
                                           ('opacity', (0,)), ('opacity', (1,))):
                            epsilon = 1e-3
                            losses = []
                            for sign in (-1, 1):
                                params = {k: np.array(v, copy=True) for k, v in original.items()}
                                params[key][index] += sign * epsilon
                                instance.apply_point_optimization(params)
                                instance.apply_device_training_step({})  # Refit after the position upload.
                                losses.append(float(instance.render_forward()['camera']['depth_distortion'][0, 0]))
                            finite_difference = (losses[1]-losses[0])/(2*epsilon)
                            actual = float(np.asarray(gradients[key])[index].item())
                            self.assertGreater(abs(finite_difference), 1e-10)
                            np.testing.assert_allclose(actual, finite_difference, rtol=.015, atol=1e-9,
                                                       err_msg=f'{key} {index}, world={world_space}, depth={depth}')


if __name__ == '__main__':
    unittest.main()
