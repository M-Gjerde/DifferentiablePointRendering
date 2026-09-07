"""Relative densification sources, radiometric scaling, and optimizer isolation."""

import unittest

import pale  # Initialize the renderer runtime before torch.
import numpy as np

from test_renderer_correctness import renderer, light, surfel, rgb


def scene_points(power=35.):
    return [light(.65, .3, power=power),
            surfel(x=.03, su=.5, sv=.4, opacity=.7, beta=-1.25),
            surfel(x=.07, z=-.002, su=.4, sv=.45, opacity=.5, beta=-1.1),
            surfel(x=.18, y=.1, z=.4, su=.12, sv=.15, opacity=.6)]


def target_for(image):
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    factor = (.85 + .05*xx/image.shape[1] + .03*yy/image.shape[0])[..., None]
    return np.ascontiguousarray(image*factor, dtype=np.float32)


class RelativeDensificationTests(unittest.TestCase):
    def test_regularizer_backward_preserves_relative_statistics(self):
        from config import OptimizationConfig
        from training import make_device_training_step_options

        for ssim in (0., .3):
            with self.subTest(ssim=ssim), renderer(scene_points(), bounces=0) as instance:
                instance.upload_training_targets({'camera': target_for(rgb(instance))})
                options = make_device_training_step_options(
                    OptimizationConfig(ssim_weight=ssim, densification_relative_error=True,
                                       densification_radiance_floor=.01),
                    active_learning_rates={}, camera_batch_scale=1., return_gradient_stats=True)
                _, reference = instance.render_rgb_loss_backward(['camera'], options)
                instance.render_forward_surface_regularizer_loss_and_adjoint(
                    ['camera'], {'use_depth_distortion': True, 'depth_distortion_weight': .1,
                                 'use_normal_consistency': True, 'normal_consistency_weight': .01})
                actual = instance.render_rgb_backward_from_current_forward(['camera'], options)
                for key in ('clone_signal_per_camera', 'clone_radiance_rms_sum_per_camera'):
                    np.testing.assert_allclose(actual['gradient_stats'][key],
                                               reference['gradient_stats'][key],
                                               rtol=4e-5, atol=2e-7)
                counts = np.asarray(actual['gradient_stats']['clone_signal_record_count_per_camera'])
                radiance = np.asarray(actual['gradient_stats']['clone_radiance_rms_sum_per_camera'])
                self.assertTrue(np.any(counts > 0))
                self.assertTrue(np.all(radiance[counts > 0] > 0))

    def test_matches_explicit_frozen_relative_source(self):
        for shared in (False, True):
            for ssim in (0., .3):
                for full_position in (False, True):
                    with self.subTest(shared=shared, ssim=ssim, full_position=full_position):
                        with renderer(scene_points(), bounces=0,
                                      share_local_layer_direct_lighting=shared,
                                      adjoint_passes=2, adjoint_q_null=.5, adjoint_q_reflect=.5) as instance:
                            image = rgb(instance)
                            target = target_for(image)
                            instance.upload_training_targets({'camera': target})
                            ordinary, ordinary_info = instance.render_rgb_loss_backward(
                                ['camera'], {'ssim_weight': ssim})
                            relative, relative_info = instance.render_rgb_loss_backward(
                                ['camera'], {'ssim_weight': ssim, 'densification_relative_error': True,
                                             'densification_full_position': full_position,
                                             'densification_radiance_floor': .01})
                            for key in ('position', 'rotation', 'scale', 'albedo', 'opacity', 'beta'):
                                np.testing.assert_allclose(relative[key], ordinary[key], rtol=3e-5, atol=2e-7)
                            # Check the actual uploaded target, not only agreement
                            # between two passes sharing the same device buffer.
                            expected_half_mse = .5*np.mean((image.astype(np.float64)-target)**2)
                            for info in (ordinary_info, relative_info):
                                actual_half_mse = info['l2_loss_values']['camera']
                                self.assertTrue(np.isfinite(actual_half_mse))
                                np.testing.assert_allclose(actual_half_mse, expected_half_mse,
                                                           rtol=3e-6, atol=1e-12)
                            self.assertAlmostEqual(relative_info['loss_values']['camera'],
                                                   ordinary_info['loss_values']['camera'], delta=2e-7)
                            denominator = ((image.astype(float)**2 + target.astype(float)**2).sum(axis=2)/6 + .01**2)
                            source = np.ascontiguousarray((image-target)/(image.size*denominator[..., None]),
                                                          dtype=np.float32)
                            expected, expected_info = instance.render_backward({'camera': source})
                            key = 'position' if full_position else 'clone_signal'
                            np.testing.assert_allclose(relative['clone_signal'], expected[key], rtol=4e-5, atol=2e-7)
                            expected_stats = expected_info['gradient_stats']
                            actual_stats = relative_info['gradient_stats']
                            np.testing.assert_allclose(actual_stats['clone_signal_per_camera'],
                                                       expected_stats[key+'_per_camera'], rtol=4e-5, atol=2e-7)
                            # The accumulator uses these counts only as an
                            # eligibility mask. Relative statistics retain tiny
                            # terms that absolute photometric culling may skip.
                            np.testing.assert_array_equal(actual_stats['clone_signal_record_count_per_camera'] > 0,
                                                          expected_stats[key+'_record_count_per_camera'] > 0)
                            counts = np.asarray(
                                actual_stats['clone_signal_record_count_per_camera'],
                                dtype=np.uint32,
                            )
                            radiance_sums = np.asarray(
                                actual_stats['clone_radiance_rms_sum_per_camera'],
                                dtype=np.float32,
                            )
                            self.assertEqual(radiance_sums.shape, counts.shape)
                            active = counts > 0
                            self.assertTrue(np.all(np.isfinite(radiance_sums)))
                            self.assertTrue(np.all(radiance_sums[active] > 0.0))
                            pixel_rms_max = np.sqrt(
                                np.max((image.astype(float) ** 2 + target.astype(float) ** 2).sum(axis=2) / 6)
                            )
                            self.assertTrue(np.all(
                                radiance_sums[active] / counts[active] <= pixel_rms_max * 1.001
                            ))

    def test_common_gain_cancels_above_radiance_floor(self):
        signals, ordinary = {}, {}
        for gain in (1e-5, .1, 1., 100.):
            with renderer([light(.65, .3, power=35*gain), surfel(beta=-1.25)],
                          bounces=0, share_local_layer_direct_lighting=False) as instance:
                image = rgb(instance)
                target = image.copy()
                target[32, 37] *= .9  # One nonzero residual, away from support/visibility edges.
                instance.upload_training_targets({'camera': target})
                gradients, _ = instance.render_rgb_loss_backward(
                    ['camera'], {'densification_relative_error': True, 'densification_radiance_floor': 1e-10})
                signals[gain] = np.asarray(gradients['clone_signal'])[1].copy()
                ordinary[gain] = np.asarray(gradients['position'])[1].copy()
        self.assertGreater(np.linalg.norm(signals[1.]), 1e-7)
        for gain in (1e-5, .1, 100.):
            np.testing.assert_allclose(signals[gain], signals[1.], rtol=6e-5, atol=2e-8)
        for gain in (.1, 100.):
            np.testing.assert_allclose(ordinary[gain]/gain**2, ordinary[1.], rtol=6e-5, atol=2e-10)

    def test_device_optimizer_ignores_relative_statistic(self):
        for ssim in (0., .3):
            parameters, states = [], []
            for enabled in (False, True):
                with renderer(scene_points(), bounces=0,
                              share_local_layer_direct_lighting=True) as instance:
                    instance.upload_training_targets({'camera': target_for(rgb(instance))})
                    instance.render_rgb_training_step(['camera'], {
                        'ssim_weight': ssim, 'densification_relative_error': enabled,
                        'densification_radiance_floor': .01, 'return_gradient_stats': True,
                        'learning_rate_position': 1e-5, 'learning_rate_rotation': 1e-4,
                        'learning_rate_scale': 1e-5, 'learning_rate_albedo': 1e-4,
                        'learning_rate_opacity': 1e-4, 'learning_rate_beta': 1e-4,
                    })
                    parameters.append(instance.get_point_parameters())
                    states.append(instance.capture_device_adam_state())
            with self.subTest(ssim=ssim):
                for key in parameters[0]:
                    np.testing.assert_allclose(parameters[1][key], parameters[0][key], rtol=2e-6, atol=2e-7)
                for key in states[0]:
                    if isinstance(states[0][key], np.ndarray):
                        np.testing.assert_allclose(states[1][key], states[0][key], rtol=4e-5, atol=2e-8)

    def test_relative_statistics_replace_albedo_boost(self):
        import torch
        from training_helpers import update_densification_statistics
        for relative in (False, True):
            accumulator = np.zeros((2, 1), np.float32)
            denominator = np.zeros((2, 1), np.float32)
            vector = np.zeros((2, 3), np.float32)
            radiance = np.zeros((2, 1), np.float32)
            update_densification_statistics(
                iteration=2, densification_interval=200, densification_cycle_start_iteration=0,
                densification_stats_skip_iterations=0,
                densify_position_grad_accum_np=accumulator, densify_position_grad_denom_np=denominator,
                densify_position_grad_vector_accum_np=vector,
                densify_radiance_rms_accum_np=radiance,
                rotations=torch.tensor([[1.,0,0,0], [1.,0,0,0]]),
                albedos=torch.tensor([[.1,.1,.1], [.9,.9,.9]]), trainable_surfel_mask=torch.tensor([True, True]),
                densify_bsdf_floor=.01, densify_bsdf_gamma=1.,
                densify_position_grad_per_camera_np=np.array([[[3,4,12], [0,10,0]]]*2, np.float32),
                densify_position_grad_per_camera_count_np=np.ones((2,2), np.uint32),
                densify_radiance_rms_sum_per_camera_np=np.ones((2,2), np.float32),
                densification_tangent_only=True, densification_relative_error=relative,
            )
            scale = np.ones(2) if relative else np.array([.1, .9])
            np.testing.assert_allclose(accumulator[:, 0], 7.5/scale, rtol=1e-6)
            np.testing.assert_allclose(vector, np.array([[1.5,7,0]]*2)/scale[:, None], rtol=1e-6)
            np.testing.assert_array_equal(denominator, np.ones((2, 1)))
            if relative:
                np.testing.assert_array_equal(radiance, np.ones((2, 1), np.float32))

    def test_ssim_auxiliary_buffers_follow_pruning(self):
        with renderer(scene_points(), bounces=0,
                      share_local_layer_direct_lighting=True) as instance:
            options = {'ssim_weight': .3, 'densification_relative_error': True,
                       'densification_radiance_floor': .01}
            for expected_count in (4, 3):
                image = rgb(instance)
                target = target_for(image)
                instance.upload_training_targets({'camera': target})
                relative, _ = instance.render_rgb_loss_backward(['camera'], options)
                self.assertEqual(np.asarray(relative['clone_signal']).shape, (expected_count, 3))
                denominator = ((image.astype(float)**2 + target.astype(float)**2).sum(axis=2)/6 + .01**2)
                source = np.ascontiguousarray((image-target)/(image.size*denominator[..., None]), dtype=np.float32)
                expected, _ = instance.render_backward({'camera': source})
                np.testing.assert_allclose(relative['clone_signal'], expected['position'], rtol=5e-5, atol=2e-7)
                if expected_count == 4:
                    instance.remove_points({'indices': np.array([2], dtype=np.int32)})

    def test_floor_limits_amplification_near_black(self):
        with renderer([light(.65, .3, power=35e-4), surfel(beta=-1.25)],
                      bounces=0, share_local_layer_direct_lighting=False) as instance:
            image = rgb(instance)
            target = image.copy()
            target[32, 37] *= .9
            instance.upload_training_targets({'camera': target})
            norms = []
            for floor in (1e-10, .01):
                gradients, _ = instance.render_rgb_loss_backward(
                    ['camera'], {'densification_relative_error': True, 'densification_radiance_floor': floor})
                self.assertTrue(np.all(np.isfinite(gradients['clone_signal'])))
                norms.append(np.linalg.norm(gradients['clone_signal']))
            self.assertGreater(norms[0], 1e-7)
            self.assertLess(norms[1], norms[0]*1e-4)


if __name__ == '__main__':
    unittest.main()
