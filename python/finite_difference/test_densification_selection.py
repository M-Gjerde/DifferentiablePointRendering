import unittest
import contextlib
import io
from unittest import mock

import numpy as np
import torch

from training_helpers import (
    position_densification_snapshot_statistics,
    should_save_image_snapshot,
    should_save_point_cloud_snapshot,
)
from config import OptimizationConfig, parse_args


class DensificationSelectionTests(unittest.TestCase):
    def brightness_snapshot(self, radiance, **overrides):
        radiance = np.asarray(radiance, dtype=np.float32).reshape(-1, 1)
        count = len(radiance)
        options = dict(
            densify_position_grad_accum_np=np.ones((count, 1), np.float32),
            densify_position_grad_denom_np=np.ones((count, 1), np.float32),
            trainable_surfel_mask=torch.ones(count, dtype=torch.bool),
            densification_grad_quantile=0.0,
            densification_grad_abs_min=1.0,
            densify_radiance_rms_accum_np=radiance,
            densification_radiance_floor=0.001,
            densification_radiance_bias_strength=0.25,
        )
        return position_densification_snapshot_statistics(**(options | overrides))

    def test_brightness_bias_changes_selection_after_band_quantiles(self):
        radiance = [0.1] * 4 + [1.0] * 4 + [10.0] * 4
        scores = np.tile([6e-5, 9e-5, 1.1e-4, 1.3e-4], 3).astype(np.float32)[:, None]
        options = dict(
            densify_position_grad_accum_np=scores,
            densification_grad_quantile=0.5,
            densification_grad_abs_min=8e-5,
            densification_radiance_quantile_bins=16,
            densification_radiance_quantile_min_bin_size=2,
        )
        unbiased = self.brightness_snapshot(radiance, **options, densification_radiance_bias_strength=0.0)
        biased = self.brightness_snapshot(radiance, **options)
        np.testing.assert_array_equal(biased[0], unbiased[0])
        np.testing.assert_array_equal(biased[1], unbiased[1])
        np.testing.assert_allclose(biased[2] / unbiased[2], [1 / 1.5] * 4 + [1] * 4 + [1 / 0.8] * 4)
        self.assertEqual(biased[3:], unbiased[3:])
        np.testing.assert_array_equal((unbiased[0] >= unbiased[2]).reshape(3, 4).sum(axis=1), [2, 2, 2])
        np.testing.assert_array_equal((biased[0] >= biased[2]).reshape(3, 4).sum(axis=1), [3, 2, 1])

    def test_bias_adjusts_absolute_threshold_without_radiance_bins(self):
        result = self.brightness_snapshot([0.1, 1.0, 10.0])
        np.testing.assert_allclose(result[2], [1 / 1.5, 1.0, 1 / 0.8])
        np.testing.assert_array_equal(result[0] >= result[2], [True, True, False])

    def test_absolute_mode_bypasses_global_and_band_quantiles(self):
        # Every score exceeds the absolute threshold: quantile=0 would still
        # raise the base threshold to the minimum score (2).
        with mock.patch('numpy.quantile', side_effect=AssertionError('Quantile evaluated')):
            result = self.brightness_snapshot(
                [0.1, 1.0, 10.0],
                densify_position_grad_accum_np=np.array([[2], [3], [4]], np.float32),
                densification_threshold_mode='absolute',
                densification_grad_quantile=0.85,
                densification_radiance_quantile_bins=16,
                densification_radiance_quantile_min_bin_size=1,
                densification_radiance_bias_strength=1.0,
                densification_radiance_bias_min_weight=0.5,
                densification_radiance_bias_max_weight=2.0,
            )
        np.testing.assert_allclose(result[2], [0.5, 1.0, 2.0])
        self.assertEqual(result[3], 1.0)
        self.assertTrue(np.isnan(result[4]))

    def test_absolute_threshold_does_not_depend_on_score_population(self):
        for scores in ([0.1, 0.2, 0.3], [2, 3, 4], [1, 10, 100]):
            result = self.brightness_snapshot(
                [0.1, 1.0, 10.0],
                densify_position_grad_accum_np=np.asarray(scores, np.float32)[:, None],
                densification_threshold_mode='absolute',
            )
            np.testing.assert_allclose(result[2], [1 / 1.5, 1.0, 1 / 0.8])

    def test_absolute_mode_can_select_no_surfels(self):
        result = self.brightness_snapshot(
            [0.1, 1.0, 10.0],
            densify_position_grad_accum_np=np.full((3, 1), 0.1, np.float32),
            densification_threshold_mode='absolute',
        )
        self.assertFalse(np.any(result[0] >= result[2]))

    def test_absolute_mode_without_brightness_data_uses_absolute_threshold(self):
        for radiance in (None, np.zeros((3, 1), np.float32)):
            result = self.brightness_snapshot(
                [0.1, 1.0, 10.0],
                densify_radiance_rms_accum_np=radiance,
                densify_position_grad_accum_np=np.full((3, 1), 10.0, np.float32),
                densification_threshold_mode='absolute',
            )
            np.testing.assert_array_equal(result[2], np.ones(3, np.float32))

    def test_absolute_mode_empty_population(self):
        result = self.brightness_snapshot([], densification_threshold_mode='absolute')
        self.assertEqual(result[2].size, 0)
        self.assertEqual(result[3], 1.0)
        self.assertTrue(np.isnan(result[4]))

    def test_unknown_threshold_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.brightness_snapshot([1.0], densification_threshold_mode='unknown')

    def test_bias_is_bounded_near_black(self):
        result = self.brightness_snapshot([1e-30, 0.1, 1e30])
        np.testing.assert_allclose(result[2], [1 / 1.5, 1.0, 1 / 0.8])
        self.assertTrue(np.isfinite(result[2]).all())

    def test_bias_weight_is_invariant_to_common_gain_above_floor(self):
        radiance = np.array([0.1, 0.5, 1.0, 2.0, 10.0], np.float32)
        reference = self.brightness_snapshot(radiance)[2]
        for gain in (0.1, 10.0, 100.0):
            np.testing.assert_allclose(self.brightness_snapshot(radiance * gain)[2], reference, rtol=1e-6)

    def test_invalid_inactive_and_untrainable_radiance_cannot_shift_median(self):
        radiance = [0.1, 1, 10, 0, np.nan, np.inf, -1, 1e8, 1e-8]
        denom = np.ones((9, 1), np.float32)
        denom[7] = 0
        trainable = torch.ones(9, dtype=torch.bool)
        trainable[8] = False
        result = self.brightness_snapshot(radiance, densify_position_grad_denom_np=denom,
                                          trainable_surfel_mask=trainable)
        np.testing.assert_allclose(result[2], [1 / 1.5, 1, 1 / 0.8] + [1] * 6)

    def test_missing_radiance_and_zero_strength_are_neutral(self):
        for options in (
            {'densify_radiance_rms_accum_np': None},
            {'densify_radiance_rms_accum_np': np.zeros((3, 1), np.float32)},
            {'densification_radiance_bias_strength': 0.0},
        ):
            with self.subTest(options=options):
                result = self.brightness_snapshot([0.1, 1, 10], **options)
                np.testing.assert_array_equal(result[2], np.ones(3, np.float32))

    def test_global_quantile_uses_all_valid_trainable_signals(self):
        signal = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        denominator = np.ones_like(signal)

        _, _, thresholds, global_threshold, quantile_threshold = (
            position_densification_snapshot_statistics(
                densify_position_grad_accum_np=signal,
                densify_position_grad_denom_np=denominator,
                trainable_surfel_mask=torch.ones(4, dtype=torch.bool),
                densification_grad_quantile=0.5,
                densification_grad_abs_min=2.5,
            )
        )

        self.assertEqual(quantile_threshold, 2.5)
        self.assertEqual(global_threshold, 2.5)
        np.testing.assert_array_equal(thresholds, np.full(4, 2.5, np.float32))

    def test_quantiles_are_independent_between_radiance_bands(self):
        signal = np.array(
            [[1.0], [2.0], [3.0], [4.0], [10.0], [20.0], [30.0], [40.0]],
            dtype=np.float32,
        ) * 1.0e-5
        denominator = np.ones_like(signal)
        radiance = np.array(
            [[0.001]] * 4 + [[0.01]] * 4,
            dtype=np.float32,
        )

        _, _, thresholds, global_threshold, _ = (
            position_densification_snapshot_statistics(
                densify_position_grad_accum_np=signal,
                densify_position_grad_denom_np=denominator,
                trainable_surfel_mask=torch.ones(8, dtype=torch.bool),
                densification_grad_quantile=0.5,
                densification_grad_abs_min=1.0e-6,
                densify_radiance_rms_accum_np=radiance,
                densification_radiance_floor=1.0e-3,
                densification_radiance_quantile_bins=4,
                densification_radiance_quantile_min_bin_size=2,
            )
        )

        self.assertAlmostEqual(global_threshold, 7.0e-5, places=10)
        np.testing.assert_allclose(thresholds[:4], 2.5e-5, rtol=1.0e-6)
        np.testing.assert_allclose(thresholds[4:], 2.5e-4, rtol=1.0e-6)


class PointCloudSnapshotScheduleTests(unittest.TestCase):
    def test_saves_first_iteration_when_ply_snapshots_are_enabled(self):
        self.assertTrue(should_save_point_cloud_snapshot(
            1001, 100, 1200, True, first_iteration=True))
        self.assertFalse(should_save_point_cloud_snapshot(
            1001, 0, 1200, True, first_iteration=True))

    def test_saves_before_densification_instead_of_periodically(self):
        saved = [i for i in range(1, 601) if should_save_point_cloud_snapshot(
            i, 100, ((i - 1) // 200 + 1) * 200, True)]
        self.assertEqual(saved, [199, 399, 599])

    def test_uses_scheduled_event_including_resumed_and_delayed_runs(self):
        self.assertTrue(should_save_point_cloud_snapshot(1249, 100, 1250, True))
        self.assertFalse(should_save_point_cloud_snapshot(1199, 100, 1250, True))
        self.assertFalse(should_save_point_cloud_snapshot(1249, 0, 1250, True))
        self.assertFalse(should_save_point_cloud_snapshot(1249, 100, None, True))

    def test_periodic_mode_is_available(self):
        self.assertTrue(should_save_point_cloud_snapshot(200, 100, None, False))
        self.assertFalse(should_save_point_cloud_snapshot(199, 100, 200, False))


class ImageSnapshotScheduleTests(unittest.TestCase):
    def test_saves_immediately_before_actual_densification(self):
        saved = [i for i in range(2, 401) if should_save_image_snapshot(
            i, 100, 400, 250, first_iteration=False)]
        self.assertEqual(saved, [249, 400])

    def test_saves_first_and_final_iteration(self):
        self.assertTrue(should_save_image_snapshot(
            1001, 100, 1300, 1200, first_iteration=True))
        self.assertTrue(should_save_image_snapshot(
            1300, 100, 1300, None, first_iteration=False))

    def test_zero_interval_keeps_first_but_disables_other_automatic_saves(self):
        self.assertTrue(should_save_image_snapshot(
            1, 0, 300, 200, first_iteration=True))
        self.assertFalse(should_save_image_snapshot(
            199, 0, 300, 200, first_iteration=False))
        self.assertFalse(should_save_image_snapshot(
            300, 0, 300, None, first_iteration=False))


class BrightnessBiasConfigTests(unittest.TestCase):
    def test_defaults_and_cli_override(self):
        with mock.patch('sys.argv', ['test']):
            config = parse_args()
        defaults = OptimizationConfig()
        for field in ('densification_radiance_bias_strength',
                      'densification_radiance_bias_min_weight',
                      'densification_radiance_bias_max_weight'):
            self.assertEqual(getattr(config, field), getattr(defaults, field))
        self.assertEqual(config.densification_threshold_mode, 'absolute')
        with mock.patch('sys.argv', ['test', '--densification-radiance-bias-strength', '0',
                                    '--densification-radiance-bias-min-weight', '0.9',
                                    '--densification-radiance-bias-max-weight', '1.25']):
            config = parse_args()
        self.assertEqual(config.densification_radiance_bias_strength, 0.0)
        self.assertEqual(config.densification_radiance_bias_min_weight, 0.9)
        self.assertEqual(config.densification_radiance_bias_max_weight, 1.25)

    def test_threshold_mode_cli_override(self):
        for mode in ('absolute', 'quantile'):
            with mock.patch('sys.argv', ['test', '--densification-threshold-mode', mode]):
                self.assertEqual(parse_args().densification_threshold_mode, mode)
        with mock.patch('sys.argv', ['test', '--densification-threshold-mode', 'unknown']), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args()
            self.assertEqual(raised.exception.code, 2)

    def test_invalid_bias_parameters_are_rejected(self):
        for field, values in (
            ('strength', ['-1', 'nan', 'inf']),
            ('min-weight', ['0', '1.1', 'nan']),
            ('max-weight', ['0.9', 'inf', 'nan']),
        ):
            for value in values:
                with self.subTest(field=field, value=value), mock.patch(
                    'sys.argv', ['test', '--densification-radiance-bias-' + field, value]
                ), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parse_args()
                    self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
