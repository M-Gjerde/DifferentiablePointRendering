import unittest

import numpy as np
import torch

from training_helpers import position_densification_snapshot_statistics


class DensificationSelectionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
