from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from finite_difference.fd_common import (
    PARAMETERS,
    _save_rgb_artifact,
    compare_gradients,
    load_suite,
    scene_paths,
    validate_case,
)


CHECK = {
    "relative_tolerance": 0.05,
    "absolute_tolerance": 1.0e-8,
    "minimum_signal": 1.0e-6,
    "fd_consistency_relative_tolerance": 0.1,
    "repeatability_tolerance": 0.0,
}


class GradientComparisonTests(unittest.TestCase):
    def test_missing_analytic_gradient_is_a_failure(self) -> None:
        result = compare_gradients(0.0, 1.0e-3, CHECK)
        self.assertFalse(result["pass"])

    def test_missing_finite_difference_gradient_is_a_failure(self) -> None:
        result = compare_gradients(1.0e-3, 0.0, CHECK)
        self.assertFalse(result["pass"])

    def test_nonfinite_values_are_failures(self) -> None:
        self.assertFalse(compare_gradients(math.nan, 1.0, CHECK)["pass"])
        self.assertFalse(compare_gradients(1.0, math.inf, CHECK)["pass"])

    def test_zero_signal_requires_explicit_permission(self) -> None:
        self.assertFalse(compare_gradients(0.0, 0.0, CHECK)["pass"])
        self.assertTrue(compare_gradients(0.0, 0.0, CHECK, allow_zero_signal=True)["pass"])

    def test_combined_absolute_and_relative_tolerance(self) -> None:
        self.assertTrue(compare_gradients(1.0, 0.97, CHECK)["pass"])
        self.assertFalse(compare_gradients(1.0, 0.90, CHECK)["pass"])


class ImageArtifactTests(unittest.TestCase):
    def test_saves_exact_linear_array_and_viewable_png(self) -> None:
        image = np.asarray([[[0.0, 0.18, 1.0], [2.0, 0.01, 0.5]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            array_path, preview_path = _save_rgb_artifact(
                image,
                Path(directory) / "images" / "baseline",
            )
            self.assertTrue(array_path.is_file())
            self.assertTrue(preview_path.is_file())
            np.testing.assert_array_equal(np.load(array_path), image)


class CaseValidationTests(unittest.TestCase):
    def _case(self) -> dict:
        return {
            "name": "case",
            "stage": "stage",
            "scene": "fd_direct",
            "ply": "single_surfel",
            "camera": "camera",
            "parameter": "opacity",
            "index": 1,
            "values": [0.5],
            "epsilons": [1.0e-3, 3.0e-4],
            "objective": {"type": "linear"},
            "settings": {
                "bounces": 1,
                "forward_passes": 1,
                "adjoint_bounces": 1,
                "adjoint_passes": 1,
                "enable_adjoint_shadow_rays": True,
                "adjoint_shadow_path_rays": 1,
                "adjoint_q_null": 0.0,
                "adjoint_q_reflect": 1.0,
                "share_local_layer_direct_lighting": False,
                "minimum_projected_footprint": False,
            },
            "check": dict(CHECK),
        }

    def test_valid_case(self) -> None:
        validate_case(self._case())

    def test_invalid_sampling_probability_is_rejected(self) -> None:
        case = self._case()
        case["settings"]["adjoint_q_null"] = 0.5
        case["settings"]["adjoint_q_reflect"] = 1.0
        with self.assertRaisesRegex(ValueError, r"q_null \+ q_reflect"):
            validate_case(case)

    def test_noncentral_bounded_stencil_is_rejected(self) -> None:
        case = self._case()
        case["values"] = [0.0]
        with self.assertRaisesRegex(ValueError, "central stencil"):
            validate_case(case)


class DirectSuiteCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite_path = Path(__file__).with_name("tests_direct.json")
        _, cls.cases = load_suite(cls.suite_path)

    def test_every_parameter_component_is_covered(self) -> None:
        covered = {str(case["parameter"]) for case in self.cases}
        self.assertEqual(PARAMETERS, covered)

    def test_progressive_geometry_stages_are_covered(self) -> None:
        covered = {str(case["stage"]) for case in self.cases}
        self.assertEqual(
            {
                "single_surfel",
                "single_slab",
                "multiple_slabs",
                "shadow_occlusion",
                "slab_occlusion",
                "minimum_footprint",
            },
            covered,
        )

    def test_important_renderer_branches_are_both_covered(self) -> None:
        settings = [case["settings"] for case in self.cases]
        self.assertEqual({False, True}, {bool(value["share_local_layer_direct_lighting"]) for value in settings})
        self.assertEqual({False, True}, {bool(value["minimum_projected_footprint"]) for value in settings})
        self.assertIn(1, {int(value["point_hit_batch_size"]) for value in settings})
        self.assertTrue(any(int(value["point_hit_batch_size"]) > 1 for value in settings))
        self.assertEqual({False, True}, {bool(value["point_hit_batch_lookahead"]) for value in settings})
        self.assertEqual({False, True}, {bool(value["enable_adjoint_shadow_rays"]) for value in settings})

    def test_fixture_files_and_ply_counts_are_valid(self) -> None:
        checked_plys: set[Path] = set()
        for case in self.cases:
            scene_xml, pointcloud_ply = scene_paths(case)
            self.assertTrue(scene_xml.is_file(), scene_xml)
            self.assertTrue(pointcloud_ply.is_file(), pointcloud_ply)
            if pointcloud_ply in checked_plys:
                continue
            checked_plys.add(pointcloud_ply)
            lines = [line.strip() for line in pointcloud_ply.read_text().splitlines()]
            vertex_line = next(line for line in lines if line.startswith("element vertex "))
            expected_count = int(vertex_line.split()[-1])
            header_end = lines.index("end_header")
            records = [line for line in lines[header_end + 1:] if line and not line.startswith("comment")]
            self.assertEqual(expected_count, len(records), pointcloud_ply)
            self.assertTrue(all(len(record.split()) == 16 for record in records), pointcloud_ply)


if __name__ == "__main__":
    unittest.main()
