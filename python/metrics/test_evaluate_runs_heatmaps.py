from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import open3d as o3d

from metrics.evaluate_runs import (
    MeshCheckpoint,
    compute_geometry_rows,
    error_heatmap_colors,
    error_heatmap_outputs_are_compatible,
    parse_args,
)


def write_triangle(path: Path, z: float) -> None:
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(
            np.asarray(
                [[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]],
                dtype=np.float64,
            )
        ),
        triangles=o3d.utility.Vector3iVector(
            np.asarray([[0, 1, 2]], dtype=np.int32)
        ),
    )
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
        raise RuntimeError(f"Failed to write test mesh: {path}")


class ErrorHeatmapTests(unittest.TestCase):
    def test_surface_sampling_is_the_cli_default(self) -> None:
        with mock.patch("sys.argv", ["evaluate_runs.py"]):
            self.assertFalse(parse_args().use_vertices)
        with mock.patch("sys.argv", ["evaluate_runs.py", "--use-vertices"]):
            self.assertTrue(parse_args().use_vertices)

    def test_colors_clamp_at_the_requested_maximum(self) -> None:
        colors = error_heatmap_colors(
            np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float32),
            max_distance=1.0,
        )
        self.assertEqual((4, 3), colors.shape)
        np.testing.assert_allclose(colors[2], colors[3])
        self.assertFalse(np.allclose(colors[0], colors[2]))

    def test_compute_geometry_rows_writes_directional_heatmaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reconstruction_path = root / "reconstruction.ply"
            ground_truth_path = root / "ground_truth.ply"
            heatmap_root = root / "heatmaps"
            write_triangle(reconstruction_path, z=0.1)
            write_triangle(ground_truth_path, z=0.0)

            rows = compute_geometry_rows(
                run_dir=root,
                checkpoints=[MeshCheckpoint(100, reconstruction_path)],
                ground_truth_path=ground_truth_path,
                samples=3,
                device_name="cpu",
                seed=0,
                scale=1.0,
                use_vertices=True,
                print_each_score=False,
                error_heatmap_output_root=heatmap_root,
                error_heatmap_max_distance=0.2,
            )

            self.assertEqual(1, len(rows))
            accuracy_path = Path(rows[0]["accuracy_error_heatmap"])
            completion_path = Path(rows[0]["completion_error_heatmap"])
            self.assertTrue(accuracy_path.is_file())
            self.assertTrue(completion_path.is_file())
            self.assertTrue((accuracy_path.parent / "color_scale.png").is_file())
            self.assertTrue(
                error_heatmap_outputs_are_compatible(
                    heatmap_root,
                    iteration=100,
                    requested_max_distance=0.2,
                )
            )

            accuracy_mesh = o3d.io.read_triangle_mesh(str(accuracy_path))
            completion_mesh = o3d.io.read_triangle_mesh(str(completion_path))
            self.assertEqual(3, len(accuracy_mesh.vertex_colors))
            self.assertEqual(3, len(completion_mesh.vertex_colors))


if __name__ == "__main__":
    unittest.main()
