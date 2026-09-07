"""Behavior checks for reusable loss buffers and training memory lifetimes."""

import copy
from pathlib import Path
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import pale
import numpy as np

from test_renderer_correctness import SETTINGS, light, renderer, surfel


class RuntimeOptimizationTests(unittest.TestCase):
    def test_regularizer_scratch_survives_batch_growth_and_disabled_losses(self):
        points = [surfel(opacity=.4), surfel(z=-.01, opacity=.5),
                  surfel(z=-.4, opacity=.6)]
        with renderer(points, width=8, height=8, fx=8, fy=8, cx=4, cy=4):
            # Add a second camera with different compositing weights, so stale
            # slots and incorrect offsets cannot pass by returning equal losses.
            assets = Path.cwd()
            scene_path = assets / "scene.xml"
            scene = ET.parse(scene_path)
            camera = copy.deepcopy(scene.getroot().find("sensor"))
            camera.set("id", "other")
            camera.find("transform/lookat").set("origin", "0.2,0,2")
            camera.find("transform/lookat").set("target", "0.2,0,0")
            scene.getroot().append(camera)
            scene.write(scene_path)
            instance = pale.Renderer(str(assets), str(scene_path), str(assets / "points.ply"),
                                     SETTINGS | {"depth_distort_world_space": True})
            try:
                instance.upload_training_targets({name: np.zeros((8, 8, 3), np.float32)
                                                  for name in ("camera", "other")})
                options = {"use_depth_distortion": True, "depth_distortion_weight": 1.,
                           "use_normal_consistency": True, "normal_consistency_weight": 1.,
                           "use_opacity_prior": True, "opacity_prior_weight": 1.,
                           "use_intra_slab_depth": True, "intra_slab_depth_weight": 1.,
                           "use_curvature_scale": True, "curvature_scale_weight": 1.}
                expected = instance.render_forward()
                depth = {name: float(frame["depth_distortion"].mean())
                         for name, frame in expected.items()}
                self.assertGreater(depth["camera"], 0.)
                self.assertNotAlmostEqual(depth["camera"], depth["other"], places=8)
                baseline = {}
                for names in (["camera"], ["camera", "other"], ["other"],
                              ["camera", "other"], ["camera"]):
                    result = instance.render_forward_surface_regularizer_loss_and_adjoint(names, options)
                    np.testing.assert_allclose(result["total_depth_distortion_loss_raw"],
                                               sum(depth[name] for name in names), rtol=2e-5)
                    for name, values in result["per_camera_loss_values"].items():
                        if name in baseline:
                            for key, value in values.items():
                                np.testing.assert_allclose(value, baseline[name][key], rtol=2e-5, atol=1e-8)
                        else:
                            baseline[name] = dict(values)
                    disabled = instance.render_forward_surface_regularizer_loss_and_adjoint(names, {})
                    for values in disabled["per_camera_loss_values"].values():
                        self.assertTrue(all(value == 0. for value in values.values()))
            finally:
                del instance

    def test_debug_images_preserve_gradients_before_and_after_rebuild(self):
        captures = []
        for debug in (False, True):
            with renderer([light(), surfel(opacity=.4)], width=4, height=4,
                          fx=4, fy=4, cx=2, cy=2, debug_images=debug) as instance:
                instance.upload_training_targets({"camera": np.zeros((4, 4, 3), np.float32)})
                for rebuild in (False, True):
                    if rebuild:
                        instance.rebuild_bvh()
                    gradients, _ = instance.render_rgb_loss_backward(["camera"])
                    captures.append({key: np.array(value, copy=True) for key, value in gradients.items()})
        for capture in captures[1:]:
            for key, expected in captures[0].items():
                np.testing.assert_allclose(capture[key], expected, rtol=2e-4, atol=1e-7,
                                           err_msg=key)

    def test_both_training_paths_save_outputs_after_releasing_images(self):
        # Run through actual initial, iteration, and final output consumers;
        # releasing an image too early must fail here rather than in a long run.
        from config import OptimizationConfig, RendererSettingsConfig
        import training

        for device_step in (False, True):
            with self.subTest(device_step=device_step):
                with renderer([light(), surfel(opacity=.4), surfel(z=-.2, opacity=.5)],
                              width=4, height=4, fx=4, fy=4, cx=2, cy=2,
                              depth_distort_world_space=True) as instance:
                    output = Path.cwd() / "training_output"
                    config = OptimizationConfig(
                        output_dir=output, iterations=3, device="cpu",
                        use_device_training_step=device_step,
                        densification_interval=0, prune_interval=0,
                        inactive_transport_prune_cycles=0, reset_opacity_interval=0,
                        mesh_extraction_interval=0, save_final_mesh=False,
                        enable_metrics=False, enable_image_preview=False,
                        save_interval=2, save_ply_files_interval=2,
                        save_ply_before_densification=False,
                        depth_distort_weight=.1, normal_consistency_weight=.01,
                        intra_slab_depth_weight=0., curvature_scale_weight=0.)
                    target = {"camera": np.zeros((4, 4, 3), np.float32)}
                    with mock.patch.object(training.helpers, "load_target_images",
                                           return_value=(target, ["camera"], ["camera"])):
                        training.run_optimization(instance, config, RendererSettingsConfig())
                    self.assertTrue((output / "points_final.ply").is_file())
                    for value in instance.get_point_parameters().values():
                        self.assertTrue(np.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()
