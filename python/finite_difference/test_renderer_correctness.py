"""Deterministic regressions for transport, camera, BVH, and optimizer fixes.

Requires the freshly built pale module on PYTHONPATH. All renderer assets and
registry writes are isolated in temporary directories. Use ACPP_VISIBILITY_MASK=omp
to run with AdaptiveCpp's CPU backend when a GPU is unavailable.
"""

from contextlib import contextmanager
import gc
import math
import os
from pathlib import Path
import tempfile
import unittest
import weakref

import pale  # Load the native runtime before torch's bundled libraries.
import numpy as np


SETTINGS = {
    "photons": 1, "bounces": 1, "forward_passes": 1,
    "primal_shadow_rays": 1, "adjoint_shadow_rays": 1,
    "gather_passes": 1, "adjoint_bounces": 1, "adjoint_passes": 1,
    "logging": 4, "seed": 42, "enable_adjoint_shadow_rays": True,
    "adjoint_shadow_path_rays": 1, "adjoint_q_null": 0.0,
    "adjoint_q_reflect": 1.0, "share_local_layer_direct_lighting": False,
    "local_layer_depth_epsilon": 0.02, "max_splat_events_per_ray": 8,
    "max_local_surfel_hits": 8, "point_hit_batch_size": 6,
    "point_hit_batch_lookahead": True,
}


def light(x=0.0, y=0.0, z=1.3, power=35.0):
    return [x, y, z, 0, 0, 0, 0, 0, 0, 1, 0.9, 0.8, 0, 0, 0, power]


def surfel(x=0.0, y=0.0, z=0.0, su=0.5, sv=0.5,
           albedo=(0.7, 0.5, 0.3), opacity=0.5, beta=0.0):
    return [x, y, z, 1, 0, 0, 0, su, sv, *albedo, opacity, beta, 0, 0]


@contextmanager
def renderer(points, *, width=64, height=64, fx=64, fy=64, cx=32, cy=32,
             **settings):
    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="pale-correctness-") as directory:
        assets = Path(directory)
        scene = assets / "scene.xml"
        scene.write_text(f'''<scene version="3.0.0">
  <sensor type="perspective" id="camera">
    <string name="camera_model" value="pinhole_intrinsics"/>
    <float name="near_clip" value="0.01"/>
    <float name="far_clip" value="20"/>
    <float name="fx" value="{fx}"/><float name="fy" value="{fy}"/>
    <float name="cx" value="{cx}"/><float name="cy" value="{cy}"/>
    <transform name="to_world">
      <lookat origin="0,0,2" target="0,0,0" up="0,1,0"/>
    </transform>
    <film type="hdrfilm">
      <integer name="width" value="{width}"/>
      <integer name="height" value="{height}"/>
      <rfilter type="box"/>
    </film>
  </sensor>
</scene>''')
        ply = assets / "points.ply"
        names = "x y z rot_w rot_x rot_y rot_z su sv albedo_r albedo_g albedo_b opacity beta shape power"
        header = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
        header += "".join(f"property float {name}\n" for name in names.split())
        ply.write_text(header + "end_header\n" +
                       "".join(" ".join(map(str, row)) + "\n" for row in points))
        instance = None
        try:
            instance = pale.Renderer(str(assets), str(scene), str(ply), SETTINGS | settings)
            yield weakref.proxy(instance)
        finally:
            del instance
            gc.collect()
            os.chdir(original_directory)


def rgb(instance):
    return np.ascontiguousarray(instance.render_forward()["camera"]["raw"][..., :3])


class RendererCorrectnessTests(unittest.TestCase):
    def test_shared_slab_opacity_gradients_match_order_average(self):
        # For up to eight coincident members, four-point Gauss quadrature
        # integrates the order-average weight polynomial exactly. Evaluate its
        # product form independently of the renderer's coefficient expansion.
        nodes, quadrature = np.polynomial.legendre.leggauss(4)
        zeta, quadrature = (nodes + 1) / 2, quadrature / 2
        channel_seed = np.array([0.3, -0.4, 0.7])
        for count in (2, 8):
            with self.subTest(members=count):
                opacity = np.linspace(0.05, 0.995, count)
                albedo = np.column_stack((np.linspace(.2, .8, count),
                                          np.linspace(.7, .3, count),
                                          np.linspace(.1, .6, count)))
                weights = np.zeros(count)
                weight_derivatives = np.zeros((count, count))
                for i in range(count):
                    other = [j for j in range(count) if j != i]
                    integral = np.dot(quadrature, np.prod(1-zeta[:, None]*opacity[other], axis=1))
                    weights[i] = opacity[i] * integral
                    for k in range(count):
                        if i == k:
                            weight_derivatives[i, k] = integral
                        else:
                            remaining = [j for j in other if j != k]
                            product = np.prod(1-zeta[:, None]*opacity[remaining], axis=1)
                            weight_derivatives[i, k] = -opacity[i]*np.dot(quadrature, zeta*product)
                irradiance_over_pi = np.array([1, .9, .8])*35/(4*math.pi**2*1.3**2)
                member_radiance = albedo * irradiance_over_pi
                points = [light()] + [surfel(opacity=opacity[i], albedo=albedo[i], beta=-1.25)
                                      for i in range(count)]
                with renderer(points, bounces=0, share_local_layer_direct_lighting=True) as instance:
                    image = rgb(instance)
                    np.testing.assert_allclose(image[32, 32], weights @ member_radiance,
                                               rtol=2e-5, atol=2e-7)
                    seed = np.zeros_like(image)
                    seed[32, 32] = channel_seed
                    gradients = instance.render_backward({"camera": seed})[0]
                    expected_opacity = weight_derivatives.T @ member_radiance @ channel_seed
                    np.testing.assert_allclose(np.asarray(gradients["opacity"])[1:],
                                               expected_opacity, rtol=2e-4, atol=2e-7)
                    expected_albedo = weights[:, None] * irradiance_over_pi * channel_seed
                    np.testing.assert_allclose(np.asarray(gradients["albedo"])[1:],
                                               expected_albedo, rtol=2e-4, atol=2e-7)

    def test_forward_overwrites_outputs_after_surface_leaves_view(self):
        with renderer([light(), surfel(opacity=0.9)]) as instance:
            first = instance.render_forward()["camera"]
            self.assertGreater(float(np.asarray(first["raw"])[..., :3].max()), 0)
            instance.set_point_translation(10.0, 0, 1)
            empty = instance.render_forward()["camera"]
            for name in ("raw", "median_depth", "mean_depth", "median_world_position",
                         "visible_normal", "normal_from_depth", "depth_distortion",
                         "opacity_prior", "intra_slab_depth", "curvature_scale",
                         "intra_slab_depth_active_slab_count", "curvature_scale_active_slab_count"):
                with self.subTest(buffer=name):
                    self.assertEqual(np.count_nonzero(empty[name]), 0)

    def test_calibrated_rectangular_camera_matches_closed_form(self):
        # One unoccluded Lambertian sheet, no stochastic path or visibility edge
        # derivative: I = eta*g*c*Phi*light_color*cos(theta)/(4*pi^2*r^2).
        with renderer([light(), surfel()], width=96, height=64,
                      fx=87, fy=73, cx=40, cy=29) as instance:
            actual = rgb(instance)
            yy, xx = np.mgrid[:64, :96]
            x, y = 2 * (xx - 40) / 87, 2 * (29 - yy) / 73
            profile = np.maximum(1 - (x / 0.5)**2 - (y / 0.5)**2, 0)**4
            geometry = 1.3 / (x*x + y*y + 1.3**2)**1.5
            color = np.array([0.7, 0.5, 0.3]) * np.array([1, 0.9, 0.8])
            expected = (0.5 * profile * geometry * 35 / (4 * math.pi**2))[..., None] * color
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-7)

    def test_bvh_refit_and_rebuild_preserve_full_profile_support(self):
        # Separate light and surface leaves so a cutoff cannot hide inside a
        # larger enclosing leaf. A zero update must leave the image unchanged.
        points = [light(), light(z=1.31), surfel(), surfel(x=0.001)]
        with renderer(points) as instance:
            before = rgb(instance)
            params_before = instance.get_point_parameters()
            instance.render_backward({"camera": np.zeros_like(before)})
            instance.apply_device_training_step({})
            after_refit = rgb(instance)
            params_after = instance.get_point_parameters()
            for key in params_before:
                np.testing.assert_array_equal(params_before[key], params_after[key])
            np.testing.assert_allclose(before, after_refit, rtol=0, atol=2e-7)
            instance.rebuild_bvh()
            np.testing.assert_allclose(before, rgb(instance), rtol=0, atol=2e-7)
            # Explicitly include contributions below the former 1% cutoff.
            self.assertGreater(float(before[32, 47, 0]), 0.0)

    def test_many_lights_do_not_truncate_opacity_gradients(self):
        # Splitting the same fixed radiant power across 16 coincident lights
        # preserves the image. dI/deta = I/eta for this isolated sheet.
        reference = None
        for shared in (False, True):
            for shadow_capacity in (1, 16):
                with self.subTest(shared=shared, shadow_capacity=shadow_capacity):
                    points = [light(power=35/16)] * 16 + [surfel()]
                    with renderer(points, share_local_layer_direct_lighting=shared,
                                  adjoint_shadow_path_rays=shadow_capacity) as instance:
                        image = rgb(instance)
                        seed = np.full_like(image, 1 / image.size)
                        gradients = instance.render_backward({"camera": seed})[0]
                        expected = float(np.sum(image.astype(np.float64) * seed) / 0.5)
                        actual = float(np.asarray(gradients["opacity"])[16])
                        self.assertGreater(expected, 1e-4)
                        self.assertAlmostEqual(actual, expected, delta=expected * 2e-4)
                        if reference is None:
                            reference = image.copy()
                        np.testing.assert_allclose(image, reference, rtol=2e-6, atol=2e-7)
                        if not shared:
                            # The individual-light XY events carry the shading
                            # position derivative. Shared anchors intentionally
                            # detach this derivative and are excluded here.
                            center_seed = np.zeros_like(image)
                            center_seed[32, 32, 0] = 1
                            center_gradients = instance.render_backward({"camera": center_seed})[0]
                            expected_depth = 2 * float(image[32, 32, 0]) / 1.3
                            actual_depth = float(np.asarray(center_gradients["position"])[16, 2])
                            self.assertAlmostEqual(actual_depth, expected_depth,
                                                   delta=expected_depth * 2e-4)

    def test_offset_shadow_receiver_position_and_rotation(self):
        # Only the center pixel is seeded; the black occluder lies on its light
        # ray, away from its camera ray. qReflect=1 therefore samples the entire
        # seeded camera objective without relying on rare visibility events.
        points = [light(0.65, 0.35),
                  surfel(opacity=0.78, albedo=(0.7, 0.46, 0.22)),
                  surfel(0.325, 0.175, 0.65, 0.145, 0.115,
                         albedo=(0, 0, 0), opacity=0.52, beta=-0.7)]
        for offset in (0.02, 0.15):
            with renderer(points, local_layer_depth_epsilon=offset) as instance:
                image = rgb(instance)
                seed = np.zeros_like(image)
                seed[32, 32, 0] = 1
                gradients = instance.render_backward({"camera": seed})[0]
                for parameter, axis, epsilon in (("position", 2, 0.0003),
                                                  ("rotation", 0, 0.02),
                                                  ("rotation", 1, 0.02)):
                    with self.subTest(offset=offset, parameter=parameter, axis=axis):
                        analytic = float(np.asarray(gradients[parameter])[1, axis])
                        if parameter == "position":
                            setter = lambda value: instance.set_point_translation(value, axis, 1)
                        else:
                            setter = lambda value: instance.set_point_rotation_degrees(value, axis, 1)
                            analytic *= math.pi / 180  # Setter uses degrees; gradient uses radians.
                        setter(epsilon)
                        plus = float(rgb(instance)[32, 32, 0])
                        setter(-epsilon)
                        minus = float(rgb(instance)[32, 32, 0])
                        setter(0.0)
                        finite_difference = (plus - minus) / (2 * epsilon)
                        self.assertGreater(abs(finite_difference), 1e-5)
                        self.assertAlmostEqual(analytic, finite_difference,
                                               delta=abs(finite_difference) * 0.003 + 2e-6)


class OptimizerMigrationTests(unittest.TestCase):
    def test_masked_steps_and_moments_survive_pruning_and_cloning(self):
        import torch
        import optimizers
        from config import OptimizationConfig
        from training_helpers import rebuild_optimizer_preserving_state

        config = OptimizationConfig(optimizer_type="masked_adam")
        shapes = {"position": (3,), "rotation": (3,), "scale": (2,),
                  "albedo": (3,), "opacity": (), "beta": (), "power": ()}
        for name in shapes:
            if name != "power":
                setattr(config, "learning_rate_" + name, 0.001)
        old = {name: torch.nn.Parameter(torch.ones((4,) + tail))
               for name, tail in shapes.items()}
        optimizer = optimizers.create_masked_optimizer(config, **dict(zip(
            ("positions", "rotation", "scales", "albedos", "opacities", "betas", "powers"),
            old.values())))
        for row in range(4):
            for parameter in old.values():
                parameter.grad = torch.full_like(parameter, 0.25 + row)
                parameter.surfelMask = torch.arange(4) >= row
                parameter.updateMask = parameter.surfelMask.reshape((4,) + (1,) * (parameter.ndim - 1))
            optimizer.step()
        for copy_source in (False, True):
            with self.subTest(copy_source=copy_source):
                new = {name: torch.nn.Parameter(value.detach()[[0, 2, 3]].clone())
                       for name, value in old.items()}
                migrated = rebuild_optimizer_preserving_state(
                    config, optimizer, old, new, np.array([True, False, True, False]),
                    np.array([3]), copy_source_state_to_new=copy_source)
                for name, parameter in new.items():
                    state = migrated.state[parameter]
                    self.assertEqual(state["surfel_step"].shape, (3,))
                    self.assertEqual(state["surfel_step"].dtype, torch.int64)
                    expected_steps = [1, 3, 4 if copy_source else 0]
                    self.assertEqual(state["surfel_step"].tolist(), expected_steps)
                    for key in ("exp_avg", "exp_avg_sq"):
                        expected = optimizer.state[old[name]][key][[0, 2, 3]].clone()
                        if not copy_source:
                            expected[2] = 0
                        torch.testing.assert_close(state[key], expected)
                    parameter.grad = torch.ones_like(parameter)
                    parameter.surfelMask = torch.ones(3, dtype=torch.bool)
                    parameter.updateMask = torch.ones_like(parameter, dtype=torch.bool)
                migrated.step()
                for parameter in new.values():
                    self.assertTrue(torch.isfinite(parameter).all())
                    self.assertEqual(migrated.state[parameter]["surfel_step"].tolist(),
                                     [2, 4, 5 if copy_source else 1])


if __name__ == "__main__":
    unittest.main()
