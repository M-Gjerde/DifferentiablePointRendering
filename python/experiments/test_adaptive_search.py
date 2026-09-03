from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import OptimizationConfig, parse_args
from image_preview import (
    discover_camera_names as discover_preview_camera_names,
    get_latest_render_path,
    get_target_path,
)
from metrics.evaluate_runs import (
    MeshCheckpoint,
    cached_geometry_iterations,
    evaluate_run,
    filter_mesh_checkpoints_by_iteration,
    find_target_image_for_camera,
    find_mesh_checkpoints,
    lazy_chamfer_imports,
    merge_geometry_rows,
    read_existing_geometry_rows,
    write_dict_csv,
)
from experiments.run_adaptive_search import (
    apply_linked_parameters,
    first_point_cap_excess,
    import_baseline_run,
    metrics_diagnostics,
    point_count_is_stable,
    point_stability_violation,
    repair_densification_parameters,
    validate_spec,
)
from experiments.run_hyperparameter_search import build_train_command


class ConfigurationTests(unittest.TestCase):
    def test_bsdf_densification_values_are_serialized(self) -> None:
        serialized = dataclasses.asdict(OptimizationConfig())
        self.assertIn("densify_bsdf_floor", serialized)
        self.assertIn("densify_bsdf_gamma", serialized)

    def test_linked_parameters_follow_the_sampled_value(self) -> None:
        parameters = apply_linked_parameters(
            {"densification_interval": 200, "densification_grad_abs_min": 0.001},
            {
                "linked_parameters": {
                    "rebuild_bvh_interval": "densification_interval",
                    "densification_grad_abs_min_final": "densification_grad_abs_min",
                }
            },
        )
        self.assertEqual(200, parameters["rebuild_bvh_interval"])
        self.assertEqual(0.001, parameters["densification_grad_abs_min_final"])

    def test_unattended_command_disables_preview(self) -> None:
        command = build_train_command(
            Path("/dataset"),
            Path("/output"),
            {"enable_image_preview": False, "iterations": 10},
        )
        self.assertIn("--no-image-preview", command)

    def test_adaptive_command_enables_live_monitors(self) -> None:
        command = build_train_command(
            Path("/dataset"),
            Path("/output"),
            {
                "enable_metrics": True,
                "enable_image_preview": True,
                "iterations": 10,
            },
        )
        self.assertIn("--metrics", command)
        self.assertIn("--image-preview", command)

    def test_config_parser_accepts_disabled_preview(self) -> None:
        with mock.patch("sys.argv", ["main.py", "--no-image-preview"]):
            config = parse_args()
        self.assertFalse(config.enable_image_preview)

    def test_lr_is_a_uniform_component_multiplier(self) -> None:
        base_config = OptimizationConfig()
        with mock.patch("sys.argv", ["main.py", "--lr", "2.0"]):
            config = parse_args()
        for field_name in (
            "learning_rate_position",
            "learning_rate_rotation",
            "learning_rate_scale",
            "learning_rate_albedo",
            "learning_rate_opacity",
            "learning_rate_beta",
        ):
            self.assertEqual(
                2.0 * getattr(base_config, field_name),
                getattr(config, field_name),
            )


class PointGuardrailTests(unittest.TestCase):
    def test_cap_excess_is_a_failed_setting_signal(self) -> None:
        rows = [
            {"iteration": "100", "num_points": "14999"},
            {"iteration": "101", "num_points": "15001"},
        ]
        self.assertEqual((101, 15001), first_point_cap_excess(rows, 15000))
        self.assertIsNone(first_point_cap_excess(rows[:1], 15000))

    def test_final_window_can_stabilize_by_absolute_growth(self) -> None:
        rows = [
            {"iteration": "9000", "num_points": "14000", "loss_rgb_mean": "0.01"},
            {"iteration": "10000", "num_points": "14150", "loss_rgb_mean": "0.009"},
        ]
        diagnostics = metrics_diagnostics(rows, 1000)
        self.assertTrue(
            point_count_is_stable(
                diagnostics,
                {"max_relative_growth": 0.005, "max_absolute_growth": 250},
            )
        )

    def test_point_cap_repair_reduces_densification_pressure(self) -> None:
        search_space = {
            "densification_grad_abs_min": {
                "type": "float",
                "low": 0.00025,
                "high": 0.0015,
            },
            "densification_interval": {
                "type": "categorical",
                "choices": [50, 100, 200],
            },
            "densification_max_new_fraction": {
                "type": "categorical",
                "choices": [0.25, 0.5, 1.0],
            },
            "curvature_violation_threshold": {
                "type": "categorical",
                "choices": [0.0, 20.0, 35.0, 60.0],
            },
        }
        repaired = repair_densification_parameters(
            {
                "densification_grad_abs_min": 0.0005,
                "densification_interval": 50,
                "densification_max_new_fraction": 1.0,
                "curvature_violation_threshold": 20.0,
            },
            search_space,
        )
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertGreater(repaired["densification_grad_abs_min"], 0.0005)
        self.assertGreater(repaired["densification_interval"], 50)
        self.assertLess(repaired["densification_max_new_fraction"], 1.0)
        self.assertGreater(repaired["curvature_violation_threshold"], 20.0)

    def test_gross_add_prune_churn_is_not_stable(self) -> None:
        diagnostics = {
            "point_growth": 0,
            "point_growth_fraction": 0.0,
            "densification_new_points": 400,
        }
        stability = {
            "max_relative_growth": 0.05,
            "max_absolute_growth": 250,
            "max_new_points": 250,
        }
        self.assertFalse(point_count_is_stable(diagnostics, stability))
        self.assertGreater(point_stability_violation(diagnostics, stability), 0.0)


class RenderLayoutTests(unittest.TestCase):
    def test_readers_use_nested_renders_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            camera_name = "DatasetCam_000"
            camera_render_dir = run_dir / "renders" / camera_name / "render"
            camera_render_dir.mkdir(parents=True)
            render_path = camera_render_dir / "0050_render.png"
            target_path = run_dir / "renders" / f"render_target_{camera_name}.png"
            render_path.write_bytes(b"render")
            target_path.write_bytes(b"target")

            self.assertEqual([camera_name], discover_preview_camera_names(run_dir))
            self.assertEqual(render_path, get_latest_render_path(run_dir, camera_name))
            self.assertEqual(target_path, get_target_path(run_dir, camera_name))
            self.assertEqual(
                target_path,
                find_target_image_for_camera(run_dir, None, camera_name),
            )

    def test_preview_reader_supports_legacy_root_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            camera_name = "DatasetCam_000"
            camera_render_dir = run_dir / camera_name / "render"
            camera_render_dir.mkdir(parents=True)
            render_path = camera_render_dir / "0050_render.png"
            render_path.write_bytes(b"render")

            self.assertEqual([camera_name], discover_preview_camera_names(run_dir))
            self.assertEqual(render_path, get_latest_render_path(run_dir, camera_name))


class GeometryCheckpointTests(unittest.TestCase):
    def test_chamfer_backend_imports_from_metrics_package(self) -> None:
        functions = lazy_chamfer_imports()
        self.assertEqual(3, len(functions))
        self.assertTrue(all(callable(function) for function in functions))

    def test_discovery_is_sorted_and_iteration_filter_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in ["iter_02000", "iter_01000", "iter_1000_extra"]:
                checkpoint_dir = run_dir / "mesh_checkpoints" / name
                checkpoint_dir.mkdir(parents=True)
                (checkpoint_dir / "fuse_post.ply").write_text("mesh", encoding="utf-8")
            checkpoints = find_mesh_checkpoints(run_dir, "fuse_post.ply")
            self.assertEqual([1000, 2000], [item.iteration for item in checkpoints])
            selected = filter_mesh_checkpoints_by_iteration(checkpoints, [2000])
            self.assertEqual([2000], [item.iteration for item in selected])

    def test_duplicate_numeric_iteration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in ["iter_01000", "iter_1000"]:
                checkpoint_dir = run_dir / "mesh_checkpoints" / name
                checkpoint_dir.mkdir(parents=True)
                (checkpoint_dir / "fuse_post.ply").write_text("mesh", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Duplicate mesh checkpoint"):
                find_mesh_checkpoints(run_dir, "fuse_post.ply")

    def test_cache_fingerprint_and_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh_path = Path(directory) / "fuse_post.ply"
            mesh_path.write_text("mesh", encoding="utf-8")
            mesh_stat = mesh_path.stat()
            checkpoint = MeshCheckpoint(1000, mesh_path)
            cached = [
                {
                    "iteration": 1000,
                    "cd": 0.2,
                    "reconstruction_size": mesh_stat.st_size,
                    "reconstruction_mtime_ns": mesh_stat.st_mtime_ns,
                }
            ]
            self.assertEqual({1000}, cached_geometry_iterations(cached, [checkpoint]))
            merged = merge_geometry_rows(cached, [{"iteration": 1000, "cd": 0.1}])
            self.assertEqual(1, len(merged))
            self.assertEqual(0.1, merged[0]["cd"])

            cache_path = Path(directory) / "cache.csv"
            write_dict_csv(cache_path, cached)
            round_tripped = read_existing_geometry_rows(cache_path)
            self.assertEqual(
                {1000},
                cached_geometry_iterations(round_tripped, [checkpoint]),
            )

    def test_full_evaluation_computes_only_new_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "metrics.csv").write_text(
                "iteration,loss_total_mean,loss_rgb_mean,num_points\n"
                "3000,0.1,0.1,10\n",
                encoding="utf-8",
            )
            ground_truth = root / "gt.ply"
            ground_truth.write_text("gt", encoding="utf-8")

            def add_checkpoint(iteration: int) -> None:
                checkpoint_dir = run_dir / "mesh_checkpoints" / f"iter_{iteration:05d}"
                checkpoint_dir.mkdir(parents=True)
                (checkpoint_dir / "fuse_post.ply").write_text("mesh", encoding="utf-8")

            add_checkpoint(1000)
            add_checkpoint(2000)

            def fake_compute(**kwargs):
                rows = []
                for checkpoint in kwargs["checkpoints"]:
                    mesh_stat = checkpoint.mesh_path.stat()
                    rows.append(
                        {
                            "run_name": "run",
                            "iteration": checkpoint.iteration,
                            "cd": 1.0 / checkpoint.iteration,
                            "accuracy": 0.1,
                            "completion": 0.1,
                            "reconstruction": str(checkpoint.mesh_path),
                            "reconstruction_size": mesh_stat.st_size,
                            "reconstruction_mtime_ns": mesh_stat.st_mtime_ns,
                        }
                    )
                return rows

            args = argparse.Namespace(
                ground_truth=ground_truth,
                full=True,
                checkpoint_iteration=[],
                force=False,
                reconstruction_name="fuse_post.ply",
                samples=500000,
                device="auto",
                seed=0,
                scale=1.0,
                use_vertices=True,
                complete_loss_only=False,
                linear_loss_y=False,
            )
            patch_prefix = "metrics.evaluate_runs"
            with (
                mock.patch(f"{patch_prefix}.compute_geometry_rows", side_effect=fake_compute) as compute,
                mock.patch(f"{patch_prefix}.plot_loss_curve"),
                mock.patch(f"{patch_prefix}.plot_geometry_curve"),
                mock.patch(f"{patch_prefix}.plot_loss_geometry_curve"),
                mock.patch(f"{patch_prefix}.compute_final_psnr_rows", return_value=[]),
            ):
                evaluate_run(run_dir, args)
                self.assertEqual(2, len(compute.call_args.kwargs["checkpoints"]))
                add_checkpoint(3000)
                evaluate_run(run_dir, args)
                self.assertEqual(
                    [3000],
                    [item.iteration for item in compute.call_args.kwargs["checkpoints"]],
                )


class SpecValidationTests(unittest.TestCase):
    def test_teapot_phase_one_searches_only_uniform_lr_multiplier(self) -> None:
        spec_path = Path(__file__).with_name("teapot_adaptive_search.json")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.assertEqual({"learning_rate"}, set(spec["search_space"]))
        self.assertEqual(0.1, spec["search_space"]["learning_rate"]["low"])
        self.assertEqual(5.0, spec["search_space"]["learning_rate"]["high"])
        self.assertTrue(spec["base_args"]["enable_metrics"])
        self.assertTrue(spec["base_args"]["enable_image_preview"])
        defaults = OptimizationConfig()
        for name, value in spec["base_args"].items():
            self.assertEqual(
                getattr(defaults, name),
                value,
                msg=f"base_args.{name} is stale relative to OptimizationConfig",
            )

    def test_uniform_learning_rate_multiplier_is_a_valid_dimension(self) -> None:
        spec = {
            "dataset_path": "/not/checked",
            "ground_truth": "/not/checked.ply",
            "output_root": "out",
            "base_args": {"iterations": 1000, "mesh_extraction_interval": 1000},
            "search_space": {
                "learning_rate": {"type": "float", "low": 0.5, "high": 2.0}
            },
            "evaluation_iterations": [1000],
            "guardrails": {
                "max_points": 15000,
                "point_stability": {
                    "window_iterations": 1000,
                    "max_relative_growth": 0.05,
                    "max_absolute_growth": 250,
                },
            },
        }
        validate_spec(spec, check_paths=False)

    def test_compatible_baseline_is_imported_without_training(self) -> None:
        import optuna

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            mesh_dir = baseline / "mesh_checkpoints" / "iter_00001"
            mesh_dir.mkdir(parents=True)
            mesh_path = mesh_dir / "fuse_post.ply"
            mesh_path.write_text("mesh", encoding="utf-8")
            (baseline / "metrics.csv").write_text(
                "iteration,num_points,loss_rgb_mean\n1,100,0.1\n",
                encoding="utf-8",
            )
            (baseline / "points_final.ply").write_text("points", encoding="utf-8")
            (baseline / "run_config.json").write_text(
                json.dumps(
                    {
                        "optimization_config": {
                            "iterations": 10,
                            "mesh_extraction_interval": 1,
                            "densification_interval": 100,
                            "rebuild_bvh_interval": 100,
                        }
                    }
                ),
                encoding="utf-8",
            )
            spec = {
                "baseline_run": str(baseline),
                "base_args": {"iterations": 1, "mesh_extraction_interval": 1},
                "search_space": {
                    "densification_interval": {
                        "type": "categorical",
                        "choices": [100, 200],
                    }
                },
                "linked_parameters": {
                    "rebuild_bvh_interval": "densification_interval"
                },
                "evaluation_iterations": [1],
                "reconstruction_name": "fuse_post.ply",
                "samples": 10,
                "evaluation_seed": 0,
                "evaluation_scale": 1.0,
                "use_vertices": True,
                "failure_cd_penalty": 1.0,
                "guardrails": {
                    "max_points": 15000,
                    "point_stability": {
                        "window_iterations": 1,
                        "max_relative_growth": 0.05,
                        "max_absolute_growth": 250,
                    },
                },
            }
            study = optuna.create_study(direction="minimize")
            geometry_row = {
                "iteration": 1,
                "cd": 0.01,
                "accuracy": 0.01,
                "completion": 0.01,
            }
            with mock.patch(
                "experiments.run_adaptive_search.compute_geometry_rows",
                return_value=[geometry_row],
            ):
                import_baseline_run(
                    optuna=optuna,
                    study=study,
                    spec=spec,
                    output_root=root / "study",
                    ground_truth_path=root / "gt.ply",
                )
            self.assertEqual(1, len(study.trials))
            self.assertEqual("IMPORTED_BASELINE", study.trials[0].user_attrs["outcome"])
            self.assertAlmostEqual(0.01, study.trials[0].value)


if __name__ == "__main__":
    unittest.main()
