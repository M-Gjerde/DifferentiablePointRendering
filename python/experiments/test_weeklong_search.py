from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import optuna

from config import OptimizationConfig, parse_args
from experiments import run_adaptive_search as search
from experiments.search_common import build_train_command


SPEC_PATH = Path(__file__).with_name("horse_weeklong_search.json")


class WeeklongSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = json.loads(SPEC_PATH.read_text())

    def parsed_parameters(self, sampled):
        parameters = search.apply_linked_parameters(dict(self.spec["base_args"], **sampled), self.spec)
        command = build_train_command(None, Path("/tmp/unused-search-test-output"), parameters)
        with mock.patch.object(sys, "argv", command[1:]):
            return parse_args()

    def test_spec_and_current_snapshot_are_valid(self):
        search.validate_spec(self.spec, check_paths=False)
        search.verify_config_snapshot(self.spec)
        self.assertEqual(23, len(self.spec["search_space"]))

    def test_baseline_reproduces_actual_learning_rate_functions(self):
        from optimizers import create_learning_rate_schedules

        baseline = OptimizationConfig()
        prepared = self.parsed_parameters(self.spec["initial_trials"][0])
        a, b = map(create_learning_rate_schedules, (baseline, prepared))
        for iteration in (0, 1, 7500, 15000, 22500, 30000):
            for group in a["base_learning_rates"]:
                def rate(schedule):
                    local = schedule["parameter_lr_scale_funcs"].get(group, lambda _: 1.0)
                    return (schedule["base_learning_rates"][group]
                            * schedule["global_lr_scale_func"](iteration) * local(iteration))
                self.assertAlmostEqual(rate(a), rate(b), places=15)
        for key in self.spec["search_space"]:
            if key != "global_lr_scale_final":
                self.assertEqual(getattr(baseline, key), getattr(prepared, key))

    def test_every_search_boundary_reaches_the_real_cli(self):
        for key, dimension in self.spec["search_space"].items():
            values = dimension.get("choices", [dimension.get("low"), dimension.get("high")])
            for value in values:
                with self.subTest(parameter=key, value=value):
                    sampled = dict(self.spec["initial_trials"][0], **{key: value})
                    cfg = self.parsed_parameters(sampled)
                    expected = value
                    if key.startswith("learning_rate_"):
                        expected *= sampled["learning_rate"]
                    self.assertEqual(expected, getattr(cfg, key))
                    self.assertEqual(cfg.densification_grad_abs_min, cfg.densification_grad_abs_min_final)
                    self.assertEqual(cfg.densification_interval, cfg.rebuild_bvh_interval)

    def test_global_multiplier_independently_controls_albedo(self):
        base = self.spec["initial_trials"][0]
        for scale in (0.4, 1, 2.5):
            cfg = self.parsed_parameters(dict(base, learning_rate=scale))
            self.assertAlmostEqual(0.0005 * scale, cfg.learning_rate_albedo)

    def test_unattended_options_keep_training_budget_and_geometry(self):
        cfg = self.parsed_parameters(self.spec["initial_trials"][0])
        self.assertEqual(30000, cfg.iterations)
        self.assertFalse(cfg.enable_metrics)
        self.assertFalse(cfg.enable_image_preview)
        self.assertTrue(cfg.densification_relative_error)
        for field in ("densification_radiance_floor", "densify_after"):
            self.assertNotIn(field, self.spec["search_space"])
            self.assertEqual(getattr(OptimizationConfig(), field), getattr(cfg, field))
        self.assertIsNone(cfg.checkpoint)
        self.assertTrue(search.point_count_is_stable({}, {"enabled": False}))
        self.assertFalse(search.point_count_is_stable({}, {}))

    def test_watchdogs_distinguish_runtime_and_no_progress(self):
        spec = {"trial_timeout_minutes": 60, "trial_no_progress_minutes": 15}
        self.assertIsNone(search.trial_timeout_reason(spec, 1800, 30))
        self.assertIn("total trial", search.trial_timeout_reason(spec, 3600, 1))
        self.assertIn("without iteration", search.trial_timeout_reason(spec, 1000, 900))
        self.assertIsNone(search.trial_timeout_reason({}, 1e9, 1e9))

    def test_restart_does_not_reset_wall_clock_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(search.time, "time", return_value=1000):
                self.assertEqual(3600, search.remaining_search_seconds(Path(tmp), 1))
            with mock.patch.object(search.time, "time", return_value=2000):
                self.assertEqual(2600, search.remaining_search_seconds(Path(tmp), 1))
            with mock.patch.object(search.time, "time", return_value=5000):
                self.assertEqual(0, search.remaining_search_seconds(Path(tmp), 1))

    def test_confirmation_repeats_are_not_deduplicated_on_resume(self):
        study = optuna.create_study()
        spec = {"initial_trials": [{"x": 1}, {"x": 2}, {"x": 1}, {"x": 2}],
                "allow_repeated_initial_trials": True}
        search.enqueue_initial_trials(study, spec)
        search.enqueue_initial_trials(study, spec)
        self.assertEqual(4, len(study.trials))
        study.optimize(lambda t: t.suggest_int("x", 1, 2), n_trials=4)
        self.assertEqual([1, 2, 1, 2], [t.params["x"] for t in study.trials])

    def test_stalled_child_is_stopped_and_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = copy.deepcopy(self.spec)
            spec.pop("config_sha256")
            spec.update(min_free_disk_gib=0.000001, trial_no_progress_minutes=0.002)
            study = optuna.create_study()
            objective = search.make_objective(optuna, study, spec, root, root, root / "gt.ply", 0.05)
            processes = []
            popen = search.subprocess.Popen

            def capture(*args, **kwargs):
                process = popen(*args, **kwargs)
                processes.append(process)
                return process

            with mock.patch.object(search, "build_train_command", return_value=[
                sys.executable, "-c", "import time; time.sleep(60)",
            ]), mock.patch.object(search.subprocess, "Popen", side_effect=capture):
                study.optimize(objective, n_trials=1, catch=(search.TrialRunError,))
            self.assertEqual(optuna.trial.TrialState.FAIL, study.trials[0].state)
            self.assertEqual("FAILED_TIMEOUT", study.trials[0].user_attrs["outcome"])
            self.assertTrue(all(p.poll() is not None for p in processes))
            state = json.loads((root / "_study/trials/trial_0000.json").read_text())
            self.assertEqual("FAILED_TIMEOUT", state["status"])

    def test_real_sqlite_pipeline_resumes_and_confirms_with_fake_training(self):
        # Real child processes, polling, Optuna suggestions, SQLite and confirmation.
        # Only rendering and mesh-distance evaluation are replaced.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "train.py"
            script.write_text(
                "import csv,sys\nfrom pathlib import Path\n"
                "p=Path(sys.argv[1]);p.mkdir(parents=True)\n"
                "m=p/'mesh_checkpoints'/'iter_00002';m.mkdir(parents=True)\n"
                "(m/'fuse_post.ply').write_text('fake mesh')\n"
                "with (p/'metrics.csv').open('w') as f:\n"
                " w=csv.writer(f);w.writerow(['iteration','num_points','loss_rgb_mean']);"
                "w.writerow([2,100,0.1])\n"
            )
            spec = copy.deepcopy(self.spec)
            spec.pop("config_sha256")
            spec.update(output_root=str(root / "study"), max_trials=2,
                        min_free_disk_gib=0.000001, search_space={"learning_rate": {
                            "type": "categorical", "choices": [0.5, 1.0]}},
                        initial_trials=[{"learning_rate": 1.0}, {"learning_rate": 0.5}],
                        evaluation_iterations=[2], linked_parameters={},
                        confirmation={"top_k": 1, "repeats": 3, "timeout_hours": 1})
            spec["base_args"] = {"iterations": 2, "mesh_extraction_interval": 2}
            search.validate_spec(spec, check_paths=False)
            args = argparse.Namespace(max_trials=None, timeout_hours=None, poll_seconds=0.1)

            def command(dataset, output, parameters):
                return [sys.executable, str(script), str(output)]

            def geometry(**kwargs):
                # Score depends on the resolved parameters in the recorded invocation.
                state_dir = kwargs["run_dir"].parent / "_study" / "trials"
                index = int(kwargs["run_dir"].name.split("_")[1])
                state = json.loads((state_dir / f"trial_{index:04d}.json").read_text())
                return [{"iteration": 2, "cd": state["parameters"]["learning_rate"]}]

            with mock.patch.object(search, "build_train_command", side_effect=command), \
                    mock.patch.object(search, "compute_geometry_rows", side_effect=geometry):
                search.run_study(root / "spec.json", spec, args)
                search.run_study(root / "spec.json", spec, args)
            summary_path = root / "study/confirmation/_study/confirmation_summary.json"
            rows = json.loads(summary_path.read_text())["candidates"]
            self.assertEqual([0.5, 1.0], [r["median_cd"] for r in rows])
            self.assertTrue(all(r["eligible"] and r["complete_repetitions"] == 3 for r in rows))
            self.assertEqual(6, len(list((root / "study/confirmation/_study/trials").glob("*.json"))))


if __name__ == "__main__":
    unittest.main()
