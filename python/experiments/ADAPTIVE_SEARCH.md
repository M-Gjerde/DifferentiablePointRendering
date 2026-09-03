# Adaptive Chamfer search

`run_adaptive_search.py` runs one `main.py` process at a time and persists the
study in SQLite. Re-running the same command resumes the study. A file lock
prevents two controllers from using the same study concurrently.

## Teapot learning-rate campaign

Validate the spec and inspect the generated command without launching training:

```bash
python experiments/run_adaptive_search.py --validate-only
python experiments/run_adaptive_search.py --dry-run
```

Start or resume the campaign:

```bash
python experiments/run_adaptive_search.py
```

The default spec is `experiments/teapot_adaptive_search.json`. Phase one searches
only the uniform `--lr` multiplier from 0.1 to 5.0. The relative component
learning rates are defined once in `config.py`; `--lr 1` uses them unchanged.
The fixed training settings in `base_args` mirror the current
`OptimizationConfig` defaults, including its regularizers, densification, and
pruning. It runs at most 18 trials, evaluates Chamfer distance at
iterations 2000, 4000, 7000, and 10000, and uses the fixed-iteration 10000 score
as the objective. The initial multipliers are 1.0, 0.5, and 2.0; later trials use
a TPE sampler and median pruning. Only one trial runs at a time. Both the metrics
window and image preview remain enabled for the active trial.

Study state is stored below:

```text
OptimizationOutput/studies/teapot_10_lr_scale_v1/_study/
├── study.sqlite3
├── study_summary.csv
├── failed_settings.csv
├── study_spec.snapshot.json
├── logs/
└── trials/
```

Use `--max-trials N` to extend an existing study without editing its frozen
spec. Changing the spec requires a new `study_name` and `output_root`, which
prevents unlike search spaces from being mixed accidentally.

## Live inspection

`--metrics` and `--image-preview` are enabled in the default spec, so the active
training trial opens the same live metric and image windows as a direct
`main.py` run.

Optuna's study-level dashboard reads the SQLite database while the controller is
running:

```bash
optuna-dashboard \
  sqlite:////home/magnus/CLionProjects/DifferentiablePointRendering/python/OptimizationOutput/studies/teapot_10_lr_scale_v1/_study/study.sqlite3
```

Open `http://127.0.0.1:8080` in a browser. The dashboard package is listed in
both dependency manifests. A lightweight CSV view is always available at
`OptimizationOutput/studies/teapot_10_lr_scale_v1/_study/study_summary.csv` after the
first completed trial.

## Point-count feasibility

The teapot point-count limit is 15,000, read from `metrics.csv:num_points`.
It is not the extracted mesh vertex count.

The controller never removes primitives to force a trial under the limit. A
trial that exceeds 15,000 is stopped and recorded as `FAILED_POINT_CAP`. The
result receives an infeasible constraint and a large Chamfer penalty so the
sampler learns to avoid that multiplier region. Densification parameters are
fixed during phase one, so no topology repair is queued. In a later phase that
searches densification, available repairs are:

- a higher densification-gradient threshold;
- a longer densification interval;
- a smaller maximum-new-point fraction; and
- a higher curvature trigger threshold when curvature densification is active.

The other sampled values remain unchanged, making the repair interpretable.

Point growth must also stabilize. Above 90% of the cap, a trial is stopped if
the last 1000 iterations grew by both more than 5% and more than 250 points. A
trial reaching iteration 10000 must satisfy the same final-window check even at
a lower point count. The same window may contain at most 250 gross additions,
so rapid add/prune churn cannot masquerade as a stable net point count. Such
settings are recorded as `FAILED_POINT_UNSTABLE`. During phase one, the fixed
topology settings remain unchanged and the sampler chooses a new learning-rate
multiplier.

## Incremental manual evaluation

`metrics/evaluate_runs.py --full` reuses compatible cached checkpoint rows and
computes only new or changed meshes. A particular checkpoint can be refreshed
with:

```bash
python metrics/evaluate_runs.py \
  --run-dir OptimizationOutput/teapot_10 \
  --gt ~/phd/models/teapot.ply \
  --full --checkpoint-iteration 10000 --force
```

The adaptive controller calls the same symmetric point-to-triangle evaluator
directly. `metrics/evaluate_2dgs_point_to_triangle.py` remains specific to the
external 2DGS `train/ours_*` directory layout and is not a campaign entry point.
