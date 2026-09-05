# Five-day search from the current configuration

`teapot_weeklong_search.json` freezes the 2026-09-05 `config.py` defaults and
searches 23 interacting parameters. It targets `~/phd/datasets/teapot_10_pbdr`
and `~/phd/models/teapot.ply`, following the existing search. This is a search
for settings for that dataset, not evidence of generalization to other scenes.

The current configuration runs first. Seven additional initial trials change
the global LR, global decay, densification pressure, tangent restriction, or
the position-statistic source. All trials start from the dataset's initial
point cloud and run up to **30,000 iterations**. No trial inherits another
trial's optimized geometry or Adam state.

## Budget and selection

- One worker on the existing renderer GPU; keep the viewer/other training jobs
  closed during the study. `--device cpu` is the existing **PyTorch host-device**
  setting; the native renderer chooses its SYCL device separately. The latest
  config snapshot has **3 adjoint passes**; this is retained, not searched.
- **108 hours of search**, with a ceiling of 800 terminal trials. At 15–20
  minutes each, this is 324–432 full trials before extraction/evaluation overhead;
  pruning may allow more, and aggressive densification may allow fewer.
- Multivariate TPE, 64 startup observations, 64 candidate proposals. The eight
  queued trials count toward startup. This is an adaptive search, not an
  exhaustive grid or a guarantee of the global optimum.
- Mesh evaluations at 7,500 / 15,000 / 22,500 / 30,000 iterations. The first 64
  completed trials run without statistical pruning. Afterward, pruning is
  allowed at 15,000 and 22,500, with at least 12 completed observations at that
  step. The 75th-percentile rule is deliberately lenient for minimization;
  Optuna compares the current trial's **best intermediate value** to previous
  trials at the same step. It does not remove exactly 25% of all trials.
- Rank by **final** symmetric mean point-to-triangle distance, not best-ever
  checkpoint, training loss, or PSNR. Evaluate 500,000 uniformly sampled surface
  points per mesh with seed 0, scale 1. This changes the old vertex-weighted
  evaluation to reduce sensitivity to mesh tessellation. It does not change
  the training objective. Accuracy and completion are also recorded.
- Automatically freeze the five best feasible, completed configurations and
  the baseline, then run each **three more times**, without statistical pruning.
  This confirmation phase has a 12-hour budget. It usually needs 4.5–6 hours
  at the stated trial time. Rank eligible candidates by median final distance;
  incomplete or failed repetitions are flagged, not silently discarded.
- Repetitions are fresh processes with the existing renderer seed policy,
  **not controlled independent random seeds**. They check run-to-run variation;
  they do not establish robustness to new camera sets or scenes.

The time limit stops submission of new trials; the current trial finishes.
Budgets are measured from each phase's first launch, including downtime, and
persist across restarts. A six-day variant can use `--timeout-hours 132`
(132 hours search plus up to 12 hours confirmation). Do not extend an already
confirmed study: prepare a new output root/study name for further exploration.

## Search space

| Group | Search |
|---|---|
| Global LR multiplier | 0.4–2.5, logarithmic |
| Position base LR | 1.375e-5–2.2e-4, logarithmic |
| Rotation base LR | 0.00125–0.02, logarithmic |
| Scale base LR | 2.5e-5–4e-4, logarithmic |
| Opacity base LR | 5e-5–8e-4, logarithmic |
| Beta base LR | 7.5e-5–0.0012, logarithmic |
| Albedo effective initial LR | 2e-4–0.00125 through the global multiplier |
| Final global schedule factor | 1 / 0.6 / 0.3 / 0.1 |
| Final position schedule factor | 10 / 5 / 3 / 1; initial factor fixed at 10 |
| Decay duration | 7,500 / 15,000 / 22,500 / 30,000 iterations |
| Densification interval | 100 / 200 / 400 |
| Gradient quantile / absolute floor | 0.60–0.95 in steps of 0.05; 1e-6–3e-4 log |
| Radiance quantile bins | 1 / 4 / 8 bins |
| Minimum split scale | 0.003–0.012 log |
| Split displacement / scale divisor | 0.15–0.60; 1.2–2 log |
| Maximum new fraction | 0.1 / 0.25 / 1 |
| Tangent-only displacement / full-position statistics | Each on/off |
| Depth distortion weight | 5–200 log |
| Normal consistency weight | 0 / 0.0005 / 0.002 / 0.008 |
| Slab-anchor weight | 0 / 1e-6 / 1e-5 / 1e-4 |
| Minimum surfel area | Current value times 0.25–4, log |

The effective rate is `global_multiplier * component_rate * global_schedule`,
with a further `position_schedule` factor for position. The current position
rate starts at **5.5e-4** and reaches **2.75e-4** at iteration 15,000.

Albedo's component rate is fixed at 5e-4 to anchor the global multiplier.
Thus the six effective parameter-group rates remain independently adjustable,
without adding a redundant seventh amplitude degree of freedom. Global decay
is always enabled in this spec; endpoints 1 and 1 reproduce the current disabled
global schedule. Position's initial schedule multiplier is likewise fixed to
avoid duplicating the searched position base rate.

The absolute densification threshold stays constant within a trial; its final
value is linked to its initial value. BVH rebuild cadence follows densification.
Relative-error densification remains enabled, with no extra albedo compensation.
Its radiance floor is fixed at **0.001**, and `densify_after` is fixed at **0**,
matching the current config. Depth distortion and normal consistency weights
are both searched over the ranges above.
Curvature densification/regularization, opacity resets, SSIM, shared lighting,
adjoint sample count, camera policy, and mesh extraction settings otherwise keep
their current defaults. Disabled or inactive options are not searched merely
because they exist.

## Unattended operation

The previous 15,000-point cap and final point-growth stability requirement are
not reused. This study has a **200,000-point resource cap** and no point-stability
ranking requirement. A still-densifying run can be a good reconstruction.
The cap is a feasibility constraint, so results are optimal only within it.
Metrics polling detects cap excess; it is not a hard allocation limit inside
the renderer.

A training process is stopped after 60 minutes total or 15 minutes without
iteration progress. These checks run in the monitor loop; synchronous mesh
evaluation can delay them. Five consecutive process failures stop the study
for diagnosis. Falling below 50 GiB free stops the study and its active child.
Per-trial logs and outcome records remain available. Failed trials do not
automatically enqueue chains of densification repairs.

Metrics logging remains at every 100 iterations. RGB snapshots are saved every
10,000, point clouds every 30,000, and live preview/metrics windows are disabled.
`--no-metrics` disables the GUI companion, **not** metrics.csv. This reduces
storage and host-side work without changing renderer sampling or updates.

Optuna's SQLite database, phase budgets, spec snapshot, and confirmation list
are persistent. Re-running the same command resumes completed-trial accounting;
an interrupted training process restarts as a trial, not from its Adam checkpoint.
The runner verifies the saved `config.py` hash before each trial so an edit to
unexposed defaults cannot silently change the objective mid-study. Keep the
source code and dataset fixed during the experiment.

## Launch

From the repository root, in your existing `pale` environment:

```bash
conda activate pale
python -m pip install -r python/experiments/requirements-optuna.txt
cd python
python experiments/run_adaptive_search.py --spec experiments/teapot_weeklong_search.json --validate-only
```

Then start it in a persistent terminal session such as tmux (if installed):

```bash
tmux new -s pale-search
python -u experiments/run_adaptive_search.py --spec experiments/teapot_weeklong_search.json 2>&1 | tee /tmp/pale-weeklong-search.log
```

Detach with Ctrl-B, D; reconnect with `tmux attach -t pale-search`.
Run the same Python command to resume after an interruption. `--dry-run`
prints a representative training command without starting training.
Without tmux, launch from the same `python` directory with:

```bash
nohup python -u experiments/run_adaptive_search.py --spec experiments/teapot_weeklong_search.json > /tmp/pale-weeklong-search.log 2>&1 < /dev/null &
```

Outputs are under
`python/OptimizationOutput/studies/teapot_current_config_5day_20260905_v1/`:

- `_study/study_summary.csv`: all trials and their final objective/status.
- `_study/logs/` and `_study/trials/`: training logs and resolved trial settings.
- `confirmation/_study/confirmation_summary.json`: repeated results and median
  ranking; only entries with `eligible: true` have all requested repetitions.
- Each run's `run_config.json`, `points_final.ply`, and final mesh preserve the
  result for inspection and later reproduction.

The renderer was not launched as part of preparing this search. Before leaving,
let the baseline finish and inspect its point count, runtime, mesh, and geometry
score. Later compare finalists on another camera set or scene before adopting
the winner as a general default.

References: [Optuna TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html),
[percentile pruning](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html).
