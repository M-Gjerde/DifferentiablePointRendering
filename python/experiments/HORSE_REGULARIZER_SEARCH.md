# Horse regularizer strengths

Run or resume the study:

```bash
python experiments/run_adaptive_search.py --spec experiments/horse_regularizer_strengths_search.json
```

Append `--validate-only` to check the spec and input paths, or `--dry-run` to
inspect the training command without starting training.

This study uses the local horse_10 dataset at `~/phd/datasets/horse_pbdr_10`
and ground truth `~/phd/models/horse.ply`. Each trial runs for 10,000 iterations.
It minimizes the final checkpoint's Chamfer distance, evaluating every 1,000
iterations. The budget is 80 sequential trials (`--max-trials N` overrides it).
Performance pruning starts at 10,000 iterations to compare full training runs.

Only these weights vary:

| Regularizer | Candidate strengths |
| --- | --- |
| Depth distortion | 0, 0.001, 0.005, 0.01, 0.02, 0.1 |
| Normal consistency | 0, 0.0005, 0.0025, 0.005, 0.01, 0.05 |
| Intra-slab depth | 0, 1e-5, 5e-5, 1e-4, 2e-4, 1e-3 |
| Curvature scale | 0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4 |

The first trial uses the current defaults: depth distortion 0.01, normal
consistency 0.005, intra-slab depth 1e-4, and curvature scale 0. The next five
trials disable all four regularizers, disable each of the three active
regularizers individually, and enable curvature scale at 1e-6. Active weights
are searched at 0, 0.1x, 0.5x, 1x, 2x, and 10x their baseline strengths.
Curvature retains its zero-inclusive logarithmic candidate grid. TPE then
searches combinations, with 16 startup trials in total.

All remaining training settings inherit the current `config.py`, including
learning rates, world-space depth distortion, densification, and output settings.
The spec records the defaults and config hash; the runner rejects config changes
to keep trials comparable. Prepare a new study if the config changes.

The existing horse study's 200,000-point feasibility cap is retained, with point
stability enforcement and automatic repair trials disabled.
Study state and results are saved under
`OptimizationOutput/studies/horse_10_regularizer_strengths_10k_v2`.
Version 2 starts a fresh study so the updated defaults and candidate strengths
are not mixed with version 1 results.
