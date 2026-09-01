# Pale finite-difference tests

The primary suite is `tests_direct.json`. It tests the direct point-light
adjoint in progressively more complicated scenes:

1. one diffuse surfel and one point light;
2. two surfels in one selected depth slab;
3. two separated visible slabs;
4. a point-light shadow occluder;
5. a slab and shadow occluder together;
6. the minimum projected-footprint branch.

Every case uses deterministic center rays and a central difference at two
epsilon values. Most cases use a fixed, spatially varying signed image
cotangent, so they test the full vector-Jacobian product rather than only an
L2 residual. One case explicitly tests the L2+SSIM image seed.

The checker fails on:

- NaN or infinite values;
- a missing analytic gradient when finite differences are nonzero;
- a missing finite-difference gradient when the analytic result is nonzero;
- a test with no gradient signal, unless it explicitly declares that zero is
  the expected result;
- any epsilon outside the combined absolute/relative tolerance;
- disagreement between the finite-difference epsilon levels;
- non-repeatable forward renders.

There is no allowed failing-row fraction and no silent tiny-gradient skip.

## Running

From `python/`, with `pale` importable:

```bash
python finite_difference/batch_fd_check.py
```

With a freshly built module in a separate build directory:

```bash
python finite_difference/batch_fd_check.py \
  --pale-module-dir /tmp/dpr-codex-build
```

List or select cases:

```bash
python finite_difference/batch_fd_check.py --list
python finite_difference/batch_fd_check.py --stage single_surfel
python finite_difference/batch_fd_check.py --case '3*_occluder_*'
python finite_difference/batch_fd_check.py --case-index 0
```

Each case runs in a separate process and writes `samples.csv`, `result.json`,
and an `images/` directory. The image directory contains:

- a baseline and repeat render for every tested parameter value;
- the `+epsilon` and `-epsilon` render for every finite-difference stencil;
- an amplified absolute-difference image for every stencil.

The PNG files are sRGB-encoded previews. Matching NPY files retain the exact
linear floating-point RGB values used by the objective. `result.json` records
all artifact paths and the display multiplier used for each amplified
difference preview. The batch runner writes a revision-stamped `summary.json`.

## Important settings covered

- `share_local_layer_direct_lighting=false`: individual surfel-to-light
  connections and shadow traversals;
- `share_local_layer_direct_lighting=true`: one detached consensus light
  connection per slab;
- `minimum_projected_footprint=false/true`;
- batched point-hit traversal and the scalar traversal (`point_hit_batch_size=1`);
- lookahead enabled/disabled while gathering a visible slab;
- deterministic first-slab selection and stochastic traversal of separated
  slabs with validated null/reflect probabilities.

The older sweep JSON files are retained as historical experiments, but they
use the superseded schema and are not consumed by the new batch runner.
