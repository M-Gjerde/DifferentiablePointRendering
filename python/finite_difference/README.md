# Pale finite-difference tests

The primary suite is `tests_direct.json`. It tests the direct point-light
adjoint in progressively more complicated scenes:

1. one diffuse surfel and one point light;
2. two surfels in one selected depth slab;
3. two separated visible slabs;
4. a point-light shadow occluder;
5. a slab and shadow occluder together.

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
- batched point-hit traversal and the scalar traversal (`point_hit_batch_size=1`);
- lookahead enabled/disabled while gathering a visible slab;
- deterministic first-slab selection and stochastic traversal of separated
  slabs with validated null/reflect probabilities.

The older sweep JSON files are retained as historical experiments, but they
use the superseded schema and are not consumed by the new batch runner.

## Renderer correctness regressions

After rebuilding `pale`, run from the repository root with the training Python
environment (including PyTorch and the image I/O dependencies):

```bash
PYTHONPATH="$PWD/cmake-build-pybind:$PWD/python" \
  python -m unittest discover -s python/finite_difference -p 'test_*.py' -v
```

`test_renderer_correctness.py` creates temporary scenes and checks calibrated
rectangular cameras against closed-form direct illumination, full beta-profile
support across BVH rebuilds/refits, many-light opacity gradients, offset shadow
ray position/rotation derivatives, stale forward buffers, and masked Adam state
migration after pruning/cloning. Set `ACPP_VISIBILITY_MASK=omp` and a writable
`ACPP_APPDB_DIR` to use AdaptiveCpp's CPU backend without a GPU.

## Optional curvature work and profiling

Curvature work is independent of relative-error densification. The camera
curvature/slab-search pass runs only when `curvature_scale_weight` is nonzero,
`enable_curvature_densification` is enabled, or the native renderer setting
`compute_curvature_diagnostics` is true. Otherwise its image and active-slab
count are reset to zero. The viewer requests it when displaying curvature or
per-primitive diagnostics that use the selected slab's identity. Changing to
one of these views triggers a fresh render.

For performance measurements, collect GPU traversal counters separately from
timings: the counters use global atomic additions. Python's debug logging
(`logging=1`) also reports `Device optimizer: parameter update` and
`Device optimizer: point BVH refit`; these costs are absent from viewer-only
render timings. Their completion waits are enabled only while timing/logging.

## Parallel refit and camera gather

The native device optimizer refits after every parameter update. It now updates
point-BVH leaves and their traversal cache in parallel, followed by parent
levels, packed binary/four-way bounds, and the TLAS. The level schedule is
cached until a full rebuild or topology change. All stages use the renderer's
in-order queue, so each parent sees completed child bounds. Instanced BLAS
ranges are scheduled only once. Full surfel support and the existing tree and
primitive permutation are preserved; no opacity cutoff or additional culling
is introduced.

Pass `parallel_bvh_refit=False` in `pale.Renderer`'s settings dictionary to use
the serial reference. `test_bvh_refit.py` compares both paths after identical
position, rotation, scale, material, opacity, and beta changes, including
pruning and addition. It also compares refit against a fresh rebuild after
nonzero native Adam updates. The comparisons hold geometry fixed to avoid
confounding refit with differences between independent floating-point Adam
trajectories.

RGB gather dispatches compiled variants for shared/individual lighting and
enabled/disabled GPU counters. This removes unused branches and private
counter state from each work item. Rendering equations, ray samples, slab
membership, and traversal capacities are unchanged. Performance comparisons
should use the same frozen checkpoint and camera, warm JIT caches, and avoid
concurrent builds or test runs on the host.

## Relative densification

Training defaults to `densification_relative_error=True`. The auxiliary source
is `(rendered - target) / (3 * pixel_count * B_squared)` in linear RGB, with
`B_squared = stop_gradient((sum(rendered**2) + sum(target**2))/6 + radiance_floor**2)`.
The floor defaults to `0.01` and is configurable with
`--densification-radiance-floor`. It deliberately limits gain invariance near black.
The optimizer retains its original RGB objective and regularizer gradients.

With half-MSE, each pixel's existing gradient contributions are weighted before
surfel/camera accumulation, so no additional adjoint traversal is needed. For
an SSIM mixture, a separate relative-MSE adjoint supplies only densification
statistics. Both modes bypass the mean-albedo boost. The scalar score retains
its tangent projection and camera/iteration averaging; the split direction
still respects `densification_tangent_only`.

Relative mode uses all implemented position derivatives by default, including
camera and shadow attenuation, while retaining the shared anchor's stop-gradient.
`--no-densification-full-position` instead weights the existing local footprint
signal. `--no-densification-relative-error` restores the albedo-compensated
photometric statistic. Existing absolute split thresholds are not converted;
they need calibration in the new statistic's units.

`test_relative_densification.py` compares against an explicitly constructed,
frozen relative source, tests common radiometric gains, checks optimizer
isolation with and without SSIM, and verifies that albedo compensation is replaced.

### World-space depth distortion

`--depth-distort-world-space` (or `OptimizationConfig.depth_distort_world_space=True`)
uses linear camera-forward depth in scene units in the pairwise distortion loss:
`sum_{i<j} w_i*w_j*(z_i-z_j)^2`. Its depth-coordinate derivative is one, so the
same layer separation is no longer attenuated with increasing camera distance.
Compositing/visibility weights and their gradients remain active. This measures
camera-forward separation, not Euclidean ray length for off-axis pixels.

The default is `False`; `--no-depth-distort-world-space` retains the existing NDC
inverse-depth loss. Direct renderer callers pass `depth_distort_world_space` in
the constructor settings dictionary. Rebuild `pale` after changing the C++ code.
The world-space loss has squared scene-distance units: retune
`--depth-distort-weight` when switching rather than assuming the old weight is
comparable. `test_world_space_distortion.py` checks both forward paths, distance
invariance, the default mode, and position/opacity finite differences in both modes.
