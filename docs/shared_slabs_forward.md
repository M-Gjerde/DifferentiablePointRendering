# Two-slab forward reconstruction

This extends the pair experiment with two explicit, camera-independent charts.
Each chart takes a contiguous range of 1–8 GPU surfel indices. Oversized ranges,
out-of-bounds ranges, emissive members, and invalid frames are rejected rather
than truncated. Membership does not depend on ray visibility. Chart origins and
normals are recomputed from the same members after ordinary scene edits.

The test fixture and launch instructions are in
`Assets/Experiments/shared_slabs/README.md`.

## Geometry

Within chart k, let N be its normal, U its first tangent, O its origin, and let
the world-space half extents be R (lateral) and h (depth). Offsets s and d move
the support, without moving the reference plane or changing the local height:

```
u = (dot(U, x-O) - s) / R
z = (dot(N, x-O) - d) / h
band_k(x) = max(1-u², 0)³ max(1-z², 0)³
```

Each member's plane residual is
`r_ki(x) = dot(n_i, x-c_i) / dot(n_i,N_k)`.
Project `x` to that plane along the chart normal, evaluate its elliptical
coordinates, and use the same cubic footprint as the pair experiment:

```
w_ki(x) = opacity_i max(1-radius_ki², 0)³
W_k(x) = sum_i w_ki(x)
F_k(x) = sum_i w_ki(x) r_ki(x) / W_k(x)
B_k(x) = band_k(x) W_k(x)
F(x) = sum_k B_k(x) F_k(x) / sum_k B_k(x)
```

The implementation expands the numerator as `sum band_k w_ki r_ki`, avoiding
division by an empty chart's W_k. A chart only stops contributing after its
weight and first two derivatives vanish. The field is C2 where total support
is positive; a smooth regular surface additionally requires nonzero field
gradient. Consistent chart signs are enforced; nearly perpendicular chart
normals are rejected in this small connected-sheet experiment.

Normals use the full spatial derivative of F, including derivatives of the
slab tapers, footprint weights, and plane residuals. Albedo uses the same
normalized weights. The slab-weight display shows B_B/(B_A+B_B).

Coverage is applied once at the reconstructed ray hit:
`alpha = 1-exp(-coverageScale * sum B_k)`. This deliberately changes the pair
mode's coverage model so contributions fade when all slab support vanishes.
It is smooth, but not invariant to chart duplication or changing support width;
the coverage scale is an explicit control. No original member plane is rendered
as a separate hit. Unselected surfels, including light carriers, remain separate.

## Ray intersection

Slab-oriented bounds conservatively enclose all positive support. Unlike the
single-chart case, the bound is not inferred from the original ellipses' world
AABB. Each ray is split at every projected footprint and slab support boundary.
Within an interval, the numerator is a polynomial of degree at most 19: three
cubic quadratic tapers and one linear plane residual.

The polynomial is assembled directly in Bernstein form. Its coefficient convex
hull bounds its values, allowing empty intervals to be discarded. De Casteljau
subdivision visits near intervals first, with a maximum depth of 27 and a
scale-dependent termination tolerance. Each candidate is checked against the
actual field and its gradient. This is a numerical forward solver, not an exact
arithmetic or completeness guarantee at degenerate roots. There is no fixed-step
ray march and no fallback to independently rendered constituent planes.

## Scope and validation

The implementation remains a forward experiment for a single point-cloud
instance and two connected charts. There is no automatic neighborhood discovery,
topology adaptation, shadow/indirect transport, or adjoint. The existing backward
entry points reject the mode. Ordinary silhouettes, lost support, tangent rays,
and singular fields are outside the smooth-interior guarantee. Separately
rendering unselected surfels can still exhibit the original overlap behavior.

Build and run the native tests from the repository root:

```bash
acpp -std=c++20 -O2 -I Pale -I vendor/glm -I vendor/entt/src \
  python/test/shared_slabs_test.cpp -o /tmp/shared_slabs_test
ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=/tmp/shared-slabs-appdb OMP_NUM_THREADS=2 \
  /tmp/shared_slabs_test
```

They load the supplied PLY and test analytic field gradients against finite
differences, both support borders, oblique nearest hits against a dense scan,
chart-order invariance, and actual forward framebuffers while lateral and depth
boundaries pass through a camera ray. The previous pair tests remain applicable.
