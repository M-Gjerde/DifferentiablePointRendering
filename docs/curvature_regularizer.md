# Directional curvature scale regularizer

The regularizer couples an elliptical surfel footprint to the renderer's local
slab depth tolerance. Let `B` be a signed symmetric curvature tensor in the
surfel tangent frame, `D = diag(su, sv)`, `h = local_layer_depth_epsilon`, and
`gamma = 0.5`. The quadratic approximation gives

```
departure_max = spectral_radius(D B D) / 2
violation = max(0, departure_max / (gamma h) - 1)
loss = mean(violation²)
```

Here the spectral radius is the largest absolute eigenvalue. In principal
directions this reduces to `max(abs(k1)*su², abs(k2)*sv²)/2`. A cylinder therefore
allows elongation along its axis; rotated ellipses and signed saddle curvature
are handled by the same expression. Transformed tangent axes pull the
world-space curvature form back into surfel coordinates, including instance
scaling and shear.

For each pixel, the renderer selects the slab closest to its mean/median depth.
For each member, it fits `B * delta_center_tangent = delta_normal_tangent` using
other members' centers and unit normals. Each observation is normalized by its
tangent baseline length. This avoids estimating curvature from changes in
pixel compositing weights or hit ordering. Normal signs are aligned before
comparison. The fit excludes coincident centers, normal disagreement over 45
degrees (or the configured slab threshold if stricter), and pairs whose chord
has more than `h` displacement along their average normal. These are local
coherence heuristics, not a guarantee of surface identity.

The symmetric least-squares fit uses a truncated inverse: a sampling direction
with less than `1e-4` of the strongest direction's support contributes no
unobserved diagonal curvature. A single neighbor constrains normal variation
along its baseline but cannot determine a complete surface shape. Isolated
surfels and slabs without valid pairs produce no curvature observation.
Neighbors in separate slabs are not compared, so this regularizer cannot
recover curvature evidence after all coherent overlap has been lost.

Losses average over observed members, then over active pixel slabs using the
existing Python/native reduction. The fit, tangent frame, membership, and
observation counts are detached. Only the two scales receive gradients.
At tied maximum absolute eigenvalues, a balanced subgradient is used.

Densification receives the same unsquared violation and a positive semidefinite
tensor of the worst-departure direction in the surfel plane. The existing
curvature-triggered splitting consumes those statistics independently of the
loss weight. It must be enabled with a positive `curvature_violation_threshold`
to supply additional surfels; scale regularization by itself does not maintain
coverage. Existing point budgets, split sizes, and scheduling still apply.
The new directional violation differs from the old scalar average-radius
violation, so loss weights and split thresholds may need retuning.

Rebuild `pale` after changing the native helpers. From the repository root:

```bash
PYTHONPATH=/path/to/build:python ACPP_VISIBILITY_MASK=omp \
  ACPP_APPDB_DIR=/tmp/pale-curvature-appdb \
  python -m unittest test.test_curvature_math test.test_curvature_work -v
```

The math test compiles the production helpers with AdaptiveCpp and checks
known curvature tensors, ellipse bounds, finite-difference derivatives,
degenerate cases, transformed geometry, and coherence filtering. Renderer
tests cover the cylinder's free axis, scale adjoints, densification statistics,
inactive outputs, and isolation from RGB and other regularizers.
