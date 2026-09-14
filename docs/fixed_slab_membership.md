# Experimental fixed slab membership

For the newer experiment with hard geometric containment, joined curved
patches, and training from the horse dataset's five surface seeds, see
[Connected patch surfaces](patch_surface.md). The fixed-membership mode below
preserves group labels; it does not constrain geometry to remain in a slab.

Enable `--fixed-slab-membership` in Python training, or pass
`{"fixed_slab_membership": True}` to `pale.Renderer`. The default remains the
existing ray-local slab renderer.

The verified local module is `build-fixed-slabs/pale.so`. From the repository
root, prepend `build-fixed-slabs:python` to `PYTHONPATH` when running training
with this build; otherwise Python may import an older installed `pale` module.

Each nonemissive surfel receives a persistent `slab_group_id`. Position,
orientation, scale, opacity, beta updates, device Adam steps, and ordinary BVH
rebuilds preserve that label. The member geometry evolves freely: the original
depth tolerance is used to construct groups, not to reject their later ray hits.

`add_points` and `remove_points` explicitly repartition the optimized cloud.
Consequently, the existing clone/split/prune training paths can create, divide,
and reshape groups without per-ray membership changes during an optimization
step. These are discrete topology changes and may change the rendered image;
gradients do not pass through the partition. To repartition after another
intentional topology operation, call `renderer.regroup_slabs()`.

## Initial partition and capacity

The initial partition is deterministic and camera independent. Nonemissive
seeds collect nearby candidates with overlapping bounding spheres, mutually
compatible center-to-plane distances, and the configured normal compatibility.
Candidates are ranked by distance; the search is bounded by the seed's extent.
This is a local heuristic, not an exact ellipse intersection algorithm or a
guarantee that every intersecting pair belongs to the same group.

Groups have at most eight members, matching the existing compositor and adjoint
event storage. Larger populations are divided into additional groups at topology
steps. Fixed mode uses the full eight-member capacity even if a lower debug
`max_local_surfel_hits` was requested. It never truncates a group per ray.
Explicit labels exceeding this capacity raise an error before scene mutation.

For a controlled experiment, set the partition yourself:

```python
import numpy as np

# One light, followed by two surfels that should always blend together.
renderer.set_slab_group_ids(np.array([0xFFFFFFFF, 12, 12], dtype=np.uint32))
ids = renderer.get_point_parameters()["slab_group_id"]
```

Labels are local to the optimized point-cloud asset. PLY checkpoints store them
as an exact `property uint slab_group_id`; initial, iteration, manual, and final
training snapshots preserve them. Loading a checkpoint in fixed mode reuses its
labels instead of inferring a new partition. Regrouping may renumber labels.

## Rendering rule

For each ray, evaluate every member footprint of a group and apply the existing
average-over-orders compositor to its contributing hits. Group hits are not
filtered by anchor depth, slab distance, normal agreement, or the candidate hit
list. A member whose footprint opacity vanishes has a vanishing contribution.

Each group is a virtual compositing layer at the opacity-weighted mean of its
member hit depths. Other groups and meshes are ordered against that depth. This
is an explicit approximation to visibility when another surface interleaves
the members. Traversal advances to the virtual depth, rather than the farthest
member, so it does not silently skip intervening surfaces. Member intersections
are evaluated along the entire ray line; this makes the virtual depth invariant
under a shift of ray origin and prevents rendering the same group twice. The
virtual depth must be in front of the ray origin. Cameras inside groups therefore
follow this virtual-layer convention, not per-member near clipping.

Fixed mode shades each member at its own hit, overriding shared slab direct
lighting. Direct-light shadow rays exclude the shading member's entire group,
and the adjoint uses the same exclusion. Other groups still cast shadows.
Camera RGB, measurement events, camera attenuation gradients, intra-slab loss,
and curvature selection use the same group collector. Depth/normal diagnostic
losses retain their original ordered physical-hit stream in both passes.

For fixed membership and group ordering, the existing analytic derivatives of
footprints and shading remain usable. This removes membership/order switches
*within* a group, not all visibility discontinuities between groups, hard shadow
boundaries, footprint derivative singularities, or discrete densification steps.
The group's virtual ordering depth is a traversal decision; no smooth visibility
gradient is claimed when two groups exchange order.

## Cost and validation

This is a correctness reference implementation. It scans group members within
each visited point-cloud instance, bypassing the capped point-hit batches for
grouped camera traversal. A group BVH with bounds refitted during optimization
is needed before expecting large-scene training performance. The existing
per-ray group/event budget and transport sampling limits still apply.

The implementation does not modify photon scattering into a grouped surface
model; indirect illumination retains the existing photon-map approximation.

Build the `pale` target, then run with that build directory first on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/build:python ACPP_VISIBILITY_MASK=omp \
  ACPP_APPDB_DIR=/tmp/pale-fixed-appdb MPLCONFIGDIR=/tmp/pale-fixed-mpl \
  python -m unittest test.test_fixed_slab_math test.test_fixed_slab_membership -v
```

Tests exercise full-group collection, depth-order changes, continuity across the
old slab boundary, member support boundaries, interleaved groups, self-shadow
exclusion, finite-difference RGB gradients, device updates, capacity validation,
topology changes, exact PLY labels, and both training paths with densification.
