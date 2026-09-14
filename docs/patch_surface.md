# Connected, slab-contained surface experiment

This experiment starts the horse from the **five random nonemissive surfels in
the dataset's `points.ply`**, with one small bicubic patch per surfel. It does
not load a reconstructed mesh or the ground-truth horse. The three point
emitters are loaded from `lights.json`, and all dataset cameras and linear EXR
targets are used. Initial patch centers, orientations, extents, and albedos
come from the surfels. The initial supports are rectangles rather than the old
elliptical beta-opacity footprints. The new surface material is opaque,
two-sided diffuse; surfel opacity and beta are not parameters of this model.

## Run the horse

From the repository root, using the existing project conda environment:

```bash
PYTHONPATH=python /home/magnus/miniconda3/envs/DifferentiablePointRendering/bin/python \
  python/patch_train.py \
  --dataset ~/phd/datasets/horse_10_pbdr \
  --output python/OptimizationOutput/horse_patches \
  --iterations 1000 --resolution 64 --spp 16 --tessellation 8
```

Use a new output directory for each run. `--resume path/to/checkpoint.pt`
restores geometry, shared topology, slab parameters, optimizer state, and the
iteration number. `--iterations` then specifies **additional** iterations.
Keep the rendering and optimization options consistent when resuming. The
runner fails rather than overwriting an existing run without `--resume`.

The optional renderer dependencies are in `python/patches/requirements.txt`.
They are already available in the environment used for local validation.
The backend is **Mitsuba 3, CPU LLVM, `direct_projective`**, bridged to PyTorch
autograd. This is a separate experimental training path. It does not change
`python/main.py`, the native Pale renderer, or `--fixed-slab-membership`.
The native surfel adjoint is not valid for these new parameters and is not used.

Outputs:

- `checkpoint.pt`: exact patch/optimizer state for resume.
- `surface_*.ply`: welded tessellated surface, normals, and vertex colors.
- `rgb_*.npy`: linear camera-zero RGB; `rgb_*.png`: sRGB display previews.
- `training.jsonl`: camera loss, gradient norm, accepted step fraction,
  containment/regularity/seam certificates, tessellation error, topology events.
- `run.json`: initialization and experiment configuration.

## What optimization can change

Each initial component can translate and rotate independently. Its control
points can change shape and extent, and its color field can change. A shared
surface component uses a single rigid transform for all joined members.

At `--topology-interval` (default 25), the runner refits the local slab frames
around the current surface when the result is feasible. The surface itself is
unchanged by refitting. This renews the allowed deformation around the current
shape instead of permanently confining it to the initial five planes.
The slab half-thickness is the configured world-space limit, default 0.005;
it is not an unrestricted learnable thickness.

At `--split-interval` (default 200), every patch splits into four if this fits
within `--max-patches` (default 100). Uniform subdivision avoids unmatched
edge subdivisions. De Casteljau subdivision preserves the analytic geometry
and color-logit field exactly while adding independent degrees of freedom.
The default five seeds therefore grow to twenty and then eighty patches.

At topology steps, `--join-distance` (default 0.02) enables candidate joins
between nearby free edges. An added Hermite bridge shares the existing edge
positions and transverse derivatives, using compatible positive parameter
scales. It leaves the existing patches unchanged. The bridge is accepted only
if its control hull fits a slab and its geometry passes the same regularity
and injectivity certificates. Invalid candidates do not mutate the model.
Set the distance to zero to disable automatic joining. This is a geometric
proximity heuristic, not a guarantee that the correct horse topology is found.

Discrete topology/refit operations reset Adam moments because they change the
parameter basis. Gradients do not pass through splitting, joining, or refitting.
Joining adds actual surface area and may change the image at that operation.

## Hard constraints and differentiation

For matching parameter scales, the shared boundary constraints are

```
P[3,j] = Q[0,j]
P[3,j] - P[2,j] = Q[1,j] - Q[0,j]
```

Seams may transpose/reverse edges and rescale the transverse chart parameter.
The same scale is used in the derivative constraint. Shared values and
compatible derivatives make the geometry and normals continuous; constraints
on the color-logit field also remove appearance seams. A global nullspace of
the seam equations handles all existing junctions together. Conflicting edge
assignments, nonorientable networks, and degenerate resulting geometry are
rejected.

Let `P0` be feasible reference control points, `Z` a nullspace basis, and
`delta = Z @ raw`. A control's signed slab-normal displacement is `a(delta)`.
Its available slack is `s = h*(1-epsilon) - abs(a(P0-C))`. For each connected
component the decoder uses

```
D = (1 + sum_controls((a(delta)/s)**8))**(1/8)
P = P0 + delta/D
```

Since `D > abs(a(delta)/s)` for every control, all decoded control points
remain strictly inside their slabs. One denominator per component preserves
the coupled seam equations. The decoder is smooth, including at zero
displacement, and autograd includes its full derivative. It is conservative:
one constrained region can reduce deformation throughout its component.

A Bézier patch is a convex combination of its controls, so control-point
containment certifies the **entire curved patch**, without footprint sampling.
Component rigid motion transforms both the patches and their slabs.

Additional polynomial bounds certify positive projected Jacobians and a
strictly monotone slab-plane projection. The latter is needed to rule out a
patch wrapping over itself: positive determinant alone is only a local test.
The certificates are conservative and may reject a valid but difficult shape.
A backtracking optimizer step accepts only certified states. This is a
feasibility-preserving optimization step, not a hidden rendering clamp. The
renderer refuses uncertified geometry. These tests certify individual patches,
not collision freedom between different patches/components.

## Rendering accuracy and remaining limits

The analytic primitive is a joined bicubic surface. The reference renderer
uses a **conforming triangle approximation** with welded boundary vertices and
normals evaluated from patch derivatives. All vertex positions, shading
normals, and colors differentiate back to the actual constrained parameters.
Mitsuba's projective integrator also estimates derivatives of primary and
shadow visibility; film border sampling is enabled.

This is not an exact curved-patch ray intersector. Gradients are for the
rendered approximation. `tessellation_error_bound` certifies a conservative
world-space position difference between the mesh and the analytic surface,
using the Bézier convex-hull property on each subpatch minus its affine
triangle map. Increase `--tessellation` (a power of two) and compare renders
and gradients to assess discretization error. The position bound does not by
itself bound normal error or image error near silhouettes. Subdivision preserves
the analytic surface but can change its triangle approximation slightly.

There is no eight-hit/group limit. Dense nullspace construction currently
limits practical patch counts; this is a correctness experiment rather than a
large-scene production implementation. Self-intersections between patches,
accidental proximity joins, missing coverage, and recovery of the horse's
topology remain possible failure modes. Smooth joins and feasible geometry do
not guarantee convergence from five random seeds. Separate components can
still occlude or intersect one another until the intended connections form.

## Validation

```bash
PYTHONPATH=python /home/magnus/miniconda3/envs/DifferentiablePointRendering/bin/python \
  -m unittest test.test_patch_surface test.test_patch_renderer -v
```

Tests cover shared geometry and normals, large latent updates, finite-difference
derivatives through active containment and rendering, short-edge bridges with
unequal parameter scales, exact subdivision, rejection of invalid geometry,
checkpoint/refit preservation, and tessellation error bounds.

The local horse smoke run is under
`python/OptimizationOutput/horse_patches_smoke`. It starts from five seeds,
exercises image optimization and checkpoint resume, and subdivides to eighty
patches. It is an execution/constraint check, not a converged reconstruction.

References: [Bézier convex-hull property](https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bezier-properties.html),
[Mitsuba projective differentiation](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/projective_sampling_integrators.html).
