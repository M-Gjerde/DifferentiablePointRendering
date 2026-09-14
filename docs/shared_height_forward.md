# Shared-height surfel forward experiment

The subsequent [two-slab extension](shared_slabs_forward.md) blends multiple
multi-surfel charts before intersection. The description below covers pair mode.

The native SYCL viewer can replace **two existing elliptical surfels** with a
single local height field. This is a forward experiment for a small scene with
one point-cloud instance (including emissive surfels) and no meshes. Other
surfels remain separate. It is not a scene-wide surface reconstruction or a
training implementation.

## Viewer

Build the `PaleRealtimeViewer` target, load your test scene, and open **Renderer
debug → Shared surface experiment** (leave **Two overlapping slabs** off for
the original pair mode). The two member inputs are GPU
surfel indices; `-1` automatically chooses the first two non-emissive surfels.
The resolved indices appear below the inputs. Change them to join a different
pair. Changing a member's position, rotation, scale, opacity, or color rebuilds
the height field on the next render. The original scene data is not rewritten.

Use **Display → Rendered** and **Height shading → Surface normals** to inspect
the reconstructed geometry, **Albedo** to see the color blend, or **Point lights
(unshadowed)** to see direct lighting. Disable the checkbox to return to the
original renderer. The toggle overrides the legacy camera-kernel selector.

Invalid indices, duplicate indices, emissive members, degenerate axes, or an
unsupported scene render blank; they do not silently fall back to plane hits.
The initial mode has no shadows, photon gathering, curvature diagnostics, or
adjoint. Both photometric and regularizer backward entry points reject it.

## Reconstruction

Let the chart origin be the mean of the two centers and its normal `N` be the
normalized mean of consistently oriented surfel normals. For a world point `x`,
project it onto the chart as `q = x - N dot(N, x - origin)`. For each member:

```
z_i(q) = dot(n_i, center_i - q) / dot(n_i, N)
x_i(q) = q + N z_i(q)
r_i²   = u_i(x_i)² + v_i(x_i)²
w_i    = opacity_i max(1 - r_i², 0)³
z(q)   = sum(w_i z_i) / sum(w_i)
F(x)   = dot(N, x - origin) - z(q)
```

Ellipse coordinates use the dual basis of the transformed ellipse axes, so
nonuniform instance scale and affine shear are supported. Membership consists
of the same explicit pair throughout the chart. There is no nearest-hit anchor,
depth-window rejection, or use of `h`. A contributor disappears only as its
footprint weight reaches zero.

The cubic kernel deliberately replaces the selected members' beta profiles in
this mode. Its zero extension is C2. Where the weight sum stays positive, the
reconstructed height is C2, including where one member enters or leaves support.
Normals include derivatives of both the member heights and the weights:

```
grad z = [sum(w_i grad z_i + z_i grad w_i) - z sum(grad w_i)] / sum(w_i)
normal = normalize(N - grad z)
```

Material color is the same normalized weighted blend. Coverage is
`1 - product(1 - w_i)`, applied once at the reconstructed hit. This retains soft
surfel coverage rather than making the entire union opaque.

## Intersection and limits

The joint world-space ellipse AABB bounds the reconstructed surface, since each
reconstructed point is a convex combination of points on the original ellipses.
Ray intervals are split at the projected ellipse support boundaries. Within
each interval, `sum(w_i F_i(ray(t)))` is a polynomial of degree at most seven.
Derivative-root isolation and bisection find candidate roots in front-to-back
order. The implementation uses double precision for the polynomial and checks
each candidate against the actual height field. It does not ray march or use
the original planes' intersections to discover the reconstructed surface.

This removes the pair's internal ±h membership seams. It does not remove true
silhouettes, disjoint-support boundaries, occlusion by unrelated surfels, or
floating-point error at ill-conditioned/tangent intersections. Outside positive
support the height is undefined. Opposed normals are oriented relative to the
first member; the orientation choice at exactly 90 degrees is a discrete chart
decision. This local chart is not a global representation of a closed horse.

The geometry is locally differentiable where the chart is valid and support is
positive. A future backward pass must differentiate the chart and normalized
weights and use implicit differentiation of `F(ray(t), parameters) = 0` at
transverse intersections. It must not reuse the existing slab adjoint or claim
differentiability at silhouettes, tangent roots, or membership edits.

## Verification

From the repository root:

```bash
acpp -std=c++20 -O2 -I Pale -I vendor/glm -I vendor/entt/src \
  python/test/shared_height_forward_test.cpp -o /tmp/shared_height_forward_test
ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=/tmp/shared-height-appdb OMP_NUM_THREADS=2 \
  /tmp/shared_height_forward_test
```

Tests execute the production helpers on host and SYCL, checking the 45-degree
pair, finite-difference normals, support-border continuity, oblique intersections
against a dense scan, and polynomial roots including a double root. A native
framebuffer smoke test calls the same forward launcher as the viewer and checks
pair-order and slab-thickness invariance and invalid-selection behavior.
