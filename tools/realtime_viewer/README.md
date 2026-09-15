# Pale Realtime Viewer

Standalone ImGui viewer for orbiting the renderer camera without editing the existing renderer executable or top-level CMake files.

## Build

From the repository root:

```bash
cmake -S tools/realtime_viewer -B build-realtime \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=/usr/bin/clang-22 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-22 \
  -DAdaptiveCpp_DIR=/usr/local/lib/cmake/AdaptiveCpp

cmake --build build-realtime -j"$(nproc)"
```

The AdaptiveCpp driver must use the same LLVM major version as the plugin it
was built with. Verify this before configuring:

```bash
acpp --help | head
```

For a Clang 22 installation, the output must report `Plugin LLVM version: 22`
and an `--acpp-clang` current value pointing to Clang 22. Compiler selection is
done on the first CMake configure; use a fresh build directory when switching
compiler versions.

By default CMake fetches Dear ImGui if `IMGUI_SOURCE_DIR` is not set. GLFW is found from the system first and fetched only if no system package or `GLFW_SOURCE_DIR` is available.

To use local copies instead:

```bash
cmake -S tools/realtime_viewer -B build-realtime \
  -DIMGUI_SOURCE_DIR=/path/to/imgui \
  -DGLFW_SOURCE_DIR=/path/to/glfw
```

## Run

Rendering and diagnostic kernel compilation run on a worker while the main
thread continues processing window-system events. This prevents a long JIT
compile from starving the Wayland connection. The title indicates that rendering
or compilation is in progress; controls are applied after the current work
finishes. OpenGL presentation remains on the main thread. Closing during a render
waits for that work to finish before releasing its resources.

Startup logs identify the window backend. Shutdown logs distinguish Escape from
a window-system close request; the latter can also indicate a lost display
connection. An informational AdaptiveCpp JIT warning alone is not an error.

```bash
./build-realtime/PaleRealtimeViewer --assets Assets --pointcloud points.ply --scene cbox.xml
```

Controls:

- Default render mode: photon mapping
- Default camera source: viewport camera only
- Viewport camera convention: world `Z` is up; camera local forward remains `-Z`
- `Camera source`: switch between the orbit viewport camera and one selected `scene.xml` camera
- Left-drag over the rendered image: orbit camera
- Right-drag or middle-drag: pan target
- Mouse wheel: zoom
- `Render`: force a render
- `Auto render`: render after camera/control changes
- `R`: load the latest optimization run PLY
- `F`: load the first `iter_*_points.ply` in the active optimization `points` folder
- `L`: load the last `iter_*_points.ply` in the active optimization `points` folder
- Left/right or down/up arrows: step through snapshots in the active optimization run

Arrow-key navigation refreshes only the active run's `points` folder, including
new snapshots written during training. It stays in that run until you use `R`
or **Load latest run PLY** to search for the latest run again.

### Slab distance modes

Under **Surfel traversal → Slab distance**, select:

- **Along surface normal**: use the anchor-normal distance
  `abs(dot(x_i - x_anchor, n_anchor)) <= h`. The candidate ray interval expands
  by `h / max(abs(dot(n_anchor, ray_direction)), 0.05)` at grazing angles.
- **Symmetric along ray** (default for the viewer and training): use the fixed interval
  `[t_anchor - h, t_anchor + h]`, where `h` is the **Ray depth half-width**.
  The interval is clipped to the active ray; because the anchor is its first
  hit, the portion before the anchor normally contains no additional hits.

Only the distance test changes. Both modes retain the normal-alignment filter,
hit/member limits, order-averaged blending, transmission, and lighting rules.
The selected mode is shared by camera rendering, adjoint traversal, and slab
diagnostics. Changing it requests a new render, including when auto-render is
disabled.

Python renderer settings expose the same choice as
`local_layer_depth_mode="normal_distance"` or `"symmetric_ray_depth"`.
The existing point-to-plane intra-slab regularizer remains the training loss;
the mean ray-depth alternative is a diagnostic preview only.

### Surface curvature map

The display menu and **+/-** cycle group **Surface curvature (magnitude)**,
**Curvature scale**, and **Curvature primitive score** consecutively. The first
ten views use number keys **1–9, then 0** in menu/cycle order: **8** is Surface
curvature, **9** is Curvature scale, and **0** is Curvature primitive score.
Position primitive score follows them without a number shortcut.

Choose **Display → Surface curvature (magnitude)** for an estimate of local
surface bending. For each fitted member of the visible slab, the map uses
`max(abs(kappa_1), abs(kappa_2))`, the largest absolute eigenvalue of the fitted
world-space normal derivative, then averages these magnitudes over valid members.
Units are inverse scene units: a sphere of radius `R` has magnitude `1/R`.
The slab is selected near median depth by default, or mean depth if that option
is enabled, using the existing curvature diagnostic's visible-slab search.

This uses neighboring surfel centers and normals, rather than differentiating
the depth image. Neighborhood coverage and slab membership affect the estimate;
unobserved tangent directions cannot be recovered from a single neighbor.
The existing **Curvature scale** map instead shows a footprint-size penalty,
which can be zero on a curved surface with sufficiently small surfels.

Valid flat estimates are zero and use the low end of the colormap. Black means
no usable estimate (including background and isolated surfels). The logarithmic
color scale starts at zero and rescales to each frame's maximum, shown below
the display selector. The map is unavailable in **Shared surface experiment**.
It is a forward diagnostic and does not enable a training loss or splitting.

Python callers can opt in with `preview_surface_curvature=True`; the forward
result then contains `surface_curvature`, with `NaN` for unavailable estimates.

### Comparing intra-slab losses

The **Display** menu offers **Intra-slab depth (plane distance)** (shortcut **7**)
and **Intra-slab depth (mean ray depth)**. The latter previews
`mean_i(((t_i - mean_j(t_j)) / h)^2)` for each slab, with equal weight for all
members. Here `t_i` is world-space distance along the camera ray and `h` is the
slab distance tolerance. Slab losses are summed per pixel, just as in the
existing plane-distance view.

Both views use the selected slab membership rule and normal filter. They share
a logarithmic color scale from zero to the maximum of both maps for the current
frame, so equal colors represent equal loss values. The viewer also displays
both means over all pixels, including zero-loss pixels. Select the ray-depth view from **Display** or cycle with **+/-**. Number keys
**1–9, then 0** select the first ten views in that same menu/cycle order.

This is a forward-only comparison: training losses, their gradients, and the
shared point used for shading are unchanged. These slab diagnostics are
unavailable while **Shared surface experiment** is enabled.

For integration checks, the Python setting `preview_intra_slab_ray_depth=True`
enables the optional `intra_slab_ray_depth_preview` forward output. It is off
by default and has no backward output.

### Training debug defaults

Debug computations follow the selected **Display** view. Normal RGB browsing
skips surface diagnostics, regularizer backward passes, curvature searches, and
CPU SSIM calculations. Select a diagnostic directly from **Display** (or cycle
with `+`/`-`) to compute it; selecting **Rendered** stops that work again.
Regularizer gradient views compute only their selected loss and share one
gradient allocation. Curvature statistics are allocated only when their view
is requested. Explicit **Adjoint profiling → Every render** remains an opt-in
background profiling operation.

The debug controls mirror the current defaults in `python/config.py`: position
threshold `0.005`, radiance-bias strength `0.5`, weight limits `[0.2, 1.5]`, and
radiance floor `0.005`. Position previews start at this threshold when loading a
snapshot; **Use saved** selects its recorded threshold and **Use config default**
restores `0.005`.

The curvature split preview starts disabled (`0`); its logarithmic threshold slider
ranges from `0` to `1`. Select a positive threshold to preview split candidates,
or Ctrl-click to type a value beyond the slider range. This changes only the viewer
preview, not the training configuration. Disabled splitting still displays curvature scores without
magenta candidate highlighting. SSIM debug defaults are weight `0`, window size
`5`, and sigma `0.75`. These defaults are copied into the viewer; it does not load
`config.py` at runtime.

### Depth-distortion previews

The viewer defaults to **World distance + Gaussian falloff**, matching training
with `depth_distort_gaussian=True`. The **Half-strength distance (m)** control
defaults to 0.10: attraction is 50% at a 10 cm camera-forward depth separation,
6.25% at 20 cm, and about 0.2% at 30 cm. Both the loss image and position-gradient
preview use this mode and distance. Pair weights are detached in backward.
The loss has scene-distance units; geometry is assumed to be in metres.

The formula has no pixel-width, focal-length, resolution, or reference-depth
normalization. Viewing changes can still change intersections and visibility.
This changes diagnostics/regularization only, not slab membership or RGB.
Choose **World distance** for the original absolute depth loss or **Normalized
depth (legacy)** for runs with `depth_distort_world_space=False` and Gaussian
falloff disabled. Pixel-footprint mode has been removed.

The loss image shows raw per-pixel distortion; the position-gradient image uses
mean image loss with unit regularizer weight. Colors rescale separately each
frame, so equal colors across frames do not imply equal loss values.

## Shared-height surface experiment

In **Renderer debug**, enable **Shared surface experiment**. Leave the
two indices at `-1` to join the first two non-emissive surfels, or select their
GPU indices explicitly. Use **Display → Rendered** and the **Height shading**
selector for unshadowed point lights, albedo, or reconstructed surface normals.
The pair shares one height field with no depth-slab membership threshold.

This forward-only experiment requires one point-cloud instance and no meshes.
It uses a cubic footprint taper for the joined pair; shadows, indirect lighting,
and backward optimization are not implemented. See
[the implementation notes](../../docs/shared_height_forward.md) for the geometry,
limitations, and native tests.

For the multi-surfel experiment, also enable **Two overlapping slabs**. The
ready-to-load [test scene](../../Assets/Experiments/shared_slabs/README.md) includes
`points.ply`, a camera XML, and exact launch instructions. `--shared-slabs` enables
this mode at startup with its default two groups of four surfels. Its width,
depth, and offset controls move smooth support boundaries; **Slab weights**
visualizes the transition between charts. The geometry is blended before ray
intersection, and coverage is applied once to the common surface hit.
