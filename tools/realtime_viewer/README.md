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
- Left-click over the rendered image: select the surfel under that pixel for editing, including after resizing or using a scene camera
- Right-drag or middle-drag: pan target
- Mouse wheel: zoom
- `Space`: restore the initial orbit view and switch to the viewport camera
- `\`: enter walk navigation from the current viewport or selected scene camera
- In walk mode: mouse look, `W/A/S/D` move, `Q/E` down/up, `Shift` faster, `Alt` slower, wheel adjusts speed
- `Enter`, left-click, or `\`: finish walking and keep the view; `Escape` or right-click: cancel and restore the previous view
- Walk navigation has no gravity or collision detection; losing window focus cancels it
- Debug-view shortcuts (`1`–`0`, `+`/`-`) and point-cloud shortcuts (`R`, `F`, `L`, `N`, `M`, arrows) remain available while walking
- Walk movement uses the latest input state after each render; released keys are not replayed across later frames. Short exit and viewer-shortcut taps still register.
- `Render`: force a render
- `Save screenshot (PNG)`: save the currently displayed render with transparent background
- `Auto render`: render after camera/control changes
- `Median depth settings`: adjust the accumulated-opacity threshold from 0.001 to 0.999 (default 0.5). `Retain last accepted depth` keeps the last surface hit when the threshold is never reached (default on). Empty rays still produce zero depth; the existing traversal-limit fallback is unchanged. Both controls request a fresh render, including depth-derived diagnostics.
- `Add surfel`: create a camera-facing surfel at the current viewport focus and select it for editing
- `Save as startup default`: save the currently edited surfels beside the original PLY and load them automatically on later runs
- `Reset to original`: reload the original startup PLY and clear the saved startup override
- `R`: load the latest optimization run PLY
- `F`: load the first `iter_*_points.ply` in the active optimization `points` folder
- `L`: load the last `iter_*_points.ply` in the active optimization `points` folder
- Left/right arrows: step through snapshots in the active optimization run
- Up/down arrows with a scene camera selected: next/previous scene camera, wrapping at either end
- Up/down arrows with the viewport camera selected: next/previous optimization snapshot

Snapshot navigation refreshes only the active run's `points` folder, including
new snapshots written during training. It stays in that run until you use `R`
or **Load latest run PLY** to search for the latest run again.

### Transparent screenshots

Click **Save screenshot (PNG)** beside the render controls. Images are saved to
`output/screenshots/viewport_<timestamp>_<sequence>.png` under the repository
root; the full saved path appears below the button. Each capture gets a new name.

The PNG uses the last displayed render's native resolution, RGB/exposure, and
rendered surface alpha, including partial opacity at footprint edges. It excludes
the gray viewport background, grid, gizmos, and UI. **Force opacity = 1** affects
the viewport display only and is ignored for screenshots. Diagnostic views save
their selected colors with the rendered surface alpha. If auto-render is off,
click **Render** first to capture camera or scene edits that have not yet rendered.

### Slab distance modes

**Renderer debug → Surfel traversal → Max splat events** defaults to **8**.
The compiled per-ray capacity is also 8, so the slider and Python's
`max_splat_events_per_ray` can reach that value. The separate local slab member
limit remains 8. A larger event limit lets partially transparent rays visit more
layers before traversal stops; the actual cost depends on opacity and overlap.

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

### Diagnostic shortcuts

Number keys select the first ten display modes: **1** Rendered, **2** Median depth,
**3** Depth distortion, **4** Mean depth, **5** Visible normal, **6** Depth normal,
**7** Intra-slab plane distance, **8** Surface overlap, **9** Position primitive
score, and **0** Intra-slab mean ray depth. Other diagnostics remain available
in **Display** and the **+/-** cycle.

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
skips surface diagnostics and regularizer backward passes. Select a diagnostic directly from **Display** (or cycle
with `+`/`-`) to compute it; selecting **Rendered** stops that work again.
Regularizer gradient views compute only their selected loss and share one
gradient allocation. Dominant primitive IDs are computed only for views that
need them. Explicit **Adjoint profiling → Every render** remains an opt-in
background profiling operation.

The debug controls are generated from `python/config.py` at build time: position
threshold `0.0005`, radiance-bias strength `0.8`, weight limits `[0.25, 1.5]`,
radiance floor `0.001`, and crowd gate `3.0` with the current defaults.
CMake regenerates these values when the config changes. Position previews start at this threshold when loading a
snapshot; **Use saved** selects its recorded threshold and **Use config default**
restores the config value.

### Crowd penalty in 3D

Select **Display → Crowd penalty (clone / split gate)**. Each primitive is colored
by its full-footprint mean slab membership divided by the selected crowd gate.
Magenta means membership is **at or above** the gate: training would block both
cloning and position splitting for that parent. This is a hard eligibility gate,
not an additive loss or a continuous reduction of the cloning signal. Below the
gate, the gradient, size and budget checks still apply. Gray means unobserved;
training allows these parents through this gate. Black is background.

**Crowd gate (mean members)** previews other thresholds; `0` disables blocking.
**Use config default** restores the build's `densification_max_mean_slab_members`.
The overlap distance and normal controls are shared with **Surface overlap**.
The calculation includes the anchor, excludes lights, and uses 64 footprint
samples per saved scene camera. The score stays attached to each primitive while
orbiting. These controls preview the gate without modifying the training config.

### Target image loss in 3D

Select **Display → Target RGB loss (projected into 3D)**. The viewer discovers the
dataset and explicit target color space from the loaded snapshot's
`run_config.json`, or looks for `images/` beside the scene XML. You can enter a
dataset or images directory and choose **Linear** or **sRGB**, then **Apply targets**.
Image stems must match scene camera names exactly, and dimensions must match.
EXR, HDR, PNG (including 16-bit), JPEG, BMP and TGA are supported. Unsupported
images, duplicate names and mismatched dimensions produce a visible error.
Runs saved with automatic color interpretation require an explicit choice here;
the viewer does not perform ICC profile conversion.

Each matching camera is rendered at its native resolution using the current
viewer renderer settings. The error is the training RGB objective
`0.5 * mean_rgb((render - target)^2)` in linear light, before exposure or output
gamma; alpha is ignored. Camera rays distribute that error with the renderer's
slab weights and accumulated transmission, including its batched traversal,
normal/depth filters, member and event limits. Each surfel stores weighted mean
error over its visible pixels. Hidden surfels receive no contribution through
opaque geometry. Placed instances have separate scores.

Choose **All target images** for the visibility-weighted mean or the worst
camera's mean, or select one camera under **Loss source camera**. The all-camera
mean normalizes each image by its pixel count, matching training's image mean.
Gray means there is no target observation; zero measured loss uses the lowest
colormap color. Colors remain fixed while orbiting. **Loss color maximum**,
**Logarithmic loss colors**, and **Fit loss colors to peak** control the scale.

The first projection renders all matching targets and may take a while. It is
cached until geometry or renderer settings change; **Reproject target loss**
refreshes targets edited on disk. Switching back to **Rendered** stops diagnostic
work. This view requires photon mapping with **CameraGatherKernel2** and the
shared surface experiment disabled. It uses current geometry and lighting, so
matching the saved training renderer settings matters when comparing losses.

This is image-error attribution, not a geometric loss or a position gradient.
Completely missing surfaces cannot be colored in 3D: the panel reports the loss
and number of pixels that have no surfel contribution (including empty rays,
mesh-only hits and fully transparent footprints). The image mean still includes
those pixels. Missing target cameras are counted explicitly. Viewport colors
use the frontmost geometric surfel, as in the position-score and overlap views.

Regression checks: `python3 -m unittest discover -s python/test -p test_target_loss_projection.py`.

### Depth-distortion previews

Choose **World distance** for absolute camera-forward depth differences or
**Normalized depth** for squared NDC differences. Match the training run's
`depth_distort_world_space` setting. Both the loss image and position-gradient
preview use the selected mode.

The loss image shows raw per-pixel distortion; the position-gradient image uses
mean image loss with unit regularizer weight. Colors rescale separately each
frame, so equal colors across frames do not imply equal loss values.

### Surface overlap and slab capacity

Press **8** for **Surface overlap (world space)**. **Normal-distance overlap** is
enabled by default. It counts intersections within **Overlap distance tolerance**
along the anchor surfel's normal, both in front and behind. The default tolerance
is `0.005` scene units. Disable the checkbox to compare symmetric camera-ray
distance instead. All three metrics recompute immediately using the selected mode.
At grazing angles the normal-distance interval widens along the ray without the
renderer's grazing-angle clamp; intersections must still fall inside the neighbor's ellipse.

Overlap mode and tolerance are independent of rendering slabs. Membership still
inherits the normal cosine and maximum local surfel hits from **Renderer debug →
Surfel traversal**. Both normals are faced toward the camera before their dot
product is compared. Cosine `-1` disables normal rejection. There is no alpha
cutoff: the diagnostic measures full geometric footprints, excluding lights.

The **Slab overflow footprint (%)** metric samples 64 locations on each surfel
from saved scene cameras, using only sample/camera pairs inside the image.
It reports the percentage whose potential membership exceeds the current slab
capacity. Counts include the anchor surfel: at capacity **8**, **9** members
overflow. **Mean slab members** and **Center slab members** show uncapped counts
averaged over sampled views. Their color maximum only controls saturation;
overflow colors always range from 0% to 100%.

Scores are cached per geometry, camera set, and membership settings. Orbiting
the viewer keeps the saved-camera scores fixed. Without saved cameras, the
displayed camera is used and moving it updates the scores. These are potential
slabs anchored at the sampled surfel, not an actual render traversal: occlusion,
transmission, candidate-batch limits, and ray-event limits are not applied.
An unobserved surfel has no score. Changing the view or its color maximum does
not alter training or split selection.

Training uses the same mean-members calculation, with a current default gate of `3`.
Set `--densification-max-mean-slab-members 2` for a stricter threshold, or `0`
to disable this gate. A value of 3 includes the parent: roughly two
other members per sample on average. The score is a mean across footprint
locations and in-frame scene-camera rays, not a count of distinct neighbors that
touch any part of the surfel. For example, half the samples at 2 members and half
at 10 give a mean of 6.

Training also defaults to symmetric normal-distance overlap. Use
`--no-surface-overlap-normal-distance` for the camera-ray comparison and
`--surface-overlap-depth-tolerance 0.005` to set the independent tolerance.
Viewer controls preview these rules without changing a training run's settings.

At each scheduled densification event, training recomputes scores from current
geometry before selecting candidates. Parents at or above the threshold are
excluded from position-triggered splits and optional exact
clones, before the candidate budget is applied. Accepted splits keep their
existing offset and scale policy. With `--densification-verbose`, each check logs
the overlap threshold, blocked candidate count and percentage, candidates remaining
after the overlap gate, unobserved candidates allowed through, and measurement
time, including when all candidates are blocked. Counts refer to gradient
candidates before size checks and the new-point budget; `added` reports the final
number of new surfels. Geometry-only scores include
hidden and transparent surfels; zero usable samples are unknown and do not block.
This experimental gate does not prune existing clusters or impose a strict
post-split maximum: simultaneous splits can raise occupancy above the threshold,
and sparse edges can dilute a crowded center's mean.

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

## Profiling the training workload

For headless timing of the viewer's normal RGB path, explicitly build
`PaleSplatEventsBenchmark` in the viewer build directory. Its source and the
alternating-build controller are in `python/test/benchmark_splat_events.cpp` and
`python/test/benchmark_splat_events.py`. The controller keeps old and new binaries
resident, warms all cameras, and alternates old-capacity/limit 8, new-capacity/limit
8, and new-capacity/limit 12 in symmetric order. It tests shared and per-surfel
lighting, excludes JIT, readback and diagnostics from timing, and collects traversal
counts and RGB differences separately. Pass `--help` to the Python controller for
its required scene, frozen PLY, executable and fresh output paths. Other GPU work
can still affect these measurements; record that load when interpreting results.

Python exposes the viewer's instrumentation through
`Renderer.set_profiling_enabled(timers=True, counters=False)`,
`reset_profiling_stats()`, and `get_profiling_stats()`. The result contains raw
timer records, named traversal/gather counters, and scene counts. Timers use the
same process-wide collector as the viewer; disable them after the measurement.
Counter attachment survives parameter uploads and BVH/topology rebuilds.

`python/test/profile_main_checkpoint.py` wraps the real `main.main()` path with
API wall timings and NVTX ranges. It restores saved renderer settings as well as
optimizer configuration. For a Restaurant checkpoint at iteration 30,000:

```bash
PYTHONPATH=cmake-build-corrections:python python python/test/profile_main_checkpoint.py \
  --checkpoint python/OptimizationOutput/paper/restaurant --iterations 30102 \
  --timing-json /tmp/restaurant-profile.json --output /tmp/restaurant-profile-run \
  --adjoint-spp 4 --local-layer-depth-epsilon 0.005 \
  --save-interval 0 --save-ply-files-interval 0 --mesh-extraction-interval 0 \
  --no-save-final-mesh --no-metrics --no-image-preview
```

Use fresh output paths. The default capture excludes the first training step;
initial/final all-camera rendering and exports are separate from training time.
Checkpoint loading restarts Adam moments and RNG state. `--use-final-ply`
explicitly selects `points_final.ply` instead of the newest numbered snapshot.

For repeated fixed-geometry measurements, add `--probe-camera Camera.096` and
request a shorter run. This uses zero learning rates and skips all-camera
exports. `--probe-train` instead applies the saved learning-rate schedules.
`--verification-npz /tmp/restaurant-verification.npz` captures parameters, Adam
moments, and the selected camera's RGB after the timed work. Zero-rate probes
have zero Adam moments, so use a nonzero-rate probe to compare gradient effects.

Use separate runs for wall timing, `--native-timers`, and `--native-counters`:
timers introduce synchronization, and counters add GPU atomics. These are
software work counters, not hardware utilization counters. Timer spans are
nested and must not be summed as independent costs. An Nsight Systems trace
can select the `training_capture` NVTX range; analyze exported SQLite files with
`python/test/analyze_training_trace.py` (`--skip-updates 0` if the trace already
excludes warm-up). Match lighting mode, samples, losses, and geometry when
comparing the viewer with training.

For comparisons while another training process is running, use the alternating
checkpoint benchmark. It keeps both native modules resident and runs only one
probe step at a time; initialization, JIT and controller waits are excluded.
Each block must cover the same cameras. Use a separate directory for each
native module so rebuilding cannot replace a module mapped by a probe:

```bash
python python/test/benchmark_checkpoint_abba.py \
  --baseline-module-dir /absolute/baseline-module \
  --optimized-module-dir /absolute/optimized-module \
  --checkpoint python/OptimizationOutput/paper/restaurant --checkpoint-iteration 30000 \
  --camera Camera.096 --camera Camera.050 --camera Camera.010 \
  --rounds 6 --block-steps 3 --warmup-steps 3 \
  --output-dir /tmp/restaurant-abba \
  -- --adjoint-spp 4 --local-layer-depth-epsilon 0.005
```

`python/test/compare_training_trace_windows.py` compares SQLite exports using
the complete fused-step NVTX spans. It reports kernel activity, GPU gaps,
transfer bytes and synchronization counts separately. A long host copy or wait
API call can be waiting for prior kernels; it is not evidence of slow PCIe
transfers. Hardware counters are needed to distinguish arithmetic limits from
GPU-memory stalls inside a kernel.

The native renderer's `adjoint_primary_slab_cache` setting defaults to true.
For a single point-cloud instance with batched hits, one adjoint bounce and
multiple samples, it reuses the exact first slab from sample zero. Every camera
and backward call overwrites hits and misses before reuse; later null-event
traversals and random samples are unchanged. The cache uses 512 bytes per pixel
in the current build (122.1 MiB at 500×500), reported by
`get_backward_allocation_stats()["primary_slab_cache_bytes"]`. Set the native
constructor setting to false to compare against recomputation. Unsupported
paths retain recomputation automatically.

### Adjoint radiance derivative / sampling variance

Select **Adjoint radiance derivative (signed)** under **Display**. This runs the
existing backward pass with source `(1,1,1)` at every pixel of the current view
camera. Each pixel shows `d(R+G+B)/d(property)` in linear radiance, without
photometric loss, pixel-count normalization, tone mapping or regularizers.
Red is positive, blue negative, white zero, and magenta non-finite.

Choose position, local rotation (radians), scale, opacity, albedo R, or beta.
**All surfels (common change)** sums the pixel derivatives for the same scalar
perturbation on all surfels; disabling it selects a global GPU surfel index. You can also click the viewport
and press **Use picked surfel**.
This is a camera-pixel Jacobian image, including shadow and camera occlusion
contributions, rather than a projection of each visible surfel's total gradient.
The current shared-lighting and traversal settings still apply.

**Derivative adjoint SPP** ranges from 1 to 256 and averages the samples. The seed
stays fixed when SPP changes; **Resample adjoint** increments it. Repeating a
fixed configuration reproduces its samples, up to floating-point atomic order.
**Full color at |derivative|** stays fixed across renders; **Fit derivative
colors** explicitly sets it to the current peak. Hold this scale, geometry,
camera, and seed fixed when comparing SPP. The ordinary screenshot button saves
the signed map, including derivatives on pixels with zero rendered opacity.

Adjoint scattering uses the configured fixed `qNull` and `qReflect` probabilities,
shown below the SPP slider. Selected branches retain their inverse-probability
weights. Positive probabilities for both branches preserve derivative support
at opacity zero and one.

Python diagnostics expose `debug_images=True`, `debug_all_surfels=True`
(or `debug_surfel_index=N`), `adjoint_passes`, `adjoint_q_null`, and
`adjoint_q_reflect`. These settings do not change forward sampling.
