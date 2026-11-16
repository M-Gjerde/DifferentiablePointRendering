# PBdR

The v1 reconstruction workspace launches the existing Python PBdR optimizer,
inspects its current scene, tunes supported hyperparameters, and edits independent
snapshots. The viewport remains the main work area, with a searchable scene tree,
an icon inspector on the right, and a collapsible bottom run monitor. Styling
follows the approved Figma/SVG palette, proportions and compact controls.

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

Building `PaleRealtimeViewer` also updates the Python training module through
the existing `pale` target in the repository's `cmake-build-pybind` directory.
That directory must already be configured with `BUILD_PYBIND=ON`; use
`-DPALE_TRAINING_BUILD_DIR=/path/to/training-build` when configuring the viewer
to select another configured training build. The update is incremental and
requires no installation or `sudo`. The viewer's **Native module directory**
must point to the training build's output directory.

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

## Start a reconstruction

Open **Reconstruction → Training setup** or the sliders icon in Properties. The
dataset dropdown discovers complete datasets directly beneath
`~/ValeDrive/phd/pbdr/datasets/pbdr_synthetic`. A complete entry has `images/`,
`scene.xml` and `points.ply`; optional ground truth lives alongside them as
`gt.ply`. Selecting a dataset configures the next reconstruction and leaves the
currently inspected scene and camera unchanged. **Load initialized point cloud**
below the dropdown explicitly opens that dataset's starting scene without
launching training, preserving the previous editing workspace on disk. **Start
reconstruction** launches the selected dataset and follows its new live scene. The last selected dataset opens on startup, unless
explicit scene/point-cloud CLI arguments are supplied.

The equivalent command is now:

```bash
python main.py -s ~/ValeDrive/phd/pbdr/datasets/pbdr_synthetic/teapot_pbdr \
  -o OptimizationOutput/teapot
```

Use **Open previous reconstruction...** in the Reconstruction panel or menu to
reopen a saved run. **App outputs** lists reconstructions beneath the configured
output root, including nested run folders, newest creation time first. The table
shows dataset, creation time, run folder and available loss/GT histories. Filter
by dataset or folder, or choose a dataset from the dropdown; click headers to
sort. A single click selects a row. Use **Open selected reconstruction**, Enter,
or a double-click to open it; Up/Down moves the selection. **Choose folder** opens
outputs saved elsewhere and lists folders by creation time, newest first. Dates
use local time; `*` marks modification time on filesystems without creation time.
Folder navigation keeps the dialog open. Select the run
folder containing `run_config.json`, `scene.xml`, `metrics.csv` and its saved
PLYs (a `points/` folder is also accepted).

Opening restores the latest saved PLY, all available loss and regularizer
history, population/runtime metrics, and recorded Chamfer, accuracy and completion
evaluations from `geometry_metrics.csv`. Logs are restored when retained. Run
monitor selects **Metrics** automatically; **Checkpoints** and `M`/`N` browse the
saved scene history, while `R` reloads the selected run's newest saved PLY and
metrics, preserving the camera. With no run selected, `R` selects the newest
loadable reconstruction under the configured output root (or anchored folder).
Incomplete output folders are skipped. The source is
read-only until an edit creates a private working snapshot. Missing GT metrics
are reported explicitly; opening a run does not recompute them.

The browser and history loading run in the background. Opening a saved run does
not launch, resume or control an optimizer, or replace the dataset/configuration
selected for the next reconstruction. Finish or stop an active reconstruction
before opening another output. Starting a new reconstruction returns to live
inspection and clears the older run's plots.

Both CLI training and the viewer discover dataset-local `gt.ply`. Disable
**Evaluate with dataset ground truth** to skip evaluation (CLI:
`--no-dataset-ground-truth`). An explicit `--gt` still overrides discovery.
**Run options** contains output root, library location, custom-folder fallback,
and checkpoint/live-publication cadence. Select the interpreter
and working directory under **Python environment**; the working directory must
contain this repository's updated `main.py`. If needed, set **Native module
directory** to the directory containing the compatible `pale.so`. An empty value
inherits `PYTHONPATH`. The interpreter needs the same dependencies as CLI training.
The initial setup prefers an active virtual environment, an existing repository
`.venv`, or a local Conda environment named `pale`. It also detects this
repository's `cmake-build-pybind/pale.so` when present. Explicit saved environment choices take precedence over detection. Legacy
preferences that only recorded the generic `python3` default migrate to the
detected environment; **Use detected environment** restores that choice.

**Start reconstruction** creates a fresh `run_<timestamp>_<sequence>` beneath the output
root. Existing runs and source datasets are not overwritten. Each launch saves
its exact argument list and stdout/stderr in a private `.viewer_sessions` sibling
directory. Execution passes arguments directly, including paths with spaces.
Only the process launched by this viewer is controlled; there is no external-run
attachment or process-discovery interface. Closing the viewer terminates its reconstruction and its worker processes.
A viewer crash also triggers cleanup. Existing logs and completed checkpoints
remain on disk; closing is not a request to finish another training update.

Each launch reads the current `python/config.py` defaults. **Override run settings
from config.py** explicitly overrides iterations, checkpoint cadence and adjoint
sampling; **Hyperparameters → Override launch defaults** independently overrides
the listed learning rates and regularizer weights. Leave both off to pick up
subsequent edits to `config.py`. Dataset, output, GT evaluation and viewer live
publication settings are still supplied by the viewer. Applying a live tuning
patch does not automatically enable overrides for the next launch.

The viewport's scene-source dropdown offers these modes:

- **Live training** follows current optimizer parameters, read-only. Publications
  use atomic binary PLY packets at the configured live interval (default one
  second), independently of persistent checkpoints. A single serialization worker
  owns its data and skips publication when busy; it retains only a small window
  of packets. GPU-to-CPU copies still have a cost, so increase the interval for
  larger scenes. Viewport rendering competes with training for GPU resources.
- **Follow saved** follows persistent checkpoints, read-only. The default save
  interval is 100 updates; completed runs also expose `points_final.ply`.
- **Browse checkpoints** opens **Run monitor → Checkpoints**. Drag the checkpoint
  scrubber, select an iteration, or use Previous/Next. **N/M** jump to the saved
  checkpoint nearest the previous/next 1,000-iteration boundary, always moving in
  that direction when possible. The inspected checkpoint stays fixed while
  optimization continues and the list grows. The previous image stays visible until
  the chosen checkpoint has rendered, and browsing preserves the camera. **R**
  returns to current live data.
- The viewport toolbar always shows the displayed surfel iteration, including
  while the Run monitor is on Metrics. During a checkpoint change it shows, for
  example, **Iteration 6000 → 7000 (loading)** until the new image is displayed.
  Hover the label for the displayed, requested and current run iterations.
- **Edit latest saved snapshot** copies the latest checkpoint into a private
  editable workspace. Geometry, material, and emitter edits do not pause or alter
  training. Switching to a training view preserves the edits; **Return to edits**
  restores them. **File → Save editable snapshot** stores the working PLY and mesh
  emitter transforms/colors/power under `output/viewer_snapshots`.

Selection controls and transform gizmos also work in read-only training views.
The first actual property/gizmo change creates an independent editable copy of
the displayed scene, preserving selection and camera. Training keeps running.
**R** preserves those edits and returns to live data; **Return to edits** restores
the private copy.

To use edits for another reconstruction, select **Initialize from edited snapshot**
and launch after the previous process finishes. The viewer freezes the input PLY
and scene metadata in `.viewer_seeds`, so subsequent editing cannot change the
new run's initialization. This starts a new optimizer; it does not resume Adam moments.

**Hyperparameters** exposes position, rotation, scale, albedo, opacity and beta
learning rates, plus alpha, depth distortion, normal consistency and intra-slab
depth weights. Leave **Override launch defaults** off to use Python defaults.
During a run, **Apply to training at next iteration** validates the complete patch,
waits for an iteration boundary, and records its effective iteration in
`parameter_changes.jsonl`. Checkpoint resume restores applicable journal entries.
Learning rates precede existing scene/schedule scaling. The current backend keeps
emitter power learning rates at zero; this UI preserves that behavior. Editable
emitter changes can seed a new run.

**Run monitor → Progress** shows training/viewed/saved iterations, objective history,
update time with iterations per second, a larger objective plot,
the output directory and log, and **Pause**, **Continue**, **Stop**, and **Save
checkpoint** actions. **Stop** immediately terminates the viewer-owned optimizer
and its mesh/glTF workers, including during finalization or an unacknowledged
command. It keeps the displayed scene and existing saved checkpoints, without
waiting for a final save or starting an export. Normal completion still performs
the configured mesh/glTF export. **Metrics** retains the complete objective and image-loss history, with an optional
rolling iteration window and log scale. Weighted normal and depth regularizers share a separate plot with independent
left (normal) and right (depth) Y-axes. Alpha/slab losses have their own plot under
**Other regularizers / raw terms / gradients**. Population, pruning, runtime, gradients and all recorded latest
CSV fields are available below. GT Chamfer distance, accuracy and completion
appear after mesh evaluations, at the cadence set by `mesh_extraction_interval`
in `config.py`; these describe the evaluated mesh, not every live update.
Hover any metrics plot to inspect the nearest recorded iteration and each
curve's original value, including values on separate Y axes or a logarithmic
scale. A vertical guide and point markers identify the inspected samples.
**Cumulative pruning** sums area, opacity and transport removals from the start
of the run; a rolling window crops the visible iterations without resetting
those totals.
**Log** interprets tqdm carriage returns/terminal line updates. Drag to select
text and use Ctrl+C, or use **Copy log**. A selection stays stable while new output
arrives; clearing it resumes display updates.

Dataset selection remains available while stopping. After launch, **Start
reconstruction** becomes **Reconstructing...** with a spinner and remains disabled.
It shows **Paused**, **Stopping...** or **Finalizing...** during those stages, then
returns to **Start reconstruction** after the process exits. A **Stop** button
appears immediately beside it in both the top bar and Reconstruction panel while
the run is active, and remains available during finalization. New reconstructions cannot be queued; starting the next one always requires
another click. Editing never sends a pause command. No command forcibly kills an
unrelated process.

## Open the viewer

Rendering and diagnostic kernel compilation run on a worker while the main
thread continues drawing the full workspace and processing input. Logs, metrics,
training commands and dataset selection stay usable during kernel compilation.
The viewport keeps its last image and overlays. Inspector panels retain their
last complete layout while work is in flight; clicks and scrolling are applied
at handoff. Preview refresh waits during active inspector edits and open menus
so controls keep their focus and layout. Reconstruction controls and logs remain
interactive throughout. Scene replacement also runs in the
background. OpenGL presentation remains on the main thread. Closing stops the owned training
process before waiting for in-flight preview work to release its resources.

Preview exceptions retain the last good image and retry up to three times with
backoff. Repeated failures leave a stable notice with **Retry preview** and
**Copy error**. A preview taking more than 15 seconds is reported once without
starting a second renderer concurrently. Records are written beneath
`output/viewer_diagnostics/preview_*.jsonl`. A persistent renderer worker avoids
creating a new OS thread and device thread-local state for every preview.
A GPU call that never returns still requires process/context recovery; the UI
can report that condition but cannot safely cancel an in-process device call.

**File → Export mesh…** extracts the currently inspected PLY using the existing
TSDF script. Textured glTF (`.glb`) is optional and off by default; enabling it
reveals texture size, UV partitions/threads, lights/cameras and scene-unit
settings. Export freezes its inputs and leaves the viewport on its point cloud.
The optimizer's existing final PLY and textured GLB export remains controlled by
`config.py`; completion does not open a mesh in the viewport. See
[mesh export details](../../docs/viewer_mesh_export.md).

Startup logs identify the window backend. Shutdown logs distinguish Escape from
a window-system close request; the latter can also indicate a lost display
connection. An informational AdaptiveCpp JIT warning alone is not an error.

```bash
./build-realtime/PaleRealtimeViewer --assets Assets --pointcloud points.ply --scene cbox.xml
```

**Scene Collection** groups surfel geometry, mesh materials, emitters and
calibrated cameras. Search by name; click a row to select its model element or
camera. Large surfel-emitter collections show the first 256 plus an explicitly
selected emitter; viewport picking remains available beyond that list.

The Properties icon rail has hover/focus tooltips:

- Cursor: selected geometry, material or emitter properties.
- Sliders: reconstruction setup and hyperparameter tuning.
- Layers: viewport display and the active diagnostic's settings.
- Camera: free-view navigation and calibrated-camera selection.
- Folder: standalone scene-file and checkpoint inspection.

**Advanced** is always available as the sixth inspector icon, exposing the existing
research diagnostics. The viewport selector includes all 22 supported debug views. **Refresh
preview** and **Auto preview** live in the application header; the viewport owns
its inspection mode, source selection, transform tools and single **Gizmos &
overlays** popover. Grid starts enabled. Local/world transform coordinates and
emitter/surfel handles are configured in that popover.
**Display** groups the view selector, context-sensitive **View options**, screenshot
capture, and collapsed **Appearance** controls. Only options for the active view
are shown; depth controls appear with the corresponding depth views.
**Scene files**, **Lights**, and **Surfel editor** keep editing controls out of the
way. Integrator selection and renderer experiments live in **Renderer debug**
under **Advanced**.

The **Profiling** panel starts closed. Open it from **Window → Profiling** or
**Advanced tools → Scene statistics & profiling**.

Controls:

- Default render mode: photon mapping
- Default camera source: viewport camera only
- Viewport camera convention: world `Z` is up; camera local forward remains `-Z`
- `Camera source`: switch between the orbit viewport camera and one selected `scene.xml` camera
- Left-drag over the rendered image: orbit camera
- Click a gold light marker: select a mesh area light or emissive surfel and open/scroll to its editing controls. Markers remain visible through geometry, show names on hover, and highlight the selection. Dragging a marker does not orbit or select a surfel.
- Selected mesh lights support move/rotate/scale; selected surfels support move/rotate in editable snapshots, in both viewport and scene camera views. The viewport's left tool rail chooses select/move/rotate/scale.
- Left-click elsewhere over the rendered image: select the surfel under that pixel for editing, including after resizing or using a scene camera
- The ground grid starts enabled; toggle it under **Gizmos & overlays → Ground grid**
- The viewport docking tab bar starts hidden; its docking menu can reveal it
- Right-drag or middle-drag: pan target
- Mouse wheel in the orbit view: zoom the camera
- Calibrated scene-camera images are centered with their aspect ratio preserved. Wheel zooms the displayed image around the pointer; right/middle-drag pans it without changing camera calibration. **Fit camera image** restores the fitted view. Left-drag switches to orbit navigation.
- `Space`: restore the initial orbit view and switch to the viewport camera
- `\`: enter walk navigation from the current viewport or selected scene camera
- In walk mode: mouse look, `W/A/S/D` move, `Q/E` down/up, `Shift` faster, `Alt` slower, wheel adjusts speed
- `Enter`, left-click, or `\`: finish walking and keep the view; `Escape` or right-click: cancel and restore the previous view
- Walk navigation has no gravity or collision detection; losing window focus cancels it
- Debug-view shortcuts (`1`–`0`, `+`/`-`) and point-cloud shortcuts (`R`, `F`, `L`, `N`, `M`, arrows) remain available while walking
- Walk movement uses the latest input state after each render; released keys are not replayed across later frames. Short exit and viewer-shortcut taps still register.
- `Refresh preview`: request a fresh viewport preview
- `Save screenshot (PNG)`: save the currently displayed render with transparent background
- `Auto preview`: refresh after camera/control changes
- `Median depth settings`: adjust the accumulated-opacity threshold from 0.001 to 0.999 (default 0.5). `Retain last accepted depth` keeps the last surface hit when the threshold is never reached (default on). Empty rays still produce zero depth; the existing traversal-limit fallback is unchanged. Both controls request a fresh render, including depth-derived diagnostics.
- `Add surfel`: create a camera-facing surfel at the current viewport focus and select it for editing
- **File → Save editable snapshot** saves the private working scene without overwriting the source PLY
- Drop a training run, its `points` folder, or a parent such as `output/pbdr_blade`: load its latest checkpoint and anchor navigation to that run. A parent selects the run with the most recently updated `metrics.csv`.
- `R`: load the selected run's latest scene (live while its owned optimizer is active; newest saved PLY otherwise). With no selected run, open the newest loadable reconstruction beneath the configured output root or anchored folder, including its plots.
- `N` / `M`: jump to the saved checkpoint nearest the previous/next 1,000-iteration boundary
- `F`: load the first `iter_*_points.ply` in the active optimization `points` folder
- `L`: load the last `iter_*_points.ply` in the active optimization `points` folder
- Left/right arrows: step through snapshots in the active optimization run
- Up/down arrows with a scene camera selected: next/previous scene camera, wrapping at either end; hold to keep cycling once per frame
- Up/down arrows with the viewport camera selected: next/previous optimization snapshot

Snapshot navigation refreshes only the active run's `points` folder, including
new snapshots written during training. After dropping a folder, `R`, `F`, `L`,
`N`/`M`, and snapshot arrows stay in the selected run even if another run starts.
Up/down retain their scene-camera behavior when a scene camera is selected.
An empty dropped folder stays anchored: press `R` or `F` after checkpoints appear.
In **Scene files**, **Reset to OptimizationOutput** clears the anchor and performs
the original global latest-run search. Loading another point-cloud PLY also clears
the folder anchor. Without an anchor, `R` opens the newest usable full
reconstruction from the configured output root. **Load latest run PLY** retains
the standalone PLY search.

These file-loading shortcuts inspect saved files in the standalone editable
workspace. For a viewer-launched run, **N/M** and the Checkpoints panel browse that run
independently of training, and **R** always returns to its live read-only view.

### Training workflow checks

```bash
python3 -m unittest discover -s test -p test_viewer_training_session.py -v
clang++-22 -std=c++20 -O2 -pthread -I. test/test_viewer_training_process.cpp \
  -o /tmp/pale_training_process_test
/tmp/pale_training_process_test
cmake --build build-realtime --target PaleViewerTrainingIntegration -j4
./build-realtime/PaleViewerTrainingIntegration "$PWD"
cmake --build build-realtime --target PaleViewerReconstructionLibrary -j4
./build-realtime/PaleViewerReconstructionLibrary
# Optional: render production UI components with Mesa EGL, no display required.
cmake --build build-realtime --target PaleViewerWorkspacePreview -j4
./build-realtime/PaleViewerWorkspacePreview "$PWD" /tmp/workspace.png selection 1440 900
```

The integration check launches a lightweight Python worker using the production
session bridge, loads its PLY through the native scene loader, exercises tuning
and pause/stop acknowledgments, and draws both ImGui panels without a display or
GPU. It does not validate optimization convergence or GPU throughput.

The opt-in `test/test_viewer_training_optimizer.py` runs both native and host
optimizer paths on a 4×4, two-surfel scene. It verifies a paused-start handshake,
tuning a participating loss term, resuming, and publishing the final checkpoint
and live scene. Run it with `PALE_RUN_TRAINING_SMOKE=1` in a configured training
environment with the built `pale` module on `PYTHONPATH`.

### Background compositing

The Rendered view composites the saved raw linear primal RGB as `C + (1-A)*B`,
then applies display encoding. `C` already contains surfel attenuation; it is
never multiplied by coverage again. Both viewport and scene cameras display an
opaque composited texture, preventing a second blend in ImGui. Background/grid
colors are display-referred UI colors, decoded for composition and kept stable
when changing scene exposure.

**Force opacity = 1** hides the background behind contributing rays without
changing primal RGB or geometry. Over black with the grid off, toggling it has
no effect on rendered RGB. The primal is captured before any backward pass
reuses its framebuffer. Raw-linear preview and training loss are unchanged.

### Grid display cost

The grid is a CPU background pass after the renderer, so its cost affects display
latency and is excluded from **Last render** timing. Toggling it only recomposites
the displayed image; it does not launch a renderer pass. Background pixels are
cached by displayed camera, resolution, scene extent, grid visibility and background
color. Projection uses the displayed renderer camera's full pose and pinhole
intrinsics, including its focal lengths and principal point. Repeated renders
at the same camera reuse the cache. A changed camera
rebuilds independent rows on up to four CPU workers, using unnormalized plane
intersections and precomputed camera constants. Colored axes use a three-pixel
width with a one-pixel antialiasing footprint derived from the projected ground
plane, so they remain readable at grazing angles. Grid lines fade smoothly with
camera depth; unresolved fine lines fade before producing horizon moire. The fade
scales with the orbit distance and scene bounds. Screenshot output excludes the grid.

A release-mode CPU microbenchmark at 1280×720 measured about 9 ms for rebuilding
the antialiased, distance-faded grid, versus 4.6 ms for the previous thinner grid.
Cached frames reuse the background rather than rebuilding it (previously measured
at about 0.06 ms for reuse/copy). These are isolated background timings, not total
viewer frame times; they vary with hardware and load.

### Transparent screenshots

Click **Save screenshot (PNG)** in **Display**. Images are saved to
`output/screenshots/viewport_<timestamp>_<sequence>.png` under the repository
root; the full saved path appears below the button. Each capture gets a new name.

The PNG uses the last displayed render's native resolution, RGB/exposure, and
rendered surface alpha, including partial opacity at footprint edges. Rendered
screenshots convert premultiplied linear RGB to straight alpha before encoding,
so coverage is not baked into RGB twice. As with any 8-bit transparent PNG,
straight HDR colors may clip; use raw EXRs for exact radiance. A viewer that
blends PNGs in encoded rather than linear space can still display them differently. It excludes
the gray viewport background, grid, gizmos, and UI. **Force opacity = 1** affects
the viewport display only and is ignored for screenshots. Diagnostic views save
their selected colors with the rendered surface alpha. If auto-render is off,
click **Render** first to capture camera or scene edits that have not yet rendered.

### Responsive navigation

While the viewport camera moves, preview rendering temporarily uses two maximum
splat events and shared local-layer direct lighting. After 180 ms without camera
motion, a full-quality preview restores the configured settings. The footer keeps
navigation guidance stable throughout a gesture. Hover it for preview timing and
rendering details; the window title and header stay fixed. This affects viewport
rendering only; it does not change
training settings or the values shown in the inspector. **View → Interaction
preview** adds independent fast-preview switches for emitter edits (on by
default) and surfel edits (off by default).

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
**3** Depth distortion, **4** Mean depth, **5** Visible normal, **6** Mean depth normal,
**7** Median depth normal, **8** Surface overlap, **9** Position primitive
score, and **0** Intra-slab mean ray depth. Other diagnostics remain available
in **Display** and the **+/-** cycle.
Shortcuts remain active while rendering or orbiting. Rapid selections are
processed in order and the latest choice takes effect at the renderer handoff;
the viewport mode dropdown immediately reflects that choice. Numeric text fields
retain keyboard input while being edited.

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

The position threshold and transport cutoff defaults are generated from
`python/config.py` at build time. CMake regenerates them when the config changes.
Position previews start at the configured threshold when loading a snapshot;
**Use saved** selects its recorded threshold and **Use config default** restores
the config value.

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

### Depth-derived normal previews

Press **6** or choose **6 Mean depth normal** from the view selector to see normals
reconstructed from the mean depth map. Press **7** or choose **7 Median depth normal**
for normals reconstructed from the median depth map. Both use the renderer's
depth-to-normal calculation and the same normal colors. Switching views recomputes the selected map.

### Depth-distortion previews

Under **Display → View options → Depth distortion**, choose
**World distance (absolute)**, **World distance (squared)**, or **Normalized depth (squared)**.
World modes use camera-forward depth in scene units. Match the training run's
`depth_distort_world_space` and `depth_distort_squared` settings.
Enable squared world-space training with `--depth-distort-world-space --depth-distort-squared`;
absolute remains the default. Retune `--depth-distort-weight` when switching. Both the loss image and position-gradient
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

Surface overlap is a viewer diagnostic. Its distance tolerance and normal-distance
controls do not affect training or densification selection.

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

### Opacity-based adjoint scattering in training

In `python/config.py`, `RendererSettingsConfig.adjoint_opacity_sampling` enables
opacity-based sampling (default `True`). Set it to `False` to restore the fixed
`adjoint_q_null` / `adjoint_q_reflect` proposal. Native callers, including the
viewer, retain fixed sampling unless they explicitly enable the option.

The enabled sampler makes one decision for the complete local slab, using its
combined opacity, including each member's radial footprint. With
`adjoint_opacity_uniform_mix = m`, reflection has probability
`(1 - m) * slab_opacity + m / 2`; transmission uses the complement. The option
`m = 0.2` keeps both probabilities at least 0.1, retaining opacity derivatives at
fully transparent or opaque surfaces. Training currently uses `m = 0.0` for pure
opacity sampling, which can lose derivative support at those endpoints. `m = 1.0` reproduces fixed
0.5/0.5 sampling, including the same seeded choices.

Probabilities are detached sampling choices; the estimator retains their inverse
weights. The primary-slab cache and sample coalescing use the same choices as
ordinary traversal. Forward rendering and regularizer evaluation are unchanged.
The enabled mode ignores fixed `adjoint_q_null` / `adjoint_q_reflect` values.
Python callers can also switch modes on an existing renderer through
`set_traversal_sampling_options({"adjoint_opacity_sampling": True,
"adjoint_opacity_uniform_mix": 0.2})`.

## Transport contribution inspection

Press **R** to load an optimization snapshot, then select **Contribution scores**
in **Display**. The first analysis
renders every saved scene camera at its native resolution using the native
camera/shadow activity counters. Results are cached while orbiting. In **Display → View options**, choose a camera, or **All training cameras (one cycle)**, and choose
combined, camera-only, or shadow-only support. Held-out cameras are available
individually but excluded from the training aggregate.

**Passing score** defaults to 10. Pink surfels have scores strictly below the
threshold; green surfels pass. Changing the threshold or score source recolors
the cached data immediately. **X-ray affected centers** draws pink markers even
behind opaque blockers; click one to inspect its per-camera camera/shadow scores.
Markers for the selected editable surfel have a white ring. At most 20,000
on-screen markers are drawn; **Select next affected surfel** also reaches entries
beyond this display limit. Geometry from other point clouds is scored and drawn,
but picking follows the viewer's existing first editable point-cloud convention.

By default analysis reads traversal, lighting-sharing and sampling settings from
`run_config.json` beside the loaded snapshot (or a parent directory). Disable
**Use saved run settings** to analyze current viewer settings instead. **Recompute
contributions** refreshes the analysis; geometry and renderer-setting changes
invalidate it automatically. This is a preview: it does not change opacity,
training configuration, or delete points.

Camera support sums incoming transmission times local-layer member opacity
weight. Shadow support sums receiver weight times incoming shadow transmission
times blocker opacity. The combined training-camera aggregate uses the same score
rule as inactive-transport pruning. It recomputes the current snapshot, rather
than recovering historical training-cycle values: geometry changed during that
cycle, and sampled shadow rays may differ. These scores measure direct camera and
shadow support, not indirect-light importance or exact loss change on deletion.

### Terminal transmittance loss preview

Select **Terminal transmittance loss** in the display-mode menu (also available
through `+`/`-` cycling). This traces only camera rays using surfel footprints and
slab compositing; it does not change opacity or the training objective.

The unit-weight loss is `max(0, target_attenuation - (1 - T_exit))^2`, with a
90% target by default. **Display → View options → Target attenuation** adjusts it
from 80% to 100%. Rays with no attenuation are ignored, including intersections
with zero-opacity surfels. Empty rays, opaque mesh hits, and rays meeting the
target are black. Other escaping rays use the scalar colormap on a fixed range
from zero to the target squared: larger loss means more remaining transmission.
Magenta marks rays that exhaust the traversal budget before escape can be
established, rather than reporting a misleading loss. This diagnostic uses the
ordinary surfel slab representation, including when inspecting experimental
rendering modes.

### Raw linear RGB preview

Select **Raw linear RGB** under **Display** to inspect primal
RGB before exposure, sRGB/gamma encoding or alpha compositing. The framebuffer
is captured before adjoint profiling can overwrite it. Values are mapped directly
to an opaque 8-bit preview: 0.2 becomes 51/255, with no normalization. Only the
preview clips values outside [0,1]; non-finite RGB is magenta. Appearance controls
are disabled in this mode. PNG screenshots save this same opaque, clipped preview.

For full-precision loss-input data, training's **s** snapshot writes
`renders/linear_raw_final_<camera>.exr`. Enable `save_snapshot_linear_raw` in the
training configuration (or `--save-snapshot-linear-raw`) for scheduled EXRs too.
