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

Curvature splitting starts disabled (`-1`); enter a positive threshold to preview
split candidates. Disabled splitting still displays curvature scores without
magenta candidate highlighting. SSIM debug defaults are weight `0`, window size
`5`, and sigma `0.75`. These defaults are copied into the viewer; it does not load
`config.py` at runtime.

### Depth-distortion previews

The viewer defaults to **World distance**, matching training with
`depth_distort_world_space=True`. The loss and position-gradient previews both
use linear camera-forward depth, so equal depth separations retain their
contribution when moved farther from the camera (for equal compositing weights).
Choose **Normalized depth (legacy)** in the **Depth distortion** selector to
inspect runs trained with `depth_distort_world_space=False`.

The loss image shows the raw per-pixel distortion; the position-gradient image
uses the mean image loss with unit regularizer weight. Colors rescale separately
for each frame, so equal colors across frames do not imply equal loss values.

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
