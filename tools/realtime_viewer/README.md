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
- Left/right or down/up arrows: step through optimization point snapshots

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
