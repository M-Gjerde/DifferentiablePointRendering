# Pale Realtime Viewer

Standalone ImGui viewer for orbiting the renderer camera without editing the existing renderer executable or top-level CMake files.

## Build

From the repository root:

```bash
cmake -S tools/realtime_viewer -B build-realtime \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang-20 \
  -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_PREFIX_PATH=/opt/AdaptiveCpp

cmake --build build-realtime -j"$(nproc)"
```

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
