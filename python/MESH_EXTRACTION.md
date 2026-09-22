# Object and scene mesh extraction

The top-level `render.py` in 2D-GS-Viser-Viewer and PALE's `python/extract_mesh.py`
use the same automatic bounded-TSDF policy. There are no `--object`/`--scene`
switches: the policy uses rendered geometry, not dataset names or camera radius.
The upstream `2d_gaussian_splatting/render.py` is unchanged. The viewer project's
`render_all.py` now calls the new top-level script.

## Automatic settings

- Maximum depth: 1.1 times the largest per-view 99th percentile of positive,
  finite rendered depth. Empty views are ignored; no valid depths is an error.
- Voxel edge: the larger of the longest robust world-space surface extent / 512
  and the median projected pixel footprint, `depth / sqrt(fx * fy)`.
  World extents use the 0.5th/99.5th percentiles of deterministically sampled,
  back-projected depths inside the chosen depth range. These bounds determine
  resolution; they do not crop the volume.
- TSDF truncation band: 4 voxel edges.
- All connected components are retained by default. Raw meshes are always saved;
  post meshes remove duplicated/degenerate elements but do not automatically
  keep only the largest 50 parts or remove non-manifold edges.

These are scale-aware defaults, not a guarantee of correct geometry. Depth
outliers occupying a substantial image region can still corrupt the estimates.
Sparse coverage, transparency, and inconsistent surfaces still affect fusion.
Inspect printed depth retention and the saved metadata before adjusting quality.
Distances are in the input scene's coordinate units, not necessarily metres.

## Usage

From 2D-GS-Viser-Viewer, in its normal 2DGS environment:

```bash
python render.py -m output/2dgs_restaurant --iteration 30000 --skip_train --skip_test
```

The default mesh outputs are `train/ours_30000/fuse_auto.ply` and
`fuse_post_auto.ply`; `extraction_auto.json` records all resolved settings,
per-view depth retention, and triangle counts. The `_auto` suffix keeps the
original extractor's files intact. Existing underscore flags remain accepted,
and equivalent hyphenated flags are available. `--output-dir` can isolate a run.
Use `--depth_ratio 0` for mean depth, or `1` (default) for median depth.

From PALE's `python` directory, in its normal renderer environment:

```bash
python extract_mesh.py --run-dir OptimizationOutput/MY_RUN --no-export-gltf
# Without --run-dir, uses the run with the newest metrics.csv under OptimizationOutput:
python extract_mesh.py --no-export-gltf
# A checkpoint can be used even when training has not written points_final.ply:
python extract_mesh.py --run-dir OptimizationOutput/MY_RUN \
  --ply OptimizationOutput/MY_RUN/points/iter_02000_points.ply \
  --mesh-output-subdir mesh_auto --no-export-gltf
```

PALE retains `fuse.ply` / `fuse_post.ply` naming and adds `extraction.json` in the
mesh directory. Its depth flag is `--depth-key median_depth` (default) or
`--depth-key mean_depth`. GLB export is still enabled unless disabled explicitly.

PALE allocates backward buffers on first use. Mesh extraction only calls
`render_forward()`, so it does not allocate adjoint ray/event scratch, gradient
sets, gradient debug images, or per-camera adjoint images. No extraction flag is
required. Forward images and scene geometry still occupy GPU memory.

For allocation diagnostics, call `renderer.get_backward_allocation_stats()`.
It reports gradient bytes by purpose, sensor-adjoint and debug-image bytes, ray
queue capacity, and whether adjoint scratch exists; it is not a total VRAM meter.
Training retains buffers after their first use. A BVH rebuild invalidates gradient
sets, which are recreated when next needed. Destroying a renderer releases its
owned scene, sensor, and scratch allocations.

## Identical numeric settings for comparisons

The same automatic policy can choose different values when two models predict
different geometry. To use *exactly* the same depth cutoff, voxel size, and TSDF
band, pass either extractor the JSON produced by the other:

```bash
python extract_mesh.py --ply /path/to/points_final.ply \
  --tsdf-settings /path/to/2dgs/train/ours_30000/extraction_auto.json \
  --mesh-output-subdir mesh_matched --no-export-gltf
```

The coordinate systems must use the same units. `--tsdf-settings` reuses the
three numeric TSDF values; it does not change the depth estimator, camera poses,
or component filtering. Explicit numeric flags override values from the JSON.

The previously tested restaurant settings remain available as explicit overrides:

```text
--depth-trunc 15 --voxel-size 0.02 --sdf-trunc 0.08
```

Other controls: `--mesh-res`, `--voxel-pixels`, `--depth-percentile`, and
`--depth-margin`. The pixel-footprint floor limits automatic voxel refinement;
use an explicit `--voxel-size` to override it. `--num-cluster 0` imposes no
largest-component limit; positive N keeps the largest N components.
`--min-cluster-triangles` defaults to **50**, removing components with fewer than
50 triangles, as in the current 2DGS/PGSR cleanup. Set it to `0` to disable this
size filter. Raw `fuse*.ply` meshes remain available alongside the cleaned
`fuse_post*.ply` meshes. The old radius-based auto cutoff is not used.

PALE training defaults and the object/scene config templates now use 512 for
mesh resolution and 0 for the largest-component limit. Small-component filtering
also applies to extraction launched by training. Explicit older command-line or
experiment settings still take precedence; already-running training retains its
in-memory settings.

## Implementation and checks

Each project carries the same standalone NumPy-based `tsdf_settings.py`, so
neither imports the other project by an absolute path. Keep the two copies
identical when changing policy. The matching `test_tsdf_settings.py` suites test
scale equivariance, outlier resistance, far-view retention, single/parallel
cameras, invalid depth handling, exact settings reuse, and component retention.
PALE also tests a real Open3D planar integration without initializing the renderer.
