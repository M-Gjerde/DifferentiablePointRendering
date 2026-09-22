"""Export reconstructed reflectance and point lights as a self-contained glTF 2.0 GLB.

The texture samples optimized surfel albedos at UV texel positions, never shaded
mesh colors. Transferring a surfel field to a triangle surface is an approximation;
the bake reports texels for which no nearby surfel footprint covers the surface.
"""
from __future__ import annotations

import json
import math
import os
import struct
import threading
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm


# Blender's standard glTF importer uses this RGB rendering convention. This is
# not a spectral watts-to-lumens conversion for arbitrary real-world emitters.
BLENDER_WATTS_TO_LUMENS = 683.0
Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)


def resolve_uv_parallelism(triangle_count: int, partitions: int = 0, threads: int = 0) -> tuple[int, int]:
    """Keep unwrap tasks bounded in size; cap automatic CPU concurrency at eight.

    More partitions introduce more UV islands, but do not simplify the geometry.
    Explicit partitions=1 restores Open3D's original single-partition behavior.
    """
    if partitions < 0 or threads < 0:
        raise ValueError("UV partitions and threads must be non-negative (0 means automatic)")
    if partitions == 0:
        partitions = min(32, max(1, math.ceil(triangle_count / 100_000)))
    if threads == 0:
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        threads = min(8, available)
    return partitions, min(threads, partitions)


def compute_uvatlas_with_elapsed_progress(
        tensor_mesh,
        *,
        texture_size: int,
        partitions: int,
        threads: int,
) -> tuple[float, int, int]:
    """Run Open3D's opaque unwrap call with an elapsed-time indicator."""
    result: dict[str, tuple[float, int, int]] = {}
    error: dict[str, BaseException] = {}
    completed = threading.Event()

    def unwrap() -> None:
        try:
            result["value"] = tensor_mesh.compute_uvatlas(
                size=texture_size,
                gutter=4.0,
                parallel_partitions=partitions,
                nthreads=threads,
            )
        except BaseException as exception:
            error["value"] = exception
        finally:
            completed.set()

    worker = threading.Thread(target=unwrap, name="uv-atlas", daemon=False)
    worker.start()
    with tqdm(
            desc="Unwrapping UV atlas",
            unit="s",
            dynamic_ncols=True,
            bar_format="{desc}: {elapsed} elapsed",
    ) as progress:
        while not completed.wait(timeout=1.0):
            progress.update(1)
    worker.join()

    if "value" in error:
        raise error["value"]
    return result["value"]


def linear_to_srgb8(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear,
                       1.055 * np.power(linear, 1.0 / 2.4) - 0.055)
    return np.rint(encoded * 255.0).astype(np.uint8)


def validate_parameters(parameters: dict) -> dict[str, np.ndarray]:
    result = {}
    count = len(parameters["position"])
    for name, width in (("position", 3), ("rotation", 4), ("scale", 2),
                        ("albedo", 3), ("opacity", 1), ("beta", 1), ("power", 1)):
        value = np.asarray(parameters[name], dtype=np.float64)
        if width == 1:
            value = value.reshape(-1)
        expected = (count,) if width == 1 else (count, width)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"Invalid surfel {name}: expected finite {expected} values")
        result[name] = value
    if np.any(result["power"] < 0):
        raise ValueError("Light power must be non-negative")
    if np.any((result["albedo"] < 0) | (result["albedo"] > 1)):
        raise ValueError("Reconstructed albedos must be linear RGB in [0, 1]")
    if np.any((result["opacity"] < 0) | (result["opacity"] > 1)):
        raise ValueError("Surfel opacity must be in [0, 1]")
    return result


def load_surfel_parameters(points_path: Path) -> dict[str, np.ndarray]:
    """Read the named reflectance/light fields from PALE's ASCII checkpoint PLY."""
    properties = []
    count = None
    element = None
    with Path(points_path).open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("Expected a PLY checkpoint")
        if stream.readline().strip() != "format ascii 1.0":
            raise ValueError("Expected a PALE ASCII PLY checkpoint")
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "end_header":
                break
            if fields[0] == "element":
                element = fields[1]
                if element == "vertex":
                    count = int(fields[2])
            if fields[0] == "property" and element == "vertex":
                if fields[1] == "list":
                    raise ValueError("List-valued vertex properties are not supported")
                properties.append(fields[-1])
        else:
            raise ValueError("Missing PLY end_header")
        if count is None or count <= 0:
            raise ValueError("Checkpoint contains no vertices")
        values = np.loadtxt(stream, max_rows=count, ndmin=2)
    if values.shape != (count, len(properties)):
        raise ValueError("Checkpoint vertex data does not match its header")
    mapping = {"position": ["x", "y", "z"], "rotation": ["rot_w", "rot_x", "rot_y", "rot_z"],
               "scale": ["su", "sv"], "albedo": ["albedo_r", "albedo_g", "albedo_b"],
               "opacity": ["opacity"], "beta": ["beta"], "power": ["power"]}
    result = {name: values[:, [properties.index(field) for field in fields]]
              for name, fields in mapping.items()}
    return validate_parameters(result)


class SurfelAlbedoSampler:
    """Transfer local reflectance, with plane/normal checks to reduce color bleeding.

    Nearest-center candidates are blended by their beta footprints, opacity and
    distance to their planes. Uncovered texels use the closest compatible surfel
    and are counted explicitly. Neither lighting nor view colors are inputs.
    """

    def __init__(self, parameters: dict, neighbors: int = 32):
        p = validate_parameters(parameters)
        keep = (p["power"] == 0) & (p["opacity"] > 0)
        if not keep.any():
            raise ValueError("No non-emissive, non-transparent surfels to bake")
        self.p = {name: value[keep] for name, value in p.items()}
        if np.any(self.p["scale"] <= 0):
            raise ValueError("Surface surfel scales must be positive")
        q = self.p["rotation"]
        lengths = np.linalg.norm(q, axis=1, keepdims=True)
        if np.any(lengths < 1e-12):
            raise ValueError("Surface surfel rotations must be nonzero quaternions")
        w, x, y, z = (q / lengths).T
        self.u = np.column_stack((1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y)))
        self.v = np.column_stack((2 * (x*y - w*z), 1 - 2 * (x*x + z*z), 2 * (y*z + w*x)))
        self.n = np.cross(self.u, self.v)
        self.tree = cKDTree(self.p["position"])
        self.neighbors = min(max(1, neighbors), len(self.p["position"]))
        self.fallback_count = 0
        self.sample_count = 0

    def sample(self, positions: np.ndarray, normals: np.ndarray, *, progress: bool = False) -> np.ndarray:
        output = np.empty((len(positions), 3), dtype=np.float32)
        started = last_report = time.perf_counter()
        for start in range(0, len(positions), 16384):
            end = min(start + 16384, len(positions))
            distance, ids = self.tree.query(positions[start:end], k=self.neighbors, workers=1)
            distance = distance.reshape(end - start, self.neighbors)
            ids = ids.reshape(end - start, self.neighbors)
            delta = positions[start:end, None, :] - self.p["position"][ids]
            u = np.sum(delta * self.u[ids], axis=-1) / self.p["scale"][ids, 0]
            v = np.sum(delta * self.v[ids], axis=-1) / self.p["scale"][ids, 1]
            plane_distance = np.sum(delta * self.n[ids], axis=-1)
            alignment = np.abs(np.sum(normals[start:end, None, :] * self.n[ids], axis=-1))
            # A narrow normal-distance band discourages transfer across thin walls.
            band = np.maximum(self.p["scale"][ids].min(axis=-1) * 0.1, 1e-8)
            base = np.maximum(1.0 - u*u - v*v, 0.0)
            exponent = 4.0 * np.exp(np.clip(self.p["beta"][ids], -80, 12))
            weights = (np.power(base, exponent) * self.p["opacity"][ids]
                       * alignment**4 * np.exp(-0.5 * (plane_distance / band)**2))
            weights[(base <= 0) | (alignment < 0.5)] = 0
            totals = weights.sum(axis=1)
            covered = totals > 1e-12
            colors = np.sum(weights[..., None] * self.p["albedo"][ids], axis=1)
            colors[covered] /= totals[covered, None]
            # Penalize incompatible normals without requiring consistently oriented
            # surfels: PALE can shade either side of a surfel.
            score = distance / np.maximum(alignment, 0.05)**2
            nearest = ids[np.arange(len(ids)), np.argmin(score, axis=1)]
            colors[~covered] = self.p["albedo"][nearest[~covered]]
            output[start:end] = colors
            self.fallback_count += int((~covered).sum())
            self.sample_count += end - start
            now = time.perf_counter()
            if progress and (now - last_report >= 5 or end == len(positions)):
                print(f"Albedo sampling: {end:,}/{len(positions):,} texels "
                      f"({end / len(positions):.0%}), {now - started:.1f}s", flush=True)
                last_report = now
        return output


def split_disconnected_vertex_fans(mesh) -> int:
    """Separate pinched TSDF vertices without moving or removing any triangles."""
    import open3d as o3d

    bad_vertices = np.asarray(mesh.get_non_manifold_vertices(), dtype=np.int64)
    if not len(bad_vertices):
        return 0
    triangles = np.asarray(mesh.triangles).copy()
    vertices = np.asarray(mesh.vertices)
    original_count = len(vertices)
    extra_vertices = []
    # Index incident faces once. Scanning the entire multi-million-face mesh for
    # each pinched vertex made topology preparation quadratic in practice.
    incident_faces = {int(vertex): [] for vertex in bad_vertices}
    rows, columns = np.nonzero(np.isin(triangles, bad_vertices))
    for face, vertex in zip(rows, triangles[rows, columns]):
        incident_faces[int(vertex)].append(int(face))
    for vertex in bad_vertices:
        incident = incident_faces[int(vertex)]
        edge_faces = {}
        for face in incident:
            for neighbor in triangles[face]:
                if neighbor != vertex:
                    edge_faces.setdefault(int(neighbor), []).append(int(face))
        adjacency = {int(face): set() for face in incident}
        for faces in edge_faces.values():
            if len(faces) == 2:
                a, b = faces
                adjacency[a].add(b)
                adjacency[b].add(a)
        remaining = set(adjacency)
        first = True
        while remaining:
            pending = [min(remaining)]
            component = []
            while pending:
                face = pending.pop()
                if face not in remaining:
                    continue
                remaining.remove(face)
                component.append(face)
                pending.extend(adjacency[face] & remaining)
            if first:
                first = False
                continue
            new_vertex = original_count + len(extra_vertices)
            extra_vertices.append(vertices[vertex].copy())
            for face in component:
                triangles[face, triangles[face] == vertex] = new_vertex
    if extra_vertices:
        mesh.vertices = o3d.utility.Vector3dVector(np.vstack((vertices, extra_vertices)))
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return len(extra_vertices)


def repair_non_manifold_edges(mesh) -> tuple[int, int]:
    """Remove the smallest set of incident triangles needed for edge manifoldness."""
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    non_manifold_edge_count = len(mesh.get_non_manifold_edges(allow_boundary_edges=True))
    if non_manifold_edge_count == 0:
        return 0, 0

    triangle_count_before = len(mesh.triangles)
    mesh.remove_non_manifold_edges()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return non_manifold_edge_count, triangle_count_before - len(mesh.triangles)


def bake_albedo_atlas(mesh, parameters: dict, texture_size: int = 2048,
                      *, uv_partitions: int = 0, uv_threads: int = 0):
    """Return a UV-mapped tensor mesh, sRGB texture, and transfer diagnostics."""
    import open3d as o3d

    if texture_size < 32 or texture_size > 16384:
        raise ValueError("Texture size must be between 32 and 16384 pixels")
    if len(mesh.triangles) == 0 or not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError("Cannot export empty or non-finite mesh geometry")
    partitions, threads = resolve_uv_parallelism(len(mesh.triangles), uv_partitions, uv_threads)
    preparation_started = time.perf_counter()
    print(f"Preparing {len(mesh.triangles):,} triangles for UV export", flush=True)
    # Work on a copy: preserve the mesh used for geometry metrics / PLY output.
    mesh = o3d.geometry.TriangleMesh(mesh)
    non_manifold_edges, removed_triangles = repair_non_manifold_edges(mesh)
    if len(mesh.triangles) == 0:
        raise ValueError("No non-degenerate triangles to export")
    if non_manifold_edges:
        print(
            f"UV repair: removed {removed_triangles:,} triangles across "
            f"{non_manifold_edges:,} non-manifold edges.",
            flush=True,
        )
    if not mesh.is_edge_manifold(allow_boundary_edges=True):
        raise ValueError("UV unwrapping requires manifold edges; repair the extracted mesh first")
    split_vertices = split_disconnected_vertex_fans(mesh)
    if not mesh.is_vertex_manifold():
        raise ValueError("UV unwrapping requires manifold vertices; repair the extracted mesh first")
    mesh.compute_vertex_normals()
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    preparation_seconds = time.perf_counter() - preparation_started
    print(f"Mesh preparation finished in {preparation_seconds:.1f}s "
          f"({split_vertices:,} pinched vertices split)", flush=True)
    print(f"Unwrapping {len(mesh.triangles):,} triangles for a {texture_size}px albedo atlas "
          f"on CPU: {partitions} requested partitions, {threads} threads. "
          "Showing elapsed time; Open3D does not report completion progress.", flush=True)
    unwrap_started = time.perf_counter()
    stretch, charts, actual_partitions = compute_uvatlas_with_elapsed_progress(
        tensor_mesh,
        texture_size=texture_size,
        partitions=partitions,
        threads=threads,
    )
    unwrap_seconds = time.perf_counter() - unwrap_started
    print(f"UV unwrap finished in {unwrap_seconds:.1f}s: {charts:,} charts, "
          f"{actual_partitions} partitions. Baking texel positions and normals...", flush=True)
    raster_started = time.perf_counter()
    # Bake surface positions, not vertex colors. This allows texture detail finer
    # than the mesh tessellation. NaN marks unused atlas pixels, including origin.
    maps = tensor_mesh.bake_vertex_attr_textures(
        texture_size, {"positions", "normals"}, margin=2.0,
        fill=float("nan"), update_material=False,
    )
    positions = maps["positions"].numpy()
    normals = maps["normals"].numpy()
    valid = np.isfinite(positions).all(axis=-1) & np.isfinite(normals).all(axis=-1)
    if not valid.any():
        raise ValueError("UV atlas contains no covered texels")
    raster_seconds = time.perf_counter() - raster_started
    print(f"Texel geometry baked in {raster_seconds:.1f}s. "
          f"Sampling {int(valid.sum()):,} texels from optimized surfel albedos...", flush=True)
    sampled_normals = normals[valid]
    sampled_normals /= np.maximum(np.linalg.norm(sampled_normals, axis=-1, keepdims=True), 1e-12)
    sampler = SurfelAlbedoSampler(parameters)
    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    sampling_started = time.perf_counter()
    texture[valid] = linear_to_srgb8(sampler.sample(positions[valid], sampled_normals, progress=True))
    diagnostics = {"texture_size": texture_size, "uv_charts": int(charts),
                   "uv_partitions": int(actual_partitions), "uv_threads": threads,
                   "preparation_seconds": preparation_seconds, "unwrap_seconds": unwrap_seconds,
                   "texel_geometry_seconds": raster_seconds,
                   "albedo_sampling_seconds": time.perf_counter() - sampling_started,
                   "split_pinched_vertices": split_vertices,
                   "uv_stretch": float(stretch), "sampled_texels": sampler.sample_count,
                   "fallback_texels": sampler.fallback_count,
                   "albedo_source": "optimized_surfel_albedo",
                   "transfer": "local_beta_footprint_with_plane_and_normal_weights"}
    if sampler.fallback_count:
        warnings.warn(f"Albedo transfer used nearest-surface fallback for "
                      f"{sampler.fallback_count / sampler.sample_count:.1%} of atlas samples")
    return tensor_mesh, texture, diagnostics


class GlbBuilder:
    def __init__(self):
        self.binary = bytearray()
        self.document = {"asset": {"version": "2.0", "generator": "PALE reconstruction exporter"},
                         "scene": 0, "scenes": [{"nodes": []}], "nodes": [],
                         "bufferViews": [], "accessors": []}

    def buffer_view(self, data: bytes, target: int | None = None) -> int:
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        view = {"buffer": 0, "byteOffset": len(self.binary), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.binary.extend(data)
        self.document["bufferViews"].append(view)
        return len(self.document["bufferViews"]) - 1

    def accessor(self, values: np.ndarray, kind: str, indices: bool = False) -> int:
        values = np.asarray(values, dtype="<u4" if indices else "<f4")
        entry = {"bufferView": self.buffer_view(values.tobytes(), 34963 if indices else 34962),
                 "componentType": 5125 if indices else 5126, "count": len(values), "type": kind}
        if kind == "VEC3":
            entry.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
        self.document["accessors"].append(entry)
        return len(self.document["accessors"]) - 1

    def node(self, value: dict):
        self.document["scenes"][0]["nodes"].append(len(self.document["nodes"]))
        self.document["nodes"].append(value)

    def write(self, path: Path):
        self.document["buffers"] = [{"byteLength": len(self.binary)}]
        encoded = json.dumps(self.document, allow_nan=False, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 4)
        binary = bytes(self.binary) + b"\0" * (-len(self.binary) % 4)
        total_length = 12 + 8 + len(encoded) + 8 + len(binary)
        with path.open("wb") as stream:
            stream.write(struct.pack("<4sII", b"glTF", 2, total_length))
            stream.write(struct.pack("<I4s", len(encoded), b"JSON"))
            stream.write(encoded)
            stream.write(struct.pack("<I4s", len(binary), b"BIN\0"))
            stream.write(binary)


def export_reconstruction_glb(mesh, parameters: dict, output_path: Path,
                              *, cameras=(), texture_size: int = 2048,
                              meters_per_unit: float = 1.0,
                              uv_partitions: int = 0, uv_threads: int = 0) -> Path:
    """Export a standard base-color UV texture, Lambertian material and point lights.

    Input coordinates use PALE's Z-up convention. Cameras use OpenCV extrinsics.
    Power is total RGB radiant flux; light conversion matches Blender's SPEC mode.
    """
    import io

    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("meters_per_unit must be positive and finite")
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".glb":
        raise ValueError("Output must have a .glb extension")
    parameters = validate_parameters(parameters)
    tensor_mesh, texture, diagnostics = bake_albedo_atlas(
        mesh, parameters, texture_size, uv_partitions=uv_partitions, uv_threads=uv_threads,
    )
    print("Packing textured mesh, lights and cameras into GLB...", flush=True)
    positions = tensor_mesh.vertex.positions.numpy()
    normals = tensor_mesh.vertex.normals.numpy()
    triangles = tensor_mesh.triangle.indices.numpy()
    uvs = tensor_mesh.triangle.texture_uvs.numpy().reshape(-1, 2).copy()
    # Open3D's atlas uses bottom-left UVs; glTF uses top-left texture coordinates.
    uvs[:, 1] = 1.0 - uvs[:, 1]
    # Split only at UV seams, preserving smooth normals across those seams.
    corners = np.column_stack((triangles.reshape(-1), uvs))
    unique, inverse = np.unique(corners, axis=0, return_inverse=True)
    vertex_ids = unique[:, 0].astype(np.int64)
    positions = (positions[vertex_ids] @ Z_UP_TO_Y_UP.T) * meters_per_unit
    normals = normals[vertex_ids] @ Z_UP_TO_Y_UP.T
    builder = GlbBuilder()
    attributes = {"POSITION": builder.accessor(positions, "VEC3"),
                  "NORMAL": builder.accessor(normals, "VEC3"),
                  "TEXCOORD_0": builder.accessor(unique[:, 1:], "VEC2")}
    indices = builder.accessor(inverse, "SCALAR", indices=True)
    png = io.BytesIO()
    Image.fromarray(texture).save(png, format="PNG")
    doc = builder.document
    doc["images"] = [{"name": "Reconstructed albedo", "mimeType": "image/png",
                       "bufferView": builder.buffer_view(png.getvalue())}]
    # Base-color PNG is sRGB; factors and light colors remain linear RGB.
    doc["samplers"] = [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}]
    doc["textures"] = [{"source": 0, "sampler": 0}]
    doc["materials"] = [{"name": "Reconstructed Lambertian albedo", "doubleSided": True,
                         "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                             "baseColorTexture": {"index": 0, "texCoord": 0},
                             "metallicFactor": 0, "roughnessFactor": 1},
                         "extensions": {"KHR_materials_specular": {"specularFactor": 0}}}]
    doc["extensionsUsed"] = ["KHR_materials_specular"]
    doc["extensionsRequired"] = ["KHR_materials_specular"]
    doc["meshes"] = [{"name": "Reconstruction", "primitives": [
        {"attributes": attributes, "indices": indices, "material": 0, "mode": 4}]}]
    builder.node({"name": "Reconstruction", "mesh": 0})

    lights = []
    for index in np.flatnonzero(parameters["power"] > 0):
        light_id = len(lights)
        power = float(parameters["power"][index])
        lights.append({"name": f"Point light {index}", "type": "point",
                       "color": parameters["albedo"][index].tolist(),
                       # Geometry rescaling needs square-scaled power to preserve
                       # irradiance at the reconstructed surface.
                       "intensity": power * meters_per_unit**2 * BLENDER_WATTS_TO_LUMENS / (4 * math.pi),
                       "extras": {"pale_power": power, "pale_surfel_index": int(index)}})
        position = Z_UP_TO_Y_UP @ parameters["position"][index] * meters_per_unit
        builder.node({"name": lights[-1]["name"], "translation": position.tolist(),
                      "extensions": {"KHR_lights_punctual": {"light": light_id}}})
    if lights:
        doc["extensions"] = {"KHR_lights_punctual": {"lights": lights}}
        doc["extensionsUsed"].append("KHR_lights_punctual")
        doc["extensionsRequired"].append("KHR_lights_punctual")

    omitted_cameras = []
    for camera in cameras:
        # Core glTF cannot express shifted principal points or arbitrary fx/fy.
        if (not np.isclose(camera.fx, camera.fy, rtol=1e-5)
                or abs(camera.cx - camera.width / 2) > 0.5
                or abs(camera.cy - camera.height / 2) > 0.5):
            omitted_cameras.append(camera.name)
            warnings.warn(f"Skipping camera {camera.name}: asymmetric intrinsics are not supported by core glTF")
            continue
        if min(camera.width, camera.height, camera.fx, camera.fy) <= 0:
            raise ValueError(f"Invalid camera intrinsics: {camera.name}")
        transform = np.linalg.inv(camera.world_to_camera) @ np.diag([1, -1, -1, 1])
        transform[:3, :3] = Z_UP_TO_Y_UP @ transform[:3, :3]
        transform[:3, 3] = Z_UP_TO_Y_UP @ transform[:3, 3] * meters_per_unit
        camera_id = len(doc.setdefault("cameras", []))
        doc["cameras"].append({"name": camera.name, "type": "perspective",
            "perspective": {"yfov": 2 * math.atan(camera.height / (2 * camera.fy)),
                            "aspectRatio": camera.width / camera.height,
                            "znear": 0.001 * meters_per_unit},
            "extras": {"width": camera.width, "height": camera.height,
                       "fx": camera.fx, "fy": camera.fy, "cx": camera.cx, "cy": camera.cy}})
        builder.node({"name": camera.name, "camera": camera_id,
                      "matrix": transform.flatten(order="F").tolist()})
    diagnostics.update(point_lights=len(lights), cameras=len(doc.get("cameras", [])),
                       omitted_cameras=omitted_cameras, meters_per_unit=meters_per_unit,
                       watts_to_lumens=BLENDER_WATTS_TO_LUMENS)
    doc["asset"]["extras"] = diagnostics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".glb.tmp")
    try:
        builder.write(temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    # The image is embedded in the GLB; this copy makes the albedo easy to inspect.
    output_path.with_name(output_path.stem + "_albedo.png").write_bytes(png.getvalue())
    print(f"Scene saved at {output_path} ({len(lights)} point lights, "
          f"{len(doc.get('cameras', []))} cameras)", flush=True)
    return output_path
