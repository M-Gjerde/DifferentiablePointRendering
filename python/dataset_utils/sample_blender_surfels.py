r"""Sample opaque Principled mesh surfaces in the active camera's frustum for PALE.

    blender -b scene.blend --python-exit-code 1 -P sample_blender_surfels.py -- \
        --output /path/to/export --count 2000000

The unlit PLY stores linear base color, unsquared roughness and metalness for GGX.
Sampling uses world-space area and retains occluded surfaces inside the frame.
Disc orientations follow the mesh's split normals. The PBdR DatasetGeneration
helpers provide the camera XML and approximate finite lights as point emitters.
For scenes outside the PBdR datasets folder, pass --dataset-tools with the path
to DatasetGeneration. Run in a background Blender process.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from dataclasses import dataclass

import bpy
import numpy as np
from mathutils import Matrix


def array_of(collection, prop, width, dtype):
    out = np.empty((len(collection), width), dtype=dtype)
    collection.foreach_get(prop, out.ravel())
    return out


def unit(vectors):
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(length).all() or np.any(length < 1e-15):
        raise ValueError("Invalid non-finite or zero-length surface normal")
    return vectors / length


def orientations(normals):
    """Shortest rotation from local +Z, written in PALE's (w,x,y,z) order."""
    n = unit(normals.astype(np.float64))
    radial = np.hypot(n[:, 0], n[:, 1])
    half_angle = 0.5 * np.arctan2(radial, n[:, 2])
    q = np.zeros((len(n), 4), dtype=np.float32)
    q[:, 0] = np.cos(half_angle)
    nonpolar = radial > 1e-15
    scale = np.sin(half_angle[nonpolar]) / radial[nonpolar]
    q[nonpolar, 1] = -n[nonpolar, 1] * scale
    q[nonpolar, 2] = n[nonpolar, 0] * scale
    q[~nonpolar & (n[:, 2] < 0), 1] = 1
    return q


def frustum_planes(matrix):
    return np.stack([matrix[3] + sign * matrix[axis]
                     for axis in range(3) for sign in (1, -1)])


def clipped_patches(vertices, indices, planes):
    """Clip in world space, carrying original-triangle barycentrics at every cut."""
    finite = np.isfinite(vertices).all(axis=1)
    if not finite[indices].all():
        raise ValueError("A mesh triangle references a non-finite vertex")
    distances = vertices @ planes[:, :3].T + planes[:, 3]
    codes = np.zeros(len(vertices), dtype=np.uint8)
    for p in range(6):
        codes |= ((distances[:, p] < 0).astype(np.uint8) << p)
    corners = codes[indices]
    mask = np.bitwise_or.reduce(corners, axis=1)
    whole = np.flatnonzero(mask == 0)
    crossing = np.flatnonzero((mask != 0) & (np.bitwise_and.reduce(corners, axis=1) == 0))
    basis = np.eye(3)
    cut_faces, cut_bary = [], []
    for face in crossing:
        polygon = list(basis)
        for p in range(6):
            if not mask[face] & (1 << p):
                continue
            d = distances[indices[face], p]
            trimmed = []
            prev = polygon[-1]
            dp = prev @ d
            for cur in polygon:
                dc = cur @ d
                if (dc < 0) != (dp < 0):
                    trimmed.append(prev + (dp / (dp - dc)) * (cur - prev))
                if dc >= 0:
                    trimmed.append(cur)
                prev, dp = cur, dc
            polygon = trimmed
            if len(polygon) < 3:
                break
        for j in range(1, len(polygon) - 1):
            cut_faces.append(face)
            cut_bary.append((polygon[0], polygon[j], polygon[j + 1]))
    faces = np.concatenate((whole, np.asarray(cut_faces, dtype=int)))
    bary = np.concatenate((np.broadcast_to(basis, (len(whole), 3, 3)),
                           np.asarray(cut_bary, dtype=float).reshape(-1, 3, 3)))
    positions = bary @ vertices[indices[faces]]
    areas = 0.5 * np.linalg.norm(np.cross(positions[:, 1] - positions[:, 0],
                                          positions[:, 2] - positions[:, 0]), axis=1)
    valid = areas > 1e-19
    return faces[valid], bary[valid], areas[valid]


@dataclass
class Surface:
    name: str
    matrix: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    uvs: dict
    active_uv: str
    generated: np.ndarray
    patch_faces: np.ndarray
    patch_bary: np.ndarray
    patch_areas: np.ndarray


@dataclass
class MaterialGroup:
    surface: Surface
    material: object
    patches: np.ndarray
    area: float


def mesh_groups(obj, graph, planes):
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=graph)
    try:
        mesh.calc_loop_triangles()
        local = array_of(mesh.vertices, 'co', 3, np.float32)
        indices = array_of(mesh.loop_triangles, 'vertices', 3, np.int32)
        matrix = np.asarray(evaluated.matrix_world, dtype=float)
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        faces, bary, areas = clipped_patches(world, indices, planes)
        if not len(areas):
            return []
        used, patch_faces = np.unique(faces, return_inverse=True)
        corners = array_of(mesh.loop_triangles, 'loops', 3, np.int32)[used]
        triangles = indices[used]
        normals = array_of(mesh.corner_normals, 'vector', 3, np.float32)[corners]
        uv_maps = {layer.name: array_of(layer.uv, 'vector', 2, np.float32)[corners]
                   for layer in mesh.uv_layers}
        active_uv = next((layer.name for layer in mesh.uv_layers if layer.active_render), '')
        rest = mesh.attributes.get('rest_position')
        rest_positions = array_of(rest.data, 'vector', 3, np.float32) if rest else local
        center = np.asarray(obj.data.texspace_location)
        half_size = np.asarray(obj.data.texspace_size)
        generated = (0.5 + 0.5 * (rest_positions[triangles] - center) / np.maximum(half_size, 1e-8)).astype(np.float32)
        surface = Surface(obj.name, matrix, local[triangles], normals, uv_maps, active_uv,
                          generated, patch_faces, bary, areas)
        slots = array_of(mesh.loop_triangles, 'material_index', 1, np.int32).ravel()[faces]
        groups = []
        for slot in np.unique(slots):
            patches = np.flatnonzero(slots == slot)
            material = evaluated.material_slots[int(slot)].material if slot < len(evaluated.material_slots) else None
            if material is None:
                raise ValueError(f"{obj.name}: sampled faces have an empty material slot {slot}")
            material = material.original
            groups.append(MaterialGroup(surface, material, patches, float(areas[patches].sum())))
        return groups
    finally:
        evaluated.to_mesh_clear()


def prepare_geometry(objects):
    for obj in objects:
        if obj.instance_type != 'NONE' or obj.type not in {'MESH', 'LIGHT', 'CAMERA', 'EMPTY'}:
            raise ValueError(f"{obj.name}: convert instanced or non-mesh geometry before using this exporter")
        if obj.type != 'MESH':
            continue
        obj.add_rest_position_attribute = True
        for modifier in obj.modifiers:
            modifier.show_viewport = modifier.show_render
            if modifier.type == 'SUBSURF':
                modifier.levels = modifier.render_levels
    bpy.context.view_layer.update()


def allocate(areas, count):
    if count < len(areas):
        raise ValueError(f"At least {len(areas)} samples are needed for the in-frustum material groups")
    quota = np.asarray(areas) / sum(areas) * (count - len(areas))
    counts = np.floor(quota).astype(int) + 1
    order = np.argsort(-(quota - np.floor(quota)), kind='stable')
    counts[order[:count - counts.sum()]] += 1
    return counts


def sample_group(group, count, rng):
    """Stratify by clipped area, then use a randomly shifted 2D R2 sequence on each triangle."""
    s = group.surface
    cdf = np.cumsum(s.patch_areas[group.patches])
    slots = np.searchsorted(cdf, (np.arange(count) + rng.random(count)) * (cdf[-1] / count))
    local_index = np.arange(count) - np.searchsorted(slots, slots, side='left')
    shifts = rng.random((len(cdf), 2))
    u = (shifts[slots, 0] + local_index * 0.7548776662466927) % 1.0
    v = (shifts[slots, 1] + local_index * 0.5698402909980533) % 1.0
    root = np.sqrt(u)
    b = np.column_stack((1 - root, root * (1 - v), root * v))
    patches = group.patches[slots]
    weights = np.einsum('ni,nij->nj', b, s.patch_bary[patches])
    return s.patch_faces[patches], weights


def transfer(source, target, links):
    if source.is_linked:
        links.new(source.links[0].from_socket, target)
    else:
        target.default_value = source.default_value


def principled(material):
    if not material or not material.node_tree:
        raise ValueError("A render-visible surface has no node material")
    output = material.node_tree.get_output_node('CYCLES')
    socket = output.inputs['Surface'] if output else None
    node = socket.links[0].from_node if socket and socket.is_linked else None
    while node and node.type == 'REROUTE':
        node = node.inputs[0].links[0].from_node
    if not node or node.type != 'BSDF_PRINCIPLED':
        raise ValueError(f"{material.name}: expected an active Principled surface")
    if node.inputs['Alpha'].is_linked or node.inputs['Alpha'].default_value < 1:
        raise ValueError(f"{material.name}: fractional material alpha needs an explicit opacity model")
    return node


def generated_inputs(tree):
    """Provide undeformed Generated coordinates for these meshes, including implicit noise coordinates."""
    attr = tree.nodes.new('ShaderNodeAttribute')
    attr.attribute_name = 'PALE_sample_generated'
    for node in list(tree.nodes):
        if node.type == 'TEX_COORD':
            for link in list(node.outputs['Generated'].links):
                tree.links.new(attr.outputs['Vector'], link.to_socket)
        if node.type in {'TEX_NOISE', 'TEX_VORONOI', 'TEX_WAVE', 'TEX_MAGIC', 'TEX_GRADIENT',
                         'TEX_CHECKER', 'TEX_BRICK', 'TEX_WHITE_NOISE'}:
            vector = node.inputs.get('Vector')
            if vector and not vector.is_linked:
                tree.links.new(attr.outputs['Vector'], vector)


def bake_material(source, parameter):
    mat = source.copy()
    mat.name = f"PALE {source.name} {parameter}"
    tree = mat.node_tree
    shader = principled(mat)
    generated_inputs(tree)
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 1.0
    if parameter == 'base':
        transfer(shader.inputs['Base Color'], emission.inputs['Color'], tree.links)
    else:
        pack = tree.nodes.new('ShaderNodeCombineColor')
        pack.mode = 'RGB'
        transfer(shader.inputs['Roughness'], pack.inputs['Red'], tree.links)
        transfer(shader.inputs['Metallic'], pack.inputs['Green'], tree.links)
        pack.inputs['Blue'].default_value = 1.0
        tree.links.new(pack.outputs[0], emission.inputs['Color'])
    tree.links.new(emission.outputs[0], tree.get_output_node('CYCLES').inputs['Surface'])
    return mat


def interp(data, faces, bary):
    return np.einsum('ni,nij->nj', bary, data[faces])


def bake_at_samples(surface, faces, bary, local, materials):
    """One temporary triangle per sample. All its corners carry the same source UVs and Generated coordinates."""
    n = len(faces)
    mesh = bpy.data.meshes.new('PALE sample locations')
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    try:
        # Retain the source tangent plane, object-space location and world transform.
        tri = surface.triangles[faces]
        pos = np.empty((n, 3, 3), dtype=np.float32)
        pos[:, 0] = local
        pos[:, 1:] = local[:, None] + 0.025 * (tri[:, 1:] - tri[:, :1])
        mesh.vertices.add(3 * n)
        mesh.vertices.foreach_set('co', pos.ravel())
        mesh.loops.add(3 * n)
        mesh.loops.foreach_set('vertex_index', np.arange(3 * n, dtype=np.int32))
        mesh.polygons.add(n)
        mesh.polygons.foreach_set('loop_start', np.arange(0, 3 * n, 3, dtype=np.int32))
        mesh.update(calc_edges=True)
        for name, corners in surface.uvs.items():
            layer = mesh.uv_layers.new(name=name, do_init=False)
            uv = interp(corners, faces, bary).astype(np.float32)
            layer.uv.foreach_set('vector', np.repeat(uv, 3, axis=0).ravel())
        if surface.active_uv:
            mesh.uv_layers[surface.active_uv].active_render = True
            mesh.uv_layers.active = mesh.uv_layers[surface.active_uv]
        else:
            mesh.uv_layers.new(name='PALE bake UV')
        attr = mesh.attributes.new('PALE_sample_generated', 'FLOAT_VECTOR', 'POINT')
        coords = interp(surface.generated, faces, bary).astype(np.float32)
        attr.data.foreach_set('vector', np.repeat(coords, 3, axis=0).ravel())
        output = mesh.color_attributes.new('PALE bake result', 'FLOAT_COLOR', 'CORNER')
        mesh.color_attributes.active_color = output
        obj.matrix_world = Matrix(surface.matrix)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        mesh.materials.append(materials[0][1])
        result = {}
        for kind, mat in materials:
            mesh.materials[0] = mat
            bpy.context.view_layer.update()
            status = bpy.ops.object.bake(type='EMIT', target='VERTEX_COLORS', use_clear=True)
            if status != {'FINISHED'}:
                raise RuntimeError(f"{mat.name}: Cycles bake returned {status}")
            rgba = array_of(output.data, 'color', 4, np.float32)[::3].copy()
            if not np.isfinite(rgba).all() or np.any(rgba[:, 3] < 0.999):
                raise ValueError(f"{mat.name}: invalid or uncovered bake samples")
            if kind == 'scalars' and not np.allclose(rgba[:, 2], 1.0, atol=1e-5):
                raise ValueError(f"{mat.name}: incomplete roughness/metalness bake")
            result[kind] = rgba
        return result
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def write_group(file, group, count, rng, radius_factor, beta, planes):
    surface = group.surface
    shader = principled(group.material)
    base, rough, metal = (shader.inputs[name] for name in ('Base Color', 'Roughness', 'Metallic'))
    constants = [*base.default_value[:3], rough.default_value, metal.default_value]
    copies = []
    if base.is_linked:
        copies.append(('base', bake_material(group.material, 'base')))
    if rough.is_linked or metal.is_linked:
        copies.append(('scalars', bake_material(group.material, 'scalars')))
    faces, weights = sample_group(group, count, rng)
    radius = radius_factor * math.sqrt(group.area / count)
    normal_transform = np.linalg.inv(surface.matrix[:3, :3])
    minima, maxima = np.full(5, np.inf), np.full(5, -np.inf)
    try:
        for start in range(0, count, 100_000):
            ix = slice(start, min(start + 100_000, count))
            tri, b = faces[ix], weights[ix]
            local = interp(surface.triangles, tri, b)
            data = np.zeros((len(b), 18), dtype='<f4')
            data[:, :3] = local @ surface.matrix[:3, :3].T + surface.matrix[:3, 3]
            if np.min(data[:, :3] @ planes[:, :3].T + planes[:, 3]) < -1e-6:
                raise ValueError(f"{surface.name}: stored center outside the camera frustum")
            data[:, 3:7] = orientations(interp(surface.normals, tri, b) @ normal_transform)
            data[:, 7:9] = radius
            data[:, 12] = 1.0
            data[:, 13] = beta
            channels = np.broadcast_to(np.asarray(constants, dtype=np.float32), (len(b), 5)).copy()
            if copies:
                samples = bake_at_samples(surface, tri, b, local, copies)
                if 'base' in samples:
                    channels[:, :3] = samples['base'][:, :3]
                if 'scalars' in samples:
                    channels[:, 3:] = samples['scalars'][:, :2]
            # The PALE loader clamps the five material channels to this same interval.
            if not np.isfinite(channels).all():
                raise ValueError(f"{group.material.name}: non-finite material data")
            np.clip(channels, 0.0, 1.0, out=channels)
            data[:, 9:12] = channels[:, :3]
            data[:, 16:18] = channels[:, 3:]
            minima = np.minimum(minima, channels.min(axis=0))
            maxima = np.maximum(maxima, channels.max(axis=0))
            file.write(data.tobytes())
    finally:
        for _, mat in copies:
            bpy.data.materials.remove(mat)
    return dict(object=surface.name, material=group.material.name, samples=int(count), area=group.area,
                radius=radius, channel_min=minima.tolist(), channel_max=maxima.tolist(),
                normal_input_linked=bool(shader.inputs['Normal'].is_linked))


def main(args):
    scene = bpy.context.scene
    source = Path(bpy.data.filepath).resolve()
    with source.open('rb') as source_file:
        source_hash = hashlib.file_digest(source_file, 'sha256').hexdigest()
    tools_dir = args.dataset_tools or source.parent.parent / 'DatasetGeneration'
    sys.path.insert(0, str(tools_dir.resolve()))
    from blender_export import camera_records, light_records, render_objects
    from common import SURFEL_PLY_FIELDS, SURFEL_MATERIAL_DEFAULTS, write_scene_xml

    expected = 'x y z rot_w rot_x rot_y rot_z su sv albedo_r albedo_g albedo_b opacity beta shape power roughness metalness'.split()
    if tuple(expected) != tuple(SURFEL_PLY_FIELDS):
        raise ValueError("PBdR's dataset schema differs from this PALE binary layout")
    if not scene.camera or scene.render.use_border:
        raise ValueError("Choose an active perspective camera with a full render frame")
    rd = scene.render
    rd.resolution_x = max(1, rd.resolution_x * rd.resolution_percentage // 100)
    rd.resolution_y = max(1, rd.resolution_y * rd.resolution_percentage // 100)
    rd.resolution_percentage = 100
    view_layer = bpy.context.view_layer
    objects = render_objects(scene, view_layer)
    prepare_geometry(objects)
    graph = bpy.context.evaluated_depsgraph_get()
    camera = scene.camera.evaluated_get(graph)
    _, _, frames = camera_records(scene, [scene.camera], graph)
    projection = np.asarray(camera.calc_matrix_camera(graph, x=rd.resolution_x, y=rd.resolution_y,
                            scale_x=rd.pixel_aspect_x, scale_y=rd.pixel_aspect_y))
    clip_matrix = projection @ np.asarray(camera.matrix_world.normalized().inverted())
    planes = frustum_planes(clip_matrix)
    lights = light_records(scene, view_layer, graph, dict(export_point_surfels=True, target=[0, 0, 0]))
    groups = []
    for obj in sorted(objects, key=lambda ob: ob.name):
        if obj.type == 'MESH':
            groups.extend(mesh_groups(obj, graph, planes))
    if not groups:
        raise ValueError("No mesh area inside the active camera frustum")
    counts = allocate([group.area for group in groups], args.count)
    rng = np.random.default_rng(args.seed)
    scratch = bpy.data.scenes.new('PALE material evaluation')
    bpy.context.window.scene = scratch
    scratch.render.engine = 'CYCLES'
    scratch.cycles.device = 'CPU'
    scratch.cycles.samples = 1
    scratch.cycles.use_denoising = False

    dest = args.output.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError(f"Choose a new or empty export directory: {dest}")
    with tempfile.TemporaryDirectory(prefix='.surfel-export-', dir=dest.parent) as tmp:
        staging = Path(tmp) / 'export'
        staging.mkdir()
        reports = []
        with (staging / 'points.ply').open('wb') as output:
            header = ['ply', 'format binary_little_endian 1.0', 'comment PBdR GGX surface samples, then fixed point emitters',
                      f'element vertex {args.count + len(lights)}', *[f'property float {field}' for field in expected],
                      'end_header', '']
            output.write('\n'.join(header).encode('ascii'))
            for group, count in zip(groups, counts):
                print(f"{group.surface.name} / {group.material.name}: {count:,} surfels", flush=True)
                reports.append(write_group(output, group, int(count), rng, args.radius_factor, args.beta, planes))
            for light in lights:
                row = [*light['position'], *light['surfel_rotation_wxyz'], *light['surfel_scale'], *light['color'],
                       1, -1000, 0, light['energy'], *SURFEL_MATERIAL_DEFAULTS.values()]
                output.write(np.asarray(row, dtype='<f4').tobytes())
        write_scene_xml(staging / 'scene.xml', frames)
        report = dict(source=str(source), source_sha256=source_hash, blender=bpy.app.version_string,
                      frame=scene.frame_current, camera=frames[0], world_to_clip=clip_matrix.tolist(),
                      surface_count=args.count, emitter_count=len(lights), seed=args.seed, beta=args.beta,
                      opacity=1, shape=0, radius_factor=args.radius_factor, clipped_area=sum(g.area for g in groups),
                      sampling='World-area stratified over six-plane-clipped evaluated triangles. Includes occluded surfaces.',
                      normals='Interpolated split/corner normals, without shader bump or normal maps.',
                      shading='Scene-linear Principled Base Color, Roughness, Metallic. PALE GGX uses these channels.',
                      channels=['albedo_r', 'albedo_g', 'albedo_b', 'roughness', 'metalness'],
                      row_groups=reports, lights=lights)
        (staging / 'sampling.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.rename(staging, dest)
    print(f"Wrote {args.count:,} surface surfels and {len(lights)} emitters to {dest}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New or empty PALE export directory')
    parser.add_argument('--count', type=int, default=2_000_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--radius-factor', type=float, default=1.45, help='Disc radius / sqrt(area per sample)')
    parser.add_argument('--beta', type=float, default=-8.0, help="PALE's stored log footprint taper")
    parser.add_argument('--dataset-tools', type=Path, help='PBdR DatasetGeneration checkout (default: beside datasets/)')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if args.count < 1 or args.seed < 0 or not math.isfinite(args.radius_factor) or args.radius_factor <= 0 or not math.isfinite(args.beta):
        parser.error('Use a positive count and radius factor, a nonnegative seed and a finite beta')
    main(args)
