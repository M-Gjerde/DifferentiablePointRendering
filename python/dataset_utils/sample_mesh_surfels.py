#!/usr/bin/env python3
"""Sample Blender OBJ/mesh PLY/GLB/glTF into PALE's quaternion surfel PLY format.

See docs/mesh_to_surfels.md. Dependencies: numpy, scipy, trimesh, Pillow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


FIELDS = ('x y z rot_w rot_x rot_y rot_z su sv albedo_r albedo_g albedo_b '
          'opacity beta shape power').split()


def srgb_to_linear(value):
    value = np.asarray(value, dtype=float)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def validate_obj_assets(path):
    """Fail visibly when trimesh would silently discard a missing base-color map."""
    libraries = []
    for line in path.read_text(errors='replace').splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0].lower() == 'mtllib':
            # Blender may put spaces in filenames. Prefer the whole name when
            # it exists; otherwise accept the OBJ list-of-libraries spelling.
            whole = path.parent / fields[1]
            libraries.extend([whole] if whole.is_file() else
                             [path.parent / name for name in fields[1].split()])
    for library in libraries:
        if not library.is_file():
            raise ValueError(f'Missing OBJ material library: {library}')
        for line in library.read_text(errors='replace').splitlines():
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2 or fields[0].lower() != 'map_kd':
                continue
            if fields[1].startswith('-'):
                raise ValueError(f'MTL texture options are unsupported: {line.strip()}. Bake the UV transform first.')
            # trimesh resolves OBJ assets relative to the OBJ directory.
            texture = path.parent / fields[1]
            try:
                with Image.open(texture) as image:
                    image.verify()
            except OSError as exc:
                raise ValueError(f'Cannot read base-color texture: {texture}') from exc


def load_meshes(path):
    if path.suffix.lower() not in ('.obj', '.ply', '.glb', '.gltf'):
        raise ValueError('Input must be OBJ, mesh PLY, GLB, or glTF.')
    if path.suffix.lower() == '.obj':
        validate_obj_assets(path)
    # Keep UV seams, material groups, and imported vertex colors intact.
    scene = trimesh.load_scene(path, process=False)
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        original = scene.geometry[name]
        if not isinstance(original, trimesh.Trimesh) or not len(original.faces):
            continue
        mesh = original.copy()
        mesh.apply_transform(transform)
        if path.suffix.lower() in ('.glb', '.gltf'):
            # glTF uses Y up; recover Blender/viewer Z up after scene transforms.
            mesh.apply_transform([[1, 0, 0, 0], [0, 0, -1, 0],
                                  [0, 1, 0, 0], [0, 0, 0, 1]])
        if not np.isfinite(mesh.vertices).all():
            raise ValueError(f'Non-finite coordinates in {name}.')
        mesh.update_faces(np.isfinite(mesh.area_faces) & (mesh.area_faces > 0))
        if len(mesh.faces):
            meshes.append(mesh)
    if not meshes:
        raise ValueError('No nondegenerate mesh faces found. Export a mesh, not a point cloud.')
    return meshes


def area_samples(mesh, count, rng):
    """Area-weighted triangles and uniform barycentric coordinates."""
    cdf = np.cumsum(mesh.area_faces)
    face = np.searchsorted(cdf, rng.random(count) * cdf[-1])
    root = np.sqrt(rng.random(count))
    v = rng.random(count)
    bary = np.column_stack((1 - root, root * (1 - v), root * v))
    positions = np.einsum('ni,nij->nj', bary, mesh.triangles[face])
    return positions, face, bary


def report_color_sources(mesh):
    """Explain what was actually exported before expensive surface sampling."""
    visual = mesh.visual
    if visual.kind != 'texture':
        print(f'  Color source: {visual.kind or "fallback"} colors', flush=True)
        return
    material = visual.material
    for mat in getattr(material, 'materials', [material]):
        is_pbr = isinstance(mat, trimesh.visual.material.PBRMaterial)
        image = mat.baseColorTexture if is_pbr else getattr(mat, 'image', None)
        name = getattr(mat, 'name', None) or '(unnamed)'
        if image is not None:
            print(f'  Material {name}: base-color texture {image.width}x{image.height}', flush=True)
            continue
        print(f'  Material {name}: constant base color; no base-color texture exported', flush=True)
        properties = getattr(mat, 'kwargs', {})
        has_normal_map = (getattr(mat, 'normalTexture', None) is not None or
                          any(key.lower() in ('map_bump', 'bump', 'norm') for key in properties))
        if has_normal_map:
            warnings.warn(f'Material {name} has a normal/bump map but no base-color texture '
                          '(OBJ map_Kd). Normal maps do not supply diffuse color. '
                          'Connect the color image to Principled BSDF Base Color and re-export.')


def even_samples(mesh, count, rng, candidates=12):
    """Dart throwing over area samples, relaxing spacing until exactly count remain.

    Normal-aware exclusion avoids merging opposite sides of thin sheets. This is
    approximate blue noise in Euclidean distance, not geodesic Poisson sampling.
    """
    points, faces, bary = area_samples(mesh, max(count * candidates, count), rng)
    normals = mesh.face_normals[faces]
    tree = cKDTree(points)
    radius = 0.85 * np.sqrt(mesh.area / count)
    selected = []
    chosen = np.zeros(len(points), dtype=bool)
    for _ in range(80):
        blocked = chosen.copy()
        for i in selected:
            nearby = np.asarray(tree.query_ball_point(points[i], radius), dtype=int)
            blocked[nearby[normals[nearby] @ normals[i] > 0.5]] = True
        for i in range(len(points)):
            if blocked[i]:
                continue
            selected.append(i)
            chosen[i] = True
            if len(selected) == count:
                selected = np.asarray(selected)
                return points[selected], faces[selected], bary[selected], radius
            nearby = np.asarray(tree.query_ball_point(points[i], radius), dtype=int)
            blocked[nearby[normals[nearby] @ normals[i] > 0.5]] = True
        radius *= 0.8
    raise ValueError('Could not separate samples; check mesh scale and duplicate geometry.')


def sample_texture(image, uv):
    """Bilinear periodic UV sampling, with OBJ's bottom-left V convention."""
    pixels = srgb_to_linear(np.asarray(image.convert('RGB'), dtype=float) / 255)
    h, w = pixels.shape[:2]
    x = np.mod(uv[:, 0], 1) * w - 0.5
    y = (1 - np.mod(uv[:, 1], 1)) * h - 0.5
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - ix)[:, None], (y - iy)[:, None]
    return ((1-fy) * ((1-fx)*pixels[iy % h, ix % w] + fx*pixels[iy % h, (ix+1) % w])
            + fy * ((1-fx)*pixels[(iy+1) % h, ix % w] + fx*pixels[(iy+1) % h, (ix+1) % w]))


def sample_colors(mesh, faces, bary, vertex_space='srgb', material_space='linear',
                  fallback=(0.5, 0.5, 0.5), multiply_texture_kd=False):
    visual = mesh.visual
    if visual.kind == 'vertex':
        rgb = np.asarray(visual.vertex_colors[:, :3], dtype=float) / 255
        if vertex_space == 'srgb':
            rgb = srgb_to_linear(rgb)
        return np.einsum('ni,nij->nj', bary, rgb[mesh.faces[faces]])
    if visual.kind == 'face':
        rgb = np.asarray(visual.face_colors[faces, :3], dtype=float) / 255
        return srgb_to_linear(rgb) if vertex_space == 'srgb' else rgb
    if visual.kind == 'texture':
        material = visual.material
        materials = getattr(material, 'materials', [material])
        face_materials = getattr(visual, 'face_materials', None)
        ids = np.zeros(len(faces), dtype=int) if face_materials is None else face_materials[faces]
        out = np.empty((len(faces), 3))
        for mid in np.unique(ids):
            mask = ids == mid
            mat = materials[int(mid)]
            # trimesh retains the original OBJ Kd floats in kwargs; avoid 8-bit
            # quantization and preserve Blender's scene-linear material values.
            is_pbr = isinstance(mat, trimesh.visual.material.PBRMaterial)
            if is_pbr:
                factor = mat.baseColorFactor
                diffuse = np.ones(3) if factor is None else np.asarray(factor[:3], dtype=float) / 255
                image = mat.baseColorTexture
            else:
                raw_kd = getattr(mat, 'kwargs', {}).get('kd')
                diffuse = (np.asarray(raw_kd, dtype=float)[:3] if raw_kd is not None
                           else np.asarray(mat.diffuse[:3], dtype=float) / 255)
                if material_space == 'srgb':
                    diffuse = srgb_to_linear(diffuse)
                image = getattr(mat, 'image', None)
            if image is None:
                out[mask] = diffuse
            else:
                if visual.uv is None:
                    raise ValueError('Texture present but UV coordinates are missing.')
                uv = np.einsum('ni,nij->nj', bary[mask], visual.uv[mesh.faces[faces[mask]]])
                # map_Kd supplies the base color; Blender commonly exports Kd as
                # a fallback, not an additional tint. Explicit tint is opt-in in CLI.
                out[mask] = sample_texture(image, uv)
                if is_pbr or multiply_texture_kd:
                    out[mask] *= diffuse
        # glTF COLOR_0 modulates the material and is already linear.
        vertex_tint = getattr(visual, 'vertex_attributes', {}).get('color')
        if vertex_tint is not None:
            tint = np.asarray(vertex_tint)[:, :3]
            if np.issubdtype(tint.dtype, np.integer):
                tint = tint.astype(float) / np.iinfo(tint.dtype).max
            out *= np.einsum('ni,nij->nj', bary, tint[mesh.faces[faces]])
        return np.clip(out, 0, 1)
    warnings.warn('Mesh has no exported color; using --fallback-color (linear RGB).')
    return np.tile(fallback, (len(faces), 1))


def nearest_surface_samples(points, normals, probes, probe_normals, spacing):
    """Find a nearby, similarly oriented tangent plane, in bounded batches."""
    tree = cKDTree(points)
    for start in range(0, len(probes), 8192):
        p = probes[start:start+8192]
        pn = probe_normals[start:start+8192]
        distance, ids = tree.query(p, k=min(32, len(points)))
        distance = distance.reshape(len(p), -1)
        ids = ids.reshape(len(p), -1)
        delta = p[:, None, :] - points[ids]
        depth = np.einsum('nkj,nkj->nk', delta, normals[ids])
        alignment = np.einsum('nj,nkj->nk', pn, normals[ids])
        valid = (alignment >= 0.75) & (np.abs(depth) <= spacing * 0.5) & (distance <= 3*spacing)
        score = np.where(valid, distance, np.inf)
        best = np.argmin(score, axis=1)
        rows = np.arange(len(p))
        tangent = np.sqrt(np.maximum(distance[rows, best]**2 - depth[rows, best]**2, 0))
        yield start, ids[rows, best], tangent, np.isfinite(score[rows, best])


def fit_radii(mesh, points, normals, rng, beta, target, overlap, probes_per_surfel):
    spacing = np.sqrt(mesh.area / len(points))
    probes, faces, _ = area_samples(mesh, max(4096, len(points)*probes_per_surfel), rng)
    covering = np.full(len(points), 0.8 * spacing)
    missing = 0
    for _, ids, tangent, valid in nearest_surface_samples(points, normals, probes, mesh.face_normals[faces], spacing):
        np.maximum.at(covering, ids[valid], tangent[valid])
        missing += int((~valid).sum())
    # alpha(r) = (1 - (r/R)^2) ** (4*exp(beta)), as in KernelHelpers.h.
    fraction = np.sqrt(-np.expm1(np.log(target) / (4*np.exp(beta))))
    radii = covering * overlap / fraction
    return radii, missing, spacing


def coverage_report(mesh, points, normals, radii, rng, beta, target, probe_count):
    probes, faces, _ = area_samples(mesh, probe_count, rng)
    alpha = np.zeros(probe_count)
    spacing = np.sqrt(mesh.area / len(points))
    for start, ids, tangent, valid in nearest_surface_samples(points, normals, probes, mesh.face_normals[faces], spacing):
        values = np.maximum(0, 1 - (tangent/radii[ids])**2) ** (4*np.exp(beta))
        alpha[start:start+len(ids)] = np.where(valid, values, 0)
    return dict(probes=probe_count, fraction_at_target=float(np.mean(alpha >= target)),
                fraction_without_nearby_support=float(np.mean(alpha == 0)),
                nearest_compatible_alpha_min=float(alpha.min()),
                nearest_compatible_alpha_p01=float(np.quantile(alpha, .01)))


def quaternions(normals):
    helper = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    u = np.cross(helper, normals)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(normals, u)
    xyzw = Rotation.from_matrix(np.stack((u, v, normals), axis=2)).as_quat()
    return xyzw[:, [3, 0, 1, 2]]


def allocate_counts(areas, count):
    if count < len(areas):
        raise ValueError(f'Need at least {len(areas)} surfels (one per mesh/material group).')
    # Reserve one sample per group, then apportion the rest by surface area.
    quota = np.asarray(areas) / np.sum(areas) * (count - len(areas))
    allocation = np.floor(quota).astype(int) + 1
    remainder = count - allocation.sum()
    allocation[np.argsort(-(quota - np.floor(quota)), kind='stable')[:remainder]] += 1
    return allocation


def perturb_normals(normals, max_degrees, rng):
    """Uniform solid-angle jitter in a bounded cone about each unit normal."""
    if max_degrees == 0:
        return normals.copy()
    helper = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    u = np.cross(helper, normals)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(normals, u)
    angle = 2 * np.arcsin(np.sqrt(rng.random(len(normals))) *
                          np.sin(np.deg2rad(max_degrees) / 2))
    azimuth = rng.uniform(0, 2*np.pi, len(normals))
    tangent = np.cos(azimuth)[:, None]*u + np.sin(azimuth)[:, None]*v
    result = np.cos(angle)[:, None]*normals + np.sin(angle)[:, None]*tangent
    return result / np.linalg.norm(result, axis=1)[:, None]


def overhead_lights(meshes, power):
    """Three white emitters at 45-degree elevation and 120-degree azimuth spacing."""
    bounds = np.asarray([mesh.bounds for mesh in meshes])
    lower, upper = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    extent = float(np.max(upper - lower))
    center = (lower + upper) / 2
    height = upper[2] - center[2] + 0.5 * extent
    angles = np.arange(3) * (2 * np.pi / 3)
    offsets = height * np.column_stack((np.cos(angles), np.sin(angles), np.ones(3)))
    positions = center + offsets
    normals = -offsets / np.linalg.norm(offsets, axis=1)[:, None]
    radius = 0.25 * extent
    return np.column_stack((positions, quaternions(normals),
                            np.full((3, 2), radius), np.ones((3, 4)),
                            np.full(3, -8.), np.zeros(3), np.full(3, power)))


def convert(args):
    meshes = load_meshes(args.input)
    counts = allocate_counts([m.area for m in meshes], args.count)
    for index, mesh in enumerate(meshes):
        print(f'Loaded mesh {index+1}/{len(meshes)}:', flush=True)
        report_color_sources(mesh)
    rng = np.random.default_rng(args.seed)
    # Separate stream: changing normal noise must not resample the geometry,
    # coverage probes, or any later mesh/material group.
    normal_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 7193]))
    rows, reports = [], []
    for index, (mesh, count) in enumerate(zip(meshes, counts)):
        print(f'Sampling mesh {index+1}/{len(meshes)}: {count:,} surfels', flush=True)
        points, faces, bary, separation = even_samples(mesh, int(count), rng, args.candidates)
        normals = mesh.face_normals[faces]
        radii, missing, spacing = fit_radii(mesh, points, normals, rng, args.beta,
                                           args.target_alpha, args.overlap, args.probes_per_surfel)
        radii *= args.scale
        if not np.all(radii.astype(np.float32) > 0):
            raise ValueError('Scaled footprint radii must remain positive in float32; increase --scale.')
        normals = perturb_normals(normals, args.normal_noise_deg, normal_rng)
        vertex_space = 'linear' if args.input.suffix.lower() in ('.glb', '.gltf') else args.vertex_color_space
        colors = sample_colors(mesh, faces, bary, vertex_space,
                               args.material_color_space, args.fallback_color,
                               args.multiply_texture_kd)
        data = np.column_stack((points, quaternions(normals), radii, radii, colors,
                                np.ones(count), np.full(count, args.beta),
                                np.zeros(count), np.zeros(count)))
        if not np.isfinite(data).all() or not np.isfinite(data.astype(np.float32)).all():
            raise ValueError('Output exceeds finite float32 range; rescale the mesh or adjust parameters.')
        rows.append(data)
        report = coverage_report(mesh, points, normals, radii, rng, args.beta,
                                 args.target_alpha, max(4096, int(count)*4))
        report.update(mesh=index, count=int(count), area=float(mesh.area),
                      nominal_spacing=float(spacing), minimum_separation=float(separation),
                      radius_min=float(radii.min()), radius_max=float(radii.max()),
                      unmatched_fitting_probes=missing)
        reports.append(report)
        print(f"  Independent coverage: {report['fraction_at_target']:.4%} at alpha >= {args.target_alpha}", flush=True)
        if report['fraction_at_target'] < .999:
            warnings.warn('Coverage below 99.9% on probes. Increase --count or --overlap; inspect thin/sharp features.')
    light_info = []
    if not args.no_light:
        lights = overhead_lights(meshes, args.light_power)
        rows.append(lights)
        normals = Rotation.from_quat(lights[:, [4, 5, 6, 3]]).apply([0, 0, 1])
        light_info = [dict(position=light[:3].tolist(), radius=float(light[7]),
                           power=args.light_power, normal=normal.tolist())
                      for light, normal in zip(lights, normals)]
        print(f'Added 3 overhead lights, 120 degrees apart: power={args.light_power:g} each')
    data = np.concatenate(rows).astype('<f4')
    if not np.isfinite(data).all():
        raise ValueError('Output exceeds finite float32 range; rescale the model/light.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = ['ply', 'format binary_little_endian 1.0',
              'comment PALE mesh surface samples; linear albedo; world-space radii',
              f'element vertex {len(data)}']
    header += [f'property float {field}' for field in FIELDS] + ['end_header']
    with args.output.open('wb') as stream:
        stream.write(('\n'.join(header) + '\n').encode('ascii'))
        stream.write(data.tobytes())
    report_path = args.output.with_suffix('.coverage.json')
    report_path.write_text(json.dumps(dict(input=str(args.input), count=args.count,
        total_ply_vertices=len(data), lights=light_info,
        seed=args.seed, beta=args.beta, target_alpha=args.target_alpha, overlap=args.overlap,
        scale=args.scale,
        normal_noise_deg=args.normal_noise_deg,
        candidates=args.candidates, probes_per_surfel=args.probes_per_surfel,
        vertex_color_space=args.vertex_color_space, material_color_space=args.material_color_space,
        multiply_texture_kd=args.multiply_texture_kd, fallback_color=args.fallback_color,
        measurement='Independent random probes, nearest compatible tangent-plane alpha; not a rendered opacity guarantee.',
        meshes=reports), indent=2) + '\n')
    print(f'Wrote {args.output}\nCoverage report: {report_path}')
    return data, reports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--count', '-n', type=int, required=True, help='Exact model surfel count (plus three lights by default)')
    parser.add_argument('--no-light', action='store_true', help='Omit all three default overhead disc lights')
    parser.add_argument('--light-power', type=float, default=100, help='Power per overhead light (default: 200 each)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--normal-noise-deg', type=float, default=0.5,
                        help='Maximum random model-normal tilt in degrees (default: 0.5; 0 disables)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Multiply model footprint radii after coverage fitting; positions/lights unchanged (default: 1)')
    parser.add_argument('--candidates', type=int, default=12, help='Area candidates per surfel (default: 12)')
    parser.add_argument('--probes-per-surfel', type=int, default=16)
    parser.add_argument('--overlap', type=float, default=1.2, help='Coverage radius safety factor (default: 1.2)')
    parser.add_argument('--beta', type=float, default=-8, help='PALE log taper parameter (default: -8)')
    parser.add_argument('--target-alpha', type=float, default=.999)
    parser.add_argument('--vertex-color-space', choices=['srgb', 'linear'], default='srgb')
    parser.add_argument('--material-color-space', choices=['srgb', 'linear'], default='linear')
    parser.add_argument('--multiply-texture-kd', action='store_true',
                        help='Multiply OBJ map_Kd texture by Kd instead of using Kd only as fallback')
    parser.add_argument('--fallback-color', type=float, nargs=3, default=[.5, .5, .5], metavar=('R', 'G', 'B'))
    args = parser.parse_args(argv)
    if not np.isfinite(args.normal_noise_deg) or not 0 <= args.normal_noise_deg < 90:
        parser.error('--normal-noise-deg must be finite and in [0, 90)')
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error('--scale must be finite and > 0')
    if not np.isfinite(args.light_power) or not 0 < args.light_power <= np.finfo(np.float32).max:
        parser.error('--light-power must be positive and finite in float32')
    if args.count < 1 or args.candidates < 2 or args.probes_per_surfel < 1 or args.seed < 0:
        parser.error('count/probes must be positive, candidates >= 2, and seed >= 0')
    if not np.isfinite(args.overlap) or args.overlap < 1:
        parser.error('--overlap must be finite and >= 1')
    if not np.isfinite(args.beta) or not -20 <= args.beta <= 2:
        parser.error('--beta must be between -20 and 2')
    if not 0 < args.target_alpha < 1:
        parser.error('--target-alpha must be strictly between 0 and 1')
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in args.fallback_color):
        parser.error('--fallback-color must contain three finite values in [0, 1]')
    if args.input.resolve() in (args.output.resolve(), args.output.with_suffix('.coverage.json').resolve()):
        parser.error('Output must not overwrite the input mesh')
    if args.output.suffix.lower() != '.ply':
        parser.error('Output must have a .ply extension')
    return args


def main():
    try:
        convert(parse_args())
    except (ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
