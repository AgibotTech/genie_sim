"""Pure material/surface geometry for independent Wuji hanger predicates.

Topology and rigid asset identities belong to the caller's preflight. True mesh
opening loops are capped with centroid triangle fans for enclosure queries. This
is an approximation for folded/self-intersecting cloth, not a contact-force or
physical support proof. Three non-axis-aligned ray parities reduce sensitivity
to individual thin-sheet crossings. See the versioned profile's limitations.
No timestamps, score, prerequisites, or latched state live in this module.
"""
from numbers import Integral, Real

import numpy as np


_LIMITS = ('lift_height_min_m', 'near_collar_max_m', 'arm_inside_fraction_min',
           'arm_surface_max_m', 'material_edge_median_ratio_min',
           'material_edge_median_ratio_max', 'upright_cos_min',
           'rod_plane_cos_min', 'hook_y_tolerance_m', 'hook_z_tolerance_m',
           'neck_height_min_m', 'neck_height_max_m')


def _finite_array(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'invalid {label}') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'expected finite {label} with shape {shape}')
    return result


def _indices(value, count, label, columns=None):
    result = np.asarray(value)
    if (not np.issubdtype(result.dtype, np.integer) or result.size == 0
            or np.any(result < 0) or np.any(result >= count)
            or (columns is None and result.ndim != 1)
            or (columns is not None and (result.ndim != 2 or result.shape[1] != columns))):
        raise ValueError(f'invalid {label}')
    return result


def _validate(points, hanger_pose, profile, baseline):
    try:
        if profile['schema_version'] != 1:
            raise ValueError('unsupported hanger profile schema')
        count = profile['mesh']['vertex_count']
        if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
            raise ValueError('invalid vertex_count')
        faces = _indices(profile['triangles'], count, 'triangles', 3)
        if len(faces) != profile['mesh']['face_count'] or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
            raise ValueError('invalid cloth triangle topology')
        boundaries = {}
        for name in ('collar', 'hem', 'cuff_left', 'cuff_right'):
            ids = _indices(profile['boundaries'][name], count, f'{name} boundary')
            region = _indices(profile['regions'][name], count, f'{name} region')
            if len(ids) < 3 or len(np.unique(ids)) != len(ids) or set(region) != set(ids):
                raise ValueError(f'invalid material opening: {name}')
            boundaries[name] = ids
        edges, multiplicity = np.unique(np.sort(np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0, return_counts=True)
        opening_edges = np.sort(np.concatenate([np.column_stack((ids, np.roll(ids, -1)))
                                               for ids in boundaries.values()]), axis=1)
        actual = edges[multiplicity == 1]
        if (np.any(multiplicity > 2) or len(actual) != len(opening_edges)
                or set(map(tuple, actual)) != set(map(tuple, opening_edges))):
            raise ValueError('opening loops do not match cloth topology')
        material_edges = _indices(profile['material_edges']['indices'], count, 'material edges', 2)
        lengths = _finite_array(profile['material_edges']['rest_lengths_m'], (len(material_edges),), 'rest lengths')
        if np.any(lengths <= 0) or set(map(tuple, np.sort(material_edges, axis=1))) != set(map(tuple, edges)):
            raise ValueError('invalid material edge rest lengths')
        limits = profile['thresholds']
        for key in _LIMITS:
            if isinstance(limits[key], bool) or not isinstance(limits[key], Real) or not np.isfinite(limits[key]):
                raise ValueError(f'invalid hanger threshold: {key}')
        if (any(limits[key] <= 0 for key in _LIMITS if key != 'neck_height_min_m')
                or limits['material_edge_median_ratio_min'] >= limits['material_edge_median_ratio_max']
                or limits['neck_height_min_m'] >= limits['neck_height_max_m']
                or any(limits[key] > 1 for key in ('arm_inside_fraction_min', 'upright_cos_min', 'rod_plane_cos_min'))):
            raise ValueError('invalid hanger threshold range')
        hanger_transform = _finite_array(profile['hanger']['world_transform'], (4, 4), 'hanger asset transform')
        if (not np.allclose(hanger_transform[3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(hanger_transform[:3, :3].T @ hanger_transform[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(hanger_transform[:3, :3]), 1, atol=1e-6)):
            raise ValueError('hanger probes require unscaled rigid asset coordinates')
        probes = profile['hanger']['local_probes']
        tip = _finite_array(probes['insertion_end_m'], (3,), 'insertion end')
        arms = [_finite_array(probes[f'arm_{side}_m'], (3, 3), f'{side} arm probes') for side in ('left', 'right')]
        neck = _finite_array(probes['neck_segment_m'], (2, 3), 'neck segment')
        pocket = _finite_array(probes['hook_pocket_center_m'], (3,), 'hook pocket')
        if (np.linalg.norm(neck[1] - neck[0]) < 1e-6
                or not np.isclose(neck[0, 1], limits['neck_height_min_m'])
                or not np.isclose(neck[1, 1], limits['neck_height_max_m'])):
            raise ValueError('degenerate neck segment')
        rack = profile['rack']
        center = _finite_array(rack['center_world_m'], (3,), 'rack center')
        axis = _finite_array(rack['axis_world'], (3,), 'rack axis')
        transform = _finite_array(rack['world_transform'], (4, 4), 'rack world transform')
        scale = np.linalg.norm(transform[:3, :3], axis=0)
        if (not np.isclose(np.linalg.norm(axis), 1, atol=1e-6) or np.any(scale <= 0)
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(transform[:3, 3], center, atol=1e-8)
                or not np.allclose(transform[:3, 2] / scale[2], axis, atol=1e-8)):
            raise ValueError('invalid rack geometry')
        for name in ('radius_m', 'length_m'):
            if isinstance(rack[name], bool) or not isinstance(rack[name], Real) or not np.isfinite(rack[name]) or rack[name] <= 0:
                raise ValueError(f'invalid rack {name}')
        if not np.allclose([rack['radius_m'] * 2, rack['length_m']], scale[[0, 2]], atol=1e-8):
            raise ValueError('rack dimensions disagree with asset transform')
        base = []
        for key in ('collar_height_m', 'hanger_height_m'):
            value = baseline[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f'invalid baseline {key}')
            base.append(float(value))
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('invalid Wuji hanger profile or baseline') from exc
    p = _finite_array(points, (count, 3), 'world cloth points')
    h = _finite_array(hanger_pose, (4, 4), 'hanger pose')
    rotation = h[:3, :3]
    if (not np.allclose(h[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError('hanger pose must be a proper rigid column-vector transform')
    return p, h, faces, boundaries, material_edges, lengths, limits, tip, arms, neck, pocket, center, axis, base


def _fan(loop):
    return np.stack((loop, np.roll(loop, -1, axis=0), np.broadcast_to(loop.mean(0), loop.shape)), axis=1)


def _ray_hits(triangles, queries, direction):
    a = triangles[:, 0]
    edge1, edge2 = triangles[:, 1] - a, triangles[:, 2] - a
    cross = np.cross(direction, edge2)
    det = np.einsum('ij,ij->i', edge1, cross)
    valid = np.abs(det) > 1e-12
    inverse = np.divide(1., det, out=np.zeros_like(det), where=valid)
    relative = queries[:, None] - a
    u = np.einsum('qij,ij->qi', relative, cross) * inverse
    v_cross = np.cross(relative, edge1)
    v = np.einsum('qij,j->qi', v_cross, direction) * inverse
    distance = np.einsum('qij,ij->qi', v_cross, edge2) * inverse
    return valid & (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > 1e-9), distance


def _inside(triangles, queries):
    votes = np.zeros(len(queries), dtype=int)
    for direction in ([1., .371, .173], [.231, 1., .419], [.337, .113, 1.]):
        hits, _ = _ray_hits(triangles, queries, np.asarray(direction))
        votes += hits.sum(axis=1) % 2
    return votes >= 2


def _surface_distances(triangles, queries):
    """Exact closest distance to triangles, including edges and vertices."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    norm2 = np.einsum('ij,ij->i', normal, normal)
    relative = queries[:, None] - a
    signed = np.einsum('qij,ij->qi', relative, normal)
    factor = np.divide(signed, norm2, out=np.zeros_like(signed), where=norm2 > 1e-24)
    projection = relative - factor[:, :, None] * normal
    d00 = np.einsum('ij,ij->i', ab, ab)
    d01 = np.einsum('ij,ij->i', ab, ac)
    d11 = np.einsum('ij,ij->i', ac, ac)
    d20 = np.einsum('qij,ij->qi', projection, ab)
    d21 = np.einsum('qij,ij->qi', projection, ac)
    denominator = d00 * d11 - d01 * d01
    u = np.divide(d11 * d20 - d01 * d21, denominator, out=np.zeros_like(d20), where=denominator > 1e-24)
    v = np.divide(d00 * d21 - d01 * d20, denominator, out=np.zeros_like(d21), where=denominator > 1e-24)
    inside = (denominator > 1e-24) & (u >= 0) & (v >= 0) & (u + v <= 1)
    best = np.where(inside, signed * factor, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        length2 = np.einsum('ij,ij->i', edge, edge)
        rel = queries[:, None] - start
        t = np.divide(np.einsum('qij,ij->qi', rel, edge), length2, out=np.zeros_like(best), where=length2 > 1e-24)
        residual = rel - np.clip(t, 0, 1)[:, :, None] * edge
        best = np.minimum(best, np.einsum('qij,qij->qi', residual, residual))
    return np.sqrt(np.maximum(best.min(axis=1), 0))


def measure_hanger(points, hanger_pose, profile, baseline):
    """Return five native bools plus finite scalar diagnostics; no temporal state.

    ``points`` are world meters, ``hanger_pose`` a proper rigid column transform.
    Baseline keys are initialized median ``collar_height_m`` and hanger origin
    ``hanger_height_m``. Malformed input raises ValueError. A finite collapsed
    garment is a geometric negative while hanger pickup remains independent.
    The caller must verify profile mesh/rigid identities before evaluating.
    """
    (p, h, faces, loops, edges, rest, limits, tip, arms, neck, pocket,
     center, axis, base) = _validate(points, hanger_pose, profile, baseline)
    local = (p - h[:3, 3]) @ h[:3, :3]
    material = local[faces]
    caps = {name: _fan(local[ids]) for name, ids in loops.items()}
    closed = np.concatenate([material, *caps.values()])
    edge_lengths = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)
    if not np.isfinite(edge_lengths).all():
        raise ValueError('cloth metric computation overflowed')
    edge_ratios = edge_lengths / rest
    ratio = float(np.median(edge_ratios))
    collar_edges = np.all(np.isin(edges, loops['collar']), axis=1)
    collar_ratio = float(np.median(edge_ratios[collar_edges]))
    material_valid = all(limits['material_edge_median_ratio_min'] <= value <= limits['material_edge_median_ratio_max']
                         for value in (ratio, collar_ratio))
    collar_height = float(np.median(p[loops['collar'], 2]))
    cloth_rise = collar_height - base[0]
    hanger_rise = float(h[2, 3]) - base[1]
    near_distance = float(_surface_distances(caps['collar'], tip[None])[0])
    # The actual material collar cap must cross the finite neck segment.
    direction = neck[1] - neck[0]
    hit, distance = _ray_hits(caps['collar'], neck[:1], direction)
    neck_crossing = bool(np.any(hit & (distance <= 1)))
    arm_probes = np.concatenate(arms)
    arm_inside = _inside(closed, arm_probes).reshape(2, 3)
    enclosure = arm_inside.mean(axis=1)
    surface = _surface_distances(material, arm_probes).reshape(2, 3)
    # Support proximity is to real cloth, never to the artificial caps.
    arm_distances = surface.min(axis=1)
    inserted = bool(material_valid and neck_crossing
                    and np.all(enclosure >= limits['arm_inside_fraction_min'])
                    and bool(np.all(arm_inside[:, -1]))
                    and np.all(arm_distances <= limits['arm_surface_max_m']))
    rod_center = (center - h[:3, 3]) @ h[:3, :3]
    rod_axis = axis @ h[:3, :3]
    normal_cos = float(abs(rod_axis[0]))
    # Bounded fallback keeps diagnostics finite for a parallel rack/hanger.
    line_parameter = float(-rod_center[0] / rod_axis[0]) if normal_cos > 1e-8 else 0.
    crossing = rod_center + line_parameter * rod_axis
    y_error, z_error = float(crossing[1] - pocket[1]), float(crossing[2] - pocket[2])
    upright = float(h[2, 1])
    engaged = bool(normal_cos >= limits['rod_plane_cos_min']
                   and abs(line_parameter) <= profile['rack']['length_m'] / 2
                   and abs(y_error) <= limits['hook_y_tolerance_m']
                   and abs(z_error) <= limits['hook_z_tolerance_m'])
    diagnostics = {'collar_height_m': collar_height, 'hanger_height_m': float(h[2, 3]),
                   'cloth_rise_m': cloth_rise, 'hanger_rise_m': hanger_rise,
                   'material_edge_median_ratio': ratio, 'collar_edge_median_ratio': collar_ratio, 'material_valid': float(material_valid),
                   'insertion_end_collar_distance_m': near_distance, 'collar_neck_crossing': float(neck_crossing),
                   'left_outer_arm_inside': float(arm_inside[0, -1]), 'right_outer_arm_inside': float(arm_inside[1, -1]),
                   'left_arm_inside_fraction': float(enclosure[0]), 'right_arm_inside_fraction': float(enclosure[1]),
                   'left_arm_surface_distance_m': float(arm_distances[0]), 'right_arm_surface_distance_m': float(arm_distances[1]),
                   'hanger_upright_cos': upright, 'rod_plane_normal_cos': normal_cos,
                   'rod_axis_parameter_m': line_parameter, 'rod_hook_y_error_m': y_error,
                   'rod_hook_z_error_m': z_error, 'hook_engaged': float(engaged)}
    if not all(np.isfinite(value) for value in diagnostics.values()):
        raise ValueError('hanger metric computation overflowed')
    return {'cloth_lifted': bool(material_valid and cloth_rise >= limits['lift_height_min_m']),
            'hanger_lifted': bool(hanger_rise >= limits['lift_height_min_m']),
            'near_collar': bool(material_valid and near_distance <= limits['near_collar_max_m']),
            'inserted': inserted, 'hung': bool(inserted and engaged and upright >= limits['upright_cos_min']),
            'diagnostics': diagnostics}
