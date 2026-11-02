"""Independent bag predicates in world meters; no scores, clocks or history.

The enclosure surface is the body alone plus a material-mouth centroid fan.
Casing tunnels and cords do not enclose the bag contents. This geometric cap is
an approximation for self intersections, not a contact/force containment proof.
"""
import hashlib
from numbers import Real

import numpy as np

from .wuji_hanger import _fan, _finite_array, _indices, _inside


_LIMITS = ('lift_height_min_m', 'bottle_inside_fraction_min',
           'body_inside_fraction_min', 'bottom_inside_fraction_min',
           'mouth_span_ratio_max', 'material_edge_median_ratio_min',
           'material_edge_median_ratio_max')


def _scalar(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'invalid {name}')
    return float(value)


def _transform(value, name, rigid=False):
    x = _finite_array(value, (4, 4), name)
    r = x[:3, :3]
    gram = r.T @ r
    if (not np.allclose(x[3], [0, 0, 0, 1], atol=1e-8)
            or np.linalg.det(r) <= 0
            or not np.allclose(gram, np.diag(np.diag(gram)), atol=1e-8)
            or (rigid and not np.allclose(gram, np.eye(3), atol=1e-6))):
        raise ValueError(f'{name} must be a proper column-vector transform')
    return x


def _identity(item, label):
    for key in ('vertex_count', 'face_count'):
        value = item[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'invalid {label} {key}')
    if not isinstance(item['path'], str) or not item['path'].startswith('/'):
        raise ValueError(f'invalid {label} path')
    for key in ('points_sha256', 'face_counts_sha256', 'face_indices_sha256'):
        value = item[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'invalid {label} {key}')


def _edges(faces):
    return np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0, return_counts=True)


def _validate(points, bottle_poses, profile, baseline):
    try:
        if profile['schema_version'] != 1:
            raise ValueError('unsupported bag profile schema')
        mesh = profile['mesh']; _identity(mesh, 'bag mesh')
        count = mesh['vertex_count']
        if count != 1201 or mesh['face_count'] != 2396:
            raise ValueError('unexpected bag mesh identity')
        faces = _indices(profile['triangles'], count, 'triangles', 3)
        if len(faces) != mesh['face_count'] or np.any(np.diff(np.sort(faces, axis=1),axis=1) == 0):
            raise ValueError('invalid bag triangles')
        for array, key in ((faces.ravel(), 'face_indices_sha256'), (np.full(len(faces), 3), 'face_counts_sha256')):
            if hashlib.sha256(np.asarray(array, dtype=np.int64).tobytes()).hexdigest() != mesh[key]:
                raise ValueError('bag topology hash mismatch')
        body_faces = _indices(profile['body_triangles'], count, 'body triangles', 3)
        regions = {name: _indices(profile['regions'][name], count, f'{name} region') for name in ('body', 'bottom', 'mouth')}
        if any(len(np.unique(ids)) != len(ids) for ids in regions.values()):
            raise ValueError('duplicate region vertices')
        if (not set(regions['bottom']).issubset(regions['body'])
                or not set(regions['mouth']).issubset(regions['body'])
                or set(np.unique(body_faces)) != set(regions['body'])
                or len(regions['mouth']) < 3
                or not set(map(tuple,body_faces)).issubset(set(map(tuple,faces)))):
            raise ValueError('invalid body or mouth region')
        edges, multiplicity = _edges(body_faces)
        mouth = regions['mouth']
        rim = np.sort(np.column_stack((mouth, np.roll(mouth, -1))),axis=1)
        if np.any(multiplicity > 2) or set(map(tuple,edges[multiplicity == 1])) != set(map(tuple,rim)):
            raise ValueError('body boundary must match ordered mouth')
        material_edges = _indices(profile['material_edges']['indices'], count, 'material edges', 2)
        rest = _finite_array(profile['material_edges']['rest_lengths_m'], (len(material_edges),), 'rest lengths')
        if np.any(rest <= 0) or len(material_edges) != len(edges) or set(map(tuple,np.sort(material_edges,axis=1))) != set(map(tuple,edges)):
            raise ValueError('invalid body material edges')
        limits = {key: _scalar(profile['thresholds'][key], key, True) for key in _LIMITS}
        if (any(limits[key] > 1 for key in ('bottle_inside_fraction_min', 'body_inside_fraction_min', 'bottom_inside_fraction_min', 'mouth_span_ratio_max'))
                or limits['material_edge_median_ratio_min'] >= limits['material_edge_median_ratio_max']):
            raise ValueError('invalid bag threshold range')
        probes=[]
        if len(profile['bottles']) != 2:
            raise ValueError('expected two bottle profiles')
        for index, item in enumerate(profile['bottles']):
            _identity(item, 'bottle')
            if (item['name'] != f'bottle{index+1}'
                    or not isinstance(item['collider_path'], str)
                    or not item['collider_path'].startswith(item['path'] + '/')):
                raise ValueError('invalid ordered bottle identity')
            _transform(item['world_transform'], 'bottle asset transform')
            probe = np.asarray(item['local_probes_m'], dtype=float)
            if probe.ndim != 2 or probe.shape[1] != 3 or len(probe) < 4 or not np.isfinite(probe).all() or np.linalg.matrix_rank(probe-probe.mean(0)) < 3:
                raise ValueError('invalid bottle volume probes')
            probes.append(probe)
        item=profile['bin']; _identity(item, 'bin')
        bin_x=_transform(item['world_transform'], 'bin world transform')
        cavity=item['cavity']
        if cavity['axis_local'] != 1:
            raise ValueError('bin cavity must use collider local Y')
        center=_finite_array(cavity['center_radial_local'], (2,), 'bin radial center')
        axial=np.asarray(cavity['axial_coordinates_m'],dtype=float)
        radii=_finite_array(cavity['inner_radii_m'], axial.shape, 'bin inner radii')
        if axial.ndim != 1 or len(axial)<2 or not np.isfinite(axial).all() or np.any(np.diff(axial)<=0) or np.any(radii<=0):
            raise ValueError('invalid bin cavity profile')
        floor=_scalar(cavity['floor_m'], 'bin floor');rim=_scalar(cavity['rim_m'], 'bin rim')
        if floor != axial[0] or rim != axial[-1]:
            raise ValueError('bin floor/rim disagree with cavity')
        base_height=_scalar(baseline['body_height_m'], 'baseline body height')
        base_span=_scalar(baseline['mouth_span_m'], 'baseline mouth span', True)
    except (KeyError, TypeError, IndexError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError('invalid Wuji bag profile or baseline') from exc
    p=_finite_array(points, (count,3), 'world bag points')
    poses=_finite_array(bottle_poses, (2,4,4), 'bottle poses')
    for pose in poses:
        _transform(pose, 'bottle pose', rigid=True)
    return p,poses,body_faces,regions,material_edges,rest,limits,probes,bin_x,center,axial,radii,base_height,base_span


def _bin_inside(points, transform, center, axial, radii):
    local=(points-transform[:3,3]) @ np.linalg.inv(transform[:3,:3]).T
    radius=np.linalg.norm(local[:,[0,2]]-center,axis=1)
    height=local[:,1]
    return (height >= axial[0]) & (height <= axial[-1]) & (radius <= np.interp(height,axial,radii))


def measure_bag(points, bottle_poses, profile, baseline):
    """Return six native bool predicates and finite float diagnostics.

    ``points`` is (1201,3) world meters; ``bottle_poses`` is two proper rigid
    column transforms of the bottle body origins, ordered bottle1 then bottle2.
    Baseline contains initial median body_height_m and full 3D mouth_span_m.
    Malformed inputs raise ValueError; collapsed finite material is negative.
    Caller preflight verifies actual asset identity against the versioned profile.
    """
    (p,poses,faces,regions,edges,rest,limits,probes,bin_x,center,axial,radii,
     initial_height,initial_span)=_validate(points,bottle_poses,profile,baseline)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        ratios=np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)/rest
        mouth=regions['mouth']; mouth_points=p[mouth]
        body_ratio=float(np.median(ratios))
        mouth_edge_mask=np.all(np.isin(edges,mouth),axis=1)
        mouth_ratio=float(np.median(ratios[mouth_edge_mask]))
        material=all(limits['material_edge_median_ratio_min'] <= r <= limits['material_edge_median_ratio_max'] for r in (body_ratio,mouth_ratio))
        span=float(np.linalg.norm(mouth_points[:,None]-mouth_points,axis=2).max())
        body_height=float(np.median(p[regions['body'],2]));rise=body_height-initial_height
        bag_bin=_bin_inside(p,bin_x,center,axial,radii)
        bottom_fraction=float(bag_bin[regions['bottom']].mean())
        body_fraction=float(bag_bin[regions['body']].mean())
        closed=np.concatenate([p[faces],_fan(mouth_points)])
        bin_fractions=[];bag_fractions=[]
        for pose,probe in zip(poses,probes):
            world=probe@pose[:3,:3].T+pose[:3,3]
            bin_fractions.append(float(_bin_inside(world,bin_x,center,axial,radii).mean()))
            bag_fractions.append(float(_inside(closed,world).mean()) if material else 0.)
        diagnostics=dict(body_height_m=body_height,body_rise_m=rise,mouth_span_m=span,
            mouth_span_ratio=span/initial_span,material_edge_median_ratio=body_ratio,
            mouth_edge_median_ratio=mouth_ratio,material_valid=float(material),
            bottom_in_bin_fraction=bottom_fraction,body_in_bin_fraction=body_fraction,
            bottle1_in_bin_fraction=bin_fractions[0],bottle2_in_bin_fraction=bin_fractions[1],
            bottle1_in_bag_fraction=bag_fractions[0],bottle2_in_bag_fraction=bag_fractions[1])
    if not all(np.isfinite(v) for v in diagnostics.values()):
        raise ValueError('bag metric computation overflowed')
    return dict(bag_lifted=bool(material and rise>=limits['lift_height_min_m']),
        bag_in_bin=bool(material and bottom_fraction>=limits['bottom_inside_fraction_min'] and body_fraction>=limits['body_inside_fraction_min']),
        bottle1_in_bin=bool(bin_fractions[0]>=limits['bottle_inside_fraction_min']),
        bottle2_in_bin=bool(bin_fractions[1]>=limits['bottle_inside_fraction_min']),
        both_in_bag=bool(material and min(bag_fractions)>=limits['bottle_inside_fraction_min']),
        mouth_cinched=bool(material and span/initial_span<=limits['mouth_span_ratio_max']),
        diagnostics=diagnostics)
