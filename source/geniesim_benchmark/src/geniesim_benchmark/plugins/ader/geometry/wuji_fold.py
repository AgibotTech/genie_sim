"""Pure NumPy material-region measurements for the Wuji T-shirt fold.

The profile fixes vertex identity in the original garment. Coordinates follow
upper-torso material anchors in the horizontal plane, so moving or yawing the
whole garment does not earn credit. This function has no temporal state: ADER
owns confirmation windows, event ordering, and latched scores.
"""

from numbers import Integral, Real

import numpy as np


def measure_fold(points, profile):
    """Return independent ``left``, ``right``, ``final`` shape predicates.

    ``points`` must be finite world-space meters with shape (vertex_count, 3).
    ``profile`` is the versioned checker_profile.json mapping, including fixed
    material ``regions`` and metric ``thresholds``. Malformed data raises
    ValueError; a valid but collapsed garment returns false predicates.

    ``diagnostics`` contains finite numeric measurements in meters (area in
    square meters). Topology identity must be verified by the caller against
    profile['mesh'] before calling: points alone cannot establish connectivity.
    Translation and gravity-axis yaw are supported; arbitrary roll/pitch is
    deliberately constrained by height/upper-torso projection checks.
    Total-width/length caps are optional; archived profiles that supply them
    retain their limits. Their measurements still define the footprint area.
    """
    try:
        count = profile['mesh']['vertex_count']
        if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
            raise ValueError('vertex_count must be a positive integer')
        regions = {}
        for name in ('collar', 'core_left', 'core_right', 'core_mid',
                     'left_shoulder', 'right_shoulder', 'left_hem', 'right_hem',
                     'hem_center'):
            raw = np.asarray(profile['regions'][name])
            if (raw.ndim != 1 or raw.size == 0
                    or not np.issubdtype(raw.dtype, np.integer)
                    or np.any(raw < 0) or np.any(raw >= count)):
                raise ValueError(f'invalid material region: {name}')
            regions[name] = raw
        material_edges = {}
        for name in regions:
            edge_profile = profile['material_edges'][name]
            edges = np.asarray(edge_profile['indices'])
            lengths = np.asarray(edge_profile['rest_lengths_m'], dtype=float)
            if (edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0
                    or not np.issubdtype(edges.dtype, np.integer)
                    or np.any(edges < 0) or np.any(edges >= count)
                    or lengths.shape != (len(edges),)
                    or not np.isfinite(lengths).all() or np.any(lengths <= 0)):
                raise ValueError(f'invalid material edges: {name}')
            material_edges[name] = (edges, lengths)
        limits = profile['thresholds']
        required = ('core_width_min_m', 'core_width_max_m',
                    'core_length_min_m', 'core_length_max_m',
                    'side_inward_q90_max_m', 'side_overshoot_q10_min_m',
                    'side_height_q90_max_m', 'final_hem_collar_q90_max_m',
                    'final_height_max_m', 'final_area_min_m2',
                    'material_edge_median_ratio_min', 'material_edge_median_ratio_max')
        optional = ('final_width_max_m', 'final_length_max_m')
        for key in required + tuple(key for key in optional if key in limits):
            value = limits[key]
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value)):
                raise ValueError(f'fold threshold must be a finite real scalar: {key}')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('invalid Wuji fold profile') from exc

    points = np.asarray(points, dtype=float)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError(f'expected finite world points with shape ({count}, 3)')

    means = {name: points[index].mean(axis=0) for name, index in regions.items()}
    lateral = means['core_right'][:2] - means['core_left'][:2]
    core_width = float(np.linalg.norm(lateral))
    # A degenerate anchor is a geometric negative, not missing/corrupt input.
    x_axis = lateral / core_width if core_width > 1e-12 else np.array([1., 0.])
    y_axis = np.array([-x_axis[1], x_axis[0]])
    center = (means['core_left'] + means['core_right']) / 2
    local_x = (points[:, :2] - center[:2]) @ x_axis
    local_y = (points[:, :2] - means['collar'][:2]) @ y_axis
    local_z = points[:, 2] - center[2]
    core_length = float(np.mean(local_y[regions['core_mid']]))
    core_valid = (limits['core_width_min_m'] <= core_width <= limits['core_width_max_m']
                  and limits['core_length_min_m'] <= core_length <= limits['core_length_max_m'])
    diagnostics = {'core_width_m': core_width, 'core_length_m': core_length,
                   'core_valid': float(core_valid)}

    material_valid = {}
    for name, (edges, rest_lengths) in material_edges.items():
        lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        ratio = float(np.median(lengths / rest_lengths))
        diagnostics[f'{name}_edge_median_ratio'] = ratio
        material_valid[name] = (limits['material_edge_median_ratio_min'] <= ratio
                                <= limits['material_edge_median_ratio_max'])
    core_valid = core_valid and all(material_valid[name] for name in
                                    ('collar', 'core_left', 'core_right', 'core_mid'))
    diagnostics['core_valid'] = float(core_valid)

    result = {}
    for side, sign in (('left', -1), ('right', 1)):
        side_valid = core_valid
        # Each part must lie in the central torso corridor. A shoulder alone, or
        # an inward hem alone, can never satisfy the complete side predicate.
        for part in ('shoulder', 'hem'):
            name = f'{side}_{part}'
            indices = regions[name]
            q10, q90 = np.quantile(sign * local_x[indices], [.1, .9])
            height = float(np.quantile(np.abs(local_z[indices]), .9))
            diagnostics[f'{name}_inward_q90_m'] = float(q90)
            diagnostics[f'{name}_overshoot_q10_m'] = float(q10)
            diagnostics[f'{name}_height_q90_m'] = height
            side_valid = (side_valid and material_valid[name] and q90 <= limits['side_inward_q90_max_m']
                          and q10 >= limits['side_overshoot_q10_min_m']
                          and height <= limits['side_height_q90_max_m'])
        result[side] = bool(side_valid)

    width = float(np.ptp(local_x))
    length = float(np.ptp(local_y))
    height = float(np.ptp(local_z))
    hem_distance = float(np.quantile(np.linalg.norm(
        points[regions['hem_center']] - means['collar'], axis=1), .9))
    diagnostics.update({'width_m': width, 'length_m': length, 'height_m': height,
                        'footprint_area_m2': width * length,
                        'hem_collar_q90_m': hem_distance})
    # These are present-tense shape conditions. The caller must still reject an
    # early final fold and enforce a new fold after the side-stage unlock frame.
    result['final'] = bool(
        result['left'] and result['right'] and material_valid['hem_center']
        and hem_distance <= limits['final_hem_collar_q90_max_m']
        and ('final_width_max_m' not in limits or width <= limits['final_width_max_m'])
        and ('final_length_max_m' not in limits or length <= limits['final_length_max_m'])
        and height <= limits['final_height_max_m']
        and width * length >= limits['final_area_min_m2'])
    result['diagnostics'] = diagnostics
    return result
