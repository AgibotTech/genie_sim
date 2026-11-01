# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Project-specific posture acceptance, not a universal anthropomorphism score."""
import numpy as np
import mujoco

LEFT_HOME_DEG = np.array([55, -65, -70, -60, 60, 0, 0], dtype=float)
MIRROR_SIGNS = np.array([-1, 1, -1, 1, -1, 1, -1])
RIGHT_HOME = np.deg2rad(LEFT_HOME_DEG * MIRROR_SIGNS)
UPPER_LIMIT_DEG = 30.0

def limits(model):
    j = [model.joint(f"Joint{i}_R").id for i in range(1, 8)]
    bounds = model.jnt_range[j].copy()
    bounds[:4, 0] = np.maximum(bounds[:4, 0], RIGHT_HOME[:4] - np.deg2rad(UPPER_LIMIT_DEG))
    bounds[:4, 1] = np.minimum(bounds[:4, 1], RIGHT_HOME[:4] + np.deg2rad(UPPER_LIMIT_DEG))
    bounds[4:, 0] = np.maximum(bounds[4:, 0], RIGHT_HOME[4:] - np.deg2rad(120))
    bounds[4:, 1] = np.minimum(bounds[4:, 1], RIGHT_HOME[4:] + np.deg2rad(120))
    return bounds

def measure(model, states, fps=20, observed=None):
    right = [model.joint(f"Joint{i}_R").id for i in range(1, 8)]
    left = [model.joint(f"Joint{i}_L").id for i in range(1, 8)]
    q = np.asarray(states)[:, model.jnt_qposadr[right]]
    delta = np.rad2deg(q - RIGHT_HOME)
    d = mujoco.MjData(model)
    heights = []
    for state in states:
        d.qpos[:] = state
        mujoco.mj_kinematics(model, d)
        heights.append(float(d.xanchor[right[3], 2] - d.xanchor[right[0], 2]))
    d.qpos[:] = states[0]; mujoco.mj_kinematics(model, d)
    mirror = np.max(np.linalg.norm(d.xanchor[right] - d.xanchor[left] * [1,-1,1], axis=1)) * 1000
    speed = np.diff(np.rad2deg(q), axis=0) * fps
    acceleration = np.diff(speed, axis=0) * fps
    peak = np.max(np.abs(delta), axis=0)
    peak_speed = np.max(np.abs(speed),axis=0) if len(speed) else np.zeros(7)
    peak_accel = np.max(np.abs(acceleration),axis=0) if len(acceleration) else np.zeros(7)
    if observed:
        peak = np.maximum(peak, observed['deviation'])
        peak_speed = np.maximum(peak_speed, observed['speed'])
        peak_accel = np.maximum(peak_accel, observed['acceleration'])
        heights.append(observed['elbow'])
    errors = []
    if mirror > 1: errors.append('Initial right/left joint anchors are not mirrored within 1 mm')
    if np.any(peak[:4] > UPPER_LIMIT_DEG + 2): errors.append('J1-J4 exceed 30 degree envelope plus 2 degree tracking tolerance')
    if np.any(peak[4:] > 122): errors.append('Wrist joint exceeds 120 degree envelope plus tracking tolerance')
    if max(heights) > -.03: errors.append('Elbow rises within 30 mm of shoulder height')
    if np.max(peak_speed) > 75: errors.append('Actual joint speed exceeds 75 deg/s')
    return {'definition': 'Mirror left home; constrain J1-J4; keep elbow below shoulder; limit actual joint speed.',
            'rms_sample_rate_hz': fps, 'peak_sample_rate_hz': 1/model.opt.timestep if observed else fps, 'upper_joint_limit_deg': UPPER_LIMIT_DEG,
            'reference_right_degrees': np.rad2deg(RIGHT_HOME).tolist(),
            'initial_mirror_error_mm': float(mirror), 'max_joint_deviation_deg': peak.tolist(),
            'rms_joint_deviation_deg': np.sqrt(np.mean(delta**2, axis=0)).tolist(),
            'max_elbow_above_shoulder_m': max(heights),
            'max_joint_speed_deg_s': peak_speed.tolist(),
            'max_joint_acceleration_deg_s2': peak_accel.tolist(),
            'passed': not errors, 'errors': errors}
