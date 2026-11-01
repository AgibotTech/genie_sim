# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Validated opt-in command; legacy SetGripperState semantics are unchanged."""
import math
from numbers import Real


def make_gripper_command(*, arm, opening_ratio, speed, max_effort,
                         mode='position', stiffness=None, damping=None,
                         tolerance=.001, timeout=5., minimum_duration=0.,
                         completion='target', stall_time=.2):
    """Use native actuator units: m/N for linear, rad/Nm for angular drives."""
    if arm not in ('left', 'right'):
        raise ValueError('arm must be left or right')
    if mode not in ('position', 'velocity'):
        raise ValueError('mode must be position or velocity')
    if completion not in ('target', 'settled'):
        raise ValueError('completion must be target or settled')
    values = dict(opening_ratio=opening_ratio, speed=speed, max_effort=max_effort,
                  tolerance=tolerance, timeout=timeout, minimum_duration=minimum_duration,
                  stall_time=stall_time, stiffness=stiffness, damping=damping)
    for key, value in values.items():
        if value is None and key in ('stiffness', 'damping'):
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
        values[key] = float(value)
    if not 0 <= values['opening_ratio'] <= 1:
        raise ValueError('opening_ratio must be in [0, 1]')
    for key in ('speed', 'max_effort', 'tolerance', 'timeout', 'stall_time'):
        if values[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 <= values['minimum_duration'] <= values['timeout']:
        raise ValueError('minimum_duration must be between zero and timeout')
    if any(values[k] is not None and values[k] < 0 for k in ('stiffness', 'damping')):
        raise ValueError('stiffness and damping must be nonnegative')
    return dict(command='ControlGripper', arm=arm, mode=mode, completion=completion, **values)
