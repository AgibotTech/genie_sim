# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Merge independent joint groups into each simulator command frame."""

import math


class JointCommandMux:
    def __init__(self, timeout_s=0.3):
        self.timeout_s = timeout_s
        self.sources = {}

    def update(self, source, names, positions, velocities, efforts, now):
        names = list(names)
        arrays = [list(values) for values in (positions, velocities, efforts)]
        if not names or len(set(names)) != len(names):
            raise ValueError("Joint command names must be nonempty and unique")
        if any(values and len(values) != len(names) for values in arrays):
            raise ValueError("Joint command fields must be empty or match the joint names")
        for other, (_, joints) in self.sources.items():
            if other != source and set(names).intersection(joints):
                raise ValueError(f"Joint command ownership conflict: {source} and {other}")
        values = [values or [math.nan] * len(names) for values in arrays]
        if any(math.isinf(value) for field in values for value in field):
            raise ValueError("Infinite joint commands are invalid")
        self.sources[source] = (now, dict(zip(names, zip(*values))))

    def snapshot(self, now, playback=False):
        names, positions, velocities, efforts = [], [], [], []
        for source, (stamp, joints) in self.sources.items():
            # Playback owns the upper body/grippers, while chassis must stop.
            if playback and source != "chassis":
                continue
            stale = now - stamp >= self.timeout_s
            for name, (position, velocity, effort) in joints.items():
                names.append(name)
                positions.append(position)
                velocities.append(0.0 if (stale or playback) and math.isfinite(velocity) else velocity)
                efforts.append(0.0 if stale and math.isfinite(effort) else effort)
        return names, positions, velocities, efforts
