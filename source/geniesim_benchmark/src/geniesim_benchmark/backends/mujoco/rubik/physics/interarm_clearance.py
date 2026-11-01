# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Explicit cross-arm geometry checks, independent of SRDF exclusions.

Only reads kinematic geometry transforms. Does not change collision masks,
exclusions, geometry, or live dynamics state.
"""
import mujoco
import numpy as np
class InterArmClearance:
    def __init__(self, model):
        self.m = model
        labels = np.zeros(model.nbody, dtype=int)
        for side, label in [('L', 1), ('R', 2)]:
            root = model.body('Link1_' + side).id
            for body in range(model.nbody):
                parent = body
                while parent and parent != root:
                    parent = model.body_parentid[parent]
                if parent == root:
                    labels[body] = label
        active = (model.geom_contype | model.geom_conaffinity) != 0
        self.a = np.flatnonzero(active & (labels[model.geom_bodyid] == 1))
        self.b = np.flatnonzero(active & (labels[model.geom_bodyid] == 2))
        self.radius = model.geom_rbound[self.a, None] + model.geom_rbound[self.b[None, :]]

    def minimum(self, data, cutoff=.003):
        delta = data.geom_xpos[self.a, None, :] - data.geom_xpos[self.b[None, :], :]
        near = np.sum(delta * delta, axis=2) <= (self.radius + max(cutoff, 0.)) ** 2
        minimum, pair = cutoff, None
        for i, j in zip(*np.where(near)):
            a, b = int(self.a[i]), int(self.b[j])
            gap = mujoco.mj_geomDistance(self.m, data, a, b, max(cutoff, 0.), None)
            if gap < minimum:
                minimum, pair = float(gap), (a, b)
        return minimum, pair

    def clear(self, data, clearance=.001):
        return self.minimum(data, max(clearance, 0.) + 1e-8)[0] >= clearance
