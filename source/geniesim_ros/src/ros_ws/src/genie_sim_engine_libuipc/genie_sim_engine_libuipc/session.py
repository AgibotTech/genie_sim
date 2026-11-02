# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bind the existing core session to the independent libuipc backend."""

import numpy as np
from genie_sim_engine_core.common.session import EngineSession, SimpleLogger
from .engine import LibuipcEngine  # noqa: F401 -- register only this backend


class LibuipcSession(EngineSession):
    def _snapshot_body_transforms(self, stage, body_paths):
        from pxr import UsdGeom

        cache = UsdGeom.XformCache()
        poses = np.zeros((len(body_paths), 7), dtype=float)
        poses[:, 3] = 1
        for row, path in enumerate(body_paths):
            prim = stage.GetPrimAtPath(path)
            if prim:
                matrix = cache.GetLocalToWorldTransform(prim)
                pos = matrix.ExtractTranslation()
                rotation = matrix.ExtractRotationQuat()
                poses[row] = [*pos, rotation.GetReal(), *rotation.GetImaginary()]
        return poses, list(body_paths)
