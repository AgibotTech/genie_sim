# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import ast
import json
import os
import time

import numpy as np

from geniesim_benchmark.plugins.ader.action.common_actions import (
    ActionBase,
    ActionEvent,
    EvaluateAction,
)
from geniesim_benchmark.plugins.logger import Logger

from .precise_geometry import rigid_pose

logger = Logger()


class Inside(EvaluateAction):
    """Check active object's center is inside passive object's AABB scaled by
    `scale`: a scalar, or a [sx,sy,sz] list for per-axis scaling (e.g. a large
    sz for flat containers like plates, whose AABB is too thin to contain the
    center of an object resting on them). An optional dictionary with `scale`
    and `height_from_bottom` caps only the world-Z upper bound, retaining the
    scaled lower and horizontal bounds. Height is measured in metres.

    An explicit ``{"frame": "reference_local", "lower": [...], "upper": [...]}``
    region instead tests the active origin in the passive object's rigid frame.
    Bounds are task-supplied metres along that frame's axes; no world AABB is
    constructed for this mode. Existing scalar/list/height regions stay in world
    coordinates. Neither mode adds release or stability requirements.
    """

    def __init__(self, env, active_obj, passive_obj, scale):
        super().__init__(env)
        self._holder_name, self._active_obj = self.placeholder_sparser(active_obj)
        self.passive_obj = passive_obj
        self._done_flag = False
        self._pass_frame = 0
        # Opt-in diagnostics only; the selector can be an object name or '*'.
        self._trace_object = os.environ.get("GENIESIM_INSIDE_TRACE_OBJECT", "")
        self._trace_update_index = 0
        self._height_from_bottom = None
        self._local_bounds = None
        if isinstance(scale, dict) or isinstance(scale, str) and scale.strip().startswith("{"):
            region = json.loads(scale) if isinstance(scale, str) else scale
            if "frame" in region:
                if set(region) != {"frame", "lower", "upper"} or region["frame"] != "reference_local":
                    raise ValueError("Inside local region requires frame=reference_local, lower and upper")
                lower, upper = (
                    np.asarray(region["lower"], dtype=float),
                    np.asarray(region["upper"], dtype=float),
                )
                if (
                    lower.shape != (3,)
                    or upper.shape != (3,)
                    or not np.isfinite(lower).all()
                    or not np.isfinite(upper).all()
                    or np.any(lower >= upper)
                ):
                    raise ValueError("Inside local bounds must be finite increasing three-vectors")
                self._local_bounds = (lower, upper)
                self.scale = 1.0
            elif set(region) != {"scale", "height_from_bottom"}:
                raise ValueError("Inside height region requires scale and height_from_bottom")
            else:
                scale = np.asarray(region["scale"], dtype=float)
                height = float(region["height_from_bottom"])
                if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                    raise ValueError("Inside height region scale must contain three finite positive values")
                if not np.isfinite(height) or height <= 0:
                    raise ValueError("Inside height_from_bottom must be finite and positive")
                self.scale = scale
                self._height_from_bottom = height
        elif isinstance(scale, str) and scale.strip().startswith("["):
            self.scale = np.array(ast.literal_eval(scale), dtype=float)
        else:
            self.scale = float(scale)

    @property
    def active_obj(self) -> str:
        if self._holder_name:
            return getattr(self, self._active_obj)
        return self._active_obj

    def update(self, delta_time: float) -> float:
        if not self.is_running() or not self.claim_observation():
            return 0.0

        pose_A = self.get_obj_pose(self.active_obj)
        pos_A = pose_A[:3, 3].reshape(
            -1,
        )
        if self._local_bounds is not None:
            # Read the current body pose every update: the bucket can translate
            # and tilt independently of its authored root Xform.
            frame = np.asarray(self.get_obj_pose(self.passive_obj), dtype=float)
            valid = rigid_pose(frame) and np.isfinite(pos_A).all()
            local = np.linalg.solve(frame, np.append(pos_A, 1.0))[:3] if valid else None
            inside = valid and self.aabb_contains_point(local, self._local_bounds)
            geometry = {
                "frame": "reference_local",
                "reference_pose_world": frame.tolist(),
                "object_origin_local": None if local is None else local.tolist(),
                "region_local_lower": self._local_bounds[0].tolist(),
                "region_local_upper": self._local_bounds[1].tolist(),
                "invalid_pose": not valid,
            }
        else:
            self.aa_B, self.bb_B = self.get_obj_aabb_new(self.passive_obj)
            mid = (self.aa_B + self.bb_B) / 2
            rescaled_size = (self.bb_B - self.aa_B) * self.scale
            aa_new = mid - rescaled_size / 2
            bb_new = mid + rescaled_size / 2
            if self._height_from_bottom is not None:
                bb_new[2] = min(bb_new[2], aa_new[2] + self._height_from_bottom)
            inside = self.aabb_contains_point(pos_A, (aa_new, bb_new))
            geometry = {
                "aabb_world_lower": self.aa_B.tolist(),
                "aabb_world_upper": self.bb_B.tolist(),
                "region_world_lower": aa_new.tolist(),
                "region_world_upper": bb_new.tolist(),
            }
        if inside:
            self._pass_frame += 1
        else:
            self._pass_frame = 0
        if self._pass_frame > 1:
            self._done_flag = True
        if self._trace_object in ("*", self.active_obj):
            self._trace_update_index += 1
            logger.info(
                "[InsideTrace] "
                + json.dumps(
                    {
                        "active_obj": self.active_obj,
                        "passive_obj": self.passive_obj,
                        "update_index": self._trace_update_index,
                        "env_step": getattr(self.env, "current_step", None),
                        "monotonic_s": time.monotonic(),
                        "delta_time_s": float(delta_time),
                        "object_origin_world": pos_A.tolist(),
                        "scale": np.asarray(self.scale).tolist(),
                        "height_from_bottom": self._height_from_bottom,
                        **geometry,
                        "inside": bool(inside),
                        "consecutive_updates": self._pass_frame,
                        "done": self._done_flag,
                    },
                    sort_keys=True,
                )
            )
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [Inside] active_obj: {self.active_obj}, evt: {event.value}")

        if event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
