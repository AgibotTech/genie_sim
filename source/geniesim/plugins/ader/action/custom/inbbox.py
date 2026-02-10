# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
import numpy as np
from geniesim.plugins.logger import Logger

logger = Logger()


class InBBox(EvaluateAction):
    """
    Check if an object is inside a specified bounding box.

    Input format: "object_id|center_x,center_y,center_z|len_x,len_y,len_z"
    - object_id: the object to check
    - center_x,center_y,center_z: the center of the bounding box
    - len_x,len_y,len_z: the size of the bounding box in each axis

    The check passes if the object remains inside the bbox for 2 consecutive frames.
    """

    def __init__(self, env, obj_id, bbox_center, bbox_size):
        super().__init__(env)
        self.obj_id = obj_id
        self.bbox_center = np.array(bbox_center, dtype=np.float64)
        self.bbox_size = np.array(bbox_size, dtype=np.float64)
        self._done_flag = False
        self._pass_frame = 0

        # Calculate bbox min and max
        self.bbox_min = self.bbox_center - self.bbox_size / 2
        self.bbox_max = self.bbox_center + self.bbox_size / 2

    def update(self, delta_time: float) -> float:
        # Find the correct object path (supporting background scene objects)
        # obj_path = self._find_obj_path(self.obj_id)
        obj_path = self._analyze_obj_name(self.obj_id)
        logger.info(f"[InBBox] obj_path: {obj_path}")
        pose = self.api_core.get_obj_world_pose_matrix(obj_path)
        pos = pose[:3, 3].reshape(-1)

        # Check if position is inside bbox
        is_inside = self.aabb_contains_point(pos, (self.bbox_min, self.bbox_max))

        # Debug info - check if pose is valid (not all zeros)
        pose_is_zero = np.allclose(pos, [0, 0, 0], atol=1e-6)
        pose_debug = "INVALID (all zeros - object not found?)" if pose_is_zero else "VALID"

        logger.info(
            f"[InBBox Check] obj={self.obj_id}\n"
            f"  Lookup path: {obj_path}\n"
            f"  Pose status: {pose_debug}\n"
            f"  Object position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n"
            f"  Full pose matrix:\n{pose}\n"
            f"  BBox center: [{self.bbox_center[0]:.4f}, {self.bbox_center[1]:.4f}, {self.bbox_center[2]:.4f}]\n"
            f"  BBox size: [{self.bbox_size[0]:.4f}, {self.bbox_size[1]:.4f}, {self.bbox_size[2]:.4f}]\n"
            f"  BBox min: [{self.bbox_min[0]:.4f}, {self.bbox_min[1]:.4f}, {self.bbox_min[2]:.4f}]\n"
            f"  BBox max: [{self.bbox_max[0]:.4f}, {self.bbox_max[1]:.4f}, {self.bbox_max[2]:.4f}]\n"
            f"  Is inside: {is_inside}, pass_frame: {self._pass_frame}"
        )

        if is_inside:
            self._pass_frame += 1
        else:
            self._pass_frame = 0

        # Need 2 consecutive frames to pass
        if self._pass_frame > 1:
            self._done_flag = True

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [InBBox] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
