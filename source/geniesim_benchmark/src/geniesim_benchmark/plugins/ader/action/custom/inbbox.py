# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim_benchmark.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
import numpy as np
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


class InBBox(EvaluateAction):
    """
    Check if an object is inside a specified bounding box.

    Input format: "object_id|center_x,center_y,center_z|len_x,len_y,len_z"
    - object_id: the object to check. Three forms are supported:
        * a single prim name ("obj1") — legacy single-object mode;
        * a comma-separated candidate list ("obj1,obj2,...") — passes when
          ANY candidate enters the bbox; the matched candidate is written
          to the env placeholder_str1 so downstream checkers (StableGrasp,
          ...) can follow the SAME object (PickUpOnGripper pattern);
        * a placeholder ("{@placeholder_str1}") — resolved at runtime to the
          object selected by an earlier checker in the tree.
    - center_x,center_y,center_z: the center of the bounding box
    - len_x,len_y,len_z: the size of the bounding box in each axis

    The check passes if an object remains inside the bbox for 2 consecutive
    frames. In multi-object mode the frames must be consecutive on the SAME
    candidate.
    """

    def __init__(self, env, obj_id, bbox_center, bbox_size):
        super().__init__(env)
        # Placeholder support: "{@placeholder_str1}" defers resolution to
        # runtime; plain names (single or comma list) resolve immediately.
        self._holder_name, self._obj_name = self.placeholder_sparser(obj_id)
        self._obj_candidates = None
        if not self._holder_name:
            self._obj_candidates = self.parse_obj_input(obj_id)

        self.bbox_center = np.array(bbox_center, dtype=np.float64)
        self.bbox_size = np.array(bbox_size, dtype=np.float64)
        self._done_flag = False
        self._pass_frame = 0
        self._matched_obj = None  # candidate currently/provisionally matched

        # Calculate bbox min and max
        self.bbox_min = self.bbox_center - self.bbox_size / 2
        self.bbox_max = self.bbox_center + self.bbox_size / 2

    @property
    def obj_id(self) -> str:
        """The object id this check currently resolves to.

        Placeholder mode returns the placeholder value once set; before that
        it returns the raw placeholder spec (single-object callers only read
        it for logging, matching the pre-existing behaviour).
        """
        if self._holder_name:
            resolved = getattr(self, self._obj_name)
            return resolved if resolved else self._obj_name
        return self._obj_name

    @property
    def obj_name_list(self) -> list:
        """Candidate list: placeholder value if set, else the parsed list."""
        if self._holder_name:
            resolved = getattr(self, self._obj_name)
            if resolved:
                return [resolved]
            return []
        return self._obj_candidates or [self._obj_name]

    def _check_candidate(self, obj_name) -> bool:
        obj_path = self._analyze_obj_name(obj_name)
        pose = self.api_core.get_obj_world_pose_matrix(obj_path)
        pos = np.asarray(pose)[:3, 3].reshape(-1)

        is_inside = self.aabb_contains_point(pos, (self.bbox_min, self.bbox_max))

        # Debug info - check if pose is valid (not all zeros)
        pose_is_zero = np.allclose(pos, [0, 0, 0], atol=1e-6)
        pose_debug = "INVALID (all zeros - object not found?)" if pose_is_zero else "VALID"

        logger.info(
            f"[InBBox Check] obj={obj_name}\n"
            f"  Lookup path: {obj_path}\n"
            f"  Pose status: {pose_debug}\n"
            f"  Object position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n"
            f"  BBox center: [{self.bbox_center[0]:.4f}, {self.bbox_center[1]:.4f}, {self.bbox_center[2]:.4f}]\n"
            f"  BBox size: [{self.bbox_size[0]:.4f}, {self.bbox_size[1]:.4f}, {self.bbox_size[2]:.4f}]\n"
            f"  BBox min: [{self.bbox_min[0]:.4f}, {self.bbox_min[1]:.4f}, {self.bbox_min[2]:.4f}]\n"
            f"  BBox max: [{self.bbox_max[0]:.4f}, {self.bbox_max[1]:.4f}, {self.bbox_max[2]:.4f}]\n"
            f"  Is inside: {is_inside}, pass_frame: {self._pass_frame}"
        )
        return is_inside

    def update(self, delta_time: float) -> float:
        candidates = self.obj_name_list
        if not candidates:
            # Placeholder not yet populated: nothing to check this tick.
            return super().update(delta_time)

        # Unified path (a single configured object is the one-candidate
        # case): ANY candidate inside; the 2 consecutive frames must be on
        # the SAME candidate — behaviour identical to the legacy
        # single-object implementation for one-candidate configs.
        matched = self.find_matching_object(candidates, self._check_candidate)
        if matched is not None:
            if matched == self._matched_obj:
                self._pass_frame += 1
            else:
                self._matched_obj = matched
                self._pass_frame = 1
            if self._pass_frame > 1:
                self._done_flag = True
                # Publish ONLY in true multi-candidate mode: single-object
                # configs (hold_pot, ...) must not clobber placeholders
                # owned by other checkers (Follow/PickUpOnGripper), and a
                # placeholder-READ config publishes nothing new (it would
                # only echo the value it resolved from).
                if self._obj_candidates is not None and len(self._obj_candidates) > 1:
                    self.env.update_place_holder("placeholder_str1", matched)
        else:
            self._pass_frame = 0
            self._matched_obj = None

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
