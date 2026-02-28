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
import ast


logger = Logger()


class Follow(EvaluateAction):
    """Evaluate whether the gripper is within the bounding box of a target object.

    Supports both single object and multiple objects (comma-separated) for obj_name.
    When multiple objects are provided, the check passes if the gripper is within
    ANY one of the objects' bounding boxes.

    Args:
        env: The simulation environment.
        obj_name: Object name or comma-separated object names to check.
            Examples: "obj1" or "obj1,obj2,obj3"
        bbox: Bounding box size string, e.g. "[0.1,0.1,0.1]".
        gripper_id: Gripper identifier, e.g. "right" or "left".
    """

    def __init__(self, env, obj_name, bbox, gripper_id):
        super().__init__(env)
        # Parse obj_name: supports single object, placeholder, or comma-separated list
        self._holder_name, self._obj_name = self.placeholder_sparser(obj_name)
        self._obj_name_list = self.parse_obj_input(obj_name) if not self._holder_name else None
        self._done_flag = False
        self.bbox = ast.literal_eval(bbox)
        self.env = env
        self._holder_id, self._gripper_id = self.placeholder_sparser(gripper_id)
        self._matched_obj = None  # Track which object satisfied the condition
        self._update_count = 0  # Debug counter to track update calls

        # Initialize a global set on env to track already-followed objects across rounds
        if not hasattr(self.env, '_followed_objects'):
            self.env._followed_objects = set()

        logger.info(
            f"[Follow] Initialized with obj_name_list={self._obj_name_list}, "
            f"bbox={self.bbox}, gripper_id={self._gripper_id}, "
            f"already_followed={self.env._followed_objects}"
        )

    @property
    def obj_name(self) -> str:
        if self._holder_name:
            return getattr(self, self._obj_name)
        return self._obj_name

    @property
    def obj_name_list(self) -> list:
        """Return the resolved list of object names to check, excluding already-followed objects."""
        if self._holder_name:
            # Placeholder mode: resolve the single placeholder value
            return [getattr(self, self._obj_name)]
        # Filter out objects that have already been followed in previous rounds
        followed = getattr(self.env, '_followed_objects', set())
        filtered = [name for name in self._obj_name_list if name not in followed]
        return filtered

    @property
    def gripper_id(self) -> str:
        if self._holder_id:
            return getattr(self, self._gripper_id)
        return self._gripper_id

    def _get_gripper_position(self):
        """Get the current gripper world position.

        Returns:
            The gripper position as a numpy array.
        """
        robot_cfg = getattr(self.env, "robot_cfg", None)
        if robot_cfg is None:
            robot_cfg = getattr(self.env, "init_task_config", {}).get("robot_cfg", "G2_omnipicker")
        if "G1" in robot_cfg:
            robot_base = "/G1"
        else:
            robot_base = "/genie"
        link_prim_path = (
            f"{robot_base}/gripper_r_center_link"
            if "right" in self.gripper_id
            else f"{robot_base}/gripper_l_center_link"
        )
        g_pos, _ = self.get_world_pose(link_prim_path)
        return g_pos

    def _check_gripper_follows_obj(self, obj_name):
        """Check if the gripper is within the bounding box of the given object.

        Args:
            obj_name: Name of the object to check against.

        Returns:
            True if the gripper position is inside the object's bounding box.
        """
        aa, bb = self.get_obj_aabb(obj_name, self.bbox)
        return self.aabb_contains_point(self._current_gripper_pos, (aa, bb))

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0

        self._update_count += 1

        try:
            # Cache gripper position for this frame (avoid re-querying per object)
            self._current_gripper_pos = self._get_gripper_position()

            # Check if gripper follows any of the target objects (excluding already-followed)
            candidates = self.obj_name_list
            if not candidates:
                logger.warning("[Follow] No remaining candidates (all objects already followed)")
                return super().update(delta_time)

            matched = self.find_matching_object(candidates, self._check_gripper_follows_obj)
            if matched is not None:
                self._done_flag = True
                self._matched_obj = matched
                self.env._followed_objects.add(matched)
                self.env.update_place_holder("placeholder_str1", matched)
                logger.info(f"[Follow] Gripper is following object '{matched}' (followed so far: {self.env._followed_objects})")

        except Exception as e:
            logger.warning(f"[Follow] Error during update #{self._update_count}: {e}")

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [Follow] {self.obj_name} evt: {event.value}")
        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
