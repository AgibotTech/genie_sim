# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np

from client.planner.action.place import PlaceStage
from client.planner.action.stage import Action, ActionSequence
from common.base_utils.logger import logger


class InsertStage(PlaceStage):
    """
    InsertStage inherits from PlaceStage.
    The difference is that it adds a pre-placement point perpendicular to the placement point before final placement.
    """

    def __init__(self, stage_config, objects):
        super().__init__(stage_config, objects)
        # Get pre-placement point offset from extra_params
        self.pre_insert_offset = self.extra_params.get("pre_insert_offset", 0.1)
        self.use_pre_place = True

    def generate_action_sequence(self, grasp_pose, pre_insert_pose=None):
        """
        Generate action sequence, adding a pre-placement point perpendicular to the placement point before final placement.

        Args:
            grasp_pose: 4x4 transformation matrix of target placement point

        Returns:
            ActionSequence: Action sequence
        """
        action_sequence = ActionSequence()
        target_pose_canonical = grasp_pose
        if pre_insert_pose is None:
            logger.error("failed to get pre insert pose")
            return None
        # Get gripper command
        gripper_cmd = self.extra_params.get("gripper_state", "open")
        error_type = ""
        if "error_data" in self.extra_params:
            error_data = self.extra_params["error_data"]
            error_data.get("params", {})
            error_type = error_data.get("type", "")
            if error_type == "KeepClose":
                gripper_cmd = None

        post_place_action = self.extra_params.get("post_place_action", None)

        # Add pre-placement point action (no gripper operation, use AvoidObs to avoid collision)
        action_sequence.add_action(
            Action(
                pre_insert_pose,
                None,
                np.eye(4),
                "AvoidObs",
                extra_params={"skip_check": True},
            )
        )

        # Add final placement action
        # Move to final placement position
        place_transform_up = np.eye(4)
        place_transform_up[:3, 3] = self.place_transform_up

        if post_place_action is not None:
            action_sequence.add_action(
                Action(target_pose_canonical, None, place_transform_up, "Simple")
            )
            post_place_distance = post_place_action.get("distance", 0.02)
            post_place_direction = np.array(post_place_action.get("direction", [0, 0, 1]))
            target_pose_canonical = target_pose_canonical.copy()
            target_pose_canonical[:3, 3] += (
                post_place_direction * post_place_distance / np.linalg.norm(post_place_direction)
            )
            action_sequence.add_action(
                Action(target_pose_canonical, gripper_cmd, np.eye(4), "simple")
            )
        else:
            action_sequence.add_action(
                Action(
                    target_pose_canonical,
                    gripper_cmd,
                    np.eye(4),
                    "AvoidObs",
                )
            )

        return action_sequence
