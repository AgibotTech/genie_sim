# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import uuid as lib_uuid
from abc import ABC
from queue import Queue
from typing import Any, Dict, List, Optional

import numpy as np

from common.base_utils.logger import logger
from common.base_utils.transform_utils import pose_difference

PLACE_LIKE_ACTIONS = ["place", "insert"]


def simple_check_completion(
    goal,
    objects,
    last_statement=None,
    pos_threshold=0.1,
    angle_threshold=70,
    is_grasped=False,
):
    (
        active_obj_id,
        passive_obj_id,
        target_pose_canonical,
        gripper_action,
        transform_world,
        motion_type,
        extra_action_params,
    ) = goal
    if target_pose_canonical is None:
        return True
    if gripper_action == "open":
        return True
    skip_check = extra_action_params.get("skip_check", False)
    if skip_check:
        logger.warning("skip check completion")
        return True
    goal_offset = extra_action_params.get("goal_offset", [0, 0, 0, 1, 0, 0, 0])
    if goal_offset != [0, 0, 0, 1, 0, 0, 0]:
        logger.info("currently simple_check_completion is not supported with goal_offset")
        return True
    current_pose_world = objects[active_obj_id].obj_pose
    if len(target_pose_canonical.shape) == 3:
        target_pose_canonical = target_pose_canonical[-1]
        transform_world = transform_world[-1]
    target_pose_world = objects[passive_obj_id].obj_pose @ target_pose_canonical
    if not is_grasped:
        target_pose_world = np.dot(transform_world, target_pose_world)

    pos_diff, angle_diff = pose_difference(current_pose_world, target_pose_world)
    success = (pos_diff < pos_threshold) and (angle_diff < angle_threshold)
    return success


def solve_target_gripper_pose(stage, objects, extra_params={}):
    (
        active_obj_ID,
        passive_obj_ID,
        target_pose_canonical,
        gripper_action,
        transform_world,
        motion_type,
    ) = stage
    if extra_params.get("use_world_pose", False):
        target_pose = transform_world @ target_pose_canonical
        return target_pose
    anchor_pose = objects[passive_obj_ID].obj_pose
    if motion_type == "Trajectory":
        assert len(target_pose_canonical.shape) == 3, "The target_pose should be a list of poses"
        target_pose = anchor_pose[np.newaxis, ...] @ target_pose_canonical
        target_pose = transform_world @ target_pose
    else:
        target_pose = anchor_pose @ target_pose_canonical
        target_pose = transform_world @ target_pose
    assert "gripper" in objects, "The gripper should be the first one in the object list"
    current_gripper_pose = objects["gripper"].obj_pose

    if active_obj_ID == "gripper":
        target_gripper_pose = target_pose
    else:
        current_obj_pose = objects[active_obj_ID].obj_pose
        gripper2obj = np.linalg.inv(current_obj_pose) @ current_gripper_pose
        if len(target_pose.shape) == 3:
            gripper2obj = gripper2obj[np.newaxis, ...]

        target_obj_pose = target_pose
        target_gripper_pose = target_obj_pose @ gripper2obj
    return target_gripper_pose


class Action:
    def __init__(self, grasp_pose, gripper_action, transform_world, motion_type, extra_params={}):
        self.grasp_pose = grasp_pose
        self.gripper_action = gripper_action
        self.transform_world = transform_world
        self.motion_type = motion_type
        self.extra_params = extra_params

    def unpack_as_list(self) -> List[Any]:
        return [
            self.grasp_pose,
            self.gripper_action,
            self.transform_world,
            self.motion_type,
            self.extra_params,
        ]


class ActionSequence:
    def __init__(self, actions: List[Action] = None):
        self.actions = list(actions) if actions is not None else []
        self.action_index = -1  # Current action index
        self.parent_stage = None  # Parent stage

    def set_parent_stage(self, stage: "Stage"):
        self.parent_stage = stage

    def add_action(self, action: Action):
        self.actions.append(action)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index) -> Action:
        return self.actions[index]

    def __iter__(self):
        return self

    def __next__(self) -> Action:
        if self.action_index < len(self.actions) - 1:
            self.action_index += 1
            action = self.actions[self.action_index]
            return action
        raise StopIteration

    def get_current_action(self) -> Optional[Action]:
        if self.action_index < len(self.actions) and self.action_index >= 0:
            return self.actions[self.action_index]
        return None

    def get_previous_action(self) -> Optional[Action]:
        if self.action_index > 0 and self.action_index <= len(self.actions):
            return self.actions[self.action_index - 1]
        return None

    def get_step_index(self) -> int:
        return self.action_index


class Stage(ABC):
    """Stage abstract base class"""

    def __init__(self, stage_config: Dict[str, Any], objects: Dict[str, Any]):
        self.stage_id = lib_uuid.uuid4().hex  # Unique identifier
        (
            self.action_type,
            self.active_obj_id,
            self.passive_obj_id,
            self.active_element,
            self.passive_element,
            self.active_primitive,
            self.passive_primitive,
            self.action_description,
            self.extra_params,
            self.checker_config,
        ) = Stage.parse_stage(stage_config, objects)

        error_data = self.extra_params.get("error_data", {})
        self.error_type = error_data.get("type", None)

        # Predecessor and successor relationships
        self.previous_stage: Stage = None
        self.next_stage: Stage = None
        self.initialized: bool = False

        # Sub-stages
        self.action_sequence_buffer: Queue[ActionSequence] = Queue()
        self.active_action_sequence: ActionSequence = None
        self.substage_index: int = 0  # Current sub-stage index
        self.step_id: int = 0  # Step index within sub-stage

        # Execution status
        self.status = "pending"  # pending, running, completed, failed
        self.start_time = None
        self.end_time = None

    def set_previous_stage(self, stage: "Stage"):
        """Add previous stage"""
        self.previous_stage = stage

    def set_next_stage(self, stage: "Stage"):
        """Add next stage"""
        self.next_stage = stage

    @classmethod
    def parse_stage(cls, stage, objects):
        action = stage["action"]
        if action in ["reset"]:
            return (
                action,
                "gripper",
                "gripper",
                None,
                None,
                None,
                None,
                stage.get("action_description", {"action_text": "", "english_action_text": ""}),
                stage.get("extra_params", {}),
                stage.get("checker", []),
            )
        # Parse active and passive object IDs
        active_obj_id = stage["active"]["object_id"]
        if "part_id" in stage["active"]:
            active_obj_id += "/%s" % stage["active"]["part_id"]

        passive_obj_id = stage["passive"]["object_id"]
        if "part_id" in stage["passive"]:
            passive_obj_id += "/%s" % stage["passive"]["part_id"]
        # Parse active and passive objects
        active_obj = objects[active_obj_id]
        passive_obj = objects[passive_obj_id]

        single_obj = action in ["pull", "rotate", "slide", "shave", "brush", "wipe"]

        gripper_only = action in ["clamp", "move"]

        def _load_element(obj, type):
            # For pick, hook, and move actions, uniformly map to grasp
            if action in ["pick", "hook", "move", "rotate"]:
                action_mapped = "grasp"
            elif action in ["insert"]:
                action_mapped = "place"
            else:
                action_mapped = action
            # Special handling for grasp action's active element (active object is gripper)
            if action_mapped == "grasp" and type == "active":
                return None, None
            elif obj.name == "gripper":
                element = obj.elements[type][action_mapped]
                return element, "default"

            primitive = stage[type]["primitive"] if stage[type]["primitive"] is not None else "default"
            if primitive != "default" or (action_mapped == "grasp" and type == "passive"):
                if action_mapped not in obj.elements[type]:
                    logger.info("No %s element for %s" % (action_mapped, obj.name))
                    return None, None
                element = obj.elements[type][action_mapped][primitive]
            else:
                element = []
                primitives = obj.elements[type][action_mapped]
                for primitive in primitives:
                    _element = primitives[primitive]
                    if isinstance(_element, list):
                        element += _element
                    else:
                        element.append(_element)

            return element, primitive

        # Parse interaction elements of active and passive objects
        if gripper_only:
            # For actions involving only gripper, set both active and passive elements to gripper
            active_element, active_primitive = _load_element(active_obj, "active")
            passive_element, passive_primitive = active_element, active_primitive
        else:
            passive_element, passive_primitive = _load_element(passive_obj, "passive")
            if not single_obj:
                active_element, active_primitive = _load_element(active_obj, "active")
            else:
                active_element, active_primitive = passive_element, passive_primitive
        return (
            action,
            active_obj_id,
            passive_obj_id,
            active_element,
            passive_element,
            active_primitive,
            passive_primitive,
            stage.get("action_description", {"action_text": "", "english_action_text": ""}),
            stage.get("extra_params", {}),
            stage.get("checker", []),
        )

    def parse_action(self, action: Action, objects: Dict[str, Any]):
        (
            gripper_pose_canonical,
            gripper_action,
            transform_world,
            motion_type,
            action_extra_params,
        ) = action.unpack_as_list()

        if motion_type == "local_gripper":
            delta_pose = gripper_pose_canonical
            gripper_pose = objects["gripper"].obj_pose
            target_gripper_pose = gripper_pose @ delta_pose
            motion_type = "Straight"
        elif self.active_obj_id == "gripper" and self.passive_obj_id == "gripper":
            target_gripper_pose = gripper_pose_canonical
        else:

            if gripper_pose_canonical is None:
                target_gripper_pose = None
            else:
                goal_datapack = [
                    self.active_obj_id,
                    self.passive_obj_id,
                    gripper_pose_canonical,
                    gripper_action,
                    transform_world,
                    motion_type,
                ]
                target_gripper_pose = solve_target_gripper_pose(goal_datapack, objects, action_extra_params)

            last_statement = {
                "objects": copy.deepcopy(objects),
                "target_gripper_pose": target_gripper_pose,
            }
            self.last_statement = last_statement
        return (
            target_gripper_pose,
            motion_type,
            gripper_action,
            self.extra_params.get("arm", "right"),
            self.action_description,
            action_extra_params,
        )

    def initialize(self, objects, robot):
        self.initialized = True

    ## action sequence
    def add_action_sequence(self, action_sequence: ActionSequence):
        action_sequence.set_parent_stage(self)
        self.action_sequence_buffer.put(action_sequence)

    def get_action_sequence(self) -> Optional[ActionSequence]:
        self.active_action_sequence = (
            self.action_sequence_buffer.get() if not self.action_sequence_buffer.empty() else None
        )
        return self.active_action_sequence

    def generate_action_sequence(self, *args, **kwargs) -> Optional[ActionSequence]:
        raise NotImplementedError

    def initialize_action_sequence_buffer(self, objects, robot):
        for poses in self.select_pose(objects, robot):
            sequence = self.generate_action_sequence(**poses)
            if sequence is not None:
                self.add_action_sequence(sequence)
        return not self.action_sequence_buffer.empty()

    def select_pose(self, objects, robot):
        gripper2obj = []
        arm = self.extra_params.get("arm", "right")
        current_gripper_pose = robot.get_ee_pose(id=arm)
        gripper2obj = [current_gripper_pose]
        return gripper2obj

    def check_completion(self, objects, robot=None):
        assert self.active_action_sequence is not None, f"Active action for stage {self.action_type} is None"
        goal_datapack = [
            self.active_obj_id,
            self.passive_obj_id,
        ] + self.active_action_sequence.get_current_action().unpack_as_list()
        succ = simple_check_completion(goal_datapack, objects)
        return succ

    def __str__(self):
        return f"Stage {self.stage_id}: {self.action_type} ({self.status})"

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.stage_id}, action={self.action_type}, status={self.status})"
