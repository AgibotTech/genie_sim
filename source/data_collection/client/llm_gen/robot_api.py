# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""High-level robot APIs wrapping gRPC commands to compose scene actions."""

import json
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
import os

from client.planner.action.stage import Stage, Action
import importlib
import sys

sim_assets = os.environ.get("SIM_ASSETS")
if not sim_assets:
    raise RuntimeError("SIM_ASSETS environment variable is not set")

# SIM_ASSETS is a package directory with __init__.py; its name may differ from assets.
# Add its parent to sys.path and dynamically import the actual directory name.
sim_assets = os.path.abspath(sim_assets)
sim_assets_parent = os.path.dirname(sim_assets)
sim_assets_pkg = os.path.basename(sim_assets)
sys.path.insert(0, sim_assets_parent)

assets_mod = importlib.import_module(sim_assets_pkg)
try:
    ASSETS_INDEX = assets_mod.ASSETS_INDEX
    ASSETS_PATH = assets_mod.ASSETS_PATH
except AttributeError as e:
    raise AttributeError(
        f"ASSETS_INDEX not found in {sim_assets}; ensure __init__.py exports it"
    ) from e


import numpy as np

from client.robot.omni_robot import IsaacSimRpcRobot
from client.robot.client import RpcClient
from common.gripper_control import make_gripper_command
from client.planner.action.grasp import PickStage
from client.planner.action.place import PlaceStage
from client.planner.action.insert import InsertStage
from client.planner.action.constraint_rotate import ConstraintRotateStage
from client.planner.action.move import MoveStage
from client.planner.action.follow_trajectory import (
    resolve_trajectory_candidates,
    validate_trajectory_options,
    waypoints_to_poses,
)
from client.planner.action.bimanual_rigid_move import BimanualRigidMoveStage
from client.layout.object import OmniObject
from common.base_utils.logger import logger
from common.base_utils.transform_utils import (
    get_pose_wxyz,
    mat2quat_wxyz,
    pose_from_position_quaternion,
    quat2mat,
    quat2mat_wxyz,
    transform_points,
)
from client.layout.object import OmniObject
from common.base_utils.transform_utils import calculate_rotation_matrix, pose_difference
from common.base_utils.transform_utils import add_random_noise_to_pose
from client.llm_gen.gather_interactions import gather_interactions


class TurnBodyConfig:
    """Upper-body turning configuration."""
    def __init__(
        self,
        step_deg: float = 30.0,
        unmovable_objects: List[str] = [],
        extra_checkers: List[Callable] = [],
        action_description: Dict[str, str] = None,
        extra_params: Dict[str, Any] = {},
    ):
        """Initialize turning options.
        step_deg is angular interpolation spacing in degrees, default 30.
        unmovable_objects must stay stationary; extra_checkers validate success.
        action_description contains zh/en text; extra_params provides custom options."""
        self.step_deg = step_deg
        self.unmovable_objects = unmovable_objects
        self.extra_checkers = extra_checkers
        self.action_description = action_description or {}
        self.extra_params = extra_params

    def to_dict(self):
        params = {
            "step_deg": self.step_deg,
            "unmovable_objects": self.unmovable_objects,
            "extra_checkers": self.extra_checkers,
            "action_description": self.action_description,
        }
        params.update(self.extra_params)
        return params


class ShakeConfig:
    """Wrist shaking configuration."""
    def __init__(
        self,
        unmovable_objects: List[str] = [],
        extra_checkers: List[Callable] = [],
        action_description: Dict[str, str] = None,
        extra_params: Dict[str, Any] = {},
    ):
        self.unmovable_objects = unmovable_objects
        self.extra_checkers = extra_checkers
        self.action_description = action_description or {}
        self.extra_params = extra_params

    def to_dict(self):
        params = {
            "unmovable_objects": self.unmovable_objects,
            "extra_checkers": self.extra_checkers,
            "action_description": self.action_description,
        }
        params.update(self.extra_params)
        return params

class GraspConfig:
    """Grasping and lifting configuration."""

    def __init__(
        self,
        grasp_offset: float = 0.0,
        pre_grasp_offset: float = 0.05,
        grasp_lower_percentile: float = 0,
        grasp_upper_percentile: float = 100,
        grasp_percentile_axis: str = None,
        disable_upside_down: bool = True,
        flip_grasp: bool = True,
        set_grasp_along_axis: str = None,
        set_grasp_pose_center: bool = False,
        is_from_up_side: bool = False,
        pick_up_distance: float = 0.12,
        pick_up_direction: str = "z",
        grasp_pose_filter: Callable = None,
        grasp_pose: np.ndarray = None,
        grasp_pose_frame: str = "world",
        grasp_width: np.ndarray = None,
        active_primitive: str = None,
        passive_primitive: str = None,
        unmovable_objects: List[str] = [],
        extra_checkers: List[Callable] = [],
        action_description: Dict[str, str] = None,
        extra_params: Dict[str, Any] = {},
    ):
        """Initialize grasp options.

        Args:
            grasp_offset: Grasp offset in meters.
            pre_grasp_offset: Approach offset in meters.
            grasp_lower_percentile: Lower percentile of grasp-point z values.
            grasp_upper_percentile: Upper percentile of grasp-point z values.
            disable_upside_down: Exclude upside-down grasps.
            flip_grasp: Flip grasp orientation.
            set_grasp_along_axis: Orient the grasp perpendicular to a local object axis.
            set_grasp_pose_center: Zero other local coordinates for the selected grasp axis.
            is_from_up_side: Prefer candidates above the object.
            pick_up_distance: Lift distance in meters.
            pick_up_direction: Lift axis: x, y, or z.
            active_primitive: Active annotation, normally the gripper.
            passive_primitive: Annotation on the grasped object.
            grasp_pose: Custom (N, 4, 4) or (4, 4) candidates replacing USD annotations;
                subsequent IK, flipping, and ranking still apply. None uses annotations.
            grasp_pose_frame: world (default) or relative to the grasped object.
                World candidates are transformed by inverse(obj_pose).
            grasp_width: Widths (N,) for custom poses; omitted widths default to 0.08 meters.
            unmovable_objects: Objects that must remain stationary during execution.
            extra_checkers: Additional success validators.
            action_description: zh/en instruction text; the arm subject is added automatically."""
        self.grasp_offset = grasp_offset
        self.pre_grasp_offset = pre_grasp_offset
        self.grasp_lower_percentile = grasp_lower_percentile
        self.grasp_upper_percentile = grasp_upper_percentile
        self.grasp_percentile_axis = grasp_percentile_axis
        self.grasp_pose_filter = grasp_pose_filter
        self.disable_upside_down = disable_upside_down
        self.flip_grasp = flip_grasp
        self.set_grasp_along_axis = set_grasp_along_axis
        self.set_grasp_pose_center = set_grasp_pose_center
        self.pick_up_distance = pick_up_distance
        self.pick_up_direction = pick_up_direction
        self.is_from_up_side = is_from_up_side
        self.active_primitive = active_primitive
        self.passive_primitive = passive_primitive
        # Custom grasp candidates bypass USD annotations; see the constructor docstring.
        self.grasp_pose = grasp_pose
        self.grasp_pose_frame = grasp_pose_frame
        self.grasp_width = grasp_width
        self.unmovable_objects = unmovable_objects
        self.extra_checkers = extra_checkers
        self.action_description = action_description or {}
        self.extra_params = extra_params

    def to_dict(self) -> Dict:
        """Convert the configuration to a dictionary."""
        params = {
            "grasp_offset": self.grasp_offset,
            "pre_grasp_offset": self.pre_grasp_offset,
            "grasp_lower_percentile": self.grasp_lower_percentile,
            "grasp_upper_percentile": self.grasp_upper_percentile,
            "grasp_percentile_axis": self.grasp_percentile_axis,
            "disable_upside_down": self.disable_upside_down,
            "grasp_pose_filter": self.grasp_pose_filter,
            "flip_grasp": self.flip_grasp,
            "set_grasp_along_axis": self.set_grasp_along_axis,
            "set_grasp_pose_center": self.set_grasp_pose_center,
            "is_from_up_side": self.is_from_up_side,
            "pick_up_distance": self.pick_up_distance,
            "pick_up_direction": self.pick_up_direction,
            "active_primitive": self.active_primitive,
            "passive_primitive": self.passive_primitive,
            "unmovable_objects": self.unmovable_objects,
            "extra_checkers": self.extra_checkers,
            "action_description": self.action_description,
        }
        # Forward custom candidates only when explicitly provided.
        # Avoid replacing defaults with None; PickStage checks for non-None values.
        if self.grasp_pose is not None:
            params["grasp_pose"] = self.grasp_pose
            params["grasp_pose_frame"] = self.grasp_pose_frame
            if self.grasp_width is not None:
                params["grasp_width"] = self.grasp_width
        params.update(self.extra_params)
        return params


class PlaceConfig:
    """Placement configuration."""

    def __init__(
        self,
        disable_upside_down: bool = True,
        active_primitive: str = None,
        passive_primitive: str = None,
        pre_insert_offset: float = None,
        unmovable_objects: List[str] = [],
        extra_checkers: List[Callable] = [],
        action_description: Dict[str, str] = None,
        extra_params: Dict[str, Any] = {},
    ):
        """Initialize placement options.
        disable_upside_down filters orientations; active_primitive belongs to the placed
        object and passive_primitive to its destination. pre_insert_offset is in meters.
        unmovable_objects must stay stationary; extra_checkers validate success.
        action_description contains zh/en text with the arm subject added automatically."""
        self.disable_upside_down = disable_upside_down
        self.active_primitive = active_primitive
        self.passive_primitive = passive_primitive
        self.pre_insert_offset = pre_insert_offset
        self.unmovable_objects = unmovable_objects
        self.extra_checkers = extra_checkers
        self.action_description = action_description or {}
        self.extra_params = extra_params

    def to_dict(self) -> Dict:
        """Convert the configuration to a dictionary."""
        params = {
            "disable_upside_down": self.disable_upside_down,
            "active_primitive": self.active_primitive,
            "passive_primitive": self.passive_primitive,
            "pre_insert_offset": self.pre_insert_offset,
            "unmovable_objects": self.unmovable_objects,
            "extra_checkers": self.extra_checkers,
            "action_description": self.action_description,
        }
        params.update(self.extra_params)
        return params

class OpenConfig:
    """Revolute-joint opening configuration."""

    def __init__(
        self,
        target_limit: str = "min",
        trajectory_step_deg: float = 1.0,
        max_consecutive_failures: int = 3,
        solvable_threshold: float = 0.5,
        unmovable_objects: List[str] = [],
        extra_checkers: List[Callable] = [],
        action_description: Dict[str, str] = None,
        extra_params: Dict[str, Any] = {},
    ):
        """Configure opening a revolute-joint object such as a cabinet door.
        target_limit selects min/max; trajectory_step_deg is angular spacing in degrees.
        max_consecutive_failures bounds following failures; solvable_threshold controls
        IK acceptance. unmovable_objects must stay stationary; extra_checkers validate
        success. action_description is required zh/en text without an arm subject.
        extra_params provides additional stage options."""
        self.target_limit = target_limit
        self.trajectory_step_deg = trajectory_step_deg
        self.max_consecutive_failures = max_consecutive_failures
        self.solvable_threshold = solvable_threshold
        self.unmovable_objects = unmovable_objects
        self.extra_checkers = extra_checkers
        self.action_description = action_description or {}
        self.extra_params = extra_params

    def to_dict(self) -> Dict:
        """Convert the configuration to a dictionary."""
        params = {
            "target_limit": self.target_limit,
            "trajectory_step_deg": self.trajectory_step_deg,
            "max_consecutive_failures": self.max_consecutive_failures,
            "solvable_threshold": self.solvable_threshold,
            "unmovable_objects": self.unmovable_objects,
            "extra_checkers": self.extra_checkers,
            "action_description": self.action_description,
        }
        params.update(self.extra_params)
        return params

class RotateConfig:
    """Rotation action configuration."""

    def __init__(
        self,
        constraint_axises: List[str] = [],
        offset_and_constraint_in_goal_frame: bool = False,
        unmovable_objects: List[str] = [],
        extra_checkers: List[Callable] = [],
        action_description: Dict[str, str] = None,
        extra_params: Dict[str, Any] = {},
    ):
        # constraint_axises locks rotations about the selected axes, e.g. x and y.
        # Use the goal/gripper frame when enabled; otherwise use the world frame.
        # unmovable_objects must remain stationary for success validation.
        # extra_checkers provides additional success validators.
        self.constraint_axises = constraint_axises
        self.offset_and_constraint_in_goal_frame = offset_and_constraint_in_goal_frame
        self.unmovable_objects = unmovable_objects
        self.extra_checkers = extra_checkers
        self.action_description = action_description or {}
        self.extra_params = extra_params

    def to_dict(self) -> Dict:
        """Convert the configuration to a dictionary."""
        params = {
            "constraint_axises": self.constraint_axises,
            "offset_and_constraint_in_goal_frame": self.offset_and_constraint_in_goal_frame,
            "unmovable_objects": self.unmovable_objects,
            "extra_checkers": self.extra_checkers,
            "action_description": self.action_description,
        }
        params.update(self.extra_params)
        return params


class MoveConfig:
    """Movement action configuration."""

    def __init__(
        self,
        path_constraint: List[float] = [],
        offset_and_constraint_in_goal_frame: bool = False,
        active_primitive: str = None,
        passive_primitive: str = None,
        gripper_action: str = None,
        unmovable_objects: List[str] = [],
        extra_checkers: List[Callable] = [],
        action_description: Dict[str, str] = None,
        extra_params: Dict[str, Any] = {},
    ):
        # path_constraint is [rx, ry, rz, x, y, z], with rotation then translation weights.
        # Zero leaves a dimension free; [0.1, 0.1, 0.1, 0, 0.1, 0.1] allows x motion.
        # Use goal/gripper-frame constraints when enabled, otherwise world-frame constraints.
        # gripper_action is open, close, or None.
        # unmovable_objects must remain stationary for success validation.
        # extra_checkers provides additional success validators.
        self.path_constraint = path_constraint
        self.offset_and_constraint_in_goal_frame = offset_and_constraint_in_goal_frame
        self.active_primitive = active_primitive
        self.passive_primitive = passive_primitive
        self.gripper_action = gripper_action
        self.unmovable_objects = unmovable_objects
        self.extra_checkers = extra_checkers
        self.action_description = action_description or {}
        self.extra_params = extra_params

    def to_dict(self) -> Dict:
        """Convert the configuration to a dictionary."""
        params = {
            "path_constraint": self.path_constraint,
            "offset_and_constraint_in_goal_frame": self.offset_and_constraint_in_goal_frame,
            "active_primitive": self.active_primitive,
            "passive_primitive": self.passive_primitive,
            "gripper_action": self.gripper_action,
            "unmovable_objects": self.unmovable_objects,
            "extra_checkers": self.extra_checkers,
            "action_description": self.action_description,
        }
        params.update(self.extra_params)
        return params


class FollowTrajectoryConfig:
    """Configuration for following one of several ordered pose trajectories."""

    def __init__(
        self,
        noise_range: List[float] = None,
        noise_frame: str = "local",
        sample_count: int = 10,
        unmovable_objects: List[str] = None,
        extra_checkers: List[Callable] = None,
        action_description: Dict[str, str] = None,
        extra_params: Dict[str, Any] = None,
    ):
        """
        Args:
            noise_range: ``[dx, dy, dz, droll, dpitch, dyaw]`` symmetric
                sampling ranges. Position is in metres and rotation is in radians.
            noise_frame: ``"local"`` (default) or ``"world"``.
            sample_count: Samples generated for each waypoint whose original IK
                target is not solvable.
            unmovable_objects: Objects that must remain stationary during the action.
            extra_checkers: Additional success checkers run after execution.
            action_description: Chinese/English action descriptions.
            extra_params: Additional parameters forwarded to the simple-action lifecycle.
        """
        self.noise_range = list(noise_range) if noise_range is not None else [0.0] * 6
        self.noise_frame = noise_frame
        self.sample_count = sample_count
        self.unmovable_objects = (
            list(unmovable_objects) if unmovable_objects is not None else []
        )
        self.extra_checkers = list(extra_checkers) if extra_checkers is not None else []
        self.action_description = action_description or {}
        self.extra_params = dict(extra_params) if extra_params is not None else {}

    def to_dict(self) -> Dict:
        params = {
            "noise_range": self.noise_range,
            "noise_frame": self.noise_frame,
            "sample_count": self.sample_count,
            "unmovable_objects": self.unmovable_objects,
            "extra_checkers": self.extra_checkers,
            "action_description": self.action_description,
        }
        params.update(self.extra_params)
        return params


class BimanualRigidMoveConfig:
    """Configuration for bimanual rigid coordinated motion.

    The leader arm plans a trajectory to the target pose, while the follower arm
    maintains its initial relative pose (T_rel = T_leader^{-1} · T_follower) and
    follows rigidly throughout the motion.
    """

    def __init__(
        self,
        leader_arm: str = "right",
        path_constraint: List[float] = [],
        offset_and_constraint_in_goal_frame: bool = False,
        goal_offset: List[float] = [0, 0, 0, 1, 0, 0, 0],
        from_current_pose: bool = False,
        ik_downsample_num: int = 200,
        max_consecutive_failures: int = 1,
        motion_run_ratio: float = 1.0,
        disable_collision_links: List[str] = [],
        unmovable_objects: List[str] = [],
        extra_checkers: List[Callable] = [],
        action_description: Dict[str, str] = None,
        extra_params: Dict[str, Any] = {},
    ):
        self.leader_arm = leader_arm
        self.path_constraint = path_constraint
        self.offset_and_constraint_in_goal_frame = offset_and_constraint_in_goal_frame
        self.goal_offset = goal_offset
        self.from_current_pose = from_current_pose
        self.ik_downsample_num = ik_downsample_num
        self.max_consecutive_failures = max_consecutive_failures
        self.motion_run_ratio = motion_run_ratio
        self.disable_collision_links = disable_collision_links
        self.unmovable_objects = unmovable_objects
        self.extra_checkers = extra_checkers
        self.action_description = action_description or {}
        self.extra_params = extra_params

    def to_dict(self) -> Dict:
        params = {
            "leader_arm": self.leader_arm,
            "path_constraint": self.path_constraint,
            "offset_and_constraint_in_goal_frame": self.offset_and_constraint_in_goal_frame,
            "goal_offset": self.goal_offset,
            "from_current_pose": self.from_current_pose,
            "ik_downsample_num": self.ik_downsample_num,
            "max_consecutive_failures": self.max_consecutive_failures,
            "motion_run_ratio": self.motion_run_ratio,
            "disable_collision_links": self.disable_collision_links,
            "unmovable_objects": self.unmovable_objects,
            "extra_checkers": self.extra_checkers,
            "action_description": self.action_description,
        }
        params.update(self.extra_params)
        return params


g2_init_arm_pose = {
    "idx21_arm_l_joint1": 0.739033,
    "idx61_arm_r_joint1": -0.739033,
    "idx05_body_joint5": 0.0,
    "idx11_head_joint1": 0.0,
    "idx22_arm_l_joint2": -0.717023,
    "idx62_arm_r_joint2": -0.717023,
    "idx04_body_joint4": 0.0,
    "idx12_head_joint2": 0.0,
    "idx23_arm_l_joint3": -1.524419,
    "idx63_arm_r_joint3": 1.524419,
    "idx03_body_joint3": 0.10025,
    "idx13_head_joint3": 0.11464,
    "idx24_arm_l_joint4": -1.537612,
    "idx64_arm_r_joint4": -1.537612,
    "idx02_body_joint2": 1.2172,
    "idx25_arm_l_joint5": 0.27811,
    "idx65_arm_r_joint5": -0.27811,
    "idx01_body_joint1": -0.83423,
    "idx26_arm_l_joint6": -0.925845,
    "idx66_arm_r_joint6": -0.925845,
    "idx27_arm_l_joint7": -0.839257,
    "idx67_arm_r_joint7": 0.839257,
    "idx111_chassis_lwheel_front_joint1": None,
    "idx121_chassis_lwheel_rear_joint1": None,
    "idx131_chassis_rwheel_front_joint1": None,
    "idx141_chassis_rwheel_rear_joint1": None,
    "idx112_chassis_lwheel_front_joint2": None,
    "idx122_chassis_lwheel_rear_joint2": None,
    "idx132_chassis_rwheel_front_joint2": None,
    "idx142_chassis_rwheel_rear_joint2": None,
    "idx31_gripper_l_inner_joint1": None,
    "idx41_gripper_l_outer_joint1": 0.85,
    "idx71_gripper_r_inner_joint1": None,
    "idx81_gripper_r_outer_joint1": 0.85,
    "idx32_gripper_l_inner_joint3": None,
    "idx42_gripper_l_outer_joint3": None,
    "idx72_gripper_r_inner_joint3": None,
    "idx82_gripper_r_outer_joint3": None,
    "idx33_gripper_l_inner_joint4": None,
    "idx43_gripper_l_outer_joint4": None,
    "idx73_gripper_r_inner_joint4": None,
    "idx83_gripper_r_outer_joint4": None,
    "idx39_gripper_l_inner_joint0": None,
    "idx49_gripper_l_outer_joint0": None,
    "idx79_gripper_r_inner_joint0": None,
    "idx89_gripper_r_outer_joint0": None,
}
g2_robot_config = {
    "init_arm_pose_noise": {
        ".*_arm_.*": {"noise_type": "uniform", "low": -0.05, "high": 0.05}
    },
    "init_arm_pose": g2_init_arm_pose,
    "robot_cfg": "G2_omnipicker_fixed_dual.json",
    "robot_id": "G2",
    "robot_init_pose": {
        "position": [-0.7, 0.0, 0],
        "quaternion": [1, 0, 0, 0],
        "random": {"delta_position": [0.06, 0.06, 0]},
    },
}


class RobotAPIConfig:
    robot_config = g2_robot_config
    auto_record = False
    recording_setting = {}
    task_description = {
        "english_task_name": "",
        "init_scene_text": "",
        "task_name": "",
    }
    task_metric = {}
    task_name = ""

    @staticmethod
    def from_json(json_file: str):
        robot_api_config = RobotAPIConfig()
        with open(json_file, "r") as f:
            config = json.load(f)
        robot_api_config.robot_config = config.get("robot", g2_robot_config)
        robot_api_config.auto_record = config.get("auto_record", False)
        robot_api_config.recording_setting = config.get("recording_setting", {})
        robot_api_config.task_description = config.get("task_description", {})
        robot_api_config.task_name = config.get("task_name", "")
        robot_api_config.task_metric = config.get("task_metric", {})
        return robot_api_config


# ---------------------------------------------------------------------------
# Synchronized dual-arm action specs.
#
# Each spec mirrors an existing high-level single-arm action API.
# The arm is supplied by dual_act through its left/right argument position.
# Specs use the private stage-building helpers
# to construct stages for dual_run_stage.
# That runner synchronizes steps and state restoration across both arms.
# ---------------------------------------------------------------------------


class DualSpec:
    """Base class for dual-arm specs; subclasses implement build_stage(api, arm).
    Configs require action_description, but dual steps record a shared description
    at their boundaries. Fill missing config descriptions with placeholders before
    building stages to satisfy validation."""

    _DEFAULT_DESC = {"zh": "(双臂同步)", "en": "(dual sync)"}

    def build_stage(self, api: "RobotAPI", arm: str) -> Tuple[Stage, Dict[str, OmniObject]]:
        raise NotImplementedError

    @classmethod
    def _ensure_action_description(cls, config) -> None:
        """Fill a missing config action_description with a placeholder for validation."""
        existing = getattr(config, "action_description", None)
        if not existing:
            config.action_description = dict(cls._DEFAULT_DESC)


@dataclass
class DualPick(DualSpec):
    actor: Union[OmniObject, str]
    grasp_config: GraspConfig = field(default_factory=GraspConfig)
    part: Optional[str] = None

    def build_stage(self, api, arm):
        self._ensure_action_description(self.grasp_config)
        return api._build_pick_stage(self.actor, arm, self.grasp_config, part=self.part)


@dataclass
class DualPlaceToActor(DualSpec):
    actor: str
    target_actor: str
    place_config: PlaceConfig = field(default_factory=PlaceConfig)

    def build_stage(self, api, arm):
        self._ensure_action_description(self.place_config)
        return api._build_place_to_actor_stage(
            self.actor, arm, self.target_actor, self.place_config,
        )


@dataclass
class DualPlaceToPose(DualSpec):
    actor: str
    target_pose: np.ndarray
    place_config: PlaceConfig = field(default_factory=PlaceConfig)

    def build_stage(self, api, arm):
        self._ensure_action_description(self.place_config)
        return api._build_place_to_pose_stage(
            self.actor, arm, self.target_pose, self.place_config,
        )


@dataclass
class DualMoveToPose(DualSpec):
    actor: str
    target_pose: np.ndarray
    move_config: MoveConfig = field(default_factory=MoveConfig)
    part: Optional[str] = None

    def build_stage(self, api, arm):
        self._ensure_action_description(self.move_config)
        return api._build_move_to_pose_stage(
            self.actor,
            arm,
            self.target_pose,
            self.move_config,
            part=self.part,
        )


@dataclass
class DualRotate(DualSpec):
    actor: str
    target_pose: np.ndarray
    rotate_config: RotateConfig = field(default_factory=RotateConfig)

    def build_stage(self, api, arm):
        self._ensure_action_description(self.rotate_config)
        return api._build_rotate_stage(
            self.actor, arm, self.target_pose, self.rotate_config,
        )


class RobotAPI:
    """High-level operations for picking, placing, moving, and scene control."""

    def __init__(
        self,
        config: RobotAPIConfig,
        scene_usd: str,
        scene_info: str,
        client_host: str = "localhost:50051",
        from_robot = None,
    ):
        """Initialize robot configuration, scene USD, and gRPC endpoint.
        position is xyz, rotation is wxyz; stand_size_x/y configure the stand,
        and robot_init_arm_pose supplies initial joint positions."""
        # Keep RobotAPIConfig so helpers can access task robot configuration,
        # including gripper_anchor_link overrides.
        self.config = config
        if from_robot is not None:
            self.robot_init_pose = from_robot.robot_init_pose
            self.robot_init_arm_pose_noise = from_robot.robot_init_arm_pose_noise
        else:
            self.robot_init_pose = {}
            if "random" in config.robot_config.get("robot_init_pose", {}):
                random_range = config.robot_config.get("robot_init_pose", {}).get(
                    "random", {}
                )
                delta_position = random_range.get("delta_position", [0, 0, 0])
                robot_init_pose = {
                    "position": [
                        config.robot_config["robot_init_pose"]["position"][0]
                        + np.random.uniform(-delta_position[0], delta_position[0]),
                        config.robot_config["robot_init_pose"]["position"][1]
                        + np.random.uniform(-delta_position[1], delta_position[1]),
                        config.robot_config["robot_init_pose"]["position"][2]
                        + np.random.uniform(-delta_position[2], delta_position[2]),
                    ],
                    "quaternion": config.robot_config["robot_init_pose"].get(
                        "quaternion", [1, 0, 0, 0]
                    ),
                }
                logger.info(f"Random robot init position{robot_init_pose}")
                self.robot_init_pose = robot_init_pose
            else:
                self.robot_init_pose = config.robot_config.get(
                    "robot_init_pose", {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]}
                )
                logger.info(f"Robot init pose {self.robot_init_pose}")
            self.robot_init_arm_pose_noise = config.robot_config.get(
                "init_arm_pose_noise", None
            )
        if from_robot is not None:
            self.body_init_poses = from_robot.body_init_poses
        else:
            self.body_init_poses = None
        self.robot = IsaacSimRpcRobot(
            robot_cfg=config.robot_config.get(
                "robot_cfg", "G2_omnipicker_fixed_dual.json"
            ),
            scene_usd=scene_usd,
            client_host=client_host,
            position=self.robot_init_pose["position"],
            rotation=self.robot_init_pose["quaternion"],
            robot_init_arm_pose=config.robot_config.get(
                "init_arm_pose", g2_init_arm_pose
            ),
            robot_init_arm_pose_noise=self.robot_init_arm_pose_noise,
            body_init_poses=self.body_init_poses,
        )
        self.body_init_poses = self.robot.body_init_poses
        self.client = self.robot.client
        self.init_position = self.robot_init_pose["position"]
        self.init_rotation = self.robot_init_pose["quaternion"]
        self.robot_init_arm_pose = config.robot_config.get(
            "init_arm_pose", g2_init_arm_pose
        )
        self.robot_cfg = config.robot_config.get(
            "robot_cfg", "G2_omnipicker_fixed_dual.json"
        )
        self.scene_info = json.load(open(scene_info))
        self.add_all_objects_info()
        layout = self.scene_info.get("layout", {})
        self.usd_list = list(
            {obj.get("usd") for obj in layout.values() if obj.get("usd")}
        )
        self.interactions = gather_interactions(self.usd_list)
        self.robot.reset()
        self.robot.open_gripper(id="right", detach=False)
        self.robot.open_gripper(id="left", detach=False)
        self.attached_obj_id = None
        self.objects = {}
        self.robot.reset_pose = {
            "right": self.robot.get_ee_pose(ee_type="gripper", id="right"),
            "left": self.robot.get_ee_pose(ee_type="gripper", id="left"),
        }
        self.robot.client.detach_obj()
        self.stage_id = 0
        self.get_all_objects()
        self.auto_record = config.auto_record
        self.robot.client.reset()
        self.robot.client.set_task_task_basic_info(config.task_description)
        if len(config.task_metric) > 0:
            self.robot.client.set_task_metric(config.task_metric)
        if config.auto_record:
            self.robot.client.start_recording(
                task_name=config.task_name,
                fps=config.recording_setting.get("fps", 30),
                data_keys={
                    "camera": {
                        "camera_prim_list": config.recording_setting.get(
                            "camera_list", []
                        ),
                        "render_depth": False,
                        "render_semantic": config.recording_setting.get(
                            "render_semantic", False
                        ),
                    },
                    "joint_position": True,
                    "gripper": True,
                    "additional_parameters": json.dumps(config.recording_setting),
                },
            )

    def exit(self):
        self.robot.client.exit()

    def stop_recording(self):
        if self.auto_record:
            self.robot.client.stop_recording()


    # Control.
    def get_object_pose(self, actor: str, articulation_part: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return object world position xyz and orientation wxyz for a prim path."""
        if actor.endswith("/entity"):
            actor = actor[:-7]
            actor = actor.split("/")[-1]
        usd_id = self.get_object_usd_id(actor)
        asset_info = ASSETS_INDEX[usd_id]
        if articulation_part is None:
            if asset_info.get("has_articulation", False):
                logger.info(f"{actor}: Object has articulation, using default part: body")
                articulation_part = "body"
        actor = self.get_object_prim_path(actor)
        if articulation_part is not None:
            actor = f"{actor}/{articulation_part}"
        pose_matrix = self.robot.get_prim_world_pose(actor)
        position = pose_matrix[:3, 3]
        rotation = self._mat_to_quat(pose_matrix[:3, :3])
        return np.array(position), np.array(rotation)

    def get_object_prim_path(self, object_name: str, articulation_part: str = None) -> str:
        """Resolve an object identifier to its prim path."""
        if articulation_part is not None:
            return f"/World/Objects/{object_name}/entity/{articulation_part}"
        return f"/World/Objects/{object_name}/entity"

    def get_articulation_joint_info(
        self, articulation_object_prim_path: str, joint_prim_path: str = None
    ) -> Dict:
        """Return joint path/name, body0/body1 paths, axis, current angle,
        and lower/upper limits in degrees."""
        return self.robot.client.command_any(
            {
                "command": "GetArticulationJointInfo",
                "articulation_object_prim_path": articulation_object_prim_path,
                "joint_prim_path": joint_prim_path,
            }
        )

    def judge_arm_to_use(self, position: List[float]) -> str:
        """Choose an arm using the target world position transformed into robot coordinates.
        Below -THRESHOLD in y choose right; above THRESHOLD choose left.
        Between them, the left-arm probability increases linearly from zero to one."""
        # Arm-selection threshold in meters.
        THRESHOLD = 0.15

        # Convert the position to a numpy array.
        position = np.array(position)

        # Build the robot-base transform.
        robot_pose = pose_from_position_quaternion(
            np.array(self.init_position), np.array(self.init_rotation)
        )
        # Compute its inverse.
        robot_pose_inv = np.linalg.inv(robot_pose)
        # Convert the world point to homogeneous coordinates.
        point_homogeneous = np.array([position[0], position[1], position[2], 1.0])
        # Transform into robot coordinates.
        point_robot_homogeneous = robot_pose_inv @ point_homogeneous
        position_robot = point_robot_homogeneous[:3]

        y_value = position_robot[1]

        if y_value < -THRESHOLD:
            return "right"
        elif y_value > THRESHOLD:
            return "left"
        else:
            # Within the threshold interval, left-arm probability increases linearly.
            # It is zero at -THRESHOLD and one at THRESHOLD.
            probability_left = (y_value + THRESHOLD) / (2 * THRESHOLD)
            if np.random.random() < probability_left:
                return "left"
            else:
                return "right"

    def back_to_origin(self, arm: str = "") -> bool:
        """Return the selected left/right arm to its predefined initial pose; report success."""
        if arm == "":
            reset_left = self.back_to_origin("left")
            reset_right = self.back_to_origin("right")
            return reset_left and reset_right
        plan_type = "AvoidObs"
        init_pose = self.robot.reset_pose[arm]
        curr_pose = self.robot.get_ee_pose(ee_type="gripper", id=arm)
        interp_pose = init_pose.copy()
        interp_pose[:3, 3] = (
            curr_pose[:3, 3] + (init_pose[:3, 3] - curr_pose[:3, 3]) * 0.25
        )
        task_success = self.robot.move_pose(
            self.robot.reset_pose[arm],
            type=plan_type,
            arm=arm,
            block=True,
        )
        if not task_success and plan_type == "AvoidObs":
            logger.error(
                f"Stage reset move to reset pose with AvoidObs failed, try Simple"
            )
            task_success = self.robot.move_pose(
                self.robot.reset_pose[arm],
                type="Simple",
                arm=arm,
                block=True,
            )
        return task_success

    def wait(self, time: float) -> bool:
        steps = int(time * 60)
        self.client.run_step(steps)
        return True

    # ---------------------------------------------------------------------
    # Synchronized dual-arm stage execution.
    #
    # Independent run_stage calls would store and restore states separately.
    # One arm could restore while the other is stepping, overwriting shared state.
    # Split each step into prepare, dispatch movement, and finish.
    # Prepare both arms before a single DualCuroboMove or DualLinearMove dispatch.
    # On failure, stop the sequence and restore both arms together.
    # try_next_sequence decides whether to advance candidates or terminate.
    # ---------------------------------------------------------------------

    DUAL_STAGE_MAX_ATTEMPTS = 3

    def dual_act(
        self,
        left: Optional[DualSpec] = None,
        right: Optional[DualSpec] = None,
        action_description: Optional[Dict[str, str]] = None,
        extra_checkers: Optional[List[Callable[[], bool]]] = None,
    ) -> Tuple[bool, str]:
        """Run synchronized dual-arm specs; None leaves an arm inactive.
        Per-arm checkers and stationary-object requirements belong to each config.
        Use action_description for the shared recorded instruction and extra_checkers
        for validation after all steps. Return (stage_success, store_name)."""
        if left is None and right is None:
            return True, f"dual_stage_{self.stage_id}"

        # Build both stages; their object dictionaries reference the same self.objects.
        left_stage: Optional[Stage] = None
        right_stage: Optional[Stage] = None
        objects: Optional[Dict[str, OmniObject]] = None
        if left is not None:
            left_stage, objects = left.build_stage(self, "left")
        if right is not None:
            right_stage, objects_r = right.build_stage(self, "right")
            objects = objects if objects is not None else objects_r

        # Caller descriptions use zh/en keys, matching the action configs.
        # Frame-state recording expects action_text/english_action_text keys.
        # Single-arm stage builders add an arm subject during conversion.
        # Dual instructions are shared, so map zh/en directly without an arm subject.
        # When None, retain each stage description in the frame-state format.
        action_description_override = None
        if action_description:
            action_description_override = {
                "action_text": action_description.get("zh", action_description.get("action_text", "")),
                "english_action_text": action_description.get(
                    "en", action_description.get("english_action_text", "")
                ),
            }

        # Log the shared instruction and per-arm config descriptions
        # to correlate recorded frame semantics with stage execution.
        _l_desc = getattr(getattr(left, "grasp_config", None) or getattr(left, "move_config", None)
                          or getattr(left, "place_config", None) or getattr(left, "rotate_config", None),
                          "action_description", None) if left is not None else None
        _r_desc = getattr(getattr(right, "grasp_config", None) or getattr(right, "move_config", None)
                          or getattr(right, "place_config", None) or getattr(right, "rotate_config", None),
                          "action_description", None) if right is not None else None
        logger.info(
            f"[dual_act] overall desc={action_description} "
            f"left={type(left).__name__ if left else None} left_desc={_l_desc} "
            f"right={type(right).__name__ if right else None} right_desc={_r_desc}"
        )

        return self.dual_run_stage(
            left_stage,
            right_stage,
            objects,  # type: ignore[arg-type]
            action_description=action_description_override,
            extra_checkers=list(extra_checkers or []),
        )

    def dual_run_stage(
        self,
        left_stage: Optional[Stage],
        right_stage: Optional[Stage],
        objects: Dict[str, OmniObject],
        action_description: Optional[Dict[str, str]] = None,
        extra_checkers: Optional[List[Callable[[], bool]]] = None,
    ) -> Tuple[bool, str]:
        """Execute two stages synchronously at atomic step boundaries.
        Store the initial state once and initialize both candidate buffers. Prepare
        both actions, dispatch movement together, then finish each arm's step.
        On a retryable failure, restore both arms and advance failed-arm candidates;
        stop when retries or candidates are exhausted. Restore again on final failure.
        left_stage/right_stage may be None; objects are updated in place.
        Return (stage_success, store_name)."""
        store_name = f"dual_stage_{self.stage_id}"
        self.robot.client.store_current_state(store_name)
        logger.info(f"Dual store state {store_name}")

        if left_stage is None and right_stage is None:
            return True, store_name

        for arm_key, st in (("left", left_stage), ("right", right_stage)):
            if st is None:
                continue
            if not st.initialize_action_sequence_buffer(objects, self.robot):
                logger.warning(
                    f"dual_run_stage: {arm_key} stage init action sequence buffer failed"
                )
                return False, store_name

        self.stage_id += 1

        dual_extra_checkers: List[Callable[[], bool]] = list(extra_checkers or [])

        # Fetch initial candidates; fail if either active arm has none.
        left_seq: Optional[Any] = None
        right_seq: Optional[Any] = None
        if left_stage is not None:
            left_seq = left_stage.get_action_sequence()
            if left_seq is None:
                logger.warning(f"dual_run_stage: left stage has no action sequence")
                self.robot.client.playback(store_name)
                return False, store_name
        if right_stage is not None:
            right_seq = right_stage.get_action_sequence()
            if right_seq is None:
                logger.warning(f"dual_run_stage: right stage has no action sequence")
                self.robot.client.playback(store_name)
                return False, store_name

        attempt_count = 0
        stage_success = False
        try_next_sequence = True
        objects_origin_pose = {
            obj.name: obj.obj_pose.copy() for obj in objects.values()
        }

        while (
            not stage_success
            and try_next_sequence
            and attempt_count < self.DUAL_STAGE_MAX_ATTEMPTS
        ):
            attempt_count += 1
            stage_success = True

            # Keep the successful arm sequence and advance only the failed arm sequence.
            # Reset action_index to -1; synchronize it explicitly at each step.
            for seq in (left_seq, right_seq):
                if seq is not None:
                    seq.action_index = -1

            max_steps = max(
                len(left_seq) if left_seq is not None else 0,
                len(right_seq) if right_seq is not None else 0,
            )


            failed_arms: set = set()
            for step_idx in range(max_steps):
                left_action = self._safe_seq_get(left_seq, step_idx)
                right_action = self._safe_seq_get(right_seq, step_idx)

                # Synchronize active sequence indices so completion checks and
                # parse_action/get_current_action refer to this step.
                if left_seq is not None and step_idx < len(left_seq):
                    left_seq.action_index = step_idx
                if right_seq is not None and step_idx < len(right_seq):
                    right_seq.action_index = step_idx

                step_success, retryable, fa = self._dual_atomic_step(
                    left_stage,
                    right_stage,
                    left_action,
                    right_action,
                    objects,
                    attempt_count,
                    step_idx,
                    action_description_override=action_description,
                )

                if not step_success:
                    stage_success = False
                    try_next_sequence = retryable
                    failed_arms = fa
                    logger.warning(
                        f"Dual stage {self.stage_id} step {step_idx} failed "
                        f"by arms={failed_arms} (try_next_sequence={try_next_sequence})"
                    )
                    break

                # As in run_stage, check that protected objects have not moved.
                # Each arm config passes unmovable_objects into stage.extra_params.
                # If ownership of the displacement is unknown, fail both arms and advance both.
                merged_unmovable = set()
                for _st in (left_stage, right_stage):
                    if _st is None:
                        continue
                    merged_unmovable.update(_st.extra_params.get("unmovable_objects", []))
                for obj_id, obj in objects.items():
                    if obj_id == "gripper":
                        continue
                    if self._is_object_allowed_to_move(
                        obj_id, left_stage
                    ) or self._is_object_allowed_to_move(obj_id, right_stage):
                        continue
                    if obj_id not in merged_unmovable:
                        continue
                    position_diff, angle_diff = pose_difference(
                        obj.obj_pose, objects_origin_pose[obj_id]
                    )
                    if position_diff > 1e-2 or angle_diff > 5:
                        logger.warning(
                            f"Object {obj_id} moved during dual stage: "
                            f"position_diff {position_diff}, angle_diff {angle_diff}"
                        )
                        stage_success = False
                        try_next_sequence = True
                        failed_arms = {a for a, s in (("left", left_stage), ("right", right_stage)) if s is not None}
                        break
                if not stage_success:
                    break

            # Run each stage checker and mark the responsible arm on failure.
            # Then run shared checkers; their failures mark both arms.
            if stage_success:
                for arm_key, st in (("left", left_stage), ("right", right_stage)):
                    if st is None:
                        continue
                    for cfg in st.checker_config or []:
                        status = self.robot.client.get_checker_status(cfg).msg
                        logger.info(
                            f"Dual stage {self.stage_id} {arm_key} checker {cfg} status: {status}"
                        )
                        if status == "fail":
                            stage_success = False
                            try_next_sequence = True
                            failed_arms.add(arm_key)
                            break
                    if not stage_success:
                        break

            if stage_success:
                for checker in dual_extra_checkers:
                    try:
                        ok = checker()
                        if not ok:
                            self.robot.client.run_step(30)
                            logger.info(
                                f"Dual stage {self.stage_id} run_step 30, recheck {checker.__name__}"
                            )
                            ok = checker()
                    except Exception as e:
                        logger.error(
                            f"Dual stage {self.stage_id} extra checker {checker.__name__} raised: {e}"
                        )
                        ok = False
                    if not ok:
                        logger.warning(
                            f"Dual stage {self.stage_id} extra checker {checker.__name__} failed"
                        )
                        stage_success = False
                        try_next_sequence = True
                        failed_arms = {a for a, s in (("left", left_stage), ("right", right_stage)) if s is not None}
                        break

            # Advance only failed-arm candidates; rerun the successful arm sequence.
            # Stop if a required candidate buffer is exhausted.
            if not stage_success and try_next_sequence:
                self.robot.client.playback(store_name)
                logger.info(f"Dual playback state {store_name}")
                if "left" in failed_arms and left_stage is not None:
                    new_seq = left_stage.get_action_sequence()
                    if new_seq is None:
                        logger.info(
                            f"Dual stage {self.stage_id} left buffer exhausted, terminate"
                        )
                        try_next_sequence = False
                    else:
                        left_seq = new_seq
                if "right" in failed_arms and right_stage is not None:
                    new_seq = right_stage.get_action_sequence()
                    if new_seq is None:
                        logger.info(
                            f"Dual stage {self.stage_id} right buffer exhausted, terminate"
                        )
                        try_next_sequence = False
                    else:
                        right_seq = new_seq

        if not stage_success:
            logger.warning(f"Dual stage failed, final playback {store_name}")
            self.robot.client.playback(store_name)
        return stage_success, store_name

    @staticmethod
    def _safe_seq_get(seq, idx):
        if seq is None or idx >= len(seq):
            return None
        return seq[idx]

    @staticmethod
    def _is_object_allowed_to_move(obj_id, stage) -> bool:
        if stage is None:
            return False
        action_type = stage.action_type
        if action_type in ("insert", "place") and obj_id == stage.active_obj_id:
            return True
        if action_type in ("pick", "grasp", "rotate") and obj_id in (
            stage.passive_obj_id,
            stage.active_obj_id,
        ):
            return True
        return False

    def _dual_atomic_step(
        self,
        left_stage: Optional[Stage],
        right_stage: Optional[Stage],
        left_action: Optional[Action],
        right_action: Optional[Action],
        objects: Dict[str, OmniObject],
        attempt_index: int,
        step_index: int,
        action_description_override: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, bool, set]:
        """Prepare both arms, dispatch movement together, and finish each step.
        Return (step_success, retryable, failed_arms). Any arm failure fails the step;
        a failed arm requesting termination makes the step nonretryable. Track failed
        arms so the outer loop advances only their candidate sequences."""
        plans: Dict[str, Optional[dict]] = {"left": None, "right": None}
        for arm_key, st, action in (
            ("left", left_stage, left_action),
            ("right", right_stage, right_action),
        ):
            if st is None or action is None:
                continue
            plans[arm_key] = self._step_prepare(
                st, action, objects, attempt_index, step_index,
                action_description_override=action_description_override,
            )

        # Record step-start markers for both arms with target poses and the shared instruction.
        for arm_key, plan in plans.items():
            if plan is None:
                continue
            self.robot.client.set_frame_state(
                plan["stage"].action_type,
                step_index,
                plan["stage"].active_obj_id,
                plan["stage"].passive_obj_id,
                self.attached_obj_id is not None,
                arm=arm_key,
                target_pose=plan["target_gripper_pose"],
                action_description=action_description_override or plan["action_description"],
                error_description=plan["error_data"],
            )

        # Dispatch both arm movements together, including single-active-arm steps.
        move_states = self._dispatch_dual_step_move(plans["left"], plans["right"], objects)

        step_success = True
        retryable = True
        failed_arms: set = set()
        finish_results: Dict[str, Tuple[bool, bool]] = {}  # arm_key → (gripper_open, gripper_close)
        for arm_key in ("left", "right"):
            plan = plans[arm_key]
            if plan is None:
                continue
            arm_state, arm_retry, set_open, set_close = self._step_finish(
                plan, objects, move_states[arm_key]
            )
            finish_results[arm_key] = (set_open, set_close)
            if not arm_state:
                step_success = False
                # need_retry=False requests termination, matching single-arm step semantics.
                # If either failed arm requests termination, do not advance sequences.
                retryable = retryable and arm_retry
                failed_arms.add(arm_key)

        # Record step-finish markers with gripper and attachment states for both arms.
        for arm_key, plan in plans.items():
            if plan is None:
                continue
            set_open, set_close = finish_results.get(arm_key, (False, False))
            self.robot.client.set_frame_state(
                plan["stage"].action_type,
                step_index,
                plan["stage"].active_obj_id,
                plan["stage"].passive_obj_id,
                self.attached_obj_id is not None,
                set_open,
                set_close,
                arm=arm_key,
                target_pose=plan["target_gripper_pose"],
                action_description=action_description_override or plan["action_description"],
                error_description=plan["error_data"],
            )

        return step_success, retryable, failed_arms

    def _step_prepare(
        self,
        stage: Stage,
        action: Action,
        objects: Dict[str, OmniObject],
        attempt_index: int,
        step_index: int,
        action_description_override: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Parse the pre-move portion of step and return its execution-plan dictionary."""
        logger.info(
            f"==> [DUAL_RUN prepare] Stage {self.stage_id} attempt {attempt_index} "
            f"step {step_index}: {stage.action_type} arm={stage.extra_params.get('arm', 'right')}"
        )
        extra_params = stage.extra_params.copy()
        arm = extra_params.get("arm", "right")
        objects_view = self.update_objects(objects, arm=arm)

        (
            target_gripper_pose,
            motion_type,
            gripper_action,
            arm,
            action_description,
            action_extra_params,
        ) = stage.parse_action(action, objects_view)
        extra_params.update(action_extra_params)
        arm = extra_params.get("arm", "right")
        if action_description_override is not None:
            action_description = action_description_override

        motion_run_ratio = 1.0
        gripper_action_timing: dict = {}
        remove_obstacles = False
        error_data: dict = {}
        if "error_data" in extra_params:
            error_data = extra_params["error_data"]
            error_params = error_data.get("params", {})
            motion_run_ratio = error_params.get("motion_run_ratio", 1.0)
            if error_data.get("type") == "Drop":
                timing = error_params.get("drop_timing", 0.2)
                gripper_action_timing = {
                    "state": "open",
                    "timing": timing,
                    "is_right": arm == "right",
                }
                gripper_action = "open"
            elif error_data.get("type") == "KeepClose":
                if stage.action_type == "pick":
                    remove_obstacles = True

        # Dual preparation and finishing do not record frame markers themselves.
        # _dual_atomic_step records both arms together at each step boundary
        # to avoid mismatched arm frames and per-config
        # description validation conflicts.

        return {
            "stage": stage,
            "arm": arm,
            "attempt_index": attempt_index,
            "step_index": step_index,
            "target_gripper_pose": target_gripper_pose,
            "motion_type": motion_type,
            "gripper_action": gripper_action,
            "action_description": action_description,
            "error_data": error_data,
            "motion_run_ratio": motion_run_ratio,
            "gripper_action_timing": gripper_action_timing,
            "remove_obstacles": remove_obstacles,
            "goal_offset": extra_params.get("goal_offset", [0, 0, 0, 1, 0, 0, 0]),
            "path_constraint": extra_params.get("path_constraint", []),
            "from_current_pose": extra_params.get("from_current_pose", False),
            "offset_and_constraint_in_goal_frame": extra_params.get(
                "offset_and_constraint_in_goal_frame", True
            ),
            "disable_collision_links": extra_params.get("disable_collision_links", []),
            "slowdown_factor": float(extra_params.get("slowdown_factor", 1.0) or 1.0),
        }

    def _dispatch_dual_step_move(
        self,
        left_plan: Optional[dict],
        right_plan: Optional[dict],
        objects: Dict[str, OmniObject],
    ) -> Dict[str, bool]:
        """Dispatch plans synchronously when supported; otherwise use serial arm moves."""
        states: Dict[str, bool] = {"left": True, "right": True}
        active: Dict[str, dict] = {}
        for arm_key, plan in (("left", left_plan), ("right", right_plan)):
            if plan is None:
                continue
            if plan["remove_obstacles"]:
                self.robot.client.remove_objs_from_obstacle(
                    [objects[plan["stage"].passive_obj_id].prim_path]
                )
            if plan["target_gripper_pose"] is None:
                # Gripper-only operation; no movement is needed.
                continue
            active[arm_key] = plan

        if not active:
            return states

        if len(active) == 1:
            arm_key, plan = next(iter(active.items()))
            states[arm_key] = self._step_dispatch_move_single(plan)
            return states

        traj_arms = {a: p["motion_type"] in ("Simple", "Straight") for a, p in active.items()}
        if all(traj_arms.values()):
            success = self._dual_dispatch_trajectory(active)
            states["left"] = states["right"] = success
        elif not any(traj_arms.values()):
            success = self._dual_dispatch_curobo(active)
            states["left"] = states["right"] = success
        else:
            logger.warning(
                "dual_run_stage: mixed motion_type "
                f"(left={active.get('left',{}).get('motion_type')}, "
                f"right={active.get('right',{}).get('motion_type')}), "
                "falling back to sequential single-arm move"
            )
            for arm_key, plan in active.items():
                states[arm_key] = self._step_dispatch_move_single(plan)
        return states

    @staticmethod
    def _target_obstacle_path(stage: Stage):
        # Handled in _dispatch_dual_step_move using object prim paths; retain the placeholder.
        return stage.passive_obj_id

    def _step_dispatch_move_single(self, plan: dict) -> bool:
        if plan["motion_type"] == "Simple":
            plan["target_gripper_pose"] = self._maybe_resample_target_with_noise(
                plan["target_gripper_pose"],
                plan["arm"],
                plan["attempt_index"],
                plan["step_index"],
            )
        return bool(
            self.robot.move_pose(
                plan["target_gripper_pose"],
                plan["motion_type"],
                arm=plan["arm"],
                block=True,
                goal_offset=plan["goal_offset"],
                path_constraint=plan["path_constraint"],
                offset_and_constraint_in_goal_frame=plan[
                    "offset_and_constraint_in_goal_frame"
                ],
                disable_collision_links=plan["disable_collision_links"],
                motion_run_ratio=plan["motion_run_ratio"],
                gripper_action_timing=plan["gripper_action_timing"],
                from_current_pose=plan["from_current_pose"],
                slowdown_factor=float(plan.get("slowdown_factor", 1.0) or 1.0),
            )
        )

    def _dual_dispatch_trajectory(self, active: Dict[str, dict]) -> bool:
        moves = []
        for arm_key, plan in active.items():
            target_pose = plan["target_gripper_pose"]
            if plan["motion_type"] == "Simple":
                target_pose = self._maybe_resample_target_with_noise(
                    target_pose,
                    plan["arm"],
                    plan["attempt_index"],
                    plan["step_index"],
                )
                plan["target_gripper_pose"] = target_pose
            target_position, target_rotation = self._world_pose_to_local_pq(target_pose)
            moves.append(
                {
                    "arm": arm_key,
                    "target_position": list(target_position),
                    "target_rotation": list(target_rotation),
                    "ee_interpolation": plan["motion_type"] == "Straight",
                    "distance_frame": 0.0008,
                }
            )
        result = self.robot.client.dual_moveto(moves)
        return result.get("success", False) if isinstance(result, dict) else bool(result)

    def _dual_dispatch_curobo(self, active: Dict[str, dict]) -> bool:
        moves = []
        for arm_key, plan in active.items():
            target_pose = plan["target_gripper_pose"]
            target_position, target_rotation = self._world_pose_to_local_pq(target_pose)
            moves.append(
                {
                    "arm": arm_key,
                    "target_position": list(target_position),
                    "target_rotation": list(target_rotation),
                    "goal_offset": plan["goal_offset"],
                    "path_constraint": plan["path_constraint"],
                    "offset_and_constraint_in_goal_frame": plan[
                        "offset_and_constraint_in_goal_frame"
                    ],
                    "disable_collision_links": plan["disable_collision_links"],
                    "from_current_pose": plan["from_current_pose"],
                    "slowdown_factor": float(plan.get("slowdown_factor", 1.0) or 1.0),
                }
            )
        result = self.robot.client.dual_moveto_curobo(moves)
        return result.get("success", False) if isinstance(result, dict) else bool(result)

    def _world_pose_to_local_pq(self, pose):
        from common.base_utils.transform_utils import (
            euler2quat,
            mat2euler,
            quat2mat_wxyz,
            quat_xyzw_to_wxyz,
        )

        if isinstance(pose, list):
            pose = np.array(pose)
        init_rot_mat = quat2mat_wxyz(self.robot.init_rotation)
        T_world_robot = np.eye(4)
        T_world_robot[:3, :3] = init_rot_mat
        T_world_robot[:3, 3] = self.robot.init_position
        target = np.linalg.inv(T_world_robot) @ pose
        R, T = target[:3, :3], target[:3, 3]
        quat_wxyz = quat_xyzw_to_wxyz(euler2quat(mat2euler(R, order="zyx"), order="zyx"))
        return T, quat_wxyz

    def _maybe_resample_target_with_noise(self, target_gripper_pose, arm, index, step_index):
        ik_success, _ = self.robot.solve_ik(
            np.array([target_gripper_pose]),
            ee_type="gripper",
            arm=arm,
            type="Simple",
        )
        if ik_success[0]:
            return target_gripper_pose
        logger.info(
            f"Substage {index} step_id {step_index}: target gripper pose not reachable, try add noise"
        )
        candidates = []
        for _ in range(10):
            candidates.append(add_random_noise_to_pose(target_gripper_pose, rot_noise=10))
        candidates = np.array(candidates)
        ok, _ = self.robot.solve_ik(
            candidates, ee_type="gripper", arm=arm, type="Simple"
        )
        candidates = candidates[ok]
        if len(candidates) == 0:
            return target_gripper_pose
        dists = np.linalg.norm(candidates[:, :3, 3] - target_gripper_pose[:3, 3], axis=1)
        chosen = candidates[int(np.argmin(dists))]
        logger.info(
            f"Substage {index} step_id {step_index}: find reachable gripper pose"
        )
        return chosen

    def _step_finish(
        self,
        plan: dict,
        objects: Dict[str, OmniObject],
        move_state: bool,
    ) -> Tuple[bool, bool]:
        """Process the post-move portion of step; return (step_success, need_retry)."""
        stage = plan["stage"]
        arm = plan["arm"]
        action_type = stage.action_type
        active_id = stage.active_obj_id
        passive_id = stage.passive_obj_id
        target_gripper_pose = plan["target_gripper_pose"]
        action_description = plan["action_description"]
        error_data = plan["error_data"]
        gripper_action = plan["gripper_action"]
        index = plan["attempt_index"]
        step_index = plan["step_index"]


        if not move_state:
            logger.info(
                f"action_stages: {self.stage_id}, action: {action_type}, "
                f"[stage {index} step_id {step_index}] failed, move pose failed"
            )
            return False, True, False, False

        set_gripper_open = gripper_action == "open"
        set_gripper_close = gripper_action == "close"
        # Do not record frame state inside dual step finishing.
        # _dual_atomic_step records both arms together after the atomic step.
        self.robot.set_gripper_action(gripper_action, arm=arm)

        objects["gripper"].obj_pose = self.robot.get_ee_pose(ee_type="gripper", id=arm)
        self.update_objects(objects, arm=arm)
        step_success = stage.check_completion(objects, self.robot)
        if not step_success:
            self.attached_obj_id = None
            logger.info(
                f"action_stages: {self.stage_id}, action: {action_type}, "
                f"[stage {index} step_id {step_index}] failed, check completion failed"
            )
        else:
            logger.info(
                f"action_stages: {self.stage_id}, action: {action_type}, "
                f"[stage {index} step_id {step_index}] success"
            )
            if (
                gripper_action == "close"
                and action_type == "pick"
                and self.attached_obj_id != stage.passive_obj_id
            ):
                self.attached_obj_id = stage.passive_obj_id
                self.robot.client.attach_obj(
                    prim_paths=[objects[self.attached_obj_id].prim_path],
                    is_right=arm == "right",
                )
            elif gripper_action == "open":
                self.attached_obj_id = None
        self.update_objects(objects, arm=arm)
        # Return gripper flags to _dual_atomic_step so completion markers
        # include set_gripper_open/set_gripper_close for recording.
        return step_success, True, set_gripper_open, set_gripper_close

    def handover_ownership_of_attached_object(self, from_arm: str) -> bool:
        """Transfer ownership of the attached object to the other arm."""
        self.open_gripper(arm=from_arm)
        self.client.attach_obj([self.get_object_prim_path(self.attached_obj_id)], is_right = from_arm != "right")
        return True

    def _build_pick_stage(
        self,
        actor: Union[OmniObject, str],
        arm: str,
        grasp_config: GraspConfig,
        part: Optional[str] = None,
    ) -> Tuple[Stage, Dict[str, OmniObject]]:
        """Prepare the actor, register objects, and build its grasp stage."""
        if isinstance(actor, str) and not actor.endswith("/entity") and "/entity/" not in actor:
            actor = self.get_object_prim_path(actor)
        actor = self.get_object(actor, part=part)
        self.add_object_to_objects("gripper", OmniObject("gripper"))
        self.add_object_to_objects(actor.name, actor)
        objects = self.objects
        self.update_objects(objects, arm=arm)
        stage = self.get_grasp_stage(
            actor=actor, objects=objects, arm=arm, grasp_config=grasp_config,
        )
        return stage, objects

    def _build_place_to_actor_stage(
        self, actor: str, arm: str, target_actor: str, place_config: PlaceConfig,
    ) -> Tuple[Stage, Dict[str, OmniObject]]:
        if not actor.endswith("/entity"):
            actor = self.get_object_prim_path(actor)
        actor = self.get_object(actor)
        target_actor = self.get_object(target_actor)
        self.add_object_to_objects(target_actor.name, target_actor)
        objects = self.objects
        self.update_objects(objects, arm=arm)
        stage = self.get_place_stage(
            actor=actor, target_actor=target_actor, objects=objects, arm=arm,
            place_config=place_config,
        )
        return stage, objects

    def _build_place_to_pose_stage(
        self, actor: str, arm: str, target_pose: np.ndarray, place_config: PlaceConfig,
    ) -> Tuple[Stage, Dict[str, OmniObject]]:
        if not actor.endswith("/entity"):
            actor = self.get_object_prim_path(actor)
        objects = self.objects
        obj_id = "fix_pose_to_place_" + actor
        actor = self.get_object(actor)
        fix_pose_object = self.get_fix_pose_object(obj_id, target_pose[0][:3, 3], [0, 0, -1])
        self.add_object_to_objects(obj_id, fix_pose_object)
        self.update_objects(objects, arm=arm)
        stage = self.get_place_stage(
            actor=actor, target_actor=fix_pose_object, target_pose=target_pose,
            objects=objects, arm=arm, place_config=place_config,
        )
        return stage, objects

    def _build_move_to_pose_stage(
        self,
        actor: str,
        arm: str,
        target_pose: np.ndarray,
        move_config: MoveConfig,
        part: Optional[str] = None,
    ) -> Tuple[Stage, Dict[str, OmniObject]]:
        if target_pose is None or len(target_pose) == 0:
            raise ValueError("target_pose is None or empty")
        self.add_object_to_objects("gripper", OmniObject("gripper"))
        if actor != "gripper":
            if not actor.endswith("/entity") and "/entity/" not in actor:
                actor = self.get_object_prim_path(actor)
            actor = self.get_object(actor, part=part)
            self.add_object_to_objects(actor.name, actor)
        else:
            actor = OmniObject("gripper")
        fix_pose_obj_id = "fix_pose_to_place_" + actor.name
        fix_pose_object = self.get_fix_pose_object(
            fix_pose_obj_id, target_pose[0][:3, 3], [0, 0, -1]
        )
        self.add_object_to_objects(fix_pose_obj_id, fix_pose_object)
        objects = self.objects
        self.update_objects(objects, arm=arm)
        stage = self.get_move_stage(
            actor=actor, target_actor=fix_pose_object, target_pose=target_pose,
            objects=objects, arm=arm, move_config=move_config,
        )
        return stage, objects

    def _build_rotate_stage(
        self, actor: str, arm: str, target_pose: np.ndarray, rotate_config: RotateConfig,
    ) -> Tuple[Stage, Dict[str, OmniObject]]:
        if not actor.endswith("/entity"):
            actor = self.get_object_prim_path(actor)
        actor = self.get_object(actor)
        objects = self.objects
        self.update_objects(objects, arm=arm)
        self.add_object_to_objects("gripper", OmniObject("gripper"))
        self.add_object_to_objects(actor.name, actor)
        stage = self.get_rotate_stage(
            actor=actor, target_pose=target_pose, objects=objects, arm=arm,
            rotate_config=rotate_config,
        )
        return stage, objects

    def pick_actor(
        self,
        actor: Union[OmniObject, str],
        arm: str = "right",
        grasp_config: GraspConfig = GraspConfig(),
        part: str = None,
    ) -> bool:
        """Pick and lift an object or an articulated part; return success.
        actor can be an OmniObject or identifier; arm selects left/right.
        grasp_config controls candidates. part selects a child pose frame; custom
        world grasp poses are still converted relative to that part's world pose."""
        # Resolve an object pose first when actor is a string identifier.
        grasp_stage, objects = self._build_pick_stage(actor, arm, grasp_config, part=part)
        grasp_success, grasp_store_name = self.run_stage(grasp_stage, objects=objects)
        if not grasp_success:
            logger.error("Grasp stage failed")
            return False
        return True

    def place_actor_to_target_actor(
        self,
        actor: str,
        arm: str,
        target_actor: str,
        place_config: PlaceConfig = PlaceConfig(),
    ) -> bool:
        """Place actor on target_actor using the selected arm and place_config; return success."""
        place_stage, objects = self._build_place_to_actor_stage(
            actor, arm, target_actor, place_config,
        )
        place_success, place_store_name = self.run_stage(place_stage, objects=objects)
        if not place_success:
            logger.error("Place stage failed")
            return False
        return True

    def place_actor_to_target_pose(
        self,
        actor: str,
        arm: str,
        target_pose: np.ndarray,
        place_config: PlaceConfig = PlaceConfig(),
    ) -> bool:
        """Place actor at a candidate (N, 4, 4) target pose using arm and place_config; return success."""
        place_stage, objects = self._build_place_to_pose_stage(
            actor, arm, target_pose, place_config,
        )
        place_success, place_store_name = self.run_stage(place_stage, objects=objects)
        if not place_success:
            logger.error("Place stage failed")
            return False
        return True

    def pick_and_place_actor_to_target_actor(
        self,
        actor: str,
        target_actor: str,
        arm: str = "right",
        grasp_config: GraspConfig = GraspConfig(),
        place_config: PlaceConfig = PlaceConfig(),
    ) -> bool:
        """Pick actor and place it on target_actor using grasp/place configs; return success."""
        if not actor.endswith("/entity"):
            actor = self.get_object_prim_path(actor)
        actor = self.get_object(actor)
        self.add_object_to_objects("gripper", OmniObject("gripper"))
        self.add_object_to_objects(actor.name, actor)

        if not target_actor.endswith("/entity"):
            target_actor = self.get_object_prim_path(target_actor)
        target_actor = self.get_object(target_actor)
        self.add_object_to_objects(target_actor.name, target_actor)
        objects = self.objects
        self.update_objects(objects, arm=arm)
        grasp_stage = self.get_grasp_stage(
            actor=actor,
            objects=objects,
            arm=arm,
            grasp_config=grasp_config,
        )
        place_stage = self.get_place_stage(
            actor=actor,
            target_actor=target_actor,
            objects=objects,
            arm=arm,
            place_config=place_config,
        )
        # grasp_stage.next_stage = place_stage
        grasp_success, grasp_store_name = self.run_stage(grasp_stage, objects=objects)
        if not grasp_success:
            logger.error("Grasp stage failed")
            return False
        self.update_objects(objects, arm=arm)
        place_success, place_store_name = self.run_stage(place_stage, objects=objects)
        if not place_success:
            logger.error("Place stage failed")
            self.robot.client.playback(grasp_store_name)
            logger.info(f"Playback state {grasp_store_name}")
            return False
        return True

    def pick_and_place_actor_to_target_pose(
        self,
        actor: str,
        target_pose: np.ndarray,
        arm: str = "right",
        grasp_config: GraspConfig = GraspConfig(),
        place_config: PlaceConfig = PlaceConfig(),
    ) -> bool:
        """Pick actor and place it at a candidate (N, 4, 4) target pose; return success."""
        if target_pose is None or len(target_pose) == 0:
            logger.error("target_pose is None or empty")
            return False
        if not actor.endswith("/entity"):
            actor = self.get_object_prim_path(actor)
        actor = self.get_object(actor)
        fix_pose_obj_id = "fix_pose_to_place_" + actor.name
        self.add_object_to_objects("gripper", OmniObject("gripper"))
        self.add_object_to_objects(actor.name, actor)
        fix_pose_object = self.get_fix_pose_object(
            fix_pose_obj_id, target_pose[0][:3, 3], [0, 0, -1]  # place direction is -z
        )
        self.add_object_to_objects(fix_pose_obj_id, fix_pose_object)
        objects = self.objects
        self.update_objects(objects, arm=arm)
        grasp_stage = self.get_grasp_stage(
            actor=actor,
            objects=objects,
            arm=arm,
            grasp_config=grasp_config,
        )
        place_stage = self.get_place_stage(
            actor=actor,
            target_actor=fix_pose_object,
            target_pose=target_pose,
            objects=objects,
            arm=arm,
            place_config=place_config,
        )
        grasp_stage.next_stage = place_stage
        grasp_success, grasp_store_name = self.run_stage(grasp_stage, objects=objects)
        if not grasp_success:
            logger.error("Grasp stage failed")
            return False
        self.update_objects(objects, arm=arm)
        place_success, place_store_name = self.run_stage(place_stage, objects=objects)
        if not place_success:
            logger.error("Place stage failed")
            self.robot.client.playback(grasp_store_name)
            logger.info(f"Playback state {grasp_store_name}")
            return False
        return True

    def rotate_actor(
        self,
        actor: str,
        target_pose: np.ndarray,
        arm: str = "right",
        rotate_config: RotateConfig = RotateConfig(),
    ):
        """Rotate actor to a candidate (N, 4, 4) target pose with the selected arm.
        rotate_config.constraint_axises locks rotations about the selected axes;
        offset_and_constraint_in_goal_frame selects goal-frame or world-frame constraints."""
        rotate_stage, objects = self._build_rotate_stage(
            actor, arm, target_pose, rotate_config,
        )
        rotate_success, rotate_store_name = self.run_stage(
            rotate_stage, objects=objects
        )
        if not rotate_success:
            logger.error("Rotate stage failed")
            return False
        return True

    def move_actor_to_target_pose(
        self,
        actor: str,
        target_pose: np.ndarray,
        arm: str = "right",
        move_config: MoveConfig = MoveConfig(),
        part: Optional[str] = None,
    ):
        """Move actor, or its optional child part, to a target world pose.
        move_config.path_constraint is [rx, ry, rz, x, y, z]: zero leaves a dimension
        free and nonzero constrains it. For example, [0.1, 0.1, 0.1, 0, 0.1, 0.1]
        allows x translation. offset_and_constraint_in_goal_frame selects goal or world frame."""
        if target_pose is None or len(target_pose) == 0:
            logger.error("target_pose is None or empty")
            return False
        move_stage, objects = self._build_move_to_pose_stage(
            actor, arm, target_pose, move_config, part=part,
        )
        move_success, move_store_name = self.run_stage(move_stage, objects=objects)
        if not move_success:
            logger.error("Move stage failed")
            return False
        return True

    def follow_trajectory(
        self,
        actor: str,
        waypoints: np.ndarray,
        arm: str = "right",
        trajectory_config: FollowTrajectoryConfig = FollowTrajectoryConfig(),
        part: Optional[str] = None,
    ) -> bool:
        """Move a gripper or a grasped object through every ordered waypoint.

        ``waypoints`` uses world-frame xyz+wxyz values and accepts either
        ``(N, 7)`` for one path or ``(M, N, 7)`` for multiple path candidates.
        For an object (or ``part``), each waypoint describes the target object
        pose; the current object-to-end-effector transform is preserved.
        """
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        if not isinstance(trajectory_config, FollowTrajectoryConfig):
            raise ValueError("trajectory_config must be a FollowTrajectoryConfig")
        if not trajectory_config.action_description:
            raise ValueError("Follow trajectory action description is required")

        validate_trajectory_options(
            trajectory_config.noise_range,
            trajectory_config.noise_frame,
            trajectory_config.sample_count,
        )
        actor_pose_candidates = waypoints_to_poses(waypoints)

        gripper = OmniObject("gripper")
        self.add_object_to_objects("gripper", gripper)
        actor_is_gripper = actor == "gripper"
        if actor_is_gripper:
            actor_object = gripper
        else:
            try:
                actor_path = actor
                if not actor_path.endswith("/entity") and "/entity/" not in actor_path:
                    actor_path = self.get_object_prim_path(actor_path)
                actor_object = self.get_object(actor_path, part=part)
            except Exception as exc:
                logger.error(f"follow_trajectory: failed to resolve {actor} part={part}: {exc}")
                return False
            self.add_object_to_objects(actor_object.name, actor_object)

        objects = self.objects
        try:
            self.update_objects(objects, arm=arm)
            current_ee_pose = self.robot.get_ee_pose(ee_type="gripper", id=arm)
            if actor_is_gripper:
                actor_to_ee = np.eye(4)
            else:
                actor_to_ee = np.linalg.inv(actor_object.obj_pose) @ current_ee_pose
        except Exception as exc:
            logger.error(f"follow_trajectory: failed to read current poses: {exc}")
            return False

        def solve_ik(poses):
            ik_success, _ = self.robot.solve_ik(
                poses,
                ee_type="gripper",
                arm=arm,
                type="AvoidObs",
            )
            return ik_success

        resolved_trajectory = resolve_trajectory_candidates(
            actor_pose_candidates,
            actor_to_ee,
            solve_ik,
            noise_range=trajectory_config.noise_range,
            noise_frame=trajectory_config.noise_frame,
            sample_count=trajectory_config.sample_count,
        )
        if resolved_trajectory is None:
            logger.error("follow_trajectory: no complete IK-solvable trajectory candidate")
            return False

        trajectory_positions = resolved_trajectory[:, :3, 3].tolist()
        trajectory_orientations = [
            mat2quat_wxyz(pose[:3, :3]).tolist() for pose in resolved_trajectory
        ]
        command_config = {
            "command": "FollowTrajectory",
            "trajectory_positions": trajectory_positions,
            "trajectory_orientations": trajectory_orientations,
            "solvable_threshold": float(np.nextafter(1.0, 0.0)),
            "max_consecutive_failures": 1,
            "is_right": arm == "right",
        }

        config = trajectory_config.to_dict()
        arm_zh = "左" if arm == "left" else "右"
        arm_en = "left" if arm == "left" else "right"
        config.update(
            {
                "arm": arm,
                "active_id": actor_object.name,
                "passive_id": "",
                "if_attached": not actor_is_gripper,
                "action_type": "follow_trajectory",
                "action_description": {
                    "action_text": (
                        f"{arm_zh}臂"
                        f"{trajectory_config.action_description.get('zh', '')}"
                    ),
                    "english_action_text": (
                        f"{arm_en} arm "
                        f"{trajectory_config.action_description.get('en', '')}"
                    ),
                },
            }
        )

        def follow_action(try_time, action_config, action_objects):
            response = self.robot.client.command_any(command_config)
            success = isinstance(response, dict) and response.get("success") is True
            return success, True

        success, _ = self.run_simple_action(follow_action, config, objects)
        return bool(success)

    def bimanual_rigid_move(
        self,
        target_pose: np.ndarray,
        config: BimanualRigidMoveConfig = BimanualRigidMoveConfig(),
    ):
        """Bimanual rigid coordinated motion: leader arm plans to target, follower
        arm maintains the initial relative EE pose (T_rel) and follows rigidly.

        Both arms are IK-filtered; the best feasible candidate is selected.

        Args:
            target_pose: Leader arm target poses, shape (N, 4, 4) homogeneous matrices.
                         Candidates are IK-filtered for both arms; best one is used.
            config: BimanualRigidMoveConfig (action_description is required).
        """
        if target_pose is None or len(target_pose) == 0:
            logger.error("bimanual_rigid_move: target_pose is None or empty")
            return False
        leader_arm = config.leader_arm
        self.add_object_to_objects("gripper", OmniObject("gripper"))
        fix_pose_obj_id = "fix_pose_to_bimanual_move"
        fix_pose_object = self.get_fix_pose_object(
            fix_pose_obj_id, target_pose[0][:3, 3], [0, 0, -1]  # place direction is -z
        )
        self.add_object_to_objects(fix_pose_obj_id, fix_pose_object)
        objects = self.objects
        self.update_objects(objects)
        stage = self.get_bimanual_rigid_move_stage(
            target_pose=target_pose,
            target_actor=fix_pose_object,
            objects=objects,
            config=config,
        )
        success, store_name = self.run_stage(stage, objects=objects)
        if not success:
            logger.error("Bimanual rigid move stage failed")
            return False
        return True

    def shake(
        self,
        actor: str,
        angle_ranges: List[List[float]],
        shake_config: ShakeConfig = ShakeConfig(),
        arm: str = "right",
        horizontal: bool = True,
        times: int = 2,
    ):
        """Oscillate wrist angles within angle_range while abs(joint6) + abs(joint7) < 1.92."""
        assert "G2" in self.robot_cfg, "Only G2 robot can shake"
        config = shake_config.to_dict()
        config["angle_ranges"] = angle_ranges
        config["arm"] = arm
        config["horizontal"] = horizontal
        config["times"] = times
        config["active_id"] = "gripper"
        config["passive_id"] = actor
        config["if_attached"] = True
        config["action_type"] = "shake"
        self.add_object_to_objects("gripper", OmniObject("gripper"))
        if actor != "gripper":
            if not actor.endswith("/entity"):
                actor = self.get_object_prim_path(actor)
            actor = self.get_object(actor)
            self.add_object_to_objects(actor.name, actor)
        else:
            actor = OmniObject("gripper")
        objects = self.objects
        self.update_objects(objects, arm=arm)

        def shake_action(try_time, config, objects) -> Tuple[bool, bool]:
            # return stage_success, terminate
            if try_time >= len(angle_ranges):
                return False, True
            terminate = try_time == len(angle_ranges) - 1
            angle_range = angle_ranges[try_time]
            logger.info(f"shake action range: {angle_range}")
            current_joint_positions = self.get_joint_positions()
            joint_name_6 = "idx66_arm_r_joint6" if arm == "right" else "idx26_arm_l_joint6"
            joint_name_7 = "idx67_arm_r_joint7" if arm == "right" else "idx27_arm_l_joint7"
            joint_6_limit = 60 * np.pi / 180
            joint_7_limit = 90 * np.pi / 180
            joint_6 = current_joint_positions[joint_name_6]
            joint_7 = current_joint_positions[joint_name_7]
            target_joint_name = joint_name_6 if horizontal else joint_name_7
            current_angle = joint_6 if horizontal else joint_7
            joint_limit = joint_6_limit if horizontal else joint_7_limit
            other_abs = abs(joint_7) if horizontal else abs(joint_6)
            limit = 1.92 - other_abs  # Absolute angle limit for the moving joints.
            angle_min_raw = current_angle + angle_range[0]
            angle_max_raw = current_angle + angle_range[1]
            angle_min_raw = min(angle_max_raw, max(angle_min_raw, -joint_limit))
            angle_max_raw = max(angle_max_raw, min(angle_max_raw, joint_limit))
            angle_min = max(angle_min_raw, -limit)
            angle_max = min(angle_max_raw, limit)
            joint_names_ordered = list(current_joint_positions.keys())
            target_idx = joint_names_ordered.index(target_joint_name)
            base_positions = list(current_joint_positions.values())
            angle_range = np.abs(angle_max - angle_min)
            wait_time = angle_range * 0.5
            for _ in range(times):
                positions_min = list(base_positions)
                positions_min[target_idx] = angle_min
                if not self.set_joint_positions(positions_min, joint_names_ordered, is_trajectory=True):
                    return False, terminate
                self.wait(wait_time)
                positions_max = list(base_positions)
                positions_max[target_idx] = angle_max
                if not self.set_joint_positions(positions_max, joint_names_ordered, is_trajectory=True):
                    return False, terminate
                self.wait(wait_time)

                base_positions = positions_max
            # set_back
            positions_back = list(base_positions)
            positions_back[target_idx] = current_angle
            if not self.set_joint_positions(positions_back, joint_names_ordered, is_trajectory=True):
                return False, terminate
            self.wait(wait_time)
            return True, terminate
        if shake_config.action_description is None or not len(
            shake_config.action_description
        ):
            raise ValueError("Move config action description is required")
        arm_zh = "左" if arm == "left" else "右"
        arm_en = "left" if arm == "left" else "right"
        config["action_description"] = {
            "action_text": f"{arm_zh}臂{shake_config.action_description.get('zh', '')}",
            "english_action_text": f"{arm_en} arm {shake_config.action_description.get('en', '')}",
        }
        logger.info(f"shake stage description: {config['action_description']}")
        success, store_name = self.run_simple_action(shake_action, config, objects)
        return success

    def turn_to(
        self,
        target_point: List[float],
        turn_body_config: TurnBodyConfig = TurnBodyConfig(),
    ) -> bool:
        """Turn the G2 upper body toward a world target using idx05_body_joint5.
        Use target x/y and interpolate from the current angle with step_deg spacing.
        turn_body_config must provide action_description. Return success."""
        assert "G2" in self.robot_cfg, "Only G2 robot can turn"

        if turn_body_config.action_description is None or not len(turn_body_config.action_description):
            raise ValueError("TurnBodyConfig action_description is required")

        from scipy.spatial.transform import Rotation as R

        body_link5_pose = self.robot.get_prim_world_pose("/G2/body_link5")
        body_link5_pos = body_link5_pose[:3, 3]

        robot_quat_wxyz = np.array(self.init_rotation)
        quat_xyzw = [robot_quat_wxyz[1], robot_quat_wxyz[2], robot_quat_wxyz[3], robot_quat_wxyz[0]]
        robot_yaw = R.from_quat(quat_xyzw).as_euler('xyz')[2]

        dx = target_point[0] - body_link5_pos[0]
        dy = target_point[1] - body_link5_pos[1]
        target_world_yaw = np.arctan2(dy, dx)

        target_joint_angle = target_world_yaw - robot_yaw
        target_joint_angle = (target_joint_angle + np.pi) % (2 * np.pi) - np.pi

        joint_lower = -3.2
        joint_upper = 3.2
        target_joint_angle = float(np.clip(target_joint_angle, joint_lower, joint_upper))

        config = turn_body_config.to_dict()
        config["action_type"] = "turn"
        config["active_id"] = "gripper"
        config["passive_id"] = "gripper"
        config["if_attached"] = False
        config["arm"] = "right"
        config["target_joint_angle"] = target_joint_angle
        config["action_description"] = {
            "action_text": turn_body_config.action_description.get("zh", "转身"),
            "english_action_text": turn_body_config.action_description.get("en", "turn body"),
        }

        self.add_object_to_objects("gripper", OmniObject("gripper"))
        objects = self.objects
        self.update_objects(objects, arm="right")

        def turn_action(try_time, config, objects) -> Tuple[bool, bool]:
            terminate = try_time >= 1
            target_angle = config["target_joint_angle"]
            step_rad = np.deg2rad(config["step_deg"])

            current_joint_positions = self.get_joint_positions()
            joint_name = "idx05_body_joint5"
            current_angle = current_joint_positions[joint_name]

            diff = target_angle - current_angle
            num_steps = max(1, int(np.ceil(np.abs(diff) / step_rad)))

            joint_names_ordered = list(current_joint_positions.keys())
            target_idx = joint_names_ordered.index(joint_name)
            base_positions = list(current_joint_positions.values())
            wait_time = step_rad * 0.5

            for i in range(1, num_steps + 1):
                ratio = i / num_steps
                interp_angle = current_angle + diff * ratio
                positions = list(base_positions)
                positions[target_idx] = interp_angle
                if not self.set_joint_positions(positions, joint_names_ordered, is_trajectory=True):
                    logger.error(f"turn_to failed at step {i}/{num_steps}")
                    return False, terminate
                # self.wait(wait_time)

            logger.info(
                f"turn_to completed: {np.rad2deg(current_angle):.1f}° -> {np.rad2deg(target_angle):.1f}°"
            )
            return True, terminate

        logger.info(f"turn stage description: {config['action_description']}")
        success, store_name = self.run_simple_action(turn_action, config, objects)
        return success

    def joint_to(
        self,
        joint_names: Union[List[str], str, Dict[str, float]],
        target_joints: Optional[Dict[str, float]] = None,
        steps: int = 50,
        delta: bool = False,
        action_description: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Move selected joints by linear interpolation; delta=True interprets values as increments."""
        if isinstance(joint_names, dict):
            if target_joints is not None:
                raise ValueError("target_joints must be None when joint_names is a dict")
            target_joints = joint_names
            joint_names = list(target_joints.keys())
        elif isinstance(joint_names, str):
            joint_names = [joint_names]

        if not joint_names:
            raise ValueError("joint_names cannot be empty")
        if not target_joints:
            raise ValueError("target_joints cannot be empty")
        missing_targets = [name for name in joint_names if name not in target_joints]
        if missing_targets:
            raise ValueError(f"target_joints missing values for joints: {missing_targets}")
        extra_targets = [name for name in target_joints if name not in joint_names]
        if extra_targets:
            raise ValueError(f"target_joints contains joints not listed in joint_names: {extra_targets}")
        if steps <= 0:
            raise ValueError("steps must be positive")

        config = {
            "joint_targets": {name: target_joints[name] for name in joint_names},
            "steps": steps,
            "delta": delta,
            "action_type": "joint_to",
            "active_id": "gripper",
            "passive_id": "gripper",
            "if_attached": False,
            "arm": "right",
            "unmovable_objects": [],
            "extra_checkers": [],
            "action_description": action_description or {
                "action_text": "移动关节",
                "english_action_text": "move joints",
            },
        }

        self.add_object_to_objects("gripper", OmniObject("gripper"))
        objects = self.objects
        self.update_objects(objects, arm="right")

        def joint_to_action(try_time, config, objects) -> Tuple[bool, bool]:
            terminate = try_time >= 1
            current_joint_positions = self.get_joint_positions()
            missing_joints = [name for name in config["joint_targets"] if name not in current_joint_positions]
            if missing_joints:
                logger.error(f"joint_to unknown joints: {missing_joints}")
                return False, terminate

            joint_names_ordered = list(current_joint_positions.keys())
            base_positions = list(current_joint_positions.values())
            joint_indices = {name: joint_names_ordered.index(name) for name in config["joint_targets"]}
            target_positions = {}
            for name, value in config["joint_targets"].items():
                if config["delta"]:
                    target_positions[name] = current_joint_positions[name] + value
                else:
                    target_positions[name] = value

            for i in range(1, config["steps"] + 1):
                ratio = i / config["steps"]
                positions = list(base_positions)
                for name, target_value in target_positions.items():
                    idx = joint_indices[name]
                    positions[idx] = current_joint_positions[name] + (target_value - current_joint_positions[name]) * ratio
                if not self.set_joint_positions(positions, joint_names_ordered, is_trajectory=True):
                    logger.error(f"joint_to failed at step {i}/{config['steps']}")
                    return False, terminate

            logger.info(f"joint_to completed: delta={config['delta']}, joints={list(config['joint_targets'].keys())}")
            return True, terminate

        logger.info(f"joint_to stage description: {config['action_description']}")
        success, store_name = self.run_simple_action(joint_to_action, config, objects)
        return success

    def open(
        self,
        actor: str,
        joint_name: str = None,
        open_config: OpenConfig = OpenConfig(),
        arm: str = "right",
    ):
        """Follow a server-generated revolute-joint trajectory to open an articulated object.
        Move to the handle and close the gripper before calling. actor identifies the
        scene object; joint_name optionally selects an explicit joint; arm is left/right.
        open_config requires action_description and may include extra_checkers to
        validate the final joint angle. An empty trajectory returns False."""
        config = open_config.to_dict()
        config["arm"] = arm
        config["joint_name"] = joint_name
        config["active_id"] = "gripper"
        config["passive_id"] = actor
        config["if_attached"] = True
        config["action_type"] = "open"
        articulation_object_prim_path = self.get_object_prim_path(actor)
        self.add_object_to_objects("gripper", OmniObject("gripper"))
        if actor != "gripper":
            if not actor.endswith("/entity"):
                actor = self.get_object_prim_path(actor)
            actor = self.get_object(actor)
            self.add_object_to_objects(actor.name, actor)
        else:
            actor = OmniObject("gripper")
        objects = self.objects
        self.update_objects(objects, arm=arm)
        # get trajectory
        command_config = {}
        command_config["command"] = "GetRevoluteJointTrajectory"
        command_config["articulation_object_prim_path"] = articulation_object_prim_path
        command_config["object_prim_path"] = "left_gripper" if arm == "left" else "right_gripper"
        command_config["target_limit"] = config["target_limit"]
        command_config["trajectory_step_deg"] = config["trajectory_step_deg"]
        command_config["joint_prim_path"] = joint_name
        trajectory_resp = self.robot.client.command_any(command_config)
        trajectory_positions = trajectory_resp.get("trajectory_positions", [])
        trajectory_orientations = trajectory_resp.get("trajectory_orientations", [])
        if len(trajectory_positions) == 0 or len(trajectory_orientations) == 0:
            logger.error(f"Get revolute joint trajectory failed for {articulation_object_prim_path}")
            return False
        def open_action(try_time, config, objects) -> Tuple[bool, bool]:
            # return stage_success, terminate
            # follow trajectory
            command_config = {}
            command_config["command"] = "FollowTrajectory"
            command_config["trajectory_positions"] = trajectory_positions
            command_config["trajectory_orientations"] = trajectory_orientations
            command_config["solvable_threshold"] = config["solvable_threshold"]
            command_config["max_consecutive_failures"] = config["max_consecutive_failures"]
            command_config["is_right"] = arm == "right"
            follow_resp = self.robot.client.command_any(command_config)
            if follow_resp.get("success", False):
                return True, True
            else:
                return False, True
        if open_config.action_description is None or not len(open_config.action_description):
            raise ValueError("Open config action description is required")
        arm_zh = "左" if arm == "left" else "右"
        arm_en = "left" if arm == "left" else "right"
        config["action_description"] = {
            "action_text": f"{arm_zh}臂{open_config.action_description.get('zh', '')}",
            "english_action_text": f"{arm_en} arm {open_config.action_description.get('en', '')}",
        }
        logger.info(f"open stage description: {config['action_description']}")
        success, store_name = self.run_simple_action(open_action, config, objects)
        return success

    def get_object_size_in_world_frame(self, actor_name: str) -> List[float] | None:
        """Transform local box vertices to world and return world AABB dimensions [sx, sy, sz]."""
        position, quat_wxyz = self.get_object_pose(actor_name)
        position = np.asarray(position)
        quat_wxyz = np.asarray(quat_wxyz)
        local_size = self.get_object_size_in_local_frame(actor_name)
        if local_size is None:
            return None
        half = np.array(local_size, dtype=np.float64) / 2.0
        vertices_local = np.array(
            [
                [-half[0], -half[1], -half[2]],
                [half[0], -half[1], -half[2]],
                [half[0], half[1], -half[2]],
                [-half[0], half[1], -half[2]],
                [-half[0], -half[1], half[2]],
                [half[0], -half[1], half[2]],
                [half[0], half[1], half[2]],
                [-half[0], half[1], half[2]],
            ],
            dtype=np.float64,
        )
        pose = get_pose_wxyz(position, quat_wxyz)
        vertices_world = transform_points(vertices_local, pose)
        aabb_min = np.min(vertices_world, axis=0)
        aabb_max = np.max(vertices_world, axis=0)
        world_size = aabb_max - aabb_min
        return world_size.tolist()

    def get_object_size_in_local_frame(self, actor_name: str) -> List[float] | None:
        """Return object dimensions in its local frame."""
        usd_id = self.get_object_usd_id(actor_name)
        if usd_id not in ASSETS_INDEX:
            logger.error(f"USD {usd_id} not found in ASSETS_INDEX")
            return None
        asset_info = ASSETS_INDEX[usd_id]
        shapes = asset_info.get("shapes", [])
        size = None
        quaternion = None
        for shape in shapes:
            if shape.get("name", "") == "bbox":
                size = shape.get("size", None)
                quaternion = shape.get("quaternion", None)
                break
        if size is None or quaternion is None:
            for shape in shapes:
                if shape.get("name", "") != "origin":
                    size = shape.get("size", None)
                    quaternion = shape.get("quaternion", None)
                    break
        if size is None or quaternion is None:
            logger.error(f"Cannot find size and quaternion for {actor_name}")
            return None
        rotation_matrix = quat2mat(np.array(quaternion))
        size = np.array(size)
        size = np.abs(rotation_matrix @ size)
        to_local_orientation = None
        if "z" == asset_info["up_axis"]:
            return size
        elif "x" == asset_info["up_axis"]:
            to_local_orientation = [0.70711, 0, -0.70711, 0]
        elif "y" == asset_info["up_axis"]:
            to_local_orientation = [0.70711, 0.70711, 0, 0]
        if to_local_orientation is not None:
            # Apply inverse to_local_orientation to the size.
            local_rot_mat = quat2mat_wxyz(np.array(to_local_orientation))
            inv_rot_mat = np.linalg.inv(local_rot_mat)
            size = np.abs(inv_rot_mat @ size)
        return size

    # Internal helpers.
    def add_all_objects_info(self):
        layout = self.scene_info.get("layout", {})
        for object_id, object_info in layout.items():
            object_prim_path = self.get_object_prim_path(object_id)
            object_usd_id = self.get_object_usd_id(object_id)
            # Objects absent from ASSETS_INDEX may already be included in scene payloads.
            # Skip add_object for them; runtime dynamic loading is unnecessary
            # when scene.usda already references the object.
            if not object_usd_id or object_usd_id not in ASSETS_INDEX:
                logger.warning(
                    f"add_all_objects_info: skip {object_id}(usd={object_info.get('usd')!r})"
                    f" — not in ASSETS_INDEX"
                )
                continue
            object_url = ASSETS_INDEX[object_usd_id].get("url", "")
            # Deformable objects must not be wrapped as rigid bodies.
            # SingleRigidPrim velocity queries on deformable meshes can cause
            # CUDA illegal memory access; detect deformable/cloth layout keywords.
            kw = set(object_info.get("keywords", []) or [])
            is_deformable = bool(kw & {"deformable", "cloth", "softbody", "fem"})
            self.client.add_object(
                usd_path=object_url,
                prim_path=object_prim_path,
                # following parameters are not used in llm gen mode
                label_name=object_id,
                target_position=[0, 0, 0],
                target_quaternion=[1, 0, 0, 0],
                target_scale=[1, 1, 1],
                color=[1, 1, 1],
                material="general",
                add_rigid_body=not is_deformable,
            )

    def run_simple_action(self, action: Callable, config: dict, objects: Dict[str, OmniObject]):
        """Execute a simple action."""
        store_name = f"stage_{self.stage_id}"
        self.robot.client.store_current_state(store_name)
        logger.info(f"Store state {store_name}")
        attempt_count = 0
        stage_success = False
        MAX_ATTEMPTIONS = 3
        self.stage_id += 1
        objects_origin_pose = {
            obj.name: obj.obj_pose.copy() for obj in objects.values()
        }
        terminate = False
        while (
            not stage_success and attempt_count < MAX_ATTEMPTIONS and not terminate
        ):
            # Execution
            stage_success = True
            attempt_count += 1
            self.robot.client.set_frame_state(
                config.get("action_type", ""),
                0,
                config.get("active_id", ""),
                config.get("passive_id", ""),
                config.get("if_attached", False),
                arm=config.get("arm", "right"),
                action_description=config.get("action_description", {}),
            )
            stage_success, terminate = action(attempt_count - 1, config, objects)
            self.robot.client.set_frame_state(
                config.get("action_type", ""),
                0,
                config.get("active_id", ""),
                config.get("passive_id", ""),
                config.get("if_attached", False),
                arm=config.get("arm", "right"),
                action_description=config.get("action_description", {}),
            )
            # check no other objects are moved
            for obj_id, obj in objects.items():
                if obj_id == "gripper":
                    continue
                if obj_id in config.get("unmovable_objects", []):
                    position_diff, angle_diff = pose_difference(
                        obj.obj_pose, objects_origin_pose[obj_id]
                    )
                    if position_diff > 1e-2 or angle_diff > 5:
                        logger.warning(
                            f"Object {obj_id} is moved: position_diff {position_diff}, angle_diff {angle_diff}"
                        )
                        stage_success = False
                        break
            if not stage_success and not terminate:
                self.robot.client.playback(store_name)
                logger.info(f"Playback state {store_name}")
                continue
            if stage_success and len(config.get("extra_checkers", [])) > 0:
                extra_checkers = config.get("extra_checkers", [])
                for extra_checker in extra_checkers:
                    try:
                        success = extra_checker()
                        if not success:
                            self.robot.client.run_step(30)
                            logger.info(f"Run step 30, to check extra checker again")
                            success = extra_checker()
                            if not success:
                                logger.warning(
                                    f"Extra checker failed: {extra_checker.__name__}, stage {self.stage_id} "
                                )
                                stage_success = False
                    except Exception as e:
                        logger.error(
                            f"Extra checker failed: {extra_checker.__name__}, stage {self.stage_id}, has exception error: {e}"
                        )
                        stage_success = False

                    if not stage_success and not terminate:
                        self.robot.client.playback(store_name)
                        logger.info(f"Playback state {store_name}")
                        break
        if not stage_success:
            logger.warning(f"Stage {self.stage_id} failed")
            self.robot.client.playback(store_name)
            logger.info(f"Playback state {store_name}")
        return stage_success, store_name

    def step(
        self,
        action: Action,
        stage: Stage,
        objects: Dict[str, OmniObject],
        index: int = 0,
        step_index: int = 0,
    ) -> bool:
        logger.info(
            f"==> [RUN] Stage {self.stage_id} Substage {index} Step {step_index}: {stage.action_type}"
        )
        step_success = False
        need_retry = True
        extra_params = stage.extra_params.copy()
        arm = extra_params.get("arm", "right")
        action_type = stage.action_type
        active_id = stage.active_obj_id
        passive_id = stage.passive_obj_id

        objects = self.update_objects(objects, arm=arm)

        (
            target_gripper_pose,
            motion_type,
            gripper_action,
            arm,
            action_description,
            action_extra_params,
        ) = stage.parse_action(action, objects)
        extra_params.update(action_extra_params)
        arm = extra_params.get("arm", "right")
        motion_run_ratio = 1.0
        remove_obstacles = False
        gripper_action_timing = {}
        error_data = {}
        if "error_data" in extra_params:
            error_data = extra_params["error_data"]
            error_params = error_data.get("params", {})
            motion_run_ratio = error_params.get("motion_run_ratio", 1.0)
            if error_data.get("type", None) == "Drop":
                timing = error_params.get("drop_timing", 0.2)
                gripper_action_timing = {
                    "state": "open",
                    "timing": timing,
                    "is_right": arm == "right",
                }
                gripper_action = "open"
            elif error_data.get("type", None) == "KeepClose":
                if action_type == "pick":
                    remove_obstacles = True

        self.robot.client.set_frame_state(
            action_type,
            step_index,
            active_id,
            passive_id,
            self.attached_obj_id is not None,
            arm=arm,
            target_pose=target_gripper_pose,
            action_description=action_description,
            error_description=error_data,
        )

        # execution action
        goal_offset = extra_params.get("goal_offset", [0, 0, 0, 1, 0, 0, 0])
        path_constraint = extra_params.get("path_constraint", [])
        from_current_pose = extra_params.get("from_current_pose", False)
        offset_and_constraint_in_goal_frame = extra_params.get(
            "offset_and_constraint_in_goal_frame", True
        )
        disable_collision_links = extra_params.get("disable_collision_links", [])
        if remove_obstacles:
            self.robot.client.remove_objs_from_obstacle(
                [objects[stage.passive_obj_id].prim_path]
            )

        # check if target_gripper_pose is reachable

        if target_gripper_pose is None:
            state = True
        else:
            if motion_type == "Simple":
                ik_success, _ = self.robot.solve_ik(
                    np.array([target_gripper_pose]),
                    ee_type="gripper",
                    arm=arm,
                    type="Simple",
                )
                if not ik_success[0]:
                    logger.info(
                        f"Substage {index} step_id {step_index}: target gripper pose not reachable, try add noise"
                    )
                    noisy_target_gripper_poses = []
                    for _ in range(10):
                        noisy_target_gripper_poses.append(
                            add_random_noise_to_pose(target_gripper_pose, rot_noise=10)
                        )
                    noisy_target_gripper_poses = np.array(noisy_target_gripper_poses)
                    ik_success, _ = self.robot.solve_ik(
                        noisy_target_gripper_poses,
                        ee_type="gripper",
                        arm=arm,
                        type="Simple",
                    )
                    noisy_target_gripper_poses = noisy_target_gripper_poses[ik_success]
                    if len(noisy_target_gripper_poses) > 0:
                        dists = np.linalg.norm(
                            noisy_target_gripper_poses[:, :3, 3]
                            - target_gripper_pose[:3, 3],
                            axis=1,
                        )
                        min_index = np.argmin(dists)
                        target_gripper_pose = noisy_target_gripper_poses[min_index]
                        logger.info(
                            f"Substage {index} step_id {step_index}: find reachable gripper pose"
                        )
            state = self.robot.move_pose(
                target_gripper_pose,
                motion_type,
                arm=arm,
                block=True,
                goal_offset=goal_offset,
                path_constraint=path_constraint,
                offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
                disable_collision_links=disable_collision_links,
                motion_run_ratio=motion_run_ratio,
                gripper_action_timing=gripper_action_timing,
                from_current_pose=from_current_pose,
                max_consecutive_failures=extra_params.get("max_consecutive_failures", 1),
                slowdown_factor=extra_params.get("slowdown_factor", 1.0),
            )
        if not state:
            step_success = False
            need_retry = True
            logger.info(
                f"action_stages: {self.stage_id}, action: {action_type}, [stage {index} step_id {step_index}] failed, move pose failed"
            )
            return step_success, need_retry
        else:
            set_gripper_open = gripper_action == "open"
            set_gripper_close = gripper_action == "close"
            self.robot.client.set_frame_state(
                action_type,
                step_index,
                active_id,
                passive_id,
                self.attached_obj_id is not None,
                set_gripper_open,
                set_gripper_close,
                arm=arm,
                target_pose=target_gripper_pose,
                action_description=action_description,
                error_description=error_data,
            )
            self.robot.set_gripper_action(gripper_action, arm=arm)

            self.robot.client.set_frame_state(
                action_type,
                step_index,
                active_id,
                passive_id,
                self.attached_obj_id is not None,
                arm=arm,
                target_pose=target_gripper_pose,
                action_description=action_description,
                error_description=error_data,
            )

            # check sub-stage completion
            objects["gripper"].obj_pose = self.robot.get_ee_pose(
                ee_type="gripper", id=arm
            )
            objects = self.update_objects(objects, arm=arm)

            step_success = stage.check_completion(objects, self.robot)
            self.robot.client.set_frame_state(
                action_type,
                step_index,
                active_id,
                passive_id,
                self.attached_obj_id is not None,
                arm=arm,
                target_pose=target_gripper_pose,
                action_description=action_description,
                error_description=error_data,
            )
            if not step_success:
                self.attached_obj_id = None
                logger.info(
                    f"action_stages: {self.stage_id}, action: {action_type}, [stage {index} step_id {step_index}] failed, check completion failed"
                )
            else:
                logger.info(
                    f"action_stages: {self.stage_id}, action: {action_type}, [stage {index} step_id {step_index}] success"
                )
                if not len(stage.active_action_sequence.actions) == step_index + 1:
                    logger.info("jump to next action")
                else:
                    logger.info("\n")

                if (
                    gripper_action == "close"
                    and action_type == "pick"
                    and self.attached_obj_id != stage.passive_obj_id
                ):
                    self.attached_obj_id = stage.passive_obj_id
                    arm = stage.extra_params.get("arm", "right")
                    self.robot.client.attach_obj(
                        prim_paths=[objects[self.attached_obj_id].prim_path],
                        is_right=arm == "right",
                    )
                elif gripper_action == "open":
                    self.attached_obj_id = None
                self.robot.client.set_frame_state(
                    action_type,
                    step_index,
                    active_id,
                    passive_id,
                    self.attached_obj_id is not None,
                    arm=arm,
                    target_pose=target_gripper_pose,
                    action_description=action_description,
                    error_description=error_data,
                )
            objects = self.update_objects(objects, arm=arm)
        return step_success, need_retry

    def run_stage(self, stage: Stage, objects: Dict[str, OmniObject]) -> bool:
        store_name = f"stage_{self.stage_id}"
        self.robot.client.store_current_state(store_name)
        logger.info(f"Store state {store_name}")
        if not stage.initialize_action_sequence_buffer(objects, self.robot):
            return False, store_name
        attempt_count = 0
        stage_success = False
        try_next_sequence = True
        MAX_ATTEMPTIONS = 3
        self.stage_id += 1
        objects_origin_pose = {
            obj.name: obj.obj_pose.copy() for obj in objects.values()
        }
        while (
            not stage_success and try_next_sequence and attempt_count < MAX_ATTEMPTIONS
        ):
            # Execution
            stage_success = True
            attempt_count += 1
            action_sequence = stage.get_action_sequence()
            if not action_sequence:
                # no more action sequence to try, fail this stage
                stage_success = False
                try_next_sequence = False
                logger.info(
                    f"Stage {self.stage_id} {stage.action_type} failed due to no more action sequence"
                )
                break
            for action in action_sequence:
                step_success, try_next_sequence = self.step(
                    action,
                    stage,
                    objects,
                    attempt_count,
                    action_sequence.get_step_index(),
                )
                if not step_success:
                    # for first step fail, try next sequence. Otherwise, fail this stage
                    if try_next_sequence:
                        logger.warning(
                            f"Stage {self.stage_id} {stage.action_type} fail at first step, try next sequence"
                        )
                    else:
                        logger.warning(
                            f"Stage {self.stage_id} {stage.action_type} fail at step {action_sequence.get_step_index()}, fail this stage"
                        )
                    stage_success = False
                    break
                # check no other objects are moved
                for obj_id, obj in objects.items():
                    if obj_id == "gripper":
                        continue
                    if (
                        stage.action_type == "insert" or stage.action_type == "place"
                    ) and obj_id == stage.active_obj_id:
                        continue
                    if (
                        stage.action_type == "pick"
                        or stage.action_type == "grasp"
                        or stage.action_type == "rotate"
                    ) and (
                        obj_id == stage.passive_obj_id or obj_id == stage.active_obj_id
                    ):
                        continue
                    if obj_id in stage.extra_params.get("unmovable_objects", []):
                        position_diff, angle_diff = pose_difference(
                            obj.obj_pose, objects_origin_pose[obj_id]
                        )
                        if position_diff > 1e-2 or angle_diff > 5:
                            logger.warning(
                                f"Object {obj_id} is moved: position_diff {position_diff}, angle_diff {angle_diff}"
                            )
                            stage_success = False
                            try_next_sequence = True
                            break
                if not stage_success:
                    break
            if (
                not stage_success
                and try_next_sequence
                and not stage.action_sequence_buffer.empty()
            ):
                self.robot.client.playback(store_name)
                logger.info(f"Playback state {store_name}")
            elif stage.checker_config:
                for check_config in stage.checker_config:
                    checker_status = self.robot.client.get_checker_status(
                        check_config
                    ).msg
                    logger.info(
                        f"stage finish: {self.stage_id}, checker: {check_config}, status: {checker_status}"
                    )
                    if checker_status == "fail":
                        stage_success = False
                        try_next_sequence = True
                        if not stage.action_sequence_buffer.empty():
                            self.robot.client.playback(store_name)
                            logger.info(f"Playback state {store_name}")
                        break
            if stage_success and len(stage.extra_params.get("extra_checkers", [])) > 0:
                extra_checkers = stage.extra_params.get("extra_checkers", [])
                for extra_checker in extra_checkers:
                    try:
                        success = extra_checker()
                        if not success:
                            self.robot.client.run_step(30)
                            logger.info(f"Run step 30, to check extra checker again")
                            success = extra_checker()
                            if not success:
                                logger.warning(
                                    f"Extra checker failed: {extra_checker.__name__}, stage {self.stage_id} {stage.action_type}"
                                )
                                stage_success = False
                    except Exception as e:
                        logger.error(
                            f"Extra checker failed: {extra_checker.__name__}, stage {self.stage_id} {stage.action_type}, has exception error: {e}"
                        )
                        stage_success = False

                    if not stage_success:
                        try_next_sequence = True
                        if not stage.action_sequence_buffer.empty():
                            self.robot.client.playback(store_name)
                            logger.info(f"Playback state {store_name}")
                        break
        if not stage_success:
            logger.warning(f"Stage {self.stage_id} {stage.action_type} failed")
            self.robot.client.playback(store_name)
            logger.info(f"Playback state {store_name}")
        return stage_success, store_name

    def get_grasp_stage(
        self,
        actor: OmniObject,
        objects: Dict[str, OmniObject],
        arm: str = "right",
        grasp_config: GraspConfig = GraspConfig(),
    ) -> Stage:
        """Build a PickStage for actor using the objects, arm, and grasp configuration."""
        config_dict = grasp_config.to_dict()

        stage_config = {
            "action": "pick",
            "active": {
                "object_id": "gripper",
                "primitive": grasp_config.active_primitive,
            },
            "passive": {
                "object_id": actor.name,
                "primitive": grasp_config.passive_primitive,
            },
            "extra_params": {
                "arm": arm,
                **config_dict,
            },
        }

        # Include the supplied action description.
        if grasp_config.action_description is None or not len(
            grasp_config.action_description
        ):
            raise ValueError("Grasp config action description is required")
        arm_zh = "左" if arm == "left" else "右"
        arm_en = "left" if arm == "left" else "right"
        stage_config["action_description"] = {
            "action_text": f"{arm_zh}臂{grasp_config.action_description.get('zh', '')}",
            "english_action_text": f"{arm_en} arm {grasp_config.action_description.get('en', '')}",
        }
        logger.info(f"pick stage description: {stage_config['action_description']}")
        pick_stage = PickStage(stage_config, objects)
        return pick_stage

    def get_place_stage(
        self,
        actor: OmniObject,
        target_actor: OmniObject,
        objects: Dict[str, OmniObject],
        target_pose: Optional[np.ndarray] = None,
        arm: str = "right",
        place_config: PlaceConfig = PlaceConfig(),
    ) -> Stage:
        """Build a placement stage using actor, target_actor, objects, arm, and place configuration."""
        config_dict = place_config.to_dict()
        action = (
            "place"
            if place_config.pre_insert_offset is None
            or place_config.pre_insert_offset == 0.0
            else "insert"
        )
        stage_config = {
            "action": action,
            "active": {
                "object_id": actor.name,
                "primitive": place_config.active_primitive,
            },
            "passive": {
                "object_id": target_actor.name,
                "primitive": place_config.passive_primitive,
            },
            "extra_params": {
                "arm": arm,
                **config_dict,
            },
        }

        # Include the supplied action description.
        if place_config.action_description is None or not len(
            place_config.action_description
        ):
            raise ValueError("Place config action description is required")
        arm_zh = "左" if arm == "left" else "右"
        arm_en = "left" if arm == "left" else "right"
        stage_config["action_description"] = {
            "action_text": f"{arm_zh}臂{place_config.action_description.get('zh', '')}",
            "english_action_text": f"{arm_en} arm {place_config.action_description.get('en', '')}",
        }
        if target_pose is not None:
            stage_config["extra_params"]["target_pose"] = target_pose
        logger.info(f"place stage description: {stage_config['action_description']}")
        return (
            PlaceStage(stage_config, objects)
            if action == "place"
            else InsertStage(stage_config, objects)
        )

    def get_rotate_stage(
        self,
        actor: OmniObject,
        target_pose: np.ndarray,
        objects: Dict[str, OmniObject],
        arm: str = "right",
        rotate_config: RotateConfig = RotateConfig(),
    ):
        """Build a rotation stage."""
        config_dict = rotate_config.to_dict()
        stage_config = {
            "action": "rotate",
            "active": {
                "object_id": actor.name,
                "primitive": None,
            },
            "passive": {
                "object_id": actor.name,
                "primitive": None,
            },
            "extra_params": {
                "arm": arm,
                "target_pose": target_pose,
                **config_dict,
            },
        }
        if rotate_config.action_description is None or not len(
            rotate_config.action_description
        ):
            raise ValueError("Grasp config action description is required")
        arm_zh = "左" if arm == "left" else "右"
        arm_en = "left" if arm == "left" else "right"
        stage_config["action_description"] = {
            "action_text": f"{arm_zh}臂{rotate_config.action_description.get('zh', '')}",
            "english_action_text": f"{arm_en} arm {rotate_config.action_description.get('en', '')}",
        }
        logger.info(f"rotate stage description: {stage_config['action_description']}")
        return ConstraintRotateStage(stage_config, objects)

    def get_move_stage(
        self,
        actor: OmniObject,
        target_actor: OmniObject,
        target_pose: np.ndarray,
        objects: Dict[str, OmniObject],
        arm: str = "right",
        move_config: MoveConfig = MoveConfig(),
    ):
        config_dict = move_config.to_dict()
        stage_config = {
            "action": "move",
            "active": {
                "object_id": actor.name,
                "primitive": move_config.active_primitive,
            },
            "passive": {
                "object_id": target_actor.name,
                "primitive": move_config.passive_primitive,
            },
            "extra_params": {
                "arm": arm,
                **config_dict,
            },
        }
        if move_config.action_description is None or not len(
            move_config.action_description
        ):
            raise ValueError("Move config action description is required")
        # strip_arm_prefix in move_config.extra_params can disable the arm subject.
        # This supports tasks whose instruction already describes the intended action.
        # Default False retains the arm prefix.
        _strip_arm = bool(move_config.extra_params.get("strip_arm_prefix", False))
        if _strip_arm:
            stage_config["action_description"] = {
                "action_text": move_config.action_description.get("zh", ""),
                "english_action_text": move_config.action_description.get("en", ""),
            }
        else:
            arm_zh = "左" if arm == "left" else "右"
            arm_en = "left" if arm == "left" else "right"
            stage_config["action_description"] = {
                "action_text": f"{arm_zh}臂{move_config.action_description.get('zh', '')}",
                "english_action_text": f"{arm_en} arm {move_config.action_description.get('en', '')}",
            }
        logger.info(f"move stage description: {stage_config['action_description']}")
        if target_pose is not None:
            stage_config["extra_params"]["target_pose"] = target_pose
        return MoveStage(stage_config, objects)

    def get_bimanual_rigid_move_stage(
        self,
        target_actor: OmniObject,
        target_pose: np.ndarray,
        objects: Dict[str, OmniObject],
        config: BimanualRigidMoveConfig = BimanualRigidMoveConfig(),
    ) -> Stage:
        config_dict = config.to_dict()
        leader_arm = config.leader_arm
        follower_arm = "left" if leader_arm == "right" else "right"
        stage_config = {
            "action": "bimanual_rigid_move",
            "active": {
                "object_id": "gripper",
                "primitive": None,
            },
            "passive": {
                "object_id": target_actor.name,
                "primitive": None,
            },
            "extra_params": {
                "arm": leader_arm,
                "follower_arm": follower_arm,
                "target_pose": target_pose,
                **config_dict,
            },
        }
        if config.action_description is None or not len(config.action_description):
            raise ValueError("BimanualRigidMoveConfig action_description is required")
        stage_config["action_description"] = {
            "action_text": f"双臂{config.action_description.get('zh', '')}",
            "english_action_text": f"both arm {config.action_description.get('en', '')}",
        }
        logger.info(f"bimanual rigid move stage description: {stage_config['action_description']}")
        return BimanualRigidMoveStage(stage_config, objects)

    def get_fix_pose_object(
        self,
        obj_id: str,
        position: List[float],
        direction: List[float],
        N_align: int = 1,
    ) -> OmniObject:
        obj = OmniObject(obj_id)
        obj.set_pose(np.array(position), np.array([0.001, 0.001, 0.001]))
        obj.elements = {
            "active": {},
            "passive": {
                "place": {
                    "default": [
                        {
                            "xyz": np.array([0, 0, 0]),
                            "direction": np.array(direction),
                            "angle_sample_num": N_align,
                        }
                    ]
                }
            },
        }
        return obj

    def get_object(self, actor: str, part: str = None):
        """Resolve an actor prim to an OmniObject, optionally referencing an articulated part.
        For a part, name is base_name/part and pose_prim selects its world pose;
        parent_prim_path remains the full entity path for DOF queries."""
        if actor.endswith("/entity"):
            actor_name = actor[:-7]
            actor_name = actor_name.split("/")[-1]
        else:
            actor_name = actor.split("/")[-1]
        cache_key = f"{actor_name}/{part}" if part else actor_name
        if self.objects.get(cache_key, None) is not None:
            return self.objects[cache_key]
        # Use the selected part world pose, or the full object pose when no part is given.
        actor_pose = self.get_object_pose(actor_name, articulation_part=part) if part else self.get_object_pose(actor)
        if actor_pose is None:
            logger.error(f"Cannot get the pose of object {actor}")
            return None

        usd_id = self.get_object_usd_id(actor_name)
        asset_info = ASSETS_INDEX[usd_id]
        object_info = {}
        # Use base/part as the object ID to match update_objects part handling.
        object_info["object_id"] = cache_key
        shapes = asset_info["shapes"]
        size = None
        for shape in shapes:
            if shape.get("name", "") == "bbox":
                size = shape.get("size", None)
                if size is not None:
                    object_info["size"] = size
                    break
        if size is None:
            if "dimensions" in asset_info["description"]:
                object_info["size"] = asset_info["description"]["dimensions"]
        object_info["scale"] = 1.0
        object_info["unit"] = (
            asset_info["description"]["unit"]
            if "unit" in asset_info["description"]
            else "m"
        )
        object_info["upAxis"] = asset_info["up_axis"]
        object_info["mass"] = 0.05
        object_info["semantic_name"] = (
            asset_info["description"]["semantic_name"][0]
            if "semantic_name" in asset_info["description"]
            else ""
        )
        object_info["model_path"] = asset_info["url"]
        object_info["data_info_dir"] = os.path.dirname(
            os.path.join(ASSETS_PATH, asset_info["url"])
        )
        object_info["obj_path"] = os.path.join(ASSETS_PATH, asset_info["url"])
        object_info["scene_object"] = False
        object_info["position"] = actor_pose[0]
        object_info["quaternion"] = actor_pose[1]
        is_key_object = len(asset_info.get("interaction", {})) > 0
        # Keep the full entity path for DOF queries.
        base_prim_path = actor if actor.endswith("/entity") else self.get_object_prim_path(actor_name)
        object_info["prim_path"] = base_prim_path
        if part is not None:
            # Set pose_part so OmniObject.pose_prim resolves base_prim_path/part.
            object_info["pose_part"] = part
        elif asset_info.get("has_articulation", False):
            articulation_part = "body"
            object_info["pose_part"] = articulation_part
        return OmniObject.from_obj_dir(
            obj_dir=object_info["data_info_dir"],
            obj_info=object_info,
            is_key_object=is_key_object,
        )

    def get_all_objects(self):
        scene_info = self.scene_info
        layout = scene_info.get("layout", {})
        for obj_id, obj_info in layout.items():
            usd_id = self.get_object_usd_id(obj_id)
            # As in add_all_objects_info, skip metadata for objects absent from ASSETS_INDEX.
            # Scene payload objects may lack data_info_dir and articulation metadata;
            # the server reads them directly from USD by prim path.
            if not usd_id or usd_id not in ASSETS_INDEX:
                continue
            asset_info = ASSETS_INDEX[usd_id]
            articulation_part = None
            if asset_info.get("has_articulation", False):
                logger.info(f"{obj_id}: Object has articulation, using default part: body")
                articulation_part = "body"
            object_path = self.get_object_prim_path(obj_id)
            object_element = self.get_object(object_path)
            self.add_object_to_objects(object_element.name, object_element)
        return

    def update_objects(self, objects, arm="right"):
        # update gripper pose
        objects["gripper"].obj_pose = self.robot.get_ee_pose(ee_type="gripper", id=arm)

        # update object pose
        for obj_id in objects:
            if obj_id == "gripper":
                continue
            if "fix_pose" in obj_id:
                if len(objects[obj_id].obj_pose) == 3:
                    position = objects[obj_id].obj_pose
                    rotation_matrix = calculate_rotation_matrix(
                        objects[obj_id].direction, [0, 0, 1]
                    )
                    objects[obj_id].obj_pose = np.eye(4)
                    objects[obj_id].obj_pose[:3, 3] = position.flatten()
                    objects[obj_id].obj_pose[:3, :3] = rotation_matrix
                continue
            if "/" in obj_id and getattr(objects[obj_id], "is_articulated", False):
                part_name = obj_id.split("/")[1]
                dof_joint = self.robot.client.get_part_dof_joint(
                    objects[obj_id].parent_prim_path, part_name
                )
                objects[obj_id].joint_position = dof_joint.joint_position
                objects[obj_id].joint_velocity = dof_joint.joint_velocity

            objects[obj_id].obj_pose = self.robot.get_prim_world_pose(
                objects[obj_id].pose_prim
            )
            if (
                hasattr(objects[obj_id], "info")
                and "simple_place" in objects[obj_id].info
                and objects[obj_id].info["simple_place"]
            ):
                down_direction_world = (
                    np.linalg.inv(objects[obj_id].obj_pose) @ np.array([0, 0, -1, 1])
                )[:3]
                down_direction_world = (
                    down_direction_world / np.linalg.norm(down_direction_world) * 0.08
                )
                objects[obj_id].elements["active"]["place"][
                    "direction"
                ] = down_direction_world
        return objects

    def add_object_to_objects(self, object_id: str, object_element: OmniObject):
        if object_id in self.objects:
            logger.warning(f"Object {object_id} already exists")
            return
        print(f"add object {object_id}")
        self.objects[object_id] = object_element

    def get_object_usd_id(self, object_id: str) -> str:
        layout = self.scene_info.get("layout", {})
        object_info = layout.get(object_id, {})
        usd_id = object_info.get("usd", "")
        if usd_id not in ASSETS_INDEX:
            logger.error(f"USD {usd_id} not found in ASSETS_INDEX")
            return ""
        return usd_id

    # Gripper control.

    def control_gripper(self, *, arm: str, opening_ratio: float, speed: float,
                        max_effort: float, mode: str = "position",
                        stiffness: Optional[float] = None, damping: Optional[float] = None,
                        tolerance: float = 0.001, timeout: float = 5.0,
                        minimum_duration: float = 0.0, completion: str = "target",
                        stall_time: float = 0.2) -> Dict[str, Any]:
        """Opt-in actuator control; legacy open/close methods retain their behavior.

        Ratio uses configured native closed/open coordinates, clipped to physical
        limits. Speed/tolerance are m/s,m for linear or rad/s,rad for angular.
        max_effort is N or Nm. Position mode requires positive current or explicit
        stiffness. Successful drive settings persist; timeout/error stops and
        restores previous settings. No automatic detach or grasp certification.
        """
        command = make_gripper_command(arm=arm, opening_ratio=opening_ratio,
            speed=speed, max_effort=max_effort, mode=mode, stiffness=stiffness,
            damping=damping, tolerance=tolerance, timeout=timeout,
            minimum_duration=minimum_duration, completion=completion, stall_time=stall_time)
        response = self.client.command_any(command)
        if not isinstance(response, dict) or not isinstance(response.get("success"), bool):
            raise RuntimeError(f"Invalid ControlGripper response: {response!r}")
        return response

    def open_gripper(self, arm: str = "right", opened_width: float = 1.0) -> None:
        """Open the selected arm gripper; opened_width=1.0 means fully open."""
        self.robot.open_gripper(id=arm, width=opened_width)

    def close_gripper(self, arm: str = "right") -> None:
        """Close the selected arm gripper; opened_width=0.0 means fully closed."""
        self.robot.close_gripper(id=arm)

    # Object operations.

    def get_ee_pose(self, arm: str = "right") -> np.ndarray:
        """Return the selected arm end-effector pose as a 4x4 homogeneous matrix."""
        return self.robot.get_ee_pose(id=arm)

    def get_joint_positions(self) -> Dict[str, float]:
        """Return the joint-name to position mapping."""
        response = self.client.get_joint_positions()
        joint_positions = {}
        for joint_state in response.states:
            joint_positions[joint_state.name] = joint_state.position
        return joint_positions

    def set_joint_positions(
        self,
        target_joint_position: List[float],
        target_joint_names: Optional[List[str]] = None,
        is_trajectory: bool = False,
    ) -> bool:
        """Set target joint positions with optional names and trajectory mode; return success."""
        response = self.client.set_joint_positions(
            target_joint_position=target_joint_position,
            target_joint_names=target_joint_names,
            is_trajectory=is_trajectory,
        )
        return response.errmsg == "Move Joint"

    def add_object(
        self,
        usd_path: str,
        prim_path: str,
        label_name: str,
        target_position: List[float],
        target_quaternion: List[float],
        target_scale: List[float],
        color: List[float],
        material: str,
        mass: float = 0.01,
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        add_rigid_body: bool = True,
        model_type: str = "convexDecomposition",
    ) -> str:
        """Add a scene object under /World/Objects and return its label_name.
        Configure the USD path, prim path, xyz position/scale, wxyz rotation, RGB color,
        material, mass, friction, rigid-body flag, and model type."""
        response = self.client.add_object(
            usd_path=usd_path,
            prim_path=prim_path,
            label_name=label_name,
            target_position=target_position,
            target_quaternion=target_quaternion,
            target_scale=target_scale,
            color=color,
            material=material,
            mass=mass,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            add_rigid_body=add_rigid_body,
            model_type=model_type,
        )
        return response.label_name

    # FEM cloth pins and nodal positions.
    # Pin any surface-deformable mesh to a selected link.
    # Task data_gen.py assembles grasp/attach/fold/release sequences.

    def _gripper_anchor_link(self, arm: str) -> str:
        """Resolve the selected arm's gripper center path for pin anchors.
        Prefer task_info robot.gripper_anchor_link, then robot-config
        end_effector_center_prim_path, then /G2/gripper_{l,r}_center_link."""
        # 1. Explicit task override.
        try:
            override = self.config.robot_config.get("gripper_anchor_link") or {}
            link = override.get(arm)
            if link:
                return link
        except Exception:
            pass

        # 2. Robot configuration gripper.end_effector_center_prim_path.
        try:
            cfg_name = self.config.robot_config.get("robot_cfg")
            if cfg_name:
                # Locate config/robot_cfg relative to the repository layout.
                here = os.path.dirname(os.path.abspath(__file__))
                # From client/llm_gen/robot_api.py, the repository root is two parents up.
                repo_root = os.path.abspath(os.path.join(here, "..", ".."))
                cfg_path = os.path.join(repo_root, "config", "robot_cfg", cfg_name)
                if os.path.isfile(cfg_path):
                    with open(cfg_path, "r") as f:
                        rcfg = json.load(f)
                    link = (rcfg.get("gripper", {})
                                .get("end_effector_center_prim_path", {})
                                .get(arm))
                    if link:
                        return link
        except Exception as exc:
            logger.warning(f"[_gripper_anchor_link] Failed to read robot_cfg; using defaults: {exc}")

        # 3. Fallback.
        return "/G2/gripper_l_center_link" if arm == "left" else "/G2/gripper_r_center_link"

    def get_cloth_nodal_positions(
        self,
        cloth_prim: str,
        indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read FEM node world positions for cloth_prim as an (N, 3) array.
        Optional indices select nodes in the requested order. Raise RuntimeError
        when the server view is unavailable or FEM is disabled."""
        cmd = {"command": "GetClothNodalPositions", "cloth_prim": cloth_prim}
        if indices is not None:
            cmd["indices"] = list(indices)
        resp = self.robot.client.command_any(cmd)
        if not resp.get("success", False):
            raise RuntimeError(
                f"GetClothNodalPositions failed: {resp.get('error')} "
                f"(cloth_prim={cloth_prim})"
            )
        return np.asarray(resp["positions"], dtype=np.float64)

    def set_fem_pin(
        self,
        arm: Optional[str],
        cloth_prim: str,
        radius: float = 0.03,
        gain: float = 3.0,
        vmax: float = 3.0,
        group_id: Optional[str] = None,
        anchor_link: Optional[str] = None,
        axis: str = "xy",
    ) -> int:
        """Attach a velocity-servo pin to FEM nodes near the current anchor.
        arm selects the gripper anchor; an explicit anchor_link overrides it.
        radius selects nodes using xy or 3d distance (axis). gain scales position-error
        velocity and vmax caps it. Reattaching group_id replaces that group; the
        anchor path is its default. Return the pinned node count; reject empty selections
        or unavailable views."""
        if anchor_link is None:
            if arm is None:
                raise ValueError("set_fem_pin: provide at least one of arm or anchor_link")
            anchor_link = self._gripper_anchor_link(arm)
        cmd = {
            "command": "FEMPinAttach",
            "cloth_prim": cloth_prim,
            "anchor_link": anchor_link,
            "radius": radius,
            "gain": gain,
            "vmax": vmax,
            "axis": axis,
        }
        if group_id is not None:
            cmd["group_id"] = group_id
        resp = self.robot.client.command_any(cmd)
        if not resp.get("success", False):
            raise RuntimeError(
                f"FEMPinAttach failed: {resp.get('error')} "
                f"(cloth_prim={cloth_prim}, anchor={anchor_link}, radius={radius})"
            )
        n = int(resp.get("attached_count", 0))
        logger.info(
            f"[fem_pin] attached {n}/{resp.get('n_total')} nodes "
            f"(arm={arm}, group={resp.get('group_id')}, radius={radius})"
        )
        return n

    def release_fem_pin(self, group_id: str = "all") -> List[str]:
        """Release one pin group or all groups, zero node velocities, and return released IDs."""
        resp = self.robot.client.command_any({
            "command": "FEMPinRelease",
            "group_id": group_id,
        })
        return list(resp.get("released", []))

    def set_rigid_pin(
        self,
        rigid_prim: str,
        arm: Optional[str] = None,
        anchor_link: Optional[str] = None,
        mode: str = "pose",
        gain_lin: float = 3.0,
        gain_ang: float = 3.0,
        vmax_lin: float = 3.0,
        vmax_ang: float = 6.0,
        group_id: Optional[str] = None,
    ) -> str:
        """Freeze a prim pose relative to an anchor and enforce it each physics step.
        arm selects the gripper anchor; anchor_link overrides it. With neither, lock
        the current world pose. Any valid prim is supported; rigid bodies use rigid
        views and other prims use Xform views.
        mode="pose" directly writes the target pose and zeros rigid-body velocities.
        mode="velocity" limits per-step translation and injects linear/angular velocities
        to retain collision response; nonrigid prims fall back to pose mode.
        gain_lin/gain_ang scale servo commands; vmax_lin/vmax_ang cap speeds.
        group_id defaults to rigid_prim and replaces an existing group with that ID.
        Return the active group ID; invalid prims or anchors raise RuntimeError."""
        if mode not in ("pose", "velocity"):
            raise ValueError(f"set_rigid_pin: mode must be 'pose' or 'velocity', got {mode!r}")
        if anchor_link is None and arm is not None:
            anchor_link = self._gripper_anchor_link(arm)
        cmd: dict = {
            "command": "RigidPinAttach",
            "rigid_prim": rigid_prim,
            "mode": mode,
            "gain_lin": gain_lin,
            "gain_ang": gain_ang,
            "vmax_lin": vmax_lin,
            "vmax_ang": vmax_ang,
        }
        if anchor_link is not None:
            cmd["anchor_link"] = anchor_link
        if group_id is not None:
            cmd["group_id"] = group_id
        resp = self.robot.client.command_any(cmd)
        if not resp.get("success", False):
            raise RuntimeError(
                f"RigidPinAttach failed: {resp.get('error')} "
                f"(rigid_prim={rigid_prim}, anchor={anchor_link}, mode={mode})"
            )
        gid = str(resp.get("group_id"))
        logger.info(
            f"[rigid_pin] attached rigid={rigid_prim} anchor={anchor_link} "
            f"group={gid} mode={mode}"
        )
        return gid

    def release_rigid_pin(self, group_id: str = "all") -> List[str]:
        """Release one rigid pin group or all groups, zero velocities, and return released IDs."""
        resp = self.robot.client.command_any({
            "command": "RigidPinRelease",
            "group_id": group_id,
        })
        return list(resp.get("released", []))

    def set_gripper_colliders(
        self,
        arm: str = "both",
        enabled: bool = True,
        links: Optional[List[str]] = None,
    ) -> int:
        """Enable or disable gripper base/finger colliders for left, right, or both arms.
        Disable after pinning cloth to prevent contacts from obstructing motion, then
        restore on release. links optionally overrides link suffixes. Return changed mesh count."""
        cmd = {
            "command": "SetGripperColliders",
            "arm": arm,
            "enabled": bool(enabled),
        }
        if links is not None:
            cmd["links"] = list(links)
        resp = self.robot.client.command_any(cmd)
        return int(resp.get("n", 0))

    def disable_gripper_colliders(self, arm: str = "both") -> int:
        """Disable gripper colliders for the selected arm."""
        return self.set_gripper_colliders(arm=arm, enabled=False)

    def enable_gripper_colliders(self, arm: str = "both") -> int:
        """Restore gripper colliders for the selected arm."""
        return self.set_gripper_colliders(arm=arm, enabled=True)

    # Debug visualization.

    def draw_debug_cube(
        self,
        name: str,
        position: Optional[List[float]] = None,
        orientation: Optional[List[float]] = None,
        pose_4x4: Optional[np.ndarray] = None,
        size: float = 0.02,
        color: Optional[List[float]] = None,
    ) -> str:
        """Create or update a named oriented cube marker without accumulating duplicates.
        Use world xyz/wxyz or pose_4x4 (which takes precedence), size in meters,
        and RGB color in [0, 1]. Return the server prim path."""
        if pose_4x4 is not None:
            T = np.asarray(pose_4x4, dtype=np.float64)
            pos = T[:3, 3].tolist()
            # rot mat → quat wxyz
            R = T[:3, :3]
            tr = R[0, 0] + R[1, 1] + R[2, 2]
            if tr > 0:
                s = np.sqrt(tr + 1.0) * 2
                w = 0.25 * s
                x = (R[2, 1] - R[1, 2]) / s
                y = (R[0, 2] - R[2, 0]) / s
                z = (R[1, 0] - R[0, 1]) / s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
            orient = [float(w), float(x), float(y), float(z)]
        else:
            pos = list(position or [0.0, 0.0, 0.0])
            orient = list(orientation or [1.0, 0.0, 0.0, 0.0])

        resp = self.robot.client.command_any({
            "command": "SpawnDebugCube",
            "name": name,
            "position": [float(c) for c in pos],
            "orientation": [float(c) for c in orient],
            "size": float(size),
            "color": [float(c) for c in (color or [1.0, 0.0, 0.0])],
        })
        return resp.get("prim_path", "")

    def clear_debug_cubes(self) -> int:
        """Delete all cubes under /World/_DebugCubes and return the count."""
        resp = self.robot.client.command_any({"command": "ClearDebugCubes"})
        return int(resp.get("removed", 0))

    def attach_obj(self, prim_paths: List[str], arm: str) -> bool:
        """Attach prim_paths to the selected left/right arm; return success."""
        response = self.client.attach_obj(
            prim_paths=prim_paths, is_right=(arm == "right")
        )
        return response.msg is not None

    def detach_obj(self) -> bool:
        """Detach the currently attached object; return success."""
        response = self.client.detach_obj()
        return response.msg is not None

    def remove_objs_from_obstacle(self, prim_paths: List[str]) -> bool:
        """Remove prim_paths from the obstacle list; return success."""
        response = self.client.remove_objs_from_obstacle(prim_paths=prim_paths)
        return response.msg is not None

    def add_objs_to_obstacle(self, prim_paths: List[str]) -> bool:
        """Add prim_paths to the obstacle list; return success."""
        response = self.client.command_any(
            {
                "command": "AddObjsToObstacle",
                "obj_prim_paths": prim_paths,
            }
        )
        return response is not None

    def set_object_parameter_action(
        self,
        prim_path: str,
        attribute_key: str,
        attribute_value,
        delay: float = 0.0,
        description: Optional[Dict[str, str]] = None,
        reference: str = "world",
    ) -> bool:
        """Set a USD attribute, optionally recording action boundaries with zh/en text.
        Before applying, advance round(delay * 60) simulation steps; advance five more
        after the server applies it. Attribute values must be JSON serializable.
        reference="world" converts translate and wxyz orient using the parent world
        transform; reference="local" writes directly. Return success."""
        action_description = {}
        if description and len(description):
            action_description = {
                "action_text": description.get("zh", ""),
                "english_action_text": description.get("en", ""),
            }

        action_type = "set_object_parameter"
        if action_description:
            self.robot.client.set_frame_state(
                action_type,
                0,
                prim_path,
                "",
                False,
                action_description=action_description,
            )

        pre_delay_frames = max(int(round(delay * 60)), 0)
        if pre_delay_frames > 0:
            self.client.run_step(pre_delay_frames)

        response = self.client.command_any(
            {
                "command": "SetObjectParameter",
                "prim_path": prim_path,
                "attribute_key": attribute_key,
                "attribute_value": attribute_value,
                "reference": reference,
            }
        )

        self.client.run_step(60)

        if action_description:
            self.robot.client.set_frame_state(
                action_type,
                0,
                prim_path,
                "",
                False,
                action_description=action_description,
            )

        return response == "success"

    # Other operations.

    def solve_ik(
        self,
        target_poses: np.ndarray,
        arm: str = "right",
        ee_type: str = "gripper",
        type: str = "Simple",
        ObsAvoid: bool = False,
        output_link_pose: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """Check candidate target_poses (N, 4, 4) for the selected arm and end effector.
        type selects Simple/AvoidObs; ObsAvoid enables obstacle avoidance and
        output_link_pose requests link poses. Return success flags and IK info."""
        return self.robot.solve_ik(
            poses=target_poses,
            arm=arm,
            type=type,
            output_link_pose=output_link_pose,
        )

    def reset(self) -> None:
        """Reset the robot to its initial state."""
        self.robot.reset()

    def _mat_to_quat(self, rotation_matrix: np.ndarray) -> List[float]:
        """Convert a 3x3 rotation matrix to a wxyz quaternion list."""
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_matrix(rotation_matrix)
        quat_xyzw = rot.as_quat()  # [x, y, z, w]
        return [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # [w, x, y, z]
