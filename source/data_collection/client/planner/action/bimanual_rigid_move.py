# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.

import numpy as np

from common.base_utils.logger import logger
from client.planner.action.stage import Action, ActionSequence, Stage
from client.planner.func.common import random_downsample


class BimanualRigidMoveStage(Stage):
    """Rigid bimanual motion: derive T_rel = inv(T_leader0) @ T_follower0 from current EEs.
    For each leader goal, use T_follower = T_leader @ T_rel. Validate both arms
    with solve_ik, trying Simple before AvoidObs.
    """

    def select_pose(self, objects, robot):
        leader_arm = self.extra_params.get("leader_arm", "right")
        follower_arm = self.extra_params.get(
            "follower_arm", "left" if leader_arm == "right" else "right"
        )
        from_current_pose = self.extra_params.get("from_current_pose", False)

        if from_current_pose:
            ee_leader = robot.get_ee_pose(ee_type="gripper", id=leader_arm)
            ee_follower = robot.get_ee_pose(ee_type="gripper", id=follower_arm)
            return [
                {
                    "leader_target_pose_world": ee_leader,
                    "follower_target_pose_world": ee_follower,
                    "leader_arm": leader_arm,
                    "follower_arm": follower_arm,
                }
            ]

        tp = self.extra_params.get("target_pose", None)
        if tp is None:
            logger.warning("bimanual_rigid_move: extra_params.target_pose is required")
            return []
        ee_leader = robot.get_ee_pose(ee_type="gripper", id=leader_arm)
        ee_follower = robot.get_ee_pose(ee_type="gripper", id=follower_arm)
        # Match server _bimanual_pose_rel and goal_p.multiply(pose_rel).
        t_rel = np.linalg.inv(ee_leader) @ ee_follower
        poses = np.asarray(tp, dtype=np.float64)
        if len(poses.shape) == 2:
            poses = poses[np.newaxis, ...]
        downsample_num = int(self.extra_params.get("ik_downsample_num", 200))
        poses, _ = random_downsample(
            transforms=poses, downsample_num=downsample_num, replace=False
        )
        if poses.shape[0] == 0:
            logger.warning("bimanual_rigid_move: no poses after downsample")
            return []

        follower_poses = poses @ t_rel

        ik_l, _ = robot.solve_ik(
            poses,
            ee_type="gripper",
            type="Simple",
            arm=leader_arm,
        )
        ik_f, _ = robot.solve_ik(
            follower_poses,
            ee_type="gripper",
            type="Simple",
            arm=follower_arm,
        )
        ok = ik_l & ik_f
        poses = poses[ok]
        follower_poses = follower_poses[ok]
        logger.info(
            f"{self.action_type}: {len(poses)} rigid pair passes isaac-sim IK (Simple) on both arms"
        )
        if len(poses) == 0:
            logger.warning(
                f"{self.action_type}: No rigid pair passes isaac-sim IK (Simple) on both arms"
            )
            return []

        ik_l, _ = robot.solve_ik(
            poses,
            ee_type="gripper",
            type="AvoidObs",
            arm=leader_arm,
        )
        ik_f, _ = robot.solve_ik(
            follower_poses,
            ee_type="gripper",
            type="AvoidObs",
            arm=follower_arm,
        )
        ok = ik_l & ik_f
        poses = poses[ok]
        follower_poses = follower_poses[ok]
        logger.info(
            f"{self.action_type}: {len(poses)} rigid pair passes curobo IK (AvoidObs) on both arms"
        )
        if len(poses) == 0:
            logger.warning(
                f"{self.action_type}: No rigid pair passes curobo IK (AvoidObs) on both arms"
            )
            return []

        result = []
        for i in range(len(poses)):
            result.append(
                {
                    "leader_target_pose_world": poses[i],
                    "follower_target_pose_world": follower_poses[i],
                    "leader_arm": leader_arm,
                    "follower_arm": follower_arm,
                }
            )
        return result

    def generate_action_sequence(
        self,
        leader_target_pose_world,
        follower_target_pose_world,
        leader_arm,
        follower_arm,
    ):
        action_sequence = ActionSequence()
        target_world = leader_target_pose_world
        path_constraint = self.extra_params.get("path_constraint", [1, 1, 1, 0, 0, 0])
        offset_and_constraint_in_goal_frame = self.extra_params.get(
            "offset_and_constraint_in_goal_frame", False
        )
        from_current_pose = self.extra_params.get("from_current_pose", False)
        action_sequence.add_action(
            Action(
                target_world,
                self.extra_params.get("gripper_action", None),
                np.eye(4),
                "BimanualRigidAvoidObs",
                extra_params={
                    "path_constraint": path_constraint,
                    "offset_and_constraint_in_goal_frame": offset_and_constraint_in_goal_frame,
                    "from_current_pose": from_current_pose,
                    "goal_offset": self.extra_params.get(
                        "goal_offset", [0, 0, 0, 1, 0, 0, 0]
                    ),
                    "arm": leader_arm,
                    "leader_arm": leader_arm,
                    "disable_collision_links": self.extra_params.get(
                        "disable_collision_links", []
                    ),
                    "motion_run_ratio": self.extra_params.get("motion_run_ratio", 1.0),
                    "max_consecutive_failures": self.extra_params.get(
                        "max_consecutive_failures", 1
                    ),
                    "use_world_pose": True,
                    "skip_check": True,
                },
            )
        )
        return action_sequence
