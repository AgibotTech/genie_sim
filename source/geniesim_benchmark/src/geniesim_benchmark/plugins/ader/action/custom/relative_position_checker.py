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


class RelativePositionChecker(EvaluateAction):
    """Evaluate whether object A is in a specific relative position to object B.

    Supports spatial relationships (all evaluated in the horizontal table plane,
    using the robot's body frame for left/right/top/bottom):
    - leftof: A is to the left of B (robot's +Y axis)
    - rightof: A is to the right of B (robot's -Y axis)
    - topof: A is further from the robot than B (robot's +X / forward axis)
    - bottomof: A is closer to the robot than B (robot's -X / backward axis)
    - aligned_x: aligned along robot's X (forward) axis, |proj| < threshold (default 0.05)
    - aligned_y: aligned along robot's Y (left) axis, |proj| < threshold (default 0.05)
    - aligned_z: |A.z - B.z| < threshold (default 0.05)

    Syntax: "RelativePosition": "obj_A|obj_B|relation"
        or: "RelativePosition": "obj_A|obj_B|relation|threshold"
    where relation is one of: leftof, rightof, topof, bottomof, aligned_x, aligned_y, aligned_z

    Args:
        env: The simulation environment.
        obj_a: Name of object A (the object to check).
        obj_b: Name of object B (the reference object).
        relation: Spatial relationship (leftof, rightof, topof, bottomof, aligned_x, aligned_y, aligned_z).
        threshold: Tolerance for aligned_* relations (default 0.05).
    """

    _DEFAULT_THRESHOLD = 0.05

    def __init__(self, env, obj_a, obj_b, relation, threshold=None):
        super().__init__(env)
        self._holder_a, self._obj_a = self.placeholder_sparser(obj_a)
        self._holder_b, self._obj_b = self.placeholder_sparser(obj_b)
        self._done_flag = False
        self.relation = relation.lower()
        self.env = env

        # Validate relation
        valid_relations = ["leftof", "rightof", "topof", "bottomof", "aligned_x", "aligned_y", "aligned_z"]
        if self.relation not in valid_relations:
            logger.warning(f"[RelativePositionChecker] Invalid relation '{relation}', must be one of {valid_relations}")
            self.relation = "leftof"  # Default fallback

        # Parse threshold: use provided value for aligned_*, ignore for others
        self._threshold = self._DEFAULT_THRESHOLD
        if self.relation.startswith("aligned_"):
            if threshold is not None:
                try:
                    self._threshold = float(threshold)
                except (ValueError, TypeError):
                    logger.warning(
                        f"[RelativePositionChecker] Invalid threshold '{threshold}', using default {self._DEFAULT_THRESHOLD}"
                    )

        # Consecutive frame count for stability (require 2 consecutive frames)
        self._consecutive_count = 0
        self._required_consecutive = 2
        self._update_count = 0

    @property
    def obj_a(self) -> str:
        if self._holder_a:
            return getattr(self, self._obj_a)
        return self._obj_a

    @property
    def obj_b(self) -> str:
        if self._holder_b:
            return getattr(self, self._obj_b)
        return self._obj_b

    def _get_robot_left_right_axis(self):
        """Get the robot's left-right axis in world coordinates.

        Returns:
            Tuple of (left_axis, right_axis) as numpy arrays.
            - left_axis: Robot's +Y direction in world coordinates
            - right_axis: Robot's -Y direction in world coordinates
        """
        rotation = self._get_robot_rotation_matrix()
        left_axis = rotation @ np.array([0, 1, 0])
        right_axis = rotation @ np.array([0, -1, 0])
        return left_axis, right_axis

    def _get_robot_forward_backward_axis(self):
        """Get the robot's forward-backward axis in world coordinates.

        Returns:
            Tuple of (forward_axis, backward_axis) as numpy arrays.
            - forward_axis: Robot's +X direction in world coordinates (away from robot)
            - backward_axis: Robot's -X direction in world coordinates (toward robot)
        """
        rotation = self._get_robot_rotation_matrix()
        forward_axis = rotation @ np.array([1, 0, 0])
        backward_axis = rotation @ np.array([-1, 0, 0])
        return forward_axis, backward_axis

    def _get_robot_rotation_matrix(self):
        """Resolve robot base rotation as a 3x3 numpy matrix; identity on failure."""
        robot_cfg = getattr(self.env, "robot_cfg", None)
        if robot_cfg is None:
            robot_cfg = getattr(self.env, "init_task_config", {}).get("robot_cfg", "G2_omnipicker")

        if "G1" in robot_cfg:
            link_path = "/G1/base_link"
        elif "arx" in robot_cfg or "agilex" in robot_cfg:
            link_path = "/robot/base_link"
        else:
            link_path = "/genie/base_link"

        try:
            _, quat = self.get_world_pose(link_path)
            qw, qx, qy, qz = np.array(quat)
            return np.array(
                [
                    [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                    [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
                    [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
                ]
            )
        except Exception as e:
            logger.warning(f"[RelativePositionChecker] Failed to get robot rotation: {e}")
            return np.eye(3)

    def _check_leftof(self, pos_a, pos_b):
        """Check if A is to the left of B from robot's perspective.

        A is left of B if A's projection onto robot's left axis is greater than B's.
        """
        left_axis, _ = self._get_robot_left_right_axis()

        # Vector from robot to each object
        vec_to_a = pos_a - pos_b
        # Project onto left axis
        proj = np.dot(vec_to_a, left_axis)

        # A is left of B if projection is positive
        is_left = proj > 0.05  # Small threshold to avoid ambiguity

        logger.info(
            f"[RelativePositionChecker] leftof check: "
            f"A={self.obj_a} pos=[{pos_a[0]:.4f},{pos_a[1]:.4f},{pos_a[2]:.4f}], "
            f"B={self.obj_b} pos=[{pos_b[0]:.4f},{pos_b[1]:.4f},{pos_b[2]:.4f}], "
            f"left_axis=[{left_axis[0]:.4f},{left_axis[1]:.4f},{left_axis[2]:.4f}], "
            f"proj={proj:.4f}, is_left={is_left}"
        )
        return is_left

    def _check_rightof(self, pos_a, pos_b):
        """Check if A is to the right of B from robot's perspective.

        A is right of B if A's projection onto robot's right axis is greater than B's.
        """
        _, right_axis = self._get_robot_left_right_axis()

        # Vector from robot to each object
        vec_to_a = pos_a - pos_b
        # Project onto right axis
        proj = np.dot(vec_to_a, right_axis)

        # A is right of B if projection is positive
        is_right = proj > 0.05

        logger.info(
            f"[RelativePositionChecker] rightof check: "
            f"A={self.obj_a} pos=[{pos_a[0]:.4f},{pos_a[1]:.4f},{pos_a[2]:.4f}], "
            f"B={self.obj_b} pos=[{pos_b[0]:.4f},{pos_b[1]:.4f},{pos_b[2]:.4f}], "
            f"right_axis=[{right_axis[0]:.4f},{right_axis[1]:.4f},{right_axis[2]:.4f}], "
            f"proj={proj:.4f}, is_right={is_right}"
        )
        return is_right

    def _check_topof(self, pos_a, pos_b):
        """Check if A is further from the robot than B (along robot's forward axis)."""
        forward_axis, _ = self._get_robot_forward_backward_axis()
        proj = np.dot(pos_a - pos_b, forward_axis)
        is_top = proj > 0.05

        logger.info(
            f"[RelativePositionChecker] topof check: "
            f"A={self.obj_a} pos=[{pos_a[0]:.4f},{pos_a[1]:.4f},{pos_a[2]:.4f}], "
            f"B={self.obj_b} pos=[{pos_b[0]:.4f},{pos_b[1]:.4f},{pos_b[2]:.4f}], "
            f"forward_axis=[{forward_axis[0]:.4f},{forward_axis[1]:.4f},{forward_axis[2]:.4f}], "
            f"proj={proj:.4f}, is_top={is_top}"
        )
        return is_top

    def _check_bottomof(self, pos_a, pos_b):
        """Check if A is closer to the robot than B (along robot's backward axis)."""
        _, backward_axis = self._get_robot_forward_backward_axis()
        proj = np.dot(pos_a - pos_b, backward_axis)
        is_bottom = proj > 0.05

        logger.info(
            f"[RelativePositionChecker] bottomof check: "
            f"A={self.obj_a} pos=[{pos_a[0]:.4f},{pos_a[1]:.4f},{pos_a[2]:.4f}], "
            f"B={self.obj_b} pos=[{pos_b[0]:.4f},{pos_b[1]:.4f},{pos_b[2]:.4f}], "
            f"backward_axis=[{backward_axis[0]:.4f},{backward_axis[1]:.4f},{backward_axis[2]:.4f}], "
            f"proj={proj:.4f}, is_bottom={is_bottom}"
        )
        return is_bottom

    def _check_aligned_x(self, pos_a, pos_b):
        """Check if A and B are aligned along robot's X (forward) axis."""
        forward_axis, _ = self._get_robot_forward_backward_axis()
        diff = abs(np.dot(pos_a - pos_b, forward_axis))
        is_aligned = diff < self._threshold

        logger.info(
            f"[RelativePositionChecker] aligned_x check: "
            f"A={self.obj_a}, B={self.obj_b}, "
            f"forward_axis=[{forward_axis[0]:.4f},{forward_axis[1]:.4f},{forward_axis[2]:.4f}], "
            f"diff={diff:.4f}, threshold={self._threshold}, is_aligned={is_aligned}"
        )
        return is_aligned

    def _check_aligned_y(self, pos_a, pos_b):
        """Check if A and B are aligned along robot's Y (left) axis."""
        left_axis, _ = self._get_robot_left_right_axis()
        diff = abs(np.dot(pos_a - pos_b, left_axis))
        is_aligned = diff < self._threshold

        logger.info(
            f"[RelativePositionChecker] aligned_y check: "
            f"A={self.obj_a}, B={self.obj_b}, "
            f"left_axis=[{left_axis[0]:.4f},{left_axis[1]:.4f},{left_axis[2]:.4f}], "
            f"diff={diff:.4f}, threshold={self._threshold}, is_aligned={is_aligned}"
        )
        return is_aligned

    def _check_aligned_z(self, pos_a, pos_b):
        """Check if A and B are aligned on Z axis (|diff| < threshold)."""
        diff = abs(pos_a[2] - pos_b[2])
        is_aligned = diff < self._threshold

        logger.info(
            f"[RelativePositionChecker] aligned_z check: "
            f"A={self.obj_a} z={pos_a[2]:.4f}, B={self.obj_b} z={pos_b[2]:.4f}, "
            f"diff={diff:.4f}, threshold={self._threshold}, is_aligned={is_aligned}"
        )
        return is_aligned

    def _check_relation(self, pos_a, pos_b):
        """Check the specified spatial relation between A and B."""
        if self.relation == "leftof":
            return self._check_leftof(pos_a, pos_b)
        elif self.relation == "rightof":
            return self._check_rightof(pos_a, pos_b)
        elif self.relation == "topof":
            return self._check_topof(pos_a, pos_b)
        elif self.relation == "bottomof":
            return self._check_bottomof(pos_a, pos_b)
        elif self.relation == "aligned_x":
            return self._check_aligned_x(pos_a, pos_b)
        elif self.relation == "aligned_y":
            return self._check_aligned_y(pos_a, pos_b)
        elif self.relation == "aligned_z":
            return self._check_aligned_z(pos_a, pos_b)
        else:
            logger.warning(f"[RelativePositionChecker] Unknown relation: {self.relation}")
            return False

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0

        self._update_count += 1

        try:
            # Get object positions (center points)
            aa_a, bb_a = self.get_obj_aabb_new(self.obj_a)
            aa_b, bb_b = self.get_obj_aabb_new(self.obj_b)

            # Calculate center positions
            pos_a = (aa_a + bb_a) / 2
            pos_b = (aa_b + bb_b) / 2

            logger.info(
                f"[RelativePositionChecker] Update #{self._update_count}: "
                f"A={self.obj_a} center=[{pos_a[0]:.4f},{pos_a[1]:.4f},{pos_a[2]:.4f}], "
                f"B={self.obj_b} center=[{pos_b[0]:.4f},{pos_b[1]:.4f},{pos_b[2]:.4f}], "
                f"relation={self.relation}"
            )

            # Check the relation
            if self._check_relation(pos_a, pos_b):
                self._consecutive_count += 1
                logger.info(
                    f"[RelativePositionChecker] Relation satisfied (count={self._consecutive_count}/{self._required_consecutive})"
                )

                if self._consecutive_count >= self._required_consecutive:
                    self._done_flag = True
            else:
                self._consecutive_count = 0
                self._done_flag = False

        except Exception as e:
            logger.warning(f"[RelativePositionChecker] Error during update #{self._update_count}: {e}")

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"
            self.progress_info["RELATION"] = self.relation
            self.progress_info["OBJECT_A"] = self.obj_a
            self.progress_info["OBJECT_B"] = self.obj_b

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [RelativePositionChecker] {self.obj_a}->{self.obj_b} ({self.relation}) evt: {event.value}")

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
