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


class PickUpOnGripper(EvaluateAction):
    """Evaluate whether the gripper successfully picks up an object.

    Supports both single object and multiple objects (comma-separated) for obj_name.
    When multiple objects are provided, the check passes if the gripper successfully
    picks up ANY one of the objects. Already-picked objects are excluded from matching.

    Backward compatible with single object input, also supports placeholder format.

    Args:
        env: The simulation environment.
        obj_name: Object name or comma-separated object names to check.
            Examples: "obj1" or "obj1,obj2,obj3"
            Can also use placeholder: "{@placeholder_str1}"
        gripper_id: Gripper identifier, e.g. "right" or "left". Use "any_gripper"
            (or "any"/"both") to accept a pickup by EITHER hand -- the check then
            tries both grippers each frame and passes on the first that succeeds.
            This keeps one checker (and so one ``progress`` row) per step, unlike
            branching into a left/right pair, which makes ``score_layout_steps``
            read the pair as two separate layout steps.

    Supported parameter formats:
        - Single object: "obj1|right_gripper"
        - Multiple objects: "obj1,obj2,obj3|right_gripper"
        - Placeholder only: "{@placeholder_str1}|right_gripper"
    """

    def __init__(self, env, obj_name, gripper_id):
        super().__init__(env)

        # Parse parameters: support multiple formats
        # Format 1: "obj_name|gripper_id" (backward compatible)
        # Format 2: "obj1,obj2,obj3|gripper_id" (multi-object)
        # Format 3: "{@placeholder_str1}|right_gripper" (placeholder only)

        parts = obj_name.split("|")
        self._holder_name, self._obj_name = self.placeholder_sparser(parts[0])
        if not self._holder_name:
            self._obj_candidates = self.parse_obj_input(self.obj_name)

        self._done_flag = False
        self._holder_id, self._gripper_id = self.placeholder_sparser(gripper_id)
        self._matched_obj = None  # Track which object was successfully picked
        self._matched_gripper = None  # Track which hand picked it (for "any_gripper")
        self._update_count = 0  # Debug counter to track update calls

        # Threshold Parameters for pickup detection
        self.z_threshold = 0.02
        self.distance_threshold = 0.2

        # Status record - per-object tracking for multi-object mode
        self._initial_z_map = {}  # {obj_name: initial_z}
        self._picked_z_map = {}  # {obj_name: current_z at pickup success}
        self.success_detected = False
        self._debug_counter = 0
        self._last_metrics = {}  # {obj_name: {gripper_path, initial_z, z, z_diff, distance}}

        # Initialize a global set on env to track already-picked objects across rounds
        if not hasattr(self.env, "_picked_objects"):
            self.env._picked_objects = set()

        logger.info(
            f"[PickUpOnGripper] Initialized with obj_name='{self.obj_name}', "
            f"obj_name_list={self.obj_name_list}, gripper_id={self.gripper_id}, "
            f"already_picked={self.env._picked_objects}"
        )

        # Initialize a global set on env to track already-picked objects across rounds
        if not hasattr(self.env, "_picked_objects"):
            self.env._picked_objects = set()

        logger.info(
            f"[PickUpOnGripper] Initialized with obj_name='{self.obj_name}', "
            f"obj_name_list={self.obj_name_list}, gripper_id={self.gripper_id}, "
            f"already_picked={self.env._picked_objects}"
        )

    @property
    def obj_name(self) -> str:
        if self._holder_name:
            return getattr(self, self._obj_name)
        return self._obj_name

    @property
    def obj_name_list(self) -> list:
        """Return the resolved list of object names to check, excluding already-picked objects."""
        if self._holder_name:
            # Placeholder mode: resolve the single placeholder value
            resolved = getattr(self, self._obj_name)
            # If placeholder resolved to empty string, fall back to _matched_obj if available
            if not resolved and self._matched_obj:
                return [self._matched_obj]
            return [resolved] if resolved else []
        if self._obj_candidates is None:
            # Single object mode: return as list
            return [self._obj_name] if self._obj_name else []
        # Multi-object mode: filter out already-picked objects, but always keep at least one
        picked = getattr(self.env, "_picked_objects", set())
        filtered = [name for name in self._obj_candidates if name not in picked]
        # If filtering would remove the last candidate, keep the first one to avoid an empty list
        if not filtered and self._obj_candidates:
            filtered = [self._obj_candidates[0]]
        return filtered

    @property
    def gripper_id(self) -> str:
        if self._holder_id:
            return getattr(self, self._gripper_id)
        return self._gripper_id

    @property
    def gripper_ids(self) -> list:
        """Grippers to try this frame.

        A hand-agnostic id ("any_gripper" / "any" / "both") expands to both
        hands so either one succeeding counts; anything else stays a
        single-element list, preserving the side-specific behaviour.
        """
        gid = self.gripper_id or ""
        if gid.split("_")[0].lower() in ("any", "both", "either"):
            return ["left_gripper", "right_gripper"]
        return [gid]

    def _get_gripper_position(self):
        """Get the current gripper world position.

        Returns:
            The gripper position as a numpy array.
        """
        g_pos, _ = self.get_world_pose(self.gripper_link_path(self.gripper_id))
        return g_pos

    def check_pickup_success(self, gripper_pose, obj_pose, obj_name, gripper_id=None):
        """Check if the object is successfully picked up.

        Args:
            gripper_pose: Current gripper pose matrix.
            obj_pose: Current object pose matrix.
            obj_name: Name of the object for tracking.

        Returns:
            True if the object is successfully picked up.
        """
        # Record initial Z-coordinate for this object
        current_z = obj_pose[2, 3]
        if obj_name not in self._initial_z_map:
            self._initial_z_map[obj_name] = current_z
            return False

        initial_z = self._initial_z_map[obj_name]
        gripper_id = gripper_id or self.gripper_id

        # Check 1: Is the object lifted?
        z_diff = current_z - initial_z
        gripper_pos = gripper_pose[:3, 3]
        obj_pos = obj_pose[:3, 3]
        distance = np.linalg.norm(gripper_pos - obj_pos)
        self._last_metrics[(obj_name, gripper_id)] = {
            "gripper_path": self.gripper_link_path(gripper_id),
            "initial_z": initial_z,
            "z": current_z,
            "z_diff": z_diff,
            "distance": distance,
        }

        if z_diff <= self.z_threshold:
            return False

        # Check 2: Distance between object and gripper
        if distance < self.distance_threshold:
            self._picked_z_map[obj_name] = current_z
            return True

        return False

    def _check_single_object_pickup(self, obj_name, gripper_poses):
        """Check if the given object is picked up by any of the given grippers.

        Args:
            obj_name: Name of the object to check.
            gripper_poses: ``{gripper_id: pose matrix}`` for the hands to try.

        Returns:
            True if the object is successfully picked up by one of them.
        """
        current_obj_pose = self.get_obj_pose(obj_name)
        for gid, pose in gripper_poses.items():
            if self.check_pickup_success(pose, current_obj_pose, obj_name, gid):
                self._matched_gripper = gid
                return True
        return False

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0

        self._update_count += 1

        # If success has already been detected, return directly
        if self._done_flag:
            return super().update(delta_time)

        try:
            gripper_poses = {}
            for gid in self.gripper_ids:
                try:
                    gripper_poses[gid] = self.get_world_pose_matrix(self.gripper_link_path(gid))
                except Exception as e:
                    logger.warning(f"[PickUpOnGripper] Could not read gripper '{gid}' pose: {e}")
            if not gripper_poses:
                return super().update(delta_time)

            # Check if gripper picks up any of the target objects (excluding already-picked)
            candidates = self.obj_name_list
            if not candidates:
                # In placeholder mode, fall back to the previously matched object so that
                # subsequent pickup checks (e.g. lifting the already-grasped object) still work.
                if self._matched_obj:
                    candidates = [self._matched_obj]
                    logger.info(
                        f"[PickUpOnGripper] Placeholder resolved to already-picked object "
                        f"'{self._matched_obj}', re-using it for pickup check"
                    )
                else:
                    logger.warning("[PickUpOnGripper] No remaining candidates (all objects already picked)")
                    return super().update(delta_time)

            # Find the first matching object that satisfies pickup condition
            matched = self.find_matching_object(
                candidates, lambda obj: self._check_single_object_pickup(obj, gripper_poses)
            )
            if matched is not None:
                self._done_flag = True
                self._matched_obj = matched
                self.success_detected = True
                self.env._picked_objects.add(matched)
                self.env.update_place_holder("placeholder_str1", matched)
                logger.info(
                    f"[PickUpOnGripper] Successfully picked object '{matched}' with "
                    f"'{self._matched_gripper}' (picked so far: {self.env._picked_objects})"
                )

        except Exception as e:
            logger.warning(f"[PickUpOnGripper] Error during update #{self._update_count}: {e}")

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag and self.success_detected:
            self.progress_info["STATUS"] = "SUCCESS"

    def _log_failure_diagnostics(self):
        """One-shot summary of the last measured metrics per object, printed
        only when the checker is cancelled without success."""
        if self.success_detected:
            return
        if not self._last_metrics:
            logger.warning(
                f"[PickUpOnGripper] Cancelled without any measurement "
                f"(obj: {self.obj_name}, gripper: {self.gripper_id})"
            )
            return
        for (obj_name, gid), m in self._last_metrics.items():
            logger.warning(
                f"[PickUpOnGripper] Cancelled without success: obj='{obj_name}' "
                f"gripper='{gid}' ref='{m['gripper_path']}' "
                f"initial_z={m['initial_z']:.4f} z={m['z']:.4f} "
                f"z_diff={m['z_diff']:.4f} (need > {self.z_threshold}) "
                f"distance={m['distance']:.4f} (need < {self.distance_threshold})"
            )

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(
            f"Action [PickUpOnGripper] {self.gripper_id}, obj: {self.obj_name}, "
            f"matched: {self._matched_obj} by {self._matched_gripper}, evt: {event.value}"
        )

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            self._log_failure_diagnostics()
        elif event == ActionEvent.FINISHED:
            # Simplified scoring: 1 point for success, 0 for failure
            if self.success_detected:
                self.progress_info["SCORE"] = 1
            else:
                self._log_failure_diagnostics()
                self.progress_info["SCORE"] = 0
