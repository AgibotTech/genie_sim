# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations
from enum import Enum
from typing import Callable, List
import numpy as np
import os
import re
import json
from geniesim.plugins.logger import Logger

from geniesim.app.controllers.api_core import APICore

logger = Logger()


def quat_wxyz_to_rotation_matrix(quat):
    qw, qx, qy, qz = quat
    return np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ]
    )


class ActionEvent(Enum):
    STARTED = 0
    PAUSED = 1
    RESUMED = 2
    FINISHED = 3
    CANCELED = 4


class ActionBase:
    class State(Enum):
        INIT = 0
        RUNNING = 1
        PAUSED = 2
        FINISHED = 3
        CANCELED = 4

    def __init__(self, env):
        self.env = env
        self.pausable: bool = True
        self.state: ActionBase.State = ActionBase.State.INIT
        self.on_event_vec: List[Callable[["ActionBase", ActionEvent], None]] = []

        self.add_on_event(self.handle_action_event)
        self.progress_info = {}

        self._placeholder_str1 = ""
        self._placeholder_str2 = ""
        self._placeholder_str3 = ""

    def start(self) -> bool:
        if self.state == ActionBase.State.INIT:
            self.state = ActionBase.State.RUNNING
            for on_event in self.on_event_vec:
                on_event(self, ActionEvent.STARTED)
            self.update(0.0)
            return True
        return False

    def resume(self) -> bool:
        if self.state == ActionBase.State.PAUSED:
            self.state = ActionBase.State.RUNNING
            for on_event in self.on_event_vec:
                on_event(self, ActionEvent.RESUMED)
            return True
        return False

    def update(self, delta_time: float) -> float:
        self._update_state()
        self.update_progress()
        self.env.task.update_progress(hex(id(self)), self.progress_info)
        return delta_time

    def pause(self) -> bool:
        if self.pausable and self.state == ActionBase.State.RUNNING:
            self.state = ActionBase.State.PAUSED
            for on_event in self.on_event_vec:
                on_event(self, ActionEvent.PAUSED)
            return True
        return False

    def stop(self) -> bool:
        if self.state in (ActionBase.State.RUNNING, ActionBase.State.PAUSED):
            self.state = ActionBase.State.CANCELED
            for on_event in self.on_event_vec:
                on_event(self, ActionEvent.CANCELED)
            return True
        return False

    def update_progress(self):
        pass

    @property
    def is_pausable(self) -> bool:
        return self.pausable

    @is_pausable.setter
    def is_pausable(self, value: bool):
        self.pausable = value

    @property
    def current_state(self) -> State:
        return self.state

    def is_init(self) -> bool:
        return self.state == ActionBase.State.INIT

    def is_running(self) -> bool:
        return self.state == ActionBase.State.RUNNING

    def is_finished(self) -> bool:
        return self.state == ActionBase.State.FINISHED

    def is_paused(self) -> bool:
        return self.state == ActionBase.State.PAUSED

    def is_canceled(self) -> bool:
        return self.state == ActionBase.State.CANCELED

    def is_end_of_life(self) -> bool:
        return self.state.value >= ActionBase.State.FINISHED.value

    def add_on_event(self, on_event: Callable[["ActionBase", ActionEvent], None]):
        self.on_event_vec.append(on_event)

    def _is_done(self) -> bool:
        return True

    def _update_state(self):
        if self._is_done():
            self.state = ActionBase.State.FINISHED
            for on_event in self.on_event_vec:
                on_event(self, ActionEvent.FINISHED)

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        pass

    def placeholder_sparser(self, placeholder_str: str) -> str:
        pattern = r"\{@([^}]+)\}"
        matches = re.findall(pattern, placeholder_str)
        if len(matches) != 0:
            return True, matches[0]
        return False, placeholder_str

    @property
    def placeholder_str1(self) -> str:
        return self._placeholder_str1

    @placeholder_str1.setter
    def placeholder_str1(self, value: str):
        self._placeholder_str1 = value

    @property
    def placeholder_str2(self) -> str:
        return self._placeholder_str2

    @placeholder_str2.setter
    def placeholder_str2(self, value: str):
        self._placeholder_str2 = value

    @property
    def placeholder_str3(self) -> str:
        return self._placeholder_str3

    @placeholder_str3.setter
    def placeholder_str3(self, value: str):
        self._placeholder_str3 = value


class EvaluateAction(ActionBase):
    def __init__(self, env):
        super().__init__(env)
        self.api_core: APICore = env.api_core

    def _analyze_obj_name(self, obj_name):
        if obj_name.startswith("/World"):
            return obj_name

        if not self.env.init_task_config.get("sub_task_name"):
            return "/World/Objects/" + obj_name
        else:
            return "/Workspace/Objects/" + obj_name

    def get_obj_pose(self, obj_name):
        pose = self.api_core.get_obj_world_pose_matrix(self._analyze_obj_name(obj_name))

        return np.array(pose)

    def get_obj_aabb(self, obj_name, obj_size):
        # Position of objects in world coordinate system
        pose = self.api_core.get_obj_world_pose_matrix(self._analyze_obj_name(obj_name))
        x_min, y_min, z_min = -obj_size[0] / 2, -obj_size[1] / 2, -obj_size[2] / 2
        x_max, y_max, z_max = obj_size[0] / 2, obj_size[1] / 2, obj_size[2] / 2
        bbox_3d = np.array(
            [
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max],
            ]
        )
        # The coordinates of the bounding box of an object under the world coordinate system
        p3d_world = np.dot(pose[:3, :3], bbox_3d.T) + pose[:3, 3:]

        x_max = np.max(p3d_world[0, :])
        x_min = np.min(p3d_world[0, :])
        y_max = np.max(p3d_world[1, :])
        y_min = np.min(p3d_world[1, :])
        z_max = np.max(p3d_world[2, :])
        z_min = np.min(p3d_world[2, :])

        room_aabb_low, room_aabb_hi = np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max])
        return room_aabb_low, room_aabb_hi

    def get_object_size(self, data_info_dir):
        assets_dir = os.environ.get("SIM_ASSETS")
        assert assets_dir is not None, "SIM_ASSETS environment variable is not set"
        object_parameters_path = os.path.join(assets_dir, data_info_dir, "object_parameters.json")
        with open(object_parameters_path, "r") as f:
            object_paramsters = json.load(f)
        obj_size = object_paramsters["size"]
        obj_scale = object_paramsters["scale"]
        return [obj_scale * v for v in obj_size]

    def get_obj_aabb_new(self, obj_name):
        aabb = self.api_core.get_obj_aabb(self._analyze_obj_name(obj_name))
        room_aabb_low, room_aabb_hi = np.array([aabb[0], aabb[1], aabb[2]]), np.array([aabb[3], aabb[4], aabb[5]])
        return room_aabb_low, room_aabb_hi

    def get_prismatic_joint(self, obj_name):
        info = self.api_core.get_obj_joint(self._analyze_obj_name(obj_name))
        prismatic_joint = []
        for v in info["joint_positions"]:
            prismatic_joint.append(v)
        return prismatic_joint

    def aabb_contains_point(self, point, container):
        lower, upper = container
        return np.less_equal(lower, point).all() and np.less_equal(point, upper).all()

    def get_world_pose(self, prim_path):
        return self.api_core.get_obj_world_pose(prim_path)

    def get_observation_image(self, disable_robot=True):
        if not disable_robot:
            return self.api_core.get_observation_image(dir={})

        robot_prim_path = None
        if hasattr(self.api_core, "robot_prim_path"):
            robot_prim_path = getattr(self.api_core, "robot_prim_path", None)

        if robot_prim_path:
            try:
                self.api_core.set_prim_visibility(robot_prim_path, False)
                import time

                time.sleep(1)
            except Exception as e:
                logger.warning(f"Failed to set robot visibility to invisible: {e}")

        try:
            image_dict = self.api_core.get_observation_image(dir={})
        finally:
            if robot_prim_path:
                try:
                    self.api_core.set_prim_visibility(robot_prim_path, True)
                except Exception as e:
                    logger.warning(f"Failed to restore robot visibility: {e}")

        return image_dict

    def get_world_pose_matrix(self, prim_path):
        pos, quat = self.api_core.get_obj_world_pose(prim_path)
        R = quat_wxyz_to_rotation_matrix(quat)
        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[:3, 3] = pos

        return matrix


class DebugAction(EvaluateAction):
    def __init__(self, env):
        super().__init__(env)

    def update(self, delta_time: float) -> float:
        self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag


# Execute the actions in turn until they are completed
class ActionList(ActionBase):
    def __init__(self, env):
        super().__init__(env)
        self._actions = []
        self._index = 0

    def resume(self) -> bool:
        if super().resume():
            if self._index < len(self._actions):
                self._actions[self._index].resume()
            return True
        return False

    def update(self, delta_time: float) -> float:
        while self._index < len(self._actions):
            current_action = self._actions[self._index]

            # Initialize the start action
            if current_action.is_init():
                current_action.start()
                # If the list itself is in a pause state, synchronize the pause sub-action
                if current_action.is_running() and self.is_paused():
                    current_action.pause()

            # Execution time update
            if delta_time > 0 and current_action.is_running():
                delta_time = current_action.update(delta_time)

            # Check the current action completion status
            if current_action.is_finished():
                if self.env.has_done:
                    break
                self._index += 1
            else:
                break

        return super().update(delta_time)

    def pause(self) -> bool:
        if super().pause():
            if self._index < len(self._actions):
                self._actions[self._index].pause()
            return True
        return False

    def stop(self) -> bool:
        if super().stop():
            if self._index < len(self._actions):
                self._actions[self._index].stop()
            return True
        return False

    def add_action(self, action: ActionBase):
        self._actions.append(action)

    @property
    def is_empty(self) -> bool:
        return len(self._actions) == 0

    def _is_done(self) -> bool:
        return self._index >= len(self._actions)


# Execute the action of the specified action
class ActionOperatorCb(ActionBase):
    def __init__(self, env, on_operate: Callable[[], None]):
        super().__init__(env)
        self.on_operate = on_operate

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        self.on_operate()
        return super().update(delta_time)


# Action of delay time
class ActionWaitForTime(ActionBase):
    def __init__(self, env, wait_time: float):
        super().__init__(env)
        self.wait_time = wait_time
        self.delayed_time = 0.0
        self._done_flag = False

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0

        self._done_flag = False
        used = delta_time

        if self.delayed_time + used >= self.wait_time:
            used = self.wait_time - self.delayed_time
            self._done_flag = True

        self.delayed_time += used
        remaining = delta_time - used
        return super().update(remaining)

    def _is_done(self) -> bool:
        return self._done_flag


# There is an action that will exit if you satisfy it
class ActionSetWaitAny(ActionBase):
    def __init__(self, env):
        super().__init__(env)
        self._actions = []

    def resume(self) -> bool:
        if super().resume():
            for action in self._actions:
                action.resume()
            return True
        return False

    def update(self, delta_time: float) -> float:
        max_left = 0.0
        for action in self._actions:
            # Automatically start uninitialized action
            if action.is_init():
                action.start()

            # Update only running actions
            if action.is_running():
                remaining = action.update(delta_time)
                if remaining > max_left:
                    max_left = remaining

        # Pass the maximum remaining time to the parent class
        return super().update(max_left)

    def pause(self) -> bool:
        if super().pause():
            for action in self._actions:
                action.pause()
            return True
        return False

    def stop(self) -> bool:
        if super().stop():
            for action in self._actions:
                action.stop()
            return True
        return False

    def add_action(self, action: ActionBase):
        self._actions.append(action)

    def _is_done(self) -> bool:
        return any(action.is_finished() for action in self._actions)

    def _update_state(self):
        super()._update_state()
        # Stop all unfinished sub-actions when finished
        if self.is_finished():
            for action in self._actions:
                if not action.is_finished():
                    action.stop()


# All actions that exit if satisfied
class ActionSetWaitAll(ActionBase):
    def __init__(self, env):
        super().__init__(env)
        self._actions = []

    def resume(self) -> bool:
        if super().resume():
            for action in self._actions:
                action.resume()
            return True
        return False

    def update(self, delta_time: float) -> float:
        max_left = 0.0
        for action in self._actions:
            # Automatically start uninitialized action
            if action.is_init():
                action.start()

            # Update only running actions
            if action.is_running():
                remaining = action.update(delta_time)
                if remaining > max_left:
                    max_left = remaining

        # Pass the maximum remaining time to the parent class
        return super().update(max_left)

    def pause(self) -> bool:
        if super().pause():
            for action in self._actions:
                action.pause()
            return True
        return False

    def stop(self) -> bool:
        if super().stop():
            for action in self._actions:
                action.stop()
            return True
        return False

    def add_action(self, action: ActionBase):
        self._actions.append(action)

    def _is_done(self) -> bool:
        return all(action.is_finished() for action in self._actions)

    def _update_state(self):
        super()._update_state()
        # Stop all unfinished sub-actions when finished
        if self.is_finished():
            for action in self._actions:
                if not action.is_finished():
                    action.stop()


class ActionSetWaitSome(ActionBase):
    def __init__(self, env, min_complete):
        super().__init__(env)
        self._actions = []
        self.min_complete = min_complete

    def resume(self) -> bool:
        if super().resume():
            for action in self._actions:
                action.resume()
            return True
        return False

    def update(self, delta_time: float) -> float:
        max_left = 0.0
        for action in self._actions:
            # Automatically start uninitialized action
            if action.is_init():
                action.start()

            # Update only running actions
            if action.is_running():
                remaining = action.update(delta_time)
                if remaining > max_left:
                    max_left = remaining

        # Pass the maximum remaining time to the parent class
        return super().update(max_left)

    def pause(self) -> bool:
        if super().pause():
            for action in self._actions:
                action.pause()
            return True
        return False

    def stop(self) -> bool:
        if super().stop():
            for action in self._actions:
                action.stop()
            return True
        return False

    def add_action(self, action: ActionBase):
        self._actions.append(action)

    def _is_done(self) -> bool:
        return sum(1 for a in self._actions if a.is_finished()) >= self.min_complete

    def _update_state(self):
        super()._update_state()
        # Stop all unfinished sub-actions when finished
        if self.is_finished():
            for action in self._actions:
                if not action.is_finished():
                    action.stop()

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [ActionSetWaitSome] evt: {event.value}")


# Action to exit if the Checker condition is met
class ActionCheckerCb(ActionBase):
    def __init__(self, env, on_check: Callable[[], bool]):
        super().__init__(env)
        self._on_check = on_check
        self._check_passed = False  # Use different names to avoid conflicts with parent class methods

    def update(self, delta_time: float) -> float:
        if self.is_running():
            self._check_passed = self._on_check()
            # Pass different remaining time according to the inspection results: pass the complete time slice when it passes, clear if it fails
            remaining_time = delta_time if self._check_passed else 0.0
            return super().update(remaining_time)
        return 0.0

    def _is_done(self) -> bool:
        return self._check_passed


# If the action is completed normally, the evaluation will be exited.
class EvalExitAction(ActionBase):
    def __init__(self, env):
        super().__init__(env)

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        if event == ActionEvent.FINISHED:
            self.env.cancel_eval()


class TimeOut(EvalExitAction):
    def __init__(self, env, time_out):
        super().__init__(env)
        self.time_out = time_out
        self.total_time = 0
        self._done_flag = False

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0

        self._done_flag = False
        used = delta_time

        if self.total_time + used >= self.time_out:
            self._done_flag = True
            self.progress_info["SCORE"] = 0

        self.total_time += used
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        progress = self.total_time

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [TimeOut] evt: %d" % (event.value))

        if event == ActionEvent.FINISHED:
            self.env.cancel_eval()


class StepOut(EvalExitAction):
    def __init__(self, env, max_step):
        super().__init__(env)
        self.ref_step = 0
        self.max_step = max_step
        self._done_flag = False

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0

        if self.env.current_step - self.ref_step > self.max_step:
            self._done_flag = True
            self.progress_info["SCORE"] = 0

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [StepOut] evt: %d" % (event.value))

        super().handle_action_event(action, event)

        if event == ActionEvent.STARTED:
            self.ref_step = self.env.current_step
