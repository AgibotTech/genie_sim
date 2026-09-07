# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import time
import json

from geniesim_benchmark.app.controllers.api_core import APICore
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


class DataCourier:
    """In-process orchestration layer between policies and the simulator.

    Used to wrap a ROS bridge (PIROSNode); now talks to api_core directly. The
    ``enable_ros`` flag is retained for backwards compatibility — it controls
    whether external ROS topics are still consumed/published, but the data
    recording path no longer depends on it.
    """

    def __init__(self, api_core: APICore, enable_ros: bool, node_name, hz=30):
        self.api_core = api_core
        self.enable_ros = enable_ros
        self.sim_ros_node = None  # legacy attribute; remains None in pure in-process mode

        if enable_ros:
            try:
                if node_name in ["pi", "corobot", ""]:
                    from geniesim_benchmark.utils.ros_nodes.pi_node import PIROSNode

                    self.sim_ros_node = PIROSNode(robot_name="G1_omnipicker")
                    self.api_core.benchmark_ros_node = self.sim_ros_node
                    if hasattr(self.sim_ros_node, "set_instruction_listener"):
                        self.sim_ros_node.set_instruction_listener(self.api_core.local_recorder.update_instruction)
                    if hasattr(api_core, "sub_task_name"):
                        # Forward initial sub_task_name to the recorder so episode
                        # directories use it from the very first frame.
                        api_core.local_recorder.set_sub_task_name(api_core.sub_task_name or "")
                        if hasattr(self.sim_ros_node, "set_sub_task_name"):
                            self.sim_ros_node.set_sub_task_name(api_core.sub_task_name or "")
            except Exception as exc:
                # Fall through to in-process mode if ROS bridge cannot start.
                logger.warning(f"ROS bridge failed to start ({exc}); falling back to in-process mode")
                self.sim_ros_node = None
        else:
            if hasattr(api_core, "sub_task_name"):
                api_core.local_recorder.set_sub_task_name(api_core.sub_task_name or "")

        self._next = time.time()
        self.hz = hz
        self.robot_cfg = None

    def set_robot_cfg(self, robot_cfg):
        self.robot_cfg = robot_cfg

    def loop_ok(self):
        if self.enable_ros and self.sim_ros_node is not None:
            import rclpy

            return rclpy.ok()
        return True

    def sleep(self):
        if self.enable_ros and self.sim_ros_node is not None and hasattr(self.sim_ros_node, "loop_rate"):
            self.sim_ros_node.loop_rate.sleep()
            return

        now = time.time()
        to_sleep = self._next - now
        if to_sleep > 0:
            time.sleep(to_sleep)
        self._next += 1 / self.hz

    def pub_static_info_msg(self, msg: str):
        # Forward task_instruction to the in-process recorder so head.webm
        # gets the overlay regardless of whether ROS is up.
        try:
            payload = json.loads(msg)
            instruction = payload.get("task_instruction", "")
            if instruction:
                self.api_core.local_recorder.update_instruction(instruction)
            sub_task = payload.get("sub_task_name", "")
            if sub_task:
                self.api_core.local_recorder.set_sub_task_name(sub_task)
        except Exception:
            pass
        if self.enable_ros and self.sim_ros_node is not None:
            self.sim_ros_node.pub_static_info_msg(msg)

    def pub_dynamic_info_msg(self, msg: str):
        if self.enable_ros and self.sim_ros_node is not None:
            self.sim_ros_node.pub_dynamic_info_msg(msg)

    def sim_time(self):
        if self.enable_ros and self.sim_ros_node is not None:
            return self.sim_ros_node.get_clock().now().nanoseconds * 1e-9
        return time.time_ns() * 1e-9

    def get_joint_state_dict(self):
        return self.api_core.get_joint_state_dict()

    _G1_CAM_DIRS = {"head": "head_camera", "left_hand": "left_camera", "right_hand": "right_camera"}
    _G2_CAM_DIRS = {"head": "head_front_camera", "left_hand": "left_camera", "right_hand": "right_camera"}
    _G2_CFGS = {"G2_omnipicker", "G2_90d_gp", "G2_90d", "G2_crsB_omnipicker"}
    _G1_CFGS = {"G1_omnipicker", "G1_120s"}

    def _camera_dirs(self):
        if self.robot_cfg in self._G1_CFGS:
            return self._G1_CAM_DIRS
        if self.robot_cfg in self._G2_CFGS:
            return self._G2_CAM_DIRS
        raise ValueError(f"Invalid robot cfg: {self.robot_cfg}")

    def get_observation_image(self):
        return self.api_core.get_observation_image(self._camera_dirs())

    def get_observation_depth(self):
        return self.api_core.get_observation_depth(self._camera_dirs())

    def get_obs_bundle(self, fetch_images=True, fetch_depth=True, link_names=None):
        # Depth never rides without images (no image fetch this step => no
        # fresh render to read depth back from either).
        if fetch_images:
            dirs = self._camera_dirs()
            image_dirs = dirs
            depth_dirs = dirs if fetch_depth else None
        else:
            image_dirs = None
            depth_dirs = None
        return self.api_core.get_obs_bundle(
            image_dirs=image_dirs,
            depth_dirs=depth_dirs,
            link_names=link_names,
            want_joint_state=True,
        )

    def set_joint_state(self, name, position):
        if self.enable_ros and self.sim_ros_node is not None and hasattr(self.sim_ros_node, "set_joint_state"):
            self.sim_ros_node.set_joint_state(name, position)

    def get_instruction(self):
        if self.enable_ros and self.sim_ros_node is not None and hasattr(self.sim_ros_node, "get_instruction"):
            return self.sim_ros_node.get_instruction()
        return None

    def get_reset(self):
        if self.enable_ros and self.sim_ros_node is not None and hasattr(self.sim_ros_node, "get_reset"):
            return self.sim_ros_node.get_reset()
        return None

    def get_infer_start(self):
        if self.enable_ros and self.sim_ros_node is not None and hasattr(self.sim_ros_node, "get_infer_start"):
            return self.sim_ros_node.get_infer_start()
        return None

    def get_shuffle(self):
        if self.enable_ros and self.sim_ros_node is not None and hasattr(self.sim_ros_node, "get_shuffle"):
            return self.sim_ros_node.get_shuffle()
        return None

    def get_teleop_recording(self):
        if self.enable_ros and self.sim_ros_node is not None and hasattr(self.sim_ros_node, "get_teleop_recording"):
            return self.sim_ros_node.get_teleop_recording()
        return False
