# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import os
import json
import argparse
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from geniesim_benchmark.robot.isaac_sim.client import Rpc_Client
from geniesim_benchmark.robot.utils import (
    get_quaternion_from_euler,
    matrix_to_euler_angles,
    get_rotation_matrix_from_quaternion,
)
from geniesim_benchmark.utils.ros_nodes.base_nodes import SimNode
import geniesim_benchmark.utils.system_utils as system_utils
import rclpy
import threading
from geniesim_benchmark.plugins.logger import Logger
from geniesim_benchmark.teleop.replay_trace import (
    ReplayTraceError,
    frame_delay,
    ordered_joint_positions,
    robot_root_command,
)

logger = Logger()


def dump_robotcfg(robot_cfg):
    try:
        cfg_path = os.path.join(system_utils.app_root_path(), "robot_cfg/", f"{robot_cfg}.json")
        new_cfg_path = os.path.join(
            system_utils.app_root_path(),
            "robot_cfg/",
            f"{robot_cfg}_recording.json",
        )
        with open(cfg_path, "r") as f:
            cfg_content = json.load(f)
        if "G1" in robot_cfg:
            cfg_content["camera"]["/G1/gripper_l_base_link/Left_Camera"] = [320, 240]
            cfg_content["camera"]["/G1/gripper_r_base_link/Right_Camera"] = [320, 240]
            cfg_content["camera"]["/G1/head_link2/Head_Camera"] = [640, 480]
        elif "G2" in robot_cfg:
            cfg_content["camera"]["/G2/gripper_l_base_link/Left_Camera"] = [320, 240]
            cfg_content["camera"]["/G2/gripper_r_base_link/Right_Camera"] = [320, 240]
            cfg_content["camera"]["/G2/head_link3/Head_Camera"] = [640, 480]
        with open(new_cfg_path, "w") as f:
            json.dump(cfg_content, f, indent=4)
        logger.info(f"Successfully dumped robot cfg to {new_cfg_path}")

    except Exception as e:
        logger.error(f"Failed to dump robot cfg to {new_cfg_path} with error {e}")


def get_object_scale(data_info_dir):
    assets_dir = os.environ.get("SIM_ASSETS")
    assert assets_dir is not None, "SIM_ASSETS environment variable is not set"
    object_parameters_path = os.path.join(assets_dir, data_info_dir, "object_parameters.json")
    with open(object_parameters_path, "r") as f:
        object_paramsters = json.load(f)
    obj_scale = object_paramsters["scale"]
    if isinstance(obj_scale, list):
        return obj_scale
    else:
        return [obj_scale] * 3


class Recording:
    def __init__(
        self,
        client_host="localhost:50051",
        state_file="task/task.json",
        task_file="task/task.json",
        fps=60,
        use_recording=False,
    ):
        self.client = Rpc_Client(client_host)
        self.data_root = os.path.dirname(__file__) + "/assets"
        self.fps = 30
        self.use_recording = use_recording

        with open(state_file, "r") as f:
            self.state = json.load(f)

        with open(task_file, "r") as f:
            self.task = json.load(f)

        scene_usd = self.state["scene"]["scene_usd"]
        self.init_frame = self.state["frames"][0] if len(self.state["frames"]) else None
        assert self.init_frame is not None, "No init frame found"
        root_pose = self.init_frame["robot"].get("root_pose")
        if root_pose is not None:
            target_position = root_pose["position"]
            xyzw = root_pose["orientation_xyzw"]
            target_rotation = [xyzw[3], *xyzw[:3]]
        else:
            robot_init_pose = self.task["robot"]["robot_init_pose"]
            target_position = [0, 0, 0]
            target_rotation = [1, 0, 0, 0]
            for _key, val in robot_init_pose.items():
                if isinstance(val, dict):
                    target_position = val["position"]
                    target_rotation = val["quaternion"]
                elif isinstance(val, list):
                    target_position = robot_init_pose["position"]
                    target_rotation = robot_init_pose["quaternion"]
                break
        translation_matrix = np.eye(4)
        translation_matrix[:3, :3] = get_rotation_matrix_from_quaternion(target_rotation)
        translation_matrix[:3, 3] = target_position

        self.init_translation_matrix = translation_matrix
        robot_cfg = self.state["robot"]["name"]
        dump_robotcfg(robot_cfg)
        self.client.InitRobot(
            robot_cfg=f"{robot_cfg}_recording.json",
            robot_usd="",
            scene_usd=scene_usd,
            init_position=target_position,
            init_rotation=target_rotation,
        )
        self.robot_pose = translation_matrix
        self.joint_name = self.init_frame["robot"]["joints"]["joint_name"]
        self.object_list = {}
        self.articulated_object_list = []
        self.run()
        # State reconstruction is a replay view, not a new task-success verdict.
        self.client.Exit()

    def fetch_object_info(self):
        self.task_objects = {}
        object_category = [
            self.task["objects"]["extra_objects"],
            self.task["objects"]["fix_objects"],
            self.task["objects"]["task_related_objects"],
            self.task["objects"].get("articulated_objects", {}),
        ]
        for object_set in object_category:
            for object in object_set:
                data_info_dir = object["data_info_dir"]
                if "metadata" in object:
                    model_path = object["metadata"]["info"]["model_path"]
                    object_scale = object["metadata"]["info"].get("scale", [1, 1, 1])
                    position = object["metadata"]["info"].get("position", None)
                    quaternion = object["metadata"]["info"].get("quaternion", None)

                else:
                    if "model_path" not in object:
                        logger.warning(f"No valid model path for {object['object_id']}")
                        continue
                    model_path = object["model_path"]
                    object_scale = get_object_scale(data_info_dir)
                    position = object.get("position", None)
                    quaternion = object.get("quaternion", None)

                self.task_objects[object["object_id"]] = {
                    "data_info_dir": data_info_dir,
                    "model_path": model_path,
                    "object_scale": object_scale,
                    "position": position,
                    "quaternion": quaternion,
                }

        for obj in self.state["objects"]:
            obj_id = obj["name"]
            if obj_id in self.init_frame["objects"].keys():
                self.object_list[obj_id] = self.init_frame["objects"][obj_id]["pose"]

        for obj in self.state["articulated_objects"]:
            self.articulated_object_list.append(obj["name"])

    def add_task_objects(self):
        for obj_id, obj_pose in self.object_list.items():
            if obj_id in self.task_objects:
                usd_path = self.task_objects[obj_id]["model_path"]
                object_scale = self.task_objects[obj_id]["object_scale"]
                if isinstance(object_scale, list):
                    object_scale = np.array(object_scale)
                else:
                    object_scale = np.array([object_scale] * 3)
                target_matrix = self.init_translation_matrix @ (np.linalg.inv(self.robot_pose) @ obj_pose)
                target_rotation_matrix, target_position = (
                    target_matrix[:3, :3],
                    target_matrix[:3, 3],
                )
                target_rotation = get_quaternion_from_euler(matrix_to_euler_angles(target_rotation_matrix), order="ZYX")
                self.client.add_object(
                    usd_path=usd_path,
                    prim_path="/World/Objects/" + obj_id,
                    label_name=obj_id,
                    target_position=target_position,
                    target_quaternion=target_rotation,
                    target_scale=object_scale,
                    color=np.array([1, 1, 1]),
                    material="general",
                    add_particle=False,
                    mass=0.01,
                )
            else:
                logger.warning("Object {} not found in task_objects".format(obj_id))

    def add_articulated_objects(self):
        for key in self.articulated_object_list:
            if key in self.task_objects:
                usd_path = self.task_objects[key]["model_path"]
                object_scale = self.task_objects[key]["object_scale"]
                target_position = self.task_objects[key]["position"]
                target_rotation = self.task_objects[key]["quaternion"]
                assert target_rotation and target_rotation, "articulated object must have init pose"
                self.client.add_object(
                    usd_path=usd_path,
                    prim_path="/World/Objects/" + key,
                    label_name=key,
                    target_position=target_position,
                    target_quaternion=target_rotation,
                    target_scale=object_scale,
                    color=np.array([1, 1, 1]),
                    material="general",
                    add_particle=False,
                    mass=0.01,
                )

    def set_scenario_light(self):
        light_infos = []
        for key, val in self.task.get("lights", {}).items():
            light_infos += val
        if len(light_infos):
            self.client.SetLight(light_infos)

    def run(self):
        rclpy.init()
        state = self.state["frames"]
        self.camera_list = self.task["recording_setting"]["camera_list"]
        task_name = self.task["task"]
        recorded_joint_names = state[0]["robot"]["joints"]["joint_name"]
        joint_names = [item.name for item in self.client.get_joint_positions().states]
        target_joint_positions = ordered_joint_positions(state[0], joint_names)

        self.sim_ros_node = SimNode()
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.sim_ros_node,))
        self.spin_thread.start()

        self.fetch_object_info()
        self.add_task_objects()
        self.add_articulated_objects()
        self.set_scenario_light()

        self.client.set_joint_positions(target_joint_positions, False)
        time.sleep(1)
        if self.use_recording:
            recording_objects_prim = self.task.get("recording_setting").get("objects_prim")
            self.client.start_recording(
                task_name=task_name,
                fps=self.fps,
                data_keys={
                    "camera": {
                        "camera_prim_list": self.camera_list,
                        "render_depth": False,
                        "render_semantic": False,
                    },
                    "pose": recording_objects_prim,
                    "joint_position": False,
                    "gripper": False,
                },
            )
        idx = 0
        previous_frame = None
        replay = self.state.get("replay", {})
        for gap in replay.get("gaps", []):
            logger.warning(f"Replay source gap: {gap}")
        while rclpy.ok() and idx < len(state):
            frame = state[idx]
            delay = frame_delay(previous_frame, frame)
            if delay:
                time.sleep(delay)

            object_poses = []
            root_command = robot_root_command(frame)
            if root_command is not None:
                object_poses.append(root_command)
            for key, value in frame["objects"].items():
                if value.get("pose_frame") == "world":
                    target_matrix = np.array(value["pose"])
                else:
                    target_matrix = self.init_translation_matrix @ (np.linalg.inv(self.robot_pose) @ value["pose"])
                target_rotation_matrix, target_position = (
                    target_matrix[:3, :3],
                    target_matrix[:3, 3],
                )
                target_rotation = get_quaternion_from_euler(matrix_to_euler_angles(target_rotation_matrix), order="ZYX")
                object_poses.append(
                    {
                        "prim_path": value.get(
                            "prim_path",
                            "/World/Objects/" + key,
                        ),
                        "position": target_position,
                        "rotation": target_rotation,
                    }
                )

            frame_joint_names = frame["robot"]["joints"]["joint_name"]
            if frame_joint_names != recorded_joint_names:
                raise ReplayTraceError("replay robot joint membership changed inside the clip")
            target_joint_positions = ordered_joint_positions(frame, joint_names)

            object_joints = []
            for key, value in frame.get("articulated_object", {}).items():
                object_joints.append(
                    {
                        "prim_path": value.get("prim_path", "/World/Objects/" + key),
                        "joint_cmd": value["joints"]["joint_position"],
                    }
                )
            self.client.SetObjectPose(
                object_poses,
                target_joint_positions,
                object_joints,
            )
            if delay is None:
                self.sim_ros_node.loop_rate.sleep()
            previous_frame = frame
            idx += 1
        self.client.stop_recording()
        self.sim_ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimGraspingAgent Command Line Interface")
    parser.add_argument(
        "--client_host",
        type=str,
        default="localhost:50051",
        help="The client host for SimGraspingAgent (default: localhost:50051)",
    )
    parser.add_argument(
        "--state_file",
        type=str,
        default="task/task.json",
        help="",
    )
    parser.add_argument("--task_file", type=str, default="", help="")
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="fps of the video",
    )
    parser.add_argument("--record", action="store_true")

    args = parser.parse_args()
    recording = Recording(
        args.client_host,
        state_file=args.state_file,
        task_file=args.task_file,
        fps=args.fps,
        use_recording=args.record,
    )
