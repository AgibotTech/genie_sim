# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import time
import pickle
import numpy as np
import os
from .base_env import BaseEnv
from geniesim.robot import Robot

from geniesim.plugins.tgs import (
    load_task_solution,
    generate_action_stages,
    split_grasp_stages,
)
from geniesim.benchmark.tasks.demo_task import DemoTask
from geniesim.benchmark.tasks.dummy_task import DummyTask
from geniesim.utils.transform_utils import calculate_rotation_matrix


from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance


class DemoEnv(BaseEnv):
    def __init__(self, robot: Robot, task_file: str, init_task_config, policy=None):
        super().__init__(robot)
        self.policy = policy
        self.attached_obj_id = None
        self.current_episode = 0
        self.current_step = 0
        self.init_task_config = init_task_config
        self.camera_list = self.init_task_config["recording_setting"]["camera_list"]
        self.specific_task_name = self.init_task_config["specific_task_name"]

        self.states_objects_by_name = {}
        # init task_file
        self.load(task_file)
        self.load_task_setup()

    def load_task_setup(self):
        if "task_name" not in self.task_info:
            self.task = DummyTask(self)
        else:
            try:
                self.task = DemoTask(self)
            except ImportError:
                raise Exception("ader is not available.")

    def load(self, task_file):
        self.generate_layout(task_file)
        self.robot.open_gripper(id="right")
        self.robot.open_gripper(id="left")
        task_info = json.load(open(task_file, "rb"))
        self.task_info = task_info
        self.policy_stages, self.policy_objects = load_task_solution(task_info)
        self.policy_objects = self.update_objects(self.policy_objects)

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self.current_episode += 1
        self.current_step = 0

    def get_capture_frame(self, data_keys):
        observation = {}
        if "camera" in data_keys:
            render_depth = data_keys["camera"]["render_depth"] if "render_depth" in data_keys["camera"] else False
            render_semantic = (
                data_keys["camera"]["render_semantic"] if "render_semantic" in data_keys["camera"] else False
            )

            cam_data = {}
            for cam_prim in data_keys["camera"]["camera_prim_list"]:
                cam_data[cam_prim] = {}
                response = self.robot.client.capture_frame(camera_prim_path=cam_prim)
                # cam_info
                cam_info = {
                    "W": response.color_info.width,
                    "H": response.color_info.height,
                    "K": np.array(
                        [
                            [response.color_info.fx, 0, response.color_info.ppx],
                            [0, response.color_info.fy, response.color_info.ppy],
                            [0, 0, 1],
                        ]
                    ),
                    "scale": 1,
                }
                # rgb
                rgb = np.frombuffer(response.color_image.data, dtype=np.uint8).reshape(cam_info["H"], cam_info["W"], 4)[
                    :, :, :3
                ]
                cam_data[cam_prim]["image"] = rgb

            observation["camera"] = cam_data

        return observation

    def get_observation(self):
        """
        # Example
            data_keys = {
                'camera': {
                    'camera_prim_list': [
                        '/World/G1/head_link/D455_Solid/TestCameraDepth'
                    ],
                    'render_depth': True,
                    'render_semantic': True
                },
                'pose': [
                    '/World/G1/head_link/D455_Solid/TestCameraDepth'
                ],
                'joint_position': True,
                'gripper': True
            }
        """

        data_keys = {
            "camera": {
                "camera_prim_list": self.camera_list[:1],
                "render_depth": False,
                "render_semantic": False,
            },
            "joint_position": True,
            "gripper": True,
        }

        observation_raw = self.robot.client.get_observation(data_keys)
        return observation_raw

    def reset(self):
        self.task.reset(self)
        self.reset_variables()
        observaion = self.get_observation()
        return observaion

    def step(self, actions):
        self.current_step += 1
        split_stages = split_grasp_stages(self.policy_stages)
        stage_id = -1
        substages = None
        for _stages in split_stages:
            extra_params = _stages[0].get("extra_params", {})
            active_id, passive_id = (
                _stages[0]["active"]["object_id"],
                _stages[0]["passive"]["object_id"],
            )
            arm = extra_params.get("arm", "right")
            action_stages = generate_action_stages(self.policy_objects, _stages, self.robot)
            if not len(action_stages):
                success = False
                logger.warning("No action stage generated.")
                break

            # Execution
            success = True

            for action, substages in action_stages:
                stage_id += 1
                logger.info(">>>>  Stage [%d]  <<<<" % (stage_id + 1))
                if action in ["reset"]:
                    init_pose = self.robot.reset_pose[arm]
                    curr_pose = self.robot.get_ee_pose(ee_type="gripper", id=arm)
                    interp_pose = init_pose.copy()
                    interp_pose[:3, 3] = curr_pose[:3, 3] + (init_pose[:3, 3] - curr_pose[:3, 3]) * 0.25
                    success = self.robot.move_pose(self.robot.reset_pose[arm], type="AvoidObs", arm=arm, block=True)
                    continue
                if action in ["grasp", "pick"]:
                    obj_id = substages.passive_obj_id
                    if obj_id.split("/")[0] not in self.articulated_objs:
                        self.robot.target_object = substages.passive_obj_id

                while len(substages):
                    # get next step actionddd
                    self.policy_objects = self.update_objects(self.policy_objects, arm=arm)
                    target_gripper_pose, motion_type, gripper_action, arm = substages.get_action(self.policy_objects)
                    arm = extra_params.get("arm", "right")
                    self.robot.client.set_frame_state(
                        action,
                        substages.step_id,
                        active_id,
                        passive_id,
                        self.attached_obj_id is not None,
                    )

                    # execution action
                    if target_gripper_pose is not None:
                        self.robot.move_pose(target_gripper_pose, motion_type, arm=arm, block=True)

                    self.robot.client.set_frame_state(
                        action,
                        substages.step_id,
                        active_id,
                        passive_id,
                        self.attached_obj_id is not None,
                    )

                    if gripper_action is not None:
                        name = ["idx81_gripper_r_outer_joint1"]
                        pos = [0.0] if gripper_action == "close" else [0.8]
                        self.policy.sim_ros_node.set_joint_state(name, pos)
                        self.robot.client.DetachObj()
                        if gripper_action == "close":
                            time.sleep(1)
                            self.robot.client.AttachObj(prim_paths=["/World/Objects/" + passive_id])

                    time.sleep(1)

                    self.robot.client.set_frame_state(
                        action,
                        substages.step_id,
                        active_id,
                        passive_id,
                        self.attached_obj_id is not None,
                    )

                    # check sub-stage completion
                    self.policy_objects["gripper"].obj_pose = self.robot.get_ee_pose(ee_type="gripper", id=arm)
                    self.policy_objects = self.update_objects(self.policy_objects, arm=arm)

                    success = substages.check_completion(self.policy_objects)
                    self.robot.client.set_frame_state(
                        action,
                        substages.step_id,
                        active_id,
                        passive_id,
                        self.attached_obj_id is not None,
                    )
                    if success == False:
                        logger.error("Failed at sub-stage %d" % substages.step_id)
                        break

                    # attach grasped object to gripper, avoid articulated objects
                    if arm == "right":
                        # make sure objects are graspped
                        if gripper_action == "close":
                            self.attached_obj_id = substages.passive_obj_id
                        elif gripper_action == "open":
                            self.attached_obj_id = None
                    self.robot.client.set_frame_state(
                        action,
                        substages.step_id,
                        active_id,
                        passive_id,
                        self.attached_obj_id is not None,
                    )

                if success == False:
                    break
            if success == False:
                break

        observaion = self.get_observation()
        self.task.step(self)
        self.action_update()
        need_update = True

        return observaion, self.has_done, need_update, self.task.task_progress

    def start_recording(self, task_name, camera_prim_list, fps):
        self.robot.client.start_recording(
            task_name=task_name,
            fps=fps,
            data_keys={
                "camera": {
                    "camera_prim_list": camera_prim_list,
                    "render_depth": False,
                    "render_semantic": False,
                },
                "pose": [],
                "joint_position": True,
                "gripper": True,
            },
        )

    def generate_layout(self, task_file):
        self.task_file = task_file
        with open(task_file, "r") as f:
            task_info = json.load(f)

        # add mass for stable manipulation
        for stage in task_info["stages"]:
            if stage["action"] in ["place", "insert", "pour"]:
                obj_id = stage["passive"]["object_id"]
                for i in range(len(task_info["objects"])):
                    if task_info["objects"][i]["object_id"] == obj_id:
                        task_info["objects"][i]["mass"] = 10
                        break

        self.articulated_objs = []
        for object_info in task_info["objects"]:
            if object_info["object_id"] == "fix_pose":
                continue
            is_articulated = object_info.get("is_articulated", False)
            if is_articulated:
                self.articulated_objs.append(object_info["object_id"])
            object_info["material"] = "general"
            self.add_object(object_info)
        time.sleep(2)

        self.arm = task_info["arm"]

        """ For G1: Fix camera rotaton to look at target object """
        task_related_objs = []
        for stage in task_info["stages"]:
            for type in ["active", "passive"]:
                obj_id = stage[type]["object_id"]
                if obj_id == "gripper" or obj_id in task_related_objs:
                    continue
                task_related_objs.append(obj_id)

        target_lookat_point = []
        for obj in task_info["objects"]:
            if obj["object_id"] not in task_related_objs:
                continue
            target_lookat_point.append(obj["position"])
        if len(target_lookat_point) != 0:
            target_lookat_point = np.mean(np.stack(target_lookat_point), axis=0)
            self.robot.client.SetTargetPoint(target_lookat_point.tolist())

        material_infos = []
        for key in task_info["object_with_material"]:
            material_infos += task_info["object_with_material"][key]
        if len(material_infos):
            self.robot.client.SetMaterial(material_infos)
            time.sleep(0.3)

        light_infos = []
        for key in task_info["lights"]:
            light_infos += task_info["lights"][key]
        if len(light_infos):
            self.robot.client.SetLight(light_infos)
            time.sleep(0.3)

    def update_objects(self, objects, arm="right"):
        # update gripper pose
        objects["gripper"].obj_pose = self.robot.get_ee_pose(ee_type="gripper", id=arm)

        # update object pose
        for obj_id in objects:
            if obj_id == "gripper":
                continue
            if "fix_pose" == obj_id:
                if len(objects[obj_id].obj_pose) == 3:
                    position = objects[obj_id].obj_pose
                    rotation_matrix = calculate_rotation_matrix(objects[obj_id].direction, [0, 0, 1])
                    objects[obj_id].obj_pose = np.eye(4)
                    objects[obj_id].obj_pose[:3, 3] = position.flatten()
                    objects[obj_id].obj_pose[:3, :3] = rotation_matrix
                continue

            if "/" in obj_id:
                obj_name = obj_id.split("/")[0]
                part_name = obj_id.split("/")[1]

                object_joint_state = self.robot.client.get_object_joint("/World/Objects/%s" % obj_name)
                for joint_name, joint_position, joint_velocity in zip(
                    object_joint_state.joint_names,
                    object_joint_state.joint_positions,
                    object_joint_state.joint_velocities,
                ):
                    if joint_name[-1] == part_name[-1]:
                        objects[obj_id].joint_position = joint_position
                        objects[obj_id].joint_velocity = joint_velocity

            objects[obj_id].obj_pose = self.robot.get_prim_world_pose("/World/Objects/%s" % obj_id)
            if "simple_place" in objects[obj_id].info and objects[obj_id].info["simple_place"]:
                down_direction_world = (np.linalg.inv(objects[obj_id].obj_pose) @ np.array([0, 0, -1, 1]))[:3]
                down_direction_world = down_direction_world / np.linalg.norm(down_direction_world) * 0.08
                objects[obj_id].elements["active"]["place"]["direction"] = down_direction_world

        return objects

    def check_task_file(self, task_file):
        with open(task_file, "r") as f:
            task_info = json.load(f)

        objs_dir = {}
        objs_interaction = {}
        for obj_info in task_info["objects"]:
            obj_id = obj_info["object_id"]
            if obj_id == "fix_pose":
                continue
            objs_dir[obj_id] = obj_info["data_info_dir"]
            if "interaction" in obj_info:
                objs_interaction[obj_id] = obj_info["interaction"]
            else:
                objs_interaction[obj_id] = json.load(open(obj_info["data_info_dir"] + "/interaction.json"))[
                    "interaction"
                ]

        for stage in task_info["stages"]:
            active_obj_id = stage["active"]["object_id"]
            passive_obj_id = stage["passive"]["object_id"]
            if active_obj_id != "gripper":
                if active_obj_id not in objs_dir:
                    logger.warning("Active obj not in objs_dir: %s" % active_obj_id)
                    return False
            if passive_obj_id != "gripper" and passive_obj_id != "fix_pose":
                if passive_obj_id not in objs_dir:
                    logger.warning("Passive obj not in objs_dir: %s" % passive_obj_id)
                    return False
            data_root = os.path.dirname(os.path.dirname(__file__)) + "/assets"
            if stage["action"] in ["grasp", "pick"]:
                passive_obj_id = stage["passive"]["object_id"]
                obj_dir = objs_dir[passive_obj_id]
                primitive = stage["passive"]["primitive"]
                if primitive is None:
                    file = "grasp_pose/grasp_pose.pkl"
                else:
                    file = objs_interaction[passive_obj_id]["passive"]["grasp"][primitive]
                    if isinstance(file, list):
                        file = file[0]
                grasp_file = os.path.join(data_root, obj_dir, file)
                if not os.path.exists(grasp_file):
                    logger.error("-- Grasp file not exist: %s" % grasp_file)
                    return False

                _data = pickle.load(open(grasp_file, "rb"))
                if len(_data["grasp_pose"]) == 0:
                    logger.error("-- Grasp file empty: %s" % grasp_file)
                    return False
        return True
