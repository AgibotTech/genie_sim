# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import time
from typing import Dict
from PIL.Image import logger
import numpy as np
from numpy.random import rand
from scipy.spatial.transform import Rotation as R, Slerp

from geniesim.plugins.ader import AderEnv, AderParams
from geniesim.app.controllers.api_core import APICore
from geniesim.utils.data_courier import DataCourier
from geniesim.utils.infer_pre_process import TaskInfo
from geniesim.benchmark.config.robot_init_states import TASK_INFO_DICT
from geniesim.utils.name_utils import robot_type_mapping, ROBOT_CONFIGS
from geniesim.utils.ikfk_utils import IKFKSolver
from geniesim.utils.generalization_utils import (
    apply_joint_pd_generalization,
    apply_material_generalization,
    apply_hdr_texture_generalization,
    _apply_camera_generalization_from_env,
)


class BaseEnv(AderEnv):
    def __init__(
        self,
        api_core,
        task_file,
        init_task_config,
        need_setup=True,
        ader_instance=0,
    ) -> None:
        super().__init__(
            api_core,
            AderParams(instance=ader_instance, task_name=init_task_config["task"]),
        )

        self.current_episode = 0
        self.current_step = 0
        self.init_task_config = init_task_config
        self.specific_task_name = self.init_task_config["specific_task_name"]
        self.sub_task_name = self.init_task_config["sub_task_name"]
        self.robot_cfg = robot_type_mapping(self.init_task_config["robot_cfg"])
        self.cfg = ROBOT_CONFIGS.get(self.robot_cfg)
        if self.cfg is None:
            raise ValueError(f"Unsupported robot_cfg: {self.robot_cfg}")

        task_info_cfg = TASK_INFO_DICT.get(self.sub_task_name, {}).get(self.robot_cfg)
        self.load(task_file)
        if task_info_cfg is not None:
            logger.info(f"Config task info for {self.sub_task_name} with {self.robot_cfg}")
            self.robot_task_info = TaskInfo(task_info_cfg, self.robot_cfg)
            (
                self.init_arm,
                self.init_head,
                self.init_waist,
                self.init_hand,
                self.init_gripper,
            ) = self.robot_task_info.init_pose()

        self.ikfk_solver = IKFKSolver(
            self.init_arm,
            self.init_head,
            self.init_waist,
            robot_cfg=self.robot_cfg,
        )
        self.task = None

        # Generalization config storage (populated via set_generalization_* interfaces)
        self.rand_init_arm = None
        self.robot_init_pose = None
        self.light_config = None
        self.joint_pd = None
        self.camera_gen_config = {}

        if need_setup:
            self.load_task_setup()

        self.data_courier: DataCourier = None
        self.need_infer = False

    def set_rand_init_arm(self, rand_init_arm):
        """Set the random initialization arm offsets for joint generalization.

        Args:
            rand_init_arm: List of 14 joint offset values to add to init_arm.
        """
        self.rand_init_arm = rand_init_arm

    def set_robot_init_pose(self, robot_init_pose):
        """Set the robot initial base pose / offsets for init_base generalization.

        Accepts two forms:
        - Pre-generated full pose: {"position": [x, y, z], "quaternion": [w, x, y, z]}
        - Dynamic offsets: {"x_thresh": float, "y_thresh": float}

        Args:
            robot_init_pose: Dict describing the pose or offset thresholds.
        """
        self.robot_init_pose = robot_init_pose

    def set_light_config(self, light_config):
        """Set the light configuration for light generalization.

        Args:
            light_config: Dict with temperature and/or intensity settings.
        """
        self.light_config = light_config

    def set_camera_gen_config(self, camera_gen_config):
        """Set the camera generalization config for camera generalization.

        Args:
            camera_gen_config: Dict containing camera_noise, camera_drop_frame,
                camera_occlusion, and/or camera_position settings.
        """
        self.camera_gen_config = camera_gen_config

    def get_rand_init_arm(self):
        """Get the random initialization arm offsets."""
        return self.rand_init_arm

    def get_robot_init_pose(self):
        """Get the robot initial base pose."""
        return self.robot_init_pose

    def get_light_config(self):
        """Get the light configuration."""
        return self.light_config

    def get_camera_gen_config(self):
        """Get the camera generalization config."""
        return self.camera_gen_config

    def set_joint_pd(self, joint_pd):
        """Set the joint PD parameters for joint_pd generalization.

        Args:
            joint_pd: Dict with enable, kp, kd settings.
        """
        self.joint_pd = joint_pd

    def get_joint_pd(self):
        """Get the joint PD parameters."""
        return self.joint_pd

    def apply_generalization(self, api_core, task_config):
        """Apply all stored generalization configs via api_core.

        This method consumes the configs stored in env members and calls
        api_core to actually apply each generalization effect.

        Args:
            api_core: The API core instance for applying robot/sim updates.
            task_config: Task configuration dictionary containing robot_cfg.
        """
        from scipy.spatial.transform import Rotation as R

        # --- init_base: perturb robot base position ---
        robot_init_pose = self.get_robot_init_pose()
        if robot_init_pose is not None:
            if "position" in robot_init_pose and "quaternion" in robot_init_pose:
                # Pre-generated pose from episode_content
                position = robot_init_pose["position"]
                quaternion = robot_init_pose["quaternion"]
            else:
                # Dynamic: use thresholds stored during update_init_env
                x_thresh = robot_init_pose.get("x_thresh", 0.1)
                y_thresh = robot_init_pose.get("y_thresh", 0.1)
                robot_prim_path = getattr(api_core, "robot_prim_path", "/genie")
                current_pos, current_quat = api_core.get_obj_world_pose(robot_prim_path)
                new_pos = np.array(current_pos)
                new_pos[0] += np.random.uniform(-x_thresh, x_thresh)
                new_pos[1] += np.random.uniform(-y_thresh, y_thresh)
                position = new_pos.tolist()
                quaternion = list(current_quat)
            api_core.update_robot_base(position, quaternion)
            logger.info(f"Applied robot base position: {position}")

        # --- init_joint: perturb init_arm via stored rand_init_arm ---
        rand_init_arm = self.get_rand_init_arm()
        if rand_init_arm is not None:
            if hasattr(self, "init_arm") and self.init_arm is not None:
                self.init_arm = list(np.array(self.init_arm) + np.array(rand_init_arm))
                logger.info(f"Applied init_joint noise: {rand_init_arm}")

        # --- lights: apply light configuration ---
        light_config = self.get_light_config()
        if light_config:
            api_core.apply_light_config(light_config)
            logger.info(f"Applied light config: {light_config}")

        # --- joint_pd: apply joint PD drive gains ---
        joint_pd = self.get_joint_pd()
        if joint_pd:
            apply_joint_pd_generalization(api_core, task_config, {"joint_pd": joint_pd})

        # --- camera: apply camera generalization configs ---
        camera_gen_config = self.get_camera_gen_config()
        if camera_gen_config:
            _apply_camera_generalization_from_env(api_core, task_config, camera_gen_config)

        # --- material: apply material generalization ---
        apply_material_generalization(api_core, task_config)

        # --- hdr_texture: apply HDR texture randomization ---
        apply_hdr_texture_generalization(api_core, task_config)

    def set_infer_status(self, need_infer):
        self.need_infer = need_infer

    def load(self, task_file):
        self.generate_layout(task_file)
        task_info = json.load(open(task_file, "rb"))
        self.task_info = task_info

    def load_task_setup(self):
        """
        To be implemented in child class
        """
        pass

    def reset_variables(self):
        pass

    def get_observation(self):
        return None

    def stop(self):
        pass

    def reset(self):
        super().reset()
        if self.task is not None:
            self.task.reset(self)
        self.reset_variables()
        observaion = self.get_observation()
        return observaion

    def step(self, actions):
        pass

    def set_data_courier(self, data_courier):
        self.data_courier = data_courier

    def set_scene_info(self, scene_info):
        self.scene_info = scene_info

    def set_current_task(self, episode_id):
        self.task.set_task(episode_id)
        self.task.do_action_parsing(self)

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
            is_articulated = object_info.get("is_articulated", False)
            if is_articulated:
                self.articulated_objs.append(object_info["object_id"])
            object_info["material"] = "general"
            self.add_object(object_info)
        time.sleep(0.3)

        self.arm = task_info["arm"]

        """ For G1: Fix camera rotaton to look at target object """
        task_related_objs = []
        for stage in task_info["stages"]:
            for type in ["active", "passive"]:
                obj_id = stage[type]["object_id"]
                if obj_id == "gripper" or obj_id in task_related_objs:
                    continue
                task_related_objs.append(obj_id)

        material_infos = []
        for key in task_info["object_with_material"]:
            material_infos += task_info["object_with_material"][key]
        if len(material_infos):
            self.api_core.SetMaterial(material_infos)
            time.sleep(0.3)

        if "lights" in task_info:
            lights_cfg = task_info["lights"]
            for key in lights_cfg:
                cfg = lights_cfg[key][0]
                self.api_core.set_light(
                    cfg["light_type"],
                    cfg["light_prim"],
                    cfg["light_temperature"],
                    cfg["light_intensity"],
                    cfg["position"],
                    cfg["rotation"],
                    cfg["texture"],
                )
            time.sleep(0.3)

    def add_object(self, object_info: dict):
        name = object_info["object_id"]
        usd_path = object_info.get("model_path")
        if not usd_path:
            return
        position = np.array(object_info["position"])
        quaternion = np.array(object_info["quaternion"])
        if "scale" not in object_info:
            object_info["scale"] = [1, 1, 1]
            size = object_info.get("size", np.array([]))
            if isinstance(size, list):
                size = np.array(size)
            if size.shape[0] == 0:
                size = np.array(100)
            if np.max(size) > 10:
                object_info["scale"] = [0.001, 0.001, 0.001]
        add_particle = False
        particle_position = [0, 0, 0]
        particle_scale = [1, 1, 1]
        if "add_particle" in object_info:
            add_particle = True
            particle_position = object_info["add_particle"]["position"]
            particle_scale = object_info["add_particle"]["scale"]
        if isinstance(object_info["scale"], list):
            scale = np.array(object_info["scale"])
        else:
            scale = np.array([object_info["scale"]] * 3)
        material = "general" if "material" not in object_info else object_info["material"]
        mass = object_info.get("mass", 0.01)
        com = object_info.get("com", [0, 0, 0])
        model_type = object_info.get("model_type", "convexDecomposition")
        static_friction = object_info.get("static_friction", 0.5)
        dynamic_friction = object_info.get("dynamic_friction", 0.5)
        self.api_core.add_usd_obj(
            usd_path=usd_path,
            prim_path="/World/Objects/%s" % name,
            label_name=name,
            position=position,
            rotation=quaternion,
            scale=scale,
            object_color=[1, 1, 1],
            object_material=material,
            object_mass=mass,
            add_particle=add_particle,
            particle_position=particle_position,
            particle_scale=particle_scale,
            particle_color=[1, 1, 1],
            object_com=com,
            model_type=model_type,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        )
