# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import time
from PIL.Image import logger
import numpy as np
from numpy.random import rand
from scipy.spatial.transform import Rotation as R, Slerp

from geniesim.plugins.ader import AderEnv, AderParams
from geniesim.app.controllers.api_core import APICore
from geniesim.utils.data_courier import DataCourier
from geniesim.utils.infer_pre_process import TaskInfo
from geniesim.benchmark.config.robot_init_states import TASK_INFO_DICT
from geniesim.utils.name_utils import robot_type_mapping


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
            gen_config = self.task_info.get("generalization_config", {})
            rand_init_arm = gen_config.get("rand_init_arm", [0] * 14)
            self.init_arm = list(np.array(self.init_arm) + np.array(rand_init_arm))
        self.task = None

        if need_setup:
            self.load_task_setup()

        self.data_courier: DataCourier = None
        self.need_infer = False

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
