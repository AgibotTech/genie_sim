# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from copy import deepcopy
import os
import json
import glob
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R


from geniesim.utils.object import OmniObject
import geniesim.utils.system_utils as system_utils
from geniesim.plugins.logger import Logger

from .utils.object import LayoutObject
from .solver_2d.solver import LayoutSolver2D

logger = Logger()  # Create singleton instance


def get_quaternion_wxyz_from_rotation_matrix(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion in the wxyz format.

    Parameters:
    R (numpy array): A 3x3 rotation matrix.

    Returns:
    numpy array: A 4x1 quaternion in the wxyz format.
    """
    # Convert the rotation matrix to a quaternion
    rot = R.from_matrix(rotation_matrix)
    quat = rot.as_quat()

    # Reorder the quaternion to the wxyz format
    if quat.shape[0] == 4:
        quaternions_wxyz = quat[[3, 0, 1, 2]]
    else:
        quaternions_wxyz = quat[:, [3, 0, 1, 2]]
    return quaternions_wxyz


def list_to_dict(data: list):
    tmp = {}
    for i in range(len(data)):
        tmp[str(i)] = data[i]
    return tmp


class LayoutGenerator:
    def __init__(
        self,
        workspace,
        obj_infos,
        objects,
        key_obj_ids,
        extra_obj_ids,
        constraint=None,
        fix_obj_ids=[],
    ):
        self.workspace = workspace
        self.objects = objects
        self.obj_infos = obj_infos

        self.key_obj_ids = key_obj_ids
        self.extra_obj_ids = extra_obj_ids
        self.fix_obj_ids = fix_obj_ids
        self.constraint = constraint

        if constraint is None:
            self.key_obj_ids_2d = self.key_obj_ids
            self.key_obj_ids_3d = []
        else:
            self.key_obj_ids_2d = [constraint["passive"]]
            self.key_obj_ids_3d = [constraint["active"]]
        self.constraint = constraint

        workspace_xyz, workspace_size = np.array(workspace["position"]), np.array(workspace["size"])
        workspace_size = workspace_size * 1000
        # extra info about workspace

        self.solver_2d = LayoutSolver2D(
            workspace_xyz,
            workspace_size,
            objects,
            fix_obj_ids=fix_obj_ids,
            obj_infos=obj_infos,
        )

        self.succ_obj_ids = []

    def __call__(self):
        """Generate Layout"""
        # import pdb;pdb.set_trace()
        if len(self.key_obj_ids_2d) > 0:
            objs_succ = self.solver_2d(
                self.key_obj_ids_2d,
                self.succ_obj_ids,
                object_extent=30,
                start_with_edge=True,
                key_obj=True,
                initial_angle=0,
            )
            self.update_obj_info(objs_succ)
            logger.info("-- 2d layout done --")

        if len(self.extra_obj_ids) > 0:
            objs_succ = self.solver_2d(
                self.extra_obj_ids,
                self.succ_obj_ids,
                object_extent=30,
                start_with_edge=False,
                key_obj=False,
            )
            self.update_obj_info(objs_succ)
            logger.info("-- extra layout done --")

        """ Check completion """
        res_infos = []
        if len(self.key_obj_ids) > 0:
            for obj_id in self.key_obj_ids:
                if obj_id not in self.succ_obj_ids:
                    return None
                res_infos.append(self.obj_infos[obj_id])
            return res_infos
        elif len(self.extra_obj_ids) > 0:
            if len(self.succ_obj_ids) > 0:
                for obj_id in self.succ_obj_ids:
                    res_infos.append(self.obj_infos[obj_id])
            return res_infos
        else:
            return res_infos

    def update_obj_info(self, obj_ids):
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        for obj_id in obj_ids:
            pose = self.objects[obj_id].obj_pose
            xyz, quat = pose[:3, 3], get_quaternion_wxyz_from_rotation_matrix(pose[:3, :3])
            self.obj_infos[obj_id]["position"] = (xyz / 1000).tolist()
            self.obj_infos[obj_id]["quaternion"] = quat.tolist()
            self.obj_infos[obj_id]["is_key"] = obj_id in self.key_obj_ids
            self.succ_obj_ids.append(obj_id)


class TaskGenerator:
    def __init__(self, task_template):
        self.data_root = str(system_utils.assets_path())
        self.init_info(task_template)

    def _load_json(self, relative_path):
        with open(os.path.join(self.data_root, relative_path), "r") as file:
            return json.load(file)

    def init_info(self, task_template):
        # Load all objects  & constraints
        self.fix_objs = task_template["objects"].get("fix_objects", [])
        all_objs = (
            task_template["objects"]["task_related_objects"] + task_template["objects"]["extra_objects"] + self.fix_objs
        )
        self.fix_obj_ids = [obj["object_id"] for obj in self.fix_objs]

        self.key_obj_ids, self.extra_obj_ids = {"0": []}, {"0": []}
        for obj in task_template["objects"]["task_related_objects"]:
            ws_id = obj.get("workspace_id", "0")
            if ws_id not in self.key_obj_ids:
                self.key_obj_ids[ws_id] = []
            self.key_obj_ids[ws_id].append(obj["object_id"])
        for obj in task_template["objects"]["extra_objects"]:
            ws_id = obj.get("workspace_id", "0")
            if ws_id not in self.extra_obj_ids:
                self.extra_obj_ids[ws_id] = []
            self.extra_obj_ids[ws_id].append(obj["object_id"])

        obj_infos = {}
        objects = {}
        all_key_objs = [obj_id for ws_id in self.key_obj_ids for obj_id in self.key_obj_ids[ws_id]]
        for obj in all_objs:
            obj_id = obj["object_id"]
            if obj_id == "fix_pose":
                info = dict()
                info["object_id"] = obj_id
                info["position"] = obj["position"]
                info["direction"] = obj["direction"]
                obj_infos[obj_id] = info
                objects[obj_id] = OmniObject("fix_pose")
            else:
                obj_dir = os.path.join(self.data_root, obj["data_info_dir"])
                if "metadata" in obj:
                    info = obj["metadata"]["info"]
                    info["interaction"] = obj["metadata"]["interaction"]
                else:
                    info = {"mass": 0.01, "upAxis": "y", "scale": 1}
                info["data_info_dir"] = obj_dir
                info["obj_path"] = obj_dir + "/Aligned.obj"
                info["model_path"] = obj_dir + "/Aligned.usd"
                info["object_id"] = obj_id
                if "extent" in obj:
                    info["extent"] = obj["extent"]
                obj_infos[obj_id] = info
                logger.info(f"obj_id {obj_id} all_key_objs {all_key_objs}")
                objects[obj_id] = LayoutObject(info, use_sdf=obj_id in all_key_objs)

        self.obj_infos, self.objects = obj_infos, objects

        self.fix_obj_infos = []
        for fix_obj in self.fix_objs:
            fix_obj["is_key"] = True
            fix_obj.update(obj_infos[fix_obj["object_id"]])
            self.fix_obj_infos.append(fix_obj)

        if "robot" not in task_template:
            arm = "right"
            robot_id = "G1"
        else:
            arm = task_template["robot"]["arm"]
            robot_id = task_template["robot"]["robot_id"]

        scene_info = task_template["scene"]
        self.scene_usd = task_template["scene"]["scene_usd"]
        self.task_template = {
            "scene_usd": self.scene_usd,
            "arm": arm,
            "task_name": task_template["task"],
            "robot_id": robot_id,
            "stages": task_template["stages"],
            "object_with_material": task_template.get("object_with_material", {}),
            "lights": task_template.get("lights", {}),
            "objects": [],
            "robot_init_pose": None,
        }
        constraint = task_template.get("constraints")
        robot_init_workspace_id = scene_info["scene_id"].split("/")[-1]

        # Retrieve scene information
        self.scene_usd = scene_info["scene_usd"]
        if "function_space_objects" in scene_info:
            workspaces = scene_info["function_space_objects"]
            if robot_init_workspace_id not in task_template["robot"]["robot_init_pose"]:
                self.robot_init_pose = task_template["robot"]["robot_init_pose"]
            else:
                self.robot_init_pose = task_template["robot"]["robot_init_pose"][robot_init_workspace_id]
        else:
            scene_info = self._load_json(scene_info["scene_info_dir"] + "/scene_parameters.json")
            workspaces = scene_info["function_space_objects"]
            # Normalize format
            if isinstance(scene_info["robot_init_pose"], list):
                scene_info["robot_init_pose"] = list_to_dict(scene_info["robot_init_pose"])
            self.robot_init_pose = scene_info["robot_init_pose"][robot_init_workspace_id]
        if isinstance(workspaces, list):
            workspaces = list_to_dict(workspaces)
            workspaces = {"0": workspaces[robot_init_workspace_id]}
        elif isinstance(workspaces, dict) and "position" in workspaces:
            workspaces = {"0": workspaces}
        self.layouts = {}

        for key in workspaces:
            ws, key_ids, extra_ids = (
                workspaces[key],
                self.key_obj_ids.get(key, []),
                self.extra_obj_ids.get(key, []),
            )
            self.layouts[key] = LayoutGenerator(
                ws,
                obj_infos,
                objects,
                key_ids,
                extra_ids,
                constraint=constraint,
                fix_obj_ids=self.fix_obj_ids,
            )

    def shuffle_joint_pd(self):
        """Generate joint PD control parameters generalization config."""
        joint_pd = self.gen_config.get("joint_pd", {})
        if not joint_pd.get("enable", False):
            return {}

        kp_range = joint_pd.get("kp", [])
        kd_range = joint_pd.get("kd", [])
        if not kp_range or not kd_range:
            return {}

        kp = float(np.random.choice(kp_range))
        kd = float(np.random.choice(kd_range))

        return {"enable": True, "kp": kp, "kd": kd}

    def shuffle_camera_noise(self):
        """Generate camera noise generalization config."""
        camera = self.gen_config.get("camera", {})
        noise = camera.get("noise", {})
        if not noise.get("enable", False):
            return {}

        noise_types = noise.get("types", [])
        if not noise_types:
            return {}

        noise_type = str(np.random.choice(noise_types))
        noise_params = {"enable": True, "type": noise_type}

        if noise_type == "gaussian":
            std_range = noise.get("gaussian", {}).get("std_range", [])
            if std_range:
                noise_params["std"] = float(np.random.choice(std_range))
        elif noise_type == "uniform":
            noise_params["low"] = float(noise.get("uniform", {}).get("low", -0.1))
            noise_params["high"] = float(noise.get("uniform", {}).get("high", 0.1))
        elif noise_type == "salt_pepper":
            amount_range = noise.get("salt_pepper", {}).get("amount_range", [])
            noise_params["salt_vs_pepper"] = float(noise.get("salt_pepper", {}).get("salt_vs_pepper", 0.5))
            if amount_range:
                noise_params["amount"] = float(np.random.choice(amount_range))
        elif noise_type == "exponential":
            scale_range = noise.get("exponential", {}).get("scale_range", [])
            if scale_range:
                noise_params["scale"] = float(np.random.choice(scale_range))

        return noise_params

    def shuffle_camera_drop_frame(self):
        """Generate camera drop frame generalization config."""
        camera = self.gen_config.get("camera", {})
        drop_frame = camera.get("drop_frame", {})
        if not drop_frame.get("enable", False):
            return {}

        drop_prob_range = drop_frame.get("drop_prob_range", [])
        if not drop_prob_range:
            return {}

        drop_prob = float(np.random.choice(drop_prob_range))
        return {"enable": True, "drop_prob": drop_prob}

    def shuffle_camera_occlusion(self):
        """Generate camera occlusion generalization config."""
        camera = self.gen_config.get("camera", {})
        occlusion = camera.get("occlusion", {})
        if not occlusion.get("enable", False):
            return {}

        ratio_range = occlusion.get("ratio_range", [])
        if not ratio_range:
            return {}

        ratio = float(np.random.choice(ratio_range))
        return {"enable": True, "ratio": ratio}

    def shuffle_camera_position(self):
        """Generate camera position perturbation generalization config."""
        camera = self.gen_config.get("camera", {})
        position = camera.get("position", {})
        if not position.get("enable", False):
            return {}

        threshold = position.get("threshold", {})
        if not threshold:
            return {}

        perturbations = {"enable": True}
        for axis in ["x", "y", "z", "roll", "pitch", "yaw"]:
            thresh = threshold.get(axis, 0.0)
            perturbations[axis] = float(np.random.uniform(-thresh, thresh))

        return perturbations

    def _get_enabled_generalizations(self):
        """Get list of enabled generalization dimensions and their loop counts.

        Returns:
            List of tuples: (dim_name, num_variants, shuffle_func)
        """
        enabled_dims = []

        # Lights
        lights_config = self.gen_config.get("lights", {})
        num_lights = lights_config.get("num", 0)
        temperature = lights_config.get("temperature", [])
        intensity = lights_config.get("intensity", [])
        has_light_values = (isinstance(temperature, list) and len(temperature) > 0) or (
            isinstance(intensity, list) and len(intensity) > 0
        )
        if lights_config.get("enable", False) and has_light_values and num_lights >= 1:
            enabled_dims.append(("lights", num_lights, self._shuffle_light_config))

        # Init base
        init_base = self.gen_config.get("init_base", {})
        num_init_base = init_base.get("num", 0)
        if init_base.get("enable", False) and num_init_base >= 1:
            enabled_dims.append(("init_base", num_init_base, self._shuffle_init_base))

        # Init joint
        init_joint = self.gen_config.get("init_joint", {})
        num_init_joint = init_joint.get("num", 0)
        if init_joint.get("enable", False) and num_init_joint >= 1:
            enabled_dims.append(("init_joint", num_init_joint, self._shuffle_init_joint))

        # Material
        material = self.gen_config.get("material", {})
        num_material = material.get("num", 0)
        if material.get("enable", False) and num_material >= 1:
            enabled_dims.append(("material", num_material, self._shuffle_material))

        # Joint PD
        joint_pd = self.gen_config.get("joint_pd", {})
        num = joint_pd.get("num", 0)
        if joint_pd.get("enable", False) and num >= 1:
            enabled_dims.append(("joint_pd", num, self.shuffle_joint_pd))

        # Camera noise
        camera = self.gen_config.get("camera", {})
        camera_noise = camera.get("noise", {})
        num = camera_noise.get("num", 0)
        if camera_noise.get("enable", False) and num >= 1:
            enabled_dims.append(("camera_noise", num, self.shuffle_camera_noise))

        # Camera drop frame
        camera_drop_frame = camera.get("drop_frame", {})
        num = camera_drop_frame.get("num", 0)
        if camera_drop_frame.get("enable", False) and num >= 1:
            enabled_dims.append(("camera_drop_frame", num, self.shuffle_camera_drop_frame))

        # Camera occlusion
        camera_occlusion = camera.get("occlusion", {})
        num = camera_occlusion.get("num", 0)
        if camera_occlusion.get("enable", False) and num >= 1:
            enabled_dims.append(("camera_occlusion", num, self.shuffle_camera_occlusion))

        # Camera position
        camera_position = camera.get("position", {})
        num = camera_position.get("num", 0)
        if camera_position.get("enable", False) and num >= 1:
            enabled_dims.append(("camera_position", num, self.shuffle_camera_position))

        return enabled_dims

    def _shuffle_light_config(self):
        """Shuffle light configuration."""
        lights_config = self.gen_config.get("lights", {})
        config = {}
        temperature = lights_config.get("temperature", [])
        if temperature:
            config["temperature"] = int(np.random.choice(temperature))
        intensity = lights_config.get("intensity", [])
        if intensity:
            config["intensity"] = int(np.random.choice(intensity))
        return config

    def _shuffle_init_base(self):
        """Shuffle robot init base pose."""
        init_base = self.gen_config.get("init_base", {})
        x_thresh = init_base.get("x_thresh", 0.1)
        y_thresh = init_base.get("y_thresh", 0.1)

        robot_init_pose = deepcopy(self.robot_init_pose)
        robot_init_pose["position"][0] += np.random.uniform(-x_thresh, x_thresh)
        robot_init_pose["position"][1] += np.random.uniform(-y_thresh, y_thresh)
        return robot_init_pose

    def _shuffle_init_joint(self):
        """Shuffle robot init joint angles."""
        init_joint = self.gen_config.get("init_joint", {})
        joint_thresh = init_joint.get("thresh", 0.1)

        return [np.random.uniform(-joint_thresh, joint_thresh) for _ in range(14)]

    def _shuffle_material(self):
        """Shuffle material configuration (placeholder for material sampling)."""
        return {}

    def generate_tasks(self, save_path, task_name, gen_config):
        """Generate task files with generalization support.

        Supports two modes:
        1. Pre-generated (num >= 1): generates multiple variant files with pre-sampled values
        2. Dynamic (num = 0 or enable=True with num=0): defers to runtime sampling
        """
        self.gen_config = gen_config

        # Clean up existing save path
        if os.path.exists(save_path):
            try:
                shutil.rmtree(save_path)
                logger.info(f"Removed existing directory: {save_path}")
            except Exception as e:
                logger.warning(f"Failed to remove directory {save_path}: {e}")

        os.makedirs(save_path, exist_ok=True)

        # Get enabled generalizations
        enabled_dims = self._get_enabled_generalizations()

        # If no generalizations enabled, generate single baseline task
        if not enabled_dims:
            output_file = os.path.join(save_path, f"{task_name}_0.json")
            self._write_task_file(output_file, {}, robot_init_pose=self.robot_init_pose)
            logger.info("Saved task json to %s (no generalization)" % output_file)
            return

        # Generate task variants
        self._generate_variants(save_path, task_name, enabled_dims)

    def _generate_variants(self, save_path, task_name, enabled_dims):
        """Generate all task variants by iterating over enabled generalization dimensions.

        Args:
            save_path: Directory to save task files.
            task_name: Base name for task files.
            enabled_dims: List of (dim_name, num_variants, shuffle_func) tuples.
        """

        def iterate_variants(dim_idx, gen_config):
            """Recursively iterate over all generalization dimensions."""
            if dim_idx >= len(enabled_dims):
                # Base case: write task file
                output_file = os.path.join(save_path, f"{task_name}_{cnt[0]}.json")
                cnt[0] += 1
                robot_init_pose = gen_config.get("robot_init_pose") or self.robot_init_pose
                self._write_task_file(output_file, gen_config, robot_init_pose=robot_init_pose)
                return

            dim_name, num_variants, shuffle_func = enabled_dims[dim_idx]

            for _ in range(num_variants):
                result = shuffle_func()

                new_config = dict(gen_config)
                if dim_name == "lights":
                    new_config["light_config"] = result
                elif dim_name == "init_base":
                    new_config["robot_init_pose"] = result
                elif dim_name == "init_joint":
                    new_config["rand_init_arm"] = result
                elif dim_name == "material":
                    pass  # Material handled at runtime
                elif dim_name == "joint_pd":
                    new_config["joint_pd"] = result
                elif dim_name == "camera_noise":
                    new_config["camera_noise"] = result
                elif dim_name == "camera_drop_frame":
                    new_config["camera_drop_frame"] = result
                elif dim_name == "camera_occlusion":
                    new_config["camera_occlusion"] = result
                elif dim_name == "camera_position":
                    new_config["camera_position"] = result

                iterate_variants(dim_idx + 1, new_config)

        cnt = [0]
        iterate_variants(0, {})

    def _write_task_file(self, output_file, gen_config, robot_init_pose):
        """Write a task file with the given generalization config.

        Args:
            output_file: Path to write the task file.
            gen_config: Generalization configuration dictionary.
            robot_init_pose: Robot initial pose.
        """
        self.task_template["objects"] = []
        self.task_template["objects"] += self.fix_obj_infos

        flag_failed = False
        for key in self.layouts:
            obj_infos = self.layouts[key]()
            if obj_infos is None:
                flag_failed = True
                break
            if obj_infos:
                self.task_template["objects"] += obj_infos

        if flag_failed:
            logger.error(f"Failed to place key object, skipping {output_file}")
            return

        self.task_template["generalization_config"] = gen_config

        logger.info("Saved task json to %s" % output_file)
        with open(output_file, "w") as f:
            json.dump(self.task_template, f, indent=4)
