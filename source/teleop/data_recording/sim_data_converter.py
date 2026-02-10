# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# from cv2.gapi.streaming import timestamp
import h5py
import numpy as np
import datetime
import os
import yaml
import json
from scipy.spatial.transform import Rotation as R
import shutil
import pytz
import math


def sim2real(sim_vec):
    """sim_vec: Any 6-dimensional vector between 0-1"""
    real_open = np.array([3.7525, 0.12217, 0.12217, 0.12217, 0.12217, 3.1067])
    real_close = np.array([3.2289, 3.4208, 3.4208, 3.4208, 3.4208, 1.0472])
    sim_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 9.3])
    sim_close = np.array([0.21, 0.838, 0.838, 0.838, 0.838, 0.0])

    sim_vec = np.asarray(sim_vec, dtype=float)
    denom = sim_close - sim_open
    scale = np.where(denom == 0, 0.0, (sim_vec - sim_open) / denom)
    return real_open + (real_close - real_open) * scale


def matrix_from_quat_and_pos(quat, pos):
    """
    Convert quaternion and position to 4x4 transformation matrix
    """
    r = R.from_quat(quat)
    t = np.eye(4)
    t[:3, :3] = r.as_matrix()
    t[:3, 3] = pos
    return t


def quat_and_pos_from_matrix(matrix):
    """
    Convert 4x4 transformation matrix to quaternion and position
    """
    r = R.from_matrix(matrix[:3, :3]).as_quat()
    t = matrix[:3, 3]

    return np.concatenate((r, t))


def get_directory_size(directory):
    """Calculate total size of specified directory (bytes)"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Skip symbolic links
            if not os.path.islink(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def reverse_relabel_gripper(g_pos):
    return min(120, max(35, 120 - (120 - 35) * pow((g_pos), 2)))


def omnipicker_reverse_relabel_gripper(g_pos):
    return min(120, max(0.0, 1.28 * 120 * (0.78 - g_pos)))


def omnipicker_sim_to_real(g_pos):
    if g_pos > 0.75:
        return 0.0
    elif g_pos < 0.6:
        return 1.0
    else:
        return min(1.0, max(0.0, (g_pos - 0.6) / 0.15))


def process_camera_parameter_file_name(camera_name):
    if "head_stereo_right" in camera_name.lower():
        camera_name = "head_right_stereo"
    elif "head_stereo_left" in camera_name.lower():
        camera_name = "head_left_stereo"
    elif "head" in camera_name.lower():
        camera_name = "head_front_rgbd"
    elif "hand_left" in camera_name.lower():
        camera_name = "hand_left_rgbd"
    elif "hand_right" in camera_name.lower():
        camera_name = "hand_right_rgbd"
    return camera_name


class SimDataConverter:
    def __init__(
        self,
        record_path,
        output_path,
        job_id,
        task_id,
        episode_id,
        gripper_names=["zhiyuan_gripper_omnipicker", "zhiyuan_gripper_omnipicker"],
        robot_type="G2",
    ):
        self._record_path = record_path
        self._output_path = output_path
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        metadata_file = os.path.join(self._record_path, "metadata.yaml")
        # metadata
        with open(metadata_file, "r") as f:
            self._metadata = yaml.load(f, Loader=yaml.FullLoader)
        self._start_ts = self._metadata["rosbag2_bagfile_information"]["starting_time"]["nanoseconds_since_epoch"]
        # state.json
        state_file = os.path.join(self._record_path, "state.json")
        with open(state_file, "r", encoding="utf-8") as f:
            self._state = json.load(f)
        self._duration = self._state["frames"][-1]["time_stamp"] - self._state["frames"][0]["time_stamp"]
        self._job_id = job_id
        self._task_id = task_id
        self._episode_id = episode_id
        self.gripper_names = gripper_names
        self.robot_type = robot_type

    def convert(self):
        self.convert_parameters()
        self.convert_meta_info()
        self.convert_h5()
        self.genrate_data_info()
        self.mkdir()
        self.print_info()

    def print_info(self):
        display_dir = (self._output_path or "").removeprefix("/geniesim/") or (self._output_path or "")
        print(f"##############[CONVERT-OK] {display_dir}\n")

    def mkdir(self):
        dirs = ["logs", "record", "post_proc"]
        for d in dirs:
            if not os.path.exists(os.path.join(self._output_path, d)):
                os.makedirs(os.path.join(self._output_path, d))
                # dump .keep file
                with open(os.path.join(self._output_path, d, ".keep"), "w"):
                    pass

    def genrate_data_info(self):
        data_info = {
            "episode_id": self._episode_id,
            "job_id": self._job_id,
            "task_id": self._task_id,
            "task_name": "",
            "data_type": "\u5e38\u89c4",
            "init_scene_text": "",
            "raw_data_path": "",
            "aligned_data_path": "",
            "visualizing_data_path": "",
            "sn_code": "G2A0004B900170",
            "label_info": {
                "error_label": "",
                "action_config": [],
                "key_frame": {"single": [], "dual": []},
                "cloud_post_processing_result": {
                    "data_valid": True,
                    "drop_frame_rate": 0.0,
                    "filter_frame_rate": 0.0,
                },
            },
        }
        try:
            with open(os.path.join(self._output_path, "data_info.json"), "w", encoding="utf-8") as f:
                json.dump(data_info, f, indent=4, ensure_ascii=False)
        except Exception:
            with open(os.path.join(self._output_path, "data_info.json"), "w", encoding="utf-8") as f:
                json.dump(data_info, f, indent=4, ensure_ascii=True)

    def move_camera_images(self):
        dest_camera_path = os.path.join(self._output_path, "camera")
        if os.path.exists(dest_camera_path):
            shutil.rmtree(dest_camera_path)
        shutil.copytree(
            os.path.join(self._record_path, "camera"),
            os.path.join(self._output_path, "camera"),
        )
        for subdir, dirs, files in os.walk(dest_camera_path):
            for file in files:
                if file.endswith(".png"):
                    os.rename(
                        os.path.join(subdir, file),
                        os.path.join(subdir, file.replace(".png", "_depth.png")),
                    )
                if file.endswith(".jpg"):
                    os.rename(
                        os.path.join(subdir, file),
                        os.path.join(subdir, file.replace(".jpg", "_color.jpg")),
                    )

    def change_names(self):
        for mp4_name in ["head", "hand_left", "hand_right"]:
            old = os.path.join(self._output_path, mp4_name + ".mp4")
            new = os.path.join(self._output_path, mp4_name + "_color.mp4")
            if os.path.isfile(old):
                os.rename(old, new)
        mapping = {
            "head.jpg": "head_color.jpg",
            "hand_left.jpg": "hand_left_color.jpg",
            "hand_right.jpg": "hand_right_color.jpg",
        }

        for dirpath, _, filenames in os.walk(self._output_path + "/camera"):
            for old_name, new_name in mapping.items():
                old_path = os.path.join(dirpath, old_name)
                new_path = os.path.join(dirpath, new_name)
                if os.path.isfile(old_path):
                    os.rename(old_path, new_path)
            for fname in filenames:
                if "front" in fname.lower():
                    new_name = "head_center_fisheye_color.jpg"
                    os.rename(os.path.join(dirpath, fname), os.path.join(dirpath, new_name))

    def convert_parameters(self):
        paramters_path = os.path.join(self._output_path, "parameters")
        if not os.path.exists(paramters_path):
            os.makedirs(paramters_path)
        camera_parameters_path = os.path.join(paramters_path, "sensor")
        if not os.path.exists(camera_parameters_path):
            os.makedirs(camera_parameters_path)
        # intrinsics
        for cam in self._state["cameras"]:
            camera = self._state["cameras"][cam]
            cam_int_params = {}
            if "fisheye" in cam:
                cam_int_params["fu"] = camera["intrinsic"]["fx"]
                cam_int_params["fv"] = camera["intrinsic"]["fy"]
                cam_int_params["pu"] = camera["intrinsic"]["ppx"]
                cam_int_params["pv"] = camera["intrinsic"]["ppy"]
                cam_int_params["distortion_model"] = "fisheyePolynomial"
                cam = "head_center_fisheye"
            else:
                cam_int_params["Fx"] = float(camera["intrinsic"]["fx"])
                cam_int_params["Fy"] = float(camera["intrinsic"]["fy"])
                cam_int_params["Cx"] = float(camera["intrinsic"]["cx"])
                cam_int_params["Cy"] = float(camera["intrinsic"]["cy"])
                cam_int_params["k1"] = float(camera["intrinsic"]["k1"])
                cam_int_params["k2"] = float(camera["intrinsic"]["k2"])
                cam_int_params["p1"] = float(camera["intrinsic"]["p1"])
                cam_int_params["p2"] = float(camera["intrinsic"]["p2"])
                cam_int_params["k3"] = float(camera["intrinsic"]["k3"])
                if "stereo" in cam:
                    if "left" in cam:
                        cam_int_params["k4"] = float(0.9570823907852173)
                        cam_int_params["k5"] = float(0.18405307829380035)
                        cam_int_params["k6"] = float(0.00723688118159771)
                    else:
                        cam_int_params["k4"] = float(0.7439410090446472)
                        cam_int_params["k5"] = float(-0.0014715819852426648)
                        cam_int_params["k6"] = float(-0.01653667911887169)
                    cam_int_params["SN"] = "H120UA-D60-H09081062"
                elif "left" in cam or "right" in cam:
                    cam_int_params["SN"] = "FD58711400003402"
                else:
                    cam_int_params["SN"] = "CPBC853000CC"
            if "stereo_left" in cam:
                cam = "intrinsic_head_left_stereo"
                with open(
                    os.path.join(camera_parameters_path, cam + ".json"),
                    "w",
                ) as f:
                    json.dump(cam_int_params, f, indent=4)
            elif "stereo_right" in cam:
                cam = "intrinsic_head_right_stereo"
                with open(
                    os.path.join(camera_parameters_path, cam + ".json"),
                    "w",
                ) as f:
                    json.dump(cam_int_params, f, indent=4)
            else:
                if "head" in cam:
                    cam = "head_front_rgb"
                if "color" in cam:
                    cam = cam.replace("color", "rgb")
                with open(
                    os.path.join(camera_parameters_path, "intrinsic_" + cam + ".json"),
                    "w",
                ) as f:
                    json.dump(cam_int_params, f, indent=4)
                cam = cam.replace("rgb", "depth")
                with open(
                    os.path.join(camera_parameters_path, "intrinsic_" + cam + ".json"),
                    "w",
                ) as f:
                    json.dump(cam_int_params, f, indent=4)
        # extrinsics
        for cam in self._state["cameras"]:
            cam_ext_params = []
            for frame in self._state["frames"]:
                cam_pose_world = np.array(frame["cameras"][cam]["pose"])
                world_to_robot = np.linalg.inv(np.array(frame["robot"]["pose"]))
                cam_pose = np.dot(world_to_robot, cam_pose_world)
                cam_ext_params.append(
                    {
                        "rotation": [
                            [cam_pose[0][0], cam_pose[0][1], cam_pose[0][2]],
                            [cam_pose[1][0], cam_pose[1][1], cam_pose[1][2]],
                            [cam_pose[2][0], cam_pose[2][1], cam_pose[2][2]],
                        ],
                        "translation": [
                            cam_pose[0][3],
                            cam_pose[1][3],
                            cam_pose[2][3],
                        ],
                    }
                )
            if "stereo_left" in cam:
                cam = "extrinsic_end_T_head_left_stereo_aligned"
                with open(
                    os.path.join(camera_parameters_path, cam + ".json"),
                    "w",
                ) as f:
                    json.dump(cam_ext_params, f, indent=4)
            elif "stereo_right" in cam:
                cam = "extrinsic_end_T_head_right_stereo_aligned"
                with open(
                    os.path.join(camera_parameters_path, cam + ".json"),
                    "w",
                ) as f:
                    json.dump(cam_ext_params, f, indent=4)
            else:
                if "head" in cam:
                    cam = "head_front_rgbd"
                if "color" in cam:
                    cam = cam.replace("color", "rgbd")
                with open(
                    os.path.join(camera_parameters_path, "extrinsic_end_T_" + cam + "_aligned.json"),
                    "w",
                ) as f:
                    json.dump(cam_ext_params, f, indent=4)
                cam = cam.replace("rgbd", "rgb")
                with open(
                    os.path.join(camera_parameters_path, "extrinsic_" + cam + "_T_depth_aligned.json"),
                    "w",
                ) as f:
                    json.dump(cam_ext_params, f, indent=4)

    def convert_meta_info(self):
        meta_info = {
            "AID": "G2",
            "author": "agibot",
            "clip_end_time": self._start_ts / 1e9 + self._duration,
            "clip_start_time": self._start_ts / 1e9,
            "create_time": datetime.datetime.fromtimestamp(
                self._start_ts / 1e9, tz=pytz.timezone("Asia/Shanghai")
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "data_validate": {"validate": True},
            "duration": self._duration,
            "ee_list": [
                {
                    "name": "left_hand",
                    "type": (self.gripper_names[0] if len(self.gripper_names) > 0 else "zhiyuan_gripper_omnipicker"),
                },
                {
                    "name": "right_hand",
                    "type": (self.gripper_names[1] if len(self.gripper_names) > 1 else "zhiyuan_gripper_omnipicker"),
                },
            ],
            "episode_id": self._episode_id,
            "episode_token": "",
            "file_size": get_directory_size(self._record_path) / 1000.0,
            "fps_validate": True,
            "integrity": {"integrity": True, "reason": 0},
            "job_id": int(self._job_id),
            "mocap_mapping_status": [],
            "robot_type": "G2A",
            "sw_version": "genie-sim",
            "task_id": self._task_id,
            "task_mode": "TDC",
            "text": '{"description":"SIM"]}',
            "version": "v0.0.2",
        }
        with open(os.path.join(self._output_path, "meta_info.json"), "w") as f:
            json.dump(meta_info, f, indent=4)

    def convert_h5(self):
        h5_file_path = os.path.join(self._record_path, "aligned_joints_all.h5")
        converted_h5_file_path = os.path.join(self._output_path, "aligned_joints.h5")
        if os.path.isfile(converted_h5_file_path):
            os.remove(converted_h5_file_path)
        # Open the HDF5 file
        with h5py.File(h5_file_path, "r") as f_in:
            with h5py.File(converted_h5_file_path, "w") as f_out:
                data_length = len(f_in["timestamp"])
                for i in range(data_length):
                    group_name = str(i)
                    group = f_out.create_group(group_name)

                    action_group = group.create_group("action")
                    state_group = group.create_group("state")
                    time_group = group.create_group("timestamp")

                    ### main_timestamp
                    time = f_in["timestamp"][i]
                    dt_us = (time * 1e6).astype(np.uint64)
                    dt_ns = (dt_us * 1e3).astype(np.uint64)
                    timestamp = (dt_ns + self._start_ts).astype(np.uint64)
                    group.create_dataset("main_timestamp", data=timestamp)

                    self.process_action_data(f_in, action_group, i)
                    self.process_state_data(f_in, state_group, i)
                    self.process_timestamp_data(timestamp, time_group)

    def process_action_data(self, f_in, action_group, index):
        prefix = "action"
        joint_position = f_in[f"{prefix}/joint/position"][index]
        # end data
        end_group = action_group.create_group("end")
        end_group.create_dataset("orientation", data=f_in[f"{prefix}/end/orientation"][index])
        end_group.create_dataset("position", data=f_in[f"{prefix}/end/position"][index])
        # head data
        head_index = [
            16,
            17,
            18,
        ]
        head_group = action_group.create_group("head")
        head_group.create_dataset("position", data=joint_position[head_index])
        # joint data
        joint_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        joint_group = action_group.create_group("joint")
        joint_group.create_dataset("position", data=joint_position[joint_index])
        # left_effector
        left_effector_position = omnipicker_sim_to_real(joint_position[14])
        left_effector_group = action_group.create_group("left_effector")
        left_effector_group.create_dataset("position", data=[left_effector_position])
        # right_effector
        right_effector_position = omnipicker_sim_to_real(joint_position[15])
        right_effector_group = action_group.create_group("right_effector")
        right_effector_group.create_dataset("position", data=[right_effector_position])
        # robot
        robot_group = action_group.create_group("robot")
        robot_group.create_dataset("velocity", data=[0.0, 0.0])
        # waist
        waist_index = [19, 20, 21, 22, 23]
        waist_group = action_group.create_group("waist")
        waist_group.create_dataset("position", data=joint_position[waist_index])

    def process_state_data(self, f_in, state_group, index):
        prefix = "state"
        joint_position = f_in[f"{prefix}/joint/position"][index]
        joint_effort = f_in[f"{prefix}/joint/effort"][index]
        joint_velocity = f_in[f"{prefix}/joint/velocity"][index]
        # end data
        end_group = state_group.create_group("end")
        end_group.create_dataset("orientation", data=f_in[f"{prefix}/end/orientation"][index])
        end_group.create_dataset("position", data=f_in[f"{prefix}/end/position"][index])
        end_group.create_dataset("arm_orientation", data=f_in[f"{prefix}/end/arm_orientation"][index])
        end_group.create_dataset("arm_position", data=f_in[f"{prefix}/end/arm_position"][index])
        left_position = np.array(f_in[f"{prefix}/end/arm_position"][index][0])
        right_position = np.array(f_in[f"{prefix}/end/arm_position"][index][1])
        left_orientation = np.array(f_in[f"{prefix}/end/arm_orientation"][index][0])
        right_orientation = np.array(f_in[f"{prefix}/end/arm_orientation"][index][1])
        end_pose = np.concatenate([left_position, left_orientation, right_position, right_orientation], axis=-1)
        end_group.create_dataset("pose", data=end_pose)
        end_group.create_dataset("errmsg", data=np.empty(0))
        end_group.create_dataset("errcode", data=np.zeros(1, dtype=np.int32))
        end_group.create_dataset("mode", data=np.array([5], dtype=np.int32))
        end_group.create_dataset("velocity", data=np.zeros((12, 0), dtype=np.float32))
        end_group.create_dataset("wrench", data=np.zeros((12, 0), dtype=np.float32))
        # head data
        head_index = [
            16,
            17,
            18,
        ]
        head_mode = [0.0] * 3
        head_group = state_group.create_group("head")
        head_group.create_dataset("effort", data=joint_effort[head_index])
        head_group.create_dataset("mode", data=head_mode)
        head_group.create_dataset("position", data=joint_position[head_index])
        head_group.create_dataset("velocity", data=joint_velocity[head_index])
        # joint data
        joint_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        joint_mode = [0.0] * 14
        joint_group = state_group.create_group("joint")
        joint_group.create_dataset("effort", data=joint_effort[joint_index])
        joint_group.create_dataset("mode", data=joint_mode)
        joint_group.create_dataset("position", data=joint_position[joint_index])
        joint_group.create_dataset("velocity", data=joint_velocity[joint_index])
        # left_effector
        left_effector_position = omnipicker_reverse_relabel_gripper(joint_position[14])
        left_effector_group = state_group.create_group("left_effector")
        left_effector_group.create_dataset("position", data=[left_effector_position])
        # right_effector
        right_effector_position = omnipicker_reverse_relabel_gripper(joint_position[15])
        right_effector_group = state_group.create_group("right_effector")
        right_effector_group.create_dataset("position", data=[right_effector_position])
        # robot
        robot_group = state_group.create_group("robot")
        robot_group.create_dataset("orientation", data=f_in[f"{prefix}/robot/orientation"][index])
        robot_group.create_dataset("position", data=f_in[f"{prefix}/robot/position"][index])
        # waist
        waist_index = [19, 20, 21, 22, 23]
        waist_mode = [0.0] * 5
        waist_group = state_group.create_group("waist")
        waist_group.create_dataset("effort", data=joint_effort[waist_index])
        waist_group.create_dataset("mode", data=waist_mode)
        waist_group.create_dataset("position", data=joint_position[waist_index])
        waist_group.create_dataset("velocity", data=joint_velocity[waist_index])

    def process_timestamp_data(self, time, group):
        camera_group = group.create_group("camera")
        camera_group.create_dataset("hand_left_color", data=time)
        camera_group.create_dataset("hand_right_color", data=time)
        camera_group.create_dataset("head_color", data=time)
        camera_group.create_dataset("head_depth", data=time)
        camera_group.create_dataset("head_stereo_left", data=time)
        camera_group.create_dataset("head_stereo_right", data=time)


if __name__ == "__main__":
    # debug
    path = ""
    converter = SimDataConverter(
        record_path=path,
        output_path=path,
        job_id=0,
        task_id=0,
        episode_id=0,
        gripper_names=["zhiyuan_gripper_omnipicker", "zhiyuan_gripper_omnipicker"],
        robot_type="G2",
    )
    converter.convert()
