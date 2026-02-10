# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, sys
import json, h5py
import numpy as np
import subprocess
import cv2

from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores
from rosbags.image import message_to_cvimage
import shutil
import asyncio

# from geniesim.utils.logger import Logger
import concurrent.futures

# logger = Logger()  # Create singleton instance
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)
from teleop.utils.async_json_writer import AsyncJSONWriter
from teleop.utils.transform_utils import (
    world_to_robot_base,
    get_xyz_euler_from_matrix,
    get_pose,
    transform_world_axis_to_robot_axis,
    calculate_y_axis_projection,
    get_quaternion_xyzw_from_rotation_matrix,
    wxyz_to_xyzw,
)

count = 0


def merge_camera(root):
    """
    Move folders 0, 1, 2... from camera_0, camera_1, camera_2 ...
    all to dst_root/camera/0, 1, 2..., with globally continuous indices
    """
    root = Path(root)
    dst = root / "camera"
    dst.mkdir(exist_ok=True)

    global_idx = 0

    # Sort by camera_0, camera_1, ... order
    for camera_dir in sorted(root.glob("camera_*")):
        if not camera_dir.is_dir():
            continue
        for seq_dir in sorted(
            camera_dir.iterdir(),
            key=lambda p: int(p.name) if p.name.isdigit() else float("inf"),
        ):
            if not seq_dir.is_dir():
                continue
            target_dir = dst / str(global_idx)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.move(str(seq_dir), str(target_dir))
            global_idx += 1
        shutil.rmtree(camera_dir)
    print("Camera merged and original files deleted.")


def check_camera(root: str):
    REQUIRED = [
        "hand_left_color.jpg",
        "hand_right_color.jpg",
        "head_color.jpg",
        "hand_left_depth.png",
        "hand_right_depth.png",
        "head_depth.png",
    ]
    root = Path(root) / "camera"
    if not root.exists():
        print("camera directory does not exist")
        return

    folders = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not folders:
        print("No integer subfolders under camera directory")
        return

    file_map = {rf: {} for rf in REQUIRED}
    for folder in folders:
        idx = int(folder.name)
        for rf in REQUIRED:
            fp = folder / rf
            if fp.exists():
                file_map[rf][idx] = fp

    max_idx = int(folders[-1].name)
    for folder in folders:
        cur_idx = int(folder.name)
        for rf in REQUIRED:
            tgt = folder / rf
            if tgt.exists():
                continue
            for j in range(cur_idx + 1, max_idx + 1):
                src = file_map[rf].get(j)
                if src and src.exists():
                    shutil.copy2(src, tgt)
                    print(f"[{cur_idx}] Missing {rf} → copied from {j}")
                    break
            else:
                print(f"[{cur_idx}] Missing {rf} and no subsequent copy, skipping")


def merge_state_json(root):
    root = Path(root)
    files = sorted(root.glob("state_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        print("No state_*.json files found")
        exit(0)
    with open(files[0], "r", encoding="utf-8") as f:
        base = json.load(f)
    merged_frames = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged_frames.extend(data.get("frames", []))
    base["frames"] = merged_frames
    with open(root / "state.json", "w", encoding="utf-8") as f:
        json.dump(base, f, indent=4, ensure_ascii=False)
    for fp in files:
        fp.unlink()
    print(f"Merged {len(files)} files → state.json, and deleted original files.")


def merge_h5(root):
    root = Path(root)
    files = sorted(root.glob("aligned_joints_all_*.h5"), key=lambda p: int(p.stem.split("_")[-1]))

    if not files:
        print("No .h5 files found to merge")
        exit(0)
    dst_path = root / "aligned_joints_all.h5"
    valid_files = []  # Store files containing valid data
    empty_files = []  # Store empty files

    # First check all files, distinguish valid files and empty files
    for fp in files:
        try:
            with h5py.File(fp, "r") as src:
                if "timestamp" in src and src["timestamp"].shape[0] > 0:
                    valid_files.append(fp)
                else:
                    empty_files.append(fp)
        except Exception as e:
            print(f"Error checking file {fp}: {e}")
            empty_files.append(fp)  # Files that cannot be opened are also considered empty

    # If no valid files, create empty target file and delete all source files
    if not valid_files:
        # Create empty target file
        with h5py.File(dst_path, "w") as dst:
            # Create an empty timestamp dataset
            dst.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)

            # Create empty state and action groups
            for grp_name in ["state", "action"]:
                grp = dst.create_group(grp_name)

        # Delete all files (including empty files)
        for f in files:
            try:
                f.unlink()
                print(f"Deleted empty file: {f}")
            except Exception as e:
                print(f"Error deleting file {f}: {e}")

        print(f"All files are empty, created empty target file and deleted all source files → {dst_path}")
        exit(0)

    # Use first valid file to create target file structure
    with h5py.File(dst_path, "w") as dst:
        # Open first valid file to get structure information
        with h5py.File(valid_files[0], "r") as src:
            # ---- timestamp ----
            dtype_ts = src["timestamp"].dtype
            dst.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype=dtype_ts)

            # ---- state / action and subgroups ----
            for grp_name in ["state", "action"]:
                grp = dst.create_group(grp_name)
                src_grp = src[grp_name]
                for sub_key in src_grp.keys():
                    sub_grp = grp.create_group(sub_key)
                    # Copy attributes
                    for k, v in src_grp[sub_key].attrs.items():
                        sub_grp.attrs[k] = v
                    # Create dataset for each inner_key
                    for inner_k, dset in src_grp[sub_key].items():
                        sub_grp.create_dataset(
                            inner_k,
                            shape=(0,) + dset.shape[1:],
                            maxshape=(None,) + dset.shape[1:],
                            dtype=dset.dtype,
                            compression=dset.compression if dset.compression else None,
                        )

        # Concatenate file by file, only process valid files
        for fp in valid_files:
            with h5py.File(fp, "r") as src:
                # Check again if empty
                if src["timestamp"].shape[0] == 0:
                    empty_files.append(fp)  # If found empty, add to empty file list
                    continue

                # 1) timestamp
                ts = src["timestamp"][:]
                ds = dst["timestamp"]
                ds.resize((ds.shape[0] + ts.shape[0],))
                ds[-ts.shape[0] :] = ts

                # 2) All leaf datasets under state / action
                for grp_name in ["state", "action"]:
                    for sub_key in dst[grp_name].keys():
                        for inner_k, target_dset in dst[grp_name][sub_key].items():
                            data = src[grp_name][sub_key][inner_k][:]
                            old_len = target_dset.shape[0]
                            new_len = old_len + data.shape[0]
                            target_dset.resize((new_len,) + data.shape[1:])
                            target_dset[old_len:] = data

    # Delete all files (including valid files and empty files)
    all_files_to_delete = valid_files + empty_files
    for f in all_files_to_delete:
        try:
            f.unlink()
            print(f"Deleted file: {f}")
        except Exception as e:
            print(f"Error deleting file {f}: {e}")

    print(f"Merged {len(valid_files)} valid files → {dst_path}")
    if empty_files:
        print(f"Deleted {len(empty_files)} empty files")


def reorder_joint_state(msg):
    target_joint_name = [
        "idx21_arm_l_joint1",
        "idx22_arm_l_joint2",
        "idx23_arm_l_joint3",
        "idx24_arm_l_joint4",
        "idx25_arm_l_joint5",
        "idx26_arm_l_joint6",
        "idx27_arm_l_joint7",
        "idx61_arm_r_joint1",
        "idx62_arm_r_joint2",
        "idx63_arm_r_joint3",
        "idx64_arm_r_joint4",
        "idx65_arm_r_joint5",
        "idx66_arm_r_joint6",
        "idx67_arm_r_joint7",
        "idx41_gripper_l_outer_joint1",
        "idx81_gripper_r_outer_joint1",
        "idx11_head_joint1",
        "idx12_head_joint2",
        "idx13_head_joint3",
        "idx01_body_joint1",
        "idx02_body_joint2",
        "idx03_body_joint3",
        "idx04_body_joint4",
        "idx05_body_joint5",
        "idx111_chassis_lwheel_front_joint1",
        "idx121_chassis_lwheel_rear_joint1",
        "idx131_chassis_rwheel_front_joint1",
        "idx141_chassis_rwheel_rear_joint1",
        "idx112_chassis_lwheel_front_joint2",
        "idx122_chassis_lwheel_rear_joint2",
        "idx132_chassis_rwheel_front_joint2",
        "idx142_chassis_rwheel_rear_joint2",
        "idx31_gripper_l_inner_joint1",
        "idx71_gripper_r_inner_joint1",
        "idx32_gripper_l_inner_joint3",
        "idx42_gripper_l_outer_joint3",
        "idx72_gripper_r_inner_joint3",
        "idx82_gripper_r_outer_joint3",
        "idx33_gripper_l_inner_joint4",
        "idx43_gripper_l_outer_joint4",
        "idx73_gripper_r_inner_joint4",
        "idx83_gripper_r_outer_joint4",
        "idx39_gripper_l_inner_joint0",
        "idx49_gripper_l_outer_joint0",
        "idx79_gripper_r_inner_joint0",
        "idx89_gripper_r_outer_joint0",
    ]
    now_joint_name = msg.name
    now_joint_position = msg.position.tolist()
    now_joint_velocity = msg.velocity.tolist()
    now_joint_effort = msg.effort.tolist()

    name_to_index = {name: i for i, name in enumerate(now_joint_name)}
    reordered_position = []
    reordered_velocity = []
    reordered_effort = []
    for target_name in target_joint_name:
        if target_name in name_to_index:
            index = name_to_index[target_name]
            reordered_position.append(now_joint_position[index])
            reordered_velocity.append(now_joint_velocity[index])
            reordered_effort.append(now_joint_effort[index])

    return target_joint_name, reordered_position, reordered_velocity, reordered_effort


class Ros_Extrater:
    def __init__(self, bag_file, output_dir, task_info, is_senmatic=False):
        self.bag_file = bag_file
        self.output_dir = output_dir
        self.robot_init_position = np.array(task_info["robot_init_position"])
        self.robot_init_rotation = np.array(task_info["robot_init_rotation"])
        self.camera_info = task_info["camera_info"]
        self.robot_name = task_info["robot_name"]
        self.scene_name = task_info["scene_name"]
        self.scene_usd = task_info["scene_usd"]
        self.scene_glb = task_info.get("scene_glb", "")
        self.code_dict = task_info.get("code_dict", {})
        self.object_names = []
        self.articulated_object_names = []
        for name in task_info["object_names"]["object_prims"]:
            if name not in self.object_names:
                self.object_names.append(name)

        for name in task_info["object_names"]["articulated_object_prims"]:
            self.articulated_object_names.append(name)
        self.frame_status = task_info["frame_status"]
        self.fps = task_info["fps"]
        self.with_img = True
        self.with_video = True
        self.with_senmatic = False
        self.light_config = task_info["light_config"]
        self.gripper_names = task_info["gripper_names"]
        self.playback_timerange = task_info["playback_timerange"]
        self.imag_file_name = []
        if "omnipicker" in self.gripper_names[0]:
            self.left_gripper_center_name = "arm_l_end_link"
        else:
            self.left_gripper_center_name = "gripper_center"
        if "omnipicker" in self.gripper_names[1]:
            self.right_gripper_center_name = "arm_r_end_link"
        else:
            self.right_gripper_center_name = "right_gripper_center"
        self.arm_base_prim_path = task_info["arm_base_prim_path"]

    def post_process_file_name(self, file_name, extra_name="", remove_name=False):
        if "G1" in self.robot_name:
            if "Head" in file_name:
                file_name = "head" + extra_name
            elif "Right" in file_name:
                file_name = "hand_right" + extra_name
            elif "Left" in file_name:
                file_name = "hand_left" + extra_name
            elif "Top" in file_name:
                file_name = "head_front_fisheye" + extra_name
            elif remove_name:
                index = file_name.rfind("_")
                file_name = file_name[:index] + extra_name
        if "G2" in self.robot_name:
            if "head_front" in file_name:
                file_name = "head" + extra_name
            elif "head_right" in file_name:
                file_name = "head_stereo_right" + extra_name
            elif "head_left" in file_name:
                file_name = "head_stereo_left" + extra_name
            elif "left_camera" in file_name:
                file_name = "hand_left" + extra_name
            elif "Left_Camera" in file_name:
                file_name = "hand_left" + extra_name
            elif "right_camera" in file_name:
                file_name = "hand_right" + extra_name
            elif "Right_Camera" in file_name:
                file_name = "hand_right" + extra_name
            elif "head_left" in file_name:
                file_name = "head_stereo_left" + extra_name
            elif "head_right" in file_name:
                file_name = "head_stereo_right" + extra_name
            elif remove_name:
                index = file_name.rfind("_")
                file_name = file_name[:index] + extra_name
        return file_name

    def get_objects_size_map(self, label_dict):
        self.size_map = {}
        for key, value in label_dict.items():
            if key == "time_stamp" or int(key) == 0 or int(key) == 1:
                continue
            semantic = next(iter(value.values()))
            if "background" in semantic or "robot" in semantic:
                continue
            if "barcode" in semantic:
                object_name = semantic.replace("_barcode", "").replace(",", "").strip()
                self.size_map[semantic] = (
                    self.code_dict.get(object_name, {}).get("barcode", {}).get("size", [-1, -1, -1])
                )
            elif "qrcode" in semantic:
                object_name = semantic.replace("_qrcode", "").replace(",", "").strip()
                self.size_map[semantic] = (
                    self.code_dict.get(object_name, {}).get("qrcode", {}).get("size", [-1, -1, -1])
                )
            else:
                object_name = semantic
                self.size_map[semantic] = self.code_dict.get(object_name, {}).get("size", [-1, -1, -1])

    def get_part_transform_matrix_map(self, label_dict):
        self.part_transform_map = {}
        for key, value in label_dict.items():
            if key == "time_stamp" or int(key) == 0 or int(key) == 1:
                continue
            semantic = next(iter(value.values()))
            if "background" in semantic or "robot" in semantic:
                continue
            if "barcode" in semantic:
                object_name = semantic.replace("_barcode", "").replace(",", "").strip()
                self.part_transform_map[semantic] = (
                    self.code_dict.get(object_name, {}).get("barcode", {}).get("transform", np.eye(4))
                )
                self.part_transform_map[semantic] = np.eye(4)
            elif "qrcode" in semantic:
                object_name = semantic.replace("_qrcode", "").replace(",", "").strip()
                self.part_transform_map[semantic] = (
                    self.code_dict.get(object_name, {}).get("qrcode", {}).get("transform", np.eye(4))
                )

    def get_size_by_semantic(self, semantic):
        for key, value in self.size_map.items():
            if key == semantic:
                return value

    def get_part_transform_by_semantic(self, semantic):
        for key, value in self.part_transform_map.items():
            if key == semantic:
                return value

    async def dump_objects_bbox3d(self, label_dict, idx, single_frame_state, chunk_index):
        files = ["head", "hand_left", "hand_right"]
        for camera_name in files:
            result = self.segmenation_image_json[idx][camera_name]
            objects = result["objects"]
            for obj_name, obj_pose in single_frame_state["objects"].items():
                object_pose = obj_pose["pose"]
                object_size = self.get_size_by_semantic(obj_name)
                robot_base_object_pose = transform_world_axis_to_robot_axis(
                    object_pose, self.robot_init_position, self.robot_init_rotation
                )
                object_position = [
                    robot_base_object_pose[0][3],
                    robot_base_object_pose[1][3],
                    robot_base_object_pose[2][3],
                ]
                object_euler = get_xyz_euler_from_matrix(np.array([row[:3] for row in robot_base_object_pose[:3]]))
                if not object_size:
                    continue
                bbox3d = {
                    "center": object_position,
                    "size": [float(x) for x in object_size],
                    "rotation": object_euler.tolist(),
                }
                group_id = get_group_id_by_semantic(label_dict, obj_name)
                for object in objects:
                    if group_id == object["id"]:
                        object["bbox3d"] = bbox3d
                        for key, value in label_dict.items():
                            if key == "time_stamp" or key == "0" or key == "1":
                                continue
                            semantic = next(iter(value.values()))
                            if obj_name in semantic:
                                if "barcode" in semantic:
                                    part_name = "barcode"
                                elif "qrcode" in semantic:
                                    part_name = "qrcode"
                                elif obj_name == semantic:
                                    continue
                                transform_matrix = self.get_part_transform_by_semantic(semantic)
                                part_pose = np.array(object_pose) @ np.array(transform_matrix)
                                part_size = self.get_size_by_semantic(semantic)
                                part_bbox = calculate_y_axis_projection(part_pose, part_size)
                                reshaped_part_bbox = change_bbox_order(part_bbox, single_frame_state, camera_name)
                                bbox = world_to_robot_base(
                                    reshaped_part_bbox,
                                    get_pose(
                                        self.robot_init_position,
                                        self.robot_init_rotation,
                                    ),
                                )
                                if object[part_name] != None:
                                    object[part_name]["corners3d"] = [arr.tolist() for arr in bbox]
            file_path = Path(self.output_dir) / f"camera_{chunk_index}" / str(idx) / (camera_name + ".json")
            await self.json_writer.add_data(result, str(file_path))

    def generate_img_json(self, img_id, segmentation_polys, label_dict, depth_dir, file_name):
        result = {"image_id": "", "objects": [], "segmentation": []}
        result["image_id"] = str(img_id)
        for semantic in segmentation_polys.keys():
            bounding_polys = []
            for poly in segmentation_polys[semantic]["polys"]["poly"]:
                bounding_polys.append(poly)
            class_id = get_class_id_by_semantic(semantic)
            group_id = get_group_id_by_semantic(label_dict, semantic)
            bbox2d = get_polys_bounding(bounding_polys)

            entry = {"cls_id": class_id, "group_id": group_id, "polys": []}
            for index in range(len(segmentation_polys[semantic]["polys"]["poly"])):
                reshaped = segmentation_polys[semantic]["polys"]["poly"][index].astype(int).reshape(-1, 2)
                poly = {
                    "hierarchy": segmentation_polys[semantic]["polys"]["hierarchy"][index],
                    "points": reshaped.astype(int).tolist(),
                }
                entry["polys"].append(poly)
            result["segmentation"].append(entry)
            if not any(key in semantic for key in ["barcode", "qrcode", "robot", "background"]):
                size = self.get_size_by_semantic(semantic)
                # get box type
                H, W, L = sorted(size)
                obj_type = "generic_box"
                if (L / W) >= 3.0 and (H / W) >= 0.4:
                    obj_type = "slender_box"
                elif (L / W) >= 3.0 and (H / W) < 0.4:
                    obj_type = "flat_long_box"
                elif max(L, W, H) / min(L, W, H) <= 1.3:
                    obj_type = "square_box"
                object = {
                    "id": group_id,
                    "type": obj_type,
                    "cls_id": class_id,
                    "bbox2d": bbox2d,
                    "occluded": False,
                    "bbox3d": None,
                    "need_flip": False,
                    "barcode": None,
                    "qrcode": None,
                }
                result["objects"].append(object)
        for semantic in segmentation_polys.keys():
            bounding_polys = []
            for poly in segmentation_polys[semantic]["polys"]["poly"]:
                bounding_polys.append(poly)
            bbox2d = get_polys_bounding(bounding_polys)
            face_orientation = ""
            if "barcode" in semantic:
                # add face_orientation
                object_name = semantic.replace("_barcode", "").replace(",", "").strip()
                face_orientation = self.code_dict.get(object_name, {}).get("barcode", {}).get("direction", "")
                res = {
                    "bbox2d": bbox2d,
                    "occluded": False,
                    "corners3d": [],
                    "face_orientation": face_orientation,
                }
                change_num = -1
                for index, object in enumerate(result["objects"]):
                    semantic_body = get_semantic_by_group_id(label_dict, object["id"])
                    semantic_object_name = semantic_body.replace("_body", "")
                    if semantic_object_name in semantic:
                        change_num = index
                        break
                if change_num != -1:
                    result["objects"][change_num]["barcode"] = res
            elif "qrcode" in semantic:
                object_name = semantic.replace("_qrcode", "").replace(",", "").strip()
                face_orientation = self.code_dict.get(object_name, {}).get("qrcode", {}).get("direction", "")
                res = {
                    "bbox2d": bbox2d,
                    "occluded": False,
                    "corners3d": [],
                    "face_orientation": face_orientation,
                }
                change_num = -1
                for index, object in enumerate(result["objects"]):
                    semantic_body = get_semantic_by_group_id(label_dict, object["id"])
                    if semantic_body in semantic:
                        change_num = index
                        break
                if change_num != -1:
                    result["objects"][change_num]["qrcode"] = res
        file_name = file_name.rsplit("_semantic", 1)[0]
        if img_id not in self.segmenation_image_json:
            self.segmenation_image_json[img_id] = {}
        self.segmenation_image_json[img_id][file_name] = result

    async def extract(self):
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        with AnyReader([Path(self.bag_file)], default_typestore=typestore) as reader:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 4)
            self.json_writer = AsyncJSONWriter(max_queue_size=100, batch_size=10, flush_interval=2.0)
            await self.json_writer.start()
            pointcloud_futures = []

            self.segmenation_image_json = {}
            # organize time stamps
            time_stamps = set()
            for connection, timestamp, msg in reader.messages():
                msg = reader.deserialize(msg, connection.msgtype)
                if hasattr(msg, "header"):
                    current_time = (float)(msg.header.stamp.sec) + (float)(msg.header.stamp.nanosec) * np.power(
                        10.0, -9
                    )
                    time_stamps.add("{:.4f}".format(current_time))
            time_stamps = sorted(list(time_stamps), key=lambda x: float(x))
            time_stamps = np.array([float(t) for t in time_stamps])
            dt = 1.0 / self.fps
            first, last = time_stamps[0], time_stamps[-1]
            desired = np.arange(first, last + dt / 2, dt)
            # desired keep 4 decimal places
            desired = np.array([float("{:.4f}".format(t + dt / 10)) for t in desired])
            temp_idx = np.searchsorted(time_stamps, desired, side="left")
            temp_idx = np.clip(temp_idx, 0, len(time_stamps) - 1)
            time_stamps = time_stamps[temp_idx].tolist()
            chunk_size = 1000
            chunks = [time_stamps[i : i + chunk_size] for i in range(0, len(time_stamps), chunk_size)]
            # semantic
            label_msgs = {"hand_left": [], "hand_right": [], "head": []}
            label_dict = None
            camera_label_dict = None
            # init_image
            for chunk_index, chunk in enumerate(chunks):
                image_topics = {}
                joint_topics = {}
                tf_topics = {}
                rgb_topics = {}
                message_step = np.inf
                physics_message_step = np.inf
                img_frames = 0
                for connection in reader.connections:
                    if connection.msgtype == "sensor_msgs/msg/Image":
                        image_topics[connection.topic] = {}
                        if connection.msgcount < message_step:
                            message_step = connection.msgcount
                    elif connection.msgtype == "sensor_msgs/msg/JointState" and (
                        connection.topic == "/joint_states" or str(connection.topic).startswith("/articulated/")
                    ):
                        joint_topics[connection.topic] = {}
                        if connection.msgcount < physics_message_step:
                            physics_message_step = connection.msgcount
                    elif connection.topic == "/tf":
                        tf_topics[connection.topic] = {}
                        if connection.msgcount < physics_message_step:
                            physics_message_step = connection.msgcount
                t_start, t_end = float(chunk[0]), float(chunk[-1])
                for connection, timestamp, msg in reader.messages():
                    if connection.msgtype == "sensor_msgs/msg/Image":
                        image_msg = reader.deserialize(msg, "sensor_msgs/msg/Image")
                        current_time = (float)(image_msg.header.stamp.sec) + (float)(
                            image_msg.header.stamp.nanosec
                        ) * np.power(10.0, -9)
                        if current_time < t_start or current_time > t_end:
                            continue
                        if "{:.4f}".format(current_time) not in image_topics[connection.topic]:
                            image_topics[connection.topic]["{:.4f}".format(current_time)] = msg
                    elif connection.msgtype == "sensor_msgs/msg/JointState" and (
                        connection.topic == "/joint_states" or str(connection.topic).startswith("/articulated/")
                    ):
                        joint_msg = reader.deserialize(msg, "sensor_msgs/msg/JointState")
                        current_time = (float)(joint_msg.header.stamp.sec) + (float)(
                            joint_msg.header.stamp.nanosec
                        ) * np.power(10.0, -9)
                        if current_time < t_start or current_time > t_end:
                            continue
                        if "{:.4f}".format(current_time) not in joint_topics[connection.topic]:
                            joint_topics[connection.topic]["{:.4f}".format(current_time)] = msg
                    elif connection.topic == "/tf":
                        tf_msg = reader.deserialize(msg, "tf2_msgs/msg/TFMessage")
                        current_time = (float)(tf_msg.transforms[0].header.stamp.sec) + (float)(
                            tf_msg.transforms[0].header.stamp.nanosec
                        ) * np.power(10.0, -9)
                        if current_time < t_start or current_time > t_end:
                            continue
                        if "{:.4f}".format(current_time) not in tf_topics[connection.topic]:
                            tf_topics[connection.topic]["{:.4f}".format(current_time)] = msg
                    elif connection.topic == "/tf_static":
                        tf_static_msg = reader.deserialize(msg, "tf2_msgs/msg/TFMessage")
                        self.tf_static_msg = tf_static_msg
                    elif connection.msgtype == "std_msgs/msg/String":
                        lable_msg = reader.deserialize(msg, "std_msgs/msg/String").data
                        if "Left" in connection.topic:
                            label_msgs["hand_left"].append(lable_msg)
                        elif "Right" in connection.topic:
                            label_msgs["hand_right"].append(lable_msg)
                        else:
                            label_msgs["head"].append(lable_msg)
                    elif connection.msgtype == "sensor_msgs/msg/CompressedImage":
                        rgb_msg = reader.deserialize(msg, "sensor_msgs/msg/CompressedImage")
                        current_time = (float)(rgb_msg.header.stamp.sec) + (float)(
                            rgb_msg.header.stamp.nanosec
                        ) * np.power(10.0, -9)
                        if current_time < t_start or current_time > t_end:
                            continue
                        if "{:.4f}".format(current_time) not in rgb_topics[connection.topic]:
                            rgb_topics[connection.topic]["{:.4f}".format(current_time)] = msg

                if self.with_senmatic:
                    if label_dict is None:
                        label_dict = {}
                    label_dict, camera_label_dict = get_semantic_dict_by_msgs(label_dict, label_msgs)
                    self.get_objects_size_map(label_dict)
                    self.get_part_transform_matrix_map(label_dict)
                # playback range
                print(f"Playback time range: {self.playback_timerange}")

                # Align timestamp
                start_time_stamp = max(
                    max([min(image_topics[topic].keys(), key=lambda x: float(x)) for topic in image_topics]),
                    max([min(joint_topics[topic].keys(), key=lambda x: float(x)) for topic in joint_topics]),
                    max([min(tf_topics[topic].keys(), key=lambda x: float(x)) for topic in tf_topics]),
                )

                end_time_stamp = min(
                    min([max(image_topics[topic].keys(), key=lambda x: float(x)) for topic in image_topics]),
                    min([max(joint_topics[topic].keys(), key=lambda x: float(x)) for topic in joint_topics]),
                    min([max(tf_topics[topic].keys(), key=lambda x: float(x)) for topic in tf_topics]),
                )

                # remove ts earlier than start_time_stamp and insert ts if not exist
                ts_to_remove = []
                for ts in chunk:
                    to_remove = False
                    if float(ts) < float(start_time_stamp) or float(ts) > float(end_time_stamp):
                        to_remove = True
                        ts_to_remove.append(ts)
                    for topic in image_topics:
                        if to_remove:
                            if ts in image_topics[topic]:
                                del image_topics[topic][ts]
                        elif ts not in image_topics[topic]:
                            image_topics[topic][ts] = None
                    for topic in rgb_topics:
                        if to_remove:
                            if ts in rgb_topics[topic]:
                                del rgb_topics[topic][ts]
                        elif ts not in rgb_topics[topic]:
                            rgb_topics[topic][ts] = None
                    for topic in joint_topics:
                        if to_remove:
                            if ts in joint_topics[topic]:
                                del joint_topics[topic][ts]
                        elif ts not in joint_topics[topic]:
                            joint_topics[topic][ts] = None
                    for topic in tf_topics:
                        if to_remove:
                            if ts in tf_topics[topic]:
                                del tf_topics[topic][ts]
                        elif ts not in tf_topics[topic]:
                            tf_topics[topic][ts] = None
                for ts in ts_to_remove:
                    chunk.remove(ts)

                # sort all topics
                for topic in image_topics:
                    image_topics[topic] = {
                        k: image_topics[topic][k] for k in sorted(image_topics[topic], key=lambda x: float(x))
                    }
                for topic in rgb_topics:
                    rgb_topics[topic] = {
                        k: rgb_topics[topic][k] for k in sorted(rgb_topics[topic], key=lambda x: float(x))
                    }
                for topic in joint_topics:
                    joint_topics[topic] = {
                        k: joint_topics[topic][k] for k in sorted(joint_topics[topic], key=lambda x: float(x))
                    }
                for topic in tf_topics:
                    tf_topics[topic] = {
                        k: tf_topics[topic][k] for k in sorted(tf_topics[topic], key=lambda x: float(x))
                    }

                # If the result of a certain frame is None, fill it with the value of the next frame
                for topic in image_topics:
                    for i in range(len(image_topics[topic]) - 2, -1, -1):
                        ts = list(image_topics[topic].keys())[i]
                        ts_next = list(image_topics[topic].keys())[i + 1]
                        if image_topics[topic][ts] is None:
                            image_topics[topic][ts] = image_topics[topic][ts_next]
                    # in case the last frame is none
                    last_idx = len(image_topics[topic]) - 1
                    last_ts = list(image_topics[topic].keys())[last_idx]
                    last_frame = image_topics[topic][last_ts]
                    while last_frame is None and last_idx > 0:
                        last_idx -= 1
                        last_ts = list(image_topics[topic].keys())[last_idx]
                        last_frame = image_topics[topic][last_ts]
                    for i in range(last_idx, len(image_topics[topic])):
                        ts = list(image_topics[topic].keys())[i]
                        image_topics[topic][ts] = last_frame
                for topic in rgb_topics:
                    for i in range(len(rgb_topics[topic]) - 2, -1, -1):
                        ts = list(rgb_topics[topic].keys())[i]
                        ts_next = list(rgb_topics[topic].keys())[i + 1]
                        if rgb_topics[topic][ts] is None:
                            rgb_topics[topic][ts] = rgb_topics[topic][ts_next]
                    # in case the last frame is none
                    last_idx = len(rgb_topics[topic]) - 1
                    last_ts = list(rgb_topics[topic].keys())[last_idx]
                    last_frame = rgb_topics[topic][last_ts]
                    while last_frame is None and last_idx > 0:
                        last_idx -= 1
                        last_ts = list(rgb_topics[topic].keys())[last_idx]
                        last_frame = rgb_topics[topic][last_ts]
                    for i in range(last_idx, len(rgb_topics[topic])):
                        ts = list(rgb_topics[topic].keys())[i]
                        rgb_topics[topic][ts] = last_frame
                for topic in joint_topics:
                    for i in range(len(joint_topics[topic]) - 2, -1, -1):
                        ts = list(joint_topics[topic].keys())[i]
                        ts_next = list(joint_topics[topic].keys())[i + 1]
                        if joint_topics[topic][ts] is None:
                            joint_topics[topic][ts] = joint_topics[topic][ts_next]
                    # in case the last frame is none
                    last_idx = len(joint_topics[topic]) - 1
                    last_ts = list(joint_topics[topic].keys())[last_idx]
                    last_frame = joint_topics[topic][last_ts]
                    while last_frame is None and last_idx > 0:
                        last_idx -= 1
                        last_ts = list(joint_topics[topic].keys())[last_idx]
                        last_frame = joint_topics[topic][last_ts]
                    for i in range(last_idx, len(joint_topics[topic])):
                        ts = list(joint_topics[topic].keys())[i]
                        joint_topics[topic][ts] = last_frame
                for topic in tf_topics:
                    for i in range(len(tf_topics[topic]) - 2, -1, -1):
                        ts = list(tf_topics[topic].keys())[i]
                        ts_next = list(tf_topics[topic].keys())[i + 1]
                        if tf_topics[topic][ts] is None:
                            tf_topics[topic][ts] = tf_topics[topic][ts_next]
                    # in case the last frame is none
                    last_idx = len(tf_topics[topic]) - 1
                    last_ts = list(tf_topics[topic].keys())[last_idx]
                    last_frame = tf_topics[topic][last_ts]
                    while last_frame is None and last_idx > 0:
                        last_idx -= 1
                        last_ts = list(tf_topics[topic].keys())[last_idx]
                        last_frame = tf_topics[topic][last_ts]
                    for i in range(last_idx, len(tf_topics[topic])):
                        ts = list(tf_topics[topic].keys())[i]
                        tf_topics[topic][ts] = last_frame

                result = {
                    "schema": "simubotix.agibot.com/episode/v6",
                    "scene": {
                        "name": self.scene_name,
                        "metadata": None,
                        "scene_usd": self.scene_usd,
                        "scene_glb": self.scene_glb,
                    },
                    "lights": self.light_config,
                    "objects": [],
                    "articulated_objects": [],
                    "cameras": self.camera_info,
                    "robot": {"name": self.robot_name, "metadata": None},
                    "frames": [],
                    "fps": self.fps,
                }

                if self.with_img:
                    img_frames = 1 if img_frames == 0 else img_frames
                    result["replay_factor"] = round(physics_message_step / img_frames) * 3

                for name in self.object_names:
                    result["objects"].append({"name": name.split("/")[-1], "metadata": None})
                for name in self.articulated_object_names:
                    result["articulated_objects"].append({"name": name.split("/")[-1], "metadata": None})
                state_info = {"timestamp": []}
                episode_state = {
                    "joint": {
                        "effort": [],
                        "position": [],
                        "velocity": [],
                        "current_value": [],
                    },
                    "end": {
                        "velocity": [],
                        "angular": [],
                        "position": [],
                        "orientation": [],
                        "arm_position": [],
                        "arm_orientation": [],
                        "wrench": [],
                    },
                    "effector": {"force": [], "position": [], "index": []},
                }
                attr_names = {
                    "joint": [],
                    "end": ["left", "right"],
                    "effector": [],
                    "robot": [self.robot_name],
                }
                robot_info = {
                    "position": [],
                    "orientation": [],
                    "position_drift": [],
                    "orientation_drift": [],
                    "velocity": [],
                }

                def in_playback_time(t, pb_range):
                    for subrange in pb_range:
                        if t >= subrange[0] and t <= subrange[1]:
                            return True
                    return False

                pic_idx = -1

                for idx, ts in enumerate(chunk):
                    if self.with_img:
                        if in_playback_time(ts, self.playback_timerange):
                            continue
                        pic_idx += 1
                        depth_dir = self.output_dir + f"/camera_{chunk_index}/{pic_idx}"
                        rgb_dir = self.output_dir + f"/camera_{chunk_index}/{pic_idx}"
                        os.makedirs(depth_dir, exist_ok=True)
                        os.makedirs(rgb_dir, exist_ok=True)
                        stamp = {}

                        depth_image_paths = {}
                        rgb_image_paths = {}

                        for key in image_topics:
                            try:
                                msg = reader.deserialize(image_topics[key][ts], "sensor_msgs/msg/Image")
                            except Exception:
                                print(f"❌ Error key={key} ts={ts}")
                                continue
                            file_name = key.split("/")[-1]
                            stamp[file_name] = (float)(msg.header.stamp.sec) + (float)(
                                msg.header.stamp.nanosec
                            ) * np.power(10.0, -9)
                            if "genie_sim" in key:
                                if "depth" in file_name:
                                    if "head" in file_name:
                                        img = message_to_cvimage(msg, "32FC1") * 1000
                                    else:
                                        img = message_to_cvimage(msg, "32FC1") * 10000
                                    file_name = self.post_process_file_name(file_name, "_depth")
                                    file_path = f"{depth_dir}/{file_name}.png"
                                    depth_image_paths[file_name] = file_path
                                    cv2.imwrite(
                                        file_path,
                                        img.astype(np.uint16),
                                    )
                                    self.imag_file_name.append(file_name)
                                elif "semantic" in file_name:
                                    img = message_to_cvimage(msg, "32FC1")
                                    file_name = self.post_process_file_name(file_name, "_semantic")
                                    colored_img, segmentation_polys = label_to_color(
                                        img,
                                        label_dict,
                                        depth_dir,
                                        file_name,
                                        camera_label_dict,
                                    )
                                    self.generate_img_json(
                                        idx,
                                        segmentation_polys,
                                        label_dict,
                                        depth_dir,
                                        file_name,
                                    )
                                    if colored_img is not None:
                                        cv2.imwrite(
                                            depth_dir + "/{}.png".format(file_name),
                                            colored_img,
                                        )
                                else:
                                    img = message_to_cvimage(msg, "bgr8")
                                    file_name = self.post_process_file_name(file_name, "_color")
                                    file_path = f"{rgb_dir}/{file_name}.jpg"
                                    rgb_image_paths[file_name] = file_path
                                    cv2.imwrite(file_path, img)
                                    self.imag_file_name.append(file_name)

                        for key in rgb_topics:
                            try:
                                msg = reader.deserialize(
                                    rgb_topics[key][ts],
                                    "sensor_msgs/msg/CompressedImage",
                                )
                            except Exception:
                                print(f"❌ Error key={key} ts={ts}")
                                continue
                            file_name = key.split("/")[-1]
                            stamp[file_name] = (float)(msg.header.stamp.sec) + (float)(
                                msg.header.stamp.nanosec
                            ) * np.power(10.0, -9)
                            img = message_to_cvimage(msg, "bgr8")  # change encoding type if needed
                            file_name = self.post_process_file_name(file_name)
                            cv2.imwrite(rgb_dir + "/{}.jpg".format(file_name), img)
                        min_value = min(stamp.values())
                        for key in stamp:
                            stamp[key] = min_value
                        with open(rgb_dir + "/time_stamp.json", "w", encoding="utf-8") as f:
                            json.dump(stamp, f, indent=4)

                    single_frame_state = {
                        "objects": {},
                        "articulated_object": {},
                        "cameras": {},
                        "ee": {},
                        "robot": {},
                    }
                    skip_frame = False
                    for key in joint_topics:
                        if key == "/joint_states":
                            msg = reader.deserialize(joint_topics[key][ts], "sensor_msgs/msg/JointState")
                            joint_timestamp = (float)(msg.header.stamp.sec) + (float)(
                                msg.header.stamp.nanosec
                            ) * np.power(10.0, -9)
                            msg_name, msg_position, msg_velocity, msg_effort = reorder_joint_state(msg)
                            single_joint_info = {
                                "joint_name": msg_name,
                                "joint_position": msg_position,
                                "joint_velocity": msg_velocity,
                                "joint_effort": msg_effort,
                            }
                        else:
                            msg = reader.deserialize(joint_topics[key][ts], "sensor_msgs/msg/JointState")
                            joint_timestamp = (float)(msg.header.stamp.sec) + (float)(
                                msg.header.stamp.nanosec
                            ) * np.power(10.0, -9)
                            single_frame_state["articulated_object"][key.split("/")[-1]] = {
                                "joints": {
                                    "joint_name": msg.name,
                                    "joint_position": msg.position.tolist(),
                                    "joint_velocity": msg.velocity.tolist(),
                                    "joint_effort": msg.effort.tolist(),
                                }
                            }
                        if in_playback_time(ts, self.playback_timerange):
                            skip_frame = True
                            break
                    if skip_frame:
                        print(f"Skip frame at {ts}")
                        continue
                    single_frame_state["time_stamp"] = ts
                    frame_idx = -1
                    if len(self.frame_status) > 0:
                        if joint_timestamp < self.frame_status[0]["time_stamp"]:
                            value = self.frame_status[0]["frame_state"]
                        else:
                            while joint_timestamp < self.frame_status[frame_idx]["time_stamp"]:
                                frame_idx -= 1
                                value = self.frame_status[frame_idx]["frame_state"]
                        single_frame_state["frame_state"] = value
                    single_frame_state["robot"]["joints"] = single_joint_info
                    single_ee_info_r = {
                        "time_stamp": 0,
                        "position": [0, 0, 0],
                        "rotation": [1, 0, 0, 0],
                    }
                    single_ee_info_l = {
                        "time_stamp": 0,
                        "position": [0, 0, 0],
                        "rotation": [1, 0, 0, 0],
                    }
                    camera_poses = {}
                    point_cloud_cameras = []
                    for key in tf_topics:
                        msg = reader.deserialize(tf_topics[key][ts], "tf2_msgs/msg/TFMessage")
                        for transform in msg.transforms:
                            position = np.array(
                                [
                                    transform.transform.translation.x,
                                    transform.transform.translation.y,
                                    transform.transform.translation.z,
                                ]
                            )
                            # for internal use, the quaternion is in the wxyz format
                            rotation = np.array(
                                [
                                    transform.transform.rotation.w,
                                    transform.transform.rotation.x,
                                    transform.transform.rotation.y,
                                    transform.transform.rotation.z,
                                ]
                            )
                            if self.right_gripper_center_name in transform.child_frame_id:
                                single_ee_info_r = {
                                    "time_stamp": (float)(transform.header.stamp.sec)
                                    + (float)(transform.header.stamp.nanosec) * np.power(10.0, -9),
                                    "position": position,
                                    "rotation": rotation,
                                }
                                single_frame_state["ee"]["right"] = {"pose": (get_pose(*(position, rotation)).tolist())}
                            elif self.left_gripper_center_name in transform.child_frame_id:
                                single_ee_info_l = {
                                    "time_stamp": (float)(transform.header.stamp.sec)
                                    + (float)(transform.header.stamp.nanosec) * np.power(10.0, -9),
                                    "position": position,
                                    "rotation": rotation,
                                }
                                single_frame_state["ee"]["left"] = {"pose": (get_pose(*(position, rotation)).tolist())}
                            elif "Camera" in transform.child_frame_id or "Fisheye" in transform.child_frame_id:
                                rotation_x_180 = np.array(
                                    [
                                        [1.0, 0.0, 0.0, 0],
                                        [0.0, -1.0, 0.0, 0],
                                        [0.0, 0.0, -1.0, 0],
                                        [0, 0, 0, 1],
                                    ]
                                )
                                camera_key = self.post_process_file_name(transform.child_frame_id, "_color")
                                camera_poses[camera_key] = get_pose(*(position, rotation))
                                if (
                                    self.with_senmatic
                                    and camera_key in depth_image_paths
                                    and camera_key in rgb_image_paths
                                ):
                                    point_cloud_cameras.append(camera_key)
                                single_frame_state["cameras"][camera_key] = {
                                    "pose": (get_pose(*(position, rotation)) @ rotation_x_180).tolist()
                                }
                            elif self.arm_base_prim_path in transform.child_frame_id:
                                single_frame_state["robot"]["arm_base_pose"] = get_pose(*(position, rotation)).tolist()
                            else:
                                if "link" not in transform.child_frame_id:
                                    single_frame_state["objects"][transform.child_frame_id] = {
                                        "pose": get_pose(*(position, rotation)).tolist()
                                    }
                                if "world" == transform.header.frame_id and "base_link" == transform.child_frame_id:

                                    single_frame_state["robot"]["pose"] = get_pose(*(position, rotation)).tolist()
                    pose = self.robot_init_position
                    rot = self.robot_init_rotation
                    single_frame_state["robot"]["pose"] = get_pose(*(pose, rot)).tolist()
                    for camera_key in point_cloud_cameras:
                        future = executor.submit(
                            generate_pointcloud,
                            depth_dir,
                            camera_key,
                            self.robot_init_position,
                            self.robot_init_rotation,
                            self.camera_info,
                            depth_image_paths[camera_key],
                            rgb_image_paths[camera_key],
                            camera_poses[camera_key],
                        )
                        pointcloud_futures.append(future)

                    if self.with_senmatic:
                        await self.dump_objects_bbox3d(
                            label_dict,
                            idx,
                            single_frame_state,
                            chunk_index,
                        )
                    result["frames"].append(single_frame_state)
                    # align hdf5
                    state_info["timestamp"].append(ts)
                    episode_state["joint"]["position"].append(single_joint_info["joint_position"])
                    episode_state["joint"]["velocity"].append(single_joint_info["joint_velocity"])
                    episode_state["joint"]["effort"].append(single_joint_info["joint_effort"])
                    if not attr_names["joint"]:
                        attr_names["joint"] = single_joint_info["joint_name"]
                    # collect ee pose in robot frame
                    l_ee_world_pose = get_pose(*(single_ee_info_l["position"], single_ee_info_l["rotation"]))
                    r_ee_world_pose = get_pose(*(single_ee_info_r["position"], single_ee_info_r["rotation"]))
                    world_to_robot = np.linalg.inv(np.array(single_frame_state["robot"]["pose"]))
                    l_ee_robot_pose = world_to_robot @ l_ee_world_pose
                    r_ee_robot_pose = world_to_robot @ r_ee_world_pose
                    episode_state["end"]["position"].append([l_ee_robot_pose[:3, 3], r_ee_robot_pose[:3, 3]])
                    l_ee_robot_quaternion = get_quaternion_xyzw_from_rotation_matrix(l_ee_robot_pose[:3, :3])
                    r_ee_robot_quaternion = get_quaternion_xyzw_from_rotation_matrix(r_ee_robot_pose[:3, :3])
                    episode_state["end"]["orientation"].append([l_ee_robot_quaternion, r_ee_robot_quaternion])
                    # record end effector pose in arm base frame
                    arm_base_pose = np.array(single_frame_state["robot"]["arm_base_pose"])
                    wolrd_to_arm_base = np.linalg.inv(arm_base_pose)
                    l_ee_arm_base_pose = wolrd_to_arm_base @ l_ee_world_pose
                    r_ee_arm_base_pose = wolrd_to_arm_base @ r_ee_world_pose
                    episode_state["end"]["arm_position"].append([l_ee_arm_base_pose[:3, 3], r_ee_arm_base_pose[:3, 3]])
                    l_ee_arm_base_quaternion = get_quaternion_xyzw_from_rotation_matrix(l_ee_arm_base_pose[:3, :3])
                    r_ee_arm_base_quaternion = get_quaternion_xyzw_from_rotation_matrix(r_ee_arm_base_pose[:3, :3])
                    episode_state["end"]["arm_orientation"].append([l_ee_arm_base_quaternion, r_ee_arm_base_quaternion])

                    episode_state["effector"]["index"].append(idx)
                    episode_state["effector"]["force"].append(0)
                    if not attr_names["effector"]:
                        attr_names["effector"] = ["left", "right"]
                    robot_info["position"].append(self.robot_init_position)
                    robot_info["orientation"].append(wxyz_to_xyzw(self.robot_init_rotation))
                    robot_info["orientation_drift"].append([0, 0, 0, 1])
                    robot_info["position_drift"].append([0, 0, 0])
                    robot_info["velocity"].append(0.0)

                state_info["state"] = {
                    "joint": episode_state["joint"],
                    "end": episode_state["end"],
                    "effector": episode_state["effector"],
                }
                state_info["action"] = {
                    "joint": episode_state["joint"],
                    "end": episode_state["end"],
                    "effector": episode_state["effector"],
                }
                state_info["state"]["robot"] = {
                    "position": robot_info["position"],
                    "orientation": robot_info["orientation"],
                    "position_drift": robot_info["position_drift"],
                    "orientation_drift": robot_info["orientation_drift"],
                }
                state_info["action"]["robot"] = {
                    "velocity": robot_info["velocity"],
                    "orientation": robot_info["orientation"],
                }

                with h5py.File(self.output_dir + f"/aligned_joints_all_{chunk_index}.h5", "w") as hdf:
                    hdf.create_dataset(
                        "timestamp",
                        data=np.array(state_info["timestamp"], dtype="float32"),
                    )
                    state_group = hdf.create_group("state")
                    for state_key, state_value in state_info["state"].items():
                        group = state_group.create_group(state_key)
                        if state_key == "joint":
                            group.attrs["name"] = attr_names["joint"]
                        elif state_key == "end":
                            group.attrs["name"] = attr_names["end"]
                        elif state_key == "effector":
                            group.attrs["name"] = attr_names["effector"]
                            group.attrs["category"] = ["continuous"]
                        elif state_key == "robot":
                            group.attrs["name"] = attr_names["robot"]
                        for inner_key, value in state_value.items():
                            if isinstance(value, (int, float, bool)):
                                dataset = group.create_dataset(inner_key, data=value)
                            elif isinstance(value, str):
                                dataset = group.create_dataset(inner_key, data=np.string_(value))
                            elif isinstance(value, list):
                                dataset = group.create_dataset(inner_key, data=np.array(value, dtype="float32"))

                    state_group_1 = hdf.create_group("action")
                    for state_key, state_value in state_info["action"].items():
                        group = state_group_1.create_group(state_key)
                        if state_key == "joint":
                            group.attrs["name"] = attr_names["joint"]
                        elif state_key == "end":
                            group.attrs["name"] = attr_names["end"]
                        elif state_key == "effector":
                            group.attrs["name"] = attr_names["effector"]
                            group.attrs["category"] = ["continuous"]
                        elif state_key == "robot":
                            group.attrs["name"] = attr_names["robot"]
                        for inner_key, value in state_value.items():
                            if isinstance(value, (int, float, bool)):
                                dataset = group.create_dataset(inner_key, data=value)
                            elif isinstance(value, str):
                                dataset = group.create_dataset(inner_key, data=np.string_(value))
                            elif isinstance(value, list):
                                dataset = group.create_dataset(inner_key, data=np.array(value, dtype="float32"))
                if label_dict:
                    lable_result = []
                    for key, value in label_dict.items():
                        if key != "1" and key != "time_stamp":
                            idx = (int)(key) - 1
                            if idx < 0:
                                idx = 0
                            value["id"] = idx
                            lable_result.append(value)
                    result["semantic_lables"] = lable_result

                state_out_dir = self.output_dir + f"/state_{chunk_index}.json"
                with open(state_out_dir, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4)
                print(f"State file saved to {state_out_dir}")

            await self.json_writer.stop()
            try:
                if "pointcloud_futures" in locals() and len(pointcloud_futures) > 0:
                    print(f"Waiting for {len(pointcloud_futures)} pointcloud tasks to finish...")
                    concurrent.futures.wait(pointcloud_futures)
                    for f in pointcloud_futures:
                        try:
                            f.result()
                        except Exception as e:
                            print(f"Pointcloud task failed: {e}")
            except Exception as e:
                print(f"Error while waiting pointcloud futures: {e}")

            def delete_db3_files(directory):
                for file in Path(directory).rglob("*.db3"):
                    try:
                        file.unlink()
                        print(f"Delete file: {file}")
                    except OSError as e:
                        print(f"Delete file failed: {file}: {e}")

            def delete_mcap_files(directory):
                for file in Path(directory).rglob("*.mcap"):
                    try:
                        file.unlink()
                        print(f"Delete file: {file}")
                    except OSError as e:
                        print(f"Delete file failed: {file}: {e}")

            delete_db3_files(self.output_dir)
            delete_mcap_files(self.output_dir)
            merge_camera(self.output_dir)
            check_camera(self.output_dir)
            merge_state_json(self.output_dir)
            merge_h5(self.output_dir)

            if self.with_video:
                try:
                    print(self.output_dir)
                    file_name = self.output_dir.split("/")[-1] + "_0"
                    if not os.path.exists(os.path.join(self.output_dir, "observations", "videos")):
                        os.makedirs(os.path.join(self.output_dir, "observations", "videos"))
                    for image_file in list(set(self.imag_file_name)):
                        camera_type = "png" if "depth" in image_file else "jpg"
                        if "depth" in image_file:
                            # Use lossless PNG encoding for depth videos
                            subprocess.run(
                                [
                                    "ffmpeg",
                                    "-framerate",
                                    "30",
                                    "-i",
                                    f"{self.output_dir}/camera/%d/{image_file}.{camera_type}",
                                    "-c:v",
                                    "png",
                                    "-pix_fmt",
                                    "rgb24",
                                    "-r",
                                    "30",
                                    f"{self.output_dir}/observations/videos/{image_file}.mp4",
                                ],
                            )
                        else:
                            # Use compressed encoding for RGB videos
                            subprocess.run(
                                [
                                    "ffmpeg",
                                    "-framerate",
                                    "30",
                                    "-i",
                                    f"{self.output_dir}/camera/%d/{image_file}.{camera_type}",
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "medium",
                                    "-crf",
                                    "18",
                                    "-g",
                                    "30",
                                    "-r",
                                    "30",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-movflags",
                                    "+faststart",
                                    f"{self.output_dir}/observations/videos/{image_file}.mp4",
                                ],
                            )
                    print(f"Video file saved to {self.output_dir}")
                    subprocess.run(["rm", "-Rf", f"{self.output_dir}/{file_name}.mcap"])
                    print("Successfully transfer h264")
                except subprocess.CalledProcessError as e:
                    print(f"Error removing file: {e}")
                    sys.exit(1)


if __name__ == "__main__":
    input_dir = ""
    output_dir = ""
    with open(f"{input_dir}/recording_info.json", "r") as f:
        task_info = json.load(f)
    ros_extrater = Ros_Extrater(
        bag_file=input_dir,
        output_dir=input_dir,
        task_info=task_info,
    )
    asyncio.run(ros_extrater.extract())
