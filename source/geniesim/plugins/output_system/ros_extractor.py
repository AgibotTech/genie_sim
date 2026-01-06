# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, sys
import json, h5py
import numpy as np
import subprocess
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from pathlib import Path
from rosbags.highlevel import AnyReader
from geniesim.plugins.logger import Logger
from rosbags.image import message_to_cvimage, compressed_image_to_cvimage
import signal, time

from PIL import Image

logger = Logger()  # Create singleton instance

count = 0

RESULT_TEMPLATE = {
    "schema": "simubotix.agibot.com/episode/v6",
    "scene": {
        "name": None,
        "metadata": None,
        "scene_usd": None,
        "scene_glb": None,
    },
    "lights": None,
    "objects": [],
    "articulated_objects": [],
    "cameras": None,
    "robot": {"name": None, "metadata": None},
    "frames": [],
    "fps": None,
    "use_video_view": True,
}

EPISODE_STATE_TEMPLATE = {
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
        "wrench": [],
    },
    "effector": {"force": [], "position": [], "index": []},
}

ROBOT_INFO_TEMPLATE = {
    "position": [],
    "orientation": [],
    "position_drift": [],
    "orientation_drift": [],
    "velocity": [],
}


class Ros_Extrater:
    def __init__(
        self,
        bag_file,
        output_dir,
        robot_init_position=None,
        robot_init_rotation=None,
        camera_info={},
        robot_name="G1",
        scene_name="test",
        scene_usd="",
        scene_glb="",
        object_names={},
        frame_status=[],
        fps=30,
        light_config=[],
        gripper_names=[],
        with_img=False,
        with_video=False,
        playback_timerange=[],
        image_downsample=1,  # Image downsampling factor, default 1 frame out of every 2
        max_io_workers=4,  # Maximum number of IO worker threads
        ffmpeg_threads=2,  # ffmpeg encoding threads, to avoid CPU contention across multiple pods
        record=False,
    ):
        self.bag_file = bag_file
        self.output_dir = output_dir
        self.robot_init_position = robot_init_position
        self.robot_init_rotation = robot_init_rotation
        self.camera_info = {
            "head": {
                "intrinsic": {
                    "width": 1280,
                    "height": 720,
                    "fx": 634.0862399675711,
                    "fy": 634.0862399675711,
                    "ppx": 640.0,
                    "ppy": 360.0,
                },
                "output": {
                    "rgb": "camera/{frame_num}/head.jpg",
                    "video": "head.mp4",
                    "depth": "camera/{frame_num}/head_depth.png",
                },
            },
            "hand_left": {
                "intrinsic": {
                    "width": 848,
                    "height": 480,
                    "fx": 270.62887111534724,
                    "fy": 270.62887111534724,
                    "ppx": 424.0,
                    "ppy": 240.0,
                },
                "output": {
                    "rgb": "camera/{frame_num}/hand_left.jpg",
                    "video": "hand_left.mp4",
                    "depth": "camera/{frame_num}/hand_left_depth.png",
                },
            },
            "hand_right": {
                "intrinsic": {
                    "width": 848,
                    "height": 480,
                    "fx": 293.18128152238654,
                    "fy": 293.18128152238654,
                    "ppx": 424.0,
                    "ppy": 240.0,
                },
                "output": {
                    "rgb": "camera/{frame_num}/hand_right.jpg",
                    "video": "hand_right.mp4",
                    "depth": "camera/{frame_num}/hand_right_depth.png",
                },
            },
            "world_img": {
                "intrinsic": {
                    "width": 1280,
                    "height": 720,
                    "fx": 634.0862399675711,
                    "fy": 634.0862399675711,
                    "ppx": 640.0,
                    "ppy": 360.0,
                },
                "output": {
                    "rgb": "camera/{frame_num}/world_img.jpg",
                    "video": "world_img.mp4",
                },
            },
        }
        self.robot_name = robot_name
        self.scene_name = scene_name
        self.scene_usd = scene_usd
        self.scene_glb = scene_glb
        self.object_names = []
        self.articulated_object_names = []
        for name in object_names["object_prims"]:
            if name not in self.object_names:
                self.object_names.append(name)

        for name in object_names["articulated_object_prims"]:
            self.articulated_object_names.append(name)
        self.frame_status = frame_status
        self.fps = fps
        self.with_img = with_img
        self.with_video = with_video
        self.light_config = light_config
        self.gripper_names = gripper_names
        self.playback_timerange = playback_timerange
        self.image_downsample = image_downsample
        self.max_io_workers = max_io_workers
        self.ffmpeg_threads = ffmpeg_threads
        if "omnipicker" in self.gripper_names[0]:
            self.left_gripper_center_name = "gripper_l_center_link"
        else:
            self.left_gripper_center_name = "gripper_center"
        if "omnipicker" in self.gripper_names[1]:
            self.right_gripper_center_name = "gripper_r_center_link"
        else:
            self.right_gripper_center_name = "right_gripper_center"
        self.record_process = []
        self.record = record

    def load_record_topics(self):
        with open("record_topics.json", "r") as f:
            record_topics = json.load(f)
            self.record_topics = record_topics.get(self.robot_name, [])
            logger.info(f"record topics: {self.record_topics}")

    def set_record_topics(self, topic_list):
        logger.info(f"set record topics: {topic_list}")
        self.record_topics = topic_list

    def record_rosbag(self):
        command_str = f"""
        unset PYTHONPATH
        unset LD_LIBRARY_PATH
        source /opt/ros/humble/setup.bash
        ros2 bag record -o {self.output_dir} {' '.join(self.record_topics)}
        """
        process = subprocess.Popen(command_str, shell=True, executable="/bin/bash", preexec_fn=os.setsid)
        logger.info("started record")
        self.record_process.append(process)

    def __enter__(self):
        if self.record:
            self.load_record_topics()
            self.record_rosbag()
            logger.info("record started")

        return self

    def __exit__(self):
        if self.record:
            for process in self.record_process:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                process.wait(timeout=10)
                time.sleep(1.0)
            self.extract()
            logger.info("record stopped")

    def post_process_file_name(self, file_name, extra_name="", remove_name=False):
        if file_name == "camera_rgb":
            return "world_img"
        if "Head" in file_name or "head" in file_name:
            file_name = "head" + extra_name
        elif "Right" in file_name or "right" in file_name:
            file_name = "hand_right" + extra_name
        elif "Left" in file_name or "left" in file_name:
            file_name = "hand_left" + extra_name
        elif "Top" in file_name or "top" in file_name:
            file_name = "head_front_fisheye" + extra_name
        elif remove_name:
            index = file_name.rfind("_")
            file_name = file_name[:index] + extra_name
        return file_name

    def extract(self):
        # Utility functions
        def to_second(msg):
            NANOSEC_TO_SEC = 1e-9
            return round(msg.header.stamp.sec + msg.header.stamp.nanosec * NANOSEC_TO_SEC, 4)

        def get_pose(xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
            def get_rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
                w, x, y, z = quat
                rot = np.array(
                    [
                        [
                            2 * (w**2 + x**2) - 1,
                            2 * (x * y - w * z),
                            2 * (x * z + w * y),
                        ],
                        [
                            2 * (x * y + w * z),
                            2 * (w**2 + y**2) - 1,
                            2 * (y * z - w * x),
                        ],
                        [
                            2 * (x * z - w * y),
                            2 * (y * z + w * x),
                            2 * (w**2 + z**2) - 1,
                        ],
                    ]
                )
                return rot

            pose = np.eye(4)
            pose[:3, :3] = get_rotation_matrix_from_quaternion(quat_wxyz)
            pose[:3, 3] = xyz
            return pose

        def in_playback_time(t, pb_range):
            for subrange in pb_range:
                if t >= subrange[0] and t <= subrange[1]:
                    return True
            return False

        def write_group_data(parent_group, state_data, attr_names_dict):
            """Helper function: writes HDF5 group data"""
            for state_key, state_value in state_data.items():
                group = parent_group.create_group(state_key)
                if state_key in attr_names_dict:
                    group.attrs["name"] = attr_names_dict[state_key]
                    if state_key == "effector":
                        group.attrs["category"] = ["continuous"]

                for inner_key, value in state_value.items():
                    if isinstance(value, (int, float, bool)):
                        group.create_dataset(inner_key, data=value)
                    elif isinstance(value, str):
                        group.create_dataset(inner_key, data=np.string_(value))
                    elif isinstance(value, list):
                        group.create_dataset(inner_key, data=np.array(value, dtype="float32"))

        def delete_db3_files(directory):
            db3_files = list(Path(directory).rglob("*.db3"))
            for file in db3_files:
                try:
                    file.unlink()
                    logger.warning(f"Delete file: {file}")
                except OSError as e:
                    logger.error(f"Delete file failed: {file}: {e}")

        def reverse_relabel_gripper(g_pos):
            return min(120, max(35, 120 - (120 - 35) * pow((g_pos), 2)))

        def omnipicker_reverse_relabel_gripper(g_pos):
            return min(120, max(0.0, 1.2 * 120 * (0.75 - g_pos)))

        # Start parsing
        with AnyReader([Path(self.bag_file)]) as reader:
            image_topics = {}
            joint_topics = {}
            tf_topics = {}
            rgb_topics = {}
            static_infos = []
            message_step = np.inf
            physics_message_step = np.inf
            img_frames = 0
            self.imag_file_name = []

            # Count and truncate image and physics information quantities
            for connection in reader.connections:
                if connection.msgtype == "sensor_msgs/msg/Image":
                    rgb_topics[connection.topic] = {}
                    if connection.msgcount < message_step:
                        message_step = connection.msgcount
                elif connection.msgtype == "sensor_msgs/msg/CompressedImage":
                    image_topics[connection.topic] = {}
                    if connection.msgcount < message_step:
                        message_step = connection.msgcount
                elif connection.msgtype == "sensor_msgs/msg/JointState" and (
                    connection.topic == "/joint_states" or connection.topic.startswith("/articulated/")
                ):
                    joint_topics[connection.topic] = {}
                    if connection.msgcount < physics_message_step:
                        physics_message_step = connection.msgcount
                elif connection.topic == "/tf":
                    tf_topics[connection.topic] = {}
                    if connection.msgcount < physics_message_step:
                        physics_message_step = connection.msgcount

            # Collect all timestamps and messages
            time_stamps_set = set()
            label_dict = None

            for connection, timestamp, msg in reader.messages():
                # Process based on message type
                if connection.msgtype == "sensor_msgs/msg/Image":
                    image_msg = reader.deserialize(msg, "sensor_msgs/msg/Image")
                    ts_key = to_second(image_msg)
                    time_stamps_set.add(ts_key)
                    if ts_key not in rgb_topics[connection.topic]:
                        rgb_topics[connection.topic][ts_key] = msg
                elif connection.msgtype == "sensor_msgs/msg/CompressedImage":
                    rgb_msg = reader.deserialize(msg, "sensor_msgs/msg/CompressedImage")
                    ts_key = to_second(rgb_msg)
                    time_stamps_set.add(ts_key)
                    if ts_key not in image_topics[connection.topic]:
                        image_topics[connection.topic][ts_key] = msg

                elif connection.msgtype == "sensor_msgs/msg/JointState" and (
                    connection.topic == "/joint_states" or connection.topic.startswith("/articulated/")
                ):
                    joint_msg = reader.deserialize(msg, "sensor_msgs/msg/JointState")
                    ts_key = to_second(joint_msg)
                    time_stamps_set.add(ts_key)
                    if ts_key not in joint_topics[connection.topic]:
                        joint_topics[connection.topic][ts_key] = msg

                elif connection.topic == "/tf":
                    tf_msg = reader.deserialize(msg, "tf2_msgs/msg/TFMessage")
                    ts_key = to_second(tf_msg.transforms[0])
                    time_stamps_set.add(ts_key)
                    if ts_key not in tf_topics[connection.topic]:
                        tf_topics[connection.topic][ts_key] = msg

                elif connection.msgtype == "std_msgs/msg/String":
                    if connection.topic == "/record/static_info":
                        static_info_str = reader.deserialize(msg, "std_msgs/msg/String").data
                        static_info = json.loads(static_info_str)
                        current_time = static_info["sim_time"]
                        task_instruction = static_info["task_instruction"]
                        static_infos.append((current_time, task_instruction))
                    else:
                        lable_msg = reader.deserialize(msg, "std_msgs/msg/String").data

                elif connection.msgtype == "sensor_msgs/msg/CompressedImage":
                    rgb_msg = reader.deserialize(msg, "sensor_msgs/msg/CompressedImage")
                    ts_key = to_second(rgb_msg)
                    time_stamps_set.add(ts_key)
                    # Uncomment to support CompressedImage
                    # if connection.topic not in rgb_topics:
                    #     rgb_topics[connection.topic] = {}
                    # if ts_key not in rgb_topics[connection.topic]:
                    #     rgb_topics[connection.topic][ts_key] = msg

            # Sort timestamps (using floats is faster)
            time_stamps = sorted(list(time_stamps_set))

            # playback range
            logger.info(f"Playback time range: {self.playback_timerange}")

            # Align timestamp
            start_time_stamp = max(
                max(min(ts_dict.keys()) for ts_dict in image_topics.values()),
                max(min(ts_dict.keys()) for ts_dict in joint_topics.values()),
                max(min(ts_dict.keys()) for ts_dict in tf_topics.values()),
            )

            end_time_stamp = min(
                min(max(ts_dict.keys()) for ts_dict in image_topics.values()),
                min(max(ts_dict.keys()) for ts_dict in joint_topics.values()),
                min(max(ts_dict.keys()) for ts_dict in tf_topics.values()),
            )

            # Filter timestamp range and align
            time_stamps = [ts for ts in time_stamps if start_time_stamp <= ts <= end_time_stamp]

            # Batch process topic alignment and padding
            all_topics_dict = {
                "image": image_topics,
                "joint": joint_topics,
                "tf": tf_topics,
                "rgb": rgb_topics,
            }

            for topics_dict in all_topics_dict.values():
                for topic in topics_dict:
                    # Remove out-of-range timestamps
                    topics_dict[topic] = {
                        ts: msg for ts, msg in topics_dict[topic].items() if start_time_stamp <= ts <= end_time_stamp
                    }

                    # Insert missing timestamps (set to None)
                    for ts in time_stamps:
                        if ts not in topics_dict[topic]:
                            topics_dict[topic][ts] = None

                    # Convert to ordered dictionary - using sorted keys
                    sorted_keys = sorted(topics_dict[topic].keys())
                    topics_dict[topic] = {k: topics_dict[topic][k] for k in sorted_keys}

                    # Forward fill None values
                    if sorted_keys:
                        last_valid = None
                        for k in reversed(sorted_keys):
                            if topics_dict[topic][k] is None:
                                if last_valid is not None:
                                    topics_dict[topic][k] = last_valid
                            else:
                                last_valid = topics_dict[topic][k]

            # Refactor timestamps - downsample timestamps
            time_stamps = time_stamps[::10]
            instruction_cnt = 1
            last_ts = 0

            if self.with_img:
                render_time_step = []
                # Pre-create directory structure
                camera_base_dir = os.path.join(self.output_dir, "camera")
                os.makedirs(camera_base_dir, exist_ok=True)

                # Apply image downsampling: process only a subset of frames to reduce IO and CPU usage
                image_time_stamps = time_stamps[:: self.image_downsample]
                logger.info(
                    f"Image downsampling: from {len(time_stamps)} frames down to {len(image_time_stamps)} frames (downsampling factor: {self.image_downsample})"
                )

                # Prepare asynchronous IO task queue
                io_tasks = []

                # Define asynchronous IO helper function (defined outside the loop for efficiency)
                def write_image_async(img_data, file_path, is_depth=False):
                    """Asynchronously writes images"""
                    try:
                        if is_depth:
                            cv2.imwrite(file_path, img_data.astype(np.uint16))
                        else:
                            # Use lower quality JPEG to reduce IO and storage
                            cv2.imwrite(file_path, img_data, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        return True
                    except Exception as e:
                        logger.error(f"Failed to write image {file_path}: {e}")
                        return False

                def write_text_async(file_path, text):
                    """Asynchronously writes text files"""
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        return True
                    except Exception as e:
                        logger.error(f"Failed to write text {file_path}: {e}")
                        return False

                def write_json_async(file_path, data):
                    """Asynchronously writes JSON files"""
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=4)
                        return True
                    except Exception as e:
                        logger.error(f"Failed to write JSON {file_path}: {e}")
                        return False

                # Batch process image frames
                for img_idx, ts in enumerate(image_time_stamps):
                    # Calculate index in original time_stamps (considering applied 10x downsampling)
                    # time_stamps already downsampled to 1 in 10, image_time_stamps further downsampled based on that
                    original_idx = img_idx * self.image_downsample
                    frame_dir = os.path.join(camera_base_dir, str(original_idx))
                    os.makedirs(frame_dir, exist_ok=True)
                    stamp = {}

                    for key in image_topics:
                        # Check if message exists to avoid NoneType errors
                        msg_raw = image_topics[key].get(ts)
                        if msg_raw is None:
                            # If current timestamp has no message, try to find the nearest valid message
                            # Find the nearest valid timestamp in this topic
                            valid_ts = None
                            for check_ts in sorted(image_topics[key].keys(), reverse=True):
                                if check_ts <= ts and image_topics[key][check_ts] is not None:
                                    valid_ts = check_ts
                                    break
                            if valid_ts is None:
                                logger.warning(f"Skipping topic {key} for timestamp {ts}, no valid message found")
                                continue
                            msg_raw = image_topics[key][valid_ts]

                        msg = reader.deserialize(msg_raw, "sensor_msgs/msg/CompressedImage")
                        file_name_parts = key.split("/")
                        file_name = file_name_parts[-1]
                        stamp[file_name] = to_second(msg)

                        if "record" in key:
                            if "depth" in file_name:
                                img = message_to_cvimage(msg, "32FC1") * 1000
                                file_name = self.post_process_file_name(file_name, remove_name=True)
                                file_path = os.path.join(frame_dir, f"{file_name}_depth.png")
                                io_tasks.append((write_image_async, (img, file_path, True)))
                            else:
                                img = compressed_image_to_cvimage(msg, "rgb8")
                                file_name = self.post_process_file_name(file_name)
                                file_path = os.path.join(frame_dir, f"{file_name}.jpg")
                                io_tasks.append((write_image_async, (img, file_path, False)))
                                if file_name not in self.imag_file_name:
                                    self.imag_file_name.append(file_name)

                    # Optimize instruction lookup
                    if len(static_infos) == 0:
                        task_instruction = "no instruction"
                    elif len(static_infos) == 1:
                        _, task_instruction = static_infos[0]
                    else:
                        if original_idx == 0:
                            _, task_instruction = static_infos[0]
                        elif len(static_infos) > instruction_cnt:
                            new_t, new_instruction = static_infos[instruction_cnt]
                            if new_t < ts and new_t >= last_ts:
                                task_instruction = new_instruction
                                instruction_cnt += 1

                    # Asynchronously write instruction
                    instruction_path = os.path.join(frame_dir, "instruction.txt")
                    io_tasks.append((write_text_async, (instruction_path, task_instruction)))

                    for key in rgb_topics:
                        # Check if topic is empty
                        if not rgb_topics[key]:
                            continue

                        # Check if message exists to avoid NoneType errors
                        msg_raw = rgb_topics[key].get(ts)
                        if msg_raw is None:
                            # If current timestamp has no message, try to find the nearest valid message
                            valid_ts = None
                            for check_ts in sorted(rgb_topics[key].keys(), reverse=True):
                                if check_ts <= ts and rgb_topics[key][check_ts] is not None:
                                    valid_ts = check_ts
                                    break
                            if valid_ts is None:
                                logger.warning(f"Skipping rgb topic {key} for timestamp {ts}, no valid message found")
                                continue
                            msg_raw = rgb_topics[key][valid_ts]

                        msg = reader.deserialize(msg_raw, "sensor_msgs/msg/CompressedImage")
                        file_name = key.split("/")[-1]
                        stamp[file_name] = to_second(msg)
                        img = message_to_cvimage(msg, "bgr8")
                        file_name = self.post_process_file_name(file_name)
                        file_path = os.path.join(frame_dir, f"{file_name}.jpg")
                        io_tasks.append((write_image_async, (img, file_path, False)))

                    # Optimization: use min directly instead of iterating
                    if stamp:
                        min_value = min(stamp.values())
                        stamp = {key: min_value for key in stamp}
                        time_stamp_path = os.path.join(frame_dir, "time_stamp.json")
                        io_tasks.append((write_json_async, (time_stamp_path, stamp)))
                        render_time_step.append(str(min_value))
                    last_ts = ts

                # Use thread pool for batch IO operations
                if io_tasks:
                    logger.info(
                        f"Starting asynchronous writing of {len(io_tasks)} IO tasks using {self.max_io_workers} worker threads"
                    )
                    with ThreadPoolExecutor(max_workers=self.max_io_workers) as executor:
                        futures = []
                        for task_func, task_args in io_tasks:
                            if isinstance(task_args, tuple):
                                future = executor.submit(task_func, *task_args)
                            else:
                                future = executor.submit(task_func, task_args)
                            futures.append(future)

                        # Wait for all tasks to complete
                        completed = 0
                        for future in as_completed(futures):
                            try:
                                future.result()
                                completed += 1
                                if completed % 100 == 0:
                                    logger.info(f"Completed {completed}/{len(io_tasks)} IO tasks")
                            except Exception as e:
                                logger.error(f"IO task execution failed: {e}")
                    logger.info(f"All IO tasks completed, total {len(io_tasks)} tasks processed")

            # Configure output
            result = RESULT_TEMPLATE
            result["scene"]["name"] = self.scene_name
            result["scene"]["scene_usd"] = self.scene_usd
            result["scene"]["scene_glb"] = self.scene_glb
            result["lights"] = self.light_config
            result["robot"]["name"] = self.robot_name
            result["fps"] = self.fps
            result["use_video_view"] = True

            state_info = {"timestamp": []}
            episode_state = EPISODE_STATE_TEMPLATE
            attr_names = {
                "joint": [],
                "end": ["left", "right"],
                "effector": [],
                "robot": [self.robot_name],
            }
            robot_info = ROBOT_INFO_TEMPLATE

            # Pre-calculate camera rotation matrix (constant)
            rotation_x_180 = np.array(
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, -1.0, 0.0, 0],
                    [0.0, 0.0, -1.0, 0],
                    [0, 0, 0, 1],
                ]
            )

            for idx, ts in enumerate(time_stamps):
                single_frame_state = {
                    "objects": {},
                    "articulated_object": {},
                    "cameras": {},
                    "ee": {},
                    "robot": {},
                }
                skip_frame = False
                single_joint_info = None
                joint_timestamp = 0

                for key in joint_topics:
                    msg = reader.deserialize(joint_topics[key][ts], "sensor_msgs/msg/JointState")
                    joint_timestamp = to_second(msg)

                    joint_info = {
                        "joint_name": msg.name,
                        "joint_position": msg.position.tolist(),
                        "joint_velocity": msg.velocity.tolist(),
                        "joint_effort": msg.effort.tolist(),
                    }

                    if key == "/joint_states":
                        single_joint_info = joint_info
                    else:
                        single_frame_state["articulated_object"][key.split("/")[-1]] = {"joints": joint_info}

                    if in_playback_time(joint_timestamp, self.playback_timerange):
                        skip_frame = True
                        break
                if skip_frame:
                    logger.info(f"Skip frame at {joint_timestamp}")
                    continue
                single_frame_state["time_stamp"] = joint_timestamp
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
                for key in tf_topics:
                    msg = reader.deserialize(tf_topics[key][ts], "tf2_msgs/msg/TFMessage")
                    for transform in msg.transforms:
                        # Optimization: directly construct numpy array
                        trans = transform.transform.translation
                        rot = transform.transform.rotation
                        position = np.array([trans.x, trans.y, trans.z])
                        rotation = np.array([rot.w, rot.x, rot.y, rot.z])

                        child_id = transform.child_frame_id

                        if self.right_gripper_center_name in child_id:
                            single_ee_info_r = {
                                "time_stamp": to_second(transform),
                                "position": position,
                                "rotation": rotation,
                            }
                            single_frame_state["ee"]["right"] = {"pose": get_pose(position, rotation).tolist()}
                        elif self.left_gripper_center_name in child_id:
                            single_ee_info_l = {
                                "time_stamp": to_second(transform),
                                "position": position,
                                "rotation": rotation,
                            }
                            single_frame_state["ee"]["left"] = {"pose": get_pose(position, rotation).tolist()}
                        elif "Camera" in child_id or "Fisheye" in child_id:
                            camera_key = self.post_process_file_name(child_id)
                            single_frame_state["cameras"][camera_key] = {
                                "pose": (get_pose(position, rotation) @ rotation_x_180).tolist()
                            }
                        else:
                            if "link" not in child_id:
                                single_frame_state["objects"][child_id] = {
                                    "pose": get_pose(position, rotation).tolist()
                                }
                            if child_id == "base_link":
                                single_frame_state["robot"]["pose"] = get_pose(position, rotation).tolist()

                result["frames"].append(single_frame_state)
                # align hdf5
                state_info["timestamp"].append(joint_timestamp)
                episode_state["joint"]["position"].append(single_joint_info["joint_position"])
                episode_state["joint"]["velocity"].append(single_joint_info["joint_velocity"])
                episode_state["joint"]["effort"].append(single_joint_info["joint_effort"])
                if not attr_names["joint"]:
                    attr_names["joint"] = single_joint_info["joint_name"]
                episode_state["end"]["position"].append([single_ee_info_l["position"], single_ee_info_r["position"]])
                episode_state["end"]["orientation"].append([single_ee_info_l["rotation"], single_ee_info_r["rotation"]])
                if len(single_joint_info["joint_position"]) > 18:
                    left_gripper_position = max(0.0, min(1.0, (1 - single_joint_info["joint_position"][18])))
                    right_gripper_position = max(0.0, min(1.0, (1 - single_joint_info["joint_position"][20])))

                    episode_state["effector"]["position"].append([left_gripper_position, right_gripper_position])
                episode_state["effector"]["index"].append(idx)
                episode_state["effector"]["force"].append(0)
                if not attr_names["effector"]:
                    attr_names["effector"] = ["left", "right"]
                robot_info["position"].append(self.robot_init_position)
                robot_info["orientation"].append(self.robot_init_rotation)
                robot_info["orientation_drift"].append([1, 0, 0, 0])
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

            # HDF5 writing
            h5_file_path = os.path.join(self.output_dir, "aligned_joints_all.h5")
            with h5py.File(h5_file_path, "w") as hdf:
                hdf.create_dataset("timestamp", data=np.array(state_info["timestamp"], dtype="float32"))
                state_group = hdf.create_group("state")
                write_group_data(state_group, state_info["state"], attr_names)

                action_group = hdf.create_group("action")
                write_group_data(action_group, state_info["action"], attr_names)
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

            state_out_dir = os.path.join(self.output_dir, "state.json")
            with open(state_out_dir, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            logger.info(f"State file saved to {state_out_dir}")

            os.makedirs(os.path.join(self.output_dir, "parameters", "camera"), exist_ok=True)

            self.imag_file_name = list(set(self.imag_file_name))
            print("imag_file_name", self.imag_file_name)

            if self.with_video:
                logger.info("start record video")
                try:
                    logger.info(self.output_dir)
                    file_name = os.path.basename(self.output_dir) + "_0"

                    # Adjust frame rate based on downsampling
                    video_fps = max(1, int(self.fps / self.image_downsample))
                    logger.info(
                        f"Video encoding frame rate: {video_fps} fps (original: {self.fps} fps, downsampled: {self.image_downsample})"
                    )

                    # Optimization: use faster ffmpeg encoding parameters to reduce CPU usage
                    for image_file in self.imag_file_name:
                        input_pattern = os.path.join(self.output_dir, "camera", "%d", f"{image_file}.jpg")
                        output_path = os.path.join(self.output_dir, f"{image_file}.webm")

                        # Use faster encoding preset and limit threads to avoid CPU contention across multiple pods
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",  # Overwrite output file
                                "-framerate",
                                str(video_fps),
                                "-i",
                                input_pattern,
                                "-vsync",
                                "0",  # Disable framerate synchronization, allow skipping missing frames
                                "-c:v",
                                "libvpx-vp9",
                                "-b:v",
                                "2000k",  # Lower bitrate to reduce encoding time
                                "-preset",
                                "fast",  # Use fast preset to increase encoding speed
                                "-crf",
                                "28",  # Increase CRF value (lower quality) to speed up encoding
                                "-threads",
                                str(self.ffmpeg_threads),  # Limit threads to avoid CPU contention across multiple pods
                                "-row-mt",
                                "0",  # Disable row-level multithreading to reduce CPU usage
                                "-speed",
                                "4",  # Increase encoding speed (0-5, 4 is faster)
                                "-an",  # No audio
                                "-loglevel",
                                "error",  # Reduce log output
                                output_path,
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,  # Redirect output to reduce IO
                            stderr=subprocess.PIPE,
                        )
                    logger.info(f"Video file saved to {self.output_dir}")

                    # Clean up db3 files
                    db3_path = os.path.join(self.output_dir, f"{file_name}.db3")
                    if os.path.exists(db3_path):
                        os.remove(db3_path)
                    logger.info(f"Successfully transfer to WebM")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error in video conversion: {e}")
                    sys.exit(1)
                except Exception as e:
                    logger.error(f"Error: {e}")
                    sys.exit(1)
