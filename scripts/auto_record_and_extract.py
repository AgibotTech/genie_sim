#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys
import time
import signal
import subprocess
import threading
import logging
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.image import compressed_image_to_cvimage

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("auto_record_extractor")

# Constant definitions
NANOSEC_TO_SEC = 1e-9
DEFAULT_JPEG_QUALITY = 85
DEFAULT_TIMEOUT = 10.0
DEFAULT_FPS = 30
DEFAULT_MAX_IO_WORKERS = 4
PROGRESS_LOG_INTERVAL = 200


def map_topic_to_camera_name(topic_name: str) -> str:
    name = topic_name.split("/")[-1]

    if "camera_rgb" == name:
        return "world_img"
    if "Head" in name or "head" in name:
        return "head"
    if "Right" in name or "right" in name:
        return "hand_right"
    if "Left" in name or "left" in name:
        return "hand_left"

    return name


class ImageForwardRecorderNode(Node):
    def __init__(
        self,
        output_dir,
        timeout=DEFAULT_TIMEOUT,
        jpeg_quality=DEFAULT_JPEG_QUALITY,
        final_output_dir: str | None = None,
        delete_db3_after: bool = False,
    ):
        super().__init__("image_forward_recorder")

        self.output_dir = output_dir
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality
        self.final_output_dir = final_output_dir
        self.delete_db3_after = delete_db3_after
        self.max_io_workers = DEFAULT_MAX_IO_WORKERS

        os.makedirs(self.output_dir, exist_ok=True)

        self.bridge = CvBridge()

        self.subscribers = {}
        self.genie_sim_subscribers = {}
        self.record_publishers = {}

        self.last_genie_sim_message_time = time.time()
        self.message_lock = threading.Lock()

        self.record_process = None

        self.is_recording = False
        self.should_stop = False

        self.topic_discovery_timer = self.create_timer(2.0, self.discover_topics)

        self.timeout_check_timer = self.create_timer(1.0, self.check_timeout)

        logger.info(f"ImageForwardRecorderNode initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Timeout: {self.timeout}s")
        if self.final_output_dir:
            logger.info(f"Final output directory (images & videos will be moved here): {self.final_output_dir}")

    def discover_topics(self):
        topic_names_and_types = self.get_topic_names_and_types()

        for topic_name, topic_types in topic_names_and_types:
            if topic_name.startswith("/record/") and "sensor_msgs/msg/CompressedImage" in topic_types:
                if topic_name not in self.subscribers:
                    self.create_subscription_for_topic(topic_name)

            if topic_name.startswith("/genie_sim/") and "sensor_msgs/msg/Image" in topic_types:
                if topic_name not in self.genie_sim_subscribers:
                    suffix = topic_name.split("/genie_sim/")[-1]
                    record_topic = f"/record/{suffix}"
                    self.create_genie_sim_subscription(topic_name, record_topic)

    def create_subscription_for_topic(self, topic_name):
        subscription = self.create_subscription(
            CompressedImage,
            topic_name,
            lambda msg, tn=topic_name: self.image_callback(msg, tn),
            10,
        )
        self.subscribers[topic_name] = subscription
        logger.info(f"Created subscription: {topic_name}")

    def create_genie_sim_subscription(self, topic_name, record_topic):
        subscription = self.create_subscription(
            Image,
            topic_name,
            lambda msg, tn=topic_name, rt=record_topic: self.genie_sim_callback(msg, tn, rt),
            10,
        )
        self.genie_sim_subscribers[topic_name] = subscription
        if record_topic not in self.record_publishers:
            publisher = self.create_publisher(CompressedImage, record_topic, 10)
            self.record_publishers[record_topic] = publisher
            logger.info(f"Created bridge: {topic_name} (Image) -> {record_topic} (CompressedImage)")

    def genie_sim_callback(self, msg, topic_name, record_topic):
        try:
            with self.message_lock:
                self.last_genie_sim_message_time = time.time()

            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                _, compressed_data = cv2.imencode(".jpg", cv_image, encode_param)

                compressed_msg = CompressedImage()
                compressed_msg.header = msg.header
                compressed_msg.format = "jpeg"
                compressed_msg.data = compressed_data.tobytes()

                if record_topic in self.record_publishers:
                    self.record_publishers[record_topic].publish(compressed_msg)
            except Exception as e:
                logger.error(f"Error forwarding {topic_name} to {record_topic}: {e}")
        except Exception as e:
            logger.error(f"Error in genie_sim_callback for {topic_name}: {e}")

    def image_callback(self, msg, topic_name):
        try:
            if not self.is_recording:
                self.start_recording()
        except Exception as e:
            logger.error(f"Error in image_callback for {topic_name}: {e}")

    def start_recording(self):
        if self.is_recording:
            return

        try:
            if not self.subscribers:
                logger.warning("No /record/ topics discovered yet, skip starting rosbag recording.")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bag_name = f"recording_{timestamp}"
            bag_path = os.path.join(self.output_dir, bag_name)

            topics = " ".join(self.subscribers.keys())
            command_str = f"""
            unset PYTHONPATH
            unset LD_LIBRARY_PATH
            source /opt/ros/jazzy/setup.bash
            ros2 bag record -o {bag_path} {topics}
            """

            self.record_process = subprocess.Popen(
                command_str, shell=True, executable="/bin/bash", preexec_fn=os.setsid
            )

            self.is_recording = True
            self.bag_path = bag_path
            logger.info(f"Started rosbag recording to {bag_path}")

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")

    def check_timeout(self):
        if self.should_stop:
            return

        with self.message_lock:
            elapsed = time.time() - self.last_genie_sim_message_time

        if elapsed > self.timeout:
            logger.info(f"No /genie_sim messages received for {self.timeout}s, stopping and exiting...")
            if self.is_recording:
                self.stop_recording()
            self.should_stop = True

    def stop_recording(self):
        if not self.is_recording or self.record_process is None:
            return

        try:
            os.killpg(os.getpgid(self.record_process.pid), signal.SIGINT)
            self.record_process.wait(timeout=10)
            logger.info("Rosbag recording stopped")

            self.is_recording = False

            time.sleep(2.0)

            self.extract_and_convert()

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")

    def extract_and_convert(self):
        logger.info("Starting extraction and video conversion...")

        try:
            bag_dirs = [d for d in Path(self.output_dir).iterdir() if d.is_dir() and d.name.startswith("recording_")]

            if not bag_dirs:
                logger.error("No recording directory found")
                return

            bag_dir = max(bag_dirs, key=lambda x: x.stat().st_mtime)
            logger.info(f"Processing bag: {bag_dir}")

            self.extract_images_from_bag(str(bag_dir))

            self.convert_images_to_video()

            if self.delete_db3_after:
                try:
                    for db3_file in Path(bag_dir).glob("*.db3"):
                        logger.info(f"Deleting db3 file: {db3_file}")
                        db3_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Error deleting db3 files in {bag_dir}: {e}")

            if self.final_output_dir:
                try:
                    os.makedirs(self.final_output_dir, exist_ok=True)

                    if hasattr(self, "images_dir") and os.path.isdir(self.images_dir):
                        dst_images = os.path.join(self.final_output_dir, "camera")
                        if os.path.exists(dst_images):
                            shutil.rmtree(dst_images)
                        shutil.move(self.images_dir, dst_images)
                        logger.info(f"Moved camera images to {dst_images}")

                    for webm_file in Path(self.output_dir).glob("*.webm"):
                        dst_file = os.path.join(self.final_output_dir, os.path.basename(webm_file))
                        if os.path.exists(dst_file):
                            os.remove(dst_file)
                        shutil.move(str(webm_file), dst_file)
                        logger.info(f"Moved video {webm_file} to {dst_file}")

                except Exception as e:
                    logger.error(f"Error moving images/videos to final output dir: {e}")

            logger.info("Extraction and conversion completed successfully")

        except Exception as e:
            logger.error(f"Error in extraction and conversion: {e}")
            import traceback

            traceback.print_exc()

    def _decode_compressed_image(self, compressed_msg):
        cv_image = compressed_image_to_cvimage(compressed_msg, "bgr8")
        if cv_image is None:
            raise ValueError("Failed to decode compressed image")

        return cv_image

    @staticmethod
    def _to_second(msg):
        return round(msg.header.stamp.sec + msg.header.stamp.nanosec * NANOSEC_TO_SEC, 4)

    def _collect_camera_frames(self, reader, compressed_topics):
        camera_frames = {}

        for connection, timestamp, msg in reader.messages():
            if connection.topic in compressed_topics:
                try:
                    compressed_msg = reader.deserialize(msg, "sensor_msgs/msg/CompressedImage")
                    raw_topic_name = connection.topic.replace("/record/", "")
                    camera_name = map_topic_to_camera_name(raw_topic_name)

                    camera_frames.setdefault(camera_name, []).append((self._to_second(compressed_msg), compressed_msg))
                except Exception as e:
                    logger.error(f"Error collecting frame from {connection.topic}: {e}")

        return camera_frames

    def _sort_and_compute_time_range(self, camera_frames):
        sorted_frames = {}
        first_ts = {}
        last_ts = {}

        for camera_name, frames in camera_frames.items():
            frames.sort(key=lambda x: x[0])
            sorted_frames[camera_name] = frames
            first_ts[camera_name] = frames[0][0]
            last_ts[camera_name] = frames[-1][0]

        return sorted_frames, first_ts, last_ts

    def _select_reference_camera(self, sorted_frames):
        """Select reference camera (prefer world_img, then head)

        Args:
            sorted_frames: {camera_name: [(timestamp, compressed_msg), ...]}

        Returns:
            str: Reference camera name
        """
        if "world_img" in sorted_frames:
            return "world_img"
        elif "head" in sorted_frames:
            return "head"
        else:
            return list(sorted_frames.keys())[0]

    def _find_closest_frame(self, frames, idx, ref_ts):
        while idx + 1 < len(frames) and frames[idx + 1][0] <= ref_ts:
            idx += 1

        best_idx = idx
        if idx + 1 < len(frames):
            diff_left = abs(frames[idx][0] - ref_ts)
            diff_right = abs(frames[idx + 1][0] - ref_ts)
            if diff_right < diff_left:
                best_idx = idx + 1

        return best_idx, frames[best_idx][1]

    def _write_images_parallel(self, io_tasks):
        logger.info(f"Start async writing {len(io_tasks)} aligned image tasks " f"with {self.max_io_workers} workers")

        unique_frame_dirs = set(task[1] for task in io_tasks)
        for frame_dir in unique_frame_dirs:
            os.makedirs(frame_dir, exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.max_io_workers) as executor:
            futures = []
            for img, frame_dir, path, camera_name, frame_idx in io_tasks:
                futures.append(executor.submit(cv2.imwrite, path, img))

            completed = 0
            for idx, fut in enumerate(futures):
                try:
                    fut.result()
                    completed += 1
                    if completed % PROGRESS_LOG_INTERVAL == 0:
                        logger.info(f"Finished {completed}/{len(io_tasks)} aligned image writes")
                except Exception as e:
                    task_info = io_tasks[idx]
                    logger.error(f"Error writing image frame {task_info[4]} " f"for camera {task_info[3]}: {e}")

    def extract_images_from_bag(self, bag_path):
        logger.info(f"Extracting images from {bag_path}")
        self.images_dir = os.path.join(self.output_dir, "camera")
        os.makedirs(self.images_dir, exist_ok=True)
        self.topic_images = {}
        typestore = get_typestore(Stores.ROS2_JAZZY)
        with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
            compressed_topics = []
            for connection in reader.connections:
                if connection.msgtype == "sensor_msgs/msg/CompressedImage":
                    if connection.topic.startswith("/record/"):
                        compressed_topics.append(connection.topic)

            logger.info(f"Found {len(compressed_topics)} compressed image topics: {compressed_topics}")

            if not compressed_topics:
                logger.warning("No compressed image topics found in bag, skip extraction.")
                return

            camera_frames = self._collect_camera_frames(reader, compressed_topics)

            if not camera_frames:
                logger.warning("No camera frames collected from bag, skip extraction.")
                return

            sorted_frames, first_ts, last_ts = self._sort_and_compute_time_range(camera_frames)

            global_start = max(first_ts.values())
            global_end = min(last_ts.values())

            if global_start >= global_end:
                logger.warning(
                    f"No overlapping time range between cameras "
                    f"(global_start={global_start}, global_end={global_end}), "
                    "skip temporal alignment."
                )
                return

            ref_camera = self._select_reference_camera(sorted_frames)
            ref_frames = sorted_frames[ref_camera]
            time_stamps = sorted([ts for ts, _ in ref_frames if global_start <= ts <= global_end])

            if not time_stamps:
                logger.warning(
                    f"No reference timestamps in overlap range "
                    f"(camera={ref_camera}, global_start={global_start}, global_end={global_end})"
                )
                return

            logger.info(f"Temporal alignment using camera '{ref_camera}', " f"frames in overlap: {len(time_stamps)}")

            camera_indices = {name: 0 for name in sorted_frames.keys()}
            io_tasks = []

            for frame_idx, ref_ts in enumerate(time_stamps):
                frame_dir = os.path.join(self.images_dir, str(frame_idx))

                for camera_name, frames in sorted_frames.items():
                    idx = camera_indices[camera_name]

                    best_idx, compressed_msg = self._find_closest_frame(frames, idx, ref_ts)

                    camera_indices[camera_name] = best_idx

                    try:
                        cv_image = self._decode_compressed_image(compressed_msg)
                        image_path = os.path.join(frame_dir, f"{camera_name}.jpg")
                        self.topic_images.setdefault(camera_name, []).append(image_path)

                        io_tasks.append((cv_image, frame_dir, image_path, camera_name, frame_idx))

                    except Exception as e:
                        logger.error(f"Error preparing aligned frame {frame_idx} for {camera_name}: {e}")

            if io_tasks:
                self._write_images_parallel(io_tasks)

            for camera_name, images in self.topic_images.items():
                logger.info(f"Extracted {len(images)} temporally aligned frames for camera: {camera_name}")

    @staticmethod
    def _extract_frame_idx(path):
        """Extract frame_idx from path camera/{frame_idx}/{camera_name}.jpg"""
        try:
            parts = Path(path).parts
            return int(parts[-2])
        except (ValueError, IndexError):
            return 0

    def _prepare_video_frames(self, camera_name, image_paths):
        topic_dir = os.path.join(self.images_dir, camera_name)
        os.makedirs(topic_dir, exist_ok=True)

        sorted_paths = sorted(image_paths, key=self._extract_frame_idx)

        frame_indices = [self._extract_frame_idx(p) for p in sorted_paths]
        expected_indices = list(range(len(sorted_paths)))
        if frame_indices != expected_indices:
            logger.warning(
                f"Frame indices are not continuous for {camera_name}. "
                f"First 10 expected: {expected_indices[:10]}, First 10 got: {frame_indices[:10]}"
            )

        for idx, src_path in enumerate(sorted_paths):
            tmp_path = os.path.join(topic_dir, f"frame_{idx:06d}.jpg")
            if os.path.abspath(src_path) != os.path.abspath(tmp_path):
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    os.link(src_path, tmp_path)
                except OSError:
                    shutil.copy2(src_path, tmp_path)

        return os.path.join(topic_dir, "frame_%06d.jpg")

    def _run_ffmpeg(self, input_pattern, output_video, fps=30):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                input_pattern,
                "-vsync",
                "0",
                "-c:v",
                "libvpx-vp9",
                "-b:v",
                "2000k",
                "-preset",
                "fast",
                "-crf",
                "28",
                "-threads",
                "4",
                "-speed",
                "4",
                "-an",  # No audio
                "-loglevel",
                "error",
                output_video,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def convert_images_to_video(self, fps=DEFAULT_FPS):
        logger.info("Converting images to video...")

        for camera_name, image_paths in self.topic_images.items():
            if len(image_paths) == 0:
                logger.warning(f"No images found for camera {camera_name}, skipping video conversion")
                continue

            try:
                # Prepare sequentially numbered frame files
                input_pattern = self._prepare_video_frames(camera_name, image_paths)

                # Output video path
                output_video = os.path.join(self.output_dir, f"{camera_name}.webm")

                logger.info(f"Creating video for {camera_name}: {output_video}")

                # Run ffmpeg conversion
                self._run_ffmpeg(input_pattern, output_video, fps)

                logger.info(f"Video created successfully: {output_video}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Error creating video for {camera_name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error for {camera_name}: {e}")

    def cleanup(self):
        logger.info("Cleaning up...")

        # Stop recording
        if self.is_recording:
            self.stop_recording()

        # Cancel timers
        if hasattr(self, "topic_discovery_timer"):
            self.topic_discovery_timer.cancel()
        if hasattr(self, "timeout_check_timer"):
            self.timeout_check_timer.cancel()

        logger.info("Cleanup completed")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Auto record ROS2 /record/* CompressedImage topics and extract images/videos."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "auto_recordings"),
        help="Temporary output directory (default: auto_recordings in current directory)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout (seconds), automatically exit if no /genie_sim/* messages received for this duration (default: 10.0)",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=85,
        help="(Reserved parameter) JPEG quality, currently only used in post-processing (default: 85)",
    )
    parser.add_argument(
        "--final_output_dir",
        type=str,
        default=None,
        help="Final output directory: if provided, will move images/ and videos/ directories to this path after extraction",
    )
    parser.add_argument(
        "--delete_db3_after",
        action="store_true",
        help="Automatically delete .db3 files generated by ros2 bag after extraction",
    )

    cli_args = parser.parse_args(args=args)

    # Configure parameters
    output_dir = cli_args.output_dir
    timeout = cli_args.timeout
    jpeg_quality = cli_args.jpeg_quality
    final_output_dir = cli_args.final_output_dir
    delete_db3_after = cli_args.delete_db3_after

    rclpy.init(args=None)

    logger.info(f"Starting auto-record-extractor")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Timeout: {timeout}s")
    logger.info(f"JPEG quality: {jpeg_quality}")
    if final_output_dir:
        logger.info(f"Final output directory: {final_output_dir}")
    logger.info(f"Delete db3 after extraction: {delete_db3_after}")

    node = ImageForwardRecorderNode(
        output_dir=output_dir,
        timeout=timeout,
        jpeg_quality=jpeg_quality,
        final_output_dir=final_output_dir,
        delete_db3_after=delete_db3_after,
    )

    try:
        while rclpy.ok() and not node.should_stop:
            rclpy.spin_once(node, timeout_sec=0.1)

        logger.info("Node stopped, performing final cleanup...")
        node.cleanup()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        node.cleanup()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        import traceback

        traceback.print_exc()
        node.cleanup()
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            logger.warning(f"rclpy.shutdown() failed or was already called: {e}")
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
