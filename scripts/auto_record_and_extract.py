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
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
import cv2

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.image import compressed_image_to_cvimage
import rosbag2_py

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("auto_record_extractor")

# Constant definitions
NANOSEC_TO_SEC = 1e-9
DEFAULT_JPEG_QUALITY = 85
DEFAULT_TIMEOUT = 60.0
DEFAULT_IMAGE_TIMEOUT = 30.0
DEFAULT_FPS = 30
DEFAULT_MAX_IO_WORKERS = 8
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
        image_timeout=DEFAULT_IMAGE_TIMEOUT,
        jpeg_quality=DEFAULT_JPEG_QUALITY,
        final_output_dir: str | None = None,
        resize_width: int | None = None,
        resize_height: int | None = None,
    ):
        super().__init__("image_forward_recorder")

        self.output_dir = output_dir
        self.timeout = timeout
        self.image_timeout = image_timeout
        self.jpeg_quality = jpeg_quality
        self.final_output_dir = final_output_dir
        self.max_io_workers = DEFAULT_MAX_IO_WORKERS
        self.resize_width = resize_width
        self.resize_height = resize_height

        os.makedirs(self.output_dir, exist_ok=True)

        self.subscribers = {}  # /record/* CompressedImage subscriptions
        self.genie_sim_subscribers = {}  # /genie_sim/* heartbeat subscriptions (no forwarding)

        self.last_genie_sim_message_time = time.time()
        self.last_image_message_time = time.time()
        self.message_lock = threading.Lock()

        self.record_process = None

        self.is_recording = False
        self.should_stop = False
        # Start from 0 so every recording always gets a sequence number in its directory name
        self.current_episode_idx = 0
        # Episode index of the bag currently being recorded (set by start_recording).
        # Unlike current_episode_idx which is also updated by episode_done, this is
        # ONLY set when a bag is opened, making it a reliable record of what was filmed.
        self._active_episode_idx = None
        self._episode_counter_lock = threading.Lock()
        self.start_lock = threading.Lock()  # serialize image_callback → start_recording

        self.extraction_in_progress = False
        self.extraction_lock = threading.Lock()

        # Background task queue: enqueues bag paths for async extraction/convert
        self.task_queue = Queue()
        self.worker_thread = threading.Thread(target=self._task_worker, daemon=True, name="extraction-worker")
        self.worker_thread.start()

        # Tracks all bag paths that have completed extraction+concat for final concatenation
        self._processed_bags: list[str] = []
        self._processed_bags_lock = threading.Lock()

        self.sub_task_name = ""
        self.sub_task_name_received = False
        self.sub_task_name_lock = threading.Lock()
        self.sub_task_name_subscription = self.create_subscription(
            String,
            "/record/sub_task_name",
            self.sub_task_name_callback,
            10,
        )
        self.episode_done_subscription = self.create_subscription(
            String,
            "/record/episode_done",
            self.episode_done_callback,
            10,
        )
        self.episode_ack_publisher = self.create_publisher(String, "/record/episode_ack", 10)

        self.topic_discovery_timer = self.create_timer(2.0, self.discover_topics)

        self.timeout_check_timer = self.create_timer(1.0, self.check_timeout)
        if self.image_timeout > 0:
            self.image_timeout_check_timer = self.create_timer(0.5, self.check_image_timeout)

        logger.info(f"ImageForwardRecorderNode initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Timeout: {self.timeout}s (genie_sim heartbeat)")
        logger.info(f"Image timeout: {self.image_timeout}s (stop recording & convert when exceeded)")
        if self.final_output_dir:
            logger.info(f"Final output directory: {self.final_output_dir}")
        if self.resize_width or self.resize_height:
            logger.info(f"Image resize: {self.resize_width or 'orig'}x{self.resize_height or 'orig'}")

    def sub_task_name_callback(self, msg):
        """Callback for /record/sub_task_name topic"""
        try:
            if self.should_stop:
                return
            with self.sub_task_name_lock:
                if msg and hasattr(msg, "data"):
                    old_received = self.sub_task_name_received
                    self.sub_task_name = msg.data
                    self.sub_task_name_received = True
                    if not old_received and self.subscribers and not self.is_recording:
                        logger.info("Sub_task_name received, will start recording on next image message...")
        except Exception as e:
            logger.warning(f"Error in sub_task_name_callback: {e}")

    def episode_done_callback(self, msg):
        """Callback for /record/episode_done: stop current bag.

        NOTE: current_episode_idx is NOT updated here. It is assigned by
        start_recording() so that the bag file name reflects the episode
        that was actually recorded, not a future one.
        """
        try:
            if self.should_stop:
                return
            idx = int(msg.data)
            logger.info(f"Episode {idx} done, stopping recording...")
            # episode_done only stops; it does NOT set the episode index here
            # (that is done in start_recording when a new bag is actually opened).
            self.stop_recording(episode_idx=idx)
        except ValueError:
            logger.warning(f"Invalid episode index in /record/episode_done: {msg.data}")
        except Exception as e:
            logger.error(f"Error in episode_done_callback: {e}")

    def discover_topics(self):
        # Cache topic list for this timer tick to avoid multiple get_topic_names_and_types() calls
        discovered = self.get_topic_names_and_types()

        for topic_name, topic_types in discovered:
            # Subscribe to /record/* CompressedImage topics for rosbag recording
            if topic_name.startswith("/record/") and "sensor_msgs/msg/CompressedImage" in topic_types:
                if topic_name not in self.subscribers:
                    self.create_subscription_for_topic(topic_name)

            # Subscribe to /genie_sim/* Image topics as heartbeat (no forwarding —
            # pi_node.py already publishes CompressedImage to /record/*).
            # Used only for timeout detection.
            if topic_name.startswith("/genie_sim/") and "sensor_msgs/msg/Image" in topic_types:
                if topic_name not in self.genie_sim_subscribers:
                    sub = self.create_subscription(
                        Image,
                        topic_name,
                        self._genie_sim_heartbeat_callback,
                        10,
                    )
                    self.genie_sim_subscribers[topic_name] = sub
                    logger.info(f"Created heartbeat subscription (no forwarding): {topic_name}")

    def _genie_sim_heartbeat_callback(self, msg):
        """Heartbeat only — update timestamp, no forwarding."""
        try:
            with self.message_lock:
                self.last_genie_sim_message_time = time.time()
        except Exception:
            pass

    def create_subscription_for_topic(self, topic_name):
        subscription = self.create_subscription(
            CompressedImage,
            topic_name,
            lambda msg, tn=topic_name: self.image_callback(msg, tn),
            10,
        )
        self.subscribers[topic_name] = subscription
        logger.info(f"Created subscription: {topic_name}")

    def image_callback(self, msg, topic_name):
        try:
            with self.message_lock:
                self.last_image_message_time = time.time()
            if not self.is_recording:
                with self.start_lock:
                    # Double-check inside lock to avoid a second image arriving
                    # after we set is_recording but before the other thread
                    # entered the outer if-not-is_recording branch.
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

            # Wait for sub_task_name to be received
            with self.sub_task_name_lock:
                if not self.sub_task_name_received or not self.sub_task_name:
                    logger.info("Waiting for sub_task_name message before starting recording...")
                    return
                sub_task_name = self.sub_task_name

            logger.info(f"Starting recording with sub_task_name: '{sub_task_name}'")

            # Create output path: output/recording_data/{sub_task_name}/recording_{episode_idx}_{timestamp}_{uuid}/
            # Microsecond timestamp + uuid guarantee no name collision even if two
            # image_callback invocations race through the start_lock at the same second.
            import uuid

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            recording_data_dir = os.path.join(self.output_dir, "recording_data", sub_task_name)
            os.makedirs(recording_data_dir, exist_ok=True)
            with self._episode_counter_lock:
                # If a bag for this episode is already done, advance to a fresh index so
                # two bags for the same logical episode never share a directory name.
                if self._active_episode_idx is not None and self._active_episode_idx >= self.current_episode_idx:
                    ep_idx = self._active_episode_idx + 1
                    self.current_episode_idx = ep_idx
                else:
                    ep_idx = self.current_episode_idx
            short_uuid = uuid.uuid4().hex[:8]
            bag_name = f"recording_{ep_idx:04d}_{timestamp}_{short_uuid}"
            bag_path = os.path.join(recording_data_dir, bag_name)

            topics = " ".join(self.subscribers.keys())
            ros_distro = os.getenv("ROS_DISTRO", "jazzy")
            command_str = f"""
            unset PYTHONPATH
            unset LD_LIBRARY_PATH
            source /opt/ros/{ros_distro}/setup.bash
            ros2 bag record -o {bag_path} {topics}
            """

            self.record_process = subprocess.Popen(
                command_str,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )

            self.is_recording = True
            self.bag_path = bag_path
            self._active_episode_idx = ep_idx

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

    def check_image_timeout(self):
        """Stop recording and trigger video conversion when image messages stop arriving."""
        if self.should_stop or self.image_timeout <= 0:
            return
        if not self.is_recording:
            return

        with self.message_lock:
            elapsed = time.time() - self.last_image_message_time

        if elapsed > self.image_timeout:
            logger.info(
                f"No image messages received for {elapsed:.1f}s (> {self.image_timeout}s timeout), "
                "stopping recording and starting video conversion..."
            )
            self.stop_recording()

    @staticmethod
    def _wait_for_bag_ready(bag_path, timeout=20, poll_interval=0.5):
        """Poll until the rosbag2 bag is fully written and readable.

        Uses a two-stage check:
        1. Basic file-system checks (metadata.yaml + data files exist and non-zero).
        2. rosbag2_py open attempt — if SequentialReader can open the bag, it is truly ready.
        Raises RuntimeError if the bag is still unreadable after `timeout`.
        """
        deadline = time.time() + timeout

        def bag_data_files_exist(bp):
            data_files = list(Path(bp).glob("*.mcap")) or list(Path(bp).glob("*.db3"))
            return data_files and all(f.stat().st_size > 0 for f in data_files)

        def metadata_yaml_is_valid(bp):
            """Return True only if metadata.yaml parses as a valid rosbag2 metadata doc."""
            p = Path(bp) / "metadata.yaml"
            if not (p.exists() and p.stat().st_size > 0):
                return False
            try:
                data = yaml.safe_load(p.read_text())
                # Must have the required rosbag2 top-level keys
                return isinstance(data, dict) and "version" in data
            except Exception:
                return False

        while time.time() < deadline:
            if bag_data_files_exist(bag_path) and metadata_yaml_is_valid(bag_path):
                # Stage-1 passed — try rosbag2_py open as stage-2 confirmation
                try:
                    reader = rosbag2_py.SequentialReader()
                    storage_opts = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="")
                    reader.open(storage_opts, rosbag2_py.ConverterOptions("", ""))
                    reader.close()
                    return  # truly readable
                except Exception:
                    pass  # not yet openable, keep polling
            time.sleep(poll_interval)

        # Timeout — one final attempt; raise so caller knows extraction may fail
        try:
            reader = rosbag2_py.SequentialReader()
            storage_opts = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="")
            reader.open(storage_opts, rosbag2_py.ConverterOptions("", ""))
            reader.close()
        except Exception as e:
            logger.error(f"Bag is not readable after {timeout}s: {e}")
            raise RuntimeError(f"Bag not ready for reading after {timeout}s: {bag_path}") from e

    def _task_worker(self):
        """Dedicated thread: waits for bag to be ready, then extracts & converts.

        extraction_in_progress is managed HERE (not in stop_recording) so that
        multiple episodes can safely queue up without being incorrectly skipped
        while the previous episode is still being extracted.
        """
        while True:
            bag_path = self.task_queue.get()
            if bag_path is None:
                break

            # Set flag BEFORE starting work (not in stop_recording) so that
            # concurrent stop_recording() calls see it as False until we
            # actually begin extracting.
            with self.extraction_lock:
                self.extraction_in_progress = True

            try:
                self._wait_for_bag_ready(bag_path, timeout=20)
                self.extract_and_convert(bag_path)
                with self._processed_bags_lock:
                    self._processed_bags.append(bag_path)
            except Exception as e:
                logger.error(f"Background extraction error for {bag_path}: {e}")
                import traceback

                logger.error(traceback.format_exc())
            finally:
                with self.extraction_lock:
                    self.extraction_in_progress = False
                self.task_queue.task_done()

    def stop_recording(self, episode_idx=None):
        if not self.is_recording:
            return

        try:
            os.killpg(os.getpgid(self.record_process.pid), signal.SIGINT)
            try:
                self.record_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"Record process {self.record_process.pid} did not stop in time, sending SIGKILL")
                os.killpg(os.getpgid(self.record_process.pid), signal.SIGKILL)
                self.record_process.wait()
            logger.info("Rosbag recording stopped, flushing filesystem...")
            time.sleep(1.5)  # give OS time to flush rosbag2 write buffers

            self.is_recording = False

            # Record the episode index of the bag that just finished.
            # Advance AFTER the bag is closed so this bag's name is stable.
            if episode_idx is not None:
                with self._episode_counter_lock:
                    self.current_episode_idx = episode_idx
                    self._active_episode_idx = episode_idx

            bag_path_copy = self.bag_path
            self.task_queue.put(bag_path_copy)
            logger.info(f"Enqueued background extraction for {bag_path_copy}")

            ack_idx = episode_idx if episode_idx is not None else -1
            ack_msg = String()
            ack_msg.data = str(ack_idx)
            self.episode_ack_publisher.publish(ack_msg)
            logger.info(f"Published episode_ack for episode {ack_idx}")

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self.is_recording = False

    def extract_and_convert(self, bag_path=None):
        logger.info("Starting extraction and video conversion...")

        try:
            # Get sub_task_name
            with self.sub_task_name_lock:
                sub_task_name = self.sub_task_name if self.sub_task_name else "unknown"

            # If no explicit bag_path, find the latest recording directory
            if bag_path is None:
                recording_data_dir = os.path.join(self.output_dir, "recording_data", sub_task_name)
                if not os.path.exists(recording_data_dir):
                    logger.error(f"Recording data directory not found: {recording_data_dir}")
                    return
                bag_dirs = [
                    d for d in Path(recording_data_dir).iterdir() if d.is_dir() and d.name.startswith("recording_")
                ]
                if not bag_dirs:
                    logger.error(f"No recording directory found in {recording_data_dir}")
                    return
                bag_dir = max(bag_dirs, key=lambda x: x.stat().st_mtime)
                bag_path = str(bag_dir)

            logger.info(f"Processing bag: {bag_path}")

            # bag_dir is always bag_path; keep for compatibility with move logic
            bag_dir = bag_path

            self.current_recording_dir = bag_path
            self.extract_images_from_bag(bag_path)

            self.convert_images_to_video()

            self._cleanup_nondata_files(bag_path)

            if self.final_output_dir:
                try:
                    os.makedirs(self.final_output_dir, exist_ok=True)

                    # Move the entire recording directory to final output
                    recording_name = os.path.basename(bag_dir)
                    dst_recording = os.path.join(self.final_output_dir, recording_name)
                    if os.path.exists(dst_recording):
                        shutil.rmtree(dst_recording)
                    shutil.move(str(bag_dir), dst_recording)
                    logger.info(f"Moved recording directory to {dst_recording}")

                except Exception as e:
                    logger.error(f"Error moving recording to final output dir: {e}")

            logger.info("Extraction and conversion completed successfully")

        except Exception as e:
            logger.error(f"Error in extraction and conversion: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _cleanup_nondata_files(self, bag_path):
        """Delete all files/directories in bag_path except the video/ subdirectory."""
        video_dir = Path(bag_path) / "video"
        to_delete = []
        for entry in Path(bag_path).iterdir():
            if entry == video_dir:
                continue
            to_delete.append(entry)

        for entry in to_delete:
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                logger.info(f"Deleted (video conversion done): {entry}")
            except Exception as e:
                logger.error(f"Error deleting {entry}: {e}")

    def concat_all_recordings(self):
        """Concatenate videos from all processed bag directories.

        Called automatically during cleanup(), but can also be triggered manually.
        """
        with self._processed_bags_lock:
            bag_paths = list(self._processed_bags)
        if not bag_paths:
            logger.info("No processed bags to concatenate")
            return
        self._concat_bag_videos(bag_paths)

    def _concat_bag_videos(self, bag_paths: list[str]):
        """Concatenate same-camera videos across bag_paths in order.

        Videos are written to the parent directory of the index folder
        (i.e. the common parent of all recording directories), keeping the
        original filenames (no "_concat" suffix).
        """
        if not bag_paths:
            return

        # All bags share the same parent directory
        output_dir = Path(bag_paths[0]).parent

        # Gather all video paths per camera, in bag order
        camera_videos: dict[str, list[str]] = {}
        for bag_path in bag_paths:
            video_dir = Path(bag_path) / "video"
            if not video_dir.is_dir():
                continue
            for vf in sorted(video_dir.glob("*.webm")):
                camera_name = vf.stem
                camera_videos.setdefault(camera_name, []).append(str(vf))

        for camera_name, video_list in camera_videos.items():
            if len(video_list) < 2:
                logger.info(f"Skipping concat for '{camera_name}': fewer than 2 segments")
                continue

            concat_output = output_dir / f"{camera_name}.webm"

            # Build ffmpeg concat file
            concat_file = output_dir / f"{camera_name}_concat_list.txt"
            concat_file.write_text("\n".join(f"file '{v}'" for v in video_list))

            try:
                logger.info(f"Concatenating {len(video_list)} segments for '{camera_name}' -> {concat_output}")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_file),
                        "-c",
                        "copy",
                        str(concat_output),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                # Remove per-segment videos after successful concat
                for v in video_list:
                    Path(v).unlink()
                concat_file.unlink()
                logger.info(f"Concat done: {concat_output}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Concat failed for '{camera_name}': {e.stderr.decode() if e.stderr else e}")
            except Exception as e:
                logger.error(f"Concat error for '{camera_name}': {e}")

        # Remove empty recording directories after concat
        for bag_path in bag_paths:
            try:
                shutil.rmtree(bag_path)
                logger.info(f"Removed empty directory: {bag_path}")
            except Exception as e:
                logger.warning(f"Could not remove {bag_path}: {e}")

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

        unique_dirs = set(task[1] for task in io_tasks)
        for d in unique_dirs:
            os.makedirs(d, exist_ok=True)

        do_resize = self.resize_width is not None or self.resize_height is not None
        resize_w = self.resize_width
        resize_h = self.resize_height

        def write_one(task):
            cv_image, cam_dir, final_path, camera_name, frame_idx = task
            if do_resize:
                cv_image = cv2.resize(cv_image, (resize_w or cv_image.shape[1], resize_h or cv_image.shape[0]))
            success, encoded = cv2.imencode(".jpg", cv_image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not success:
                raise RuntimeError(f"cv2.imencode failed for {final_path}")
            with open(final_path, "wb") as f:
                f.write(encoded.tobytes())

        with ThreadPoolExecutor(max_workers=self.max_io_workers) as executor:
            futures = {executor.submit(write_one, task): task for task in io_tasks}
            completed = 0
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    fut.result()
                    completed += 1
                    if completed % PROGRESS_LOG_INTERVAL == 0:
                        logger.info(f"Finished {completed}/{len(io_tasks)} aligned image writes")
                except Exception as e:
                    logger.error(f"Error writing {task[2]}: {e}")

    def extract_images_from_bag(self, bag_path):
        logger.info(f"Extracting images from {bag_path}")
        self.images_dir = os.path.join(bag_path, "camera")
        self.topic_images = {}  # {camera_name: [path, ...]}
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

            # Pre-create all per-camera output dirs
            for camera_name in sorted_frames.keys():
                cam_dir = os.path.join(self.images_dir, camera_name)
                os.makedirs(cam_dir, exist_ok=True)

            camera_indices = {name: 0 for name in sorted_frames.keys()}
            io_tasks = []  # (cv_image, cam_dir, final_path, camera_name, frame_idx)

            for frame_idx, ref_ts in enumerate(time_stamps):
                for camera_name, frames in sorted_frames.items():
                    idx = camera_indices[camera_name]
                    best_idx, compressed_msg = self._find_closest_frame(frames, idx, ref_ts)
                    camera_indices[camera_name] = best_idx

                    try:
                        cv_image = self._decode_compressed_image(compressed_msg)
                        final_path = os.path.join(self.images_dir, camera_name, f"frame_{frame_idx:06d}.jpg")
                        self.topic_images.setdefault(camera_name, []).append(final_path)
                        cam_dir = os.path.join(self.images_dir, camera_name)
                        io_tasks.append((cv_image, cam_dir, final_path, camera_name, frame_idx))
                    except Exception as e:
                        logger.error(f"Error preparing aligned frame {frame_idx} for {camera_name}: {e}")

            if io_tasks:
                self._write_images_parallel(io_tasks)

            for camera_name, images in self.topic_images.items():
                logger.info(f"Extracted {len(images)} temporally aligned frames for camera: {camera_name}")

    @staticmethod
    def _extract_frame_idx(path):
        """Extract frame_idx from path camera/{camera_name}/frame_{frame_idx}.jpg"""
        try:
            return int(Path(path).stem.split("_")[1])
        except (ValueError, IndexError):
            return 0

    def _prepare_video_frames(self, camera_name, image_paths):
        topic_dir = os.path.join(self.images_dir, camera_name)

        sorted_paths = sorted(image_paths, key=self._extract_frame_idx)

        frame_indices = [self._extract_frame_idx(p) for p in sorted_paths]
        expected_indices = list(range(len(sorted_paths)))
        if frame_indices != expected_indices:
            logger.warning(
                f"Frame indices are not continuous for {camera_name}. "
                f"First 10 expected: {expected_indices[:10]}, First 10 got: {frame_indices[:10]}"
            )

        # Images are already at camera/{camera_name}/frame_{frame_idx:06d}.jpg,
        # so just verify and return the pattern directly.
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

        # Determine the recording directory (parent of images_dir)
        if hasattr(self, "current_recording_dir"):
            recording_dir = self.current_recording_dir
        elif hasattr(self, "images_dir"):
            recording_dir = os.path.dirname(self.images_dir)
        else:
            recording_dir = self.output_dir

        # Create video directory in the recording directory
        video_dir = os.path.join(recording_dir, "video")
        os.makedirs(video_dir, exist_ok=True)

        for camera_name, image_paths in self.topic_images.items():
            if len(image_paths) == 0:
                logger.warning(f"No images found for camera {camera_name}, skipping video conversion")
                continue

            try:
                # Prepare sequentially numbered frame files
                input_pattern = self._prepare_video_frames(camera_name, image_paths)

                # Output video path - save in video subdirectory of recording directory
                output_video = os.path.join(video_dir, f"{camera_name}.webm")

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

        # Set should_stop flag first to prevent callbacks from processing
        self.should_stop = True

        # Stop recording
        if self.is_recording:
            self.stop_recording()

        # Stop worker thread: send sentinel and drain pending tasks
        try:
            if hasattr(self, "task_queue"):
                pending = []
                while True:
                    try:
                        item = self.task_queue.get_nowait()
                        if item is None:
                            break
                        pending.append(item)
                    except:
                        break
                if pending:
                    logger.info(f"Re-enqueuing {len(pending)} pending extraction(s) before shutdown...")
                    for item in pending:
                        self.task_queue.put(item)
                self.task_queue.put(None)
                if hasattr(self, "worker_thread"):
                    self.worker_thread.join(timeout=60)
                    if self.worker_thread.is_alive():
                        logger.warning("Extraction worker thread did not exit gracefully")

                # Concatenate all processed bag videos now that extraction is done
                self.concat_all_recordings()
        except Exception as e:
            logger.warning(f"Error stopping extraction worker: {e}")
        try:
            if hasattr(self, "topic_discovery_timer"):
                self.topic_discovery_timer.cancel()
            if hasattr(self, "timeout_check_timer"):
                self.timeout_check_timer.cancel()
            if hasattr(self, "image_timeout_check_timer"):
                self.image_timeout_check_timer.cancel()
        except Exception as e:
            logger.warning(f"Error canceling timers: {e}")

        # Destroy dynamically created subscribers
        try:
            for sub in self.subscribers.values():
                sub.destroy()
            for sub in self.genie_sim_subscribers.values():
                sub.destroy()
        except Exception as e:
            logger.warning(f"Error destroying dynamic subscriptions: {e}")

        # Destroy fixed subscriptions
        try:
            if hasattr(self, "sub_task_name_subscription"):
                self.sub_task_name_subscription.destroy()
            if hasattr(self, "episode_done_subscription"):
                self.episode_done_subscription.destroy()
        except Exception as e:
            logger.warning(f"Error destroying fixed subscriptions: {e}")

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
        default=30.0,
        help="Timeout (seconds), automatically exit if no /genie_sim/* messages received for this duration (default: 10.0)",
    )
    parser.add_argument(
        "--image_timeout",
        type=float,
        default=30.0,
        help="If no image messages received for this duration while recording, automatically stop recording "
        "and start video conversion (default: 3.0). Set to 0 to disable.",
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
        help="Final output directory: if provided, will move the recording directory to this path after extraction",
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=None,
        help="Resize images to this width (px) before saving. If not set, keep original width.",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=None,
        help="Resize images to this height (px) before saving. If not set, keep original height. "
        "Aspect ratio is NOT preserved — use both width and height to get exact size.",
    )

    cli_args = parser.parse_args(args=args)

    # Configure parameters
    output_dir = cli_args.output_dir
    timeout = cli_args.timeout
    image_timeout = cli_args.image_timeout
    jpeg_quality = cli_args.jpeg_quality
    final_output_dir = cli_args.final_output_dir

    rclpy.init(args=None)

    logger.info(f"Starting auto-record-extractor")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Timeout: {timeout}s (genie_sim heartbeat)")
    logger.info(f"Image timeout: {image_timeout}s")
    logger.info(f"JPEG quality: {jpeg_quality}")
    if final_output_dir:
        logger.info(f"Final output directory: {final_output_dir}")
    if cli_args.resize_width or cli_args.resize_height:
        logger.info(f"Image resize: {cli_args.resize_width or 'orig'}x{cli_args.resize_height or 'orig'}")

    node = ImageForwardRecorderNode(
        output_dir=output_dir,
        timeout=timeout,
        image_timeout=image_timeout,
        jpeg_quality=jpeg_quality,
        final_output_dir=final_output_dir,
        resize_width=cli_args.resize_width,
        resize_height=cli_args.resize_height,
    )

    try:
        while rclpy.ok() and not node.should_stop:
            try:
                rclpy.spin_once(node, timeout_sec=0.1)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # Log but continue to allow cleanup
                logger.warning(f"Error in spin_once: {e}")
                if node.should_stop:
                    break

        logger.info("Node stopped, performing final cleanup...")
        node.cleanup()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        node.should_stop = True
        node.cleanup()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        import traceback

        logger.error(traceback.format_exc())
        node.should_stop = True
        node.cleanup()
    finally:
        try:
            node.destroy_node()
        except Exception as e:
            logger.warning(f"Error destroying node: {e}")
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            logger.warning(f"rclpy.shutdown() failed or was already called: {e}")
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
