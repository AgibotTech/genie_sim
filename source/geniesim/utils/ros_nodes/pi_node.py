# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .base_nodes import *
from std_msgs.msg import String, Header, Bool
import cv2
from scipy.spatial.transform import Rotation as R
import os
import threading


class PIROSNode(SimNode):
    def __init__(self, robot_name="G1_120s", node_name="pi_node", device="cuda:0"):
        super().__init__(robot_name=robot_name, node_name=node_name)
        # fmt: off
        # sub
        self.sub_head_g1 = self.create_subscription(Image, "/genie_sim/head_camera_rgb", lambda msg: self.callback_image(msg, "head"), 1)
        self.sub_head_g2 = self.create_subscription(Image, "/genie_sim/head_front_camera_rgb", lambda msg: self.callback_image(msg, "head"), 1)
        self.sub_l = self.create_subscription(Image, "/genie_sim/left_camera_rgb", lambda msg: self.callback_image(msg, "left_hand"), 1)
        self.sub_r = self.create_subscription(Image, "/genie_sim/right_camera_rgb", lambda msg: self.callback_image(msg, "right_hand"), 1)
        self.sub_world = self.create_subscription(Image, "/genie_sim/camera_rgb", lambda msg: self.callback_image(msg, "world"), 1)
        self.sub_head_depth = self.create_subscription(Image, "/genie_sim/head_camera_depth", lambda msg: self.callback_image(msg, "head_depth"), 1)
        self.sub_l_depth = self.create_subscription(Image, "/genie_sim/left_camera_depth", lambda msg: self.callback_image(msg, "left_hand_depth"), 1)
        self.sub_r_depth = self.create_subscription(Image, "/genie_sim/right_camera_depth", lambda msg: self.callback_image(msg, "right_hand_depth"), 1)

        # pub
        self.pub_head_g1 = self.create_publisher(CompressedImage, "/record/head_camera_rgb", 1)
        self.pub_head_g2 = self.create_publisher(CompressedImage, "/record/head_front_camera_rgb", 1)

        self.pub_l = self.create_publisher(CompressedImage, "/record/left_camera_rgb", 1)
        self.pub_r = self.create_publisher(CompressedImage, "/record/right_camera_rgb", 1)
        self.pub_world = self.create_publisher(CompressedImage, "/record/camera_rgb", 1)


        #pub
        self.pub_static_info = self.create_publisher(
            String,
            "/record/static_info",
            QOS_PROFILE_LATEST,
        )
        self.pub_dynamic_info = self.create_publisher(
            String,
            "/genie_sim/dynamic_info",
            QOS_PROFILE_LATEST,
        )
        # fmt: on

        # Subscribe to /sim/instruction, /sim/reset, /sim/infer_start and /sim/shuffle
        self.sub_instruction = self.create_subscription(String, "/sim/instruction", self.callback_instruction, 1)
        self.sub_reset = self.create_subscription(Bool, "/sim/reset", self.callback_reset, 1)
        self.sub_infer_start = self.create_subscription(Bool, "/sim/infer_start", self.callback_infer_start, 1)
        self.sub_shuffle = self.create_subscription(Bool, "/sim/shuffle", self.callback_shuffle, 1)

        # Store received messages
        self.instruction_lock = threading.Lock()
        self.instruction_msg = ""
        self.reset_lock = threading.Lock()
        self.reset_msg = False
        self.infer_start_lock = threading.Lock()
        self.infer_start_msg = False
        self.shuffle_lock = threading.Lock()
        self.shuffle_msg = False

        self.img_buffer = {
            "head": None,
            "left_hand": None,
            "right_hand": None,
            "world": None,
            "top": None,
            "head_depth": None,
            "left_hand_depth": None,
            "right_hand_depth": None,
        }
        self.infer_img_keys = ["head", "left_hand", "right_hand"]

    def create_header(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"
        return header

    def pub_static_info_msg(self, msg: str):
        msg_out = String()
        msg_out.data = msg
        self.pub_static_info.publish(msg_out)

    def pub_dynamic_info_msg(self, msg: str):
        msg_out = String()
        msg_out.data = msg
        self.pub_dynamic_info.publish(msg_out)

    def callback_image(self, msg, type):
        if type not in self.img_buffer:
            raise Exception("Invalid image type")
        self.img_buffer[type] = msg

    def image_to_compressed(self, img_msg):
        if img_msg is None:
            return None

        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            success, compressed_data = cv2.imencode(".jpg", cv_image)
            if not success:
                self.get_logger().error("Failed to encode image to JPEG")
                return None

            compressed_msg = CompressedImage()
            compressed_msg.header = img_msg.header
            compressed_msg.format = "jpg"
            compressed_msg.data = compressed_data.tobytes()

            return compressed_msg

        except Exception as e:
            self.get_logger().error(f"Error converting image to compressed: {e}")
            return None

    def publish_image(self):
        if self.img_buffer["head"] is not None:
            compressed_head = self.image_to_compressed(self.img_buffer["head"])
            if compressed_head is not None:
                self.pub_head_g1.publish(compressed_head)
                self.pub_head_g2.publish(compressed_head)

        if self.img_buffer["left_hand"] is not None:
            compressed_l = self.image_to_compressed(self.img_buffer["left_hand"])
            if compressed_l is not None:
                self.pub_l.publish(compressed_l)

        if self.img_buffer["right_hand"] is not None:
            compressed_r = self.image_to_compressed(self.img_buffer["right_hand"])
            if compressed_r is not None:
                self.pub_r.publish(compressed_r)

        if self.img_buffer["world"] is not None:
            compressed_world = self.image_to_compressed(self.img_buffer["world"])
            if compressed_world is not None:
                self.pub_world.publish(compressed_world)

    def get_image(self, img):
        img_cpu = self.img2np(img)
        img_cpu = img_cpu.astype(np.uint8)
        return img_cpu

    def get_observation_image(self):
        ret = {}
        if all(self.img_buffer[v] is not None for v in self.infer_img_keys):
            print("image ready")
        else:
            print("image empty")
            return ret

        for k in self.infer_img_keys:
            ret[k] = self.get_image(self.img_buffer[k])
        return ret

    def callback_instruction(self, msg):
        """Callback function: receive /sim/instruction message"""
        with self.instruction_lock:
            self.instruction_msg = msg.data

    def callback_reset(self, msg):
        """Callback function: receive /sim/reset message"""
        with self.reset_lock:
            self.reset_msg = msg.data

    def callback_infer_start(self, msg):
        """Callback function: receive /sim/infer_start message"""
        with self.infer_start_lock:
            self.infer_start_msg = msg.data

    def callback_shuffle(self, msg):
        """Callback function: receive /sim/shuffle message"""
        with self.shuffle_lock:
            self.shuffle_msg = msg.data

    def get_instruction(self):
        """Get the latest instruction message"""
        with self.instruction_lock:
            return self.instruction_msg

    def get_reset(self):
        """Get the latest reset message"""
        with self.reset_lock:
            return self.reset_msg

    def get_infer_start(self):
        """Get the latest infer_start message"""
        with self.infer_start_lock:
            return self.infer_start_msg

    def get_shuffle(self):
        """Get the latest shuffle message"""
        with self.shuffle_lock:
            return self.shuffle_msg
