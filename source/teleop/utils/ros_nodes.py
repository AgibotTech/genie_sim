# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from collections import deque
import threading, time
import os, sys, math, random
from std_msgs.msg import Bool
from .name_utils import *
from datetime import datetime
from std_msgs.msg import Header
from geniesim_msg.msg import GeniesimReactiveControl, GeniesimRetargetGroup
from config.robot_interface import RobotType, robot_desc_map
import scipy.spatial.transform as tf

from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Pose

QOS_PROFILE_LATEST = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

QOS_PROFILE_VOLATILE = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)


def distance_error(xyz1, xyz2):
    return math.sqrt((xyz1[0] - xyz2[0]) ** 2 + (xyz1[1] - xyz2[1]) ** 2 + (xyz1[2] - xyz2[2]) ** 2)


def print_rotation_info(quat, label=""):
    """Print rotation details (quaternion and euler angles)."""
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler("xyz", degrees=True)
    print(f"{label}:")
    print(f"  ====> quaternion: {quat}")
    print(f"  ====> euler (deg): {euler_angles}")
    print()


class SimNode(Node):
    def __init__(self, robot_name="G2", node_name="sim_ros_node"):
        super().__init__(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.init_pose = None
        self.robot_name = robot_name
        self.config_robot()
        self.bridge = CvBridge()
        self.pub_joint_command = self.create_publisher(
            JointState,
            "/joint_command",
            QOS_PROFILE_VOLATILE,
        )
        self.sub_js = self.create_subscription(
            JointState,
            "/joint_states",
            self.callback_joint_state,
            QOS_PROFILE_VOLATILE,
        )
        self.publisher_playback = self.create_publisher(Bool, "/sim/playback_flag", 1)
        self.publisher_recording = self.create_publisher(Bool, "/sim/is_recording", 1)
        self.publisher_mc = self.create_publisher(GeniesimReactiveControl, "/wbc/retarget", 10)
        self.publisher_mc_debug = self.create_publisher(Pose, "/debug/retarget", 10)
        self.lock_js = threading.Lock()
        self.lock_tf = threading.Lock()
        self.js_msg = JointState()
        self.tf_msg = TFMessage()
        self.mc_tf_init = False
        # init for mc
        self.robot_type = RobotType.G2
        robot_desc = robot_desc_map[self.robot_type]
        self.parts = [
            {
                "id": "left",
                "frame_id": "base_link",
                "control_type": 0,
                "group_id": GeniesimRetargetGroup.GROUP_LEFT_ARM,
                "reference_frame_name": robot_desc["ref_frames"][0],
                "target_frame_name": robot_desc["ee_frames"][0],
                "init": False,
                "pose": Pose(),
            },
            {
                "id": "right",
                "frame_id": "base_link",
                "control_type": 0,
                "group_id": GeniesimRetargetGroup.GROUP_RIGHT_ARM,
                "reference_frame_name": robot_desc["ref_frames"][1],
                "target_frame_name": robot_desc["ee_frames"][1],
                "init": False,
                "pose": Pose(),
            },
            {
                "id": "arm_base",
                "frame_id": "base_link",
                "control_type": 1,
                "group_id": GeniesimRetargetGroup.GROUP_WAIST,
                "reference_frame_name": "",
                "target_frame_name": "arm_base_link",
                "init": False,
                "pose": Pose(),
            },
        ]
        self.base_frame = "base_link"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.loop_rate = self.create_rate(30.0)
        self.dubug_num = 0

    def config_robot(self):
        if "G1" in self.robot_name:
            self.robot_id = "G1"
        elif "G2" in self.robot_name:
            self.robot_id = "G2"
        else:
            raise Exception(f"Invalid robot name {self.robot_name}")

        if self.robot_id == "G1":
            self.joint_names = G1_JOINT_NAMES
        elif self.robot_id == "G2":
            self.joint_names = G2_JOINT_NAMES
        self.config_eef()

    def config_eef(self):
        if "omnipicker" in self.robot_name:
            self.joint_names += OMNIPICKER_AJ_NAMES
        else:
            raise Exception(f"Invalid eef {self.robot_name}")

    def callback_joint_state(self, msg):
        with self.lock_js:
            self.js_msg = msg

    def get_joint_state(self):
        with self.lock_js:
            return self.js_msg

    def set_joint_state(self, name, position):
        cmd_msg = JointState()
        cmd_msg.position = position
        cmd_msg.name = name
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_joint_command.publish(cmd_msg)

    def set_joint_position_and_velocity(self, name, position, velocity):
        cmd_msg = JointState()
        cmd_msg.name = name
        cmd_msg.position = position
        cmd_msg.velocity = velocity
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_joint_command.publish(cmd_msg)

    def pub_playback(self, val):
        msg = Bool()
        msg.data = val
        self.publisher_playback.publish(msg)

    def pub_recording(self, val):
        msg = Bool()
        msg.data = val
        self.publisher_recording.publish(msg)

    def pub_debug_pos(self, xyz, xyzw):
        pose = Pose()
        pose.position.x = xyz[0]
        pose.position.y = xyz[1]
        pose.position.z = xyz[2]
        pose.orientation.x = xyzw[0]
        pose.orientation.y = xyzw[1]
        pose.orientation.z = xyzw[2]
        pose.orientation.w = xyzw[3]
        self.publisher_mc_debug.publish(pose)

    def pub_mc(self, poses, body_lift_pose=None):
        retarget_msg = GeniesimReactiveControl()
        retarget_msg.header = self.create_header()
        retarget_msg.retarget_groups = []
        for part in self.parts:
            if (
                (part.get("id") == "left" and poses[0] == None)
                or (part.get("id") == "right" and poses[1] == None)
                or (part.get("id") == "body" and body_lift_pose == None)
            ):
                continue
            retarget_group = GeniesimRetargetGroup()
            retarget_group.group_id = part["group_id"]
            retarget_group.control_type = part["control_type"]
            if part["control_type"] == 0:
                retarget_group.frame_id = "arm_base_link"
                ee_names = part.get("target_frame_name", "")
                retarget_group.target_frame_names = [ee_names]
                retarget_group.target_frame_poses = [poses[0] if part.get("id") == "left" else poses[1]]
            elif part["control_type"] == 1:
                continue
            elif part["control_type"] == 3:
                if isinstance(body_lift_pose, Pose):
                    retarget_group.control_type = 0
                    retarget_group.frame_id = "base_link"
                    ee_names = part.get("target_frame_name", "")
                    retarget_group.target_frame_names = [ee_names]
                    retarget_group.target_frame_poses = [body_lift_pose]
                else:
                    retarget_group.target_joint_positions = body_lift_pose
            retarget_msg.retarget_groups.append(retarget_group)

        right_tool = GeniesimRetargetGroup()
        right_tool.group_id = GeniesimRetargetGroup.GROUP_RIGHT_TOOL
        right_tool.control_type = 3
        right_tool.target_joint_positions = [0.0]
        retarget_msg.retarget_groups.append(right_tool)
        print(f"pub_mc: {retarget_msg}")
        self.publisher_mc.publish(retarget_msg)

    def pub_waist_pose(self, init_waist_angle, waist_yaw, waist_pitch):
        retarget_msg = GeniesimReactiveControl()
        retarget_msg.header = self.create_header()
        retarget_msg.retarget_groups = []

        waist_lift_group = GeniesimRetargetGroup()
        waist_lift_group.group_id = GeniesimRetargetGroup.GROUP_WAIST_LIFT
        waist_lift_group.control_type = 3  # absolute joint angle
        waist_lift_group.target_joint_positions = [
            init_waist_angle[4],
            init_waist_angle[3],
            waist_pitch,
            init_waist_angle[1],
            waist_yaw,
        ]
        retarget_msg.retarget_groups.append(waist_lift_group)

        self.publisher_mc.publish(retarget_msg)

    def pub_robot_pose(self, body_position, head_position, left_arm_position, right_arm_position):
        if body_position is None and head_position is None and left_arm_position is None and right_arm_position is None:
            return

        retarget_msg = GeniesimReactiveControl()
        retarget_msg.header = self.create_header()
        retarget_msg.retarget_groups = []

        if left_arm_position != None:
            left_arm_group = GeniesimRetargetGroup()
            left_arm_group.group_id = GeniesimRetargetGroup.GROUP_LEFT_ARM
            left_arm_group.control_type = 3  # absolute joint angle
            left_arm_group.target_joint_positions = left_arm_position
            retarget_msg.retarget_groups.append(left_arm_group)

        if right_arm_position != None:
            right_arm_group = GeniesimRetargetGroup()
            right_arm_group.group_id = GeniesimRetargetGroup.GROUP_RIGHT_ARM
            right_arm_group.control_type = 3  # absolute joint angle
            right_arm_group.target_joint_positions = right_arm_position
            retarget_msg.retarget_groups.append(right_arm_group)

        if body_position != None:
            waist_lift_group = GeniesimRetargetGroup()
            waist_lift_group.group_id = GeniesimRetargetGroup.GROUP_WAIST_LIFT
            waist_lift_group.control_type = 3  # absolute joint angle
            waist_lift_group.target_joint_positions = body_position
            retarget_msg.retarget_groups.append(waist_lift_group)

        if head_position != None:
            head_yaw_group = GeniesimRetargetGroup()
            head_yaw_group.group_id = GeniesimRetargetGroup.GROUP_HEAD_YAW
            head_yaw_group.control_type = 3  # absolute joint angle
            head_yaw_group.target_joint_positions = [head_position[0]]
            retarget_msg.retarget_groups.append(head_yaw_group)

            head_roll_group = GeniesimRetargetGroup()
            head_roll_group.group_id = GeniesimRetargetGroup.GROUP_HEAD_ROLL
            head_roll_group.control_type = 3  # absolute joint angle
            head_roll_group.target_joint_positions = [head_position[1]]
            retarget_msg.retarget_groups.append(head_roll_group)

            head_pitch_group = GeniesimRetargetGroup()
            head_pitch_group.group_id = GeniesimRetargetGroup.GROUP_HEAD_PITCH
            head_pitch_group.control_type = 3  # absolute joint angle
            head_pitch_group.target_joint_positions = [head_position[2]]
            retarget_msg.retarget_groups.append(head_pitch_group)

        self.publisher_mc.publish(retarget_msg)

    def initialize(self):
        for i in range(len(self.parts)):
            if self.parts[i]["init"]:
                continue
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.parts[i]["frame_id"],
                    self.parts[i]["target_frame_name"],
                    rclpy.time.Time(),
                )
                self.get_logger().info("Transform received. Initializing marker.")
                self.parts[i]["pose"] = self.transform_to_pose_ros(transform)
                self.parts[i]["init"] = True
                pose = self.parts[i]["pose"]
                R = tf.Rotation.from_quat(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                ).as_matrix()
                print(f"{self.parts[i]}: \n{R}")
            except Exception as e:
                print(f"##############{e}\n")
        if all(self.parts[i]["init"] for i in range(len(self.parts))):
            return True
        else:
            return False

    def update_ee_pos(self, id="left"):
        if id == "left":
            self.parts[0]["init"] = False
            update_flag = False
            while not update_flag:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.parts[0]["frame_id"],
                        self.parts[0]["target_frame_name"],
                        rclpy.time.Time(),
                    )
                    self.parts[0]["pose"] = self.transform_to_pose_ros(transform)
                    self.parts[0]["init"] = True
                    update_flag = True
                    pose = self.parts[0]["pose"]
                    R = [
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]

                    print(f"==========>[pico] left update pos:{R}")
                except Exception as e:
                    print(f"##############{e}\n")
        else:
            self.parts[1]["init"] = False
            update_flag = False
            while not update_flag:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.parts[1]["frame_id"],
                        self.parts[1]["target_frame_name"],
                        rclpy.time.Time(),
                    )
                    self.parts[1]["pose"] = self.transform_to_pose_ros(transform)
                    self.parts[1]["init"] = True
                    update_flag = True
                    pose = self.parts[1]["pose"]
                    R = [
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                    print(f"==========>[pico] right update pos:{R}")
                except Exception as e:
                    print(f"##############{e}\n")

    def update_without_judge(self):
        for i in range(len(self.parts)):
            self.parts[i]["init"] = False
            print(f"self.parts[{i}]['pose']:{self.parts[i]['pose']}")
        update_flag = False
        while not update_flag:
            for i in range(len(self.parts)):
                if self.parts[i]["init"]:
                    continue
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.parts[i]["frame_id"],
                        self.parts[i]["target_frame_name"],
                        rclpy.time.Time(),
                    )
                    self.get_logger().info("Transform received. Initializing marker.")
                    self.parts[i]["pose"] = self.transform_to_pose_ros(transform)
                    self.parts[i]["init"] = True
                    pose = self.parts[i]["pose"]
                    R = tf.Rotation.from_quat(
                        [
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        ]
                    ).as_matrix()
                    print(f"{self.parts[i]}: \n{R} \n{pose}")
                    # self.get_logger().info(f"{self.parts[i]}: \n{R} \n{pose}")
                except Exception as e:
                    print(f"##############{e}\n")
            if all(self.parts[i]["init"] for i in range(len(self.parts))):
                update_flag = True

    def parse_delta_pose(self, delta_xyzxyzw, id="left"):
        delta_pos = delta_xyzxyzw[0:3]
        delta_xyzw = delta_xyzxyzw[3:7]
        ee_pos, ee_xyzw, arm_base_pos, arm_base_xyzw = None, None, None, None
        for part in self.parts:
            if part["id"] == id:
                ee_pos = [
                    part["pose"].position.x,
                    part["pose"].position.y,
                    part["pose"].position.z,
                ]
                ee_xyzw = [
                    part["pose"].orientation.x,
                    part["pose"].orientation.y,
                    part["pose"].orientation.z,
                    part["pose"].orientation.w,
                ]
            elif part["id"] == "arm_base":
                arm_base_pos = [
                    part["pose"].position.x,
                    part["pose"].position.y,
                    part["pose"].position.z,
                ]
                arm_base_xyzw = [
                    part["pose"].orientation.x,
                    part["pose"].orientation.y,
                    part["pose"].orientation.z,
                    part["pose"].orientation.w,
                ]
        if any(v is None for v in (ee_pos, ee_xyzw, arm_base_pos, arm_base_xyzw)):
            return None

        ee_rot = R.from_quat(ee_xyzw)  # end-effector current orientation
        delta_rot = R.from_quat(delta_xyzw)  # orientation delta
        # print(f"id:{id}")
        # print_rotation_info(ee_xyzw, "ee_xyzw")
        # print(f"ee_pos:{ee_pos}")
        delta_in_ee_frame = ee_rot.inv() * delta_rot * ee_rot  # transform delta from base to EE frame
        # delta_in_ee_frame = ee_rot.inv() * delta_rot  # apply delta in EE frame
        new_ee_rot = ee_rot * delta_in_ee_frame
        new_ee_xyzw = new_ee_rot.as_quat()
        # print_rotation_info(new_ee_xyzw, "new_ee_xyzw")
        # self.pub_debug_pos(ee_pos, new_ee_xyzw)
        new_ee_pos = [ee_pos[0] + delta_pos[0], ee_pos[1] + delta_pos[1], ee_pos[2] + delta_pos[2]]

        # base_link → arm_base_link
        T_base_arm = np.eye(4)
        T_base_arm[:3, 3] = arm_base_pos
        T_base_arm[:3, :3] = R.from_quat(arm_base_xyzw).as_matrix()

        T_base_ee_new = np.eye(4)
        T_base_ee_new[:3, 3] = new_ee_pos
        T_base_ee_new[:3, :3] = R.from_quat(new_ee_xyzw).as_matrix()

        T_arm_base = np.linalg.inv(T_base_arm)
        T_arm_ee_new = T_arm_base @ T_base_ee_new

        new_pos = T_arm_ee_new[:3, 3]
        new_quat = R.from_matrix(T_arm_ee_new[:3, :3]).as_quat()  # [x,y,z,w]
        print(f"id:{id} new_pos:{new_pos} new_quat:{new_quat}")

        pose = Pose()
        pose.position.x = new_pos[0]
        pose.position.y = new_pos[1]
        pose.position.z = new_pos[2]
        pose.orientation.x = new_quat[0]
        pose.orientation.y = new_quat[1]
        pose.orientation.z = new_quat[2]
        pose.orientation.w = new_quat[3]
        return pose

    def parse_keyboard_pose(self, keyborad_poses):
        if (keyborad_poses[0] == None) and (keyborad_poses[1] == None):
            return [None, None]
        arm_base_pos, arm_base_xyzw, left_pose, right_pose = None, None, None, None
        for part in self.parts:
            if part["id"] == "arm_base":
                arm_base_pos = [
                    part["pose"].position.x,
                    part["pose"].position.y,
                    part["pose"].position.z,
                ]
                arm_base_xyzw = [
                    part["pose"].orientation.x,
                    part["pose"].orientation.y,
                    part["pose"].orientation.z,
                    part["pose"].orientation.w,
                ]
        if keyborad_poses[0] != None:
            left_xyz = keyborad_poses[0][0:3]
            left_xyzw = keyborad_poses[0][3:7]
            # base_link → ee
            T_base_ee = np.eye(4)
            T_base_ee[:3, 3] = left_xyz
            T_base_ee[:3, :3] = R.from_quat(left_xyzw).as_matrix()
            # base_link → arm_base_link
            T_base_arm = np.eye(4)
            T_base_arm[:3, 3] = arm_base_pos
            T_base_arm[:3, :3] = R.from_quat(arm_base_xyzw).as_matrix()
            T_arm_base = np.linalg.inv(T_base_arm)
            T_arm_ee_new = T_arm_base @ T_base_ee

            new_pos = T_arm_ee_new[:3, 3]
            new_quat = R.from_matrix(T_arm_ee_new[:3, :3]).as_quat()  # [x,y,z,w]
            left_pose = Pose()
            left_pose.position.x = new_pos[0]
            left_pose.position.y = new_pos[1]
            left_pose.position.z = new_pos[2]
            left_pose.orientation.x = new_quat[0]
            left_pose.orientation.y = new_quat[1]
            left_pose.orientation.z = new_quat[2]
            left_pose.orientation.w = new_quat[3]
        if keyborad_poses[1] != None:
            right_xyz = keyborad_poses[1][0:3]
            right_xyzw = keyborad_poses[1][3:7]
            # base_link → ee
            T_base_ee = np.eye(4)
            T_base_ee[:3, 3] = right_xyz
            T_base_ee[:3, :3] = R.from_quat(right_xyzw).as_matrix()
            # base_link → arm_base_link
            T_base_arm = np.eye(4)
            T_base_arm[:3, 3] = arm_base_pos
            T_base_arm[:3, :3] = R.from_quat(arm_base_xyzw).as_matrix()
            T_arm_base = np.linalg.inv(T_base_arm)
            T_arm_ee_new = T_arm_base @ T_base_ee

            new_pos = T_arm_ee_new[:3, 3]
            new_quat = R.from_matrix(T_arm_ee_new[:3, :3]).as_quat()  # [x,y,z,w]
            right_pose = Pose()
            right_pose.position.x = new_pos[0]
            right_pose.position.y = new_pos[1]
            right_pose.position.z = new_pos[2]
            right_pose.orientation.x = new_quat[0]
            right_pose.orientation.y = new_quat[1]
            right_pose.orientation.z = new_quat[2]
            right_pose.orientation.w = new_quat[3]

        return [left_pose, right_pose]

    def get_link_transform_from_base_link(self, link_name: str):
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                link_name,
                rclpy.time.Time(),
            )
            return self.transform_to_pose_ros(transform)
        except:
            return None

    def create_header(self):
        now = datetime.now()
        header = Header()
        header.stamp.sec = int(now.timestamp())
        header.stamp.nanosec = int(now.microsecond * 1000)
        header.frame_id = self.base_frame
        return header

    def transform_to_pose_ros(self, transform):
        pose = Pose()
        pose.position.x = transform.transform.translation.x
        pose.position.y = transform.transform.translation.y
        pose.position.z = transform.transform.translation.z
        pose.orientation.x = transform.transform.rotation.x
        pose.orientation.y = transform.transform.rotation.y
        pose.orientation.z = transform.transform.rotation.z
        pose.orientation.w = transform.transform.rotation.w
        return pose
