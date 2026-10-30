# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import threading
from datetime import datetime

import numpy as np
import rclpy
import scipy.spatial.transform as tf
from geniesim_msg.msg import GeniesimReactiveControl, GeniesimRetargetGroup
from geometry_msgs.msg import Pose, Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Header
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from geniesim_teleop.config.gripper import gripper_control
from geniesim_teleop.config.robot_interface import RobotType, robot_desc_map
from geniesim_teleop.utils.name_utils import G1_JOINT_NAMES, G2_JOINT_NAMES

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

# BEST_EFFORT accepts the engine's SensorDataQoS joint-state publisher.
QOS_PROFILE_SENSOR = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class SimNode(Node):
    def __init__(self, robot_name="G2", node_name="sim_ros_node", arm_only=False):
        super().__init__(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.init_pose = None
        self.robot_name = robot_name
        self.arm_only = arm_only
        self.config_robot()
        self.pub_joint_command = self.create_publisher(
            JointState,
            "/teleop/gripper_command",
            QOS_PROFILE_VOLATILE,
        )
        self.sub_js = self.create_subscription(
            JointState,
            "/joint_states",
            self.callback_joint_state,
            QOS_PROFILE_SENSOR,
        )
        self.publisher_playback = self.create_publisher(Bool, "/sim/playback_flag", 1)
        self.publisher_recording = self.create_publisher(Bool, "/sim/is_recording", 1)
        self.publisher_mc = self.create_publisher(GeniesimReactiveControl, "/wbc/retarget", 10)
        self.publisher_chassis = self.create_publisher(Twist, "/cmd_twist", 1)
        self.lock_js = threading.Lock()
        self.lock_tf = threading.Lock()
        self.js_msg = JointState()
        self.tf_msg = TFMessage()
        self.mc_tf_init = False
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
                "control_type": 0,
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

    def pub_chassis_twist(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        self.publisher_chassis.publish(msg)

    def config_robot(self):
        if "G1" in self.robot_name:
            self.robot_id = "G1"
        elif "G2" in self.robot_name:
            self.robot_id = "G2"
        else:
            raise Exception(f"Invalid robot name {self.robot_name}")

        if self.robot_id == "G1":
            self.joint_names = list(G1_JOINT_NAMES)
        elif self.robot_id == "G2":
            self.joint_names = list(G2_JOINT_NAMES)
        self.config_eef()

    def config_eef(self):
        if self.arm_only:
            # Arm-only teleop leaves all hand DOFs to the external Wuji bridge.
            return
        if "omnipicker" in self.robot_name or "90d" in self.robot_name:
            self.joint_names += list(gripper_control(self.robot_name).joint_names)
        elif "wuji" in self.robot_name:
            self.joint_names += [
                f"hand_{side}_{finger}_joint{joint}"
                for side in ("l", "r")
                for finger in ("thumb", "index", "middle", "ring", "pinky")
                for joint in range(1, 5)
            ]
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

    def set_joint_velocity(self, name, velocity):
        cmd_msg = JointState()
        cmd_msg.name = name
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

    def pub_mc(self, poses, waist_pose=None):
        retarget_msg = GeniesimReactiveControl()
        retarget_msg.header = self.create_header()
        retarget_msg.retarget_groups = []
        for part in self.parts:
            pose = {"left": poses[0], "right": poses[1], "arm_base": waist_pose}.get(part["id"])
            if pose is None or (self.arm_only and part["id"] == "arm_base"):
                continue
            retarget_group = GeniesimRetargetGroup()
            retarget_group.group_id = part["group_id"]
            retarget_group.control_type = 0
            retarget_group.frame_id = "base_link" if part["id"] == "arm_base" else "arm_base_link"
            retarget_group.target_frame_names = [part["target_frame_name"]]
            retarget_group.target_frame_poses = [pose]
            retarget_msg.retarget_groups.append(retarget_group)

        # Grippers publish /teleop/gripper_command for the bridge to merge, never through MC.
        if retarget_msg.retarget_groups:
            self.publisher_mc.publish(retarget_msg)

    def parse_waist_delta_pose(self, delta_xyzxyzw, anchor):
        # Half-scale relative motion, expressed in base_link. Scale the
        # rotation delta, never the absolute controller orientation.
        delta = np.asarray(delta_xyzxyzw, dtype=float)
        pose = Pose()
        pose.position.x = float(anchor.position.x + 0.5 * delta[0])
        pose.position.y = float(anchor.position.y + 0.5 * delta[1])
        pose.position.z = float(anchor.position.z + 0.5 * delta[2])
        initial = R.from_quat([anchor.orientation.x, anchor.orientation.y, anchor.orientation.z, anchor.orientation.w])
        rotation = R.from_rotvec(0.5 * R.from_quat(delta[3:]).as_rotvec()) * initial
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, rotation.as_quat())
        return pose

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
            waist_lift_group.group_id = GeniesimRetargetGroup.GROUP_WAIST
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

    def update_without_judge(self, held_targets=None):
        """Refresh the clutch frame, preserving any arm targets MC still holds."""
        try:
            latest = [
                self.tf_buffer.lookup_transform(part["frame_id"], part["target_frame_name"], rclpy.time.Time())
                for part in self.parts
            ]
            stamps = [rclpy.time.Time.from_msg(item.header.stamp) for item in latest]
            # Time zero denotes a static transform/latest lookup, not a dated
            # sample. Choose a time available to every dynamic chain.
            dated = [stamp for stamp in stamps if stamp.nanoseconds > 0]
            stamp = min(dated, key=lambda value: value.nanoseconds) if dated else rclpy.time.Time()
            poses = [
                self.transform_to_pose_ros(
                    self.tf_buffer.lookup_transform(part["frame_id"], part["target_frame_name"], stamp)
                )
                for part in self.parts
            ]
        except Exception as exc:
            # Do not partially replace anchors or block input processing. The
            # next Pico frame retries the rising edge with a fresh controller pose.
            self.get_logger().warning(f"Waiting for a coherent arm TF snapshot: {exc}", throttle_duration_sec=2.0)
            return False

        if held_targets is not None:
            base = next(pose for part, pose in zip(self.parts, poses) if part["id"] == "arm_base")
            base_pos = np.array([base.position.x, base.position.y, base.position.z])
            base_rot = R.from_quat([base.orientation.x, base.orientation.y, base.orientation.z, base.orientation.w])
            held_by_side = dict(zip(("left", "right"), held_targets))
            for i, part in enumerate(self.parts):
                held = held_by_side.get(part["id"])
                if held is None:
                    continue
                # Preserve held targets to avoid injecting wrist tracking error on re-grip.
                position = base_pos + base_rot.apply([held.position.x, held.position.y, held.position.z])
                rotation = base_rot * R.from_quat(
                    [held.orientation.x, held.orientation.y, held.orientation.z, held.orientation.w]
                )
                measured = poses[i].position
                gap_mm = 1000 * np.linalg.norm(position - [measured.x, measured.y, measured.z])
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = map(float, position)
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(
                    float, rotation.as_quat()
                )
                poses[i] = pose
                self.get_logger().info(f"Arm clutch {part['id']}: preserving held target; measured gap={gap_mm:.2f} mm")

        # Mixing a wrist at t1 with arm_base at t2 turns torso motion into an
        # arm target jump even when the controller's relative delta is zero.
        for part, pose in zip(self.parts, poses):
            part["pose"] = pose
            part["init"] = True
        return True

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
        delta_in_ee_frame = ee_rot.inv() * delta_rot * ee_rot  # transform delta from base to EE frame
        new_ee_rot = ee_rot * delta_in_ee_frame
        new_ee_xyzw = new_ee_rot.as_quat()
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
