# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance

from omni.usd import get_world_transform_matrix
import omni.replicator.core as rep

import numpy as np

from cv_bridge import CvBridge
from pxr import Gf, Sdf, UsdPhysics

from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import TransformStamped, Point, Vector3, Quaternion
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


MAP_DYNAMIC_TF_NAMES = {
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
    "idx21_arm_l_joint1",
    "idx61_arm_r_joint1",
    "idx22_arm_l_joint2",
    "idx62_arm_r_joint2",
    "idx23_arm_l_joint3",
    "idx63_arm_r_joint3",
    "idx24_arm_l_joint4",
    "idx64_arm_r_joint4",
    "idx25_arm_l_joint5",
    "idx65_arm_r_joint5",
    "idx26_arm_l_joint6",
    "idx66_arm_r_joint6",
    "idx27_arm_l_joint7",
    "idx67_arm_r_joint7",
    "idx31_gripper_l_inner_joint1",
    "idx41_gripper_l_outer_joint1",
    "idx71_gripper_r_inner_joint1",
    "idx81_gripper_r_outer_joint1",
    "idx32_gripper_l_inner_joint3",
    "idx42_gripper_l_outer_joint3",
    "idx72_gripper_r_inner_joint3",
    "idx82_gripper_r_outer_joint3",
    "idx33_gripper_l_inner_joint4",
    "idx43_gripper_l_outer_joint4",
    "idx73_gripper_r_inner_joint4",
    "idx83_gripper_r_outer_joint4",
    "idx54_gripper_l_inner_joint0",
    "idx53_gripper_l_outer_joint0",
    "idx94_gripper_r_inner_joint0",
    "idx93_gripper_r_outer_joint0",
    "idx111_chassis_lwheel_front_joint1",
    "idx112_chassis_lwheel_front_joint2",
    "idx131_chassis_rwheel_front_joint1",
    "idx132_chassis_rwheel_front_joint2",
    "idx121_chassis_lwheel_rear_joint1",
    "idx122_chassis_lwheel_rear_joint2",
    "idx141_chassis_rwheel_rear_joint1",
    "idx142_chassis_rwheel_rear_joint2",
}


class RobotInterface(Node):

    def __init__(self):
        super().__init__("geniesim_sensor_node")

        self._sec = 0
        self._nanosec = 0
        self._header = Header(frame_id="base_link")

        self._articulation = None

        self._bridge = CvBridge()

        self._static_tf_tree = None
        self._dynamic_tf_tree = None

        self._enable_ros_pub = True

        # cache
        self.annotators = {}
        self.parameters = {}
        self.publisher_map = {}

        self.articulated_obj_publishers = []
        self.articulated_objs = []

        # JointState for the main robot
        self._js_msg = JointState()
        self._js_msg.name = []  # will be filled once when we know the robot DOFs
        self._js_msg.position = []
        self._js_msg.velocity = []
        self._js_msg.effort = []

        # One JointState per articulated object
        self._joint_state_cache = {}  # articulation -> JointState

        # Pre-allocated Image messages (one per camera)
        self._img_data_cache = {}  # camera_id -> Image Data
        self._img_msg_cache = {}  # camera_id -> Image

        # Pre-built TransformStamped lists for static & dynamic TFs
        self._static_tfs_prebuilt = []  # list[TransformStamped]
        self._dynamic_tfs_prebuilt = []  # list[TransformStamped]

        # Pre-built Python containers that never change size
        self._dof_pos_cache = None
        self._dof_vel_cache = None
        self._dof_eff_cache = None

    def disable_ros_pub(self):
        self._enable_ros_pub = False

    def register_joint_state(self, articulation):
        self._articulation = articulation
        self._articulation_dof_names = articulation.dof_names
        self._articulation_pos = self._articulation.get_joint_positions()
        self._articulation_vel = self._articulation.get_joint_velocities()
        self._articulation_eff = self._articulation.get_measured_joint_efforts()

        self.pub_js = self.create_publisher(JointState, "/joint_states", 1)
        self.pub_ee = self.create_publisher(JointState, "/joint_states_ee", 1)

        self._js_msg.name = articulation.dof_names
        N = len(self._js_msg.name)
        # contiguous numpy buffers; .tolist() is still cheap
        self._dof_pos_cache = np.empty(N, dtype=np.float64)
        self._dof_vel_cache = np.empty(N, dtype=np.float64)
        self._dof_eff_cache = np.empty(N, dtype=np.float64)
        # pre-size the lists to avoid list-resize inside the loop
        self._js_msg.position = [0.0] * N
        self._js_msg.velocity = [0.0] * N
        self._js_msg.effort = [0.0] * N

        self._metadata = articulation._articulation_view._metadata
        self.joint_names_ee = ["idx51_ee_l_joint", "idx91_ee_r_joint"]
        self.joint_indices_ee = 1 + np.array([self._metadata.joint_indices[jn] for jn in self.joint_names_ee])

    def register_articulated_obj(self, articulated_objs):
        for prim_path, articulation in articulated_objs.items():
            self.articulated_objs.append(articulation)
            self.articulated_obj_publishers.append(
                self.create_publisher(JointState, f"/articulated/{prim_path.split('/')[-1]}", 1)
            )

    def register_robot_tf(self, stage, robot_ns):
        robot_ns = robot_ns.replace("/", "")
        if not self._articulation:
            logger.error("register_robot_tf failed before articulation is intialized")
            return
        self.all_joints = []
        for prim in stage.Traverse():
            # print(self.all_joints)
            if prim.IsA(UsdPhysics.Joint):
                self.all_joints.append(prim)

        self._static_tf_tree = []
        self._dynamic_tf_tree = []
        self.build_tf_tree(stage, stage.GetPrimAtPath(f"/{robot_ns}/base_link"), None, None)

        def _build_tf_list(tf_tree):
            tfs = []
            for prim, parent in tf_tree:
                tf = TransformStamped()
                tf.header.frame_id = parent.GetName() if parent else "odom"
                tf.child_frame_id = prim.GetName()
                tfs.append(tf)
            return tfs

        self._static_tfs_prebuilt = _build_tf_list(self._static_tf_tree)
        self._dynamic_tfs_prebuilt = _build_tf_list(self._dynamic_tf_tree)

        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_broadcaster = TransformBroadcaster(self)
        # self.publish_transforms(self._static_tf_tree, self.static_broadcaster)
        self.static_broadcaster.sendTransform(self._static_tfs_prebuilt)

    def register_obj_tf(self, object_prim):
        self._dynamic_tf_tree.append((object_prim, None))

    def build_tf_tree(self, stage, prim, parent, joint_name):
        prim_name = prim.GetName().split("/")[-1]
        if joint_name in MAP_DYNAMIC_TF_NAMES or "base_link" in prim_name:
            self._dynamic_tf_tree.append((prim, parent))
        else:
            self._static_tf_tree.append((prim, parent))

        for child in self.all_joints:
            joint = UsdPhysics.Joint(child)
            _joint_name = child.GetName().split("/")[-1]
            if "Joint" in _joint_name:
                continue
            first = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            if first == prim:
                second = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
                if not (
                    any(t[0] == second for t in self._static_tf_tree)
                    or any(t[0] == second for t in self._dynamic_tf_tree)
                ):
                    self.build_tf_tree(stage, second, first, _joint_name)

    def register_camera(self, camera_prim, resolution, every_n_frame):
        # register
        try:
            rp = rep.create.render_product(camera_prim, (resolution[0], resolution[1]))
            camera_id = camera_prim.split("/")[-1].lower()
            camera_param = {
                "path": camera_prim,
                "every_n_frame": every_n_frame,
                "resolution": {
                    "width": resolution[0],
                    "height": resolution[1],
                },
                "topic_name": {
                    "rgb": "genie_sim/" + camera_id + "_rgb",
                    "depth": "genie_sim/" + camera_id + "_depth",
                },
                "publish": [
                    "rgb:/" + camera_id + "_rgb",
                    "depth:/" + camera_id + "_depth",
                ],
            }

            if "Fisheye" in camera_prim or "Top" in camera_prim:
                camera_param["publish"] = [
                    "rgb:/" + camera_id + "_rgb",
                ]
            else:
                camera_param["publish"] = [
                    "rgb:/" + camera_id + "_rgb",
                    "depth:/" + camera_id + "_depth",
                ]

            # param
            self.parameters[camera_id] = camera_param

            # pub
            self.publisher_map[camera_id] = self.create_publisher(Image, camera_param["topic_name"]["rgb"], 1)

        except Exception as e:
            logger.warning(f"Failed to register camera {camera_prim}: {e}")
            return

        self.annotators[camera_id] = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotators[camera_id].attach(rp)

        img = Image()
        img.header.frame_id = "camera_optical_frame"
        img.width = resolution[0]
        img.height = resolution[1]
        img.encoding = "rgba8"
        img.step = resolution[0] * 4
        self._img_msg_cache[camera_id] = img
        self._img_data_cache[camera_id] = np.empty((resolution[1], resolution[0], 4), dtype=np.uint8)

    def tick(self, current_time: float, current_step_index: int):
        # on_tick
        self._current_step_index = current_step_index
        self._sec = int(current_time)
        self._nanosec = int((current_time - self._sec) * 1e9)

        self._header.stamp.sec = self._sec
        self._header.stamp.nanosec = self._nanosec

        # prepare data
        self.prepare_data()

        # pub
        if self._enable_ros_pub:
            self.pub_joint_state(self.pub_js, self._articulation)
            self.pub_joint_state_ee(self._articulation)
            self.pub_tf()
            self.pub_articulated_object()
            for cam in self.publisher_map:
                if 0 == self._current_step_index % self.parameters[cam]["every_n_frame"]:
                    self.pub_camera(cam)

    def prepare_data(self):
        if self._articulation is None:
            return

        # joint_state
        self._articulation_pos = self._articulation.get_joint_positions()
        self._articulation_vel = self._articulation.get_joint_velocities()
        self._articulation_eff = self._articulation.get_measured_joint_efforts()

        np.copyto(self._dof_pos_cache, self._articulation_pos)
        np.copyto(self._dof_vel_cache, self._articulation_vel)
        np.copyto(self._dof_eff_cache, self._articulation_eff)

        # tf
        for tf, (prim, parent) in zip(self._dynamic_tfs_prebuilt, self._dynamic_tf_tree):
            tf.header.stamp = self._header.stamp
            matrix = get_world_transform_matrix(prim)
            if parent:
                matrix = matrix * (get_world_transform_matrix(parent).GetInverse())
            translate = matrix.ExtractTranslation()
            orient = matrix.ExtractRotationQuat()
            tf.transform.translation.x = translate[0]
            tf.transform.translation.y = translate[1]
            tf.transform.translation.z = translate[2]
            tf.transform.rotation.x = orient.imaginary[0]
            tf.transform.rotation.y = orient.imaginary[1]
            tf.transform.rotation.z = orient.imaginary[2]
            tf.transform.rotation.w = orient.real

        # cam
        for cam in self.publisher_map:
            if 0 == self._current_step_index % self.parameters[cam]["every_n_frame"]:
                img = self.annotators[cam].get_data()
                if img is None or img.size == 0:
                    continue
                np.copyto(self._img_data_cache[cam], img)

    def publish_transforms(self, tf_tree, broadcaster):
        transforms = []
        if broadcaster == self.static_broadcaster:
            tf = TransformStamped()
            tf.header.frame_id = "map"
            tf.child_frame_id = "odom"
            tf.header.stamp = self._header.stamp
            tf.transform.translation = Vector3(x=0.0, y=0.0, z=0.0)
            tf.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            transforms.append(tf)
        for prim, parent in tf_tree:
            tf = TransformStamped()
            tf.header.frame_id = parent.GetName() if parent else "odom"
            tf.header.stamp = self._header.stamp
            tf.child_frame_id = prim.GetName()
            matrix = get_world_transform_matrix(prim)
            if parent:
                matrix = matrix * (get_world_transform_matrix(parent).GetInverse())
            translate = matrix.ExtractTranslation()
            orient = matrix.ExtractRotationQuat()
            tf.transform.translation = Vector3(x=translate[0], y=translate[1], z=translate[2])
            tf.transform.rotation = Quaternion(
                x=orient.imaginary[0],
                y=orient.imaginary[1],
                z=orient.imaginary[2],
                w=orient.real,
            )
            transforms.append(tf)
        broadcaster.sendTransform(transforms)

    def pub_tf(self):
        # if not self._dynamic_tf_tree or not self._static_tf_tree:
        #     return
        # self._dynamic_tf_tree = list(set(self._dynamic_tf_tree))
        # self.publish_transforms(self._dynamic_tf_tree, self.dynamic_broadcaster)
        if not self._dynamic_tfs_prebuilt:
            return

        self.dynamic_broadcaster.sendTransform(self._dynamic_tfs_prebuilt)

    def pub_joint_state(self, pub, articulation):
        if articulation is None:
            return

        if pub is self.pub_js:  # main robot
            msg = self._js_msg
            msg.position = self._dof_pos_cache.tolist()
            msg.velocity = self._dof_vel_cache.tolist()
            msg.effort = self._dof_eff_cache.tolist()

        else:  # articulated objects
            msg = JointState()
            msg.name = articulation.dof_names
            msg.position = articulation.get_joint_positions().tolist()
            msg.velocity = articulation.get_joint_velocities().tolist()
            msg.effort = articulation.get_measured_joint_efforts().tolist()

        msg.header = self._header
        pub.publish(msg)

    def pub_joint_state_ee(self, articulation):
        if articulation is None:
            return

        eef_6d_forces = articulation.get_measured_joint_forces(self.joint_indices_ee)

        msg = JointState()
        msg.header = self._header
        msg.name = []
        for idx, n in enumerate(self.joint_names_ee):
            readings = eef_6d_forces[idx]
            msg.name.append(f"{n}.linear.x")
            msg.name.append(f"{n}.linear.y")
            msg.name.append(f"{n}.linear.z")
            msg.name.append(f"{n}.angular.x")
            msg.name.append(f"{n}.angular.y")
            msg.name.append(f"{n}.angular.z")
            msg.effort.extend(readings.tolist())

        self.pub_ee.publish(msg)

    def pub_camera(self, camera_id):
        if 0 != self._current_step_index % self.parameters[camera_id]["every_n_frame"]:
            return

        try:
            # data = self.annotators[camera_id].get_data()  # H x W x 4, uint8
            # data is already uint8; cv_bridge expects a contiguous array
            img_msg = self._bridge.cv2_to_imgmsg(
                np.ascontiguousarray(self._img_data_cache[camera_id]),
                encoding="rgba8",
                # header=self._header,
            )

            # img_msg = self._bridge.cv2_to_imgmsg(data, encoding="rgba8", header=self._header)

            # shallow-copy back into pre-allocated object so we still reuse one msg
            img = self._img_msg_cache[camera_id]
            img.header.stamp = self._header.stamp
            img.data = img_msg.data  # list[int]
            img.height = img_msg.height
            img.width = img_msg.width
            img.step = img_msg.step
            img.encoding = img_msg.encoding
            img.is_bigendian = img_msg.is_bigendian
            self.publisher_map[camera_id].publish(img)
        except Exception as e:
            print(f"[ERROR] Failed to capture image from {camera_id}: {e}")

    def pub_articulated_object(self):
        for idx, articulation in enumerate(self.articulated_objs):
            self.pub_joint_state(self.articulated_obj_publishers[idx], articulation)

    def get_joint_state_names(self):
        return self._articulation_dof_names

    def get_joint_state_position(self):
        return self._articulation_pos

    def get_joint_state_velocity(self):
        return self._articulation_vel

    def get_joint_state_effort(self):
        return self._articulation_eff

    def get_camera_images_raw(self):
        return self._img_data_cache

    def get_camera_image_raw(self, camera):
        if camera in self._img_data_cache:
            return self._img_data_cache[camera]
        else:
            return np.ndarray()

    def get_camera_image_rgb(self, camera):
        if camera in self._img_data_cache:
            return self._img_data_cache[camera][..., :3]
        else:
            return np.ndarray()

    def get_observation_image(self, dir):
        ret = {}
        if dir == {}:
            for k in self.annotators.keys():
                ret[k] = self.annotators[k].get_data()[..., :3]  # Remove Alpha channel, (H, W, 3)
        else:
            for k, v in dir.items():
                ret[k] = self.annotators[v].get_data()[..., :3]  # Remove Alpha channel, (H, W, 3)

        return ret

    def get_joint_state_dict(self):
        return {
            self._articulation.dof_names[i]: self._articulation.get_joint_positions().tolist()[i]
            for i in range(len(self._articulation.dof_names))
        }

    def get_joint_state(self):
        return self._articulation.get_joint_positions().tolist()

    def get_joint_indices_by_name(self, name: str):
        if self._metadata:
            return self._metadata.joint_indices[name]
        else:
            return None

    def get_joint_indices_by_names(self, names: list):
        if self._metadata:
            return 1 + np.array([self._metadata.joint_indices[n] for n in names])
        else:
            return np.array()
