# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()  # Create singleton instance

from omni.usd import get_world_transform_matrix
import omni.replicator.core as rep
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import get_prim_object_type
from geniesim_benchmark.utils.usd_utils import *

import numpy as np

from pxr import UsdPhysics, Usd

from std_msgs.msg import Header
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import TransformStamped
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


class RobotInterface:
    """In-process sensor cache used by LocalRecorder and obs APIs, and a
    ROS publisher of robot TF + ``/joint_states`` for teleoperation.

    Camera frames / joint state are cached in numpy arrays so that
    LocalRecorder and the obs APIs (get_observation_image, …) can read them
    directly without ROS. This object is NOT a ROS node: the eval flow
    (enable_ros=false) uses only the in-process caches and never touches rclpy.
    On the teleop flow it lazily creates a backing rclpy node (``_ensure_node``)
    to broadcast the robot TF tree (base_link, arm_base_link, cameras, …) and
    publish ``/joint_states`` for downstream consumers (motion control, teleop).
    """

    def __init__(self):
        # Backing rclpy Node, created lazily for the teleop publish path only.
        # Stays None on the eval flow so no ROS node is ever instantiated.
        self._node = None

        self._articulation = None

        # Master gate for the ROS publish path. register_robot_tf would
        # otherwise flip _enable_ros_pub on for everyone (it runs on the eval
        # path too, via _init_robot), so the eval flow (enable_ros=false) calls
        # disable_ros_pub() to latch this off and keep tick() ROS-free.
        self._ros_pub_allowed = True
        # Off until register_robot_tf wires up broadcasters/publishers.
        self._enable_ros_pub = False

        # cache
        self.annotators = {}
        self.depth_annotators = {}
        self.parameters = {}

        self.articulated_objs = []

        self._articulation_dof_names = []
        self._articulation_pos = None
        self._articulation_vel = None
        self._articulation_eff = None

        # Pre-allocated image buffers (one per camera). RGBA uint8.
        self._img_data_cache = {}  # camera_id -> H×W×4 uint8

        # Pre-built per-frame containers for joint state (kept for parity).
        self._dof_pos_cache = None
        self._dof_vel_cache = None
        self._dof_eff_cache = None

        self._current_step_index = 0
        self._metadata = None
        self.joint_names_ee = []
        self.joint_indices_ee = np.array([])
        self.all_joints = []
        self._static_tf_tree = []
        self._dynamic_tf_tree = []
        self.articulat_objects = {}

        # --- ROS TF / joint_states publishing (teleop) ---
        self._sec = 0
        self._nanosec = 0
        self._header = Header(frame_id="base_link")
        self._base_link_prim = None
        self._js_msg = JointState()
        self.pub_js = None
        self._static_tfs_prebuilt = []  # list[TransformStamped]
        self._dynamic_tfs_prebuilt = []  # list[TransformStamped]
        self.static_broadcaster = None
        self.dynamic_broadcaster = None

        # --- ROS RGB camera publishing (teleop) ---
        # Image msgs are pre-built in register_camera() and published as-is in
        # pub_camera() — no cv_bridge, which isn't importable under omni_python's
        # Python 3.11 (jazzy ships cv_bridge only for 3.12). See pub_camera().
        self.publisher_map = {}  # camera_id -> Image publisher
        self._img_msg_cache = {}  # camera_id -> pre-allocated Image msg

    def disable_ros_pub(self):
        # Latch publishing off for the eval flow: the recorder reads cached
        # frames in-process, so tick() must not publish. register_robot_tf
        # honours this latch.
        self._ros_pub_allowed = False
        self._enable_ros_pub = False

    def _ensure_node(self):
        # Lazily create the backing rclpy Node — only reached on the teleop
        # publish path, where rclpy.init() has already run (enable_ros=true).
        if self._node is None:
            from rclpy.node import Node

            self._node = Node("geniesim_sensor_node")
        return self._node

    def destroy(self):
        # Tear down the backing Node if one was created (teleop). No-op on the
        # eval path, which never instantiated a node.
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

    def register_joint_state(self, articulation):
        self._articulation = articulation
        self._articulation_dof_names = articulation.dof_names
        self._articulation_pos = self._articulation.get_joint_positions()
        self._articulation_vel = self._articulation.get_joint_velocities()
        self._articulation_eff = self._articulation.get_measured_joint_efforts()

        n_dof = len(self._articulation_dof_names)
        self._dof_pos_cache = np.empty(n_dof, dtype=np.float64)
        self._dof_vel_cache = np.empty(n_dof, dtype=np.float64)
        self._dof_eff_cache = np.empty(n_dof, dtype=np.float64)

        # /joint_states publisher (teleop only); eval flow stays node-less.
        if self._ros_pub_allowed:
            self.pub_js = self._ensure_node().create_publisher(JointState, "/joint_states", 1)
        self._js_msg.name = list(self._articulation_dof_names)
        self._js_msg.position = [0.0] * n_dof
        self._js_msg.velocity = [0.0] * n_dof
        self._js_msg.effort = [0.0] * n_dof

        self._metadata = articulation._articulation_view._metadata
        self.joint_names_ee = ["idx51_ee_l_joint", "idx91_ee_r_joint"]
        self.joint_names_ee = [jn for jn in self.joint_names_ee if jn in self._metadata.joint_indices]
        if self.joint_names_ee:
            self.joint_indices_ee = 1 + np.array([self._metadata.joint_indices[jn] for jn in self.joint_names_ee])
        else:
            self.joint_indices_ee = np.array([])

    def register_articulated_obj(self, articulated_objs):
        for _prim_path, articulation in articulated_objs.items():
            self.articulated_objs.append(articulation)

    def register_robot_tf(self, stage, robot_ns):
        robot_ns = robot_ns.replace("/", "")
        if not self._articulation:
            logger.error("register_robot_tf failed before articulation is intialized")
            return
        self._base_link_prim = stage.GetPrimAtPath(f"/{robot_ns}/base_link")
        self.all_joints = []
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                self.all_joints.append(prim)

        self._static_tf_tree = []
        self._dynamic_tf_tree = []
        self.articulat_objects = {}
        self.build_tf_tree(stage, stage.GetPrimAtPath(f"/{robot_ns}/base_link"), None, None)

        def _build_tf_list(tf_tree):
            tfs = []
            for prim, parent in tf_tree:
                if not prim:
                    continue
                tf = TransformStamped()
                tf.header.frame_id = parent.GetName() if parent else "odom"
                name = prim.GetName()
                if "link" in name or "Camera" in name:
                    tf.child_frame_id = prim.GetName()
                else:
                    tf.child_frame_id = str(prim.GetPrimPath()).split("/")[-2]
                tfs.append(tf)
            return tfs

        # Hard-coded world-frame links (parent=None) consumed by teleop:
        # arm end/base links + gripper centers + cameras.
        for _link in ("arm_l_end_link", "arm_r_end_link", "arm_base_link"):
            _prim = stage.GetPrimAtPath(f"/{robot_ns}/{_link}")
            if _prim.IsValid():
                self._dynamic_tf_tree.append((_prim, None))
        for _center_link in ("gripper_l_center_link", "gripper_r_center_link"):
            _center_prim = stage.GetPrimAtPath(f"/{robot_ns}/{_center_link}")
            if _center_prim.IsValid():
                self._dynamic_tf_tree.append((_center_prim, None))
        for _cam_path in (
            f"/{robot_ns}/head_link3/head_front_Camera",
            f"/{robot_ns}/gripper_l_base_link/Left_Camera",
            f"/{robot_ns}/gripper_r_base_link/Right_Camera",
        ):
            _cam_prim = stage.GetPrimAtPath(_cam_path)
            if _cam_prim.IsValid():
                self._dynamic_tf_tree.append((_cam_prim, None))

        for prim in stage.Traverse():
            prim_path = str(prim.GetPrimPath())
            prim_type = get_prim_object_type(prim_path)
            if prim_type == "articulation" and prim_path.startswith("/World/Objects"):
                self.articulat_objects[prim_path] = SingleArticulation(prim_path)
                logger.info(f"register articulated prim DOF={self.articulat_objects[prim_path].num_dof}")
                for child in Usd.PrimRange(prim):
                    if child.HasAPI(UsdPhysics.RigidBodyAPI):
                        child_path = str(child.GetPath())
                        already = any(str(p.GetPath()) == child_path for p, _ in self._dynamic_tf_tree if p)
                        if not already:
                            self._dynamic_tf_tree.append((child, None))
        logger.info(f"record {len(self.articulat_objects)} articulated prim(s) TF")

        self.register_articulated_obj(self.articulat_objects)

        # Build prebuilt TF messages and broadcast the static tree once.
        self._static_tfs_prebuilt = _build_tf_list(self._static_tf_tree)
        self._dynamic_tfs_prebuilt = _build_tf_list(self._dynamic_tf_tree)

        # ROS publishing setup (backing node + broadcasters) — teleop only. The
        # eval flow latched _ros_pub_allowed off in disable_ros_pub() and stays
        # node-less / ROS-free; _enable_ros_pub remains its False default.
        if self._ros_pub_allowed:
            node = self._ensure_node()
            self.static_broadcaster = StaticTransformBroadcaster(node)
            self.dynamic_broadcaster = TransformBroadcaster(node)
            self.static_broadcaster.sendTransform(self._static_tfs_prebuilt)
            self._enable_ros_pub = True

    def register_obj_tf(self, object_prim):
        self._dynamic_tf_tree.append((object_prim, None))

    def register_perception(self, stage, robot_ns):
        # No-op stub: perception sensors used to be wired through ROS topics.
        # Kept so robot configs with `perception=True` don't crash; consumers
        # should read directly from annotators / depth_annotators instead.
        return

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
            body0_targets = joint.GetBody0Rel().GetTargets()
            body1_targets = joint.GetBody1Rel().GetTargets()
            if not body0_targets or not body1_targets:
                continue
            first = stage.GetPrimAtPath(body0_targets[0])
            if first == prim:
                second = stage.GetPrimAtPath(body1_targets[0])
                if not (
                    any(t[0] == second for t in self._static_tf_tree)
                    or any(t[0] == second for t in self._dynamic_tf_tree)
                ):
                    self.build_tf_tree(stage, second, first, _joint_name)

    def register_camera(self, camera_prim, resolution, every_n_frame):
        try:
            rp = rep.create.render_product(camera_prim, (resolution[0], resolution[1]))
            camera_id = camera_prim.split("/")[-1].lower()
            self.parameters[camera_id] = {
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
            }
            # RGB Image publisher (teleop only; eval stays node-less).
            if self._ros_pub_allowed:
                self.publisher_map[camera_id] = self._ensure_node().create_publisher(
                    Image, self.parameters[camera_id]["topic_name"]["rgb"], 1
                )
        except Exception as e:
            logger.warning(f"Failed to register camera {camera_prim}: {e}")
            return

        self.annotators[camera_id] = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotators[camera_id].attach(rp)

        if "Fisheye" not in camera_prim and "Top" not in camera_prim:
            self.depth_annotators[camera_id] = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            self.depth_annotators[camera_id].attach(rp)

        self._img_data_cache[camera_id] = np.empty((resolution[1], resolution[0], 4), dtype=np.uint8)

        img = Image()
        img.header.frame_id = "camera_optical_frame"
        img.width = resolution[0]
        img.height = resolution[1]
        img.encoding = "rgba8"
        img.step = resolution[0] * 4
        self._img_msg_cache[camera_id] = img

    def tick(self, current_time: float, current_step_index: int):
        self._current_step_index = current_step_index
        self._sec = int(current_time)
        self._nanosec = int((current_time - self._sec) * 1e9)
        self._header.stamp.sec = self._sec
        self._header.stamp.nanosec = self._nanosec

        self.prepare_data()

        # Teleop only: broadcast dynamic TF + publish /joint_states + RGB.
        if self._enable_ros_pub:
            self.pub_joint_state()
            self.pub_tf()
            for cam in self.publisher_map:
                if 0 == self._current_step_index % self.parameters[cam]["every_n_frame"]:
                    self.pub_camera(cam)

    def prepare_data(self):
        if self._articulation is None:
            return

        # joint state cache
        self._articulation_pos = self._articulation.get_joint_positions()
        self._articulation_vel = self._articulation.get_joint_velocities()
        self._articulation_eff = self._articulation.get_measured_joint_efforts()

        if self._dof_pos_cache is not None:
            np.copyto(self._dof_pos_cache, np.asarray(self._articulation_pos, dtype=np.float64))
            np.copyto(self._dof_vel_cache, np.asarray(self._articulation_vel, dtype=np.float64))
            np.copyto(self._dof_eff_cache, np.asarray(self._articulation_eff, dtype=np.float64))

        # dynamic TF poses (teleop only — list is empty unless register_robot_tf ran)
        for tf, (prim, parent) in zip(self._dynamic_tfs_prebuilt, self._dynamic_tf_tree):
            if not prim:
                continue
            tf.header.stamp = self._header.stamp
            matrix = get_world_transform_matrix(prim)
            if parent:
                matrix = matrix * (get_world_transform_matrix(parent).GetInverse())
            translate = matrix.ExtractTranslation()
            orient = matrix.ExtractRotationQuat()
            tf.transform.translation.x = float(translate[0])
            tf.transform.translation.y = float(translate[1])
            tf.transform.translation.z = float(translate[2])
            tf.transform.rotation.x = float(orient.imaginary[0])
            tf.transform.rotation.y = float(orient.imaginary[1])
            tf.transform.rotation.z = float(orient.imaginary[2])
            tf.transform.rotation.w = float(orient.real)

        # camera images: pull out latest BGRA from annotator into the cache
        # the recorder will read.
        for cam, params in self.parameters.items():
            every = max(1, int(params.get("every_n_frame", 1)))
            if 0 != self._current_step_index % every:
                continue
            annot = self.annotators.get(cam)
            if annot is None:
                continue
            img = annot.get_data()
            if img is None or img.size == 0:
                continue
            buf = self._img_data_cache.get(cam)
            if buf is None or buf.shape != img.shape:
                self._img_data_cache[cam] = np.array(img, copy=True)
            else:
                np.copyto(buf, img)

    def pub_tf(self):
        if not self._dynamic_tfs_prebuilt or self.dynamic_broadcaster is None:
            return
        self.dynamic_broadcaster.sendTransform(self._dynamic_tfs_prebuilt)

    def pub_joint_state(self):
        if self.pub_js is None or self._dof_pos_cache is None:
            return
        msg = self._js_msg
        msg.header = self._header
        msg.position = self._dof_pos_cache.tolist()
        msg.velocity = self._dof_vel_cache.tolist()
        msg.effort = self._dof_eff_cache.tolist()
        self.pub_js.publish(msg)

    def pub_camera(self, camera_id):
        # Publish the cached RGBA frame as a sensor_msgs/Image. _img_data_cache
        # is refreshed in prepare_data() each tick. The Image msg is pre-built
        # in register_camera() with fixed width/height/encoding/step, so we only
        # refresh the timestamp and pixel bytes here.
        #
        # The msg is assembled inline rather than via cv_bridge.cv2_to_imgmsg:
        # this runs under omni_python (Python 3.11), and ROS jazzy's cv_bridge is
        # built only for 3.12 (its cv_bridge_boost.so can't load under 3.11). For
        # a contiguous rgba8 buffer the conversion is just a .tobytes() copy.
        buf = self._img_data_cache.get(camera_id)
        if buf is None or buf.size == 0:
            return
        try:
            img = self._img_msg_cache[camera_id]
            img.header.stamp = self._header.stamp
            img.data = np.ascontiguousarray(buf).tobytes()
            self.publisher_map[camera_id].publish(img)
        except Exception as e:
            logger.warning(f"Failed to publish camera {camera_id}: {e}")

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
        return np.ndarray([])

    def get_camera_image_rgb(self, camera):
        if camera in self._img_data_cache:
            return self._img_data_cache[camera][..., :3]
        return np.ndarray([])

    def get_observation_image(self, dir):
        ret = {}
        if dir == {}:
            for k, annot in self.annotators.items():
                ret[k] = annot.get_data()[..., :3]  # drop alpha
        else:
            for k, v in dir.items():
                ret[k] = self.annotators[v].get_data()[..., :3]
        return ret

    def get_observation_depth(self, dir):
        ret = {}
        if dir == {}:
            for k, annot in self.depth_annotators.items():
                d = annot.get_data()
                if d is not None and d.size > 0:
                    ret[k] = d.squeeze()
        else:
            for k, v in dir.items():
                if v not in self.depth_annotators:
                    continue
                d = self.depth_annotators[v].get_data()
                if d is not None and d.size > 0:
                    ret[k] = d.squeeze()
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
        return None

    def get_joint_indices_by_names(self, names: list):
        if self._metadata:
            return 1 + np.array([self._metadata.joint_indices[n] for n in names])
        return np.array([])
