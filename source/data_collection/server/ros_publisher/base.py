# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import omni.graph.core as og
import omni.usd
from isaacsim.core.nodes.scripts.utils import set_target_prims
from omni.usd import get_world_transform_matrix
from pxr import Gf, UsdPhysics

from common.base_utils.logger import logger


def _world_pose_for_tf(prim):
    """Read the current world translation and quaternion for TF publication.

    With GPU physics and Fabric, USD transforms can stay stale after cuRobo
    attaches an object. Read physical poses directly from SingleRigidPrim tensors.
    For a container, find a rigid descendant and compose its live pose with the
    static relative transform: target_static * inverse(rigid_static) * rigid_world.
    This preserves offsets between an entity and its rigid body.
    If no rigid descendant exists, search rigid ancestors only with torch tensors
    (e.g. a camera attached to an articulation link). CPU physics synchronizes USD,
    so nonphysical prims can use the USD world-transform fallback directly.
    Cache views by prim path to avoid constructing them on every publication.
    Return Gf.Vec3d translation and Gf.Quatd rotation, matching the USD fallback."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415
    rigid_prim_for_tensor = None
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_prim_for_tensor = prim
    else:
        try:
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_prim_for_tensor = child
                    break
        except Exception:
            pass
        # With no rigid descendant, search ancestors on the GPU pipeline.
        # CPU physics synchronizes USD, so the fallback needs no tensor probes.
        if rigid_prim_for_tensor is None:
            try:
                from sim_backend import tensors_as as _tensors_as  # noqa: PLC0415
                _is_gpu = _tensors_as() == "torch"
            except Exception:
                _is_gpu = False
            if _is_gpu:
                try:
                    parent = prim.GetParent()
                    while parent is not None and parent.IsValid() and str(parent.GetPath()) != "/":
                        if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                            rigid_prim_for_tensor = parent
                            break
                        parent = parent.GetParent()
                except Exception:
                    pass
    if rigid_prim_for_tensor is not None:
        try:
            rigid_path = str(rigid_prim_for_tensor.GetPath())
            rp = _RIGID_VIEW_CACHE.get(rigid_path)
            if rp is None:
                from isaacsim.core.prims import SingleRigidPrim  # noqa: PLC0415

                rp = SingleRigidPrim(prim_path=rigid_path)
                _RIGID_VIEW_CACHE[rigid_path] = rp
            from sim_backend import backend as _bk  # noqa: PLC0415

            poss, oris = rp._rigid_prim_view.get_world_poses(usd=False)
            pos = _bk.to_numpy(poss[0])
            quat_wxyz = _bk.to_numpy(oris[0])
            anc_world = Gf.Matrix4d(1)
            anc_world.SetRotate(Gf.Quatd(
                float(quat_wxyz[0]),
                Gf.Vec3d(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
            ))
            anc_world.SetTranslateOnly(
                Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
            )
            if rigid_prim_for_tensor.GetPath() == prim.GetPath():
                composed = anc_world
            else:
                # entity_current_world = entity_static * inv(anc_static) * anc_world
                # Gf row vectors: p * A * B applies A first, then B.
                target_static = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default())
                anc_static = UsdGeom.Xformable(
                    rigid_prim_for_tensor
                ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                delta = target_static * anc_static.GetInverse()
                composed = delta * anc_world
            return composed.ExtractTranslation(), composed.ExtractRotationQuat()
        except Exception:
            pass
    matrix = get_world_transform_matrix(prim)
    return matrix.ExtractTranslation(), matrix.ExtractRotationQuat()


# Cache SingleRigidPrim views by path within a recording to avoid repeated setup.
# Scene reloads recreate prims at the same paths and invalidate these views.
# Rotate the cache before each recording so object poses and joint states both
# reference the current PhysX scene.
_RIGID_VIEW_CACHE: dict = {}
# Retain retired views: the physics scene may still reference their native
# resources, and garbage collection can cause get_world_poses to crash.
_RIGID_VIEW_GRAVEYARD: list = []


def rotate_rigid_view_cache():
    """Use fresh rigid views for the next recording while retaining retired views.
    Reloading a scene recreates prims at the same paths and invalidates old views.
    Retain their native resources to avoid crashes from premature garbage collection."""
    global _RIGID_VIEW_CACHE
    n = len(_RIGID_VIEW_CACHE)
    if n:
        _RIGID_VIEW_GRAVEYARD.append(_RIGID_VIEW_CACHE)
        _RIGID_VIEW_CACHE = {}
    logger.info(
        f"Rotated rigid view cache ({n} stale entries retired, "
        f"graveyard={len(_RIGID_VIEW_GRAVEYARD)})"
    )

# Import ROS node classes only when a publish method needs them.
# sim_ros_node imports cv_bridge and rclpy; delaying it lets the server start
# without ROS dependencies when ROS publication is disabled.
def _ros_node_classes():
    from common.base_utils.ros_nodes.sim_ros_node import (  # noqa: PLC0415
        JointStatePubRosNode, TFPubRosNode,
    )
    return JointStatePubRosNode, TFPubRosNode

from sim_backend import physics, render
from isaacsim.core.prims import SingleArticulation as Articulation

class USDBase:
    def __init__(self):
        pass

    def _init_sensor(self, ros_domain_id):
        self.ros_domain_id = (int)(ros_domain_id)
        logger.info(f"ROS domain ID: {self.ros_domain_id}")

    def step(self):
        pass

    def init_all_sensors(self):
        if "sensors" in self.config:
            for sensor in self.config["sensors"]:
                if sensor["type"] == "Camera":
                    self._init_camera(sensor)
                if sensor["type"] == "Lidar":
                    self._init_lidar(sensor)
                if sensor["type"] == "IMU":
                    self._init_imu(sensor)

    def _init_lidar(self, param):
        from .lidar import publish_lidar_pointcloud, publish_lidar_scan

        lidar = render.make_lidar(prim_path=param["path"])
        lidar.initialize()
        approx_freq = param["frequency"]
        for publish in param["publish"]:
            if publish is None:
                continue
            split = publish.split(":", 2)
            topic = ""
            if len(split) > 1:
                publish = split[0]
                topic = split[1]
            if publish == "pointcloud":
                publish_lidar_pointcloud(lidar, approx_freq, topic)
            elif publish == "scan":
                publish_lidar_scan(lidar, approx_freq, topic)

    def _init_camera(self, param):
        from .camera import (
            publish_boundingbox2d_loose,
            publish_boundingbox2d_tight,
            publish_boundingbox3d,
            publish_camera_info,
            publish_depth,
            publish_noised_rgb,
            publish_pointcloud_from_depth,
            publish_rgb,
            publish_semantic_segment,
        )

        camera = render.make_camera(
            prim_path=param["path"],
            frequency=param["frequency"],
            resolution=(param["resolution"]["width"], param["resolution"]["height"]),
        )
        camera.initialize()

        step_size = param["frequency"]
        camera_graph = []
        ros_nodes = []
        for publish in param["publish"]:
            if publish is None:
                continue
            split = publish.split(":", 2)
            topic = ""
            if len(split) > 1:
                publish = split[0]
                topic = split[1]
            if publish == "rgb":
                if not param.get("noised", False):
                    ros_nodes.append(publish_rgb(camera, step_size, ""))
                else:
                    ros_nodes.append(
                        publish_noised_rgb(
                            camera=camera,
                            step_size=step_size,
                            topic="",
                            **param["noise_parameters"],
                        )
                    )
            elif publish == "info":
                camera_graph.append(publish_camera_info(camera, step_size, topic))
            elif publish == "pointcloud":
                publish_pointcloud_from_depth(camera, step_size, topic)
            elif publish == "depth":
                camera.add_distance_to_image_plane_to_frame()
                ros_nodes.append(publish_depth(camera, step_size, ""))
            elif publish == "bbox2_loose":
                publish_boundingbox2d_loose(camera, step_size, topic)
            elif publish == "bbox2_tight":
                publish_boundingbox2d_tight(camera, step_size, topic)
            elif publish == "bbox3":
                publish_boundingbox3d(camera, step_size, topic)
            elif publish == "semantic":
                publish_semantic_segment(camera, step_size, topic)

        # Note: camera.initialize() already called above, no need to call again
        return camera_graph, ros_nodes

    def _init_imu(self, param):
        from .imu import publish_imu

        imu = render.make_imu(prim_path=param["path"])
        # imu.initialize()
        approx_freq = param["frequency"]
        for publish in param["publish"]:
            if publish is None:
                continue
            split = publish.split(":", 2)
            topic = ""
            if len(split) > 1:
                publish = split[0]
                topic = split[1]
            else:
                topic = param["path"] + "_imu"
            if publish == "imu":
                publish_imu(imu, approx_freq, topic)

    def reset_graph(self):
        ros_tf_graph_path = "/World/RobotTFActionGraph"

        set_target_prims(
            primPath=ros_tf_graph_path + "/RosPublishTransformTree",
            inputName="inputs:targetPrims",
            targetPrimPaths=[],
        )

    def register_robot_tf(self, robot_prim):
        stage = omni.usd.get_context().get_stage()
        robot_ns = robot_prim.replace("/", "")
        self.all_joints = []
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint) and robot_prim in str(prim.GetPath()):
                self.all_joints.append(prim)
        # Prefer /{robot_ns}/base_link.
        # Resolve the articulation root link when the conventional path is missing.
        base_prim = stage.GetPrimAtPath(f"/{robot_ns}/base_link")
        if not (base_prim and base_prim.IsValid()):
            base_prim = self._resolve_base_link(stage, robot_prim)
        self.build_tf_tree(stage, base_prim, None, None)

    def _resolve_base_link(self, stage, robot_prim):
        """Resolve a robot root link when it is not named base_link.
        Use body0 of an articulatedRoot joint, then the root's first Xform child,
        and finally the root itself."""
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(robot_prim):
                continue
            if prim.IsA(UsdPhysics.Joint):
                art_attr = prim.GetAttribute("articulationType")
                if art_attr and art_attr.Get() == "articulatedRoot":
                    targets = prim.GetRelationship("physics:body0").GetTargets()
                    if targets:
                        cand = stage.GetPrimAtPath(targets[0])
                        if cand and cand.IsValid():
                            return cand
        root = stage.GetPrimAtPath(robot_prim)
        if root and root.IsValid():
            for child in root.GetChildren():
                if child.GetTypeName() == "Xform":
                    return child
        return root


    def build_tf_tree(self, stage, prim, parent, joint_name):
        self.tf_tree.append((prim, parent))

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
                    any(t[0] == second for t in self.tf_tree)
                ):
                    self.build_tf_tree(stage, second, first, _joint_name)

    def publish_tf(self, robot_prim, targets, step_size, topic_name="/tf"):
        from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
        self.tf_tree = []
        def get_frame_id(prim):
            prim_path = str(prim.GetPath())
            if prim_path.endswith("/entity"):
                prim_path = prim_path[1:]
                return prim_path.replace("/", "_")
            if "/entity/" in prim_path:
                prim_path = prim_path[1:]
                return prim_path.replace("/", "_")
            return prim.GetName()
        def _build_tf_list(tf_tree, tfs):
            for prim, parent in tf_tree:
                tf = TransformStamped()
                tf.header.frame_id = get_frame_id(parent) if parent else "world"
                tf.child_frame_id = get_frame_id(prim)
                tfs.append(tf)
            return tfs
        stage = omni.usd.get_context().get_stage()
        for target in targets:
            if target == robot_prim:
                self.register_robot_tf(robot_prim)
                continue
            obj_prim = stage.GetPrimAtPath(target)
            self.tf_tree.append((obj_prim, None))
        self.prebuilt_tfs = []
        self.prebuilt_tfs = _build_tf_list(self.tf_tree, self.prebuilt_tfs)

        def get_tf_list():
            for tf, (prim, parent) in zip(self.prebuilt_tfs, self.tf_tree):
                translate, orient = _world_pose_for_tf(prim)
                if parent:
                    parent_t, parent_o = _world_pose_for_tf(parent)
                    # Preserve the USD child * inverse(parent) convention using GfMatrix4d.
                    child_m = Gf.Matrix4d().SetTranslateOnly(translate).SetRotateOnly(orient)
                    parent_m = Gf.Matrix4d().SetTranslateOnly(parent_t).SetRotateOnly(parent_o)
                    rel = child_m * parent_m.GetInverse()
                    translate = rel.ExtractTranslation()
                    orient = rel.ExtractRotationQuat()
                tf.transform.translation.x = translate[0]
                tf.transform.translation.y = translate[1]
                tf.transform.translation.z = translate[2]
                tf.transform.rotation.x = orient.imaginary[0]
                tf.transform.rotation.y = orient.imaginary[1]
                tf.transform.rotation.z = orient.imaginary[2]
                tf.transform.rotation.w = orient.real
                if "Camera" in prim.GetName():
                    tf.transform.rotation.x = orient.real
                    tf.transform.rotation.y = orient.imaginary[2]
                    tf.transform.rotation.z = -orient.imaginary[1]
                    tf.transform.rotation.w = -orient.imaginary[0]
            return self.prebuilt_tfs

        node = _ros_node_classes()[1](  # TFPubRosNode
            topic_name,
            get_tf_list,
            "world",
            node_name="tf_pub_node",
            step_size=step_size,
        )
        return node

    def publish_joint(self, robot_prim,  robot, step_size, topic_name="/joint_states"):
        frame_id = topic_name

        def get_joint_state():
            try:
                joint_positions = physics.get_joint_positions(robot)
                joint_velocities = physics.get_joint_velocities(robot)
                # Get joint efforts if available, otherwise use zeros
                try:
                    joint_efforts = physics.get_measured_joint_efforts(robot)
                except (AttributeError, NotImplementedError):
                    joint_efforts = np.zeros_like(joint_positions)

                joint_dict = {
                    "position": joint_positions,
                    "velocity": joint_velocities,
                    "effort": joint_efforts,
                    "joint_names": physics.get_dof_names(robot),
                }
                return joint_dict
            except Exception as e:
                logger.error(f"Error getting joint state: {e}")
                return None

        node = _ros_node_classes()[0](  # JointStatePubRosNode
            topic_name,
            get_joint_state,
            frame_id,
            node_name=f"joint_state_node_{robot_prim.split('/')[-1]}",
            step_size=step_size,
        )
        return node

    def publish_articulation_action(self, robot, step_size, topic_name="/articulation_action"):
        frame_id = topic_name

        def get_articulation_action():
            ac = robot.get_articulation_controller()
            articulation_action = ac.get_applied_action()
            action_dict = {}
            if articulation_action is not None:
                action_dict["position"] = articulation_action.joint_positions
                action_dict["velocity"] = articulation_action.joint_velocities
                action_dict["effort"] = articulation_action.joint_efforts
                action_dict["joint_names"] = physics.get_dof_names(robot)
                return action_dict
            else:
                return None

        node = _ros_node_classes()[0](  # JointStatePubRosNode
            topic_name,
            get_articulation_action,
            frame_id,
            node_name="articulation_action_node",
            step_size=step_size,
        )
        return node

    def publish_clock(self, clock_graph_path="/ClockActionGraph"):
        ros_clock_graph_path = clock_graph_path
        og.Controller.edit(
            {
                "graph_path": ros_clock_graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RosContext", "isaacsim.ros2.bridge.ROS2Context"),
                    ("RosPublisher", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "RosPublisher.inputs:execIn"),
                    ("OnPlaybackTick.outputs:time", "RosPublisher.inputs:timeStamp"),
                    ("RosContext.outputs:context", "RosPublisher.inputs:context"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("RosContext.inputs:domain_id", self.ros_domain_id),
                ],
            },
        )
