# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys

import grpc
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.append(current_directory)

# Add scripts directory (formerly server-side aimdk directory) to sys.path for importing aimdk modules
# Find project root directory (directory containing scripts) by traversing up from current file location
project_root = current_directory
while project_root != os.path.dirname(project_root):  # Until reaching filesystem root
    if os.path.exists(os.path.join(project_root, "scripts")):
        break
    project_root = os.path.dirname(project_root)
scripts_dir = os.path.join(project_root, "scripts")
if os.path.exists(scripts_dir) and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
import json
import os
import time

import pinocchio

# Arm protocol
from common.aimdk.protocol.hal.arm import arm_pb2, arm_pb2_grpc

# Joint protocol
from common.aimdk.protocol.hal.joint import joint_channel_pb2, joint_channel_pb2_grpc

# Observation protocol
# Object protocol
# Gripper protocol
from common.aimdk.protocol.sim import (
    sim_gripper_service_pb2,
    sim_gripper_service_pb2_grpc,
    sim_object_service_pb2,
    sim_object_service_pb2_grpc,
    sim_observation_service_pb2,
    sim_observation_service_pb2_grpc,
)
from common.base_utils.logger import logger


def find_urdf_in_robot_cfg(robot_urdf_name, project_root):
    """
    Traverse the robot_cfg directory to find a URDF file matching robot_urdf_name.

    Args:
        robot_urdf_name: URDF filename
        project_root: Project root directory path

    Returns:
        Full path to the URDF file

    Raises:
        FileNotFoundError: If no matching URDF file is found
    """
    robot_cfg_dir = os.path.join(project_root, "config/robot_cfg")
    if not os.path.exists(robot_cfg_dir):
        raise FileNotFoundError(f"robot_cfg directory does not exist: {robot_cfg_dir}")

    # Traverse robot_cfg directory and its subdirectories
    for root, dirs, files in os.walk(robot_cfg_dir):
        for file in files:
            if file == robot_urdf_name and file.endswith(".urdf"):
                urdf_path = os.path.join(root, file)
                return urdf_path

    # Raise error if not found
    raise FileNotFoundError(
        f"URDF file not found in robot_cfg directory: {robot_urdf_name}\n" f"Search path: {robot_cfg_dir}"
    )


# All rotation angles in the code are in degrees
class RpcClient:
    def __init__(self, client_host, robot_urdf=""):
        for i in range(600):
            try:
                self.channel = grpc.insecure_channel(
                    client_host, options=[("grpc.max_receive_message_length", 16094304)]
                )
                grpc.channel_ready_future(self.channel).result(timeout=5)
                self.robot_urdf = robot_urdf
                # Find and store URDF path during initialization
                self.urdf_path = find_urdf_in_robot_cfg(self.robot_urdf, project_root)
                break
            except grpc.FutureTimeoutError as e:
                logger.error(f"Failed to connect to gRPC server[{i}]: {e}")
                time.sleep(3)
                if i >= 599:
                    raise e
            except grpc.RpcError as e:
                logger.error(f"Failed to connect to gRPC server[{i}]: {e}")
                time.sleep(3)
                if i >= 599:
                    raise e

    def set_task_task_basic_info(self, task_description):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetFrameStateReq()
        frame_state = {"task_description": task_description}
        _frame_state: str = json.dumps(frame_state)
        req.frame_state = _frame_state
        response = stub.set_frame_state(req)
        return response

    def set_task_metric(self, task_metric):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetTaskMetricReq()
        metric = {"task_metric": task_metric}
        _task_metric: str = json.dumps(metric)
        req.metric = _task_metric
        response = stub.set_task_metric(req)
        return response

    def set_frame_state(
        self,
        action: str,
        substage_id: int,
        active_id: str,
        passive_id: str,
        if_attached: bool,
        set_gripper_open=False,
        set_gripper_close=False,
        arm="default",
        target_pose=None,
        action_description={},
        error_description={},
    ):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetFrameStateReq()
        if target_pose is not None:
            target_pose = target_pose.tolist()
        frame_state = {
            "action": action,
            "substage_id": substage_id,
            "if_attached": if_attached,
            "set_gripper_open": set_gripper_open,
            "set_gripper_close": set_gripper_close,
            "active_id": active_id,
            "passive_id": passive_id,
            "arm": arm,
            "target_pose": target_pose,
            "action_description": action_description,
            "error_description": error_description,
        }
        _frame_state: str = json.dumps(frame_state)
        req.frame_state = _frame_state
        response = stub.set_frame_state(req)
        return response

    def moveto(
        self,
        target_position,
        target_quaternion,
        arm_name,
        is_backend=True,
        ee_interpolation=False,
        distance_frame=0.0008,
        goal_offset=[0, 0, 0, 1, 0, 0, 0],
        path_constraint=[],
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links=[],
        motion_run_ratio=1.0,
        gripper_action_timing={},
        from_current_pose=False,
    ):
        stub = arm_pb2_grpc.ArmControlServiceStub(self.channel)
        req = arm_pb2.LinearMoveReq()
        req.robot_name = arm_name
        req.pose.position.x, req.pose.position.y, req.pose.position.z = target_position
        req.pose.rpy.rw, req.pose.rpy.rx, req.pose.rpy.ry, req.pose.rpy.rz = target_quaternion
        req.is_block = is_backend
        req.ee_interpolation = ee_interpolation
        req.distance_frame = distance_frame
        req.goal_offset[:] = goal_offset
        req.path_constraint[:] = path_constraint
        req.offset_and_constraint_in_goal_frame = offset_and_constraint_in_goal_frame
        req.disable_collision_links[:] = disable_collision_links
        req.motion_run_ratio = motion_run_ratio
        req.gripper_action_timing = json.dumps(gripper_action_timing)
        req.from_current_pose = from_current_pose
        response = stub.linear_move(req)
        return response

    def set_joint_positions(
        self,
        target_joint_position,
        target_joint_indices=None,
        target_joint_names=None,
        is_trajectory=False,
    ):
        stub = joint_channel_pb2_grpc.JointControlServiceStub(self.channel)
        req = joint_channel_pb2.SetJointReq()
        req.is_trajectory = is_trajectory
        for idx, pos in enumerate(target_joint_position):
            joint_position = joint_channel_pb2.JointCommand()
            joint_position.position = pos
            if target_joint_names is not None:
                joint_position.name = target_joint_names[idx]
            elif target_joint_indices is None:
                joint_position.sequence = idx
            else:
                joint_position.sequence = target_joint_indices[idx]
            req.commands.append(joint_position)
        response = stub.set_joint_position(req)
        return response

    def get_joint_positions(self):
        stub = joint_channel_pb2_grpc.JointControlServiceStub(self.channel)
        req = joint_channel_pb2.GetJointReq()
        req.serial_no = "robot"
        response = stub.get_joint_position(req)
        return response

    # usd_path is currently fixed to directories under data, e.g., genie3D/01.usd
    # prim_path must be specified within "/World/Objects/xxx" for easy reset
    def add_object(
        self,
        usd_path,
        prim_path,
        label_name,
        target_position,
        target_quaternion,
        target_scale,
        color,
        material,
        mass=0.01,
        add_particle=True,
        particle_position=[0, 0, 0],
        particle_scale=[0.1, 0.1, 0.1],
        particle_color=[1, 1, 1],
        static_friction=0.5,
        dynamic_friction=0.5,
        add_rigid_body=True,
        model_type="convexDecomposition",
    ):
        stub = sim_object_service_pb2_grpc.SimObjectServiceStub(self.channel)
        req = sim_object_service_pb2.AddObjectReq()
        req.usd_path = usd_path
        req.prim_path = prim_path
        req.label_name = label_name
        req.object_color.r, req.object_color.g, req.object_color.b = color
        (
            req.object_pose.position.x,
            req.object_pose.position.y,
            req.object_pose.position.z,
        ) = target_position
        (
            req.object_pose.rpy.rw,
            req.object_pose.rpy.rx,
            req.object_pose.rpy.ry,
            req.object_pose.rpy.rz,
        ) = target_quaternion
        req.object_scale.x, req.object_scale.y, req.object_scale.z = target_scale
        req.object_material = material
        req.object_mass = mass
        req.add_particle = add_particle
        req.static_friction = static_friction
        req.dynamic_friction = dynamic_friction
        req.particle_position.x, req.particle_position.y, req.particle_position.z = (
            particle_position
        )
        req.particle_scale.x, req.particle_scale.y, req.particle_scale.z = particle_scale
        req.particle_color.r, req.particle_color.g, req.particle_color.b = particle_color
        req.add_rigid_body = add_rigid_body
        req.model_type = model_type

        response = stub.add_object(req)
        return response

    def get_object_pose(self, prim_path):
        stub = sim_object_service_pb2_grpc.SimObjectServiceStub(self.channel)
        req = sim_object_service_pb2.GetObjectPoseReq()
        req.prim_path = prim_path
        response = stub.get_object_pose(req)
        return response

    def get_part_dof_joint(self, object_prim_path, part_name):
        stub = sim_object_service_pb2_grpc.SimObjectServiceStub(self.channel)
        req = sim_object_service_pb2.GetPartDofJointReq()
        req.prim_path = object_prim_path
        req.part_name = part_name
        response = stub.get_part_dof_joint(req)
        return response

    def set_gripper_state(self, gripper_command, is_right, opened_width):
        stub = sim_gripper_service_pb2_grpc.SimGripperServiceStub(self.channel)
        req = sim_gripper_service_pb2.SetGripperStateReq()
        req.gripper_command = gripper_command
        req.is_right = is_right
        req.opened_width = opened_width
        response = stub.set_gripper_state(req)
        return response

    # Client-side start recording
    def start_recording(self, data_keys, fps, task_name):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.GetObservationReq()
        req.startRecording = True
        req.fps = fps
        req.task_name = task_name
        if "camera" in data_keys:
            data_keys["camera"]["render_depth"]
            data_keys["camera"]["render_semantic"]
            data_keys["camera"]["camera_prim_list"]
            req.isCam = True
            req.CameraReq.render_depth = data_keys["camera"]["render_depth"]
            req.CameraReq.render_semantic = data_keys["camera"]["render_semantic"]
            req.CameraReq.additional_parameters = data_keys.get("additional_parameters", "")
            for prim in data_keys["camera"]["camera_prim_list"]:
                req.CameraReq.camera_prim_list.append(prim)
        req.isJoint = data_keys["joint_position"]
        req.isGripper = data_keys["gripper"]
        response = stub.get_observation(req)
        return response

    # Client-side stop recording
    def stop_recording(self):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.GetObservationReq()
        req.stopRecording = True
        response = stub.get_observation(req)
        return response

    def reset(self):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.ResetReq()
        req.reset = True
        response = stub.reset(req)
        return response

    def attach_obj(self, prim_paths, is_right=True):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.AttachReq()
        req.is_right = is_right
        for prim in prim_paths:
            req.obj_prims.append(prim)
        response = stub.attach_obj(req)
        return response

    def detach_obj(self):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.DetachReq()
        req.detach = True
        response = stub.detach_obj(req)
        return response

    def remove_objs_from_obstacle(self, prim_paths):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.RemoveObstacleReq()
        for prim in prim_paths:
            req.obj_prims.append(prim)
        response = stub.remove_objs_from_obstacle(req)
        return response

    def send_task_status(self, isSuccess, fail_stage_step):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.TaskStatusReq()
        for step in fail_stage_step:
            req.failStep.append(step)
        req.isSuccess = isSuccess
        response = stub.task_status(req)
        return response

    def exit(self):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.ExitReq()
        req.exit = True
        response = stub.exit(req)
        return response

    def get_ee_pose(self, is_right):
        stub = joint_channel_pb2_grpc.JointControlServiceStub(self.channel)
        req = joint_channel_pb2.GetEEPoseReq()
        req.is_right = is_right
        response = stub.get_ee_pose(req)
        return response

    def get_ik_status(self, target_poses, is_right, ObsAvoid=False, output_link_pose=False):
        stub = joint_channel_pb2_grpc.JointControlServiceStub(self.channel)
        req = joint_channel_pb2.GetIKStatusReq()
        req.is_right = is_right
        req.ObsAvoid = ObsAvoid
        req.output_link_pose = output_link_pose
        model = pinocchio.buildModelFromUrdf(self.urdf_path)
        data = model.createData()
        joint_names = []
        for name in model.names:
            joint_names.append(str(name))

        for pose in target_poses:
            _pose = sim_observation_service_pb2.SE3MatrixPose()
            _pose.elements.extend(pose.flatten().tolist())
            req.target_pose.append(_pose)
        response = stub.get_ik_status(req)
        result = []
        for ik_status in response.IKStatus:
            joint_datas = {}
            for name, position in zip(ik_status.joint_names, ik_status.joint_positions):
                joint_datas[name] = position

            joint_positions = []
            for name in model.names:
                if name == "universe":
                    continue
                if name not in joint_datas:
                    joint_positions.append(0)
                else:
                    joint_positions.append(joint_datas[name])
            joint_positions = np.array(joint_positions)

            pinocchio.forwardKinematics(model, data, joint_positions)
            if "G1" in self.robot_urdf:
                J = pinocchio.computeJointJacobian(model, data, joint_positions, 24)
            elif "G2" in self.robot_urdf:
                J = pinocchio.computeJointJacobian(
                    model, data, joint_positions, 45
                )
            else:
                J = pinocchio.computeJointJacobian(model, data, joint_positions, 7)
            manip = np.sqrt(np.linalg.det(np.dot(J, J.T)))
            tmp_result = {
                "status": ik_status.isSuccess,
                "joint_positions": ik_status.joint_positions,
                "joint_names": ik_status.joint_names,
            }
            tmp_result["Jacobian"] = manip
            if ObsAvoid and output_link_pose:
                link_poses_dict = {}
                link_poses = ik_status.link_poses
                for link_pose in link_poses:
                    link_poses_dict[link_pose.link_name] = link_pose.link_pose
                tmp_result["link_poses"] = link_poses_dict
            result.append(tmp_result)
        return result

    def add_camera(
        self,
        camera_prim,
        camera_position,
        camera_rotation,
        width,
        height,
        focus_length,
        horizontal_aperture,
        vertical_aperture,
        is_local,
    ):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.AddCameraReq()
        req.camera_prim = camera_prim
        (
            req.camera_pose.position.x,
            req.camera_pose.position.y,
            req.camera_pose.position.z,
        ) = camera_position
        (
            req.camera_pose.rpy.rw,
            req.camera_pose.rpy.rx,
            req.camera_pose.rpy.ry,
            req.camera_pose.rpy.rz,
        ) = camera_rotation
        req.focus_length = focus_length
        req.horizontal_aperture = horizontal_aperture
        req.vertical_aperture = vertical_aperture
        req.width = width
        req.height = height
        req.is_local = is_local
        response = stub.add_camera(req)
        return response

    def init_robot(
        self,
        robot_cfg,
        robot_usd,
        scene_usd,
        init_position=[0, 0, 0],
        init_rotation=[1, 0, 0, 0],
        stand_type="cylinder",
        stand_size_x=0.1,
        stand_size_y=0.1,
        robot_init_arm_pose=None,
    ):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.InitRobotReq()
        req.robot_cfg_file, req.robot_usd_path, req.scene_usd_path = (
            robot_cfg,
            robot_usd,
            scene_usd,
        )
        (
            req.robot_pose.position.x,
            req.robot_pose.position.y,
            req.robot_pose.position.z,
        ) = init_position
        (
            req.robot_pose.rpy.rw,
            req.robot_pose.rpy.rx,
            req.robot_pose.rpy.ry,
            req.robot_pose.rpy.rz,
        ) = init_rotation
        if robot_init_arm_pose is not None:  # Final initial position is not determined by this
            for joint_name, pos in robot_init_arm_pose.items():
                if pos is not None:
                    joint_position = joint_channel_pb2.JointCommand()
                    joint_position.position = pos
                    # joint_position.sequence = idx
                    joint_position.name = joint_name
                    req.joint_cmd.append(joint_position)
                # idx += 1
        req.stand_type = stand_type
        req.stand_size_x = stand_size_x
        req.stand_size_y = stand_size_y
        response = stub.init_robot(req)
        return response

    def set_object_pose(self, object_info, joint_cmd, object_joint_info=None):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetObjectPoseReq()
        for object in object_info:
            object_pose = sim_observation_service_pb2.ObjectPose()
            object_pose.prim_path = object["prim_path"]
            (
                object_pose.pose.position.x,
                object_pose.pose.position.y,
                object_pose.pose.position.z,
            ) = object["position"]
            (
                object_pose.pose.rpy.rw,
                object_pose.pose.rpy.rx,
                object_pose.pose.rpy.ry,
                object_pose.pose.rpy.rz,
            ) = object["rotation"]
            req.object_pose.append(object_pose)
        for pos in joint_cmd:
            cmd = sim_observation_service_pb2.JointCommand()
            cmd.position = pos
            req.joint_cmd.append(cmd)
        if object_joint_info is not None:
            for joint in object_joint_info:
                object_joint = sim_observation_service_pb2.ObjectJoint()
                object_joint.prim_path = joint["prim_path"]
                for pos in joint["joint_cmd"]:
                    cmd = sim_observation_service_pb2.JointCommand()
                    cmd.position = pos
                    object_joint.joint_cmd.append(cmd)
                req.object_joint.append(object_joint)
        response = stub.set_object_pose(req)
        return response

    def set_trajectory_list(self, trajectory_list, is_block=False):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetTrajectoryListReq()
        req.is_block = is_block
        for point in trajectory_list:
            pose = sim_observation_service_pb2.SE3RpyPose()
            pose.position.x, pose.position.y, pose.position.z = point[0]
            pose.rpy.rw, pose.rpy.rx, pose.rpy.ry, pose.rpy.rz = point[1]
            req.trajectory_point.append(pose)
        response = stub.set_trajectory_list(req)
        return response

    def set_target_point(self, position):
        stub = sim_object_service_pb2_grpc.SimObjectServiceStub(self.channel)
        req = sim_object_service_pb2.SetTargetPointReq()
        req.point_position.x, req.point_position.y, req.point_position.z = position
        response = stub.set_target_point(req)
        return response

    def set_material(self, material_info):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetMaterailReq()
        for mat in material_info:
            logger.info(mat)
            mat_info = sim_observation_service_pb2.MaterialInfo()
            mat_info.object_prim = mat["object_prim"]
            mat_info.material_name = mat["material_name"]
            mat_info.material_path = mat["material_path"]
            if "label_name" in mat:
                mat_info.label_name = mat["label_name"]
            req.materials.append(mat_info)
        response = stub.set_material(req)
        return response

    def set_light(self, light_info):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetLightReq()
        for light in light_info:
            logger.info(light)
            light_cfg = sim_observation_service_pb2.LightCfg()
            light_cfg.light_type = light["light_type"]
            light_cfg.light_prim = light["light_prim"]
            light_cfg.light_temperature = light["light_temperature"]
            light_cfg.light_intensity = light["light_intensity"]
            (
                light_cfg.light_rotation.rw,
                light_cfg.light_rotation.rx,
                light_cfg.light_rotation.ry,
                light_cfg.light_rotation.rz,
            ) = light["rotation"]
            light_cfg.light_texture = light["texture"]
            req.lights.append(light_cfg)
        response = stub.set_light(req)
        return response

    def store_current_state(self, playback_id: str):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.StoreCurrentStateReq()
        req.playback_id = playback_id
        response = stub.store_current_state(req)
        return response

    def playback(self, playback_id: str):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.PlaybackReq()
        req.playback_id = playback_id
        response = stub.playback(req)
        return response

    def get_checker_status(self, checker_config):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.GetCheckerStatusReq()
        req.checker = json.dumps(checker_config)
        response = stub.get_checker_status(req)
        return response
