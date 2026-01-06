# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np

# Arm protocol
from common.aimdk.protocol.hal.arm import arm_pb2, arm_pb2_grpc

# Joint protocol
from common.aimdk.protocol.hal.joint import joint_channel_pb2, joint_channel_pb2_grpc

# Camera protocol
from common.aimdk.protocol.hal.sensors import rs2_camera_pb2, rs2_camera_pb2_grpc

# Observation protocol
# Object protocol
# Hand protocol
from common.aimdk.protocol.sim import (
    sim_camera_service_pb2,
    sim_camera_service_pb2_grpc,
    sim_gripper_service_pb2,
    sim_gripper_service_pb2_grpc,
    sim_object_service_pb2,
    sim_object_service_pb2_grpc,
    sim_observation_service_pb2,
    sim_observation_service_pb2_grpc,
)
from common.base_utils.logger import logger
from server.command_enum import Command


class CameraService(rs2_camera_pb2_grpc.CameraService):
    def __init__(self, server_function):
        self.server_function = server_function

    def SetCameraInfo(self, camera_info, width, height, ppx, ppy, fx, fy):
        camera_info.width = width
        camera_info.height = height
        camera_info.ppx = ppx
        camera_info.ppy = ppy
        camera_info.fx = fx
        camera_info.fy = fy

    def get_camera_data(self, req, rsp):
        rsp = rs2_camera_pb2.GetCameraDataResponse()
        rsp.serial_no = req.serial_no
        current_camera = self.server_function.blocking_start_server(
            data={
                "Cam_prim_path": req.serial_no,
                "isRGB": True,
                "isDepth": True,
                "isSemantic": False,
                "isGN": False,
            },
            Command=Command.GET_CAMERA_DATA,
        )
        camera_info = current_camera["camera_info"]
        self.SetCameraInfo(
            rsp.color_info,
            camera_info["width"],
            camera_info["height"],
            camera_info["ppx"],
            camera_info["ppy"],
            camera_info["fx"],
            camera_info["fy"],
        )
        rgb_camera = current_camera["rgb"]
        rsp.color_image.format = "rgb"
        if rgb_camera is not None:
            rsp.color_image.data = rgb_camera.tobytes()
        depth_camera = current_camera["depth"]
        rsp.depth_image.format = "depth"
        if depth_camera is not None:
            rsp.depth_image.data = depth_camera.tobytes()
        return rsp


class SimCameraService(sim_camera_service_pb2_grpc.SimCameraService):
    def __init__(self, server_function):
        self.server_function = server_function

    def get_semantic_data(self, req, rsp):
        rsp = sim_camera_service_pb2.GetSemanticResponse()
        rsp.serial_no = req.serial_no
        current_camera = self.server_function.blocking_start_server(
            data={
                "Cam_prim_path": req.serial_no,
                "isRGB": False,
                "isDepth": False,
                "isSemantic": True,
                "isGN": False,
            },
            Command=Command.GET_SEMANTIC_DATA,
        )
        semantic_image = current_camera["semantic"][0]
        label_ids = current_camera["semantic"][1]
        rsp.semantic_mask.name = "semantic"
        if semantic_image is not None:
            rsp.semantic_mask.data = semantic_image.tobytes()
        for key in label_ids:
            label = sim_camera_service_pb2.SemanticLabel()
            label.label_name = key
            label.label_id = label_ids[key]
            rsp.label_dict.append(label)
        return rsp


class armService(arm_pb2_grpc.ArmControlService):
    def __init__(self, server_function):
        self.server_function = server_function

    def linear_move(self, req, rsp):
        rsp = arm_pb2.LinearMoveRsp()
        isArmRight = False
        if req.robot_name == "right":
            isArmRight = True
        target_position = np.array([req.pose.position.x, req.pose.position.y, req.pose.position.z])
        target_rotation = np.array(
            [req.pose.rpy.rw, req.pose.rpy.rx, req.pose.rpy.ry, req.pose.rpy.rz]
        )
        is_backend = req.is_block
        motion_run_ratio = req.motion_run_ratio
        try:
            logger.info(f"gripper_action_timing{req.gripper_action_timing}")
            gripper_action_timing = json.loads(req.gripper_action_timing)
            logger.info(f"gripper_action_timing{gripper_action_timing}")
        except Exception as e:
            logger.error(f"Failed to parse gripper_action_timing{e}")
            gripper_action_timing = {}
        if "timing" not in gripper_action_timing or "state" not in gripper_action_timing:
            gripper_action_timing = None

        isSuccess = self.server_function.blocking_start_server(
            data={
                "isArmRight": isArmRight,
                "target_position": target_position,
                "target_rotation": target_rotation,
                "is_backend": is_backend,
                "ee_interpolation": req.ee_interpolation,
                "distance_frame": req.distance_frame,
                "goal_offset": req.goal_offset,
                "path_constraint": req.path_constraint,
                "offset_and_constraint_in_goal_frame": req.offset_and_constraint_in_goal_frame,
                "disable_collision_links": req.disable_collision_links,
                "motion_run_ratio": motion_run_ratio,
                "gripper_action_timing": gripper_action_timing,
                "from_current_pose": req.from_current_pose,
            },
            Command=Command.LINEAR_MOVE,
        )
        rsp.errmsg = str(isSuccess)
        return rsp


class JointService(joint_channel_pb2_grpc.JointControlService):
    def __init__(self, server_function):
        self.server_function = server_function

    def get_joint_position(self, req, rsp):
        rsp = joint_channel_pb2.GetJointRsp()
        joint_positions = self.server_function.blocking_start_server(
            data="get_joints", Command=Command.GET_JOINT_POSITION
        )
        for joint_name in joint_positions:
            joint_state = joint_channel_pb2.JointState()
            joint_state.name = joint_name
            joint_state.position = joint_positions[joint_name]
            rsp.states.append(joint_state)
        return rsp

    def set_joint_position(self, req, rsp):
        rsp = joint_channel_pb2.SetJointRsp()
        joint_num = len(req.commands)
        target_joint_position = []
        target_joint_names = []
        is_trajectory = req.is_trajectory
        for index in range(joint_num):
            v = req.commands[index].position
            name = req.commands[index].name
            if v is None:
                continue
            if not np.isfinite(v):
                continue
            target_joint_position.append(v)
            # target_joint_indices.append(idc)
            target_joint_names.append(name)
        self.server_function.blocking_start_server(
            data={
                "target_joints_position": target_joint_position,
                "is_trajectory": is_trajectory,
                # "target_joints_indices": target_joint_indices,
                "target_joint_names": target_joint_names,
            },
            Command=Command.SET_JOINT_POSITION,
        )
        rsp.errmsg = "Move Joint"
        return rsp

    def get_ee_pose(self, req, rsp):
        rsp = joint_channel_pb2.GetEEPoseRsp()
        is_right = req.is_right
        position, rotation = self.server_function.blocking_start_server(
            data={"isRight": is_right}, Command=Command.GET_EE_POSE
        )

        rsp.ee_pose.position.x, rsp.ee_pose.position.y, rsp.ee_pose.position.z = position
        (
            rsp.ee_pose.rpy.rw,
            rsp.ee_pose.rpy.rx,
            rsp.ee_pose.rpy.ry,
            rsp.ee_pose.rpy.rz,
        ) = rotation
        return rsp

    def get_ik_status(self, req, rsp):
        rsp = joint_channel_pb2.GetIKStatusRsp()
        target_poses = []
        for pose in req.target_pose:
            pose = np.array(pose.elements)
            pose = pose.reshape(4, 4).tolist()
            target_poses.append(pose)
        is_right = req.is_right
        ObsAvoid = req.ObsAvoid
        ik_result = self.server_function.blocking_start_server(
            data={
                "target_poses": target_poses,
                "isRight": is_right,
                "ObsAvoid": ObsAvoid,
                "output_link_pose": req.output_link_pose,
            },
            Command=Command.GET_IK_STATUS,
        )
        for result in ik_result:
            ik_status = joint_channel_pb2.IKStatus()
            ik_status.isSuccess = result[0]
            for key, value in result[1].items():
                ik_status.joint_names.append(key)
                ik_status.joint_positions.append(value)
            if req.output_link_pose and ObsAvoid:
                for key, value in result[2].items():
                    link_pose = joint_channel_pb2.LinkPose()
                    link_pose.link_name = key
                    position, rotation = value
                    (
                        link_pose.link_pose.position.x,
                        link_pose.link_pose.position.y,
                        link_pose.link_pose.position.z,
                    ) = position
                    (
                        link_pose.link_pose.rpy.rw,
                        link_pose.link_pose.rpy.rx,
                        link_pose.link_pose.rpy.ry,
                        link_pose.link_pose.rpy.rz,
                    ) = rotation
                    ik_status.link_poses.append(link_pose)
            rsp.IKStatus.append(ik_status)
        return rsp

    def SetJointState(self, req, rsp):
        joint_channel_pb2.SetJointStateRsp()


class ObjectService(sim_object_service_pb2_grpc.SimObjectService):
    def __init__(self, server_function):
        self.server_function = server_function

    def add_object(self, req, rsp):
        rsp = sim_object_service_pb2.AddObjectRsp()
        usd_path = req.usd_path
        prim_path = req.prim_path
        label_name = req.label_name
        object_mass = req.object_mass
        object_color = np.array([req.object_color.r, req.object_color.g, req.object_color.b])
        object_material = req.object_material
        target_position = np.array(
            [
                req.object_pose.position.x,
                req.object_pose.position.y,
                req.object_pose.position.z,
            ]
        )
        target_rotation = np.array(
            [
                req.object_pose.rpy.rw,
                req.object_pose.rpy.rx,
                req.object_pose.rpy.ry,
                req.object_pose.rpy.rz,
            ]
        )
        target_scale = np.array([req.object_scale.x, req.object_scale.y, req.object_scale.z])
        static_friction = req.static_friction
        dynamic_friction = req.dynamic_friction
        add_rigid_body = req.add_rigid_body
        model_type = req.model_type
        self.server_function.blocking_start_server(
            data={
                "usd_object_path": usd_path,
                "usd_object_prim_path": prim_path,
                "usd_label_name": label_name,
                "usd_object_position": target_position,
                "usd_object_rotation": target_rotation,
                "usd_object_scale": target_scale,
                "object_color": object_color,
                "object_material": object_material,
                "object_mass": object_mass,
                "static_friction": static_friction,
                "dynamic_friction": dynamic_friction,
                "add_rigid_body": add_rigid_body,
                "model_type": model_type,
            },
            Command=Command.ADD_OBJECT,
        )
        rsp.label_name = label_name
        return rsp

    def get_object_pose(self, req, rsp):
        rsp = sim_object_service_pb2.GetObjectPoseRsp()
        prim_path = req.prim_path
        object_pose = self.server_function.blocking_start_server(
            data={"object_prim_path": prim_path}, Command=Command.GET_OBJECT_POSE
        )
        rsp.prim_path = prim_path
        position, rotation = object_pose
        (
            rsp.object_pose.position.x,
            rsp.object_pose.position.y,
            rsp.object_pose.position.z,
        ) = position
        (
            rsp.object_pose.rpy.rw,
            rsp.object_pose.rpy.rx,
            rsp.object_pose.rpy.ry,
            rsp.object_pose.rpy.rz,
        ) = rotation
        return rsp

    def get_part_dof_joint(self, req, rsp):
        rsp = sim_object_service_pb2.GetPartDofJointRsp()
        prim_path = req.prim_path
        part_name = req.part_name
        joint_state = self.server_function.blocking_start_server(
            data={"object_prim_path": prim_path, "part_name": part_name},
            Command=Command.GET_PART_DOF_JOINT,
        )
        rsp.joint_name = joint_state["joint_name"]
        rsp.joint_position = joint_state["joint_position"]
        rsp.joint_velocity = joint_state["joint_velocity"]
        return rsp

    def set_target_point(self, req, rsp):
        rsp = sim_object_service_pb2.SetTargetPointRsp()
        target_position = np.array(
            [req.point_position.x, req.point_position.y, req.point_position.z]
        )
        rsp.msg = self.server_function.blocking_start_server(
            data={"target_position": target_position}, Command=Command.SET_TARGET_POINT
        )
        return rsp


class GripperService(sim_gripper_service_pb2_grpc.SimGripperService):
    def __init__(self, server_function):
        self.server_function = server_function

    def set_gripper_state(self, req, rsp):
        rsp = sim_gripper_service_pb2.SetGripperStateRsp()
        gripper_command = req.gripper_command
        is_right = req.is_right
        front_msg = "left"
        if is_right:
            front_msg = "right"
        opened_width = req.opened_width
        msg = self.server_function.blocking_start_server(
            data={
                "gripper_state": gripper_command,
                "is_gripper_right": is_right,
                "opened_width": opened_width,
            },
            Command=Command.SET_GRIPPER_STATE,
        )
        rsp.msg = front_msg + " " + msg
        return rsp


class ObservationService(sim_observation_service_pb2_grpc.SimObservationService):
    def __init__(self, server_function):
        self.server_function = server_function

    def SetCameraInfo(self, camera_info, width, height, ppx, ppy, fx, fy):
        camera_info.width = width
        camera_info.height = height
        camera_info.ppx = ppx
        camera_info.ppy = ppy
        camera_info.fx = fx
        camera_info.fy = fy

    def get_observation(self, req, rsp):
        rsp = sim_observation_service_pb2.GetObservationRsp()
        isCam = req.isCam
        isJoint = req.isJoint
        isPose = req.isPose
        isGripper = req.isGripper
        render_depth = req.CameraReq.render_depth
        render_semantic = req.CameraReq.render_semantic
        additional_cam_parameters = req.CameraReq.additional_parameters
        startRecording = req.startRecording
        stopRecording = req.stopRecording
        fps = req.fps
        task_name = req.task_name
        camera_prim_list = []
        for prim in req.CameraReq.camera_prim_list:
            camera_prim_list.append(prim)
        result = self.server_function.blocking_start_server(
            data={
                "startRecording": startRecording,
                "stopRecording": stopRecording,
                "isCam": isCam,
                "isJoint": isJoint,
                "isPose": isPose,
                "isGripper": isGripper,
                "camera_prim_list": camera_prim_list,
                "additional_cam_parameters": additional_cam_parameters,
                "render_depth": render_depth,
                "render_semantic": render_semantic,
                "fps": fps,
                "task_name": task_name,
            },
            Command=Command.GET_OBSERVATION,
        )
        if startRecording or stopRecording:
            rsp.recordingState = result
            return rsp

        if isPose:
            for _pose in result["object"]:
                object_rsp = sim_observation_service_pb2.ObjectRsp()
                position, rotation = _pose
                (
                    object_rsp.object_pose.position.x,
                    object_rsp.object_pose.position.y,
                    object_rsp.object_pose.position.z,
                ) = position
                (
                    object_rsp.object_pose.rpy.rw,
                    object_rsp.object_pose.rpy.rx,
                    object_rsp.object_pose.rpy.ry,
                    object_rsp.object_pose.rpy.rz,
                ) = rotation
                rsp.pose.append(object_rsp)
        if isCam:
            for _cam in result["camera"]:
                camera_data = sim_observation_service_pb2.CameraRsp()
                camera_info = _cam["camera_info"]
                self.SetCameraInfo(
                    camera_data.camera_info,
                    camera_info["width"],
                    camera_info["height"],
                    camera_info["ppx"],
                    camera_info["ppy"],
                    camera_info["fx"],
                    camera_info["fy"],
                )
                rgb_camera = _cam["rgb"]
                camera_data.rgb_camera.format = "rgb"
                if rgb_camera is not None:
                    camera_data.rgb_camera.data = rgb_camera.tobytes()
                depth_camera = _cam["depth"]
                camera_data.depth_camera.format = "depth"
                if depth_camera is not None:
                    camera_data.depth_camera.data = depth_camera.tobytes()

                if _cam["semantic"] is not None:
                    semantic_image = _cam["semantic"][0]
                    label_ids = _cam["semantic"][1]
                    camera_data.semantic_mask.name = "semantic"
                    if semantic_image is not None:
                        camera_data.semantic_mask.data = semantic_image.tobytes()
                    for key in label_ids:
                        label = sim_observation_service_pb2.SemanticDict()
                        label.label_name = key
                        label.label_id = label_ids[key]
                        camera_data.label_dict.append(label)
                rsp.camera.append(camera_data)
        if isJoint:
            joint_positions = result["joint"]
            for joint_name in joint_positions:
                joint_state = sim_observation_service_pb2.JointState()
                joint_state.name = joint_name
                joint_state.position = joint_positions[joint_name]
                rsp.joint.left_arm.append(joint_state)
        if isGripper:
            left_pose, left_rotation = result["gripper"]["left"]
            right_pose, right_rotation = result["gripper"]["right"]
            (
                rsp.gripper.right_gripper.position.x,
                rsp.gripper.right_gripper.position.y,
                rsp.gripper.right_gripper.position.z,
            ) = right_pose
            (
                rsp.gripper.right_gripper.rpy.rw,
                rsp.gripper.right_gripper.rpy.rx,
                rsp.gripper.right_gripper.rpy.ry,
                rsp.gripper.right_gripper.rpy.rz,
            ) = right_rotation

            (
                rsp.gripper.left_gripper.position.x,
                rsp.gripper.left_gripper.position.y,
                rsp.gripper.left_gripper.position.z,
            ) = left_pose
            (
                rsp.gripper.left_gripper.rpy.rw,
                rsp.gripper.left_gripper.rpy.rx,
                rsp.gripper.left_gripper.rpy.ry,
                rsp.gripper.left_gripper.rpy.rz,
            ) = left_rotation
        return rsp

    def store_current_state(self, req, rsp):
        rsp = sim_observation_service_pb2.StoreCurrentStateRsp()
        rsp.msg = self.server_function.blocking_start_server(
            data={"playback_id": req.playback_id}, Command=Command.STORE_CURRENT_STATE
        )
        return rsp

    def playback(self, req, rsp):
        rsp = sim_observation_service_pb2.PlaybackRsp()
        rsp.msg = self.server_function.blocking_start_server(
            data={"playback_id": req.playback_id}, Command=Command.PLAYBACK
        )
        return rsp

    def reset(self, req, rsp):
        rsp = sim_observation_service_pb2.ResetRsp()
        Reset = req.reset
        rsp.msg = self.server_function.blocking_start_server(
            data={"reset", Reset}, Command=Command.RESET
        )
        return rsp

    def attach_obj(self, req, rsp):
        rsp = sim_observation_service_pb2.AttachRsp()
        prim_paths = []
        for prim in req.obj_prims:
            prim_paths.append(prim)
        is_right = req.is_right
        rsp.msg = self.server_function.blocking_start_server(
            data={"obj_prim_paths": prim_paths, "is_right": is_right}, Command=Command.ATTACH_OBJ
        )
        return rsp

    def detach_obj(self, req, rsp):
        rsp = sim_observation_service_pb2.DetachRsp()
        detach = req.detach
        rsp.msg = self.server_function.blocking_start_server(
            data={"detach", detach}, Command=Command.DETACH_OBJ
        )
        return rsp

    def remove_objs_from_obstacle(self, req, rsp):
        rsp = sim_observation_service_pb2.RemoveObstacleRsp()
        prim_paths = []
        for prim in req.obj_prims:
            prim_paths.append(prim)
        rsp.msg = self.server_function.blocking_start_server(
            data={"obj_prim_paths": prim_paths}, Command=Command.REMOVE_OBJS_FROM_OBSTACLE
        )
        return rsp

    def task_status(self, req, rsp):
        rsp = sim_observation_service_pb2.TaskStatusRsp()
        isSuccess = req.isSuccess
        failStep = []
        for step in req.failStep:
            failStep.append(step)
        rsp.msg = self.server_function.blocking_start_server(
            data={"isSuccess": isSuccess, "failStep": failStep}, Command=Command.TASK_STATUS
        )
        return rsp

    def exit(self, req, rsp):
        rsp = sim_observation_service_pb2.ExitRsp()
        exit = req.exit
        rsp.msg = self.server_function.blocking_start_server(
            data={"exit": exit}, Command=Command.EXIT
        )
        return rsp

    def add_camera(self, req, rsp):
        rsp = sim_observation_service_pb2.AddCameraRsp()
        target_position = np.array(
            [
                req.camera_pose.position.x,
                req.camera_pose.position.y,
                req.camera_pose.position.z,
            ]
        )
        target_rotation = np.array(
            [
                req.camera_pose.rpy.rw,
                req.camera_pose.rpy.rx,
                req.camera_pose.rpy.ry,
                req.camera_pose.rpy.rz,
            ]
        )
        rsp.msg = self.server_function.blocking_start_server(
            data={
                "camera_prim": req.camera_prim,
                "camera_position": target_position,
                "camera_rotation": target_rotation,
                "focus_length": req.focus_length,
                "horizontal_aperture": req.horizontal_aperture,
                "vertical_aperture": req.vertical_aperture,
                "width": req.width,
                "height": req.height,
                "is_local": req.is_local,
            },
            Command=Command.ADD_CAMERA,
        )
        return rsp

    def init_robot(self, req, rsp):
        rsp = sim_observation_service_pb2.InitRobotRsp()
        target_position = np.array(
            [
                req.robot_pose.position.x,
                req.robot_pose.position.y,
                req.robot_pose.position.z,
            ]
        )
        target_rotation = np.array(
            [
                req.robot_pose.rpy.rw,
                req.robot_pose.rpy.rx,
                req.robot_pose.rpy.ry,
                req.robot_pose.rpy.rz,
            ]
        )
        joint_num = len(req.joint_cmd)
        init_joint_position = []
        init_joint_names = []
        for index in range(joint_num):
            v = req.joint_cmd[index].position
            name = req.joint_cmd[index].name
            if v is None:
                continue
            if not np.isfinite(v):
                continue
            init_joint_position.append(v)
            init_joint_names.append(name)
        rsp.msg = self.server_function.blocking_start_server(
            data={
                "robot_cfg_file": req.robot_cfg_file,
                "robot_usd_path": req.robot_usd_path,
                "scene_usd_path": req.scene_usd_path,
                "robot_position": target_position,
                "robot_rotation": target_rotation,
                "stand_type": req.stand_type,
                "stand_size_x": req.stand_size_x,
                "stand_size_y": req.stand_size_y,
                "init_joint_position": init_joint_position,
                "init_joint_names": init_joint_names,
            },
            Command=Command.INIT_ROBOT,
        )
        return rsp

    def set_object_pose(self, req, rsp):
        rsp = sim_observation_service_pb2.SetObjectPoseRsp()
        object_poses = []
        for object in req.object_pose:
            object_pose = {}
            object_pose["prim_path"] = object.prim_path
            object_pose["position"] = [
                object.pose.position.x,
                object.pose.position.y,
                object.pose.position.z,
            ]
            object_pose["rotation"] = [
                object.pose.rpy.rw,
                object.pose.rpy.rx,
                object.pose.rpy.ry,
                object.pose.rpy.rz,
            ]
            object_poses.append(object_pose)
        object_joints = []
        for joint in req.object_joint:
            object_joint = {}
            object_joint["prim_path"] = joint.prim_path
            object_joint["object_joint"] = []
            for joint in joint.joint_cmd:
                object_joint["object_joint"].append(joint.position)
            object_joints.append(object_joint)
        joint_num = len(req.joint_cmd)
        target_joint_position = np.zeros(joint_num)
        for index in range(joint_num):
            target_joint_position[index] = req.joint_cmd[index].position
        rsp.msg = self.server_function.blocking_start_server(
            data={
                "object_poses": object_poses,
                "joint_position": target_joint_position,
                "object_joints": object_joints,
            },
            Command=Command.SET_OBJECT_POSE,
        )
        return rsp

    def set_trajectory_list(self, req, rsp):
        rsp = sim_observation_service_pb2.SetTrajectoryListRsp()
        trajectory_list = []
        is_block = req.is_block
        for trajectory_point in req.trajectory_point:
            position = np.array(
                [
                    trajectory_point.position.x,
                    trajectory_point.position.y,
                    trajectory_point.position.z,
                ]
            )
            rotation = np.array(
                [
                    trajectory_point.rpy.rw,
                    trajectory_point.rpy.rx,
                    trajectory_point.rpy.ry,
                    trajectory_point.rpy.rz,
                ]
            )
            pose = position, rotation
            trajectory_list.append(pose)
        rsp.msg = self.server_function.blocking_start_server(
            data={"trajectory_list": trajectory_list, "is_block": is_block},
            Command=Command.SET_TRAJECTORY_LIST,
        )
        return rsp

    def set_frame_state(self, req, rsp):
        rsp = sim_observation_service_pb2.SetFrameStateRsp()
        frame_state = req.frame_state
        rsp.msg = self.server_function.blocking_start_server(
            data={"frame_state": frame_state}, Command=Command.SET_FRAME_STATE
        )
        return rsp

    def set_material(self, req, rsp):
        rsp = sim_observation_service_pb2.SetMaterialRsp()
        logger.info(req)
        materials = []
        for mat in req.materials:
            materials.append(
                {
                    "object_prim": mat.object_prim,
                    "material_name": mat.material_name,
                    "material_path": mat.material_path,
                    "label_name": mat.label_name,
                }
            )
        rsp.msg = self.server_function.blocking_start_server(
            data=materials, Command=Command.SET_MATERIAL
        )
        return rsp

    def set_light(self, req, rsp):
        rsp = sim_observation_service_pb2.SetLightRsp()
        lights = []
        for light in req.lights:
            lights.append(
                {
                    "light_type": light.light_type,
                    "light_prim": light.light_prim,
                    "light_temperature": light.light_temperature,
                    "light_intensity": light.light_intensity,
                    "light_rotation": [
                        light.light_rotation.rw,
                        light.light_rotation.rx,
                        light.light_rotation.ry,
                        light.light_rotation.rz,
                    ],
                    "light_texture": light.light_texture,
                }
            )
        rsp.msg = self.server_function.blocking_start_server(data=lights, Command=Command.SET_LIGHT)
        return rsp

    def set_task_metric(self, req, rsp):
        rsp = sim_observation_service_pb2.SetTaskMetricRsp()
        task_metric = req.metric
        rsp.msg = self.server_function.blocking_start_server(
            data={"task_metric": task_metric}, Command=Command.SET_TASK_METRIC
        )
        return rsp

    def get_checker_status(self, req, rsp):
        rsp = sim_observation_service_pb2.GetCheckerStatusRsp()
        rsp.msg = self.server_function.blocking_start_server(
            data={"checker": req.checker}, Command=Command.GET_CHECKER_STATUS
        )
        return rsp


class GrpcServer:
    def __init__(self, server_function):
        self.server_function = server_function

    def start(self):
        server_thread = threading.Thread(target=self.server)
        server_thread.start()

    def server(self):
        self._server = grpc.server(
            ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_send_message_length", 16094304),
                ("grpc.max_receive_message_length", 16094304),
            ],
        )
        rs2_camera_pb2_grpc.add_CameraServiceServicer_to_server(
            CameraService(self.server_function), self._server
        )
        sim_camera_service_pb2_grpc.add_SimCameraServiceServicer_to_server(
            SimCameraService(self.server_function), self._server
        )
        arm_pb2_grpc.add_ArmControlServiceServicer_to_server(
            armService(self.server_function), self._server
        )
        joint_channel_pb2_grpc.add_JointControlServiceServicer_to_server(
            JointService(self.server_function), self._server
        )
        sim_object_service_pb2_grpc.add_SimObjectServiceServicer_to_server(
            ObjectService(self.server_function), self._server
        )
        sim_gripper_service_pb2_grpc.add_SimGripperServiceServicer_to_server(
            GripperService(self.server_function), self._server
        )
        sim_observation_service_pb2_grpc.add_SimObservationServiceServicer_to_server(
            ObservationService(self.server_function), self._server
        )
        self.stop()
        self._server.add_insecure_port("0.0.0.0:50051")
        self._server.start()

    def stop(self):
        if self._server:
            self._server.stop(0)
