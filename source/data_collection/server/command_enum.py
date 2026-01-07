# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Command enumeration for gRPC server and command controller

from enum import IntEnum


class Command(IntEnum):
    """Command enumeration for gRPC server communication"""

    # Camera commands
    GET_CAMERA_DATA = 1
    GET_SEMANTIC_DATA = 1  # Same as GET_CAMERA_DATA

    # Motion commands
    LINEAR_MOVE = 2
    SET_JOINT_POSITION = 3
    GET_JOINT_POSITION = 8
    GET_EE_POSE = 18
    GET_IK_STATUS = 19
    SET_TRAJECTORY_LIST = 25

    # Gripper commands
    GET_GRIPPER_STATE = 4
    SET_GRIPPER_STATE = 9

    # Object commands
    GET_OBJECT_POSE = 5
    ADD_OBJECT = 6
    GET_ROBOT_LINK_POSE = 7
    GET_OBJECT_JOINT = 26
    GET_PART_DOF_JOINT = 32
    SET_OBJECT_POSE = 24
    SET_TARGET_POINT = 27
    SET_LINEAR_VELOCITY = 33
    ATTACH_OBJ = 13
    DETACH_OBJ = 14
    ATTACH_OBJ_TO_PARENT = 50
    DETACH_OBJ_FROM_PARENT = 51
    REMOVE_OBJS_FROM_OBSTACLE = 52

    # Observation and recording commands
    GET_OBSERVATION = 11
    START_RECORDING = 11  # Same as GET_OBSERVATION with startRecording flag
    STOP_RECORDING = 11  # Same as GET_OBSERVATION with stopRecording flag

    # System commands
    RESET = 12
    EXIT = 17
    INIT_ROBOT = 21
    TASK_STATUS = 16

    # Camera setup commands
    ADD_CAMERA = 22

    # State and configuration commands
    SET_FRAME_STATE = 28
    SET_LIGHT = 30
    SET_CODE_FACE_ORIENTATION = 34
    SET_TASK_METRIC = 53

    # Replay related
    STORE_CURRENT_STATE = 54
    PLAYBACK = 55

    # Checker related
    GET_CHECKER_STATUS = 56


# int to string
command_value_to_string = {}
for k, v in Command.__members__.items():
    if v not in command_value_to_string:
        command_value_to_string[v] = k
    else:
        command_value_to_string[v] = f"{command_value_to_string[v]}, {k}"
