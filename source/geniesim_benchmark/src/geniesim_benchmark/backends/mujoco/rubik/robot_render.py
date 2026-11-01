# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Forward kinematics for the standard USD robot, without PhysX simulation."""

import numpy as np


def rotation_matrix(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def joint_frame(position, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix(quaternion)
    matrix[:3, 3] = position
    return matrix


def native_to_usd_joint(name):
    if name.startswith("Joint"):
        joint, side = name.split("_")
        number = joint.removeprefix("Joint")
        return f"idx{2 if side=='L' else 6}{number}_arm_{side.lower()}_joint{number}"
    return "hand_" + name.replace("_finger", "")


class RobotPoseTree:
    def __init__(self, joints):
        self.joints = joints

    def evaluate(self, positions, base_pose):
        pending = list(self.joints)
        poses = {}
        while pending:
            progress = False
            for joint in pending[:]:
                parent, child = joint["parent"], joint["child"]
                if parent and parent not in poses:
                    continue
                if child.endswith("/base_link"):
                    matrix = np.asarray(base_pose, dtype=float)
                else:
                    motion = np.eye(4)
                    if joint["kind"] == "PhysicsRevoluteJoint":
                        angle = float(positions.get(joint["name"], 0.0))
                        axis = "XYZ".index(joint["axis"])
                        c, s = np.cos(angle), np.sin(angle)
                        if axis == 0:
                            motion[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
                        elif axis == 1:
                            motion[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
                        else:
                            motion[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
                    elif joint["kind"] != "PhysicsFixedJoint":
                        raise ValueError("Unsupported render joint: " + joint["kind"])
                    matrix = (
                        poses.get(parent, np.eye(4))
                        @ np.asarray(joint["frame0"])
                        @ motion
                        @ np.linalg.inv(joint["frame1"])
                    )
                poses[child] = matrix
                pending.remove(joint)
                progress = True
            if not progress:
                raise ValueError("USD robot joint graph is disconnected or cyclic")
        return poses
