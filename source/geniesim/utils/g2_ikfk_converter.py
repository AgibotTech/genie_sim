#!/usr/bin/env python
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from geniesim.utils.ikfk_utils import (  # type: ignore
    IKFKSolver,
    mat2xyzrpy,
    xyzquat_to_xyzrpy,
    xyzrpy_to_xyzquat,
)


def parse_floats(s: str, expected_len: int) -> List[float]:
    values = [float(x) for x in s.replace(" ", "").split(",") if x != ""]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} floats, got {len(values)}: {values}")
    return values


def joints_to_eef(
    solver: IKFKSolver,
    left_joints: List[float],
    right_joints: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given left and right 7-DOF arm joints, return end-effector poses
    relative to arm_base_link as xyzrpy.
    Uses ik_solver.Solver.compute_fk(joints) to compute forward kinematics.
    """

    left_joints_arr = np.asarray(left_joints, dtype=np.float32)
    right_joints_arr = np.asarray(right_joints, dtype=np.float32)

    left_fk_raw = solver.left_solver.compute_fk(left_joints_arr)
    right_fk_raw = solver.right_solver.compute_fk(right_joints_arr)

    left_mat = np.asarray(left_fk_raw, dtype=np.float32).reshape(4, 4)
    right_mat = np.asarray(right_fk_raw, dtype=np.float32).reshape(4, 4)

    left_xyzrpy = mat2xyzrpy(left_mat)
    right_xyzrpy = mat2xyzrpy(right_mat)

    return left_xyzrpy, right_xyzrpy


def eef_to_joints(
    solver: IKFKSolver,
    left_eef_xyzrpy: List[float],
    right_eef_xyzrpy: List[float],
    arm_joint_init: List[float],
) -> Tuple[List[float], List[float]]:
    """
    Use IK to convert left and right end-effector xyzrpy poses
    to corresponding 7-DOF arm joint values.
    """
    solver.left_solver.sync_target_with_joints(arm_joint_init[:7])
    solver.right_solver.sync_target_with_joints(arm_joint_init[7:14])

    left_arr = np.array(left_eef_xyzrpy, dtype=np.float32)
    right_arr = np.array(right_eef_xyzrpy, dtype=np.float32)

    left_target = xyzrpy_to_xyzquat(left_arr)
    right_target = xyzrpy_to_xyzquat(right_arr)

    solver.left_solver.update_target_quat(
        target_pos=left_target[:3],
        target_quat=left_target[3:],
    )
    solver.right_solver.update_target_quat(
        target_pos=right_target[:3],
        target_quat=right_target[3:],
    )

    left_joints = solver.left_solver.solve()
    right_joints = solver.right_solver.solve()

    return left_joints.tolist(), right_joints.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="G2 dual-arm IK/FK tool: joint <-> EEF, relative to arm_base_link.",
        epilog=(
            "Examples:\n"
            "  FK with RPY output:\n"
            "    /isaac-sim/python.sh g2_ikfk_converter.py \\\n"
            "      --mode fk \\\n"
            "      --eef-type rpy \\\n"
            "      --left-joints 0,0,0,0,0,0,0 \\\n"
            "      --right-joints 0,0,0,0,0,0,0 \\\n"
            "      --robot-cfg G2\n"
            "\n"
            "  FK with quaternion (xyz+xyzw) output:\n"
            "    /isaac-sim/python.sh g2_ikfk_converter.py \\\n"
            "      --mode fk \\\n"
            "      --eef-type quat \\\n"
            "      --left-joints 0,0,0,0,0,0,0 \\\n"
            "      --right-joints 0,0,0,0,0,0,0 \\\n"
            "      --robot-cfg G2\n"
            "\n"
            "  IK from RPY EEF poses:\n"
            "    /isaac-sim/python.sh g2_ikfk_converter.py \\\n"
            "      --mode ik \\\n"
            "      --eef-type rpy \\\n"
            "      --left-eef 0.3,0.2,0.5,0.0,1.57,0.0 \\\n"
            "      --right-eef 0.3,-0.2,0.5,0.0,1.57,0.0 \\\n"
            "      --left-joints 0,0,0,0,0,0,0 \\\n"
            "      --right-joints 0,0,0,0,0,0,0 \\\n"
            "      --robot-cfg G2\n"
            "\n"
            "  IK from quaternion (xyz+xyzw) EEF poses:\n"
            "    /isaac-sim/python.sh g2_ikfk_converter.py \\\n"
            "      --mode ik \\\n"
            "      --eef-type quat \\\n"
            "      --left-eef 0.3,0.2,0.5,0,0,0,1 \\\n"
            "      --right-eef 0.3,-0.2,0.5,0,0,0,1 \\\n"
            "      --left-joints 0,0,0,0,0,0,0 \\\n"
            "      --right-joints 0,0,0,0,0,0,0 \\\n"
            "      --robot-cfg G2\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["fk", "ik"],
        required=True,
        help="fk: compute EEF from joints; ik: compute joints from EEF.",
    )
    parser.add_argument(
        "--left-joints",
        type=str,
        help="Left arm 7 joint values, comma-separated, e.g. 0.0,0.1,... (used for mode=fk or as IK initial guess).",
    )
    parser.add_argument(
        "--right-joints",
        type=str,
        help="Right arm 7 joint values, comma-separated, e.g. 0.0,0.1,... (used for mode=fk or as IK initial guess).",
    )
    parser.add_argument(
        "--left-eef",
        type=str,
        help="Left EEF pose, format depends on --eef-type; rpy: x,y,z,rx,ry,rz; quat: x,y,z,qx,qy,qz,qw.",
    )
    parser.add_argument(
        "--right-eef",
        type=str,
        help="Right EEF pose, format depends on --eef-type; rpy: x,y,z,rx,ry,rz; quat: x,y,z,qx,qy,qz,qw.",
    )
    parser.add_argument(
        "--robot-cfg",
        type=str,
        default="G2",
        help="robot_cfg passed into IKFKSolver (must contain 'G2', default 'G2').",
    )
    parser.add_argument(
        "--eef-type",
        choices=["rpy", "quat"],
        default="rpy",
        help="EEF representation type: rpy (xyz+roll,pitch,yaw) or quat (xyz+xyzw quaternion).",
    )

    args = parser.parse_args()

    # Construct IKFKSolver, arm_init_joint_position requires 14 values
    if args.left_joints and args.right_joints:
        left_init = parse_floats(args.left_joints, 7)
        right_init = parse_floats(args.right_joints, 7)
        arm_init = left_init + right_init
    else:
        arm_init = [0.0] * 14

    head_init = [0.0, 0.0, 0.0]
    waist_init = [0.0, 0.0, 0.0]

    solver = IKFKSolver(
        arm_init_joint_position=arm_init,
        head_init_position=head_init,
        waist_init_position=waist_init,
        robot_cfg=args.robot_cfg,
    )

    if args.mode == "fk":
        if not args.left_joints or not args.right_joints:
            raise ValueError("mode=fk must provide --left-joints and --right-joints")

        left_joints = parse_floats(args.left_joints, 7)
        right_joints = parse_floats(args.right_joints, 7)

        left_xyzrpy, right_xyzrpy = joints_to_eef(solver, left_joints, right_joints)

        if args.eef_type == "rpy":
            print("Left EEF xyzrpy:", left_xyzrpy.tolist())
            print("Right EEF xyzrpy:", right_xyzrpy.tolist())
        else:
            # Output xyz + xyzw quaternion (API uses xyzw)
            left_rot = R.from_euler("xyz", left_xyzrpy[3:6]).as_matrix()
            right_rot = R.from_euler("xyz", right_xyzrpy[3:6]).as_matrix()

            left_xyz = left_xyzrpy[0:3]
            right_xyz = right_xyzrpy[0:3]
            left_quat_xyzw = R.from_matrix(left_rot).as_quat(scalar_first=False)
            right_quat_xyzw = R.from_matrix(right_rot).as_quat(scalar_first=False)

            left_xyzquat = np.concatenate([left_xyz, left_quat_xyzw])
            right_xyzquat = np.concatenate([right_xyz, right_quat_xyzw])

            print("Left EEF xyzquat (xyzw):", left_xyzquat.tolist())
            print("Right EEF xyzquat (xyzw):", right_xyzquat.tolist())

    elif args.mode == "ik":
        if not args.left_eef or not args.right_eef or not args.left_joints or not args.right_joints:
            raise ValueError("mode=ik must provide --left-eef, --right-eef, --left-joints, and --right-joints")

        if args.eef_type == "rpy":
            left_eef = parse_floats(args.left_eef, 6)
            right_eef = parse_floats(args.right_eef, 6)
        else:
            # Input is xyz + xyzw; convert to xyz + wxyz before using xyzquat_to_xyzrpy
            left_vals = parse_floats(args.left_eef, 7)
            right_vals = parse_floats(args.right_eef, 7)

            left_xyz = left_vals[0:3]
            left_xyzw = left_vals[3:7]
            right_xyz = right_vals[0:3]
            right_xyzw = right_vals[3:7]

            # xyzw -> wxyz
            left_wxyz = [left_xyzw[3]] + left_xyzw[0:3]
            right_wxyz = [right_xyzw[3]] + right_xyzw[0:3]

            left_xyzquat_wxyz = np.array(left_xyz + left_wxyz, dtype=np.float32)
            right_xyzquat_wxyz = np.array(right_xyz + right_wxyz, dtype=np.float32)

            left_eef = xyzquat_to_xyzrpy(left_xyzquat_wxyz).tolist()
            right_eef = xyzquat_to_xyzrpy(right_xyzquat_wxyz).tolist()

        left_joints, right_joints = eef_to_joints(solver, left_eef, right_eef, arm_init)

        print("Left joints:", left_joints)
        print("Right joints:", right_joints)


if __name__ == "__main__":
    main()
