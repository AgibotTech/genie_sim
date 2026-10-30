# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Geometry helpers for recorded arm and effector poses."""

import logging
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_link_chain_cache: dict = {}


# Omnipicker centre: arm-end local z offset, unchanged orientation.
GRIPPER_CENTER_OFFSET_Z = {"omnipicker": 0.1431}


def effector_from_arm_end(arm_pos, arm_quat_xyzw, gripper_model):
    """Derive effector pose using a local z offset; unknown models use omnipicker.

    Use effector_from_arm_end_ctek90d for the 90D rigid transform."""
    z = GRIPPER_CENTER_OFFSET_Z.get(gripper_model, GRIPPER_CENTER_OFFSET_Z["omnipicker"])
    arm_pos = np.asarray(arm_pos, dtype=np.float64)
    quat = np.asarray(arm_quat_xyzw, dtype=np.float64)
    pos = arm_pos + R.from_quat(quat).as_matrix() @ np.array([0.0, 0.0, z])
    return pos.astype(np.float64), quat.astype(np.float64)


# ctek90d arm-end to gripper-centre transform.
# Measured from G2_crsB_90d/robot_fix.usda at rest: +0.22922 m along the
# arm-end local z axis, plus +90 degrees on the left and -90 on the right.
_CTEK90D_CENTER_TZ = 0.22922
_SQRT_HALF = float(np.sqrt(0.5))
# Relative xyzw quaternion about z: left +90 degrees, right -90 degrees.
_CTEK90D_CENTER_QREL = {
    "left": np.array([0.0, 0.0, _SQRT_HALF, _SQRT_HALF], dtype=np.float64),
    "right": np.array([0.0, 0.0, -_SQRT_HALF, _SQRT_HALF], dtype=np.float64),
}


def effector_from_arm_end_ctek90d(arm_pos, arm_quat_xyzw, side):
    """Derive 90D gripper-centre pose using the measured rigid transform.

    T_center = T_arm_end @ T_rel, with a 0.22922 m z offset and side-specific
    90-degree rotation. ``side`` accepts left/right or an l/r prefix.
    """
    arm_pos = np.asarray(arm_pos, dtype=np.float64)
    R_arm = R.from_quat(np.asarray(arm_quat_xyzw, dtype=np.float64))
    key = "right" if str(side).lower().startswith("r") else "left"
    pos = arm_pos + R_arm.as_matrix() @ np.array([0.0, 0.0, _CTEK90D_CENTER_TZ])
    quat = (R_arm * R.from_quat(_CTEK90D_CENTER_QREL[key])).as_quat()
    return pos.astype(np.float64), quat.astype(np.float64)


def effector_in_base(T_world_base, T_world_eec):
    """Transform ee_center from world coordinates into base_link coordinates."""
    T_world_base = np.asarray(T_world_base, dtype=np.float64)
    T_world_eec = np.asarray(T_world_eec, dtype=np.float64)
    T = np.linalg.inv(T_world_base) @ T_world_eec
    pos = T[:3, 3].astype(np.float64)
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat().astype(np.float64)
    return pos, quat_xyzw


def load_link_chain(urdf_path, base_link, target_link):
    """Return the ordered URDF joint chain from base_link to target_link.

    Each joint includes name, type, rpy, and axis. Return None if no chain
    exists; cache results by URDF path and link pair.
    """
    key = (urdf_path, base_link, target_link)
    if key in _link_chain_cache:
        return _link_chain_cache[key]
    chain = None
    try:
        root = ET.parse(urdf_path).getroot()
        child2joint = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            ax = j.find("axis")
            child2joint[j.find("child").get("link")] = {
                "name": j.get("name"),
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "xyz": [float(x) for x in o.get("xyz", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0],
                "rpy": [float(x) for x in o.get("rpy", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0],
                "axis": [float(x) for x in ax.get("xyz").split()] if ax is not None else [0.0, 0.0, 1.0],
            }
        walk, link, seen = [], target_link, set()
        while link != base_link:
            if link not in child2joint or link in seen:
                walk = None
                break
            seen.add(link)
            walk.append(child2joint[link])
            link = child2joint[link]["parent"]
        if walk is not None:
            walk.reverse()
            chain = walk
    except Exception as e:
        logging.warning("URDF chain parse failed (%s -> %s): %s", base_link, target_link, e)
    _link_chain_cache[key] = chain
    return chain


def chain_transform(chain, angle_by_name):
    """Run forward kinematics along a loaded chain and return a 4x4 transform.

    Revolute joints include their axis rotation; fixed joints use only their
    origin transform. Missing joint angles default to zero; an empty chain
    returns the identity transform.
    """
    T = np.eye(4)
    if not chain:
        return T
    for jt in chain:
        O = np.eye(4)
        O[:3, :3] = R.from_euler("xyz", jt["rpy"]).as_matrix()
        O[:3, 3] = jt.get("xyz", [0.0, 0.0, 0.0])
        T = T @ O
        if jt["type"] in ("revolute", "continuous"):
            q = float(angle_by_name.get(jt["name"], 0.0))
            J = np.eye(4)
            J[:3, :3] = R.from_rotvec(np.asarray(jt["axis"], dtype=np.float64) * q).as_matrix()
            T = T @ J
    return T


def link3_orientation(arm_joints_by_name, side, urdf_path, base_link="arm_base_link"):
    """Return arm_{l,r}_link3 orientation in arm_base_link coordinates.

    Joint names must match the URDF. Missing or invalid chains return the
    identity quaternion in xyzw order.
    """
    if not urdf_path:
        return _IDENTITY_QUAT.copy()
    target = "arm_l_link3" if side == "left" else "arm_r_link3"
    chain = load_link_chain(urdf_path, base_link, target)
    if not chain:
        return _IDENTITY_QUAT.copy()
    try:
        rot = np.eye(3)
        for jt in chain:
            rot = rot @ R.from_euler("xyz", jt["rpy"]).as_matrix()
            if jt["type"] in ("revolute", "continuous"):
                q = float(arm_joints_by_name.get(jt["name"], 0.0))
                rot = rot @ R.from_rotvec(np.asarray(jt["axis"]) * q).as_matrix()
        return R.from_matrix(rot).as_quat().astype(np.float64)
    except Exception as e:
        logging.warning("link3 orientation fallback (%s): %s", side, e)
        return _IDENTITY_QUAT.copy()


def assemble_end_pose(arm_pos, arm_ori, eff_pos, eff_ori):
    """Assemble the 28-value end pose from arm and effector poses.

    arm_pos/eff_pos: [2,3]; arm_ori/eff_ori: [2,4] (xyzw).
    """
    arm_pos = np.asarray(arm_pos, dtype=np.float64)
    arm_ori = np.asarray(arm_ori, dtype=np.float64)
    eff_pos = np.asarray(eff_pos, dtype=np.float64)
    eff_ori = np.asarray(eff_ori, dtype=np.float64)
    return np.concatenate(
        [
            arm_pos[0],
            arm_ori[0],
            arm_pos[1],
            arm_ori[1],
            eff_pos[0],
            eff_ori[0],
            eff_pos[1],
            eff_ori[1],
        ]
    ).astype(np.float64)
