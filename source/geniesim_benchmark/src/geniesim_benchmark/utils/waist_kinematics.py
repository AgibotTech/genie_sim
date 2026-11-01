# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Forward kinematics of the base_link -> arm_base_link chain (the waist).

`arm_base_link` hangs off the end of the waist chain, so the transform between
it and `base_link` is a function of the waist joints — it changes whenever the
torso moves. The runtime measures that transform from the scene every
observation, which is correct for *observing* the EEF but one step stale for
*commanding* it: a step that drives the waist applies the arm and waist targets
together, so IK has to solve against the waist pose the step is moving to.

This module supplies the missing piece, FK over the waist chain, and
`ArmBaseTransformPredictor` combines it with the measured transform.

Kinematics come from the sim-facing ``app/robot_cfg/<cfg>/<cfg>.urdf``, the only
URDF whose joint names match the articulation the sim reports joint states for.
(The IK-SDK URDFs describe the same robots under different joint names and, for
G2, different axis signs — they are not interchangeable here.)

Deliberately free of Isaac / IK-SDK imports so the frame math stays testable
outside the container.
"""

import os
import threading
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()

_ROBOT_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "robot_cfg")

# robot_cfg keys whose URDF lives under a differently-named directory. Both 90d
# variants are the same body with different grippers, so they share a chain.
_URDF_DIR_ALIASES = {
    "G2_90d": "G2_90D",
    "G2_90d_gp": "G2_90D",
}

ROOT_LINK = "base_link"
TIP_LINK = "arm_base_link"

# Below this, a commanded waist counts as "not moving" and the measured
# transform is passed through untouched. Covers radians and metres alike.
_STILL_EPS = 1e-4

# How far FK may disagree with the measured transform before the URDF is judged
# not to describe this articulation.
_MODEL_POS_TOL_M = 0.01
_MODEL_ROT_TOL_RAD = np.deg2rad(2.0)

# Consecutive disagreements tolerated before prediction is switched off. A
# single bad sample can just be a reset transient; a run of them cannot.
_MODEL_MISMATCH_LIMIT = 3


def _floats(text, default):
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def _joint_spec(node):
    origin = node.find("origin")
    xyz = _floats(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    rpy = _floats(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))

    origin_mat = np.eye(4)
    # URDF rpy is fixed-axis roll/pitch/yaw, i.e. scipy's extrinsic "xyz".
    origin_mat[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    origin_mat[:3, 3] = xyz

    axis_node = node.find("axis")
    axis = _floats(axis_node.get("xyz") if axis_node is not None else None, (1.0, 0.0, 0.0))
    norm = float(np.linalg.norm(axis))
    axis = axis / norm if norm > 0.0 else np.array([1.0, 0.0, 0.0])

    return {
        "name": node.get("name"),
        "type": node.get("type"),
        "origin": origin_mat,
        "axis": axis,
    }


def _parse_chain(urdf_path, root_link=ROOT_LINK, tip_link=TIP_LINK):
    """Joint specs from `root_link` down to `tip_link`, root-first."""
    root = ET.parse(urdf_path).getroot()

    joint_by_child = {}
    for node in root.findall("joint"):
        child = node.find("child")
        if child is not None and node.find("parent") is not None:
            joint_by_child[child.get("link")] = node

    chain = []
    link = tip_link
    while link != root_link:
        node = joint_by_child.get(link)
        if node is None:
            raise ValueError(f"{tip_link} does not trace back to {root_link} in {urdf_path}")
        chain.append(_joint_spec(node))
        link = node.find("parent").get("link")
        if len(chain) > len(joint_by_child):
            raise ValueError(f"Cycle in the {root_link} -> {tip_link} chain in {urdf_path}")
    chain.reverse()
    return chain


class WaistChain:
    """FK over the fixed joint chain from base_link to arm_base_link."""

    def __init__(self, joints):
        self._joints = joints

    @property
    def actuated_joint_names(self):
        return [j["name"] for j in self._joints if j["type"] != "fixed"]

    def transform(self, joint_positions):
        """4x4 arm_base_link -> base_link transform at the given waist pose.

        Args:
            joint_positions: {joint_name: value}. Joints the caller doesn't
                mention are treated as being at zero, matching URDF convention.
        """
        mat = np.eye(4)
        for joint in self._joints:
            mat = mat @ joint["origin"]
            kind = joint["type"]
            if kind == "fixed":
                continue
            value = float(joint_positions.get(joint["name"], 0.0))
            step = np.eye(4)
            if kind == "prismatic":
                step[:3, 3] = joint["axis"] * value
            else:
                step[:3, :3] = R.from_rotvec(joint["axis"] * value).as_matrix()
            mat = mat @ step
        return mat


_CHAIN_CACHE = {}
_CHAIN_CACHE_LOCK = threading.Lock()


def urdf_path_for(robot_cfg):
    dir_name = _URDF_DIR_ALIASES.get(robot_cfg, robot_cfg)
    return os.path.normpath(os.path.join(_ROBOT_CFG_DIR, dir_name, f"{dir_name}.urdf"))


def get_waist_chain(robot_cfg):
    """Cached `WaistChain` for this robot, or None if its URDF isn't usable.

    Returning None rather than raising is deliberate: the chain is only needed
    to sharpen an already-working transform, so a robot without a parseable
    URDF should degrade to the measured transform, not fail to run.
    """
    with _CHAIN_CACHE_LOCK:
        if robot_cfg in _CHAIN_CACHE:
            return _CHAIN_CACHE[robot_cfg]

        chain = None
        path = urdf_path_for(robot_cfg)
        try:
            chain = WaistChain(_parse_chain(path))
        except (OSError, ET.ParseError, ValueError) as e:
            logger.warning(
                f"No waist chain for robot_cfg={robot_cfg!r} ({type(e).__name__}: {e}); "
                f"base_link EEF actions will be reframed with the measured transform only"
            )
        _CHAIN_CACHE[robot_cfg] = chain
        return chain


class ArmBaseTransformPredictor:
    """Transform to reframe a base_link EEF action with, given a waist command.

    The measured transform is the anchor: FK is only ever used for the *delta*
    between the measured waist pose and the commanded one. So a URDF that is
    slightly offset from the asset cancels out, and a step that doesn't move the
    waist reproduces today's behaviour exactly.

    FK is checked against the measured transform on every prediction. A URDF
    that disagrees is describing a different robot, and prediction switches off
    rather than trading a one-step lag for a systematic error.
    """

    def __init__(self, robot_cfg, waist_joint_names):
        self._waist_joint_names = list(waist_joint_names or [])
        self._chain = get_waist_chain(robot_cfg) if self._waist_joint_names else None
        self._enabled = self._chain is not None
        self._mismatch_streak = 0

    @property
    def enabled(self):
        return self._enabled

    def predict(self, measured_tf, measured_waist, commanded_waist):
        """4x4 transform at the commanded waist pose.

        Falls back to `measured_tf` whenever the prediction can't be trusted or
        isn't needed. Never raises.
        """
        if not self._enabled or measured_tf is None or commanded_waist is None:
            return measured_tf

        names = self._waist_joint_names
        if measured_waist is None or len(measured_waist) < len(names):
            return measured_tf

        measured = {name: float(v) for name, v in zip(names, measured_waist)}
        target = dict(measured)
        # PiEnv.step pairs command values with waist joints 1:1 and clamps to
        # the shorter list; mirror that so the prediction describes the pose
        # that actually gets commanded.
        for name, value in zip(names, commanded_waist):
            target[name] = float(value)

        if all(abs(target[n] - measured[n]) < _STILL_EPS for n in names):
            return measured_tf

        fk_measured = self._chain.transform(measured)
        if not self._model_agrees(fk_measured, measured_tf):
            return measured_tf

        return measured_tf @ np.linalg.inv(fk_measured) @ self._chain.transform(target)

    def _model_agrees(self, fk_tf, measured_tf):
        delta = np.linalg.inv(fk_tf) @ measured_tf
        pos_err = float(np.linalg.norm(delta[:3, 3]))
        rot_err = float(np.linalg.norm(R.from_matrix(delta[:3, :3]).as_rotvec()))

        if pos_err <= _MODEL_POS_TOL_M and rot_err <= _MODEL_ROT_TOL_RAD:
            self._mismatch_streak = 0
            return True

        self._mismatch_streak += 1
        if self._mismatch_streak >= _MODEL_MISMATCH_LIMIT:
            self._enabled = False
            logger.warning(
                f"Waist FK disagrees with the measured arm_base_link transform "
                f"(pos {pos_err * 1000:.1f} mm, rot {np.rad2deg(rot_err):.2f} deg) — the URDF does "
                f"not describe this articulation. Falling back to the measured transform, so "
                f"base_link EEF actions lag the waist by one step while it turns."
            )
        return False
