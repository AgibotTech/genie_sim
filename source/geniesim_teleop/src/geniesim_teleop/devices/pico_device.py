#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
from .teleop_device import TeleopDevice
from geniesim_teleop.utils.vr_server import VRServer
import numpy as np
from geniesim_teleop.utils.logger import logger
from scipy.spatial.transform import Rotation as R
import math


def scale_quat(q, s):
    return q * s


def round_v(val):
    PRECISION = 3
    return [round(v, PRECISION) for v in val]


def relative_controller_pose(current_pose, anchor_pose):
    """Return the controller motion since the clutch was engaged."""
    current = np.asarray(current_pose, dtype=float)
    anchor = np.asarray(anchor_pose, dtype=float)
    if current.shape != (7,) or anchor.shape != (7,):
        raise ValueError("controller poses must contain xyz + xyzw")

    delta_pos = current[:3] - anchor[:3]
    delta_rot = R.from_quat(current[3:]) * R.from_quat(anchor[3:]).inv()
    return np.concatenate((delta_pos, delta_rot.as_quat()))


class ControllerPoseClutch:
    def __init__(self):
        self._anchor = None

    def update(self, current_pose, active, reanchor=False):
        if not active:
            self._anchor = None
            return None
        if current_pose is None:
            return None
        if reanchor or self._anchor is None:
            self._anchor = np.asarray(current_pose, dtype=float).copy()
        return relative_controller_pose(current_pose, self._anchor)


class PicoDevice(TeleopDevice):
    def __init__(self, host_ip=None, port=8080, robot_cfg="G2_omnipicker", input_timeout_s=0.25):
        self.host_ip = host_ip
        self.port = port
        self.robot_cfg = robot_cfg
        self.input_timeout_s = input_timeout_s
        self.output = {"left": None, "right": None}
        self.coef_pos = 1.0
        self.coef_rot = 1.0

    def initialize(self):
        self.vr_server = VRServer(host=self.host_ip, port=self.port, input_timeout_s=self.input_timeout_s)

    def update(self, debug=False):
        self.content = self.vr_server.on_update()
        if self.content:
            try:
                self.validate_packet(self.content)
                self.pico_command = self.parse_pico_command(robot_type=self.robot_cfg)
            except (KeyError, TypeError, ValueError):
                return {}
            dpos_l = self.coef_pos * self.pico_command["l"]["position"]
            dpos_r = self.coef_pos * self.pico_command["r"]["position"]
            drot_l = scale_quat(self.pico_command["l"]["quaternion"], self.coef_rot)
            drot_r = scale_quat(self.pico_command["r"]["quaternion"], self.coef_rot)

            self.output["left"] = list(round_v(dpos_l)) + list(round_v(drot_l))
            self.output["right"] = list(round_v(dpos_r)) + list(round_v(drot_r))
            self.output["l_eef"] = self.pico_command["l"]["gripper"]
            self.output["r_eef"] = self.pico_command["r"]["gripper"]
            self.output["l_on"] = self.pico_command["l"]["On"]
            self.output["r_on"] = self.pico_command["r"]["On"]
            self.output["r_b"] = self.pico_command["r"]["resetbh"]
            self.output["r_a"] = self.pico_command["r"]["reset"]
            self.output["l_x"] = self.pico_command["l"]["reset"]

            self.output["r_axisX"] = self.pico_command["r"]["axisX"]
            self.output["r_axisY"] = self.pico_command["r"]["axisY"]

            self.output["l_axisX"] = self.pico_command["l"]["axisX"]
            self.output["l_axisY"] = self.pico_command["l"]["axisY"]

            if debug:
                logger.debug(f"[PICO]\n {self.output['left']},\n, {self.output['right']}")
            return self.output
        else:
            return {}

    def close(self):
        if hasattr(self, "vr_server"):
            self.vr_server.close()

    @staticmethod
    def validate_packet(content):
        if not isinstance(content, list) or len(content) != 2:
            raise ValueError("Pico packet must contain two controllers")
        for hand in content:
            values = [hand["position"][axis] for axis in "xyz"]
            quat = [hand["rotation"][axis] for axis in "xyzw"]
            values += quat + [hand[key] for key in ("axisX", "axisY", "indexTrig", "handTrig")]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError("Pico pose and axes must be finite numbers")
            if np.linalg.norm(quat) < 1e-6:
                raise ValueError("Pico quaternion must be nonzero")
            for key in ("axisClick", "keyOne", "keyTwo"):
                if hand[key] not in (True, False, "true", "false"):
                    raise ValueError("Invalid Pico button")

    def reset(self):
        self.output = {"left": None, "right": None}

    def parse_pico_command(self, robot_type="G2_omnipicker"):
        """
        Args:
            content (list(dict)): parse pico control command to local frame
        """
        [l_sig, r_sig] = self.content
        ret = {"l": {}, "r": {}}
        ret["l"]["position"] = np.array([l_sig["position"]["z"], -l_sig["position"]["x"], l_sig["position"]["y"]])
        l_original_quat = np.array(
            [
                l_sig["rotation"]["x"],
                l_sig["rotation"]["y"],
                l_sig["rotation"]["z"],
                l_sig["rotation"]["w"],
            ]
        )
        l_original_rot = R.from_quat(l_original_quat)
        l_original_matrix = l_original_rot.as_matrix()
        l_transform_matrix = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
        l_transform_inv = l_transform_matrix.T
        l_new_matrix = l_transform_matrix @ l_original_matrix @ l_transform_inv
        l_new_rot = R.from_matrix(l_new_matrix)
        ret["l"]["quaternion"] = l_new_rot.as_quat()

        ret["l"]["axisMode"] = "reset" if l_sig["axisClick"] in (True, "true") else "move"
        ret["l"]["axisX"] = l_sig["axisX"]
        ret["l"]["axisY"] = l_sig["axisY"]
        ret["l"]["gripper"] = 1 - l_sig["indexTrig"]
        ret["l"]["On"] = l_sig["handTrig"] > 0.8
        ret["l"]["reset"] = l_sig["keyOne"] in (True, "true")
        ret["l"]["against"] = l_sig["keyTwo"] in (True, "true")

        # Pose coordinates are independent of buttons; A routes this pose
        # to the waist in TeleOp, while B only starts recording.
        ret["r"]["position"] = np.array([r_sig["position"]["z"], -r_sig["position"]["x"], r_sig["position"]["y"]])
        r_original_quat = np.array(
            [
                r_sig["rotation"]["x"],
                r_sig["rotation"]["y"],
                r_sig["rotation"]["z"],
                r_sig["rotation"]["w"],
            ]
        )
        r_original_rot = R.from_quat(r_original_quat)
        r_original_matrix = r_original_rot.as_matrix()
        r_transform_matrix = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
        r_transform_inv = r_transform_matrix.T
        r_new_matrix = r_transform_matrix @ r_original_matrix @ r_transform_inv
        r_new_rot = R.from_matrix(r_new_matrix)
        ret["r"]["quaternion"] = r_new_rot.as_quat()

        ret["r"]["axisX"] = r_sig["axisX"]
        ret["r"]["axisY"] = r_sig["axisY"]
        ret["r"]["axisMode"] = "head" if r_sig["axisClick"] in (True, "true") else "waist"
        ret["r"]["gripper"] = 1 - r_sig["indexTrig"]
        ret["r"]["On"] = r_sig["handTrig"] > 0.8
        ret["r"]["reset"] = r_sig["keyOne"] in (True, "true")
        ret["r"]["resetbh"] = r_sig["keyTwo"] in (True, "true")

        return ret

    def extra_l(self):
        if hasattr(self, "pico_command"):
            return self.pico_command["l"]["against"]
        else:
            return False

    def extra_r(self):
        if hasattr(self, "pico_command"):
            return self.pico_command["r"]["resetbh"]
        else:
            return False
