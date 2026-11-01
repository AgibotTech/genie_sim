# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""ROS-free extract of geniesim_teleop's Pico signal path (VRServer UDP +
parse_pico_command math). Debug-only. coef_pos=coef_rot=1.0 per teleop.py."""

from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

_L_XFORM = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], dtype=float)
_R_XFORM = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], dtype=float)  # right "else" branch


def _remap_pos(p):
    return np.array([p["z"], -p["x"], p["y"]], dtype=float)


def _remap_quat_xyzw(rot_dict, xform):
    q = np.array([rot_dict["x"], rot_dict["y"], rot_dict["z"], rot_dict["w"]], dtype=float)
    m = R.from_quat(q).as_matrix()
    return R.from_matrix(xform @ m @ xform.T).as_quat()  # xyzw


def parse_pico_frame(content) -> dict:
    l_sig, r_sig = content
    out = {}
    out["left"] = np.concatenate(
        [_remap_pos(l_sig["position"]), _remap_quat_xyzw(l_sig["rotation"], _L_XFORM)]
    ).tolist()
    out["right"] = np.concatenate(
        [_remap_pos(r_sig["position"]), _remap_quat_xyzw(r_sig["rotation"], _R_XFORM)]
    ).tolist()
    out["l_grip"] = 1.0 - float(l_sig["indexTrig"])
    out["r_grip"] = 1.0 - float(r_sig["indexTrig"])
    out["l_on"] = float(l_sig["handTrig"]) > 0.8
    out["r_on"] = float(r_sig["handTrig"]) > 0.8
    out["l_reset"] = str(l_sig.get("keyOne", "false")) == "true"
    # debug passthrough: raw trigger values as received, before any mapping
    out["_raw_trig"] = (l_sig.get("indexTrig"), r_sig.get("indexTrig"), l_sig.get("handTrig"), r_sig.get("handTrig"))
    return out


class PicoTeleopSource:
    def __init__(self, host: str, port: int):
        from geniesim_teleop.utils.vr_server import VRServer  # ROS-free

        self._vr = VRServer(host=host, port=port)
        self.frames = 0  # debug: count of polls that returned parsed data

    def poll(self) -> Optional[dict]:
        content = self._vr.on_update()
        if not content:
            return None
        self.frames += 1
        return parse_pico_frame(content)

    @property
    def udp_alive(self) -> bool:
        """True once the polling loop has received Pico input."""
        return self.frames > 0

    def close(self):
        self._vr.close()
