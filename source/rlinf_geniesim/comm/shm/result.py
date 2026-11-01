# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build a ResultBuffer whose fields are shm views (zero copy)."""
from __future__ import annotations
from rlinf_geniesim.comm.base import ResultBuffer
from rlinf_geniesim.comm.shm.layout import ShmRegion


def make_shm_result_buffer(ctrl: ShmRegion, frame: ShmRegion) -> ResultBuffer:
    return ResultBuffer(
        states=ctrl.field("states_buf"),
        joint_states=ctrl.field("joint_states_buf"),
        body_poses=ctrl.field("body_poses_buf"),
        object_poses=ctrl.field("object_poses_buf"),
        images=frame.field("rgb_buf"),
        rewards=ctrl.field("rewards_buf"),
        terminations=ctrl.field("dones_buf"),
        truncations=ctrl.field("truncated_buf"),
        sim_time=ctrl.field("sim_time_buf"),
        success=ctrl.field("success_buf"),
        global_tick=ctrl.field("global_tick"),
        frame_tick=frame.field("frame_tick"),
    )


__all__ = ["make_shm_result_buffer"]
