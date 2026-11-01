# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import json
import time

from .hook_base import HookBase
from geniesim_benchmark.plugins.logger import Logger
import geniesim_benchmark.utils.system_utils as system_utils

logger = Logger()

# Place tasks whose target object starts already held in a gripper. scene.usda
# pre-positions the block (kinematic, so it does not fall during reset) at the
# gripper center. This hook forms a real friction grasp (make the block dynamic
# but weightless, close the gripper onto it, restore gravity), then keeps the
# holding gripper forced closed every step so the block is held reliably --
# until the policy's own gripper command shows an opening trend, at which point
# control is handed back and the block drops naturally. No constraint to break.
PREGRASP_TASKS = set()
# Joint DOF index into cfg["gripper_joints"] ([left, right]).
_HOLD_IDX = {"left_gripper": 0, "right_gripper": 1}
# GP_90D gripper hold joint range is [-0.909 (closed), 0.0 (open)].
_CLOSED = -0.9
_CLOSE_STEPS = 15  # incremental drive targets 0 -> _CLOSED (slow, gentle grip)
_CLOSE_DT = 0.08
# Release once the policy commands the holding gripper this much more open than
# the tightest it has held (an opening trend), so a real place lets go.
_OPEN_DELTA = 0.35


def _load_pregrasp_spec(env):
    """Return {"object": name, "gripper": side} for this instance, or None."""
    cfg = env.init_task_config
    path = os.path.join(
        system_utils.benchmark_conf_path(),
        "llm_task",
        cfg["sub_task_name"],
        str(cfg["scene"]["scene_instance_id"]),
        "instructions.json",
    )
    try:
        with open(path) as f:
            spec = json.load(f).get("pregrasp")
    except (OSError, json.JSONDecodeError):
        return None
    if spec and spec.get("object") and spec.get("gripper"):
        return spec
    return None


class PreGraspHook(HookBase):
    """Form a friction grasp at start; hold the gripper closed until the policy
    signals an opening trend, then release."""

    def __init__(self):
        self._active = False
        self._hold_idx = None
        self._released = True
        self._min_cmd = None

    def start_callback(self, env, _):
        self._active = False
        self._released = True
        if env.sub_task_name not in PREGRASP_TASKS:
            return
        spec = _load_pregrasp_spec(env)
        if spec is None:
            logger.warning(f"[pregrasp] no pregrasp spec for {env.sub_task_name}; skipping")
            return

        side = spec["gripper"]
        prefix = "/Workspace/Objects/" if env.sub_task_name else "/World/Objects/"
        obj_body = prefix + spec["object"] + "/entity"
        gripper_joints = env.cfg["gripper_joints"]
        idx = [env.robot_joint_indices[v] for v in gripper_joints]
        self._hold_idx = _HOLD_IDX[side]

        # Block was kinematic at the gripper (no fall during reset). Make it a
        # weightless dynamic body so the fingers can clamp it without it falling
        # or being knocked away while the gripper closes.
        env.api_core.run_on_physics_loop(self._set_gravity, obj_body, disable=True, make_dynamic=True)

        # Slowly close the holding gripper via drive targets (is_trajectory=True
        # respects contact; is_trajectory=False would teleport through the block).
        base = list(env.cfg.get("init_gripper_open", [0.0] * len(gripper_joints)))
        for k in range(1, _CLOSE_STEPS + 1):
            base[self._hold_idx] = _CLOSED * k / _CLOSE_STEPS
            env.api_core.set_joint_positions_batched([(list(base), idx, True)])
            time.sleep(_CLOSE_DT)

        # Restore gravity: the block is now clamped and held by friction.
        env.api_core.run_on_physics_loop(self._set_gravity, obj_body, disable=False, make_dynamic=False)
        self._active = True
        self._released = False
        self._min_cmd = None
        logger.info(f"[pregrasp] friction grasp formed on {spec['object']} ({side}); holding {side} closed")

    def step_callback(self, env, action):
        if not self._active or self._released or not isinstance(action, dict):
            return
        gripper = action.get("gripper")
        if gripper is None or len(gripper) <= self._hold_idx:
            return
        cmd = float(gripper[self._hold_idx])
        self._min_cmd = cmd if self._min_cmd is None else min(self._min_cmd, cmd)
        if cmd > self._min_cmd + _OPEN_DELTA:
            # Policy is opening the gripper -> release: let its command through.
            self._released = True
            logger.info(f"[pregrasp] release: policy gripper cmd {cmd:.3f} > hold {self._min_cmd:.3f}+{_OPEN_DELTA}")
            return
        # Otherwise force the holding gripper closed.
        g = list(gripper)
        g[self._hold_idx] = _CLOSED
        action["gripper"] = g

    def _set_gravity(self, obj_body, disable, make_dynamic):
        from pxr import PhysxSchema
        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        prim = stage.GetPrimAtPath(obj_body)
        if make_dynamic:
            prim.GetAttribute("physics:kinematicEnabled").Set(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(disable)

    def gather_results(self):
        return {}
