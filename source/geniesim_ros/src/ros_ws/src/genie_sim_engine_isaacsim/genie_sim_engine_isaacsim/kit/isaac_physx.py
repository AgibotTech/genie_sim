#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""PhysX physics engine implementation.

Thin wrapper around ``IsaacSimStage``.  All physics is driven by
``get_physx_simulation_interface().simulate() / fetch_results()``
(PhysX 5 via ``omni.physx``).  ``tick_extras`` is a no-op.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from genie_sim_engine_core.engine.base import PhysicsEngine
from genie_sim_engine_core.schema.scene import PinBaseMode


@PhysicsEngine.register("isaac_physx")
class IsaacPhysXEngine(PhysicsEngine):
    """Isaac Sim PhysX engine — delegates to :class:`IsaacSimStage`."""

    def __init__(
        self,
        *,
        robot_prefix: str,
        scene_usda: str,
        robot_usda: str,
        render_layer_usda: str,
        physics_hz: float,
        render_hz: float,
        simulation_app: Any,
        logger: Any,
        params: Any,
        robot_from_urdf: bool,
        init_joint_pos: Any,
        runtime_usd_dump_path: str,
        pin_base_to_world: bool,
        pin_base_mode: str = "",  # declared so the downgrade below is visible, not swallowed by **_unused
        newton_solvers_path: str = "",
        scene_cfg: dict | None = None,
        scene_yaml_path: str = "",  # accepted for API parity; unused
        physics_solver: str = "mujoco-warp",
        render_mode: str = "raster",
        **_unused: Any,  # tolerate backend-specific kwargs (substep, mass-matrix interval, mujoco_pd_*, convert_joints_to_fixed)
    ) -> None:
        self._logger = logger
        self._reset_epoch = 0
        self._render_mode = (render_mode or "raster").strip().lower()
        # ``pin_base_mode: mocap`` has no implementation on this engine:
        # the base is welded via ``fix_base`` below, which HOLDS it (so
        # nothing falls) but cannot re-place it at reset.  Downgrade to
        # the default weld and say so — being swallowed by ``**_unused``
        # is how a scene yaml ends up quietly not doing what it asks.
        # The Newton engine refuses instead of downgrading, because there
        # the consequence is a base under gravity with nothing holding
        # it; here it is only a lost capability.
        if PinBaseMode.resolve(pin_base_to_world, pin_base_mode).needs_base_clamp:
            logger.warn(
                "[isaac_physx] pin_base_mode=mocap is not implemented on this engine "
                "(no per-substep base clamp) — using the default weld instead. The "
                "base is held but cannot be re-placed at reset; use physics_engine "
                "newton_standalone with physics_solver mujoco_warp_multi_env for mocap."
            )
        from genie_sim_engine_isaacsim.kit.stage import IsaacSimStage

        self._stage_obj = IsaacSimStage(
            robot_prefix=robot_prefix,
            scene_usda=scene_usda,
            robot_usda=robot_usda,
            render_layer_usda=render_layer_usda,
            physics_hz=physics_hz,
            render_hz=render_hz,
            simulation_app=simulation_app,
            logger=logger,
            params=params,
            robot_from_urdf=robot_from_urdf,
            init_joint_pos=init_joint_pos,
            runtime_usd_dump_path=runtime_usd_dump_path,
            physics_engine="isaac_physx",
            # IsaacSimStage's internal arg is still ``fix_base`` —
            # the canonical scene-yaml name is ``pin_base_to_world``;
            # both refer to "weld base_link → world" in
            # ``_apply_fix_base_policy``.
            fix_base=pin_base_to_world,
            newton_solvers_path="",  # unused for PhysX
            scene_cfg=scene_cfg,
        )

        # Viewport / Hydra render mode — see IsaacNewtonEngine for the
        # placement rationale. PhysX uses the same stack so the call is
        # identical.
        try:
            from genie_sim_engine_isaacsim.kit.bootstrap import configure_viewport_for_debug

            configure_viewport_for_debug(render_mode=self._render_mode)
        except Exception as exc:
            self._logger.warn(f"[isaac_physx] viewport config failed: {exc}")

    # ------------------------------------------------------------------
    # PhysicsEngine interface
    # ------------------------------------------------------------------

    @property
    def reset_epoch(self) -> int:
        return self._reset_epoch

    def episode_reset(self, env_ids=None, seeds=None, instances=None) -> None:
        from genie_sim_engine_isaacsim.kit.episode_reset import validate_reset

        if not validate_reset(env_ids, seeds, instances):
            return
        self._stage_obj._episode_snapshot.restore()
        self._reset_epoch += 1

    @property
    def stage(self) -> Any:
        return self._stage_obj.stage

    @property
    def robot_prefix(self) -> str:
        return self._stage_obj._robot_prefix

    @property
    def joint_names(self) -> List[str]:
        return self._stage_obj.joint_names

    @property
    def joint_prim_map(self) -> Dict[str, str]:
        return self._stage_obj.joint_prim_map

    @property
    def body_paths(self) -> List[str]:
        return self._stage_obj.body_paths

    def startup(self, headless: bool) -> None:
        self._stage_obj.startup_loop_setup(headless)

    def step(self, dt: float, step_start: float) -> float:
        t0 = time.monotonic()
        self._stage_obj.physx_sim.simulate(dt, step_start)
        self._stage_obj.physx_sim.fetch_results()
        return (time.monotonic() - t0) * 1000.0

    def tick_extras(self) -> None:
        pass  # no-op for PhysX

    def apply_commands(self, cmd_positions, cmd_4ws_steer_pos, cmd_4ws_drive_vel, cmd_4ws_stamp) -> None:
        self._stage_obj.apply_commands(
            cmd_positions=cmd_positions,
            cmd_4ws_steer_pos=cmd_4ws_steer_pos,
            cmd_4ws_drive_vel=cmd_4ws_drive_vel,
            cmd_4ws_stamp=cmd_4ws_stamp,
        )

    # ------------------------------------------------------------------
    # State readback — PhysX writes USD attributes every tick, so a
    # ``snapshot_*`` over the live stage is the source of truth.
    # ------------------------------------------------------------------

    def get_joint_states(self):
        # A tensor reset can precede USD writeback; publish the live state so
        # a new reset epoch is never paired with pre-reset joint attributes.
        return self._stage_obj.get_joint_states()

    def get_body_transforms(self):
        from genie_sim_engine_isaacsim.kit.stage import snapshot_body_transforms

        return snapshot_body_transforms(self.stage, self.body_paths)

    def get_odom(self, sim_time):
        from genie_sim_engine_isaacsim.kit.stage import snapshot_odom

        return snapshot_odom(self.stage, self.robot_prefix, sim_time)

    def shutdown(self) -> None:
        self._stage_obj.shutdown()
