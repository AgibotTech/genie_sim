# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Vectorized benchmark execution: N task envs cloned on one Isaac Sim stage.

The per-env ``PiEnv.get_observation`` / ``.step`` / ``.reset`` paths route
through ``api_core.get_obs_bundle``, which reads the single root robot and is
NOT vec-aware. So this module bypasses them: it reads observations, applies
actions, and resets joints directly through the per-env forked ``APICore``
views (``fork_for_env``) and the shared-camera render cache. The envs are kept
only for their task (``LLMTask``), config, and generalization plumbing.

Policy contract (corobot): ``CoRobotPolicy.act()`` already returns a
post-processed action dict ``{"arm", "gripper", optional "waist"}``; IK and
gripper mapping happen inside the policy. The env adapter therefore applies the
returned dict directly to the per-env articulation without re-processing.
"""

import time
import types
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from geniesim_benchmark.plugins.logger import Logger
from geniesim_benchmark.app.controllers.api_core import APICore
from geniesim_benchmark.utils.data_courier import DataCourier

logger = Logger()


class BenchmarkVecEnv:
    """Synchronises reset/step/obs across N sub-envs cloned on one stage."""

    def __init__(
        self,
        api_core: APICore,
        envs: list,
        data_couriers: List[DataCourier],
        hz: float = 30.0,
    ):
        self._api_core = api_core
        self._envs = envs
        self._data_couriers = data_couriers
        self._n_envs = len(envs)
        self._hz = hz
        self._next_tick = time.time()

        for env in self._envs:
            VecEnvAdapter.patch(env)

    @property
    def n_envs(self) -> int:
        return self._n_envs

    def get_env(self, idx: int):
        return self._envs[idx]

    def all_done(self) -> bool:
        return all(env.has_done for env in self._envs)

    def do_eval_actions(self):
        for env in self._envs:
            env.do_eval_action()

    def set_infer_status(self, need_infer_list: List[bool]):
        for env, need in zip(self._envs, need_infer_list):
            env.need_infer = need

    def set_depth_status(self, need_depth_list: List[bool]):
        for env, need in zip(self._envs, need_depth_list):
            env.need_depth = need

    # ----------------------------- reset --------------------------------
    def reset_all(self) -> List:
        obs_list = self._batched_reset(list(range(self._n_envs)))
        self._next_tick = time.time()
        self._force_initial_render()
        self._refresh_obs_images(obs_list)
        return obs_list

    def _force_initial_render(self) -> None:
        api = self._api_core
        env_paths = [f"/World/envs/env_{i}" for i in range(self._n_envs)]
        api.prepare_shared_render_cache()
        for idx in range(self._n_envs):
            api.render_env(idx, env_paths=env_paths)

    def _batched_reset(self, env_indices: List[int]) -> List:
        envs = [self._envs[i] for i in env_indices]

        for env in envs:
            env._reset_py_state()

        eps = 1e-2
        max_retry = 10
        converged = {id(env): False for env in envs}
        for _attempt in range(max_retry):
            pending = [env for env in envs if not converged[id(env)]]
            if not pending:
                break

            def _write_joints():
                for env in pending:
                    env._reset_write_joints()

            self._api_core.run_on_physics_loop(_write_joints)
            time.sleep(0.1)

            check_result: Dict[int, bool] = {}

            def _read_and_check():
                for env in pending:
                    check_result[id(env)] = env._reset_check_converged(eps)

            self._api_core.run_on_physics_loop(_read_and_check)
            for env in pending:
                if check_result.get(id(env), False):
                    converged[id(env)] = True

        unconverged = [i for i, env in zip(env_indices, envs) if not converged[id(env)]]
        if unconverged:
            logger.warning(
                f"[vec_reset] arm/waist did not reach init within {max_retry} retries for envs {unconverged}"
            )

        time.sleep(1.0)
        self._api_core.reset_env()

        obs_results: Dict[int, dict] = {}

        def _read_obs_no_images():
            for i, env in enumerate(envs):
                obs_results[i] = self._read_obs_direct(env, env_indices[i], skip_images=True)

        self._api_core.run_on_physics_loop(_read_obs_no_images)
        return [obs_results[i] for i in range(len(envs))]

    def _refresh_obs_images(self, obs_list: List, only: List[int] = None) -> None:
        indices = range(self._n_envs) if only is None else only

        def _read_images():
            for i in indices:
                obs = obs_list[i]
                if not isinstance(obs, dict):
                    continue
                env = self._envs[i]
                if not getattr(env, "need_infer", True):
                    continue
                api_view = env.api_core
                images = api_view._get_observation_image(env.data_courier._camera_dirs())
                if images:
                    obs["images"] = images
                # Mirrors the image fetch above so a fresh episode's first
                # observation (built via reset -> _batched_reset -> here,
                # bypassing _read_obs_direct's normal per-step path) still
                # carries real depth when the policy wants it.
                if getattr(env, "need_depth", True):
                    depths = api_view._get_observation_depth(env.data_courier._camera_dirs())
                    if depths:
                        obs["depth"] = depths

        self._api_core.run_on_physics_loop(_read_images)

    # ----------------------------- step ---------------------------------
    def step(self, actions: list) -> Tuple[List, List[bool], List[bool], List]:
        assert len(actions) == self._n_envs
        self._batch_apply_actions(actions)
        obs_list, dones, need_updates, task_progresses = self._batch_read_obs_and_eval(actions)
        self._rate_sleep()
        return obs_list, dones, need_updates, task_progresses

    def _batch_apply_actions(self, actions: list):
        def _apply():
            for idx in range(self._n_envs):
                action = actions[idx]
                if action is None:
                    continue
                self._envs[idx]._apply_action_to_articulation(action)

        self._api_core.run_on_physics_loop(_apply)

    def _any_needs_images(self, actions: list) -> bool:
        for idx in range(self._n_envs):
            if actions[idx] is None:
                continue
            if getattr(self._envs[idx], "need_infer", True):
                return True
        return False

    def _maybe_render_on_demand(self, actions: list) -> None:
        if not self._any_needs_images(actions):
            return
        # render_env schedules its own render-loop work; don't nest it under
        # run_on_render_loop or the render barrier deadlocks.
        api = self._api_core
        api.prepare_shared_render_cache()
        env_paths = [f"/World/envs/env_{i}" for i in range(self._n_envs)]
        for idx in range(self._n_envs):
            if actions[idx] is not None and getattr(self._envs[idx], "need_infer", True):
                api.render_env(idx, env_paths=env_paths)

    def _batch_read_obs_and_eval(self, actions: list) -> Tuple[List, List[bool], List[bool], List]:
        self._maybe_render_on_demand(actions)

        # task.step / action_update run on the task thread — they may schedule
        # run_on_physics_loop themselves, which would deadlock if nested.
        need_updates: Dict[int, bool] = {}
        for idx in range(self._n_envs):
            env = self._envs[idx]
            if actions[idx] is None:
                env.has_done = True
                need_updates[idx] = False
                continue

            env.current_step += 1
            if env.current_step != 1 and env.current_step % 30 == 0:
                env.task.step(env)
                env.action_update()
                need_updates[idx] = True
            else:
                need_updates[idx] = False

        obs_results: Dict[int, dict] = {}

        def _read_all():
            for idx in range(self._n_envs):
                if actions[idx] is None:
                    obs_results[idx] = None
                    continue
                obs_results[idx] = self._read_obs_direct(self._envs[idx], idx)

        self._api_core.run_on_physics_loop(_read_all)

        obs_list, dones, need_updates_list, progresses = [], [], [], []
        for i in range(self._n_envs):
            env = self._envs[i]
            obs_list.append(obs_results.get(i))
            dones.append(env.has_done)
            need_updates_list.append(need_updates[i])
            progresses.append(getattr(env.task, "task_progress", []))
        return obs_list, dones, need_updates_list, progresses

    # ----------------------------- obs ----------------------------------
    def _read_obs_direct(self, env, env_idx: int, skip_images: bool = False) -> dict:
        """Build per-env obs inside a physics callback (mirrors PiEnv.get_observation)."""
        api_view = env.api_core

        images = {}
        depths = {}
        if getattr(env, "need_infer", True) and not skip_images:
            # _camera_dirs() is the same robot->camera mapping the serial path uses.
            images = api_view._get_observation_image(env.data_courier._camera_dirs())
            if getattr(env, "need_depth", True):
                depths = api_view._get_observation_depth(env.data_courier._camera_dirs())

        art = api_view._get_articulation()
        if art is not None:
            positions = art.get_joint_positions()
            positions = positions.tolist() if positions is not None else []
            full_joint_states = {art.dof_names[i]: positions[i] for i in range(len(positions))}
        else:
            full_joint_states = api_view._get_joint_state_dict()

        if not full_joint_states:
            return {"images": images, "states": [], "depth": depths}

        def jv(names):
            return [full_joint_states[name] for name in names]

        cfg = env.cfg
        gripper_values = jv(cfg["gripper_joints"])
        states = {
            "left_arm": jv(cfg["left_arm_joints"]),
            "right_arm": jv(cfg["right_arm_joints"]),
            "left_gripper": gripper_values[:1],
            "right_gripper": gripper_values[1:2],
            "waist": jv(cfg["waist_joints"]),
            "head": jv(cfg.get("obs_extra_joints", [])),
        }
        env.cur_arm = states["left_arm"] + states["right_arm"]
        return {"images": images, "states": states, "depth": depths}

    def _rate_sleep(self):
        now = time.time()
        to_sleep = self._next_tick - now
        if to_sleep > 0:
            time.sleep(to_sleep)
        self._next_tick = max(self._next_tick + 1.0 / self._hz, now)


class VecEnvAdapter:
    """Patches batched apply/reset hooks onto PiEnv/DummyEnv for vec mode."""

    @staticmethod
    def patch(env):
        if hasattr(env, "_apply_action_to_articulation"):
            return env
        VecEnvAdapter._patch_apply(env)
        VecEnvAdapter._patch_reset(env)
        return env

    @staticmethod
    def _patch_apply(env):
        def _apply(self, action):
            """Apply an already post-processed action dict to this env's robot.

            ``CoRobotPolicy.act()`` has already run IK / gripper mapping, so the
            arm values are joint targets and gripper values are final — write
            them straight to the per-env articulation.
            """
            if action is None:
                return
            art = self.api_core._get_articulation()
            if art is None:
                return
            from isaacsim.core.utils.types import ArticulationAction

            cfg = self.cfg

            def _send(values, joint_names):
                if not values:
                    return
                indices = np.array([self.robot_joint_indices[v] for v in joint_names], dtype=np.int32)
                art.apply_action(
                    ArticulationAction(
                        joint_positions=np.array([float(v) for v in values], dtype=np.float32),
                        joint_indices=indices,
                    )
                )

            _send(action.get("arm"), cfg["arm_joints"])
            _send(action.get("gripper"), cfg["gripper_joints"])
            waist = action.get("waist")
            if waist is not None:
                waist_joints = cfg["waist_joints"]
                _send(waist, waist_joints[: len(waist)])

        env._apply_action_to_articulation = types.MethodType(_apply, env)

    @staticmethod
    def _patch_reset(env):
        def _reset_py_state(self):
            self._followed_objects = set()
            self._picked_objects = set()
            self.last_update_time = time.time()
            self.has_done = False
            self.need_infer = True
            self.need_depth = True
            self.current_step = 0
            if self.task is not None:
                self.task.reset(self)
            self.robot_joint_indices = self.api_core.get_robot_joint_indices()

        def _reset_write_joints(self):
            art = self.api_core._get_articulation()
            if art is None:
                return
            cfg = self.cfg
            init_gripper = list(cfg.get("init_gripper_open", [0.0, 0.0]))
            for values, names in (
                (list(self.init_arm), cfg["arm_joints"]),
                (list(self.init_waist), cfg["waist_joints"]),
                (list(self.init_head), cfg["head_joints"]),
                (init_gripper, cfg["gripper_joints"]),
            ):
                indices = np.array([self.robot_joint_indices[v] for v in names], dtype=np.int32)
                art.set_joint_positions(np.asarray(values, dtype=np.float32), joint_indices=indices)

        def _reset_check_converged(self, eps: float = 1e-2) -> bool:
            art = self.api_core._get_articulation()
            if art is None:
                return True
            positions = art.get_joint_positions()
            if positions is None or len(positions) == 0:
                return False
            cfg = self.cfg
            try:
                arm_pos = np.asarray([float(positions[self.robot_joint_indices[v]]) for v in cfg["arm_joints"]])
                waist_pos = np.asarray([float(positions[self.robot_joint_indices[v]]) for v in cfg["waist_joints"]])
            except (KeyError, IndexError):
                return False
            init_arm = np.asarray(self.init_arm, dtype=np.float64)
            init_waist = np.asarray(self.init_waist, dtype=np.float64)
            c1 = np.max(np.abs(arm_pos - init_arm)) < eps if init_arm.size else True
            c2 = np.max(np.abs(waist_pos - init_waist)) < eps if init_waist.size else True
            return bool(c1 and c2)

        env._reset_py_state = types.MethodType(_reset_py_state, env)
        env._reset_write_joints = types.MethodType(_reset_write_joints, env)
        env._reset_check_converged = types.MethodType(_reset_check_converged, env)


class VecPolicyWrapper:
    """Wraps N policies; runs per-env act() concurrently across a thread pool.

    ``CoRobotPolicy.act()`` performs WebSocket inference when its action buffer
    is empty (the parallel win — N servers/connections hit at once) and pops a
    locally post-processed action otherwise. Each policy owns its own WS
    connection. NOTE: ``CoRobotPolicy`` shares a process-wide IK/FK solver,
    which is only exercised for EEF_ABS actions — JOINT_ABS (the default) does
    not touch it. If running EEF_ABS vectorized, confirm the solver is
    re-entrant or serialize that branch.
    """

    def __init__(self, policies: list, max_infer_workers: int = 0):
        self._policies = policies
        self._n = len(policies)
        if max_infer_workers <= 0:
            max_infer_workers = self._n
        self._infer_pool = ThreadPoolExecutor(max_workers=max_infer_workers, thread_name_prefix="vec_infer")

    @property
    def n_envs(self) -> int:
        return self._n

    def get_policy(self, idx: int):
        return self._policies[idx]

    def act_batch(
        self,
        observations: list,
        instructions: List[str],
        gen_configs: list,
        step_nums: List[int],
        active_mask: List[bool] = None,
    ) -> list:
        if active_mask is None:
            active_mask = [True] * self._n

        results: List = [None] * self._n

        def _one(idx):
            try:
                return idx, self._policies[idx].act(
                    observations[idx],
                    step_num=step_nums[idx],
                    task_instruction=instructions[idx],
                    gen_config=gen_configs[idx],
                )
            except Exception as e:
                logger.error(f"[env{idx}] inference failed: {e}")
                return idx, None

        idxs = [i for i in range(self._n) if active_mask[i]]
        futures = [self._infer_pool.submit(_one, i) for i in idxs]
        for future in futures:
            idx, action = future.result(timeout=180)
            results[idx] = action
        return results

    def need_infer_list(self) -> List[bool]:
        return [p.need_infer() for p in self._policies]

    def need_depth_list(self) -> List[bool]:
        return [p.need_depth() for p in self._policies]

    def update_task_status_batch(self, dones: List[bool], progresses: list):
        for i, policy in enumerate(self._policies):
            if hasattr(policy, "update_task_status"):
                policy.update_task_status(dones[i], progresses[i])

    def reset_all(self):
        for p in self._policies:
            p.reset()

    def reset_policy(self, idx: int):
        self._policies[idx].reset()

    def shutdown(self):
        self._infer_pool.shutdown(wait=False)
        for p in self._policies:
            drop = getattr(p, "_drop_connection", None)
            if callable(drop):
                drop()
