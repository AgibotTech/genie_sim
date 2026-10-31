# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import time

import numpy as np
import os, json
import geniesim_benchmark.utils.system_utils as system_utils
from geniesim_benchmark.plugins.logger import Logger

from .base_task import BaseTask

logger = Logger()


class LLMTask(BaseTask):
    """
    Instruction-following task: loads instructions.json, drives the per-instruction
    / subtask checker logic and language (instruction) perturbation.
    """

    def __init__(self, env):
        super(LLMTask, self).__init__(env)
        self.current_episode_id = 0

        instance_id = env.init_task_config["scene"]["scene_instance_id"]
        sub_task_name = env.init_task_config["sub_task_name"]
        path = os.path.join(
            system_utils.benchmark_conf_path(),
            "llm_task",
            sub_task_name,
            str(instance_id),
        )
        try:
            with open(
                os.path.join(path, "instructions.json"),
                "r",
            ) as f:
                instr_info = json.load(f)
                self.instructions = instr_info["instructions"]
                self._subtask_steps_per_episode = instr_info.get("subtask_steps")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            logger.warning(f"No instructions found for {path}")
            self.instructions = [""]
            self._subtask_steps_per_episode = None

        self.instruction_mode = env.init_task_config.get("instruction_mode", "full")
        if self.instruction_mode == "subtask" and not self._subtask_steps_per_episode:
            logger.warning(
                f"instruction_mode=subtask requested but instructions.json for "
                f"{sub_task_name}/{instance_id} has no `subtask_steps`; falling back to full."
            )
            self.instruction_mode = "full"
        if self.instruction_mode == "subtask":
            # Also downgrade if the scoring table has no layout for this task —
            # otherwise runtime would drive subtask checkers but eval_utils
            # would score against the full-mode TASK_STEPS, silently misaligned.
            from geniesim_benchmark.plugins.output_system.eval_utils import TASK_SUBTASK_STEPS
            import re as _re

            key = _re.sub(r"_\d+$", "", sub_task_name)
            if key not in TASK_SUBTASK_STEPS:
                logger.warning(
                    f"instruction_mode=subtask requested but TASK_SUBTASK_STEPS "
                    f"has no entry for {key}; falling back to full."
                )
                self.instruction_mode = "full"

        # Subtask runtime state, rebuilt each episode in reset().
        self._subtask_active = None
        self._subtask_idx = 0
        self._subtask_last_update_time = None

        if env.init_task_config.get("language_perturbation"):
            self._apply_language_perturbation(env, sub_task_name, instance_id)

    def _apply_language_perturbation(self, env, sub_task_name, instance_id):
        # Overlay paraphrased instruction strings keyed by (sub_task_name, instance_id).
        # Length must match the canonical instructions list; non-instruction fields
        # (target.id1, gripper, ...) are preserved from the original entry.
        if not self.instructions or self.instructions[0] == "":
            return
        cfg_path = env.init_task_config.get("language_perturbation_config") or os.path.join(
            system_utils.benchmark_conf_path(), "instruction_perturbation.json"
        )
        try:
            with open(cfg_path, "r") as f:
                table = json.load(f)
        except FileNotFoundError:
            logger.warning(f"language_perturbation config not found: {cfg_path}")
            return
        except Exception as e:
            logger.warning(f"failed to read language_perturbation config {cfg_path}: {e}")
            return

        entries = table.get(sub_task_name, {}).get(str(instance_id))
        if not entries:
            logger.warning(
                f"no perturbation entries for {sub_task_name}/{instance_id} in {cfg_path}; "
                f"using canonical instructions"
            )
            return
        if len(entries) != len(self.instructions):
            logger.warning(
                f"perturbation length mismatch for {sub_task_name}/{instance_id}: "
                f"{len(entries)} vs {len(self.instructions)}; using canonical instructions"
            )
            return

        overlaid = []
        for orig, pert in zip(self.instructions, entries):
            merged = dict(orig)
            merged["instruction"] = pert["instruction"]
            overlaid.append(merged)
        self.instructions = overlaid
        logger.info(f"language_perturbation: overlaid {len(overlaid)} instructions for {sub_task_name}/{instance_id}")

    def set_task(self, episode_id):
        self.current_episode_id = episode_id

    def reset_subtask_runtime(self):
        """Rebuild subtask state for the current episode. Called from env.reset()."""
        # Drop subtask checker entries from previous episode — legacy problems.json
        # checkers re-register themselves on reset, but our subtask checkers are
        # built lazily on first tick so we need to evict their stale rows here.
        if hasattr(self, "task_progress") and self.task_progress is not None:
            self.task_progress[:] = [e for e in self.task_progress if "subtask_step_idx" not in e]
        if self.instruction_mode != "subtask" or not self._subtask_steps_per_episode:
            self._subtask_active = None
            return
        # subtask_steps is either a flat list (one set for all episodes) or a
        # list-of-lists indexed by episode.
        steps = self._subtask_steps_per_episode
        if steps and isinstance(steps[0], list):
            idx = self.current_episode_id % len(steps)
            steps = steps[idx]
        # Deep-ish copy so per-episode runtime keys (_checker, _step_anchor,
        # _passthrough_anchor) don't leak across episodes.
        self._subtask_active = [{k: v for k, v in s.items() if not k.startswith("_")} for s in steps]
        self._subtask_idx = 0
        self._subtask_last_update_time = time.time()

    def _build_step_checker(self, step):
        """Lazily instantiate the ader checker for a step (no StepOut wrapping —
        we drive it ourselves via update())."""
        if step.get("_checker") is not None:
            return step["_checker"]
        spec = step.get("checker")
        if spec == "passthrough" or spec is None:
            step["_checker"] = "passthrough"
            return step["_checker"]
        # Lazy import to avoid a top-level cycle.
        from geniesim_benchmark.plugins.ader.action.action_parsing import parse_action

        if not isinstance(spec, dict):
            raise ValueError(f"checker must be dict or 'passthrough', got: {spec!r}")
        # `spec` is the same shape as a problems.json action node, e.g.
        #   {"LiftUp": "benchmark_pot_09aadb57|0.1"}
        #   {"ActionSetWaitAll": [{"InBBox": "..."}, {"Upright": "..."}]}
        # Register checker entries into the real task_progress so their
        # SCORE/STATUS are visible to eval_utils. Tag the top-level entry as
        # the subtask "root" so the scoring layer can pick exactly one row per
        # sub-instruction and ignore inner WaitAll/WaitAny bookkeeping rows.
        before = len(self.task_progress)
        checker = parse_action(spec, self.task_progress, self.env)
        if checker is None:
            raise ValueError(f"failed to build checker for step: {step}")
        added = self.task_progress[before:]
        if added:
            added[0]["subtask_role"] = "root"
            cached = getattr(self.env, "_placeholders", {}) or {}
            for entry in added:
                entry["subtask_step_idx"] = self._subtask_idx
                # Replay placeholders set by earlier subtask steps so that
                # `{@placeholder_str1}` references in this step resolve to the
                # object selected by a previous Follow/PickUpOnGripper.
                for k, v in cached.items():
                    setattr(entry["acion_obj"], k, v)
        checker.start()
        step["_checker"] = checker
        return checker

    def get_instruction(self):
        if self.instruction_mode == "subtask" and self._subtask_active:
            steps = self._subtask_active
            idx = min(self._subtask_idx, len(steps) - 1)
            step = steps[idx]
            return [
                step["instruction"],
                step.get("target", {}).get("id1", ""),
                step.get("gripper", ""),
            ]
        if self.instructions[0] != "":
            idx = self.current_episode_id % len(self.instructions)
            return [
                self.instructions[idx]["instruction"],
                self.instructions[idx].get("target", {}).get("id1", ""),
                self.instructions[idx].get("gripper", ""),
            ]
        else:
            return ["default", "", ""]

    def all_subtasks_done(self) -> bool:
        return (
            self.instruction_mode == "subtask"
            and self._subtask_active is not None
            and self._subtask_idx >= len(self._subtask_active)
        )

    def advance_subtask_if_passed(self):
        """Tick the current step's checker; if it has finished, advance the
        instruction pointer. Returns True iff the index moved this call."""
        if self.instruction_mode != "subtask" or not self._subtask_active:
            return False
        if self._subtask_idx >= len(self._subtask_active):
            return False

        step = self._subtask_active[self._subtask_idx]
        checker = self._build_step_checker(step)

        # Time slice driven by wallclock between consecutive task.step() calls.
        now = time.time()
        delta = now - (self._subtask_last_update_time or now)
        self._subtask_last_update_time = now

        passed = False
        timeout_steps = step.get("step_out", 0)
        if checker == "passthrough":
            # No state-based checker; pass after step_out frames (counted via
            # env.current_step) or immediately if step_out is 0/missing.
            current = getattr(self.env, "current_step", 0)
            anchor = step.setdefault("_passthrough_anchor", current)
            passed = timeout_steps <= 0 or (current - anchor) >= timeout_steps
        else:
            if checker.is_running():
                checker.update(max(delta, 0.0))
            passed = checker.is_finished()
            if not passed and timeout_steps > 0:
                current = getattr(self.env, "current_step", 0)
                anchor = step.setdefault("_step_anchor", current)
                if (current - anchor) >= timeout_steps:
                    checker.stop()
                    passed = True
                    logger.warning(
                        f"subtask step_out hit for index {self._subtask_idx} "
                        f"({step.get('instruction','')[:40]!r}); advancing anyway."
                    )

        if passed:
            logger.info(
                f"subtask step {self._subtask_idx + 1}/{len(self._subtask_active)} "
                f"passed: {step.get('instruction','')!r}"
            )
            self._subtask_idx += 1
            self._subtask_last_update_time = time.time()
            return True
        return False

    def reset(self, env):
        super().reset(env)
        self.reset_subtask_runtime()

    def step(self, env):
        if self.instruction_mode == "subtask":
            self.advance_subtask_if_passed()
            if self.all_subtasks_done():
                # Tell the env loop the episode is complete; the original
                # problems.json eval action keeps running as a safety net but
                # we don't need to wait for it.
                env.cancel_eval()

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset scene objects or floor plane

        :param env: environment instance
        """
        return

    def reset_agent(self, env):
        """
        Task-specific agent reset (no-op stub).

        :param env: environment instance
        """
        return

    def get_task_obs(self, env):
        """
        Task-specific observation (none — returns an empty array).

        :param env: environment instance
        :return: task-specific observation
        """
        return np.zeros(0, dtype=np.float32)
