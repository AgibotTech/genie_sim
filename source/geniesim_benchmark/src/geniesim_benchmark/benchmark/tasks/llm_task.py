# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import os, json
import geniesim_benchmark.utils.system_utils as system_utils
from geniesim_benchmark.plugins.logger import Logger

from geniesim_benchmark.benchmark.instruction_segments import get_episode_segments

from .base_task import BaseTask

logger = Logger()


class LLMTask(BaseTask):
    """
    Load full episode instructions and segment text, with optional language perturbation.
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
                self._instruction_segments = instr_info.get("subtask_steps")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            logger.warning(f"No instructions found for {path}")
            self.instructions = [""]
            self._instruction_segments = None

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

    def get_instruction(self):
        if self.instructions[0] != "":
            idx = self.current_episode_id % len(self.instructions)
            return [
                self.instructions[idx]["instruction"],
                self.instructions[idx].get("target", {}).get("id1", ""),
                self.instructions[idx].get("gripper", ""),
            ]
        else:
            return ["default", "", ""]

    def get_instruction_segments(self):
        return get_episode_segments(self._instruction_segments, self.current_episode_id)

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
