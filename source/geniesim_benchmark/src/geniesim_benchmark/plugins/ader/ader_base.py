# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import time
from dataclasses import dataclass

from geniesim_benchmark.plugins.ader import ActionManager
from geniesim_benchmark.plugins.ader import ActionBase
from geniesim_benchmark.plugins.ader.action.action_parsing import do_parsing
from geniesim_benchmark.app.controllers.api_core import APICore


@dataclass
class AderParams:
    instance: int = 0
    task_name: str = "demo_task"


class AderEnv(object):
    def __init__(self, api_core, params: AderParams) -> None:
        self.api_core: APICore = api_core
        self.action_executor = ActionManager()
        self.last_update_time = time.time()
        self.params = params
        self.has_done = False
        self.execute_action = None
        # Cache of placeholder values set during this episode. Subtask-mode
        # checkers are built lazily; this lets us replay already-set
        # placeholders onto newly-registered action objects.
        self._placeholders = {}

    def do_action(self, slot: str, name: str, action: ActionBase):
        self.action_executor.start(slot, name, action)

    def cancel_action(self, slot):
        self.action_executor.stop(slot)

    def exist_eval_action(self):
        return self.action_executor.exist_action("eval")

    def do_eval_action(self):
        if getattr(self.task, "instruction_mode", "full") == "subtask":
            return
        self.do_action("eval", self.params.task_name, self.execute_action)

    def cancel_eval(self):
        self.has_done = True

    def reset(self):
        self.last_update_time = time.time()
        self.has_done = False
        self._followed_objects = set()
        self._picked_objects = set()
        self._placeholders = {}
        self.api_core.reset()

    def exist_eval_action(self):
        return self.action_executor.exist_action("eval")

    def action_update(self):
        # LLMTask.step drives the subtask checkers and their local deadlines.
        # Starting the original tree would compete for placeholders and exits.
        if getattr(self.task, "instruction_mode", "full") == "subtask":
            return
        if not self.exist_eval_action():
            self.has_done = True
            return
        delta_time = time.time() - self.last_update_time
        self.action_executor.update(delta_time)
        self.last_update_time = time.time()
        if self.has_done:
            self.cancel_action("eval")

    def update_place_holder(self, key: str, value: str):
        self._placeholders[key] = value
        for item in self.task.task_progress:
            setattr(item["acion_obj"], key, value)


class AderTask(object):
    def __init__(self, env: AderEnv, task_definitions_path) -> None:
        self.task_definitions_path = task_definitions_path

    def do_action_parsing(self, env: AderEnv):
        self.problem_name, self.objects, self.action, self.task_progress = do_parsing(env, self.task_definitions_path)
        env.execute_action = self.action

    def reset(self, env: AderEnv):
        self.problem_name, self.objects, self.action, self.task_progress = do_parsing(env, self.task_definitions_path)
        env.execute_action = self.action

    def update_progress(self, id, progress):
        for item in self.task_progress:
            if item["id"] == id:
                item["progress"] = progress
                break
