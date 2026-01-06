# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import time
from dataclasses import dataclass

from geniesim.plugins.ader import ActionManager
from geniesim.plugins.ader import ActionBase
from geniesim.plugins.ader.action.action_parsing import do_parsing
from geniesim.app.controllers.api_core import APICore


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

    def do_action(self, slot: str, name: str, action: ActionBase):
        self.action_executor.start(slot, name, action)

    def cancel_action(self, slot):
        self.action_executor.stop(slot)

    def exist_eval_action(self):
        return self.action_executor.exist_action("eval")

    def do_eval_action(self):
        self.do_action("eval", self.params.task_name, self.execute_action)

    def cancel_eval(self):
        self.has_done = True

    def reset(self):
        self.last_update_time = time.time()
        self.has_done = False
        self.api_core.reset()

    def exist_eval_action(self):
        return self.action_executor.exist_action("eval")

    def action_update(self):
        if not self.exist_eval_action():
            self.has_done = True
            return
        delta_time = time.time() - self.last_update_time
        self.action_executor.update(delta_time)
        self.last_update_time = time.time()
        if self.has_done:
            self.cancel_action("eval")

    def update_place_holder(self, key: str, value: str):
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
