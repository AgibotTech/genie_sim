# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""An explicit score monitor around a native stage; it never changes its exit rule."""

from .common_actions import ActionBase, ActionEvent, ActionSetWaitAny, StepOut
from geniesim_benchmark.plugins.output_system.policy_budget import (
    POLICY_BUDGET_VERSIONS,
    assess_policy_budget,
    policy_budget_baseline,
)


class PolicyDispatchBudget(ActionBase):
    def __init__(
        self,
        env,
        stage: ActionBase,
        step: int,
        version: str,
        points: float | None = None,
        step_out: int | None = None,
    ):
        super().__init__(env)
        if version not in POLICY_BUDGET_VERSIONS:
            raise ValueError("Unsupported PolicyDispatchBudget version")
        if type(step) is not int or step < 0:
            raise ValueError("PolicyDispatchBudget.step must be a nonnegative integer")
        if step_out is None:
            if not isinstance(stage, ActionSetWaitAny):
                raise ValueError("PolicyDispatchBudget needs a stage-local or inherited StepOut")
            exits = [a for a in stage._actions if isinstance(a, StepOut)]
            if len(exits) != 1:
                raise ValueError("PolicyDispatchBudget needs exactly one stage-local StepOut")
            step_out = exits[0].max_step
        if points is not None and (type(points) not in (int, float) or not 0 <= points <= 50):
            raise ValueError("PolicyDispatchBudget.points must be finite and between 0 and 50")
        self.stage = stage
        self._version = version
        self._start_count = None
        self._skip_started_update = False
        self._step_out = step_out
        self.progress_info["POLICY_BUDGET"] = {
            "version": version,
            "step": step,
            "step_out": self._step_out,
            "baseline": policy_budget_baseline(self._step_out, version),
            "count": None,
        }
        if points is not None:
            self.progress_info["POLICY_BUDGET"]["points"] = points

    def _count(self) -> int:
        count = getattr(self.env, "policy_dispatch_count", None)
        if type(count) is not int or count < 0:
            raise ValueError("PolicyDispatchBudget requires a measured policy dispatch counter")
        return count

    def handle_action_event(self, action, event):
        if event == ActionEvent.STARTED:
            self._start_count = self._count()
            # ActionBase.start calls update(0) after this callback. The native
            # stage.start already updates its child, so do not check it twice.
            self._skip_started_update = True
            self.stage.start()

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        remaining = delta_time
        if self._skip_started_update:
            self._skip_started_update = False
        elif self.stage.is_running():
            remaining = self.stage.update(delta_time)
        count = self._count() - self._start_count
        self.progress_info["POLICY_BUDGET"].update(assess_policy_budget(self._step_out, count, self._version))
        return super().update(remaining)

    def _is_done(self) -> bool:
        return self.stage.is_finished()

    def pause(self) -> bool:
        if not super().pause():
            return False
        self.stage.pause()
        return True

    def resume(self) -> bool:
        if not super().resume():
            return False
        self.stage.resume()
        return True

    def stop(self) -> bool:
        if not super().stop():
            return False
        self.stage.stop()
        return True
