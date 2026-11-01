# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""An explicit success-first alternative to the legacy parallel exit set."""

from .common_actions import ActionBase, ActionSetWaitAny, EvalExitAction


class ActionSetSuccessFirst(ActionSetWaitAny):
    """Check goals before failure exits, even when both can finish this update."""

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        max_left = 0.0
        for failure_pass in (False, True):
            for action in self._actions:
                if isinstance(action, EvalExitAction) != failure_pass:
                    continue
                if action.is_init():
                    action.start()
                if action.is_running():
                    max_left = max(max_left, action.update(delta_time))
                if failure_pass and action.is_finished():
                    break
            if any(action.is_finished() for action in self._actions):
                break
        # The inherited state transition stops unfinished siblings. Calling
        # the base directly avoids running the legacy parallel update again.
        return ActionBase.update(self, max_left)
