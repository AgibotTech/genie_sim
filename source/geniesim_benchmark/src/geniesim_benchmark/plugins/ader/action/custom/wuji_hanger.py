# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Independent hanger leaves confirmed using physical simulation time."""
from geniesim_benchmark.plugins.ader.action.common_actions import ActionEvent, EvaluateAction


class _WujiHangerStage(EvaluateAction):
    predicate = None

    def __init__(self, env):
        super().__init__(env)
        self._done_flag = False
        self._last_tick = None
        self._pass_start = None
        self._pass_frames = 0
        self.progress_info.update(SCORE=0, STATUS='PENDING')

    def update(self, delta_time):
        if self.is_running() and self.env.current_step != self._last_tick:
            self._last_tick = self.env.current_step
            if self.env.hanger_observation[self.predicate]:
                if self._pass_start is None:
                    self._pass_start = self.env.hanger_sim_time_s
                self._pass_frames += 1
                confirm = (self.env.hung_confirm_sim_s if self.predicate == 'hung'
                           else self.env.hanger_confirm_sim_s)
                elapsed = self.env.hanger_sim_time_s - self._pass_start
                self._done_flag = self._pass_frames >= 2 and elapsed + 1e-12 >= confirm
            else:
                self._pass_start = None
                self._pass_frames = 0
        return super().update(delta_time)

    def _is_done(self):
        return self._done_flag

    def update_progress(self):
        self.progress_info.update(SCORE=int(self._done_flag),
                                  STATUS='SUCCESS' if self._done_flag else self.state.name)

    def handle_action_event(self, action, event):
        if event == ActionEvent.FINISHED:
            self.env.hanger_event(f'{self.predicate}_completed')


class WujiLiftCloth(_WujiHangerStage):
    predicate = 'cloth_lifted'


class WujiLiftHanger(_WujiHangerStage):
    predicate = 'hanger_lifted'


class WujiHangerNearCollar(_WujiHangerStage):
    predicate = 'near_collar'


class WujiHangerInserted(_WujiHangerStage):
    predicate = 'inserted'


class WujiGarmentHung(_WujiHangerStage):
    predicate = 'hung'
