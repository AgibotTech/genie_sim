# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Wuji fold leaves; ordinary ADER composites own their prerequisites."""
from geniesim_benchmark.plugins.ader.action.common_actions import ActionEvent, EvaluateAction


class _WujiFoldStage(EvaluateAction):
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
            matched = self.env.fold_observation[self.predicate]
            if self._accept_frame(matched):
                if self._pass_start is None:
                    self._pass_start = self.env.fold_sim_time_s
                self._pass_frames += 1
                elapsed = self.env.fold_sim_time_s - self._pass_start
                self._done_flag = (self._pass_frames >= 2
                                   and elapsed + 1e-12 >= self.env.fold_confirm_sim_s)
            else:
                self._pass_start = None
                self._pass_frames = 0
        return super().update(delta_time)

    def _accept_frame(self, matched):
        return matched

    def _is_done(self):
        return self._done_flag

    def update_progress(self):
        self.progress_info.update(SCORE=int(self._done_flag),
                                  STATUS='SUCCESS' if self._done_flag else self.state.name)

    def handle_action_event(self, action, event):
        if event == ActionEvent.FINISHED:
            self.env.fold_event(f'{self.predicate}_completed')


class WujiFoldLeftSide(_WujiFoldStage):
    predicate = 'left'


class WujiFoldRightSide(_WujiFoldStage):
    predicate = 'right'


class WujiFoldHemToCollar(_WujiFoldStage):
    predicate = 'final'

    def handle_action_event(self, action, event):
        if event == ActionEvent.STARTED:
            self._unlock_tick = self.env.current_step
            # A final shape already present on unlock must be left before it
            # can establish a fresh fold. The unlock frame never confirms it.
            self._armed = not self.env.fold_observation['final']
            self.env.fold_event('final_unlocked')
        super().handle_action_event(action, event)

    def _accept_frame(self, matched):
        if self.env.current_step <= self._unlock_tick:
            return False
        if not matched:
            self._armed = True
        return self._armed and matched
