"""Latched bag items with physical-time holds and loaded closure ordering."""
from geniesim_benchmark.plugins.ader.action.common_actions import ActionEvent, EvaluateAction


class _WujiBagStage(EvaluateAction):
    predicate = None
    completion_event = None

    def __init__(self, env):
        super().__init__(env)
        self._done_flag = False
        self._last_tick = None
        self._pass_start = None
        self._pass_last_time = None
        self._pass_frames = 0
        self._event_sent = False
        self.progress_info.update(SCORE=0, STATUS='PENDING')

    def _hold(self, valid, duration):
        now = self.env.bag_sim_time_s
        if not valid:
            self._pass_start = self._pass_last_time = None
            self._pass_frames = 0
            return False
        if self._pass_start is None:
            self._pass_start = self._pass_last_time = now
            self._pass_frames = 1
        elif now > self._pass_last_time:
            self._pass_last_time = now
            self._pass_frames += 1
        return self._pass_frames >= 2 and now - self._pass_start + 1e-12 >= duration

    def _observe(self):
        return self._hold(self.env.bag_observation[self.predicate], self.env.bag_confirm_sim_s)

    def update(self, delta_time):
        if self.is_running() and self.env.current_step != self._last_tick:
            self._last_tick = self.env.current_step
            self._done_flag = self._observe()
        return super().update(delta_time)

    def _is_done(self):
        return self._done_flag

    def update_progress(self):
        self.progress_info.update(SCORE=int(self._done_flag),
                                  STATUS='SUCCESS' if self._done_flag else self.state.name)

    def handle_action_event(self, action, event):
        if event == ActionEvent.FINISHED and not self._event_sent:
            self._event_sent = True
            self.env.bag_event(self.completion_event)


class WujiBagPlaced(_WujiBagStage):
    completion_event = 'bag_placed_completed'

    def __init__(self, env):
        super().__init__(env)
        self._lift_confirmed = False

    def _observe(self):
        observation = self.env.bag_observation
        if not self._lift_confirmed:
            if self._hold(observation['bag_lifted'], self.env.bag_confirm_sim_s):
                self._lift_confirmed = True
                self._hold(False, self.env.bag_confirm_sim_s)
            # The lift confirmation frame cannot also start placement.
            return False
        return self._hold(observation['bag_in_bin'], self.env.bag_confirm_sim_s)


class WujiBottle1InBin(_WujiBagStage):
    predicate = 'bottle1_in_bin'
    completion_event = 'bottle1_in_bin_completed'


class WujiBottle2InBin(_WujiBagStage):
    predicate = 'bottle2_in_bin'
    completion_event = 'bottle2_in_bin_completed'


class WujiBothBottlesInBag(_WujiBagStage):
    predicate = 'both_in_bag'
    completion_event = 'both_in_bag_completed'


class WujiBagCinched(_WujiBagStage):
    completion_event = 'mouth_cinched_completed'

    def __init__(self, env):
        super().__init__(env)
        self._unlock_tick = None

    def handle_action_event(self, action, event):
        if event == ActionEvent.STARTED:
            self._unlock_tick = self.env.current_step
        super().handle_action_event(action, event)

    def _observe(self):
        observation = self.env.bag_observation
        valid = (self.env.current_step != self._unlock_tick
                 and observation['both_in_bag'] and observation['mouth_cinched'])
        return self._hold(valid, self.env.cinch_confirm_sim_s)
