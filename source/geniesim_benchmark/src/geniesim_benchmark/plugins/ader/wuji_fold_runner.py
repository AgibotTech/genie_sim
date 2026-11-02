# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run the ordinary fold ADER tree on validated IPC physical frames.

The caller verifies cloth path and topology against profile['mesh']; this
adapter validates frame identity, simulation time and world-space vertices.
It deliberately does not use wall-clock ADER action_update() for scoring.
"""
from copy import deepcopy
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from geniesim_benchmark.plugins.ader.ader_base import AderEnv, AderParams, AderTask
from geniesim_benchmark.plugins.ader.action.action_parsing import parse_action
from geniesim_benchmark.plugins.ader.geometry.wuji_fold import measure_fold
from geniesim_benchmark.plugins.output_system.eval_utils import TASK_STEPS, score_layout_steps

_CONFIG = Path(__file__).resolve().parents[2] / 'benchmark/config/llm_task/wuji_ipc/fold_cloth'


class _FoldEnv(AderEnv):
    def __init__(self, confirm_sim_s):
        super().__init__(None, AderParams(task_name='fold_cloth'))
        self.task = AderTask(self, str(_CONFIG))
        self.task.task_progress = []
        self.current_step = None
        self.fold_sim_time_s = None
        self.fold_confirm_sim_s = confirm_sim_s
        self.fold_observation = None
        self.fold_events = []

    def fold_event(self, event):
        self.fold_events.append({'event': event, 'tick': self.current_step,
                                 'sim_time_s': self.fold_sim_time_s})


class WujiFoldRunner:
    def __init__(self, profile, measure=measure_fold):
        try:
            confirm = profile['confirm_sim_s']
            count = profile['mesh']['vertex_count']
            if (isinstance(confirm, bool) or not isinstance(confirm, Real)
                    or not np.isfinite(confirm) or confirm < 0):
                raise ValueError('confirm_sim_s must be finite and nonnegative')
            if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
                raise ValueError('vertex_count must be a positive integer')
        except (KeyError, TypeError) as exc:
            raise ValueError('invalid fold runner profile') from exc
        self.profile = deepcopy(profile)
        self.measure = measure
        self.env = _FoldEnv(float(confirm))
        problem = json.loads((_CONFIG / 'problems.json').read_text())['problem1']
        self.env.execute_action = parse_action(problem['Acts'][0], self.env.task.task_progress, self.env)
        self._last_points = None
        self._diagnostics = {}
        self._started = False

    def update(self, tick, sim_time_s, points):
        """Consume one physical frame; identical repeated frames are ignored.

        Conflicting duplicates, reversed clocks and malformed observations raise
        ValueError before advancing any checker. Distinct ticks may share a
        simulation time, but cannot accrue confirmation duration that way.
        """
        if isinstance(tick, bool) or not isinstance(tick, Integral) or tick < 0:
            raise ValueError('tick must be a nonnegative integer')
        if (isinstance(sim_time_s, bool) or not isinstance(sim_time_s, Real)
                or not np.isfinite(sim_time_s) or sim_time_s < 0):
            raise ValueError('sim_time_s must be finite and nonnegative')
        try:
            points = np.asarray(points, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('invalid cloth world points') from exc
        count = self.profile['mesh']['vertex_count']
        if points.shape != (count, 3) or not np.isfinite(points).all():
            raise ValueError(f'expected finite cloth points with shape ({count}, 3)')
        previous_tick = self.env.current_step
        previous_time = self.env.fold_sim_time_s
        if previous_tick is not None:
            if tick < previous_tick or sim_time_s < previous_time:
                raise ValueError('physical frame tick and simulation time must be monotonic')
            if tick == previous_tick:
                if sim_time_s != previous_time or not np.array_equal(points, self._last_points):
                    raise ValueError('conflicting duplicate physical frame')
                return
        report = self.measure(points, self.profile)
        if (not isinstance(report, dict)
                or any(type(report.get(key)) is not bool for key in ('left', 'right', 'final'))
                or not isinstance(report.get('diagnostics'), dict)):
            raise ValueError('invalid fold geometry report')
        diagnostics = report['diagnostics']
        if any(isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
               for value in diagnostics.values()):
            raise ValueError('fold diagnostics must contain finite numbers')
        self.env.current_step = int(tick)
        self.env.fold_sim_time_s = float(sim_time_s)
        self.env.fold_observation = report
        self._last_points = points.copy()
        self._diagnostics = deepcopy(diagnostics)
        if not self._started:
            self.env.do_eval_action()
            self._started = True
        else:
            # ActionList dispatches only positive time slices. Leaves use the
            # physical simulation clock above, including a zero-duration frame.
            delta = max(float(sim_time_s - previous_time), np.finfo(float).eps)
            self.env.action_executor.update(delta)
        if self.env.execute_action.is_finished():
            self.env.cancel_eval()

    def result(self):
        """Return normalized score, fixed layout rows and latest diagnostics."""
        stages = score_layout_steps(self.env.task.task_progress, TASK_STEPS['fold_cloth'])
        return {'score': sum(stage['score'] for stage in stages) / 3,
                'success': bool(stages[-1]['score'] == 1 and self.env.execute_action.is_finished()),
                'stages': stages, 'diagnostics': deepcopy(self._diagnostics),
                'events': deepcopy(self.env.fold_events)}
