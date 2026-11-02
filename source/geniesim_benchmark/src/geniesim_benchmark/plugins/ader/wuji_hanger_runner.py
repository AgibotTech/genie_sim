# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run the ordinary parallel hanger ADER tree on validated physical frames.

The caller verifies cloth topology and rigid asset identity against the profile.
This adapter validates each observation before committing clocks or progress.
"""
from copy import deepcopy
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from geniesim_benchmark.plugins.ader.ader_base import AderEnv, AderParams, AderTask
from geniesim_benchmark.plugins.ader.action.action_parsing import parse_action
from geniesim_benchmark.plugins.ader.geometry.wuji_hanger import measure_hanger
from geniesim_benchmark.plugins.output_system.eval_utils import TASK_STEPS, score_layout_steps

_CONFIG = Path(__file__).resolve().parents[2] / 'benchmark/config/llm_task/wuji_ipc/cloth_hanger'
_PREDICATES = ('cloth_lifted', 'hanger_lifted', 'near_collar', 'inserted', 'hung')


class _HangerEnv(AderEnv):
    def __init__(self, confirm_sim_s, hung_confirm_sim_s):
        super().__init__(None, AderParams(task_name='cloth_hanger'))
        self.task = AderTask(self, str(_CONFIG))
        self.task.task_progress = []
        self.current_step = None
        self.hanger_sim_time_s = None
        self.hanger_confirm_sim_s = confirm_sim_s
        self.hung_confirm_sim_s = hung_confirm_sim_s
        self.hanger_observation = None
        self.hanger_events = []

    def hanger_event(self, event):
        self.hanger_events.append({'event': event, 'tick': self.current_step,
                                   'sim_time_s': self.hanger_sim_time_s})


class WujiHangerRunner:
    def __init__(self, profile, measure=measure_hanger):
        try:
            for key in ('confirm_sim_s', 'hung_confirm_sim_s'):
                value = profile[key]
                if (isinstance(value, bool) or not isinstance(value, Real)
                        or not np.isfinite(value) or value < 0):
                    raise ValueError(f'{key} must be finite and nonnegative')
            count = profile['mesh']['vertex_count']
            if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
                raise ValueError('vertex_count must be a positive integer')
            collar = np.asarray(profile['regions']['collar'])
            if (collar.ndim != 1 or not collar.size
                    or not np.issubdtype(collar.dtype, np.integer)
                    or any(isinstance(index, (bool, np.bool_)) for index in profile['regions']['collar'])
                    or np.any(collar < 0) or np.any(collar >= count)):
                raise ValueError('collar must contain valid material vertex indices')
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError('invalid hanger runner profile') from exc
        self.profile = deepcopy(profile)
        self.measure = measure
        self.env = _HangerEnv(float(profile['confirm_sim_s']), float(profile['hung_confirm_sim_s']))
        problem = json.loads((_CONFIG / 'problems.json').read_text())['problem1']
        self.env.execute_action = parse_action(problem['Acts'][0], self.env.task.task_progress, self.env)
        self._last_points = None
        self._last_pose = None
        self._baseline = None
        self._diagnostics = {}
        self._started = False

    def update(self, tick, sim_time_s, points, hanger_pose):
        """Consume one distinct frame, ignoring exact duplicates without credit."""
        if isinstance(tick, bool) or not isinstance(tick, Integral) or tick < 0:
            raise ValueError('tick must be a nonnegative integer')
        if (isinstance(sim_time_s, bool) or not isinstance(sim_time_s, Real)
                or not np.isfinite(sim_time_s) or sim_time_s < 0):
            raise ValueError('sim_time_s must be finite and nonnegative')
        try:
            points = np.asarray(points, dtype=float)
            hanger_pose = np.asarray(hanger_pose, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('invalid cloth points or hanger pose') from exc
        count = self.profile['mesh']['vertex_count']
        if points.shape != (count, 3) or not np.isfinite(points).all():
            raise ValueError(f'expected finite cloth points with shape ({count}, 3)')
        if hanger_pose.shape != (4, 4) or not np.isfinite(hanger_pose).all():
            raise ValueError('expected finite hanger pose with shape (4, 4)')
        rotation = hanger_pose[:3, :3]
        if (not np.allclose(hanger_pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
            raise ValueError('hanger pose must be a proper rigid column transform')
        previous_tick = self.env.current_step
        previous_time = self.env.hanger_sim_time_s
        if previous_tick is not None:
            if tick < previous_tick or sim_time_s < previous_time:
                raise ValueError('physical frame tick and simulation time must be monotonic')
            if tick == previous_tick:
                if (sim_time_s != previous_time or not np.array_equal(points, self._last_points)
                        or not np.array_equal(hanger_pose, self._last_pose)):
                    raise ValueError('conflicting duplicate physical frame')
                return
        baseline = self._baseline
        if baseline is None:
            baseline = {'collar_height_m': float(np.median(points[self.profile['regions']['collar'], 2])),
                        'hanger_height_m': float(hanger_pose[2, 3])}
        report = self.measure(points, hanger_pose, self.profile, deepcopy(baseline))
        if (not isinstance(report, dict)
                or any(type(report.get(key)) is not bool for key in _PREDICATES)
                or not isinstance(report.get('diagnostics'), dict)):
            raise ValueError('invalid hanger geometry report')
        diagnostics = report['diagnostics']
        if any(isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
               for value in diagnostics.values()):
            raise ValueError('hanger diagnostics must contain finite numbers')
        self._baseline = baseline
        self.env.current_step = int(tick)
        self.env.hanger_sim_time_s = float(sim_time_s)
        self.env.hanger_observation = report
        self._last_points = points.copy()
        self._last_pose = hanger_pose.copy()
        self._diagnostics = deepcopy(diagnostics)
        if not self._started:
            self.env.do_eval_action()
            self._started = True
        else:
            self.env.action_executor.update(float(sim_time_s - previous_time))
        if self.env.execute_action.is_finished():
            self.env.cancel_eval()

    def result(self):
        """Return the five equal latched item scores in the standard layout."""
        stages = score_layout_steps(self.env.task.task_progress, TASK_STEPS['cloth_hanger'])
        return {'score': sum(stage['score'] for stage in stages) / 5,
                'success': bool(all(stage['score'] == 1 for stage in stages)
                                and self.env.execute_action.is_finished()),
                'stages': stages, 'diagnostics': deepcopy(self._diagnostics),
                'events': deepcopy(self.env.hanger_events)}
