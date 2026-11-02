# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run the ordinary bag ADER tree on validated, monotonic physical frames."""
from copy import deepcopy
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from geniesim_benchmark.plugins.ader.ader_base import AderEnv, AderParams, AderTask
from geniesim_benchmark.plugins.ader.action.action_parsing import parse_action
from geniesim_benchmark.plugins.ader.geometry.wuji_bag import measure_bag
from geniesim_benchmark.plugins.output_system.eval_utils import TASK_STEPS, score_layout_steps

_CONFIG = Path(__file__).resolve().parents[2] / 'benchmark/config/llm_task/wuji_ipc/drawstring_bag'
_PREDICATES = ('bag_lifted', 'bag_in_bin', 'bottle1_in_bin', 'bottle2_in_bin',
               'both_in_bag', 'mouth_cinched')


class _BagEnv(AderEnv):
    def __init__(self, confirm_sim_s, cinch_confirm_sim_s):
        super().__init__(None, AderParams(task_name='drawstring_bag'))
        self.task = AderTask(self, str(_CONFIG))
        self.task.task_progress = []
        self.current_step = None
        self.bag_sim_time_s = None
        self.bag_confirm_sim_s = confirm_sim_s
        self.cinch_confirm_sim_s = cinch_confirm_sim_s
        self.bag_observation = None
        self.bag_events = []

    def bag_event(self, event):
        self.bag_events.append({'event': event, 'tick': self.current_step,
                                'sim_time_s': self.bag_sim_time_s})


class WujiBagRunner:
    def __init__(self, profile, measure=measure_bag):
        try:
            for key in ('confirm_sim_s', 'cinch_confirm_sim_s'):
                value = profile[key]
                if (isinstance(value, bool) or not isinstance(value, Real)
                        or not np.isfinite(value) or value < 0):
                    raise ValueError(f'{key} must be finite and nonnegative')
            count = profile['mesh']['vertex_count']
            if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
                raise ValueError('vertex_count must be a positive integer')
            for name in ('body', 'mouth'):
                region = np.asarray(profile['regions'][name])
                if (region.ndim != 1 or not region.size
                        or not np.issubdtype(region.dtype, np.integer)
                        or any(isinstance(index, (bool, np.bool_)) for index in profile['regions'][name])
                        or np.any(region < 0) or np.any(region >= count)):
                    raise ValueError(f'{name} must contain valid material vertex indices')
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError('invalid bag runner profile') from exc
        self.profile = deepcopy(profile)
        self.measure = measure
        self.env = _BagEnv(float(profile['confirm_sim_s']), float(profile['cinch_confirm_sim_s']))
        problem = json.loads((_CONFIG / 'problems.json').read_text())['problem1']
        self.env.execute_action = parse_action(problem['Acts'][0], self.env.task.task_progress, self.env)
        self._last_points = None
        self._last_poses = None
        self._baseline = None
        self._diagnostics = {}
        self._started = False

    def update(self, tick, sim_time_s, points, bottle_poses):
        """Validate before committing any baseline, clock, event or item credit."""
        if isinstance(tick, bool) or not isinstance(tick, Integral) or tick < 0:
            raise ValueError('tick must be a nonnegative integer')
        if (isinstance(sim_time_s, bool) or not isinstance(sim_time_s, Real)
                or not np.isfinite(sim_time_s) or sim_time_s < 0):
            raise ValueError('sim_time_s must be finite and nonnegative')
        try:
            points = np.asarray(points, dtype=float)
            bottle_poses = np.asarray(bottle_poses, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('invalid bag points or bottle poses') from exc
        count = self.profile['mesh']['vertex_count']
        if points.shape != (count, 3) or not np.isfinite(points).all():
            raise ValueError(f'expected finite bag points with shape ({count}, 3)')
        if bottle_poses.shape != (2, 4, 4) or not np.isfinite(bottle_poses).all():
            raise ValueError('expected finite bottle poses with shape (2, 4, 4)')
        for pose in bottle_poses:
            rotation = pose[:3, :3]
            if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
                    or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
                raise ValueError('bottle pose must be a proper rigid column transform')
        previous_tick = self.env.current_step
        previous_time = self.env.bag_sim_time_s
        if previous_tick is not None:
            if tick < previous_tick or sim_time_s < previous_time:
                raise ValueError('physical frame tick and simulation time must be monotonic')
            if tick == previous_tick:
                if (sim_time_s != previous_time or not np.array_equal(points, self._last_points)
                        or not np.array_equal(bottle_poses, self._last_poses)):
                    raise ValueError('conflicting duplicate physical frame')
                return
        baseline = self._baseline
        if baseline is None:
            mouth = points[self.profile['regions']['mouth']]
            with np.errstate(over='ignore', invalid='ignore'):
                baseline = {'body_height_m': float(np.median(points[self.profile['regions']['body'], 2])),
                            'mouth_span_m': float(np.linalg.norm(mouth[:, None] - mouth, axis=2).max())}
            if (not all(np.isfinite(value) for value in baseline.values())
                    or baseline['mouth_span_m'] <= 0):
                raise ValueError('initial body height and positive mouth span must be finite')
        report = self.measure(points, bottle_poses, self.profile, deepcopy(baseline))
        if (not isinstance(report, dict)
                or any(type(report.get(key)) is not bool for key in _PREDICATES)
                or not isinstance(report.get('diagnostics'), dict)):
            raise ValueError('invalid bag geometry report')
        diagnostics = report['diagnostics']
        if any(isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
               for value in diagnostics.values()):
            raise ValueError('bag diagnostics must contain finite numbers')
        self._baseline = baseline
        self.env.current_step = int(tick)
        self.env.bag_sim_time_s = float(sim_time_s)
        self.env.bag_observation = deepcopy(report)
        self._last_points = points.copy()
        self._last_poses = bottle_poses.copy()
        self._diagnostics = deepcopy(diagnostics)
        if not self._started:
            self.env.do_eval_action()
            self._started = True
        else:
            # ActionList dispatches only positive deltas. Leaves use the actual
            # physical clock, so a zero-time frame can reset a hold without
            # gaining any elapsed time from this dispatch epsilon.
            self.env.action_executor.update(max(float(sim_time_s - previous_time), 1e-12))
        if self.env.execute_action.is_finished():
            self.env.cancel_eval()

    def result(self):
        """Return all five equal, latched scores, including unstarted leaves."""
        stages = score_layout_steps(self.env.task.task_progress, TASK_STEPS['drawstring_bag'])
        return {'score': sum(stage['score'] for stage in stages) / 5,
                'success': bool(all(stage['score'] == 1 for stage in stages)
                                and self.env.execute_action.is_finished()),
                'stages': stages, 'diagnostics': deepcopy(self._diagnostics),
                'events': deepcopy(self.env.bag_events)}
