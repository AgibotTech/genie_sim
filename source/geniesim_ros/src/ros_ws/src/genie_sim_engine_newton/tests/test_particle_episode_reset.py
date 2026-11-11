# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Episode reset must restore food as well as the robot, in both state buffers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "genie_sim_engine_newton/engine/newton/lifecycle.py"
spec = importlib.util.spec_from_file_location("newton_lifecycle_under_test", SOURCE)
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)


class Array:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32).copy()
        self.fail = False

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        if self.fail:
            raise RuntimeError("particle restore failed")
        self.value[:] = value

    def zero_(self):
        self.assign(np.zeros_like(self.value))


class Engine(lifecycle._LifecycleMixin):
    def __init__(self):
        self.messages = []
        self._logger = SimpleNamespace(info=self.messages.append, warn=self.messages.append)
        self._num_envs = 1
        self._scene_cfg = {}
        self._jindex = None
        self._adapter = SimpleNamespace()
        points = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self._model = SimpleNamespace(joint_q=Array([0.2]), particle_q=Array(points))
        self._control = SimpleNamespace(joint_target_q=Array([0.2]))
        self._state_0 = self.state(points)
        self._state_1 = self.state(points)
        self._snapshot_reset_state()

    @staticmethod
    def state(points):
        return SimpleNamespace(
            joint_q=Array([0.2]),
            joint_qd=Array([0]),
            body_q=None,
            particle_q=Array(points),
            particle_qd=Array(np.zeros((2, 3))),
            particle_f=Array(np.zeros((2, 3))),
        )


class ParticleResetTests(unittest.TestCase):
    def setUp(self):
        self.imports = patch.dict(
            "sys.modules",
            {
                "newton": SimpleNamespace(eval_fk=lambda *args: None),
                "warp": SimpleNamespace(synchronize_device=lambda: None),
            },
        )
        self.imports.start()
        self.addCleanup(self.imports.stop)

    def test_scattered_food_restores_in_place_and_can_step_then_reset_again(self):
        engine = Engine()
        engine.episode_reset()
        expected = engine._state_0.particle_q.numpy()
        arrays = [(s.particle_q, s.particle_qd, s.particle_f) for s in (engine._state_0, engine._state_1)]
        for attempt in range(2):
            for state in (engine._state_0, engine._state_1):
                state.particle_q.assign(expected + 3 + attempt)
                state.particle_qd.assign(np.full((2, 3), 4))
                state.particle_f.assign(np.full((2, 3), 5))
            engine.episode_reset()
            for state, original in zip((engine._state_0, engine._state_1), arrays):
                np.testing.assert_array_equal(state.particle_q.numpy(), expected)
                np.testing.assert_array_equal(state.particle_qd.numpy(), np.zeros((2, 3)))
                np.testing.assert_array_equal(state.particle_f.numpy(), np.zeros((2, 3)))
                self.assertIs(state.particle_q, original[0])
                self.assertIs(state.particle_qd, original[1])
                self.assertIs(state.particle_f, original[2])
        self.assertEqual(engine._reset_epoch, 3)

    def test_failed_food_restore_does_not_acknowledge_a_new_epoch(self):
        engine = Engine()
        engine.episode_reset()
        before = engine._reset_epoch
        engine._state_1.particle_q.fail = True
        engine.episode_reset()
        self.assertEqual(engine._reset_epoch, before)
        self.assertTrue(any("particle restore failed" in item for item in engine.messages))

    def test_nonfinite_settled_food_cannot_replace_a_clean_baseline(self):
        engine = Engine()
        before = engine._reset_init_joint_q.copy()
        engine._state_0.joint_q.assign([123])
        engine._state_0.particle_q.assign(np.full((2, 3), np.nan))
        engine.resnapshot_settled()
        np.testing.assert_array_equal(engine._reset_init_joint_q, before)

    def test_coupled_feedback_is_reset_without_overwriting_authored_state_then_control_syncs(self):
        engine = Engine()
        calls = []

        def reset(state, **options):
            self.assertIs(state, engine._state_0)
            self.assertEqual(options, {"flags": 0})
            np.testing.assert_array_equal(state.particle_qd.numpy(), np.zeros((2, 3)))
            calls.append("coupled")

        engine._coupled_solver = SimpleNamespace(reset=reset)
        engine._adapter = SimpleNamespace(
            sync_state_into_solver=lambda *args: calls.append("state"),
            sync_control_into_solver=lambda *args: calls.append("control"),
        )
        engine.episode_reset()
        self.assertEqual(calls, ["coupled", "state", "control"])
        self.assertEqual(engine._reset_epoch, 1)

    def test_coupled_reset_failure_cannot_acknowledge_a_new_epoch(self):
        engine = Engine()

        def fail(*args, **kwargs):
            raise RuntimeError("coupling reset failed")

        engine._coupled_solver = SimpleNamespace(reset=fail)
        engine.episode_reset()
        self.assertEqual(engine._reset_epoch, 0)
        self.assertTrue(any("coupling reset failed" in message for message in engine.messages))


if __name__ == "__main__":
    unittest.main()
