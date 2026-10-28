# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Stubs for the publish-path tests.

These drive ``loop._publish_tick`` directly rather than through
``EngineRunLoop.spin()``: the thing under test is which publisher gets
called with which arrays, and a real tick loop would only add a clock to
script.  Nothing here touches rclpy — ``_core`` is a recorder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make `genie_sim_engine_core` importable without going through
# ament_python_install (colcon test runs before install by default).
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from genie_sim_engine_core.common import loop as loop_mod  # noqa: E402
from genie_sim_engine_core.engine.publish_plan import PublishPlan  # noqa: E402


class RecordingCore:
    """``_core`` stand-in that records every publish call verbatim."""

    def __init__(self) -> None:
        self.joint_states = []  # (sim_time, pos, vel, eff, frame_id)
        self.joint_states_scene = []  # (sim_time, names, pos, vel, eff)

    def publish_clock(self, t):
        pass

    def publish_joint_states(self, t, pos, vel, eff, frame_id=""):
        self.joint_states.append((t, np.asarray(pos), np.asarray(vel), eff, frame_id))

    def publish_joint_states_scene(self, t, names, pos, vel, eff):
        self.joint_states_scene.append((t, list(names), np.asarray(pos), np.asarray(vel), eff))

    def publish_body_tf_scene(self, *a):
        pass

    def publish_tf_render(self, *a):
        pass

    def publish_odom(self, *a):
        pass

    def publish_contact_states(self, *a):
        pass


class PublishSim:
    """Backend stub exposing exactly what ``_publish_tick`` reads.

    ``joint_states`` is the FULL engine joint vector the backend hands over.
    Whether the scene DoFs are in it is the whole question the two routing
    paths answer differently, so each test builds it accordingly.

    The routing decision itself is NOT here — it travels as a ``PublishPlan``
    argument, so a stub cannot accidentally satisfy the publish path by
    growing an attribute name.
    """

    def __init__(self, jpos, jvel, jeff=None) -> None:
        # ``_publish_tick`` stamps this into /joint_states header.frame_id as
        # the reset credential; 0 is the ABC default, "no reset completed".
        self.reset_epoch = 0
        self._jpos = np.asarray(jpos, dtype=np.float64)
        self._jvel = np.asarray(jvel, dtype=np.float64)
        self._jeff = None if jeff is None else np.asarray(jeff, dtype=np.float64)
        self.scene_reads = 0

    def get_joint_states(self):
        if self._jeff is None:
            return self._jpos, self._jvel
        return self._jpos, self._jvel, self._jeff

    def get_body_transforms(self):
        return np.zeros((0, 7), dtype=np.float64), []

    def get_odom(self, sim_time):
        return None

    def get_contact_states(self):
        return None

    def note_publish_phase(self, **kw):
        pass


@pytest.fixture
def core():
    return RecordingCore()


@pytest.fixture
def publish_tick():
    """``_publish_tick(sim, plan, sim_time, core)``, called directly."""
    return loop_mod._publish_tick


@pytest.fixture(name="PublishPlan")
def publish_plan_cls():
    """The routing value, so tests can build the shape each path needs."""
    return PublishPlan


@pytest.fixture(name="PublishSim")
def publish_sim_cls():
    """The backend stub class, so tests can build several shapes of it."""
    return PublishSim
