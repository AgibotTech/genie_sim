# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise production step/reset methods without loading Newton, CUDA or ROS."""

from __future__ import annotations

import ast
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

_ENGINE = Path(__file__).resolve().parents[1] / "engine/newton"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_plugin = _load("snapshot_plugin", _ENGINE / "plugin.py")
_lifecycle = _load("snapshot_lifecycle", _ENGINE / "lifecycle.py")


class _Array:
    device = "cuda:0"

    def __init__(self, value=0):
        self.value = value

    def numpy(self):
        return np.array([self.value])


class _Mirror:
    def __init__(self):
        self.values = []
        self.synchronize = Mock()

    def refresh(self, src, stream):
        self.values.append(src.value)

    def read(self):
        return np.array([self.values[-2]]) if len(self.values) >= 2 else None


@pytest.fixture
def sim(monkeypatch):
    warp = SimpleNamespace(get_stream=Mock(return_value=None), synchronize_device=Mock())
    monkeypatch.setitem(sys.modules, "warp", warp)
    monkeypatch.setitem(sys.modules, "newton", SimpleNamespace(eval_fk=Mock()))
    mirror_factory = Mock(side_effect=_Mirror)
    monkeypatch.setitem(
        sys.modules,
        "genie_sim_engine_newton.engine.newton.async_mirror",
        SimpleNamespace(AsyncMirror=mirror_factory),
    )
    # Importing engine_base would load the real GPU runtime. Compile just its
    # production methods; this still tests their complete control flow.
    path = _ENGINE / "engine_base.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_NewtonStandaloneBase")
    names = {"_update_record_snapshot_mirror", "step", "tick_extras"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"time": time, "wp": warp}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    engine_type = type(
        "SnapshotEngine", (_plugin._PluginMixin, _lifecycle._LifecycleMixin), {name: namespace[name] for name in names}
    )
    engine = engine_type()
    engine._model = object()
    engine._state_0 = SimpleNamespace(**{name: _Array() for name in ("joint_q", "joint_qd", "body_q", "particle_q")})
    engine._state_1 = None
    engine._control = None
    engine._adapter = None
    engine._paused = False
    engine._cloth_solver = None
    engine._cuda_graph = None
    engine._scene_plugin = None
    engine._pq_mirror = None
    engine._jq_mirror, engine._jqd_mirror, engine._bq_mirror = _Mirror(), _Mirror(), _Mirror()
    engine._simulate_substeps = Mock()
    engine._plugin_ctx = Mock(return_value=None)
    engine._logger = SimpleNamespace(warn=Mock())
    engine.get_body_transforms = Mock(return_value=(np.array([[1, 2, 3, 1, 0, 0, 0]]), ["/World/pot"]))
    engine._physics_step_event = None
    engine._deformable_pub = engine._deformable_pc_pub = engine._object_pub = None
    engine._render_hz = 0
    engine._t_log_start = time.monotonic()
    for field in (
        "_sim_time_acc",
        "_step_ms_acc",
        "_step_ms_max",
        "_frame_id",
        "_writeback_ms_acc",
        "_writeback_ms_max",
        "_tick_count",
    ):
        setattr(engine, field, 0)
    engine.mirror_factory = mirror_factory
    return engine


def _tick(sim, value):
    sim._state_0.particle_q.value = value
    sim.step(0.01, 0.0)
    sim.tick_extras()


@pytest.mark.parametrize(
    "plugin",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(record_snapshot_enabled=lambda: True),
        SimpleNamespace(on_record_snapshot=Mock(), record_snapshot_enabled=lambda: False),
    ],
)
def test_no_consumer_skips_particle_allocation_and_body_read(sim, plugin):
    sim._scene_plugin = plugin
    for value in range(3):
        _tick(sim, value)
    sim.mirror_factory.assert_not_called()
    sim.get_body_transforms.assert_not_called()
    assert sim._pq_mirror is None
    assert len(sim._jq_mirror.values) == len(sim._jqd_mirror.values) == len(sim._bq_mirror.values) == 3
    if plugin is not None and hasattr(plugin, "on_record_snapshot"):
        plugin.on_record_snapshot.assert_not_called()


@pytest.mark.parametrize("legacy", [False, True])
def test_enabled_and_legacy_consumers_receive_delayed_snapshots(sim, legacy):
    callback = Mock()
    sim._scene_plugin = SimpleNamespace(on_record_snapshot=callback)
    if not legacy:
        sim._scene_plugin.record_snapshot_enabled = lambda: True
    _tick(sim, 11)
    _tick(sim, 22)
    sim.mirror_factory.assert_called_once()
    assert callback.call_args_list[0].args[0] is None
    assert callback.call_args_list[1].args[0].tolist() == [11]
    assert callback.call_args_list[1].args[1].tolist() == [[1, 2, 3, 1, 0, 0, 0]]
    assert callback.call_args_list[1].args[2:4] == (["/World/pot"], 0.02)
    sim._pq_mirror.synchronize.assert_not_called()


def test_disable_retires_mirror_and_reenable_has_no_stale_snapshot(sim):
    active = [True]
    callback = Mock()
    sim._scene_plugin = SimpleNamespace(on_record_snapshot=callback, record_snapshot_enabled=lambda: active[0])
    _tick(sim, 1)
    _tick(sim, 2)
    retired = sim._pq_mirror
    active[0] = False
    _tick(sim, 3)
    assert sim._pq_mirror is None
    retired.synchronize.assert_called_once()
    assert retired.values == [1, 2]
    assert callback.call_count == 2
    active[0] = True
    _tick(sim, 4)
    assert sim._pq_mirror is not retired
    assert callback.call_args.args[0] is None
    _tick(sim, 5)
    assert callback.call_args.args[0].tolist() == [4]


def test_disable_before_extras_skips_callback_and_body_read(sim):
    active = [True]
    sim._scene_plugin = SimpleNamespace(on_record_snapshot=Mock(), record_snapshot_enabled=lambda: active[0])
    sim.step(0.01, 0.0)
    retired = sim._pq_mirror
    active[0] = False
    sim.tick_extras()
    retired.synchronize.assert_called_once()
    sim.get_body_transforms.assert_not_called()
    sim._scene_plugin.on_record_snapshot.assert_not_called()


@pytest.mark.parametrize("active", [False, True])
def test_reset_flushes_only_active_particle_mirror(sim, active):
    sim._scene_plugin = SimpleNamespace(on_record_snapshot=Mock(), record_snapshot_enabled=lambda: True)
    _tick(sim, 1)
    mirror = sim._pq_mirror
    sim._scene_plugin.record_snapshot_enabled = lambda: active
    sim._state_0.particle_q.value = 99
    sim._commit_state_write([], None, None)
    sim._logger.warn.assert_not_called()
    assert len(sim._jq_mirror.values) == len(sim._jqd_mirror.values) == len(sim._bq_mirror.values) == 3
    if active:
        assert sim._pq_mirror is mirror
        assert mirror.read().tolist() == [99]
        mirror.synchronize.assert_not_called()
    else:
        assert sim._pq_mirror is None
        assert mirror.values == [1]
        mirror.synchronize.assert_called_once()


def test_reset_without_a_consumer_does_not_allocate_particle_mirror(sim):
    sim._commit_state_write([], None, None)
    sim.mirror_factory.assert_not_called()
    sim._logger.warn.assert_not_called()
    assert len(sim._jq_mirror.values) == len(sim._jqd_mirror.values) == len(sim._bq_mirror.values) == 2


@pytest.mark.parametrize("failed", [False, True])
def test_retirement_waits_for_pending_copies(monkeypatch, failed):
    warp = SimpleNamespace(synchronize=Mock(), synchronize_event=Mock())
    monkeypatch.setitem(sys.modules, "warp", warp)
    module = _load("snapshot_async_mirror", _ENGINE / "async_mirror.py")
    mirror = module.AsyncMirror()
    events = [object(), object()]
    mirror._events = events
    mirror._failed = failed
    mirror.synchronize()
    if failed:
        warp.synchronize.assert_called_once()
        warp.synchronize_event.assert_not_called()
    else:
        assert [call.args[0] for call in warp.synchronize_event.call_args_list] == events
        warp.synchronize.assert_not_called()
