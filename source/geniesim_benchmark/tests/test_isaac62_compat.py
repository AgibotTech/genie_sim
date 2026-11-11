# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise the production compatibility functions without importing Kit."""

import ast
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src/geniesim_benchmark"


def load_function(path, name, namespace=None):
    tree = ast.parse(path.read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "playing,capture_on_play,stepped,error,resumed",
    [
        (True, True, True, None, True),
        (False, True, True, None, False),
        (True, False, True, None, False),
        (True, True, False, None, False),
        (True, True, True, "step", True),
        (True, True, True, "read", True),
    ],
)
def test_calibrated_capture_restores_continuous_playback(
    monkeypatch, playing, capture_on_play, stepped, error, resumed
):
    events = []
    observation = object()
    stepped_status = object()

    def step(**kwargs):
        events.append("step")
        assert kwargs == {"rt_subframes": 1, "pause_timeline": False, "delta_time": 0.0, "wait_for_render": True}
        if error == "step":
            raise RuntimeError("capture failed")

    def read(*args):
        events.append("read")
        if error == "read":
            raise RuntimeError("capture failed")
        return observation

    omni = ModuleType("omni")
    replicator = ModuleType("omni.replicator")
    core = ModuleType("omni.replicator.core")
    omni.replicator = replicator
    replicator.core = core
    core.orchestrator = SimpleNamespace(
        step=step,
        resume=lambda: events.append("resume"),
        get_status=lambda: stepped_status if stepped else object(),
        Status=SimpleNamespace(STEPPED=stepped_status),
    )
    carb = ModuleType("carb")
    carb.settings = SimpleNamespace(get_settings=lambda: SimpleNamespace(get_as_bool=lambda name: capture_on_play))
    for name, module in (
        ("omni", omni),
        ("omni.replicator", replicator),
        ("omni.replicator.core", core),
        ("carb", carb),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    owner = SimpleNamespace(
        want_camera_calibration=True,
        is_vec_mode=False,
        ui_builder=SimpleNamespace(my_world=SimpleNamespace(is_playing=lambda: playing)),
        run_on_render_loop=lambda callback: callback(),
        _get_obs_bundle=read,
    )
    capture = load_function(SOURCE / "app/controllers/api_core.py", "get_obs_bundle")
    if error:
        with pytest.raises(RuntimeError, match="capture failed"):
            capture(owner, image_dirs=["camera"], depth_dirs=["camera"])
    else:
        assert capture(owner, image_dirs=["camera"], depth_dirs=["camera"]) is observation
    expected = ["step"] + ([] if error == "step" else ["read"]) + (["resume"] if resumed else [])
    assert events == expected


@pytest.mark.parametrize("modern", [False, True])
def test_websocket_keepalive_does_not_leak_into_legacy_socket_kwargs(modern):
    calls = []

    def legacy_connect(uri, *, open_timeout=None, **socket_kwargs):
        assert "ping_interval" not in socket_kwargs and "ping_timeout" not in socket_kwargs
        calls.append((uri, open_timeout, socket_kwargs))
        return "connected"

    def modern_connect(uri, *, open_timeout=None, ping_interval=None, ping_timeout=None, **socket_kwargs):
        assert (ping_interval, ping_timeout) == (20, 60)
        calls.append((uri, open_timeout, socket_kwargs))
        return "connected"

    client = SimpleNamespace(connect=modern_connect if modern else legacy_connect)
    connect = load_function(
        SOURCE / "utils/comm/websocket_client.py",
        "ws_connect_compat",
        {"inspect": inspect, "Any": Any, "websockets": SimpleNamespace(sync=SimpleNamespace(client=client))},
    )
    assert (
        connect(
            "ws://localhost:8999", open_timeout=30, ping_interval=20, ping_timeout=60, source_address=("127.0.0.1", 0)
        )
        == "connected"
    )
    assert calls == [("ws://localhost:8999", 30, {"source_address": ("127.0.0.1", 0)})]
