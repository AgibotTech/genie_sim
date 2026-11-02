# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0

"""Check opt-in capture initialization without starting Kit."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def setup_capture(node):
    script = Path(__file__).resolve().parents[1] / "scripts" / "isaacsim_render.py"
    tree = ast.parse(script.read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "IsaacSimRenderNode")
    method = next(
        (item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "_setup_sync_capture"), None
    )
    assert method is not None, "capture setup must extend the original initialization sequence"
    namespace = {"json": json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(script), "exec"), namespace)
    namespace["_setup_sync_capture"](node)


def test_live_mode_does_not_import_capture(monkeypatch):
    monkeypatch.setitem(sys.modules, "_sync_capture", None)
    setup_capture(SimpleNamespace(_sync_topic=""))


@pytest.mark.parametrize("selection", [{}, {"head": "/missing"}, {"a": "/rgb", "b": "/rgb"}])
def test_invalid_sync_cameras_clean_up_before_raising(monkeypatch, selection):
    events = []
    monkeypatch.setitem(sys.modules, "_sync_capture", SimpleNamespace(SyncObservationRenderer=lambda *a, **kw: None))
    node = SimpleNamespace(
        _sync_topic="/observations",
        get_parameter=lambda name: SimpleNamespace(value=json.dumps(selection)),
        _manifest=SimpleNamespace(cameras=[SimpleNamespace(topic="/rgb")]),
        shutdown=lambda: events.append("shutdown"),
        destroy_node=lambda: events.append("destroy"),
    )
    with pytest.raises(ValueError, match="distinct configured manifest cameras"):
        setup_capture(node)
    assert events == ["shutdown", "destroy"]


def test_sync_cameras_keep_manifest_paths_and_sizes(monkeypatch):
    captured = {}

    def create(*args, **kwargs):
        captured.update(cameras=args[2], apply=args[3], completed=kwargs["completed_capture"])
        return captured

    monkeypatch.setitem(sys.modules, "_sync_capture", SimpleNamespace(SyncObservationRenderer=create))
    node = SimpleNamespace(
        _sync_topic="/observations",
        get_parameter=lambda name: SimpleNamespace(value='{"head": "/rgb"}'),
        _manifest=SimpleNamespace(cameras=[SimpleNamespace(topic="/rgb", width=640, height=400)]),
        _camera_prim_path=lambda camera: "/World/Head/Camera",
        _apply_observation_snapshot=object(),
        _record_material_capture=object(),
    )
    setup_capture(node)
    assert captured["cameras"] == {"head": ("/World/Head/Camera", (640, 400))}
    assert node._sync_capture is captured
    assert captured["apply"] is node._apply_observation_snapshot
    assert captured["completed"] is node._record_material_capture
