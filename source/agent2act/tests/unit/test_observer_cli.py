# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""UI switches do not change planner options or execution settings."""

from pathlib import Path

import pytest

from agent2act import cli
from agent2act.configuration import load_config

CONFIG = Path(__file__).resolve().parents[2] / "examples/demo.toml"


def test_ui_auto_trace_and_cleanup_after_service_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    closed, opened, served = [], [], []

    class Viewer:
        url = "http://127.0.0.1:12345/fixture/"

        def close(self):
            closed.append(True)

    def create(root, port):
        opened.append((root, port))
        return Viewer()

    def serve(factory, settings):
        served.append(settings)
        raise ValueError("fixture construction failure")

    monkeypatch.setattr(cli, "create_observer", create)
    monkeypatch.setattr(cli, "serve_application", serve)
    assert cli.run(["serve", "--config", str(CONFIG), "--ui", "--no-browser"]) == 2
    assert opened[0][0].is_dir() and opened[0][1] == 0
    assert closed == [True]
    original = load_config(CONFIG)
    assert served[0].planner == original.planner
    assert served[0].execution == original.execution


def test_no_ui_does_not_create_observer_and_failure_is_optional(tmp_path, monkeypatch):
    served = []

    def fail(*args):
        raise OSError("occupied port")

    monkeypatch.setattr(cli, "create_observer", fail)
    monkeypatch.setattr(cli, "serve_application", lambda factory, settings: served.append(settings))
    assert cli.run(["serve", "--config", str(CONFIG)]) == 0
    assert served[0].logging.trace_dir is None
    assert (
        cli.run(["serve", "--config", str(CONFIG), "--ui", "--no-browser", "--trace-dir", str(tmp_path)]) == 0
    )
    assert len(served) == 2


def test_offline_observe_does_not_construct_planner(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_observe", lambda *args: calls.append(args))
    monkeypatch.setattr(
        cli, "load_config", lambda *args, **kwargs: pytest.fail("must not load planner config")
    )
    assert cli.run(["observe", "--trace-dir", str(tmp_path), "--no-browser"]) == 0
    assert calls == [(tmp_path, 0, True)]
    assert cli.run(["observe", "--trace-dir", str(tmp_path), "--ui-port", "-1"]) == 2


def test_process_construction_failure_closes_pipes(tmp_path, monkeypatch):
    from agent2act.service.observer import process

    closed = []

    class Pipe:
        def close(self):
            closed.append(self)

    endpoints = (Pipe(), Pipe())

    class Context:
        def Pipe(self, **kwargs):
            return endpoints

        def Process(self, **kwargs):
            raise RuntimeError("fixture process construction failure")

    monkeypatch.setattr(process.multiprocessing, "get_context", lambda method: Context())
    with pytest.raises(RuntimeError, match="construction"):
        process.ObserverProcess(tmp_path)
    assert set(closed) == set(endpoints)
