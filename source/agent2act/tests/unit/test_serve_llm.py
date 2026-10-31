# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The convenience launcher preserves configured routing and explicit overrides."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", [None, "auto", "motion", "policy"])
def test_launcher_routing(mode, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("serve_llm", root / "scripts/serve_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "run", lambda args: calls.append(args) or 0)
    args = [
        "serve_llm",
        "--config",
        str(root / "examples/demo.toml"),
        "--log-file",
        str(tmp_path / "agent.jsonl"),
        "--no-trace",
    ]
    if mode is not None:
        args += ["--mode", mode]
    monkeypatch.setattr(sys, "argv", args)
    assert module.main() == 0
    if mode is None:
        assert "--mode" not in calls[0]
    else:
        assert calls[0][calls[0].index("--mode") + 1] == mode


def test_launcher_requires_configured_api_key_env(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("serve_llm_key", root / "scripts/serve_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("AGENT2ACT_LIVE_LLM_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_llm",
            "--config",
            str(root / "examples/g2_llm_recipe.json"),
            "--log-file",
            str(tmp_path / "agent.jsonl"),
            "--no-trace",
        ],
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_launcher_forwards_independent_observation_flags(enabled, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("serve_llm_flags", root / "scripts/serve_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "run", lambda args: calls.append(args) or 0)
    flags = ["--" + ("" if enabled else "no-") + name for name in ("depth", "gripper-center", "height-prior")]
    calibration = str(tmp_path / "heights.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_llm",
            "--config",
            str(root / "examples/demo.toml"),
            "--no-trace",
            *flags,
            "--height-calibration",
            calibration,
        ],
    )
    assert module.main() == 0
    assert all(flag in calls[0] for flag in flags)
    assert calls[0][calls[0].index("--height-calibration") + 1] == calibration


def test_launcher_forwards_cluster_listen_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT2ACT_LIVE_LLM_KEY", "offline-launcher-test")
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("serve_llm_cluster", root / "scripts/serve_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "run", lambda args: calls.append(args) or 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_llm",
            "--config",
            str(root / "examples/g2_llm_recipe.json"),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--depth",
            "--no-trace",
            "--log-file",
            str(tmp_path / "agent.jsonl"),
        ],
    )
    assert module.main() == 0
    assert calls[0][calls[0].index("--host") + 1] == "0.0.0.0"
    assert calls[0][calls[0].index("--port") + 1] == "8000"
    assert "--depth" in calls[0]


def test_launcher_ui_requires_trace_and_forwards_options(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("serve_llm_ui", root / "scripts/serve_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "run", lambda args: calls.append(args) or 0)
    args = [
        "serve_llm",
        "--config",
        str(root / "examples/demo.toml"),
        "--ui",
        "--no-browser",
        "--ui-port",
        "9123",
        "--log-file",
        str(tmp_path / "run.jsonl"),
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert module.main() == 0
    assert "--ui" in calls[0] and "--no-browser" in calls[0]
    assert calls[0][calls[0].index("--ui-port") + 1] == "9123"
    assert "--trace-dir" in calls[0]
    monkeypatch.setattr(sys, "argv", args + ["--no-trace"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert len(calls) == 1
