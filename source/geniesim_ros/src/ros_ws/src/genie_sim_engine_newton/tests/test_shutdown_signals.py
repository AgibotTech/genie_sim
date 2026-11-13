# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Real process signals must unwind the engine and preserve cleanup."""

import ast
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from genie_sim_engine_newton.common.shutdown_signals import shutdown_signals


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_inherited_ignored_signal_unwinds_once_and_finishes_cleanup(signum):
    source = """
import os, signal, time
from genie_sim_engine_newton.common.shutdown_signals import shutdown_signals
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with shutdown_signals():
    try:
        print('ready', flush=True)
        while True:
            time.sleep(.01)
    except KeyboardInterrupt:
        print('interrupted', flush=True)
    finally:
        # A launcher can signal the node again while encoders are draining.
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(.01)
        print('finalized', flush=True)
print('restored', signal.getsignal(signal.SIGINT) == signal.SIG_IGN, flush=True)
"""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signum)
        output, errors = process.communicate(timeout=5)
        assert process.returncode == 0, errors
        assert output.splitlines() == ["interrupted", "finalized", "restored True"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_original_handlers_restored_after_exception():
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(RuntimeError, match="startup failed"):
        with shutdown_signals():
            raise RuntimeError("startup failed")
    assert {signum: signal.getsignal(signum) for signum in previous} == previous


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_entrypoint_guard_covers_run_and_finally_without_loading_sdk(cleanup_fails, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "genie_sim_engine_newton_node.py"
    tree = ast.parse(script.read_text())
    main = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "main")
    namespace = {
        "shutdown_signals": shutdown_signals,
        "sys": sys,
        "_parse_args": lambda *args, **kwargs: {},
    }
    events = []

    def run(params):
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            events.append("run_cleanup")
            os.kill(os.getpid(), signal.SIGINT)
            if cleanup_fails:
                raise RuntimeError("video encoder failed")

    class App:
        def close(self):
            os.kill(os.getpid(), signal.SIGTERM)
            events.append("app_cleanup")

    namespace.update(run=run, simulation_app=App())
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(script), "exec"), namespace)
    assert namespace["main"]() == (1 if cleanup_fails else 0)
    assert events == ["run_cleanup", "app_cleanup"]
    if cleanup_fails:
        assert "fatal: video encoder failed" in capsys.readouterr().err
