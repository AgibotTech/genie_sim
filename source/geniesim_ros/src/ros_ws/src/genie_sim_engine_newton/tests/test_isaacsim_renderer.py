# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Real local process/Pipe checks; no Kit, GPU, ROS or model connection."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest


PACKAGE = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "tested_isaacsim_renderer", PACKAGE / "genie_sim_engine_newton/engine/newton/visualizers/isaacsim.py"
)
bridge = module_from_spec(SPEC)
sys.modules[SPEC.name] = bridge
SPEC.loader.exec_module(bridge)


PRELUDE = """
import argparse
from multiprocessing.connection import Connection
import os
import signal
import struct
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--ipc-fd', type=int, required=True)
args = parser.parse_args()
connection = Connection(args.ipc_fd)
print('CPU fixture worker started', flush=True)
"""


def worker(tmp_path, source):
    path = tmp_path / "worker.py"
    path.write_text(textwrap.dedent(PRELUDE) + "\n" + textwrap.dedent(source))
    return path


def echo_worker(tmp_path, response="connection.send(reply)"):
    return worker(
        tmp_path,
        """
connection.send({'ready': True, 'renderer': 'isaacsim'})
while True:
    request = connection.recv()
    if request['op'] == 'close':
        break
    assert set(request) == {'op', 'sequence', 'scene', 'cameras', 'width', 'height'}
    assert request['scene']['vertices'].dtype == np.float32
    assert [row['name'] for row in request['cameras']] == ['head', 'hand_left', 'hand_right']
    reply = {
        'renderer': 'isaacsim',
        'sequence': request['sequence'],
        'images': {
            row['name']: np.full((request['height'], request['width'], 3), request['sequence'], np.uint8)
            for row in request['cameras']
        },
    }
"""
        + textwrap.indent(textwrap.dedent(response), "    "),
    )


@pytest.fixture
def cameras():
    return [
        dict(name=name, eye=[0, 0, 1], target=[1, 0, 1], up=[0, 0, 1], fov_degrees=65)
        for name in ("head", "hand_left", "hand_right")
    ]


@pytest.fixture
def scene():
    return {"vertices": np.arange(9, dtype=np.float32).reshape(3, 3)}


@pytest.fixture
def processes(monkeypatch):
    real_spawn = subprocess.Popen
    records = []

    def spawn(*args, **kwargs):
        process = real_spawn(*args, **kwargs)
        if "pass_fds" in kwargs:
            records.append((process, args, kwargs))
        return process

    monkeypatch.setattr(bridge.subprocess, "Popen", spawn)
    yield records
    for process, _, _ in records:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


@pytest.fixture
def pipes(monkeypatch):
    real_pipe = bridge.Pipe
    endpoints = []

    def pipe(**kwargs):
        pair = real_pipe(**kwargs)
        endpoints.extend(pair)
        return pair

    monkeypatch.setattr(bridge, "Pipe", pipe)
    return endpoints


def assert_released(instance, processes):
    assert instance._closed
    assert instance._connection is None
    assert instance._log is None
    assert instance._process is None
    assert instance._io_thread is None
    for process, _, kwargs in processes:
        assert process.poll() is not None
        assert kwargs["stdout"] is kwargs["stderr"]
        assert kwargs["stdout"].closed


def test_real_pipe_carries_numpy_scene_and_strict_sequence_rgb_batches(tmp_path, cameras, scene, processes):
    log = tmp_path / "worker.log"
    instance = bridge.IsaacSimRenderer(echo_worker(tmp_path), log)
    try:
        first = instance.render(scene, cameras, 8, 6)
        second = instance.render(scene, cameras, width=8, height=6)
        assert set(first) == {"head", "hand_left", "hand_right"}
        for name in first:
            assert first[name].shape == (6, 8, 3)
            assert first[name].dtype == np.uint8
            assert first[name].flags.c_contiguous
            np.testing.assert_array_equal(first[name], 1)
            np.testing.assert_array_equal(second[name], 2)
        first["head"][:] = 99
        np.testing.assert_array_equal(second["head"], 2)
        command = processes[0][1][0]
        kwargs = processes[0][2]
        assert command[0] == sys.executable
        assert command[2] == "--ipc-fd"
        assert kwargs["pass_fds"] == (int(command[3]),)
        assert kwargs["start_new_session"] is True
    finally:
        instance.close()
    instance.close()
    assert_released(instance, processes)
    assert "CPU fixture worker started" in log.read_text()
    with pytest.raises(RuntimeError, match="closed"):
        instance.render(scene, cameras, 8, 6)


@pytest.mark.parametrize(
    "response, message",
    [
        ("reply['sequence'] -= 1\nconnection.send(reply)", "sequence mismatch"),
        ("reply['sequence'] = True\nconnection.send(reply)", "sequence mismatch"),
        ("reply['renderer'] = 'gl'\nconnection.send(reply)", "identity"),
        ("del reply['images']['head']\nconnection.send(reply)", "incomplete camera batch"),
        ("reply['images']['extra'] = reply['images']['head']\nconnection.send(reply)", "incomplete camera batch"),
        ("reply['images']['head'] = np.zeros((6, 8, 3), np.float32)\nconnection.send(reply)", "invalid RGB"),
        ("reply['images']['head'] = np.zeros((8, 6, 3), np.uint8)\nconnection.send(reply)", "invalid RGB"),
        ("reply['images']['head'] = np.zeros((6, 8, 4), np.uint8)\nconnection.send(reply)", "invalid RGB"),
        ("reply['images']['head'] = [[[0, 0, 0]]]\nconnection.send(reply)", "invalid RGB"),
        ("connection.send({'error': 'GPU readback failed'})", "GPU readback failed"),
        ("connection.send(None)", "identity"),
        ("connection.close()\nraise SystemExit(0)", "disconnected"),
    ],
)
def test_bad_batch_is_fatal_reaped_and_never_retried(tmp_path, cameras, scene, processes, response, message):
    instance = bridge.IsaacSimRenderer(echo_worker(tmp_path, response), tmp_path / "worker.log")
    with pytest.raises(RuntimeError, match=message):
        instance.render(scene, cameras, 8, 6)
    assert instance._poisoned
    assert_released(instance, processes)
    with pytest.raises(RuntimeError, match="poisoned"):
        instance.render(scene, cameras, 8, 6)
    assert len(processes) == 1


@pytest.mark.parametrize(
    "source, error, message",
    [
        ("connection.send({'ready': True, 'renderer': 'gl'})\ntime.sleep(60)", RuntimeError, "readiness"),
        ("connection.send({'ready': 1, 'renderer': 'isaacsim'})\ntime.sleep(60)", RuntimeError, "readiness"),
        ("connection.send({'error': 'stage load failed'})\ntime.sleep(60)", RuntimeError, "stage load failed"),
        ("connection.close()", RuntimeError, "disconnected"),
        ("time.sleep(60)", TimeoutError, "timed out"),
    ],
)
def test_constructor_failure_releases_process_pipe_and_log(tmp_path, processes, pipes, source, error, message):
    path = worker(tmp_path, source)
    with pytest.raises(error, match=message):
        bridge.IsaacSimRenderer(path, tmp_path / "worker.log", startup_timeout_s=0.5 if error is TimeoutError else 5)
    assert len(processes) == 1
    process, _, kwargs = processes[0]
    assert process.poll() is not None
    assert kwargs["stdout"].closed
    assert len(pipes) == 2 and all(endpoint.closed for endpoint in pipes)


def test_spawn_failure_releases_both_pipe_ends_and_log(tmp_path, monkeypatch):
    endpoints = []
    real_pipe = bridge.Pipe
    logs = []

    def pipe(**kwargs):
        pair = real_pipe(**kwargs)
        endpoints.extend(pair)
        return pair

    def spawn(*args, **kwargs):
        logs.append(kwargs["stdout"])
        raise OSError("spawn failed")

    monkeypatch.setattr(bridge, "Pipe", pipe)
    monkeypatch.setattr(bridge.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="spawn failed"):
        bridge.IsaacSimRenderer(worker(tmp_path, "pass"), tmp_path / "worker.log")
    assert len(endpoints) == 2 and all(endpoint.closed for endpoint in endpoints)
    assert logs[0].closed


@pytest.mark.parametrize("partial", [False, True])
def test_render_timeout_also_bounds_partial_recv_and_reaps_child(tmp_path, cameras, scene, processes, partial):
    response = "time.sleep(60)"
    if partial:
        response = "os.write(connection.fileno(), struct.pack('!i', 4096) + b'x')\ntime.sleep(60)"
    instance = bridge.IsaacSimRenderer(echo_worker(tmp_path, response), tmp_path / "worker.log", timeout_s=0.1)
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        instance.render(scene, cameras, 8, 6)
    assert time.monotonic() - start < 3
    assert_released(instance, processes)


def test_timeout_also_bounds_large_snapshot_send_to_nonreading_child(tmp_path, cameras, processes):
    path = worker(tmp_path, "connection.send({'ready': True, 'renderer': 'isaacsim'})\ntime.sleep(60)")
    instance = bridge.IsaacSimRenderer(path, tmp_path / "worker.log", timeout_s=0.1)
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        instance.render({"large": np.zeros(4_000_000, np.float32)}, cameras, 8, 6)
    assert time.monotonic() - start < 3
    assert_released(instance, processes)


def test_close_escalates_to_kill_for_worker_ignoring_close_and_sigterm(tmp_path, processes):
    path = worker(
        tmp_path,
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "connection.send({'ready': True, 'renderer': 'isaacsim'})\n"
        "while True:\n    time.sleep(60)",
    )
    instance = bridge.IsaacSimRenderer(path, tmp_path / "worker.log")
    start = time.monotonic()
    instance.close()
    assert time.monotonic() - start < 4
    assert processes[0][0].returncode == -9
    assert_released(instance, processes)


def test_log_is_exclusive_and_is_never_overwritten(tmp_path, processes):
    log = tmp_path / "worker.log"
    log.write_text("prior experiment")
    with pytest.raises(FileExistsError):
        bridge.IsaacSimRenderer(worker(tmp_path, "pass"), log)
    assert log.read_text() == "prior experiment"
    assert not processes


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), float("nan"), "bad"])
def test_invalid_timeout_does_not_spawn(tmp_path, processes, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        bridge.IsaacSimRenderer(worker(tmp_path, "pass"), tmp_path / "worker.log", timeout_s=timeout)
    assert not processes


def test_invalid_camera_or_size_is_rejected_before_send_without_poison(tmp_path, cameras, scene, processes):
    instance = bridge.IsaacSimRenderer(echo_worker(tmp_path), tmp_path / "worker.log")
    try:
        with pytest.raises(ValueError, match="positive integers"):
            instance.render(scene, cameras, True, 6)
        with pytest.raises(ValueError, match="unique"):
            instance.render(scene, [cameras[0], cameras[0]], 8, 6)
        with pytest.raises(ValueError, match="finite eye"):
            instance.render(scene, [{**cameras[0], "eye": [np.nan, 0, 1]}], 8, 6)
        assert instance._sequence == 0
        np.testing.assert_array_equal(instance.render(scene, cameras, 8, 6)["head"], 1)
    finally:
        instance.close()
    assert_released(instance, processes)


def test_parent_module_import_requires_no_kit_ros_newton_or_warp():
    script = f"""
import importlib.abc
import importlib.util
import sys
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'isaacsim', 'omni', 'carb', 'pxr', 'rclpy', 'newton', 'warp'}}:
            raise AssertionError('unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, RejectRuntime())
spec = importlib.util.spec_from_file_location('kit_free_renderer', {str(SPEC.origin)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.IsaacSimRenderer
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10, capture_output=True, text=True)
