# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run the shared example with both applications and ROS/GPU imports blocked."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("joint_target", [False, True])
def test_example_uses_only_shared_client_and_closes_transport(tmp_path, joint_target):
    scene = tmp_path / "scene.yaml"
    scene.write_text("robot: {}\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"scene_yaml": str(scene)}))
    arguments = [
        "--launcher",
        str(PACKAGE / "config/launcher_client_newton.yaml"),
        "--manifest",
        str(manifest),
        "--steps",
        "2",
    ]
    if joint_target:
        arguments += ["--joint", "j1", "--target", "0.25"]
    script = f"""
import sys, importlib.abc
sys.path[:0] = {[str(PACKAGE), str(PACKAGE / 'tests')]!r}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'geniesim_benchmark', 'rlinf_geniesim', 'isaacsim', 'omni', 'carb', 'pxr', 'rclpy', 'newton', 'warp'}}:
            raise AssertionError('forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from genie_sim_client import SimulationClient, client
from test_client import Transport
bridge = Transport()
client.RosEngineBridge = lambda **kwargs: bridge
from genie_sim_client.examples import newton_joint_step
# Keep the real client and fake only ROS transport/process ownership.
newton_joint_step.SimulationClient = lambda config: SimulationClient(config, spawn=False)
state = newton_joint_step.main({arguments!r})
assert state['step_requests'] == len(bridge.sent) == 2
assert state['completion'] == 'arrival_v1'
assert state['reset_epoch'] == 1
assert state['joint_positions'] == {[[0.25 if joint_target else 0.0, 0.0]]!r}
assert bridge.reset_args == ([0], None, None)
assert bridge.closed
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
