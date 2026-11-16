# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Runner pause barriers, CLI forwarding and synchronous native integration."""

import ast
import base64
from pathlib import Path
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest

from geniesim_benchmark.libuipc import adapter, runner as ipc

PACKAGE = Path(ipc.__file__).parents[1]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_libuipc_freezes_observation_through_reply_and_resumes_actions(tmp_path, enabled, fail):
    class Client:
        paused = False
        tick = 0
        steps = 0

        def pause(self):
            assert not self.paused
            self.paused = True

        def resume(self):
            assert self.paused
            self.paused = False

        def wall_tick(self):
            if not self.paused:
                self.tick += 1

        def observe(self, **kwargs):
            self.wall_tick()
            return {
                "snapshot": {
                    "request_id": "request",
                    "source_id": "source",
                    "snapshot_id": "snapshot",
                    "epoch": 0,
                    "tick": self.tick,
                    "cloth_tick": self.tick,
                    "sim_time": self.tick / 60,
                    "captured_wall_time_ns": 1,
                    "joints": dict.fromkeys(adapter.DRIVE_JOINTS, 0.0),
                    "bodies": {
                        "/robot/" + n: [0, 0, 0, 1, 0, 0, 0]
                        for n in ("base_link", "arm_base_link", "arm_l_end_link", "arm_r_end_link")
                    },
                },
                "images": {
                    name: {
                        "encoding": "JPEG",
                        "image_data": base64.b64encode(b"jpeg").decode(),
                        "width": 2,
                        "height": 2,
                    }
                    for name in ("head", "hand_left", "hand_right")
                },
            }

        def step(self, target, duration, *, on_accepted, **kwargs):
            assert not self.paused
            self.wall_tick()
            self.steps += 1
            on_accepted({"status": "accepted"})
            return {"executed_sim_time_s": duration}

    client = Client()

    class Policy:
        def infer(self, payload, **kwargs):
            assert client.paused is enabled
            before = client.tick
            for _ in range(5):
                client.wall_tick()  # Model latency would otherwise advance physics.
            assert (client.tick == before) is enabled
            if fail:
                raise TimeoutError("delayed inference failed")
            return {
                name: {"kind": "JOINT_ABS", "values": [[0.0] * width]}
                for name, width in (("left_arm", 7), ("right_arm", 7), ("left_effector", 20), ("right_effector", 20))
            }

    result = ipc.run_episode(
        client,
        Policy(),
        task="cloth_hanger",
        instruction="Hang clothes",
        fixed_joints=dict.fromkeys(adapter.WAIST_JOINTS + adapter.HEAD_JOINTS, 0.0),
        episode_dir=tmp_path,
        max_steps=2,
        pause_physics_during_inference=enabled,
    )
    assert result["execution_status"] == ("failed" if fail else "completed"), result
    assert client.steps == (0 if fail else 2)
    assert client.paused is enabled  # Finish can seal a paused episode without extra ticks.


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_newton_inference_pause_and_exception_cleanup(monkeypatch, enabled, fail):
    from geniesim_benchmark.newton import runner

    state = NS(schema=NS(num_envs=1, joint_names=("arm",)), joint_positions=np.zeros((1, 1)), reset_epoch=0)

    class Client:
        paused = False
        steps = 0

        def pause(self):
            self.paused = True

        def resume(self):
            self.paused = False

        def step(self, values):
            assert not self.paused
            self.steps += 1
            return NS(snapshot=state)

    client = Client()

    def capture(*args, **kwargs):
        client.pause()
        return state, 1, {}

    monkeypatch.setattr(runner, "pause_observation", capture)
    monkeypatch.setattr(runner, "build_payload", lambda *args, **kwargs: {})

    class Policy:
        def infer(self, *args, **kwargs):
            assert client.paused is enabled
            if fail:
                raise TimeoutError("model timeout")
            return NS(targets=[{"arm": 0.1}], history_interval=0, server_episode_done=False)

    kwargs = dict(
        instruction="Toss",
        initial_positions={},
        max_steps=2,
        groups=NS(left_arm=("arm",), right_arm=(), left_effector=(), right_effector=()),
        pause_physics_during_inference=enabled,
    )
    if fail:
        with pytest.raises(TimeoutError, match="model timeout"):
            runner.run_episode(client, None, Policy(), **kwargs)
    else:
        assert runner.run_episode(client, None, Policy(), **kwargs)["issued_steps"] == 2
    assert client.paused


@pytest.mark.parametrize("flag", ["true", "false"])
def test_cli_and_yaml_accept_the_shared_flag(tmp_path, monkeypatch, flag):
    from geniesim_benchmark.backends.mujoco.rubik.config import load_config
    from geniesim_benchmark.newton import runner, instances

    arg = "--benchmark.pause_physics_during_inference=" + flag
    assert ipc.make_parser().parse_args(["--config=x.yaml", arg]).pause_physics_during_inference is (flag == "true")
    cfg = load_config(
        PACKAGE / "config/g2crsbwuji_dexmanip_rubik_cube_solve.yaml",
        [arg, "--rubik.package_root=" + str(tmp_path), "--benchmark.infer_host=localhost:8999"],
    )
    assert cfg.benchmark.pause_physics_during_inference is (flag == "true")
    captured = []
    monkeypatch.setattr(instances, "run_instances", lambda opts, path, config: captured.append(config) or 0)
    assert (
        runner.main(
            ["--config", str(PACKAGE / "config/g2crsbwuji_dexmanip_wok_flip.yaml"), "--output-dir", str(tmp_path), arg]
        )
        == 0
    )
    assert captured[0]["benchmark"]["pause_physics_during_inference"] is (flag == "true")


@pytest.mark.parametrize("enabled", [False, True])
def test_mujoco_state_does_not_advance_during_slow_inference(tmp_path, monkeypatch, enabled):
    mujoco = pytest.importorskip("mujoco")
    from geniesim_benchmark.backends.mujoco.rubik import runner, config

    class Env:
        graphics = viewer = None
        initial_height = 0.0
        maximum_integration_error = 0.0
        controller = None
        steps = 0

        def __init__(self):
            self.model = mujoco.MjModel.from_xml_string(
                '<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'
            )
            self.data = mujoco.MjData(self.model)

        def sample(self):
            return dict(
                time=float(self.data.time),
                height=float(self.data.qpos[2]),
                support_fingers=0,
                cube_checker=dict(
                    simulation_time=float(self.data.time),
                    hold_seconds=0.0,
                    solved=False,
                    success=False,
                    held_seconds=0.0,
                    assembled=False,
                ),
            )

        def reset(self, *args):
            mujoco.mj_resetData(self.model, self.data)
            return self.sample()

        def images(self):
            return {}

        def set_action(self, action, steps):
            return steps

        def tick(self):
            mujoco.mj_step(self.model, self.data)
            self.steps += 1
            return self.sample()

        def action_contract(self):
            return {}

        checker_contract = arm_speed_diagnostics = robot_collision_diagnostics = action_contract

    env = Env()
    cfg = config.Config()
    cfg.benchmark.pause_physics_during_inference = enabled
    cfg.benchmark.record = True
    cfg.benchmark.record_state = True  # Legacy option must not restore large state dumps.
    cfg.rubik.render = False
    cfg.rubik.timeout_seconds = 0.05
    monkeypatch.setattr(runner, "payload_for", lambda *a: {"params": {"states": {}}})
    monkeypatch.setattr(runner, "parse_actions", lambda *a, **kw: ([np.array([0.0])], False))
    calls = []

    class Policy:
        def reset(self):
            pass

        def infer(self, payload):
            before = env.data.qpos.copy(), env.data.qvel.copy(), env.data.time
            time.sleep(0.02)
            np.testing.assert_array_equal(env.data.qpos, before[0])
            np.testing.assert_array_equal(env.data.qvel, before[1])
            assert env.data.time == before[2]
            calls.append(env.data.time)
            return {}

    result = runner.run_episode(env, cfg, Policy(), 0, "", tmp_path / "episode")
    assert result["valid"], result
    assert len(calls) >= 2 and env.steps > 0 and env.data.qpos[2] < 1
    assert result["recorded_raw"] is False
    assert not (tmp_path / "episode/raw-1000hz.npz").exists()


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_isaac_pause_always_restores_physics(vector, fail):
    # Import only the production call boundary; the full module imports Isaac/Kit.
    tree = ast.parse((PACKAGE / "benchmark/task_benchmark.py").read_text())
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "resume_physics"
            for statement in node.finalbody
            for n in ast.walk(statement)
        )
    ]
    guard = min(
        (
            node
            for node in guards
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == ("act_batch" if vector else "act")
                for n in ast.walk(node)
            )
        ),
        key=lambda node: len(list(ast.walk(node))),
    )
    events = []

    def infer(*args, **kwargs):
        events.append("infer")
        if fail:
            raise TimeoutError("model timeout")
        return {}

    owner = NS(
        api_core=NS(resume_physics=lambda: events.append("resume")),
        task_mode="infer",
        args=NS(interactive=False, record=False),
        policy=NS(act=infer),
        env=NS(
            current_step=0,
            task=NS(get_instruction=lambda: ["Insert"], get_instruction_segments=lambda: []),
            get_camera_gen_config=lambda: {},
        ),
    )
    scope = dict(
        self=owner,
        should_pause=True,
        observaion={},
        vec_policy=NS(act_batch=infer),
        obs_list=[],
        instructions=[],
        gen_configs=[],
        step_nums=[],
        active=[],
        instruction_segments=[],
    )
    code = compile(ast.Module(body=[guard], type_ignores=[]), "inference-boundary", "exec")
    if fail:
        with pytest.raises(TimeoutError):
            exec(code, scope)
    else:
        exec(code, scope)
    assert events == ["infer", "resume"]
