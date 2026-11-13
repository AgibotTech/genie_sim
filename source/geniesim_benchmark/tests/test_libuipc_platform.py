# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Platform flags, native EEF conversion and durable episode reuse."""

import copy
import base64
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from geniesim_benchmark.libuipc import adapter, runner
from geniesim_benchmark.utils.hard_ik import HardIKFKSolver


def test_platform_record_resume_flags_accept_equals_and_separate_values(tmp_path):
    options = runner.make_parser().parse_args([
        "--config", "task.yaml", "--benchmark.record=true", "--benchmark.resume", "true",
        "--benchmark.output_dir", str(tmp_path), "--benchmark.resume_id=job-42",
        "--benchmark.record_state=false",
    ])
    assert options.record and options.resume
    assert options.output_dir == tmp_path
    assert options.resume_id == "job-42"
    assert not options.record_state


def joint_reply(count=2):
    return {
        name: {"kind": "JOINT_ABS", "values": [[0.0] * width for _ in range(count)]}
        for name, width in (("left_arm", 7), ("right_arm", 7), ("left_effector", 20), ("right_effector", 20))
    }


@pytest.mark.parametrize("frame", ["arm_base_link", "base_link"])
def test_eef_roundtrip_uses_hard_ik_and_measured_torso(frame):
    pytest.importorskip("ik_solver")
    solver = HardIKFKSolver(robot_cfg="G2_crsB_wuji")
    joints = np.clip(np.zeros(14), solver.lower + 0.1, solver.upper - 0.1)
    measured = dict(zip(solver.joint_names, joints))
    measured.update(dict(zip(adapter.WAIST_JOINTS, [0.2, 0.1, -0.1, 0.15, 0.1])))
    transform = solver.base_transform(measured)
    poses = solver.compute_eef(joints)
    reply = joint_reply()
    for side in ("left", "right"):
        pose = poses[side]
        position, rotation = np.array(pose[:3]), Rotation.from_quat(np.array(pose[3:])[[1, 2, 3, 0]])
        if frame == "base_link":
            position = transform[:3, :3] @ position + transform[:3, 3]
            rotation = Rotation.from_matrix(transform[:3, :3]) * rotation
        values = [*position, *rotation.as_quat()]
        reply[side + "_arm"] = {"kind": "EEF_ABS", "base_link": frame, "values": [values, values]}
    original = copy.deepcopy(reply)
    chunk = adapter.parse_result(reply, measured_positions=measured, ik_solver=solver)
    assert len(chunk.targets) == 2
    np.testing.assert_allclose([chunk.targets[0][name] for name in adapter.ARM_JOINTS], joints, atol=1e-7)
    assert all(chunk.targets[0][name] == 0 for name in adapter.GRIPPER_JOINTS)
    assert reply == original


def test_bad_tail_is_rejected_before_any_ik():
    reply = joint_reply()
    for side in ("left", "right"):
        reply[side + "_arm"]["kind"] = "EEF_ABS"
    reply["right_effector"]["values"][-1][0] = float("nan")

    class Solver:
        def base_transform(self, positions):
            pytest.fail("invalid action chunks must be rejected before IK")

    with pytest.raises(ValueError, match="finite"):
        adapter.parse_result(reply, measured_positions={}, ik_solver=Solver())


def test_record_captures_measured_joints_and_labelled_video(tmp_path):
    import cv2
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required")
    from geniesim_benchmark.libuipc.recording import EpisodeRecording

    image = np.full((120, 320, 3), 150, dtype=np.uint8)
    jpeg = cv2.imencode(".jpg", image)[1].tobytes()
    packet = {"snapshot": {"joints": dict.fromkeys(adapter.DRIVE_JOINTS, 0.25), "sim_time": 0.1,
                           "bodies": {"/robot/base_link": [1, 2, 3, 1, 0, 0, 0]}},
              "images": {name: {"encoding": "JPEG", "image_data": jpeg, "width": 320, "height": 120}
                         for name in ("head", "hand_left", "hand_right")}}
    recording = EpisodeRecording(tmp_path, 7, "Hang the clothes", fps=30, record_state=True,
                                 fixed_joints=dict.fromkeys(adapter.WAIST_JOINTS + adapter.HEAD_JOINTS, 0.1))
    recording.capture(packet, 0)
    recording.capture(packet, 0)  # Same control tick must not add another video/state frame.
    packet["snapshot"]["joints"][adapter.ARM_JOINTS[0]] = 0.5
    recording.capture(packet, 1)
    bag = Path(recording.close())
    rows = [json.loads(line) for line in (bag / "state.jsonl").read_text().splitlines()]
    assert rows[0]["episode_idx"] == 7
    assert [row["step"] for row in rows[1:]] == [0, 1]
    name_index = rows[1]["joints"]["name"].index(adapter.ARM_JOINTS[0])
    assert [row["joints"]["position"][name_index] for row in rows[1:]] == [0.25, 0.5]
    assert rows[1]["base_pose"]["orientation_xyzw"] == [0, 0, 0, 1]
    from test_local_recorder_episode_overlay import read_frames

    frame = read_frames(bag / "video/head.webm")[0]
    assert np.any(frame[-45:, -180:] > 200), "zero-based episode label missing"


def test_runner_records_every_executed_control_tick_without_changing_policy_images(tmp_path):
    import cv2
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required")
    jpeg = cv2.imencode(".jpg", np.full((120, 320, 3), 150, dtype=np.uint8))[1].tobytes()

    class Client:
        steps = 0

        def observe(self, **kwargs):
            return {"snapshot": {
                **dict.fromkeys(("request_id", "source_id", "snapshot_id"), "sample"),
                **dict.fromkeys(("epoch", "tick", "cloth_tick", "captured_wall_time_ns"), self.steps),
                "sim_time": self.steps / 30, "joints": dict.fromkeys(adapter.DRIVE_JOINTS, self.steps / 10),
                "bodies": {"/robot/" + name: [0, 0, 0, 1, 0, 0, 0] for name in
                           ("base_link", "arm_base_link", "arm_l_end_link", "arm_r_end_link")}},
                    "images": {name: {"encoding": "JPEG", "image_data": base64.b64encode(jpeg).decode(), "width": 320, "height": 120}
                               for name in ("head", "hand_left", "hand_right")}}

        def step(self, target, duration, *, on_accepted, **kwargs):
            assert len(target) == 54
            on_accepted({"accepted": True})
            self.steps += 1
            return {"executed_sim_time_s": duration}

    class Policy:
        def infer(self, payload, **kwargs):
            assert payload["params"]["images"]["head"]["image_data"] == jpeg
            return joint_reply()

    result = runner.run_episode(Client(), Policy(), task="cloth_hanger", instruction="Hang clothes",
                                fixed_joints=dict.fromkeys(adapter.WAIST_JOINTS + adapter.HEAD_JOINTS, 0),
                                episode_dir=tmp_path / "episode", max_steps=2, record=True,
                                recording_root=tmp_path)
    assert result["execution_status"] == "completed"
    rows = [json.loads(line) for line in (Path(result["recording"]) / "state.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows[1:]] == [0, 1, 2]
    assert len(list((tmp_path / "episode/observations").glob("*.json"))) == 2


@pytest.mark.parametrize("record", [False, True])
def test_resume_keeps_only_committed_episodes_and_skips_finished_without_startup(tmp_path, monkeypatch, record):
    """A failed second episode is retried; its stale detail is not counted twice."""
    config = {
        "repo_root": str(tmp_path),
        "benchmark": {"num_instances": 3, "num_episode": 1, "resume": True, "record": record},
        "libuipc": {"task": "cloth_hanger", "max_steps": 30},
    }
    monkeypatch.setattr(runner, "checkpoint_identity", lambda *args: {"task": "cloth_hanger"})
    monkeypatch.setattr(runner, "write_summary", lambda directory, rows: runner.write_json(
        directory / "evaluate_ret_00.json", {"details": rows, "statistics": {}}))
    calls = []
    fail = True
    if record:
        import cv2
        import shutil
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg required")
        from geniesim_benchmark.libuipc.recording import EpisodeRecording
        jpeg = cv2.imencode(".jpg", np.full((120, 320, 3), 150, dtype=np.uint8))[1].tobytes()

    class Session:
        def __init__(self, config, endpoint, root, directory, **kwargs):
            self.episodes = [{"instance_id": i, "attempt_index": 0} for i in range(3)]
            calls.append("construct")

        def __enter__(self):
            calls.append("start")
            return self

        def __exit__(self, *args):
            calls.append("close")

        def skip_completed_episode(self, index):
            calls.append(("skip", index))

    monkeypatch.setattr(runner, "LibuipcRunSession", Session)

    def episode(session, index, *, evaluations):
        calls.append(index)
        result = {"episode_idx": index, "execution_status": "failed" if fail and index == 1 else "completed",
                  "scoring_status": "scored", "score": 0, "success": False}
        if record:
            recorder = EpisodeRecording(tmp_path, index, "Hang clothes", fps=30, record_state=True,
                                        fixed_joints={}, run_id=session.checkpoint_run_id)
            recorder.capture({"snapshot": {"joints": dict.fromkeys(adapter.DRIVE_JOINTS, index / 10),
                                           "sim_time": 0.1, "bodies": {"/robot/base_link": [0, 0, 0, 1, 0, 0, 0]}},
                              "images": {name: {"encoding": "JPEG", "image_data": jpeg, "width": 320, "height": 120}
                                         for name in ("head", "hand_left", "hand_right")}}, 0)
            result["recording"] = recorder.close()
        evaluations.append({"execution": result})
        return result

    monkeypatch.setattr(runner, "run_one_episode", episode)
    with pytest.raises(RuntimeError):
        runner.evaluate(config, "model:9000", tmp_path, tmp_path, headless=True)
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    assert [e["episode_idx"] for e in manifest["completed"]] == [0]
    calls.clear()
    if record:
        from geniesim_benchmark.plugins.output_system.checkpoint import CheckpointRejected
        committed = tmp_path / manifest["completed"][0]["bag"]
        displaced = committed.with_name("temporarily-missing")
        committed.rename(displaced)
        try:
            with pytest.raises(CheckpointRejected, match="recording is missing"):
                runner.evaluate(config, "model:9000", tmp_path, tmp_path, headless=True)
            assert calls == []
        finally:
            displaced.rename(committed)
    fail = False
    runner.evaluate(config, "model:9000", tmp_path, tmp_path, headless=True)
    assert calls == ["construct", "start", ("skip", 0), 1, 2, "close"]
    rows = json.loads((tmp_path / "evaluate_ret_00.json").read_text())["details"]
    assert [r["episode_coord"]["instance_id"] for r in rows] == [0, 1, 2]
    assert len(json.loads((tmp_path / "summary.json").read_text())["results"]) == 3
    if record:
        assert len([p for p in tmp_path.glob("recording_*") if p.is_dir()]) == 3  # Failed attempt was discarded.
        from test_local_recorder_episode_overlay import read_frames

        assert len(read_frames(tmp_path / "head.webm")) == 3
        assert len(list(tmp_path.glob("*.state.jsonl"))) == 3
    calls.clear()
    runner.evaluate(config, "model:9000", tmp_path, tmp_path, headless=True)
    assert calls == []
    if record:
        # Once finalized, uploads may retain only the published videos/states.
        # Finalized retries must also work without the intermediate bag folders.
        for directory in tmp_path.glob("recording_*"):
            if directory.is_dir():
                shutil.rmtree(directory)
        runner.evaluate(config, "model:9000", tmp_path, tmp_path, headless=True)
        assert calls == []


def test_resume_rejects_changed_inputs_before_starting_simulation(tmp_path, monkeypatch):
    from geniesim_benchmark.plugins.output_system.checkpoint import CheckpointRejected, EvaluationCheckpoint

    config = {"benchmark": {"num_instances": 1, "num_episode": 1, "resume": True}, "libuipc": {}}
    with EvaluationCheckpoint(tmp_path) as checkpoint:
        checkpoint._result_file = "evaluate_ret_00.json"
        checkpoint.prepare({"endpoint": "original"}, [dict(episode_idx=0, instance_id=0, file_idx=0, episode_id=0)], [], {})
    monkeypatch.setattr(runner, "checkpoint_identity", lambda *args: {"endpoint": "different"})
    monkeypatch.setattr(runner, "LibuipcRunSession", lambda *a, **kw: pytest.fail("must reject before starting"))
    with pytest.raises(CheckpointRejected, match="changed"):
        runner.evaluate(config, "model:9000", tmp_path, tmp_path, headless=True)


def test_real_session_can_open_fresh_engine_logs_on_retry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import genie_sim_client
    from geniesim_benchmark.libuipc import config as native_config, scoring

    config = {"libuipc": {"launcher": "launcher", "wait_timeout": 1, "evaluation_scene": "scene"}}
    monkeypatch.setattr(scoring, "prepare_assets", lambda *a, **kw: {"assets_root": str(tmp_path), "process_environment": {}})
    monkeypatch.setattr(scoring, "assembled_geometry", lambda *a, **kw: {})

    def sequence(config, directory, **kwargs):
        scene = directory / "scene.yaml"
        scene.write_text("{}")
        manifest_dir = tmp_path / "scenes" / scene.stem
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "manifest.json").write_text(json.dumps({"robot_usda": "robot.usda"}))
        return scene, [{"directory": directory}]

    class Client:
        identity = {"joint_names": list(adapter.DRIVE_JOINTS)}

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(native_config, "materialize_sequence", sequence)
    monkeypatch.setattr(genie_sim_client, "IPCClient", Client)
    monkeypatch.setattr(genie_sim_client.IPCClientConfig, "from_files", lambda *a: None)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(poll=lambda: None, pid=123))
    monkeypatch.setattr(runner, "stop_process_group", lambda *a: None)
    monkeypatch.setattr(runner, "shutdown_recorded_scene", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "episode_domain", lambda *a: 20)
    for _ in range(2):
        with runner.LibuipcRunSession(config, "model:9000", tmp_path, tmp_path, headless=True):
            pass
    assert len(list((tmp_path / "_attempts").glob("*/scene.log"))) == 2
