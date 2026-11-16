# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""IPC recording reuses policy/history captures without introducing render work."""

import base64
from pathlib import Path

import pytest
import yaml

from geniesim_benchmark.libuipc import adapter, runner


@pytest.mark.parametrize("history_interval,expected_captures", [(0, 3), (1, 6)])
def test_recording_does_not_add_observations_or_change_policy_inputs(
    tmp_path, monkeypatch, history_interval, expected_captures
):
    from geniesim_benchmark.libuipc import recording

    recorded = []
    image = b"already-rendered-policy-image"

    class Recorder:
        def __init__(self, *args, **kwargs):
            pass

        def capture(self, packet, step):
            recorded.append((packet["snapshot"]["snapshot_id"], step))

        def close(self):
            return "retained-recording"

    monkeypatch.setattr(recording, "EpisodeRecording", Recorder)

    class Client:
        def __init__(self):
            self.steps = self.captures = 0

        def observe(self, **kwargs):
            self.captures += 1
            return {
                "snapshot": {
                    "request_id": f"request-{self.captures}",
                    "snapshot_id": f"snapshot-{self.captures}",
                    "source_id": "same-world",
                    "epoch": 0,
                    "tick": 2 * self.steps,
                    "cloth_tick": 2 * self.steps,
                    "captured_wall_time_ns": self.captures,
                    "sim_time": self.steps / 30,
                    "joints": dict.fromkeys(adapter.DRIVE_JOINTS, 0.25),
                    "bodies": {
                        "/robot/" + name: [0, 0, 0, 1, 0, 0, 0]
                        for name in ("base_link", "arm_base_link", "arm_l_end_link", "arm_r_end_link")
                    },
                },
                "images": {
                    name: {
                        "encoding": "JPEG",
                        "image_data": base64.b64encode(image).decode(),
                        "width": 320,
                        "height": 120,
                    }
                    for name in ("head", "hand_left", "hand_right")
                },
            }

        def step(self, target, duration, *, on_accepted, **kwargs):
            on_accepted({"accepted": True})
            self.steps += 1
            return {"executed_sim_time_s": duration}

    class Policy:
        def __init__(self):
            self.inputs = []

        def infer(self, payload, **kwargs):
            self.inputs.append(payload)
            return {
                **{
                    name: {"kind": "JOINT_ABS", "values": [[0.0] * width for _ in range(2)]}
                    for name, width in (
                        ("left_arm", 7),
                        ("right_arm", 7),
                        ("left_effector", 20),
                        ("right_effector", 20),
                    )
                },
                "hist_frame_interval": history_interval,
            }

    results = []
    for enabled in (False, True):
        client, policy = Client(), Policy()
        result = runner.run_episode(
            client,
            policy,
            task="cloth_hanger",
            instruction="Hang clothes",
            fixed_joints=dict.fromkeys(adapter.WAIST_JOINTS + adapter.HEAD_JOINTS, 0),
            episode_dir=tmp_path / str(enabled),
            max_steps=3,
            record=enabled,
        )
        assert result["execution_status"] == "completed", result
        assert client.captures == expected_captures
        assert result["steps"] == 3 and result["inference_requests"] == 2
        results.append(policy.inputs)
    assert results[0] == results[1]
    assert len(recorded) == expected_captures
    assert [identity for identity, _ in recorded] == [f"snapshot-{i + 1}" for i in range(expected_captures)]


def test_only_three_ipc_evaluation_scenes_enable_idle_pause():
    configs = Path(__file__).resolve().parents[2] / "geniesim_ros/src/ros_ws/src/genie_sim_bringup/config"
    enabled = set()
    for path in configs.glob("scene_*.yaml"):
        scene = yaml.safe_load(path.read_text())
        if scene.get("libuipc", {}).get("pause_when_idle", False):
            enabled.add(path.name)
    assert enabled == {
        f"scene_flat_g2_wuji_ipc_{task}_eval.yaml" for task in ("cloth_hanger", "fold_cloth", "drawstring_bag")
    }
