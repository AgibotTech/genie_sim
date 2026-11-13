# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Check replay state survives recording, episode boundaries and publication."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from geniesim_benchmark.config.params import BenchmarkConfig
from geniesim_benchmark.plugins.output_system.local_recorder import LocalRecorder, STATE_SCHEMA


def test_benchmark_record_defaults_to_saving_replay_state():
    assert BenchmarkConfig(record=True).record_state is True


@pytest.fixture
def camera():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for video recording")
    image = np.full((120, 160, 4), 160, dtype=np.uint8)
    return (
        SimpleNamespace(_img_data_cache={"head_front": image}),
        [{"camera_id": "head_front", "width": 160, "height": 120, "every_n_frame": 2}],
    )


@pytest.mark.parametrize("checkpoint", [False, True])
def test_default_state_recording_survives_episode_publication(tmp_path, camera, checkpoint):
    robot, specs = camera
    recorder = LocalRecorder()
    bags = []
    if checkpoint:
        recorder.restore_checkpoint("replay-test", [])

    for episode_idx in (0, 5):
        if checkpoint and bags:
            recorder = LocalRecorder()
            recorder.restore_checkpoint("replay-test", [str(p) for p in bags])
        bag = Path(recorder.start(str(tmp_path / "recordings"), episode_idx, specs, sub_task_name="task"))
        samples = []

        def sample():
            samples.append(step)
            return {
                "base_pose": {"position": [episode_idx, 0, 0], "orientation_xyzw": [0, 0, 0, 1]},
                "base_velocity": {"linear": [0.1, 0, 0], "angular": [0, 0, 0.2]},
                "joints": {
                    "name": ["arm_joint", "gripper_joint"],
                    "position": [step / 10, episode_idx / 10],
                    "velocity": [0.3, 0.4],
                },
            }

        try:
            for step in range(6):
                recorder.write_frames(robot, step, sample, sim_time=step / 60)
        finally:
            recorder.stop(episode_idx)
        assert samples == [0, 2, 4]
        state_path = bag / "state.jsonl"
        assert state_path.is_file()
        rows = [json.loads(line) for line in state_path.read_text().splitlines()]
        assert rows[0] == {"schema": STATE_SCHEMA, "episode_idx": episode_idx, "fps": 30, "every_n_frame": 2}
        assert [row["frame_index"] for row in rows[1:]] == [0, 1, 2]
        assert [row["step"] for row in rows[1:]] == [0, 2, 4]
        assert [row["sim_time"] for row in rows[1:]] == pytest.approx([0, 2 / 60, 4 / 60])
        assert [row["joints"]["position"] for row in rows[1:]] == [
            [0, episode_idx / 10],
            [0.2, episode_idx / 10],
            [0.4, episode_idx / 10],
        ]
        assert all(row["joints"]["name"] == ["arm_joint", "gripper_joint"] for row in rows[1:])
        assert all(row["joints"]["velocity"] == [0.3, 0.4] for row in rows[1:])
        assert rows[1]["base_pose"]["position"] == [episode_idx, 0, 0]
        assert rows[1]["base_velocity"]["linear"] == [0.1, 0, 0]
        bags.append(bag)

    destination = tmp_path / "published"
    if checkpoint:
        artifacts = recorder.concat_all(preserve_bags=True)
        recorder.publish_checkpoint_artifacts(artifacts, bags[0].parent, str(destination))
    else:
        recorder.concat_all(final_output_dir=str(destination))
    for bag in bags:
        state = destination / "task" / f"{bag.name}.state.jsonl"
        assert state.is_file()
        assert len(state.read_text().splitlines()) == 4
    assert (destination / "task/head.webm").is_file()


def test_record_state_can_be_explicitly_disabled(tmp_path, camera):
    robot, specs = camera
    recorder = LocalRecorder(record_state=False)
    bag = Path(recorder.start(str(tmp_path), 0, specs))
    samples = []

    def sample():
        samples.append(True)
        return {}

    try:
        recorder.write_frames(robot, 0, sample, sim_time=0)
    finally:
        recorder.stop(0)
    assert samples == []
    assert not (bag / "state.jsonl").exists()
