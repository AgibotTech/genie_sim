# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Video recording never samples or exports replay state, including legacy flags."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from geniesim_benchmark.config.params import BenchmarkConfig
from geniesim_benchmark.plugins.output_system.local_recorder import LocalRecorder


def test_benchmark_record_defaults_to_video_only():
    assert BenchmarkConfig(record=True).record_state is False


@pytest.mark.parametrize("record_state", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_recording_and_export_are_video_only(tmp_path, record_state, checkpoint):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required")
    robot = SimpleNamespace(_img_data_cache={"head_front": np.full((120, 160, 4), 160, dtype=np.uint8)})
    specs = [{"camera_id": "head_front", "width": 160, "height": 120}]
    recorder = LocalRecorder(record_state=record_state)
    if checkpoint:
        recorder.restore_checkpoint("video-only", [])
    bag = Path(recorder.start(str(tmp_path / "recordings"), 5, specs, sub_task_name="task"))
    try:
        for step in range(3):
            recorder.write_frames(
                robot, step, lambda: pytest.fail("Replay state must not be sampled"), sim_time=step / 30
            )
    finally:
        recorder.stop(5)
    assert not (bag / "state.jsonl").exists()
    # An older recording may contain state; publication must not export it.
    (bag / "state.jsonl").write_text("legacy state")
    destination = tmp_path / "published"
    if checkpoint:
        artifacts = recorder.concat_all(preserve_bags=True)
        recorder.publish_checkpoint_artifacts(artifacts, bag.parent, str(destination))
    else:
        recorder.concat_all(final_output_dir=str(destination))
    assert not list(destination.rglob("*.jsonl"))
    from test_local_recorder_episode_overlay import read_frames

    assert len(read_frames(destination / "task/head.webm")) == 3
