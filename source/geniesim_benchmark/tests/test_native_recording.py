# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Native camera/state files use Isaac's recorder and survive final export."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from geniesim_benchmark.plugins.output_system.observations import ObservationRecording, finalize_recordings


def test_three_camera_video_state_and_episode_overlay(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    output = tmp_path / "task" / "sub_task"
    frames = {key: np.full((120, 320, 3), 150, dtype=np.uint8) for key in ("head", "hand_left", "hand_right")}
    state = {"joints": {"name": ["joint"], "position": [0.25]}}
    recorder = ObservationRecording(output, 8, "Pan tossing")
    recorder.capture(frames, 0, state)
    recorder.capture(frames, 0, state)
    recorder.capture(frames, 1, state)
    bag = Path(recorder.close())
    assert bag.parent == output
    rows = [json.loads(line) for line in (bag / "state.jsonl").read_text().splitlines()]
    assert rows[0]["episode_idx"] == 8
    assert [row["step"] for row in rows[1:]] == [0, 1]
    report = output / "evaluate_ret_00.json"
    report.write_text('{"details": []}')
    artifacts = finalize_recordings(output, [bag], tmp_path / "export", reports=[report])
    assert (tmp_path / "export/sub_task/evaluate_ret_00.json").read_bytes() == report.read_bytes()
    assert len(artifacts) == 4
    for name in frames:
        path = output / f"{name}.webm"
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        assert int(json.loads(probe.stdout)["streams"][0]["nb_read_frames"]) == 2
        assert (tmp_path / "export/sub_task" / path.name).read_bytes() == path.read_bytes()
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(output / "head.webm"),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    assert np.any(np.frombuffer(raw, dtype=np.uint8).reshape(120, 320, 3)[-45:, -180:] > 200)
    assert (bag / "state.jsonl").is_file()
