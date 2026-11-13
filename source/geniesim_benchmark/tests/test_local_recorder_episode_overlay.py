# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise episode labels through recording, encoding and video concatenation."""

import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from geniesim_benchmark.plugins.output_system.local_recorder import LocalRecorder


pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video recording")


def read_frames(path):
    # Isaac's OpenCV build has no FFmpeg video decoder; use the recorder's tools.
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required for video checks")
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    stream = json.loads(probe)["streams"][0]
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-threads", "1", "pipe:1"]
    )
    assert raw, f"No readable frames in {path}"
    return list(np.frombuffer(raw, dtype=np.uint8).reshape(-1, stream["height"], stream["width"], 3))


@pytest.mark.parametrize("instruction", ["", "Pick up the cup."])
def test_head_video_labels_follow_episode_index_after_concat(tmp_path, instruction):
    width, height = 640, 360
    camera_ids = ("head_front", "left_hand", "right_hand", "camera_rgb")
    rgba = np.full((height, width, 4), 160, dtype=np.uint8)
    rgba[..., 3] = 255
    robot = SimpleNamespace(_img_data_cache={camera_id: rgba for camera_id in camera_ids})
    camera_specs = [{"camera_id": camera_id, "width": width, "height": height} for camera_id in camera_ids]
    recorder = LocalRecorder(record_world_camera=True)
    recorder.update_instruction(instruction)

    for episode_idx in (0, 17):
        recorder.start(str(tmp_path), episode_idx, camera_specs, sub_task_name="overlay")
        try:
            for step in range(3):
                recorder.write_frames(robot, step)
        finally:
            recorder.stop(episode_idx)

    recorder.concat_all()
    head = read_frames(tmp_path / "overlay/head.webm")
    assert len(head) == 6
    first, last = head[0], head[-1]
    bottom_right = np.s_[-55:-5, -220:-5]
    assert np.any(first[bottom_right] > 200), "Episode 0 needs a visible text label without an instruction"
    assert np.mean(np.abs(first[bottom_right].astype(float) - 160)) > 3
    assert np.mean(np.abs(first[bottom_right].astype(float) - last[bottom_right])) > 1, "Label must change on resume"
    assert np.mean(np.abs(first[120:240, 20:300].astype(float) - 160)) < 3
    assert np.mean(np.abs(first[-55:-5, :200].astype(float) - 160)) < 3
    if instruction:
        assert np.any(first[:70] > 200), "The instruction overlay must remain visible"
    else:
        assert np.mean(np.abs(first[:70].astype(float) - 160)) < 3

    for camera_name in ("hand_left", "hand_right", "world_img"):
        frames = read_frames(tmp_path / f"overlay/{camera_name}.webm")
        assert len(frames) == 6
        assert all(np.max(np.abs(frame.astype(float) - 160)) < 5 for frame in frames)
    assert np.all(rgba[..., :3] == 160), "Video overlays must not modify the simulator's observation cache"
