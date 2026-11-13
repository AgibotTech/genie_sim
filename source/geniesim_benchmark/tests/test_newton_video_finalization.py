# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Decode real MP4s independently of process shutdown bookkeeping."""

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from geniesim_benchmark.newton import layout_batch, shared_layouts


@pytest.fixture
def video(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required")
    shared = tmp_path / "shared"
    shared.mkdir()
    path = shared / "observer.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x48:r=10",
            "-frames:v",
            "5",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=10,
    )
    (shared / "observer.frames.jsonl").write_text(
        "".join(json.dumps({"frame": frame, "reset_epoch": 1}) + "\n" for frame in range(5))
    )
    return path


def test_closed_video_decodes_all_frames(video):
    result = layout_batch.verify_video(video, expected_frames=5)
    assert result["status"] == "verified"
    assert result["decoded_frames"] == 5


@pytest.mark.parametrize("damage", ["missing", "empty", "truncated", "wrong_count"])
def test_incomplete_video_is_rejected_even_after_graceful_shutdown(video, damage):
    if damage == "missing":
        video.unlink()
    elif damage == "empty":
        video.write_bytes(b"")
    elif damage == "truncated":
        video.write_bytes(video.read_bytes()[:100])
    with pytest.raises(RuntimeError, match="video"):
        layout_batch.verify_video(video, expected_frames=6 if damage == "wrong_count" else 5)


def test_video_changing_during_decode_is_not_finalized(video, monkeypatch):
    run = subprocess.run

    def changing(*args, **kwargs):
        result = run(*args, **kwargs)
        with video.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(layout_batch.subprocess, "run", changing)
    with pytest.raises(RuntimeError, match="changed during verification"):
        layout_batch.verify_video(video)


def test_graceful_detached_cleanup_preserves_score_and_produces_verified_clip(video, tmp_path):
    output = tmp_path / "layout00"
    output.mkdir()
    plan = SimpleNamespace(layout_id=0, run_dir=output)
    row = {"status": "evaluated", "score": 2, "episode_results": [{"evaluation": {"reset_epoch": 1}}]}
    summary = {"cleanup": {"forced": False, "detached_cleanup": [123]}}
    shared_layouts._finalize_videos(video.parent, [plan], [row], summary)
    assert row["status"] == "evaluated" and row["score"] == 2
    assert row["video_verification"]["decoded_frames"] == 5
    assert summary["video_verification"]["decoded_frames"] == 5
    assert json.loads((output / "video.json").read_text())["frame_range_exclusive"] == [0, 5]


def test_actual_escalation_still_fails_after_independent_video_check(video, tmp_path):
    plan = SimpleNamespace(layout_id=0, run_dir=tmp_path)
    row = {"status": "evaluated", "episode_results": [{"evaluation": {"reset_epoch": 1}}]}
    summary = {"cleanup": {"forced": True}}
    with pytest.raises(RuntimeError, match="SIGTERM/SIGKILL"):
        shared_layouts._finalize_videos(video.parent, [plan], [row], summary)
    assert summary["video_verification"]["status"] == "verified"


def test_shared_video_must_match_frame_sidecar(video, tmp_path):
    path = video.parent / "observer.frames.jsonl"
    path.write_text(path.read_text() + json.dumps({"frame": 5, "reset_epoch": 1}) + "\n")
    plan = SimpleNamespace(layout_id=0, run_dir=tmp_path)
    row = {"status": "evaluated", "episode_results": [{"evaluation": {"reset_epoch": 1}}]}
    with pytest.raises(RuntimeError, match="frame count mismatch"):
        shared_layouts._finalize_videos(video.parent, [plan], [row], {"cleanup": {"forced": False}})
