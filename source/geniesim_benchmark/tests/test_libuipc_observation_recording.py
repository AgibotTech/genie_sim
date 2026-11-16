# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""IPC videos retain simulation time while replay state stays observation sparse."""

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from geniesim_benchmark.libuipc.recording import EpisodeRecording

CAMERAS = ("head", "hand_left", "hand_right")


def packet(snapshot_id, sim_time, *, value=50, source_id="source", epoch=4):
    jpeg = cv2.imencode(".jpg", np.full((120, 320, 3), value, dtype=np.uint8))[1].tobytes()
    return {
        "snapshot": {
            "source_id": source_id,
            "epoch": epoch,
            "snapshot_id": snapshot_id,
            "request_id": "request-" + snapshot_id,
            "sim_time": sim_time,
            "tick": round(sim_time * 1000),
            "cloth_tick": round(sim_time * 100),
            "joints": {"measured": value / 100},
            "bodies": {"/robot/base_link": [1, 2, 3, 1, 0, 0, 0]},
        },
        "images": {name: {"encoding": "JPEG", "image_data": jpeg, "width": 320, "height": 120} for name in CAMERAS},
    }


def recorder(output, *, record_state=True):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    return EpisodeRecording(
        output, 4, "Hold the observed cloth", fps=10, record_state=record_state, fixed_joints={"welded": 0.25}
    )


def rows(bag):
    return [json.loads(line) for line in (Path(bag) / "state.jsonl").read_text().splitlines()]


def video(path):
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return stream, np.frombuffer(raw, dtype=np.uint8).reshape(-1, 120, 320, 3)


def test_sparse_observations_preserve_simulation_duration_and_paired_state(tmp_path):
    recording = recorder(tmp_path)
    recording.capture(packet("initial", 1.25, value=30), 0)
    recording.capture(packet("final", 2.25, value=210), 500)
    bag = Path(recording.close())
    samples = rows(bag)
    assert [row["sim_time"] for row in samples[1:]] == [1.25, 2.25]
    assert [row.get("snapshot_id") for row in samples[1:]] == ["initial", "final"]
    assert [row["frame_index"] for row in samples[1:]] == [0, 10]
    assert [row["step"] for row in samples[1:]] == [0, 500]
    assert samples[1]["joints"] == {"name": ["measured", "welded"], "position": [0.3, 0.25]}
    assert samples[1]["base_pose"] == {"position": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]}
    for name in CAMERAS:
        stream, frames = video(bag / "video" / (name + ".webm"))
        assert stream["r_frame_rate"] == "10/1"
        assert int(stream["nb_read_frames"]) == 11
        assert np.max(np.abs(frames[:-1, 65:75, :20].astype(int) - 30)) < 10
        assert np.max(np.abs(frames[-1, 65:75, :20].astype(int) - 210)) < 10
    assert recording.close() == str(bag)


def test_snapshot_identity_dedup_keeps_distinct_snapshots_on_same_action_step(tmp_path):
    recording = recorder(tmp_path)
    first = packet("first", 0.0, value=30)
    recording.capture(first, 0)
    recording.capture(packet("second", 0.0, value=90), 0)
    recording.capture(first, 999)  # Exact packet reused non-consecutively at this timestamp.
    recording.capture(packet("third", 0.4, value=90), 0)  # Identical image does not imply identical snapshot.
    bag = Path(recording.close())
    samples = rows(bag)[1:]
    assert [row["snapshot_id"] for row in samples] == ["first", "second", "third"]
    assert [row["source_id"] for row in samples] == ["source"] * 3
    assert [row["epoch"] for row in samples] == [4] * 3
    assert [row["sim_time"] for row in samples] == [0.0, 0.0, 0.4]
    assert [row["frame_index"] for row in samples] == [0, 0, 4]
    _, frames = video(bag / "video/hand_left.webm")
    assert len(frames) == 5
    assert np.max(np.abs(frames[:, 65:75, :20].astype(int) - 90)) < 10


@pytest.mark.parametrize(
    "change",
    [
        {"sim_time": 0.5},
        {"sim_time": float("nan")},
        {"sim_time": -1.0},
        {"source_id": "foreign"},
        {"epoch": 5},
        {"snapshot_id": ""},
    ],
)
def test_invalid_or_foreign_snapshots_fail_before_new_outputs(tmp_path, change):
    recording = recorder(tmp_path)
    recording.capture(packet("first", 1.0), 0)
    changed = packet("second", 1.1)
    changed["snapshot"].update(change)
    try:
        with pytest.raises(ValueError, match="(sim_time|identity|snapshot|epoch|source)"):
            recording.capture(changed, 1)
    finally:
        bag = Path(recording.close())
    assert len(rows(bag)) == 2
    _, frames = video(bag / "video/hand_left.webm")
    assert len(frames) == 1


def test_no_state_flag_still_preserves_simulation_duration(tmp_path):
    recording = recorder(tmp_path, record_state=False)
    recording.capture(packet("first", 0.0), 0)
    recording.capture(packet("last", 0.5), 20)
    bag = Path(recording.close())
    assert not (bag / "state.jsonl").exists()
    _, frames = video(bag / "video/hand_right.webm")
    assert len(frames) == 6


def test_hold_intervals_exceeding_encoder_queue_keep_every_timeline_frame(tmp_path):
    recording = recorder(tmp_path, record_state=False)
    recording.capture(packet("first", 0.0), 0)
    recording.capture(packet("last", 30.0), 900)
    bag = Path(recording.close())
    for name in CAMERAS:
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
                str(bag / "video" / (name + ".webm")),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert int(json.loads(probe.stdout)["streams"][0]["nb_read_frames"]) == 301


def test_close_preserves_frame_order_when_worker_already_holds_first_frame(tmp_path, monkeypatch):
    """Actual shared encoder close must not drain a newer frame ahead of its worker."""
    from queue import Queue
    from threading import Event, Thread
    from types import SimpleNamespace
    import time

    from geniesim_benchmark.libuipc import recording as recording_module
    from geniesim_benchmark.plugins.output_system.local_recorder import _CameraEncoder

    taken, release, first_written = Event(), Event(), Event()
    written = []

    class DelayedQueue(Queue):
        def get(self, block=True, timeout=None):
            frame = super().get(block=block, timeout=timeout)
            if block and not taken.is_set():
                taken.set()
                assert release.wait(5), "test worker was never released"
            return frame

    class Pipe:
        closed = False

        def write(self, data):
            written.append(data[0])
            if data[0] == 1:
                release.set()  # The old shared closer writes this before frame zero.
            else:
                first_written.set()

        def close(self):
            assert first_written.wait(5), "worker never wrote the earlier frame"
            self.closed = True

    encoder = object.__new__(_CameraEncoder)
    encoder.camera_name = "hand_left"
    encoder.queue = DelayedQueue(maxsize=2)
    encoder._stop = Event()
    encoder.frames_written = encoder.frames_dropped = 0
    encoder._proc = SimpleNamespace(stdin=Pipe(), poll=lambda: None, wait=lambda **kwargs: None)
    encoder.queue.put(np.array([0], dtype=np.uint8))
    encoder.queue.put(np.array([1], dtype=np.uint8))
    encoder._worker = Thread(target=encoder._drain, name="ipc-order-regression", daemon=True)
    encoder._worker.start()
    assert taken.wait(5), "worker never acquired the earlier frame"

    recording = EpisodeRecording(tmp_path, 4, "Preserve frame order", fps=10, record_state=False, fixed_joints={})
    recording._next_frame = 2
    recording.recorder._encoders = {"hand_left": encoder}
    monkeypatch.setattr(recording.recorder, "stop", lambda index: encoder.close())

    def allow_worker_to_finish(seconds):
        release.set()  # A correct IPC drain wait allows the in-flight first frame to finish.
        time.sleep(seconds)

    monkeypatch.setattr(recording_module, "time", SimpleNamespace(sleep=allow_worker_to_finish))
    try:
        recording.close()
    finally:
        release.set()
        encoder._stop.set()
        encoder._worker.join(timeout=5)
    with (tmp_path / "encoded-order.json").open("x") as stream:
        json.dump(written, stream)
    assert written == [0, 1]
    assert encoder.frames_written == 2


def test_publish_observation_states_preserves_sources_and_is_idempotent(tmp_path):
    from geniesim_benchmark.libuipc import recording as recording_module

    publish = getattr(recording_module, "publish_observation_states", None)
    assert callable(publish), "IPC must publish its own sparse paired state files"
    bags = [tmp_path / "bags/recording_0004_first", tmp_path / "bags/recording_0005_second"]
    expected = []
    output = tmp_path / "published"
    for index, bag in enumerate(bags):
        bag.mkdir(parents=True)
        data = (json.dumps({"snapshot_id": "sample-" + str(index), "sim_time": index / 10}) + "\n").encode()
        with (bag / "state.jsonl").open("xb") as stream:
            stream.write(data)
        expected.append((output / (bag.name + ".state.jsonl"), data))
    artifacts = publish(bags, output)
    assert artifacts == [path for path, _ in expected]
    assert publish(bags, output) == artifacts
    for bag, (destination, data) in zip(bags, expected):
        assert destination.read_bytes() == data
        assert (bag / "state.jsonl").read_bytes() == data


def test_publish_observation_states_rejects_different_existing_output(tmp_path):
    from geniesim_benchmark.libuipc import recording as recording_module

    publish = getattr(recording_module, "publish_observation_states", None)
    assert callable(publish), "IPC must publish its own sparse paired state files"
    bag = tmp_path / "recording_0004"
    bag.mkdir()
    source = bag / "state.jsonl"
    with source.open("xb") as stream:
        stream.write(b"new sparse measurement\n")
    destination = tmp_path / (bag.name + ".state.jsonl")
    with destination.open("xb") as stream:
        stream.write(b"existing sparse measurement\n")
    with pytest.raises(FileExistsError, match="different|existing|overwrite"):
        publish([bag], tmp_path)
    assert destination.read_bytes() == b"existing sparse measurement\n"
    assert source.read_bytes() == b"new sparse measurement\n"


def test_publish_observation_states_requires_each_committed_state_file(tmp_path):
    from geniesim_benchmark.libuipc import recording as recording_module

    publish = getattr(recording_module, "publish_observation_states", None)
    assert callable(publish), "IPC must publish its own sparse paired state files"
    bag = tmp_path / "recording_0004"
    bag.mkdir()
    with pytest.raises(FileNotFoundError, match="state.jsonl"):
        publish([bag], tmp_path / "published")
    assert not (tmp_path / "published" / (bag.name + ".state.jsonl")).exists()


def test_interrupted_state_copy_preserves_partial_staging_and_allows_retry(tmp_path, monkeypatch):
    from geniesim_benchmark.libuipc import recording as recording_module

    bag = tmp_path / "recording_0004"
    bag.mkdir()
    payload = b'{"snapshot_id":"measured-observation","sim_time":0.25}\n'
    source = bag / "state.jsonl"
    with source.open("xb") as stream:
        stream.write(payload)
    output = tmp_path / "published"
    destination = output / (bag.name + ".state.jsonl")
    copy_file = recording_module.shutil.copyfileobj

    def interrupted_copy(reader, writer):
        writer.write(reader.read(7))
        raise OSError("injected interrupted state copy")

    monkeypatch.setattr(recording_module.shutil, "copyfileobj", interrupted_copy)
    with pytest.raises(OSError, match="injected interrupted"):
        recording_module.publish_observation_states([bag], output)
    assert not destination.exists(), "an incomplete copy must never occupy the final artifact path"
    partials = [path for path in output.iterdir() if path.is_file()]
    assert len(partials) == 1
    assert partials[0].read_bytes() == payload[:7]
    monkeypatch.setattr(recording_module.shutil, "copyfileobj", copy_file)
    assert recording_module.publish_observation_states([bag], output) == [destination]
    assert destination.read_bytes() == payload
    assert source.read_bytes() == payload
    assert partials[0].read_bytes() == payload[:7], "interrupted staging data must be retained"
    assert recording_module.publish_observation_states([bag], output) == [destination]
