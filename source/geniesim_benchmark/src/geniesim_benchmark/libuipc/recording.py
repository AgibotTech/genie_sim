# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Sample-and-hold IPC videos paired with sparse, identified observations."""

import json
import math
import os
import shutil
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import cv2
import numpy as np

from .adapter import _compose_fixed_pose, wire_images
from geniesim_benchmark.plugins.output_system.local_recorder import LocalRecorder

# The sparse IPC state writer owns this schema independently of the video-only
# shared recorder. Retain the established header value for existing consumers.
STATE_SCHEMA = "geniesim.local-recorder-state.v1"


class EpisodeRecording:
    """Reuse observed pixels on a simulation-time grid; never request a render.

    ``state.jsonl`` contains one row per unique snapshot, including snapshots at
    the same simulation time. Its ``frame_index`` is the first video grid index
    at or after that observation, and can be shared when a newer observation
    supersedes it before that grid tick. The exact snapshot identity remains in
    every state row. Only the latest decoded packet is retained in memory.
    """

    def __init__(
        self, output, episode_idx, instruction, *, fps, record_state, fixed_joints, frame_bindings=None, run_id=None
    ):
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0 < fps < math.inf:
            raise ValueError("Recording fps must be finite and positive")
        self.output = Path(output)
        self.episode_idx, self.fps = episode_idx, fps
        self.fixed_joints, self.frame_bindings = fixed_joints, frame_bindings
        # State sampling is IPC-owned: repeated video frames must not invent
        # repeated measured states. Leave the shared recorder's defaults intact.
        self.recorder = LocalRecorder(fps=fps, record_state=False)
        self.recorder.restore_checkpoint(run_id or uuid.uuid4().hex, [])
        self.recorder.update_instruction(instruction)
        self._record_state = record_state
        self._state_file = None
        self._bag_path = None
        self._source_epoch = None
        self._first_time = self._last_time = None
        self._timestamp_snapshots = set()
        self._cache = None
        self._next_frame = 0
        self._closed = False

    def _validate_snapshot(self, snapshot):
        source, epoch, token = (snapshot.get(key) for key in ("source_id", "epoch", "snapshot_id"))
        if not isinstance(source, str) or not source or type(epoch) is not int or epoch < 0:
            raise ValueError("Recording requires a valid source_id and epoch identity")
        if not isinstance(token, str) or not token:
            raise ValueError("Recording requires a nonempty snapshot_id")
        if self._source_epoch is not None and (source, epoch) != self._source_epoch:
            raise ValueError("Recording snapshot belongs to a foreign source_id or epoch")
        clock = snapshot.get("sim_time")
        if isinstance(clock, bool) or not isinstance(clock, (int, float)) or not math.isfinite(clock) or clock < 0:
            raise ValueError("Recording sim_time must be finite and nonnegative")
        if self._last_time is not None and clock < self._last_time:
            raise ValueError("Recording sim_time cannot move backwards")
        return (source, epoch, token), clock

    def _sample(self, snapshot):
        joints = {**snapshot["joints"], **self.fixed_joints}
        if self.frame_bindings is not None:
            binding = self.frame_bindings["frames"]["base_link"]
            pose = _compose_fixed_pose(snapshot["bodies"][binding["body_path"]], binding["body_from_frame_pose"])
        else:
            matches = [pose for path, pose in snapshot["bodies"].items() if path.rsplit("/", 1)[-1] == "base_link"]
            if len(matches) != 1:
                raise ValueError("Recording requires an unambiguous base_link pose")
            pose = matches[0]
        return {
            "joints": {"name": list(joints), "position": list(joints.values())},
            "base_pose": {"position": list(pose[:3]), "orientation_xyzw": [*pose[4:], pose[3]]},
        }

    def _grid_position(self, clock):
        value = (clock - self._first_time) * self.fps
        # Physics accumulates floating-point dt. Snap only numerical noise near
        # integer grid positions, so a nominal 0.3 s at 10 Hz stays frame three.
        nearest = round(value)
        return nearest if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-7) else value

    def _write_video_until(self, stop_index):
        while self._next_frame < stop_index:
            # Shared encoders deliberately drop full queues for live rendering.
            # IPC fills gaps in bursts, so wait for their bounded queues here.
            # These are the only private recorder details needed by this adapter.
            encoders = tuple(self.recorder._encoders.values())
            while any(encoder.queue.full() for encoder in encoders):
                if any(encoder._proc.poll() is not None or not encoder._worker.is_alive() for encoder in encoders):
                    raise RuntimeError("IPC recording encoder stopped before completing the simulation timeline")
                time.sleep(0.001)
            self.recorder.write_frames(SimpleNamespace(_img_data_cache=self._cache), self._next_frame)
            self._next_frame += 1

    def capture(self, packet, step):
        if self._closed:
            raise RuntimeError("Cannot capture into a closed IPC recording")
        snapshot = packet["snapshot"]
        identity, clock = self._validate_snapshot(snapshot)
        if clock == self._last_time and identity in self._timestamp_snapshots:
            return
        images = wire_images(packet["images"])
        cache = {}
        for name, row in images.items():
            image = cv2.imdecode(np.frombuffer(row["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (row["height"], row["width"]):
                raise ValueError("Recording requires a decodable camera image with declared dimensions")
            cache[name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        sample = self._sample(snapshot) if self._record_state else None
        if self._first_time is None:
            self._first_time = clock
        frame_index = math.ceil(self._grid_position(clock))
        if sample is not None:
            sample = {
                "frame_index": frame_index,
                "step": step,
                "sim_time": clock,
                **{
                    key: snapshot[key]
                    for key in (
                        "source_id",
                        "epoch",
                        "snapshot_id",
                        "request_id",
                        "tick",
                        "cloth_tick",
                        "captured_wall_time_ns",
                    )
                    if key in snapshot
                },
                **sample,
            }
            serialized = json.dumps(sample, allow_nan=False) + "\n"
        if not self.recorder.is_recording:
            specs = [
                {"camera_id": name, "width": row["width"], "height": row["height"]} for name, row in images.items()
            ]
            self._bag_path = self.recorder.start(
                str(self.output.parent), self.episode_idx, specs, sub_task_name=self.output.name
            )
            if self._record_state:
                self._state_file = (Path(self._bag_path) / "state.jsonl").open("x", encoding="utf-8")
                self._state_file.write(
                    json.dumps(
                        {
                            "schema": STATE_SCHEMA,
                            "episode_idx": self.episode_idx,
                            "fps": self.fps,
                            "every_n_frame": 1,
                            "sampling": "unique_observation_snapshot",
                            "video_time_origin": self._first_time,
                            "frame_index_semantics": "first_grid_tick_at_or_after_observation",
                        }
                    )
                    + "\n"
                )
        if self._cache is not None:
            self._write_video_until(frame_index)
        if self._state_file is not None:
            self._state_file.write(serialized)
            self._state_file.flush()
        if clock != self._last_time:
            self._timestamp_snapshots = set()
        self._timestamp_snapshots.add(identity)
        self._source_epoch = identity[:2]
        self._last_time = clock
        self._cache = cache

    def close(self):
        if self._closed:
            return self._bag_path
        encoders = tuple(self.recorder._encoders.values())
        try:
            if self._cache is not None:
                # Include the final observed pixels, rounded to the next grid
                # tick if needed. No video time is inferred from wall time.
                self._write_video_until(math.ceil(self._grid_position(self._last_time)) + 1)
            # The shared closer drains queued frames before joining its worker.
            # Wait for both queued and in-flight frames here so its main thread
            # cannot write a newer frame ahead of an earlier worker-held frame.
            while any(encoder.frames_written != self._next_frame for encoder in encoders):
                if any(encoder._proc.poll() is not None or not encoder._worker.is_alive() for encoder in encoders):
                    raise RuntimeError("IPC recording encoder stopped before completing the simulation timeline")
                time.sleep(0.001)
        finally:
            try:
                self.recorder.stop(self.episode_idx)
            finally:
                if self._state_file is not None:
                    self._state_file.close()
                self._cache = None
                self._timestamp_snapshots = set()
                self._closed = True
        if any(encoder.frames_written != self._next_frame or encoder.frames_dropped for encoder in encoders):
            raise RuntimeError("IPC recording encoder did not preserve every simulation timeline frame")
        return self._bag_path


def publish_observation_states(bags, output_dir):
    """Copy committed sparse state files without moving or replacing any data.

    Each source remains in its episode bag. Repeated publication accepts only
    byte-identical existing outputs; conflicting outputs fail explicitly.
    """
    sources = [Path(bag) / "state.jsonl" for bag in bags]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Committed IPC observation state is missing: {source}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for source in sources:
        destination = output_dir / f"{source.parent.name}.state.jsonl"
        # Copy into a unique retained file first. A failed copy never occupies
        # the final name, and hard-link publication cannot overwrite a winner.
        staging = output_dir / f".{destination.name}.{uuid.uuid4().hex}.staging"
        with source.open("rb") as reader, staging.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        try:
            os.link(staging, destination)
        except FileExistsError:
            with source.open("rb") as reader, destination.open("rb") as existing:
                while True:
                    expected = reader.read(1024 * 1024)
                    actual = existing.read(1024 * 1024)
                    if expected != actual:
                        raise FileExistsError(f"Cannot overwrite different existing IPC state: {destination}")
                    if not expected:
                        break
        artifacts.append(destination)
    return artifacts
