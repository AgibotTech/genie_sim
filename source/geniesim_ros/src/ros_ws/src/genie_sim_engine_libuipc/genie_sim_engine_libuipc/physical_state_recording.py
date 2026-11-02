# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Lossless committed IPC states, independent of ROS publication and scoring.

Locally adapted from the physical state recorder in history 0ab2b58;
no sibling backend is imported.

Scene YAML ``libuipc.state_recording: {enabled: true, output_dir: ...}`` is the
only configuration route. One initial state and every subsequent physical
frame are appended without dtype conversion. Each array file and the index
are flushed per frame; an interrupted solver retains all committed frames.
This is not a power-loss durability guarantee. Readers reject partial tails,
missing ticks, schema changes, nonfinite values and checksum disagreement.

All six arrays are mandatory, including valid zero-count arrays. An empty
array is accepted only when its name is listed in caller-provided metadata
``zero_count_arrays``. This policy never permits a missing array. Metadata
also identifies the caller's ``source_id`` and nonnegative integer ``epoch``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

STATE_ARRAYS = ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")


def _metadata(value):
    if not isinstance(value, dict):
        raise ValueError("physical recording metadata must be an object")
    if not isinstance(value.get("source_id"), str) or not value["source_id"].strip():
        raise ValueError("physical recording metadata requires source_id")
    if type(value.get("epoch")) is not int or value["epoch"] < 0:
        raise ValueError("physical recording metadata requires a nonnegative integer epoch")
    policy = value.get("zero_count_arrays", [])
    if (
        not isinstance(policy, list)
        or any(not isinstance(name, str) for name in policy)
        or len(policy) != len(set(policy))
        or set(policy) - set(STATE_ARRAYS)
    ):
        raise ValueError("zero_count_arrays must list unique physical array names")


def _schemas(manifest):
    _metadata(manifest.get("metadata"))
    schemas = manifest.get("arrays")
    if not isinstance(schemas, dict) or set(schemas) != set(STATE_ARRAYS):
        raise ValueError("physical state requires all six array schemas")
    for name, schema in schemas.items():
        try:
            if not isinstance(schema, dict) or not isinstance(schema.get("dtype"), str):
                raise ValueError("invalid schema/dtype")
            dtype = np.dtype(schema["dtype"])
            shape = schema["shape"]
            if (
                dtype.kind not in "fiu"
                or not isinstance(shape, list)
                or any(type(value) is not int or value < 0 for value in shape)
                or schema["file"] != name + ".bin"
                or type(schema["frame_bytes"]) is not int
                or schema["frame_bytes"] != math.prod(shape) * dtype.itemsize
            ):
                raise ValueError("invalid shape/dtype/size")
            if not math.prod(shape) and name not in manifest["metadata"].get("zero_count_arrays", []):
                raise ValueError("empty array without a zero_count_arrays policy")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid physical array schema: {name}") from exc
    return schemas


def _manifest_markers(manifest):
    if not isinstance(manifest, dict) or manifest.get("format") != "geniesim_physical_state_v1":
        raise ValueError("unsupported physical state format")
    if type(manifest.get("recording_closed")) is not bool:
        raise ValueError("physical recording_closed marker is required and must be a bool")
    if type(manifest.get("solver_shutdown_clean")) is not bool:
        raise ValueError("invalid physical shutdown marker")
    if "recording_clean" in manifest and (
        type(manifest["recording_clean"]) is not bool
        or manifest["recording_clean"] != manifest["solver_shutdown_clean"]
    ):
        raise ValueError("physical recording clean markers disagree")
    if manifest["solver_shutdown_clean"] and not manifest["recording_closed"]:
        raise ValueError("physical manifest is clean before recording closed")
    if manifest["solver_shutdown_clean"] and "recording_failure" in manifest:
        raise ValueError("physical recording failed and cannot be clean")
    if type(manifest.get("frame_count")) is not int or manifest["frame_count"] < 0:
        raise ValueError("invalid physical manifest frame count")


def _saved_array(saved, *, index, size, name):
    if (
        not isinstance(saved, dict)
        or type(saved.get("offset")) is not int
        or type(saved.get("bytes")) is not int
        or saved["offset"] != index * size
        or saved["bytes"] != size
        or not isinstance(saved.get("sha256"), str)
        or len(saved["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in saved["sha256"])
    ):
        raise ValueError(f"invalid physical array offset/size/hash: {name}")


@dataclass(frozen=True)
class PhysicalStateRecordingConfig:
    output_dir: str

    @classmethod
    def from_scene(cls, scene_cfg):
        block = (scene_cfg.get("libuipc") or {}).get("state_recording")
        if block is None:
            return None
        if not isinstance(block, dict) or set(block) - {"enabled", "output_dir"}:
            raise ValueError("libuipc.state_recording permits only enabled and output_dir")
        enabled = block.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("libuipc.state_recording.enabled must be a bool")
        if not enabled:
            return None
        output = block.get("output_dir")
        if not isinstance(output, str) or not output.strip() or "\x00" in output:
            raise ValueError("libuipc.state_recording.output_dir is required when enabled")
        return cls(output)


def _timing(tick, sim_time, dt, initial, previous=None):
    if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
        raise ValueError("physical tick must be a nonnegative integer")
    if not isinstance(initial, bool):
        raise ValueError("initial must be a bool")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (sim_time, dt)):
        raise ValueError("sim_time and dt must be finite numbers")
    if previous is None:
        if tick != 0 or not initial or sim_time != 0 or dt != 0:
            raise ValueError("first frame must be the initial tick 0 at simulation time 0")
    elif (
        tick != previous["tick"] + 1
        or initial
        or dt <= 0
        or sim_time <= previous["sim_time"]
        or not math.isclose(sim_time, previous["sim_time"] + dt, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError("physical frames require contiguous ticks and increasing simulation time")


class PhysicalStateRecorder:
    """Append all committed IPC state arrays; close explicitly on shutdown."""

    def __init__(self, config, *, metadata):
        _metadata(metadata)
        # Detach nested caller-owned objects and reject non-JSON/nonfinite data
        # before creating the fresh output directory.
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self.directory = Path(config.output_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self._files = {}
        self._index = (self.directory / "frames.jsonl").open("w", encoding="utf-8")
        self._previous = None
        self._closed = False
        self._failure = None
        self._manifest = {
            "format": "geniesim_physical_state_v1",
            "coverage": "initial_and_every_committed_ipc_physical_step",
            "units": {"length": "m", "time": "s", "joint_angles": "rad"},
            "metadata": metadata,
            "arrays": {},
            "frame_count": 0,
            "solver_shutdown_clean": False,
            "recording_clean": False,
            "recording_closed": False,
        }
        try:
            self._checkpoint()
        except BaseException:
            self._index.close()
            self._closed = True
            raise

    @property
    def frame_count(self):
        return self._manifest["frame_count"]

    @property
    def closed(self):
        return self._closed

    def _checkpoint(self):
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(self._manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.directory / "manifest.json")

    def record(self, state, *, tick, sim_time, dt, initial=False, deformable_tick=None):
        self.require_healthy()
        try:
            self._record(state, tick=tick, sim_time=sim_time, dt=dt, initial=initial, deformable_tick=deformable_tick)
        except Exception as exc:
            # The engine loop can catch step errors and try stepping again.
            # A failed post-solve write must prevent every subsequent solve;
            # otherwise the archive could silently omit physical states.
            self._failure = exc
            self._manifest["recording_failure"] = f"{type(exc).__name__}: {exc}"
            self._checkpoint()
            raise

    def require_healthy(self):
        """Reject all later advances/writes after the first recording failure."""
        if self._failure is not None:
            raise RuntimeError(f"physical recording failed; solver advance blocked: {self._failure}") from self._failure
        if self._closed:
            raise ValueError("physical state recorder is closed")

    def _record(self, state, *, tick, sim_time, dt, initial=False, deformable_tick=None):
        _timing(tick, sim_time, dt, initial, self._previous)
        snapshots, schemas = {}, {}
        for name in STATE_ARRAYS:
            value = getattr(state, name, None)
            if value is None:
                raise ValueError(f"physical state requires {name}; missing arrays are not permitted")
            array = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError(f"physical state {name} must be a finite numeric array")
            if not array.size and name not in self._manifest["metadata"].get("zero_count_arrays", []):
                raise ValueError(f"physical state {name} is empty without a zero_count_arrays policy")
            array = np.ascontiguousarray(array)
            snapshots[name] = array.tobytes(order="C")
            schemas[name] = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "frame_bytes": array.nbytes,
                "file": name + ".bin",
            }
        if self.frame_count and schemas != self._manifest["arrays"]:
            raise ValueError("physical state array shapes/dtypes/availability changed")
        if not self.frame_count:
            self._manifest["arrays"] = schemas
            for name, schema in schemas.items():
                self._files[name] = (self.directory / schema["file"]).open("wb")
            self._checkpoint()
        frame = {
            "tick": tick,
            "sim_time": float(sim_time),
            "dt": float(dt),
            "initial": initial,
            "deformable_tick": deformable_tick,
            "arrays": {},
        }
        for name, payload in snapshots.items():
            output = self._files[name]
            frame["arrays"][name] = {
                "offset": output.tell(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            if output.write(payload) != len(payload):
                raise OSError(f"short physical array write: {name}")
            output.flush()
        line = json.dumps(frame, separators=(",", ":")) + "\n"
        if self._index.write(line) != len(line):
            raise OSError("short physical frame index write")
        self._index.flush()
        self._previous = frame
        self._manifest["frame_count"] += 1
        self._manifest["last_tick"] = tick
        self._manifest["last_sim_time"] = float(sim_time)
        if self.frame_count % 128 == 0:
            self._checkpoint()

    def close(self, *, clean=True):
        if self._closed:
            return
        error = None
        try:
            for output in (*self._files.values(), self._index):
                try:
                    output.flush()
                except Exception as exc:
                    error = error or exc
                try:
                    output.close()
                except Exception as exc:
                    error = error or exc
            self._manifest["solver_shutdown_clean"] = False
            self._manifest["recording_clean"] = False
            # The reader must keep waiting throughout final verification;
            # publishing a dirty terminal marker here would race clean close.
            self._checkpoint()
            if error is None and clean and self.frame_count and self._failure is None:
                try:
                    # Verify every committed byte before publishing a clean
                    # terminal manifest. A checksum failure poisons this owner.
                    archive = PhysicalStateArchive(self.directory, require_clean=False)
                    for _ in archive.iter_frames():
                        pass
                    self._manifest["solver_shutdown_clean"] = True
                    self._manifest["recording_clean"] = True
                except Exception as exc:
                    error = exc
            if error is not None:
                self._failure = self._failure or error
                self._manifest["recording_failure"] = f"{type(error).__name__}: {error}"
            self._manifest["recording_closed"] = True
            self._checkpoint()
        finally:
            self._closed = True
        if error is not None:
            raise error


class PhysicalStateArchive:
    """Strict streaming reader; allow interrupted archives only explicitly."""

    def __init__(self, directory, *, require_clean=True):
        self.directory = Path(directory)
        try:
            self.manifest = json.loads((self.directory / "manifest.json").read_text())
            raw = (self.directory / "frames.jsonl").read_text()
            if not raw or not raw.endswith("\n"):
                raise ValueError("physical frame index is empty or has a partial tail")
            self._frames = [json.loads(line) for line in raw.splitlines()]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid physical archive: {exc}") from exc
        _manifest_markers(self.manifest)
        clean = self.manifest.get("solver_shutdown_clean") is True
        if require_clean and (not clean or not self.manifest["recording_closed"]):
            raise ValueError("physical state solver shutdown is not clean")
        schemas = _schemas(self.manifest)
        previous = None
        for index, frame in enumerate(self._frames):
            if (
                not isinstance(frame, dict)
                or not isinstance(frame.get("arrays"), dict)
                or set(frame["arrays"]) != set(schemas)
            ):
                raise ValueError("physical index arrays disagree with manifest")
            _timing(frame.get("tick"), frame.get("sim_time"), frame.get("dt"), frame.get("initial"), previous)
            for name, schema in schemas.items():
                _saved_array(frame["arrays"][name], index=index, size=schema["frame_bytes"], name=name)
            previous = frame
        for name, schema in schemas.items():
            path = self.directory / schema["file"]
            if not path.is_file() or path.stat().st_size != len(self._frames) * schema["frame_bytes"]:
                raise ValueError(f"physical array truncated or has an extra tail: {name}")
        checkpoint_count = self.manifest.get("frame_count")
        if (
            isinstance(checkpoint_count, bool)
            or not isinstance(checkpoint_count, int)
            or not 0 <= checkpoint_count <= len(self._frames)
            or (self.manifest["recording_closed"] and checkpoint_count != len(self._frames))
        ):
            raise ValueError("physical manifest frame count disagrees with index")
        if self.manifest["recording_closed"] and (
            type(self.manifest.get("last_tick")) is not int
            or self.manifest["last_tick"] != previous["tick"]
            or self.manifest.get("last_sim_time") != previous["sim_time"]
        ):
            raise ValueError("physical manifest final clock disagrees with index")

    def iter_frames(self):
        files = {}
        try:
            for name, schema in self.manifest["arrays"].items():
                files[name] = (self.directory / schema["file"]).open("rb")
            for frame in self._frames:
                arrays = {}
                for name, schema in self.manifest["arrays"].items():
                    expected = frame["arrays"][name]
                    payload = files[name].read(expected["bytes"])
                    if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected.get(
                        "sha256"
                    ):
                        raise ValueError(f'physical array checksum mismatch: {name} tick {frame["tick"]}')
                    array = np.frombuffer(payload, dtype=np.dtype(schema["dtype"])).reshape(schema["shape"])
                    if not np.isfinite(array).all():
                        raise ValueError(f'nonfinite physical state: {name} tick {frame["tick"]}')
                    arrays[name] = array
                yield {**{k: v for k, v in frame.items() if k != "arrays"}, "state": arrays}
        finally:
            for source in files.values():
                source.close()


class PhysicalStateArchiveFollower:
    """Read committed index lines while recording, verify only after clean close.

    Construction does not wait. ``initialize()`` waits for the first schema;
    ``iter_frames()`` also initializes automatically and may be consumed once.
    ``timeout_s`` is a total monotonic deadline, including consumer work.
    ``cancel`` is a callable returning true to raise InterruptedError.

    Array bytes are checked against each committed row before yielding. At
    close, one sequential byte-hash pass detects rewrites of already-consumed
    data; it neither decodes arrays again nor reruns any geometry/scoring.
    Early generator close, cancellation and dirty shutdown never set verified.
    """

    _DYNAMIC = frozenset(
        (
            "frame_count",
            "last_tick",
            "last_sim_time",
            "solver_shutdown_clean",
            "recording_clean",
            "recording_closed",
            "recording_failure",
        )
    )

    def __init__(self, directory, *, timeout_s=None, cancel=None, poll_seconds=0.05):
        if (
            isinstance(poll_seconds, bool)
            or not isinstance(poll_seconds, (int, float))
            or not math.isfinite(poll_seconds)
            or poll_seconds <= 0
        ):
            raise ValueError("poll_seconds must be finite and positive")
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s < 0
        ):
            raise ValueError("timeout_s must be finite and nonnegative")
        if cancel is not None and not callable(cancel):
            raise TypeError("cancel must be callable")
        self.directory = Path(directory)
        self.manifest = None
        self.verified = False
        self.verification = None
        self.frames_read = 0
        self._deadline = None if timeout_s is None else time.monotonic() + timeout_s
        self._cancel = cancel
        self._poll_seconds = poll_seconds
        self._static_manifest = None
        self._iterated = False

    def _control(self):
        if self._cancel is not None and self._cancel():
            raise InterruptedError("physical archive following cancelled")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutError("physical archive following timed out")

    def _wait(self):
        self._control()
        delay = self._poll_seconds
        if self._deadline is not None:
            delay = min(delay, max(0.0, self._deadline - time.monotonic()))
        time.sleep(delay)
        self._control()

    def _manifest(self):
        try:
            raw = (self.directory / "manifest.json").read_bytes()
        except FileNotFoundError:
            if self.manifest is not None:
                raise ValueError("physical manifest disappeared during recording")
            return None, None
        try:
            manifest = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid physical manifest") from exc
        _manifest_markers(manifest)
        if "recording_failure" in manifest:
            raise ValueError("physical recording failed: " + str(manifest["recording_failure"]))
        if manifest.get("recording_closed") is True and not manifest["solver_shutdown_clean"]:
            raise ValueError("physical state solver shutdown is not clean")
        if manifest.get("recording_closed") is False and manifest["solver_shutdown_clean"]:
            raise ValueError("physical manifest is clean before recording closed")
        count = manifest.get("frame_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("invalid physical manifest frame count")
        static = {key: value for key, value in manifest.items() if key not in self._DYNAMIC}
        if self._static_manifest is not None and static != self._static_manifest:
            raise ValueError("physical manifest schema or metadata changed")
        return manifest, raw

    def initialize(self):
        self._control()
        if self.manifest is not None:
            return self
        while True:
            manifest, _ = self._manifest()
            if manifest is not None:
                schemas = manifest.get("arrays")
                if schemas == {} and not manifest["solver_shutdown_clean"]:
                    self._wait()
                    continue
                _schemas(manifest)
                self.manifest = manifest
                self._static_manifest = {key: value for key, value in manifest.items() if key not in self._DYNAMIC}
                return self
            self._wait()

    @staticmethod
    def _identity(source):
        stat = os.fstat(source.fileno())
        return stat.st_dev, stat.st_ino

    def _final_hash(self, path, expected_size, identity):
        """One raw sequential reread, to catch modifications after a frame yielded."""
        before = path.stat()
        if (before.st_dev, before.st_ino) != identity or before.st_size != expected_size:
            raise ValueError(f"physical file changed, truncated or has an extra tail: {path.name}")
        value = hashlib.sha256()
        with path.open("rb") as source:
            if self._identity(source) != identity:
                raise ValueError(f"physical file replaced: {path.name}")
            while True:
                self._control()
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                value.update(chunk)
        after = path.stat()
        attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in attributes):
            raise ValueError(f"physical file changed during final verification: {path.name}")
        return value.hexdigest()

    def _finish(self, manifest, manifest_raw, index, index_bytes, index_hash, files, array_hashes, previous):
        count = self.frames_read
        if (
            not count
            or manifest["frame_count"] != count
            or type(manifest.get("last_tick")) is not int
            or isinstance(manifest.get("last_sim_time"), bool)
            or not isinstance(manifest.get("last_sim_time"), (int, float))
            or manifest.get("last_tick") != previous["tick"]
            or manifest.get("last_sim_time") != previous["sim_time"]
        ):
            raise ValueError("physical manifest final clock/count disagrees with committed index")
        index_digest = self._final_hash(self.directory / "frames.jsonl", index_bytes, self._identity(index))
        if index_digest != index_hash.hexdigest():
            raise ValueError("physical index changed after committed rows were consumed")
        hashes = {}
        for name, schema in manifest["arrays"].items():
            value = self._final_hash(
                self.directory / schema["file"], count * schema["frame_bytes"], self._identity(files[name])
            )
            if value != array_hashes[name].hexdigest():
                raise ValueError(f"physical array checksum changed after consumption: {name}")
            hashes[name] = value
        final_manifest, final_raw = self._manifest()
        if final_raw != manifest_raw:
            raise ValueError("physical manifest changed during final verification")
        self._control()
        self.manifest = final_manifest
        self.verification = {
            "physical_frames": count,
            "all_physical_frames_verified": True,
            "solver_shutdown_clean": True,
            "recording_closed": True,
            "recording_clean": True,
            "physical_manifest_sha256": hashlib.sha256(final_raw).hexdigest(),
            "physical_index_sha256": index_digest,
            "array_sha256": hashes,
        }
        self.verified = True

    def iter_frames(self):
        if self._iterated:
            raise RuntimeError("physical archive follower can only be consumed once")
        self._iterated = True
        self.initialize()
        schemas = self.manifest["arrays"]
        files = {}
        previous, index_bytes = None, 0
        index_hash = hashlib.sha256()
        array_hashes = {name: hashlib.sha256() for name in schemas}
        try:
            for name, schema in schemas.items():
                files[name] = (self.directory / schema["file"]).open("rb")
            with (self.directory / "frames.jsonl").open("rb") as index:
                while True:
                    self._control()
                    index.seek(index_bytes)
                    raw = index.readline()
                    if not raw.endswith(b"\n"):
                        manifest, manifest_raw = self._manifest()
                        if manifest["recording_closed"]:
                            # EOF may have raced the last append before close.
                            index.seek(index_bytes)
                            tail = index.readline()
                            if tail.endswith(b"\n"):
                                continue
                            if tail:
                                raise ValueError("physical frame index has a partial tail after close")
                            self._finish(
                                manifest, manifest_raw, index, index_bytes, index_hash, files, array_hashes, previous
                            )
                            return
                        self._wait()
                        continue
                    try:
                        frame = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError("invalid committed physical frame index line") from exc
                    if (
                        not isinstance(frame, dict)
                        or not isinstance(frame.get("arrays"), dict)
                        or set(frame["arrays"]) != set(schemas)
                    ):
                        raise ValueError("physical index arrays disagree with manifest")
                    _timing(frame.get("tick"), frame.get("sim_time"), frame.get("dt"), frame.get("initial"), previous)
                    arrays = {}
                    for name, schema in schemas.items():
                        saved = frame["arrays"][name]
                        size = schema["frame_bytes"]
                        _saved_array(saved, index=self.frames_read, size=size, name=name)
                        payload = files[name].read(size)
                        if len(payload) != size or hashlib.sha256(payload).hexdigest() != saved.get("sha256"):
                            raise ValueError(f'physical array checksum mismatch: {name} tick {frame["tick"]}')
                        array = np.frombuffer(payload, dtype=np.dtype(schema["dtype"])).reshape(schema["shape"])
                        if not np.isfinite(array).all():
                            raise ValueError(f'nonfinite physical state: {name} tick {frame["tick"]}')
                        array_hashes[name].update(payload)
                        arrays[name] = array
                    index_hash.update(raw)
                    index_bytes += len(raw)
                    self.frames_read += 1
                    previous = frame
                    yield {**{key: value for key, value in frame.items() if key != "arrays"}, "state": arrays}
        except OSError as exc:
            if isinstance(exc, (InterruptedError, TimeoutError)):
                raise
            raise ValueError(f"invalid physical archive while following: {exc}") from exc
        finally:
            for source in files.values():
                source.close()
