# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Single-writer, episode-level evaluation commits. No simulator dependencies."""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

COORD_KEYS = ("instance_id", "file_idx", "episode_id")

CHECKPOINT_REJECTED_EXIT_CODE = 78


class CheckpointRejected(ValueError):
    """A stored checkpoint cannot be resumed from. Distinguishes "this
    checkpoint is unusable" from "the run failed", so the caller can retry
    without resume instead of failing the evaluation."""


def atomic_write_json(path, value):
    """Publish a complete JSON document in the destination filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
            os.fchmod(stream.fileno(), mode)
            json.dump(value, stream, indent=4, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def coordinate(entry):
    values = tuple(entry[key] for key in COORD_KEYS)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Invalid episode coordinate")
    return values


def build_episode_plan(instance_ids, task_files, instruction_counts, repetitions):
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("Cannot checkpoint duplicate instance ids")
    if type(repetitions) is not int or repetitions < 1 or not instance_ids or not task_files:
        raise ValueError("An evaluation plan must contain episodes")
    plan = []
    for instance_id in instance_ids:
        count = instruction_counts[instance_id]
        if type(count) is not int or count < 1:
            raise ValueError("An evaluation instance must contain instructions")
        for file_idx, _ in enumerate(task_files):
            for episode_id in range(repetitions * count):
                plan.append(
                    dict(episode_idx=len(plan), instance_id=instance_id, file_idx=file_idx, episode_id=episode_id)
                )
    return plan


class EvaluationCheckpoint:
    """The manifest is the commit marker; an uncommitted result tail is retryable.

    Keep the lock for the entire task, including input preparation and artifact
    finalization. A new non-resume run gets its own identity and result file.
    """

    def __init__(self, out_dir, resume=False):
        self.out_dir = Path(out_dir).resolve()
        self.path = self.out_dir / "checkpoint_manifest.json"
        self.resume = resume
        self.state = None
        self._lock = None

    def __enter__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = open(self.out_dir / ".evaluation.lock", "a+b")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.close()
            raise RuntimeError(f"Evaluation directory is already in use: {self.out_dir}") from None
        try:
            if self.resume and self.path.exists():
                with self.path.open() as stream:
                    self.state = json.load(stream)
                if self.state.get("version") != 1 or not isinstance(self.state.get("run_id"), str):
                    raise CheckpointRejected("Unsupported evaluation checkpoint; start in a new output directory")
                if uuid.UUID(hex=self.state["run_id"]).hex != self.state["run_id"]:
                    raise CheckpointRejected("Invalid checkpoint run identity")
                self._local_path(self.state["result_file"])
                self._validate_plan()
            else:
                self.run_id = uuid.uuid4().hex
                index = 0
                while (self.out_dir / f"evaluate_ret_{index:02d}.json").exists():
                    index += 1
                self._result_file = f"evaluate_ret_{index:02d}.json"
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    @property
    def run_id(self):
        return self.state["run_id"] if self.state is not None else self._run_id

    @run_id.setter
    def run_id(self, value):
        self._run_id = value

    @property
    def task_dir(self):
        return self.out_dir / "_checkpoint_tasks" / self.run_id

    @property
    def result_path(self):
        return self._local_path(self.state["result_file"] if self.state is not None else self._result_file)

    def _local_path(self, name):
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Checkpoint paths must be local file names")
        path = self.out_dir / name
        if path.is_symlink():
            raise ValueError("Checkpoint paths must not be symlinks")
        return path

    def _validate_plan(self, state=None):
        state = self.state if state is None else state
        plan = state["plan"]
        if not plan or [p["episode_idx"] for p in plan] != list(range(len(plan))):
            raise ValueError("Invalid checkpoint episode plan")
        entries = {coordinate(p): p for p in plan}
        if len(entries) != len(plan):
            raise ValueError("Duplicate checkpoint episode coordinates")
        seen = set()
        for entry in state["completed"] + state["skipped"]:
            key = coordinate(entry)
            if key not in entries or key in seen or entry["episode_idx"] != entries[key]["episode_idx"]:
                raise ValueError("Invalid or duplicate committed episode")
            seen.add(key)
            if entry.get("bag") is not None:
                self._local_path(entry["bag"])
        if state is self.state:
            self._plan = entries
            self._completed = {coordinate(e) for e in state["completed"]}
            self._skipped = {coordinate(e) for e in state["skipped"]}

    def _publish(self, state):
        self._validate_plan(state)
        atomic_write_json(self.path, state)
        self.state = state
        self._validate_plan()

    def prepare(self, identity, plan, task_files, task_config):
        if self.state is not None:
            raise ValueError("Checkpoint is already prepared")
        state = dict(
            version=1,
            run_id=self.run_id,
            result_file=self._result_file,
            identity=deepcopy(identity),
            plan=deepcopy(plan),
            completed=[],
            skipped=[],
            artifacts=None,
            task_config=deepcopy(task_config),
            task_files=[dict(name=Path(p).name, sha256=file_digest(p)) for p in task_files],
        )
        self._publish(state)

    def restore(self, identity):
        if digest(identity) != digest(self.state["identity"]):
            raise CheckpointRejected("Checkpoint configuration or task inputs changed; use a new evaluation directory")
        if {p.name for p in self.task_dir.glob("*.json")} != {e["name"] for e in self.state["task_files"]}:
            raise CheckpointRejected("Checkpoint generated task file inventory changed")
        for item in self.state["task_files"]:
            path = self.task_dir / item["name"]
            if Path(item["name"]).name != item["name"] or not path.is_file() or file_digest(path) != item["sha256"]:
                raise CheckpointRejected("Checkpoint generated task file is missing or changed")
        if not self.result_path.is_file():
            if self.state["completed"]:
                raise CheckpointRejected("Checkpoint result file is missing")
            return []
        with self.result_path.open() as stream:
            document = json.load(stream)
        committed = {coordinate(e): e for e in self.state["completed"]}
        recovered = {}
        for row in document["details"]:
            key = coordinate(row["episode_coord"])
            if key not in committed:
                continue
            if key in recovered or digest(row) != committed[key]["result_digest"]:
                raise CheckpointRejected("Checkpoint result is duplicated or changed")
            recovered[key] = row
        if recovered.keys() != committed.keys():
            raise CheckpointRejected("Checkpoint result is missing a committed episode")
        return [
            recovered[coordinate(entry)] for entry in sorted(self.state["completed"], key=lambda e: e["episode_idx"])
        ]

    def is_completed(self, entry):
        return coordinate(entry) in self._completed

    def is_skipped(self, entry):
        return coordinate(entry) in self._skipped

    def episode_index(self, entry):
        return self._plan[coordinate(entry)]["episode_idx"]

    @property
    def progress(self):
        planned = len(self.state["plan"])
        committed = len(self.state["completed"])
        skipped = len(self.state["skipped"])
        remaining = planned - committed - skipped
        return dict(planned=planned, committed=committed, skipped=skipped, remaining=remaining, finished=remaining == 0)

    @property
    def finished(self):
        return self.progress["finished"]

    def commit(self, detail, bag=None, skip_remaining=False):
        key = coordinate(detail["episode_coord"])
        if (
            key not in self._plan
            or self.is_completed(detail["episode_coord"])
            or self.is_skipped(detail["episode_coord"])
        ):
            raise ValueError("Episode is outside the plan or already committed")
        # Confirm that the result was published before exposing the commit marker.
        with self.result_path.open() as stream:
            rows = json.load(stream)["details"]
        matches = [r for r in rows if coordinate(r["episode_coord"]) == key]
        if len(matches) != 1 or digest(matches[0]) != digest(detail):
            raise ValueError("Cannot commit an episode before its result is saved")
        bag_name = None
        if bag is not None:
            bag_path = Path(bag).resolve()
            if bag_path.parent != self.out_dir or not bag_path.is_dir():
                raise ValueError("Recording is outside the evaluation directory or missing")
            bag_name = bag_path.name
        state = deepcopy(self.state)
        entry = {**self._plan[key], "bag": bag_name, "result_digest": digest(detail)}
        state["completed"].append(entry)
        if skip_remaining:
            state["skipped"].extend(
                {**p, "reason": "episode loop aborted"}
                for p in state["plan"]
                if p["instance_id"] == key[0]
                and p["file_idx"] == key[1]
                and p["episode_id"] > key[2]
                and coordinate(p) not in self._completed
                and coordinate(p) not in self._skipped
            )
        self._publish(state)

    def recording_paths(self):
        paths = [
            self._local_path(e["bag"])
            for e in sorted(self.state["completed"], key=lambda e: e["episode_idx"])
            if e["bag"] is not None
        ]
        if not self.recordings_ready and any(not p.is_dir() for p in paths):
            raise ValueError("A committed recording is missing before finalization")
        return [str(p) for p in paths]

    @property
    def recordings_ready(self):
        artifacts = self.state["artifacts"]
        return artifacts is not None and all(
            self._local_path(e["name"]).is_file() and self._local_path(e["name"]).stat().st_size == e["size"]
            for e in artifacts
        )

    def finalize_recordings(self, artifacts):
        state = deepcopy(self.state)
        state["artifacts"] = [dict(name=Path(p).name, size=Path(p).stat().st_size) for p in artifacts]
        self._publish(state)

    def cleanup_recordings(self, committed=False):
        """Only remove bags marked as owned by this run, never another run's data."""
        keep = {e["bag"] for e in self.state["completed"] if e["bag"] is not None}
        for path in self.out_dir.glob("recording_*"):
            marker = path / "checkpoint_run.json"
            if path.is_symlink() or not path.is_dir() or not marker.is_file():
                continue
            with marker.open() as stream:
                owner = json.load(stream)
            if owner.get("run_id") == self.run_id and (committed or path.name not in keep):
                shutil.rmtree(path)
