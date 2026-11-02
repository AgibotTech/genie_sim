#!/usr/bin/env python3
"""Run a CPU-only archive follower and publish its score after final verification."""

from __future__ import annotations

import argparse
import hashlib
import secrets
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

SCORERS = ("fold_cloth", "cloth_hanger", "drawstring_bag")
STOP_GRACE_SECONDS = 5.0
KILL_GRACE_SECONDS = 5.0


def _validate_success_token(token):
    if token is not None and (not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{32}", token) is None):
        raise ValueError("success token must contain 32 lowercase hexadecimal characters")


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {description}: expected a JSON object")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_count(verification):
    if (
        not isinstance(verification, dict)
        or verification.get("solver_shutdown_clean") is not True
        or verification.get("all_physical_frames_verified") is not True
        or verification.get("recording_closed") is not True
    ):
        raise ValueError("Streaming score requires a verified, clean physical archive")
    count = verification.get("physical_frames")
    if type(count) is not int or count < 2:
        raise ValueError("Verified physical frame count must be an integer >= 2")
    return count


def _require_worker_checkout(repo_root):
    expected = (Path(repo_root) / "source/geniesim_benchmark/src/geniesim_benchmark/libuipc" / "streaming.py").resolve()
    if expected != Path(__file__).resolve():
        raise ValueError("Streaming scoring worker and repo_root must use the same checkout")


def _live_group(pgid):
    """Return live members of the private session, excluding unreapable zombies."""
    live = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid or fields[0] in ("Z", "X"):
                continue
            if int(fields[3]) != pgid or process.stat().st_uid != os.getuid():
                raise RuntimeError("Streaming worker group ownership cannot be verified")
            live.append(int(process.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # A restricted /proc may hide other users' processes. Our own
            # private session must remain inspectable before cleanup/publish.
            try:
                if process.stat().st_uid == os.getuid():
                    raise RuntimeError("Streaming worker group cannot be inspected")
            except (FileNotFoundError, ProcessLookupError):
                continue
    return live


class StreamingScoringJob:
    """Own one scorer process; partial output stays private until finish()."""

    def __init__(self, task, episode_dir, repo_root, *, drain_timeout_s=900.0, success_token=None):
        if task not in SCORERS:
            raise ValueError("Unknown Wuji scoring task")
        if (
            isinstance(drain_timeout_s, bool)
            or not isinstance(drain_timeout_s, (int, float))
            or not math.isfinite(drain_timeout_s)
            or drain_timeout_s <= 0
        ):
            raise ValueError("scoring drain timeout must be finite and positive")
        _validate_success_token(success_token)
        self.task = task
        self.episode_dir = Path(episode_dir).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.drain_timeout_s = float(drain_timeout_s)
        self.success_token = secrets.token_hex(16) if success_token is None else success_token
        self.staging = self.episode_dir / ".scoring_stream"
        self.output = self.episode_dir / "scoring"
        self.log_path = self.episode_dir / "stream_scoring.log"
        self.process = None
        self._log = None
        self._started = False
        self._stopped = False
        self._report = None
        self._success = None

    def _command(self):
        command = [
            sys.executable,
            "-B",
            "-m",
            "geniesim_benchmark.libuipc.streaming",
            "--worker",
            "--task",
            self.task,
            "--episode",
            str(self.episode_dir),
            "--repo-root",
            str(self.repo_root),
        ]
        if self.success_token is not None:
            command += ["--success-token", self.success_token]
        return command

    def start(self):
        if self._started or self._stopped:
            raise RuntimeError("Streaming scoring job can only be started once")
        _require_worker_checkout(self.repo_root)
        if not self.episode_dir.is_dir():
            raise ValueError("Episode directory must exist before starting streaming scoring")
        for path in (self.staging, self.output, self.log_path, self.episode_dir / "stream_scoring_worker.json"):
            if os.path.lexists(path):
                raise FileExistsError(f"Streaming scoring output already exists: {path}")
        try:
            self._log = self.log_path.open("xb")
            self.process = subprocess.Popen(
                self._command(),
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._started = True
        except BaseException:
            self.stop()
            raise
        return self

    def check(self):
        if self.process is None:
            raise RuntimeError("Streaming scoring worker has not started")
        code = self.process.poll()
        if code is not None and code != 0:
            raise RuntimeError(f"Streaming scoring worker exited {code}; inspect {self.log_path}")

    def success_notification(self):
        """Return confirmed private evidence bound to this running worker and episode."""
        if not self._started or self._stopped:
            raise RuntimeError("Streaming scoring job is not running")
        self.check()
        if self.success_token is None:
            return None
        path = self.staging / "checker_success.json"
        if not os.path.lexists(path):
            if self._success is not None:
                raise ValueError("Accepted success notification disappeared")
            return None
        if self.staging.is_symlink() or path.is_symlink() or not path.is_file():
            raise ValueError("Invalid success notification file")
        notification = _read_json(path, "success notification")
        expected = {
            "schema": "wuji_success_v1",
            "token": self.success_token,
            "task": self.task,
            "episode": str(self.episode_dir),
        }
        if any(notification.get(key) != value for key, value in expected.items()):
            raise ValueError("Success notification identity differs from this scoring job")
        canonical = _read_json(
            self.repo_root
            / "source/geniesim_benchmark/src/geniesim_benchmark/benchmark/config"
            / "llm_task/wuji_ipc"
            / self.task
            / "checker_profile.json",
            "canonical scoring profile",
        )
        staged = _read_json(self.staging / "checker_profile.json", "success notification profile")
        if staged != canonical or notification.get("profile_id") != canonical.get("profile_id"):
            raise ValueError("Success notification differs from the canonical scoring profile")
        score, sim_time = notification.get("score"), notification.get("sim_time_s")
        if (
            notification.get("success") is not True
            or type(score) not in (int, float)
            or not math.isfinite(score)
            or score != 1
            or type(sim_time) not in (int, float)
            or not math.isfinite(sim_time)
            or sim_time < 0
            or type(notification.get("tick")) is not int
            or notification["tick"] < 0
            or type(notification.get("observed_frames")) is not int
            or notification["observed_frames"] < 2
        ):
            raise ValueError("Success notification contains invalid score, success or frame evidence")
        if self._success is not None and notification != self._success:
            raise ValueError("Accepted success notification changed")
        self._success = notification.copy()
        return notification

    def _validate_success_report(self, report, count):
        notification = self.success_notification()
        if notification is None:
            return
        if report.get("success") is not True or report.get("score") != 1 or notification["observed_frames"] > count:
            raise ValueError("Final streaming report contradicts its success notification")
        try:
            for path, clock_key, check_score in (
                (self.staging / "checker_frames.jsonl", "sim_time_s", True),
                (self.episode_dir / "physical_state/frames.jsonl", "sim_time", False),
            ):
                with path.open() as stream:
                    for index, line in enumerate(stream, 1):
                        if index == notification["observed_frames"]:
                            frame = json.loads(line)
                            break
                    else:
                        raise ValueError("Success notification frame is missing from final archive")
                if (
                    not isinstance(frame, dict)
                    or frame.get("tick") != notification["tick"]
                    or frame.get(clock_key) != notification["sim_time_s"]
                    or (check_score and (type(frame.get("score")) not in (int, float) or frame["score"] != 1))
                ):
                    raise ValueError("Success notification frame differs from final archive")
        except (OSError, ValueError) as exc:
            raise ValueError("Final archive does not include the success notification frame") from exc

    def _validate_report(self, report, count):
        if report.get("scoring_status") != "scored" or report.get("scoring_framework") != "ader":
            raise ValueError("Streaming report does not contain a completed ADER score")
        if not isinstance(report.get("profile_id"), str) or not report["profile_id"].strip():
            raise ValueError("Streaming report is missing its scoring profile")
        canonical = _read_json(
            self.repo_root
            / "source/geniesim_benchmark/src/geniesim_benchmark/benchmark/config"
            / "llm_task/wuji_ipc"
            / self.task
            / "checker_profile.json",
            "canonical scoring profile",
        )
        staged = _read_json(self.staging / "checker_profile.json", "streaming scoring profile")
        if staged != canonical or report["profile_id"] != canonical.get("profile_id"):
            raise ValueError("Streaming scoring profile differs from this checkout's canonical profile")
        score = report.get("score")
        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
            or type(report.get("success")) is not bool
        ):
            raise ValueError("Streaming report contains an invalid score or success value")
        if type(report.get("observed_frames")) is not int or report["observed_frames"] != count:
            raise ValueError("Streaming report frame count differs from verified archive")
        manifest = _read_json(self.episode_dir / "physical_state/manifest.json", "physical manifest")
        if (
            manifest.get("solver_shutdown_clean") is not True
            or manifest.get("recording_closed") is not True
            or manifest.get("frame_count") != count
        ):
            raise ValueError("Physical manifest does not match the verified clean frame count")
        hashes = {
            "asset_identity_sha256": self.episode_dir / "asset_identity.json",
            "physical_manifest_sha256": self.episode_dir / "physical_state/manifest.json",
            "physical_index_sha256": self.episode_dir / "physical_state/frames.jsonl",
        }
        for key, path in hashes.items():
            try:
                matches = report.get(key) == _sha(path)
            except OSError as exc:
                raise ValueError(f"Streaming report hash source is unavailable: {path}") from exc
            if not matches:
                raise ValueError(f"Streaming report hash mismatch: {key}")

    def finish(self, verification):
        if self._report is not None:
            return self._report
        if not self._started or self._stopped:
            raise RuntimeError("Streaming scoring job is not running")
        try:
            count = _verified_count(verification)
            try:
                self.process.wait(timeout=self.drain_timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("Streaming scoring worker exceeded its drain timeout") from exc
            self.check()
            if _live_group(self.process.pid):
                raise RuntimeError("Streaming scoring worker group remained live after leader exit")
            if not self.staging.is_dir() or self.staging.is_symlink():
                raise ValueError("Streaming scoring staging report directory is missing or invalid")
            report = _read_json(self.staging / "checker_result.json", "streaming scoring report")
            self._validate_report(report, count)
            self._validate_success_report(report, count)
            if os.path.lexists(self.output):
                raise FileExistsError(f"Refusing to replace existing scoring directory: {self.output}")
            # Drain and reap the worker before taking the final full archive
            # snapshot. Validating before wait() leaves its entire drain window
            # open to array mutations that report/index hashes cannot detect.
            saved = _read_json(self.episode_dir / "recording_verified.json", "archive verification")
            if _verified_count(saved) != count:
                raise ValueError("Archive verification frame counts disagree")
            from genie_sim_engine_libuipc.physical_state_recording import PhysicalStateArchiveFollower

            final = PhysicalStateArchiveFollower(
                self.episode_dir / "physical_state", timeout_s=self.drain_timeout_s
            ).initialize()
            for _ in final.iter_frames():
                pass
            for key in (
                "physical_frames",
                "all_physical_frames_verified",
                "solver_shutdown_clean",
                "recording_closed",
                "physical_manifest_sha256",
                "physical_index_sha256",
                "array_sha256",
            ):
                if saved.get(key) != final.verification.get(key) or verification.get(key) != final.verification.get(
                    key
                ):
                    raise ValueError(f"Archive verification differs from finalized archive: {key}")
            # Staging and destination share a parent/filesystem. Only this rename
            # exposes a scored checker report to callers.
            self.staging.rename(self.output)
            self._report = report
            self._stopped = True
            return report
        except BaseException:
            self.stop()
            raise
        finally:
            if self.process is not None and self.process.poll() is not None:
                self._close_log()

    def _close_log(self):
        if self._log is not None:
            self._log.close()
            self._log = None

    def _wait_for_empty_group(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            code = self.process.poll()  # Reap the direct child as soon as it exits.
            if not _live_group(self.process.pid) and code is not None:
                self.process.wait(timeout=0)
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.01, remaining))

    def stop(self):
        """Bound termination, escalation and reaping, including orphaned group members."""
        try:
            if self._stopped:
                return
            if self.process is None:
                self._stopped = True
                return
            if self._wait_for_empty_group(0):
                self._stopped = True
                return
            for signum, grace in ((signal.SIGTERM, STOP_GRACE_SECONDS), (signal.SIGKILL, KILL_GRACE_SECONDS)):
                if _live_group(self.process.pid):
                    try:
                        os.killpg(self.process.pid, signum)
                    except ProcessLookupError:
                        pass
                elif self.process.poll() is None:
                    # The leader may be crossing exec/exit while /proc is read.
                    self.process.send_signal(signum)
                if self._wait_for_empty_group(grace):
                    self._stopped = True
                    return
            raise RuntimeError("Streaming scoring worker group survived forced termination")
        finally:
            self._close_log()


def _load_scorer(task):
    from functools import partial
    from types import SimpleNamespace
    from geniesim_benchmark.libuipc.scoring import score_physical_episode

    return SimpleNamespace(score_physical_episode=partial(score_physical_episode, task))


def _worker_resources():
    # This executes only after the independent child has started, before NumPy
    # or task scorers are imported. The parent and simulation environment stay intact.
    limits = {
        name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    }
    os.environ.update(limits)
    os.nice(5)
    cpu = max(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    return {
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "allowed_cpus": sorted(os.sched_getaffinity(0)),
        "thread_limits": limits,
    }


def _worker_main(task, episode_dir, repo_root, success_token=None):
    episode_dir, repo_root = Path(episode_dir).resolve(), Path(repo_root).resolve()
    _require_worker_checkout(repo_root)
    _validate_success_token(success_token)
    started = time.time_ns()
    resources = _worker_resources()
    receipt_path = episode_dir / "stream_scoring_worker.json"
    receipt = {
        "phase": "running",
        "task": task,
        "episode": str(episode_dir),
        "repo_root": str(repo_root),
        "pid": os.getpid(),
        "started_wall_ns": started,
        "first_progress_wall_ns": None,
        "policy_calls": 0,
        **resources,
    }
    _write_json(receipt_path, receipt)
    staging = episode_dir / ".scoring_stream"
    stopped = threading.Event()
    progress = {}

    def observe_progress():
        while not stopped.wait(0.01):
            try:
                with (staging / "checker_frames.jsonl").open() as stream:
                    first = stream.readline()
                if not first.endswith("\n"):
                    continue
                frame = json.loads(first)
                progress.update(first_progress_wall_ns=time.time_ns(), first_progress_tick=frame.get("tick"))
                return
            except (OSError, ValueError):
                pass

    observer = threading.Thread(target=observe_progress, name="stream-score-progress", daemon=True)
    observer.start()
    try:
        scorer = _load_scorer(task)
        options = {}
        if success_token is not None:

            def on_success(evidence):
                _write_json(
                    staging / "checker_success.json",
                    {
                        **evidence,
                        "schema": "wuji_success_v1",
                        "token": success_token,
                        "task": task,
                        "episode": str(episode_dir),
                    },
                )

            options["on_success"] = on_success
        scorer.score_physical_episode(
            episode_dir, staging, repo_root, follow=True, cancel=None, timeout_s=None, **options
        )
        receipt["phase"] = "completed"
    except BaseException:
        receipt.update(phase="failed", error=traceback.format_exc())
        raise
    finally:
        stopped.set()
        observer.join(timeout=1.0)
        receipt.update(progress)
        receipt["finished_wall_ns"] = time.time_ns()
        _write_json(receipt_path, receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task", choices=SCORERS, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--success-token")
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("This entry point requires --worker")
    _worker_main(args.task, args.episode, args.repo_root, success_token=args.success_token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
