from __future__ import annotations

import base64
import fcntl
import hashlib
import io
import json
import logging
import queue
import shutil
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .adapters import (
    HTTPEvaluator,
    HTTPPolicy,
    LocalVideoEvaluator,
    ReplayPolicy,
    image_base64,
    save_json,
)
from .config import DreamConfig
from .contracts import EvaluationRequest, RunRequest, SessionRequest, StepRequest
from .examples import Example, discover_examples
from .runtime import Runtime, create_runtime, history_indices
from .session_actions import (
    assemble_normalized_actions,
    write_session_actions,
)
from .vision import (
    CONCAT_LAYOUT,
    CONCAT_SIZE,
    RAW_FD_SIZE,
    ensure_session_geometry_consistency,
    to_canonical_frame,
    validate_raw_output_geometry,
)

logger = logging.getLogger(__name__)


class Conflict(ValueError):
    pass


class Busy(ValueError):
    pass


@dataclass
class Session:
    session_id: str
    domain: str
    caption: str
    policy: str
    directory: Path
    current_image: Path
    example: Example | None
    state: dict | None
    state_source: str
    history: list[Path] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    next_step: int = 0
    active_job: str | None = None
    stop: bool = False
    finished: bool = False
    touched: float = field(default_factory=time.time)
    initial_image: Path | None = None
    num_steps: int = 10
    seed: int = 42
    output_geometry: tuple[int, int] | None = None
    video_metadata: dict | None = None
    chunk_length: int = 32
    actions_meta: dict | None = None


class DreamPlatform:
    def __init__(self, config: DreamConfig, runtime: Runtime | None = None):
        self.config = config.for_model(config.default_model_id or config.model_id)
        config = self.config
        self.manual_model_selection = bool(
            config.model_profiles and config.default_model_id is None and runtime is None
        )
        self.config.output_dir = self.config.output_dir.resolve()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_lock = (self.config.output_dir / ".platform.lock").open("a")
        try:
            fcntl.flock(self.output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.output_lock.close()
            raise Busy("Another Dream process owns this output_dir") from None
        self.examples, self.rejected = discover_examples(config)
        self.sessions: dict[str, Session] = {}
        self.jobs: dict[str, dict] = {}
        self.requests: dict[tuple[str, str], tuple[str, str]] = {}
        self.lock = threading.RLock()
        self.tasks = queue.Queue(maxsize=config.queue_size)
        self.runtime = runtime
        self.ready = False
        self.closed = False
        self.startup_error: str | None = None
        self.switch_state = "unselected" if self.manual_model_selection else "loading"
        self.switch_error: str | None = None
        self.switching_model = False
        self.pending_model_id = config.default_model_id
        self.worker_busy = False
        self.vlm_registry = None
        self.vlm_evaluator: LocalVideoEvaluator | None = None
        self.vlm_error: str | None = None
        self.worker = threading.Thread(target=self._worker, name="dream-gpu-worker", daemon=True)
        self.worker.start()

    def _should_init_local_vlm(self) -> bool:
        if self.config.evaluator_url:
            return False
        if self.config.vlm_models_config is not None:
            return True
        return self.config.backend == "cosmos"

    def _init_local_vlm(self) -> None:
        """Preload default VLM (qwen3_5); other keys switch on demand."""
        try:
            from vlm_registry import default_models_config_path, load_model_registry

            config_path = self.config.vlm_models_config or default_models_config_path()
            default_key = self.config.vlm_default_model_key or "qwen3_5"
            registry = load_model_registry(
                Path(config_path),
                preload_keys=[default_key],
                default_model_key=default_key,
            )
            evaluator = LocalVideoEvaluator(
                registry=registry,
                shared_output_root=self.config.output_dir,
            )
            with self.lock:
                self.vlm_registry = registry
                self.vlm_evaluator = evaluator
                self.vlm_error = None
            logger.info(
                "In-process VLM ready: active=%s loaded=%s",
                registry.active_key,
                registry.loaded_paths(),
            )
        except Exception as error:  # noqa: BLE001
            logger.exception("In-process VLM initialization failed")
            with self.lock:
                self.vlm_registry = None
                self.vlm_evaluator = None
                self.vlm_error = f"{type(error).__name__}: {error}"

    def _evaluation_available(self) -> bool:
        return self.vlm_evaluator is not None or bool(self.config.evaluator_url)

    def _worker(self):
        try:
            if self.runtime is None and not self.manual_model_selection:
                self.runtime = create_runtime(self.config)
            if self.runtime is not None:
                # Preload default VLM before advertising ready (same GPU worker thread).
                # Skip mock unless vlm_models_config is explicitly set (keeps unit tests light).
                if self._should_init_local_vlm():
                    self._init_local_vlm()
                with self.lock:
                    self.ready = True
                    self.switch_state = "ready"
                    self.pending_model_id = None
        except BaseException:
            logger.exception("Dream runtime startup failed")
            with self.lock:
                self.startup_error = "Runtime initialization failed; see server logs"
                self.switch_state = "failed"
        while True:
            task = self.tasks.get()
            if task is None:
                self.tasks.task_done()
                break
            function, future = task
            with self.lock:
                self.worker_busy = True
            try:
                if future.set_running_or_notify_cancel():
                    future.set_result(function())
            except BaseException as error:
                future.set_exception(error)
            finally:
                with self.lock:
                    self.worker_busy = False
                self.tasks.task_done()
        if self.runtime:
            self.runtime.close()

    def model_catalog(self):
        if self.config.model_profiles:
            return [
                {
                    "id": profile.id,
                    "label": profile.label,
                    "label_en": profile.label_en or profile.label,
                    "supported_domains": profile.supported_domains,
                }
                for profile in self.config.model_profiles
            ]
        return [
            {
                "id": self.config.model_id,
                "label": self.config.model_id,
                "label_en": self.config.model_id,
                "supported_domains": list(self.config.domains),
            }
        ]

    def model_status(self):
        return {
            "models": self.model_catalog(),
            "active_model_id": self.config.model_id if self.ready else None,
            "default_num_steps": self.config.num_steps,
            "pending_model_id": self.pending_model_id,
            "ready": self.ready,
            "switch_state": self.switch_state,
            "switching": self.switching_model,
            "switch_error": self.switch_error,
        }

    def _supported_domains(self):
        if not self.ready:
            return None
        if self.config.model_profiles:
            profile = next(
                (item for item in self.config.model_profiles if item.id == self.config.model_id),
                None,
            )
            return set(profile.supported_domains) if profile else set()
        return set(self.config.domains) if self.config.domains else None

    def available_examples(self):
        supported = self._supported_domains()
        return [
            example.public()
            for example in self.examples.values()
            if supported is None or example.domain in supported
        ]

    def select_model(self, model_id: str):
        with self.lock:
            if self.closed:
                raise Busy("Platform is shutting down")
            if self.config.model_profiles:
                next_config = self.config.for_model(model_id)
            elif model_id == self.config.model_id:
                return self.model_status()
            else:
                raise ValueError(f"Unknown model profile: {model_id}")
            if model_id == self.config.model_id and self.ready:
                return self.model_status()
            if not self.ready and self.switch_state not in {"failed", "unselected"}:
                raise Busy("Model is not ready or another model switch is in progress")
            if self.sessions:
                raise Conflict("Delete all sessions before switching models")
            if self.worker_busy or not self.tasks.empty():
                raise Busy("Wait for the GPU worker queue to become idle")
            # Replay arrays may be normalized with different stats for each WM.
            # Prepare its catalog before changing the active runtime, and publish
            # both together only after the replacement has loaded successfully.
            next_examples, next_rejected = discover_examples(next_config)
            previous_config = self.config
            previous_runtime = self.runtime
            self.ready = False
            self.switching_model = True
            self.switch_state = "loading"
            self.switch_error = None
            self.pending_model_id = model_id
            self.startup_error = None
            future = Future()
            self.tasks.put_nowait(
                (
                    lambda: self._replace_runtime(
                        next_config, previous_config, previous_runtime, next_examples, next_rejected
                    ),
                    future,
                )
            )
            return self.model_status()

    def _replace_runtime(
        self, next_config, previous_config, previous_runtime, next_examples, next_rejected
    ):
        try:
            if previous_runtime is not None:
                previous_runtime.close()
            self.runtime = None
            runtime = create_runtime(next_config)
        except Exception as error:
            logger.exception("Failed to switch to model %s", next_config.model_id)
            restored_runtime = None
            if previous_runtime is not None:
                try:
                    restored_runtime = create_runtime(previous_config)
                except Exception:
                    logger.exception("Failed to restore model %s", previous_config.model_id)
            with self.lock:
                self.runtime = restored_runtime
                self.config = previous_config
                self.ready = restored_runtime is not None
                self.switching_model = False
                self.pending_model_id = None
                self.startup_error = (
                    None if self.ready else "Runtime initialization failed; see server logs"
                )
                self.switch_state = "failed"
                self.switch_error = f"Could not load {next_config.model_id}: {error}"
            return
        with self.lock:
            self.runtime = runtime
            self.config = next_config
            self.examples = next_examples
            self.rejected = next_rejected
            self.ready = True
            self.switching_model = False
            self.pending_model_id = None
            self.startup_error = None
            self.switch_state = "ready"
            self.switch_error = None
        if self._should_init_local_vlm() and self.vlm_evaluator is None:
            self._init_local_vlm()

    def close(self):
        with self.lock:
            self.closed = True
            for session in self.sessions.values():
                session.stop = True
        self.tasks.put(None)
        self.worker.join()
        self.ready = False
        self.output_lock.close()

    def wait_ready(self, timeout: float = 30):
        deadline = time.monotonic() + timeout
        while not self.ready and self.startup_error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.ready

    def capabilities(self):
        return {
            "protocol": "dream.v1",
            "model_id": self.config.model_id if self.ready else None,
            "backend": self.config.backend,
            "fresh_inference": self.config.backend == "cosmos",
            "ready": self.ready,
            "chunk_length": self.config.chunk_length,
            "fps": 10,
            "num_conditional_frames": 1,
            "history_bucket_frames": [1, 1, 1, 1],
            "history_sampling": "endpoints_fixed",
            "action_representation": "stateend_29d_backward_framewise_rot6d",
            "domains": {
                name: {"id": domain.domain_id, "stats_sha256": domain.stats_sha256}
                for name, domain in self.config.domains.items()
            },
            "policies": ["replay", "manual"] + (["external"] if self.config.policy_url else []),
            "evaluation_available": self._evaluation_available(),
            "vlm_error": self.vlm_error,
            "vlm_active_model_key": getattr(self.vlm_registry, "active_key", None),
            "state_prediction": False,
            "physics_ground_truth": False,
            "replay_policy_is_observation_conditioned": False,
        }

    def _expire(self):
        now = time.time()
        expired = [
            key
            for key, session in self.sessions.items()
            if session.active_job is None and now - session.touched > self.config.idle_ttl_seconds
        ]
        for key in expired:
            self._remove_session(key)

    def _remove_session(self, session_id: str):
        session = self.sessions.pop(session_id)
        shutil.rmtree(session.directory)
        self._forget_session_requests(session_id)

    def _forget_session_requests(self, session_id: str):
        for key in list(self.requests):
            if key[0] == session_id:
                _, job_id = self.requests.pop(key)
                self.jobs.pop(job_id, None)

    def _load_persisted_session(self, session_id: str):
        try:
            if uuid.UUID(hex=session_id).hex != session_id:
                raise ValueError
        except ValueError:
            raise KeyError(session_id) from None
        directory = self.config.output_dir / session_id
        metadata_path = directory / "session.json"
        if directory.is_symlink() or not directory.is_dir():
            raise KeyError(session_id)
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise KeyError(session_id)
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise KeyError(session_id) from None
        if not isinstance(metadata, dict):
            raise KeyError(session_id)
        if metadata.get("protocol") != "dream.v1" or metadata.get("session_id") != session_id:
            raise KeyError(session_id)
        return directory, metadata

    def list_sessions(self):
        with self.lock:
            sessions = []
            for session_id, session in self.sessions.items():
                sessions.append(
                    {
                        "id": session_id,
                        "example_id": session.example.example_id if session.example else None,
                        "example_title": session.example.title if session.example else None,
                        "domain": session.domain,
                        "policy": session.policy,
                        "next_step": session.next_step,
                        "total_frames": len(session.history),
                        "active_job": session.active_job is not None,
                        "finished": session.finished,
                        "archived": False,
                        "orphaned": False,
                        "touched_at": session.touched,
                        "video_url": self.artifact_url(session, session.directory / "vision.mp4")
                        if (session.directory / "vision.mp4").is_file()
                        else None,
                        "step_videos": [step["video_url"] for step in session.steps],
                    }
                )
            for directory in self.config.output_dir.iterdir():
                if (
                    directory.name in self.sessions
                    or directory.is_symlink()
                    or not directory.is_dir()
                ):
                    continue
                try:
                    directory, metadata = self._load_persisted_session(directory.name)
                except KeyError:
                    continue
                metadata_path = directory / "session.json"
                saved_steps = metadata.get("steps", [])
                if not isinstance(saved_steps, list):
                    saved_steps = []
                step_videos = [
                    step["video_url"]
                    for step in saved_steps
                    if isinstance(step, dict) and isinstance(step.get("video_url"), str)
                ]
                example_id = metadata.get("example_id")
                example_title = metadata.get("example_title")
                sessions.append(
                    {
                        "id": directory.name,
                        "example_id": example_id if isinstance(example_id, str) else None,
                        "example_title": example_title
                        if isinstance(example_title, str)
                        else None,
                        "domain": metadata.get("domain", "unknown"),
                        "policy": metadata.get("policy", "unknown"),
                        "next_step": metadata.get("next_step", 0),
                        "total_frames": None,
                        "active_job": False,
                        "finished": bool(metadata.get("finished", False)),
                        "archived": bool(metadata.get("archived", False)),
                        "orphaned": not bool(metadata.get("archived", False)),
                        "touched_at": metadata_path.stat().st_mtime,
                        "video_url": f"/artifacts/{directory.name}/vision.mp4"
                        if (directory / "vision.mp4").is_file()
                        else None,
                        "step_videos": step_videos,
                    }
                )
            sessions.sort(key=lambda entry: entry["touched_at"], reverse=True)
            return {"sessions": sessions}

    def create_session(self, request: SessionRequest):
        with self.lock:
            if self.closed:
                raise Busy("Platform is shutting down")
            if self.switching_model:
                raise Busy("Model is switching; wait until it is ready before creating a session")
            if not self.ready and self.switch_state in {"unselected", "failed"}:
                raise Busy("Select and load a model before creating a session")
            self._expire()
            if len(self.sessions) >= self.config.max_sessions:
                raise Busy("Session limit reached; close or finish an existing session")
            if request.policy == "external" and not self.config.policy_url:
                raise ValueError("External policy is not configured")
            example = None
            domain = request.domain
            if request.example_id:
                example = self.examples.get(request.example_id)
                if example is None:
                    raise ValueError("Unknown example_id")
                if example.external_only and request.policy != "external":
                    raise ValueError("This example is a pi05 seed; choose external policy")
                domain = example.domain
                image = example.conditioning_image()
                if request.policy == "replay":
                    expected = int(example.chunk_length)
                    if request.chunk_length != expected:
                        raise ValueError(
                            f"replay policy replays the example's {expected}-row actions; "
                            f"choose chunk_length {expected} for this example"
                        )
            elif request.policy == "replay":
                raise ValueError("replay policy requires an example_id")
            else:
                try:
                    image_bytes = base64.b64decode(request.initial_image_png_b64, validate=True)
                except (ValueError, TypeError) as error:
                    raise ValueError("Invalid base64 image") from error
                with Image.open(io.BytesIO(image_bytes)) as source:
                    image = source.convert("RGB")
            supported_domains = self._supported_domains()
            if supported_domains is not None and domain not in supported_domains:
                raise ValueError("Domain is not supported by the active model")
            if domain not in self.config.domains and self.config.backend != "mock":
                raise ValueError("Domain is not available")
            if self.config.backend == "recorded" and (
                example is None or request.policy != "replay"
            ):
                raise ValueError("Recorded backend is only a recorded-action replay, not fresh FD")
            if self.config.backend == "recorded" and request.caption not in {None, example.caption}:
                raise ValueError("Recorded backend cannot change the original prompt")
            if self.config.backend == "recorded" and not all(
                path.is_file() for path in example.outputs
            ):
                raise ValueError("This example has missing recorded outputs; use cosmos backend")
            if image.width * image.height > self.config.max_image_pixels:
                raise ValueError("Initial image exceeds pixel limit")
            session_id = uuid.uuid4().hex
            directory = self.config.output_dir / session_id
            directory.mkdir()
            first = directory / "initial.png"
            try:
                image.save(first)
            except Exception:
                shutil.rmtree(directory)
                raise
            initial_state = request.initial_state
            state_source = request.state_source
            if (
                initial_state is None
                and example is not None
                and example.manifest.is_file()
                and request.policy == "external"
            ):
                metadata = json.loads(example.manifest.read_text())
                from .examples import _resolve_seed_fields

                initial_state, _ = _resolve_seed_fields(metadata)
                if initial_state is not None:
                    state_source = metadata.get("state_source", "commanded")
            session = Session(
                session_id,
                domain,
                request.caption
                if request.caption is not None
                else (example.caption if example else ""),
                request.policy,
                directory,
                first,
                example,
                initial_state,
                state_source,
                initial_image=first,
                num_steps=request.num_steps
                if request.num_steps is not None
                else self.config.num_steps,
                seed=request.seed,
                chunk_length=request.chunk_length,
            )
            try:
                self._save_session(session)
            except Exception:
                shutil.rmtree(directory)
                raise
            self.sessions[session_id] = session
            return self.session_info(session_id)

    def _save_session(self, session: Session, *, archived: bool = False):
        save_json(
            session.directory / "session.json",
            {
                "protocol": "dream.v1",
                "model_id": self.config.model_id,
                "backend": self.config.backend,
                "session_id": session.session_id,
                "example_id": session.example.example_id if session.example else None,
                "example_title": session.example.title if session.example else None,
                "domain": session.domain,
                "caption": session.caption,
                "policy": session.policy,
                "num_steps": session.num_steps,
                "seed": session.seed,
                "chunk_length": session.chunk_length,
                "fps": 10,
                "history_bucket_frames": [1, 1, 1, 1],
                "steps": session.steps,
                "next_step": session.next_step,
                "state": session.state,
                "state_source": session.state_source,
                "evaluations": session.evaluations,
                "finished": session.finished,
                "archived": archived,
                "output_geometry": (
                    {"width": session.output_geometry[0], "height": session.output_geometry[1]}
                    if session.output_geometry
                    else None
                ),
                "video_metadata": session.video_metadata,
                "stats_sha256": self.config.domains[session.domain].stats_sha256
                if session.domain in self.config.domains
                else None,
                "actions": session.actions_meta,
            },
        )

    def session_info(self, session_id: str):
        with self.lock:
            session = self.sessions[session_id]
            session.touched = time.time()
            return {
                "id": session_id,
                "example_id": session.example.example_id if session.example else None,
                "example_title": session.example.title if session.example else None,
                "domain": session.domain,
                "caption": session.caption,
                "policy": session.policy,
                "num_steps": session.num_steps,
                "seed": session.seed,
                "chunk_length": session.chunk_length,
                "backend": self.config.backend,
                "next_step": session.next_step,
                "total_frames": len(session.history),
                "active_job": session.active_job,
                "finished": session.finished,
                "observation_url": self.artifact_url(session, session.current_image),
                "state": session.state,
                "state_source": session.state_source,
                "steps": list(session.steps),
                "evaluations": list(session.evaluations),
                "max_replay_chunks": (
                    session.example.chunk_count
                    if session.example and session.policy == "replay"
                    else None
                ),
                "video_url": self.artifact_url(session, session.directory / "vision.mp4")
                if (session.directory / "vision.mp4").is_file()
                else None,
                "video_metadata": session.video_metadata,
                "evaluation_available": self._evaluation_available(),
                "vlm_error": self.vlm_error,
                "vlm_active_model_key": getattr(self.vlm_registry, "active_key", None),
            }

    @staticmethod
    def artifact_url(session: Session, path: Path):
        return f"/artifacts/{session.session_id}/{path.relative_to(session.directory).as_posix()}"

    def observation(self, session: Session):
        return {
            "image_png_b64": image_base64(session.current_image),
            "layout": "concat_view",
            "domain": session.domain,
            "prompt": session.caption,
            "fps": 10,
            "chunk_length": session.chunk_length,
            "state": session.state,
            "state_source": session.state_source,
            "state_is_predicted_by_fd": False,
        }

    def submit(
        self, session_id: str, request: StepRequest | RunRequest | EvaluationRequest, kind="step"
    ):
        with self.lock:
            session = self.sessions[session_id]
            fingerprint = hashlib.sha256((kind + request.model_dump_json()).encode()).hexdigest()
            key = (session_id, request.request_id)
            if key in self.requests:
                old_hash, job_id = self.requests[key]
                if old_hash != fingerprint:
                    raise Conflict("request_id reused with different payload")
                return dict(self.jobs[job_id])
            if self.startup_error:
                raise Busy(self.startup_error)
            if self.closed:
                raise Busy("Platform is shutting down")
            if not self.ready:
                raise Busy("Model is still initializing; wait for /health/ready")
            if session.active_job is not None or (session.finished and kind != "evaluate"):
                raise Conflict("Session is busy or finished")
            if kind != "evaluate":
                if request.expected_step != session.next_step:
                    raise Conflict("expected_step does not match session")
                count = request.chunks if isinstance(request, RunRequest) else 1
                if session.next_step + count > self.config.max_chunks:
                    raise ValueError("Maximum rollout length exceeded")
                if request.chunk is not None and len(request.chunk.action) != session.chunk_length:
                    raise ValueError(f"Session requires {session.chunk_length} action rows")
                if request.chunk is None and session.policy == "manual":
                    raise ValueError("Manual policy requires chunk")
                if count > 1 and request.chunk is not None:
                    raise ValueError("Explicit chunk is only supported for a single step")
                if session.policy == "replay" and request.chunk is None:
                    remaining = session.example.chunk_count - session.next_step
                    if count > remaining:
                        raise ValueError(
                            f"Replay actions exhausted: requested {count}, only {remaining} remain"
                        )
            else:
                if not self._evaluation_available():
                    detail = self.vlm_error or "configure vlm_models_config or evaluator_url"
                    raise ValueError(f"VLM evaluator is not available: {detail}")
                instruction = self._resolve_evaluation_instruction(session, request)
                video_path = session.directory / "vision.mp4"
                if not session.finished:
                    raise Conflict("Finish the session before running VLM evaluation")
                if not video_path.is_file():
                    raise ValueError("Finished session is missing root vision.mp4")
                if not session.video_metadata:
                    raise ValueError("Finished session is missing video_metadata")
                if not instruction:
                    raise ValueError(
                        "Evaluation requires a non-empty instruction or session caption"
                    )
            if self.tasks.full():
                raise Busy("GPU queue is full")
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "session_id": session_id,
                "kind": kind,
                "status": "queued",
                "completed_chunks": 0,
                "error": None,
            }
            self.jobs[job_id] = job
            self.requests[key] = (fingerprint, job_id)
            session.active_job = job_id
            session.stop = False
            session.touched = time.time()
            future = Future()
            self.tasks.put_nowait((lambda: self._execute(session, job, request, kind), future))
            return dict(job)

    def _execute(self, session, job, request, kind):
        job["status"] = "running"
        try:
            if kind == "evaluate":
                evidence = self._build_video_evaluation_evidence(session, request, job["id"])
                if self.config.evaluator_url:
                    result = HTTPEvaluator(self.config).evaluate(evidence)
                elif self.vlm_evaluator is not None:
                    result = self.vlm_evaluator.evaluate(evidence)
                else:
                    raise ValueError(
                        f"VLM evaluator is not available: {self.vlm_error or 'not initialized'}"
                    )
                summary = result.model_dump()
                artifact_path = session.directory / "evaluations" / f"{job['id']}.json"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                save_json(artifact_path, {"evidence": evidence, "result": summary})
                summary_for_session = {
                    **summary,
                    "details": {
                        **(summary.get("details") or {}),
                        "result_url": self.artifact_url(session, artifact_path),
                        "judge_result": None,
                    },
                }
                with self.lock:
                    session.evaluations.append(summary_for_session)
                    try:
                        self._save_session(session)
                    except Exception:
                        if session.evaluations and session.evaluations[-1] is summary_for_session:
                            session.evaluations.pop()
                        raise
                    job["result"] = summary_for_session
            else:
                count = request.chunks if isinstance(request, RunRequest) else 1
                for _ in range(count):
                    if session.stop:
                        break
                    chunk = request.chunk
                    if chunk is None:
                        if session.policy == "replay":
                            policy = ReplayPolicy(session.example)
                        else:
                            policy = HTTPPolicy(self.config)
                        chunk = policy.predict(self.observation(session), session.next_step)
                    self._step(session, chunk)
                    job["completed_chunks"] += 1
            job["status"] = "stopped" if session.stop else "completed"
        except (Exception, SystemExit):
            logger.exception("Dream job %s failed", job["id"])
            job["status"] = "failed"
            job["error"] = "Job failed; inspect server logs. Session retains last committed step."
        finally:
            with self.lock:
                session.active_job = None
                session.touched = time.time()

    def _step(self, session: Session, chunk):
        if len(chunk.action) != session.chunk_length:
            raise ValueError(
                f"Session requires {session.chunk_length} action rows, got {len(chunk.action)}"
            )
        started = time.monotonic()
        step = session.next_step
        temporary = session.directory / f".step_{step:04d}_{uuid.uuid4().hex}"
        final = session.directory / f"step_{step:04d}"
        temporary.mkdir()
        try:
            frames = self.runtime.generate(
                image=session.current_image,
                history=session.history,
                chunk=chunk,
                domain=session.domain,
                caption=session.caption,
                step=step,
                workdir=temporary,
                example=session.example,
                num_steps=session.num_steps,
                seed=session.seed,
            )
            expected_frames = session.chunk_length + 1
            if frames.ndim != 4 or frames.shape[0] != expected_frames or frames.shape[-1] != 3:
                raise ValueError(f"FD output must have shape [{session.chunk_length + 1},H,W,3]")
            if frames.dtype != np.uint8:
                raise ValueError("FD runtime must return uint8 RGB frames")
            if frames.shape[1] < 1 or frames.shape[2] < 1:
                raise ValueError("FD output canvas is empty")
            if self.config.backend == "cosmos":
                geometry = validate_raw_output_geometry(frames)
            else:
                geometry = {
                    "raw_width": int(frames.shape[2]),
                    "raw_height": int(frames.shape[1]),
                    "geometry_kind": "passthrough",
                    "layout": CONCAT_LAYOUT,
                }
            with self.lock:
                session.output_geometry = ensure_session_geometry_consistency(
                    session.output_geometry, geometry["raw_width"], geometry["raw_height"]
                )
            new_frames = frames if step == 0 else frames[1:]
            frame_dir = temporary / "frames"
            frame_dir.mkdir()
            for index, frame in enumerate(new_frames):
                Image.fromarray(frame).save(frame_dir / f"{index:03d}.png")
            Image.fromarray(frames[-1]).save(temporary / "observation.png")
            imageio.mimsave(
                temporary / "chunk.mp4",
                new_frames,
                fps=10,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
            )
            save_json(temporary / "action.json", chunk.model_dump())
            save_json(
                temporary / "provenance.json",
                {
                    "model_id": self.config.model_id,
                    "backend": self.config.backend,
                    "domain": session.domain,
                    "step_index": step,
                    "num_steps": session.num_steps,
                    "seed": session.seed,
                    "history_indices": history_indices(len(session.history)),
                    "conditioning_sha256": hashlib.sha256(
                        session.current_image.read_bytes()
                    ).hexdigest(),
                    "action_sha256": hashlib.sha256(chunk.model_dump_json().encode()).hexdigest(),
                    "stats_sha256": self.config.domains[session.domain].stats_sha256
                    if session.domain in self.config.domains
                    else None,
                    "output_geometry": geometry,
                },
            )
            temporary.rename(final)
            with self.lock:
                prior = (
                    session.next_step,
                    session.current_image,
                    session.state,
                    session.state_source,
                    session.output_geometry,
                )
                old_history_length, old_steps_length = len(session.history), len(session.steps)
                session.history.extend(sorted((final / "frames").glob("*.png")))
                session.current_image = final / "observation.png"
                session.next_step += 1
                if chunk.state is not None:
                    session.state, session.state_source = chunk.state, chunk.state_source
                else:
                    session.state, session.state_source = None, "unavailable"
                session.steps.append(
                    {
                        "step_index": step,
                        "new_frames": len(new_frames),
                        "video_url": self.artifact_url(session, final / "chunk.mp4"),
                        "observation_url": self.artifact_url(session, session.current_image),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "output_geometry": geometry,
                    }
                )
                try:
                    self._save_session(session)
                except Exception:
                    (
                        session.next_step,
                        session.current_image,
                        session.state,
                        session.state_source,
                        session.output_geometry,
                    ) = prior
                    del session.history[old_history_length:]
                    del session.steps[old_steps_length:]
                    shutil.rmtree(final)
                    raise
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def stop_session(self, session_id):
        with self.lock:
            self.sessions[session_id].stop = True
            return self.session_info(session_id)

    def archive_session(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                directory, metadata = self._load_persisted_session(session_id)
                metadata["archived"] = True
                save_json(directory / "session.json", metadata)
                return {"archived": session_id}
            if session.active_job:
                raise Conflict("Stop and wait for the active chunk before archiving")
            self._save_session(session, archived=True)
            self.sessions.pop(session_id)
            self._forget_session_requests(session_id)
            return {"archived": session_id}

    def delete_session(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                directory, _ = self._load_persisted_session(session_id)
                shutil.rmtree(directory)
                return {"deleted": session_id}
            if session.active_job:
                raise Conflict("Stop and wait for the active chunk before deleting")
            self._remove_session(session_id)
            return {"deleted": session_id}

    @staticmethod
    def _resolve_evaluation_instruction(session: Session, request: EvaluationRequest) -> str:
        explicit = (request.instruction or "").strip()
        if explicit:
            return explicit
        return (session.caption or "").strip()

    def _build_video_evaluation_evidence(
        self, session: Session, request: EvaluationRequest, evaluation_id: str
    ) -> dict:
        instruction = self._resolve_evaluation_instruction(session, request)
        video_path = (session.directory / "vision.mp4").resolve()
        metadata = dict(session.video_metadata or {})
        actual_sha = hashlib.sha256(video_path.read_bytes()).hexdigest()
        if metadata.get("sha256") and metadata["sha256"] != actual_sha:
            raise ValueError("vision.mp4 hash does not match saved video_metadata")
        if metadata.get("frame_count") != len(session.history):
            raise ValueError("vision.mp4 frame_count does not match session history")
        return {
            "evaluation_id": evaluation_id,
            "session_id": session.session_id,
            "instruction": instruction,
            "video": {
                "path": str(video_path),
                "sha256": actual_sha,
                "width": int(metadata.get("width") or CONCAT_SIZE[0]),
                "height": int(metadata.get("height") or CONCAT_SIZE[1]),
                "fps": int(metadata.get("fps") or 10),
                "frame_count": int(metadata.get("frame_count") or len(session.history)),
                "layout": metadata.get("layout") or CONCAT_LAYOUT,
            },
            "steps": session.next_step,
            "domain": session.domain,
            "backend": self.config.backend,
            "fd_model_id": self.config.model_id,
            "physics_ground_truth": False,
            "question": request.question,
        }

    def finish(self, session_id):
        with self.lock:
            session = self.sessions[session_id]
            if session.active_job:
                raise Conflict("Session has an active job")
            if not session.history:
                raise ValueError("No generated frames")
            video_path = session.directory / "vision.mp4"
            if (
                session.finished
                and video_path.is_file()
                and session.video_metadata
                and session.video_metadata.get("frame_count") == len(session.history)
                and session.video_metadata.get("sha256")
                == hashlib.sha256(video_path.read_bytes()).hexdigest()
            ):
                return self.session_info(session_id)

            temporary = session.directory / "vision.tmp.mp4"
            if temporary.exists():
                temporary.unlink()
            transform_summary = None
            require_canonical = self.config.backend == "cosmos"
            with imageio.get_writer(
                temporary,
                fps=10,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
            ) as writer:
                for path in session.history:
                    image = Image.open(path).convert("RGB")
                    if image.size in {CONCAT_SIZE, RAW_FD_SIZE}:
                        frame, transform_summary = to_canonical_frame(image)
                    elif require_canonical:
                        raise ValueError(
                            f"unsupported_output_geometry: cosmos finish got {image.size}, "
                            f"expected {CONCAT_SIZE} or {RAW_FD_SIZE}"
                        )
                    else:
                        frame = np.array(image)
                        transform_summary = {"geometry_kind": "passthrough"}
                    writer.append_data(frame)
            # Probe by re-reading encoded frames via imageio.
            encoded = imageio.mimread(temporary, memtest=False)
            if len(encoded) != len(session.history):
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"Finished video frame count {len(encoded)} != history {len(session.history)}"
                )
            height, width = encoded[0].shape[:2]
            if require_canonical and (width, height) != CONCAT_SIZE:
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"Finished video must be {CONCAT_SIZE[0]}x{CONCAT_SIZE[1]}, got {width}x{height}"
                )
            sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
            metadata = {
                "path": "vision.mp4",
                "sha256": sha256,
                "width": width,
                "height": height,
                "fps": 10,
                "frame_count": len(encoded),
                "duration_seconds": round(len(encoded) / 10.0, 3),
                "layout": CONCAT_LAYOUT,
                "geometry": transform_summary,
            }
            temporary.replace(video_path)
            previous_finished = session.finished
            previous_metadata = session.video_metadata
            previous_actions_meta = session.actions_meta
            session.finished = True
            session.video_metadata = metadata
            if session.policy == "external" and session.next_step > 0:
                try:
                    actions = assemble_normalized_actions(session.directory)
                    stats_sha = (
                        self.config.domains[session.domain].stats_sha256
                        if session.domain in self.config.domains
                        else None
                    )
                    representation = None
                    first_action = session.directory / "step_0000" / "action.json"
                    if first_action.is_file():
                        try:
                            representation = json.loads(first_action.read_text()).get(
                                "representation"
                            )
                        except (OSError, json.JSONDecodeError, TypeError):
                            representation = None
                    session.actions_meta = write_session_actions(
                        session.directory,
                        actions,
                        stats_sha256=stats_sha,
                        representation=representation
                        if isinstance(representation, str)
                        else None,
                    )
                except Exception:
                    logger.exception(
                        "failed to write session actions.npy for session %s",
                        session.session_id,
                    )
            try:
                self._save_session(session)
            except Exception:
                session.finished = previous_finished
                session.video_metadata = previous_metadata
                session.actions_meta = previous_actions_meta
                raise
            return self.session_info(session_id)

    def artifact(self, session_id: str, relative_path: str):
        with self.lock:
            session = self.sessions.get(session_id)
            directory = (
                session.directory
                if session is not None
                else self._load_persisted_session(session_id)[0]
            )
            path = (directory / relative_path).resolve()
            if not path.is_relative_to(directory.resolve()) or any(
                part.startswith(".") for part in Path(relative_path).parts
            ):
                raise ValueError("Invalid artifact path")
            if not path.is_file():
                raise KeyError("Artifact not found")
            return path
