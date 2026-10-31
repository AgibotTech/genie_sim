# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Translate simulator messages at the service boundary, outside application logic."""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import uuid4

from agent2act.adapters.simulators.codec import SimulatorCodec
from agent2act.adapters.simulators.corobot import CoRobotCodec
from agent2act.adapters.telemetry import capture_artifacts, record_artifact
from agent2act.application.runtime import AgentRuntime
from agent2act.domain.errors import ModelIdentityError

from .action_trace import action_log_path
from .observability import log_context, timed_event
from .trace import RequestTrace

logger = logging.getLogger(__name__)


class AgentSession:
    def __init__(
        self,
        runtime: AgentRuntime,
        session_id: str,
        trace_dir: str | None = None,
        codec: SimulatorCodec | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.trace_dir = Path(trace_dir).resolve() if trace_dir else None
        self.runtime = runtime
        self.session_id = session_id
        self.codec = codec if codec is not None else CoRobotCodec(runtime.spec, {})
        self._closed = False
        self._identity_aborted_episode: tuple[str, int, str] | None = None
        if self.trace_dir is not None and run_metadata is not None:
            folder = self.trace_dir / self.session_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "hardware.json").write_text(
                json.dumps(run_metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("session is closed")
        inference_id = uuid4().hex
        trace = (
            RequestTrace(
                self.trace_dir / self.session_id / inference_id,
                action_log=action_log_path(self.trace_dir),
            )
            if self.trace_dir
            else None
        )
        with log_context(session_id=self.session_id, inference_id=inference_id), capture_artifacts(trace):
            try:
                return self._handle(request)
            except Exception as exc:
                record_artifact("request-error", {"error_type": type(exc).__name__})
                raise

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        with timed_event(logger, "inference_request"):
            record_artifact("simulator-request", request)
            with timed_event(logger, "observation_decode"):
                observation = self.codec.decode(request)
            record_artifact(
                "observer-observation",
                {
                    "end_pose": observation.end_pose,
                    "images": observation.images,
                },
            )
            with timed_event(logger, "runtime"):
                episode = (*observation.episode_key, observation.instruction)
                if self._identity_aborted_episode == episode and not observation.done:
                    raise ModelIdentityError("episode interrupted by model identity failure; restart episode")
                if self._identity_aborted_episode != episode or observation.done:
                    self._identity_aborted_episode = None
                try:
                    chunk = self.runtime.step(observation)
                except ModelIdentityError:
                    self._identity_aborted_episode = episode
                    event = {
                        "episode": episode,
                        "sequence": self.runtime.sequence,
                        "status": "episode_interrupted",
                        "error_code": "model_identity_exhausted",
                        "task_success": "unverified",
                        "action_issued": False,
                    }
                    record_artifact("episode-interrupted", event)
                    logger.warning("episode_interrupted", extra={"event": event})
                    raise
            path = chunk.metadata.get("cartesian_path", {})
            if not isinstance(path, dict):
                path = {}
            record_artifact(
                "observer-action",
                {
                    "sequence": chunk.metadata.get("sequence", self.runtime.sequence),
                    "frame": chunk.frame,
                    "kind": chunk.kind,
                    "steps": chunk.steps,
                    "endpoint": {"left": chunk.left_arm[-1], "right": chunk.right_arm[-1]},
                    "endpoint_fk": {
                        arm: poses[chunk.steps - 1]
                        for arm, poses in path.items()
                        if isinstance(poses, list) and len(poses) >= chunk.steps
                    },
                },
            )
            chunk.metadata["session_id"] = self.session_id
            with timed_event(logger, "response_encode"):
                response = self.codec.encode(
                    chunk,
                    observation,
                    history=self.runtime.config.history,
                    need_depth=self.runtime.config.need_depth,
                )
            record_artifact("simulator-response", response)
            return response

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            with ExitStack() as cleanup:
                cleanup.callback(self.codec.close)
                cleanup.callback(self.runtime.close)

    def __enter__(self) -> AgentSession:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()
