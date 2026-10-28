# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Per-tick chef task progress on the existing lifecycle ROS node.

This is a task observer, never a physics/controller owner. Scoring failures
become sticky terminal progress instead of escaping into the core run loop's
step-error retry path. It owns only its publisher, not a ROS context/executor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
import math
import sys
import time
from typing import Any
import uuid


@dataclass(frozen=True)
class WokEvaluationRosConfig:
    topic: str
    thresholds: Any

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> WokEvaluationRosConfig | None:
        """Read the launcher-only JSON parameter; absent/empty disables scoring."""
        raw = params.get("wok_evaluation_json", "")
        if raw is None or raw == "":
            return None
        if not isinstance(raw, str):
            raise ValueError("wok_evaluation_json must be a JSON string")
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) - {"enabled", "topic", "thresholds"}:
            raise ValueError("wok_evaluation_json accepts enabled, topic and thresholds")
        enabled = data.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("wok evaluation enabled must be a boolean")
        if not enabled:
            return None
        topic = data.get("topic", "/wuji_eval/task_progress")
        if not isinstance(topic, str) or not topic.startswith("/") or topic == "/":
            raise ValueError("wok evaluation topic must be an absolute ROS topic")
        thresholds = data.get("thresholds", {})
        if not isinstance(thresholds, dict):
            raise ValueError("wok evaluation thresholds must be an object")
        from .wok_evaluation import WokEvaluationConfig

        return cls(topic, WokEvaluationConfig.from_mapping(thresholds))


def _json_default(value: Any) -> Any:
    # Sampler metrics can contain NumPy scalars/arrays; no GPU object crosses
    # this boundary. Normalization also makes allow_nan=False effective.
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"task progress contains unsupported {type(value).__name__}")


class WokEvaluationProducer:
    """Publish a versioned, sticky task status for every completed physics tick.

    ``after_step`` attaches to session.post_step_hooks. ``refresh`` follows
    paused image publication, reevaluates at dt=0 and never increments tick.
    The optional sampler/evaluator injection is for CPU tests only.
    """

    def __init__(
        self,
        node: Any,
        sim: Any,
        config: WokEvaluationRosConfig,
        *,
        dt: float,
        session_id: str | None = None,
        sampler: Any = None,
        evaluator: Any = None,
    ) -> None:
        if isinstance(dt, bool) or not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("wok evaluation dt must be finite and positive")
        self._node = node
        self._sim = sim
        self._dt = float(dt)
        self._session_id = session_id or str(uuid.uuid4())
        self._epoch = int(sim.reset_epoch)
        self._tick = 0
        self._sim_time = float(sim._sim_time_acc)
        self._published_ns = 0
        self._error: str | None = None
        self._closed = False
        self._sampler = sampler
        self._evaluator = evaluator
        thresholds = asdict(config.thresholds) if is_dataclass(config.thresholds) else dict(config.thresholds)
        self._status = dict(
            score=0,
            max_score=2,
            milestones=dict(stable_lift=False, food_toss=False),
            failed=False,
            failure_reason=None,
            metrics={},
            thresholds=thresholds,
        )
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self._message_type = String
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._publisher = node.create_publisher(String, config.topic, qos)
        # The caller registers close immediately after construction. Catch pure
        # sampler construction failures as progress, keeping this publisher live.
        try:
            if self._sampler is None:
                from .wok_evaluation import WokEvaluationSampler

                self._sampler = WokEvaluationSampler(sim)
            if self._evaluator is None:
                from .wok_evaluation import WokEvaluator

                self._evaluator = WokEvaluator(self._sampler.geometry, config.thresholds)
        except Exception as exc:
            self._latch_error(exc)

    def _latch_error(self, error: Exception) -> None:
        if self._error is None:
            self._error = f"{type(error).__name__}: {error}"
            print(f"[wok-evaluation] terminal observer error: {self._error}", file=sys.stderr, flush=True)
        self._status = {**self._status, "failed": True, "failure_reason": "evaluation_error"}

    def _payload(self) -> dict[str, Any]:
        self._published_ns = max(time.time_ns(), self._published_ns + 1)
        result = {
            **self._status,
            "version": 1,
            "task_name": "wok_flip",
            "session_id": self._session_id,
            "reset_epoch": self._epoch,
            "tick": self._tick,
            "sim_time": self._sim_time,
            "published_ns": self._published_ns,
            # Two earned milestones do not stop the action budget: a later
            # dropped wok must still terminate the episode as a failure.
            "terminal": bool(self._status["failed"]),
        }
        if self._error is not None:
            result["error"] = self._error
        return result

    def _publish(self) -> None:
        try:
            message = self._message_type()
            message.data = json.dumps(self._payload(), allow_nan=False, default=_json_default)
            self._publisher.publish(message)
        except Exception as exc:
            self._latch_error(exc)
            try:
                message = self._message_type()
                message.data = json.dumps(self._payload(), allow_nan=False, default=_json_default)
                self._publisher.publish(message)
            except Exception as publish_error:
                # A broken transport cannot carry its own failure. Keep it
                # visible in logs; the consumer's bounded progress wait fails.
                print(f"[wok-evaluation] progress publication failed: {publish_error}", file=sys.stderr, flush=True)

    def _advance(self, *, stepped: bool) -> None:
        if self._closed:
            return
        try:
            # Identity keeps tracking the engine after a scoring failure. A
            # reset must expose that same sticky observer error in the new
            # epoch, rather than leaving the consumer waiting on an old epoch.
            epoch, sim_time = int(self._sim.reset_epoch), float(self._sim._sim_time_acc)
            if epoch < 0 or not math.isfinite(sim_time) or sim_time < 0:
                raise ValueError("task sample requires a nonnegative epoch and finite simulation time")
            if epoch != self._epoch:
                self._epoch = epoch
                self._tick = 0
            self._sim_time = sim_time
            if stepped:
                self._tick += 1
        except Exception as exc:
            self._latch_error(exc)
        if self._error is None:
            try:
                sample = self._sampler.sample()
                if int(sample.reset_epoch) != self._epoch or not math.isclose(float(sample.sim_time), self._sim_time):
                    raise ValueError("task sample identity does not match the completed engine state")
                status = self._evaluator.update(sample, dt=self._dt if stepped else 0.0)
                # Validate before replacing the last known-good earned score.
                json.dumps(status, allow_nan=False, default=_json_default)
                self._status = status
            except Exception as exc:
                self._latch_error(exc)
        self._publish()

    def after_step(self, _loop_sim_time: float) -> None:
        """Score a completed engine tick; the loop argument precedes its dt increment."""
        self._advance(stepped=True)

    def refresh(self) -> None:
        """Publish after paused RGB; check resets/drop without adding time or ticks."""
        self._advance(stepped=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        publisher, self._publisher = self._publisher, None
        if publisher is not None:
            self._node.destroy_publisher(publisher)
