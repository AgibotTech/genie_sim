from __future__ import annotations

import base64
import importlib
import json
import os
from pathlib import Path
from typing import Protocol

import httpx

from .config import DreamConfig
from .contracts import ActionChunk, EvaluationResult
from .examples import Example


class PolicyAdapter(Protocol):
    def predict(self, observation: dict, step: int) -> ActionChunk: ...


class Evaluator(Protocol):
    def evaluate(self, evidence: dict) -> EvaluationResult: ...


def image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


class ReplayPolicy:
    def __init__(self, example: Example):
        self.example = example

    def predict(self, observation: dict, step: int) -> ActionChunk:
        return self.example.action(step)


class HTTPPolicy:
    def __init__(self, config: DreamConfig):
        if not config.policy_url:
            raise ValueError("No policy_url configured")
        self.config = config

    def predict(self, observation: dict, step: int) -> ActionChunk:
        token = os.environ.get(self.config.policy_token_env)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = httpx.post(
            self.config.policy_url,
            json={"protocol": "dream.policy.v1", "step_index": step, "observation": observation},
            headers=headers,
            timeout=self.config.remote_timeout_seconds,
        )
        response.raise_for_status()
        return ActionChunk.model_validate(response.json())


class HTTPEvaluator:
    def __init__(self, config: DreamConfig):
        if not config.evaluator_url:
            raise ValueError("No evaluator_url configured")
        self.config = config

    def evaluate(self, evidence: dict) -> EvaluationResult:
        token = os.environ.get(self.config.evaluator_token_env)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        protocol = (
            "dream.evaluation.video.v1"
            if isinstance(evidence.get("video"), dict)
            else "dream.evaluation.v1"
        )
        timeout = (
            self.config.evaluator_timeout_seconds
            if protocol == "dream.evaluation.video.v1"
            else self.config.remote_timeout_seconds
        )
        try:
            response = httpx.post(
                self.config.evaluator_url,
                json={"protocol": protocol, "evidence": evidence},
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as error:
            raise TimeoutError(f"Evaluator timed out after {timeout}s") from error
        if response.status_code == 503:
            raise RuntimeError(f"evaluator_busy: {response.text[:500]}")
        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = {"error": {"message": response.text[:500]}}
            detail = payload.get("error") if isinstance(payload, dict) else None
            message = (
                detail.get("message")
                if isinstance(detail, dict)
                else response.text[:500]
            )
            raise RuntimeError(f"Evaluator HTTP {response.status_code}: {message}")
        payload = response.json()
        if isinstance(payload, dict):
            details = payload.get("details") or {}
            if details.get("success") is None and details.get("error"):
                raise RuntimeError(f"Evaluator error: {details.get('error')}")
        return EvaluationResult.model_validate(payload)


class LocalVideoEvaluator:
    """In-process geniesim_dream_evaluation judge with lazy per-task VLM weights."""

    def __init__(self, *, registry, shared_output_root: Path, judge_fn=None):
        from dream_adapter import DreamVideoEvaluator

        self._inner = DreamVideoEvaluator(
            shared_output_root=shared_output_root,
            registry=registry,
            judge_fn=judge_fn,
        )

    def evaluate(self, evidence: dict) -> EvaluationResult:
        body = {"protocol": "dream.evaluation.video.v1", "evidence": evidence}
        try:
            payload = self._inner.evaluate(body)
        except Exception as error:
            # dream_contracts.EvaluationError → plain RuntimeError for Dream job layer
            code = getattr(error, "code", None)
            message = getattr(error, "message", None) or str(error)
            if code:
                raise RuntimeError(f"local_evaluator_{code}: {message}") from error
            raise
        return EvaluationResult.model_validate(payload)


def load_adapter(spec: str, config: dict):
    module_name, factory_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), factory_name)(config)


def save_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
    temporary.replace(path)
