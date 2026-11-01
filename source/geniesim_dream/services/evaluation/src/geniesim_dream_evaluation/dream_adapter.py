"""Bridge Dream video evidence to geniesim_dream_evaluation judge_pair."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Callable

from geniesim_dream_evaluation.dream_contracts import (
    ADAPTER_VERSION,
    EvaluationError,
    validate_video_evidence,
)

def _load_converter_module():
    from . import converter

    return converter


def resolve_shared_video_path(path: str, shared_output_root: Path) -> Path:
    candidate = Path(path).resolve()
    root = shared_output_root.resolve()
    if not candidate.is_relative_to(root):
        raise EvaluationError(
            "path_escape",
            "video path escapes configured shared_output_root",
            http_status=400,
        )
    if not candidate.is_file():
        raise EvaluationError("missing_video", f"video not found: {candidate}", http_status=400)
    return candidate


def map_judge_result(
    judge_result: dict[str, Any],
    *,
    evaluation_id: str,
    video_sha256: str,
    vlm_model_id: str,
    vlm_model_key: str | None = None,
    task_family: str | None = None,
) -> dict[str, Any]:
    success = judge_result.get("success")
    error = judge_result.get("error")
    if success is None or error:
        raise EvaluationError(
            "judge_error",
            str(error or "judge returned success=null"),
            evaluation_id=evaluation_id,
            http_status=500,
        )

    completion_status = str(judge_result.get("completion_status") or "").lower()
    if completion_status == "uncertain" or (
        success == 0 and str(judge_result.get("resolver_reason") or "").endswith("uncertain")
    ):
        verdict = "uncertain"
        score = 0.0
    elif int(success) == 1:
        verdict = "success"
        score = 1.0
    else:
        verdict = "failure"
        score = 0.0

    assessment = (
        judge_result.get("thinking")
        or judge_result.get("resolver_reason")
        or judge_result.get("assessment")
        or verdict
    )
    details = {
        "evaluation_id": evaluation_id,
        "success": int(success),
        "error": None,
        "verdict": verdict,
        "task_family": task_family if task_family is not None else judge_result.get("task_family"),
        "judge_version": judge_result.get("judge_version"),
        "resolver_reason": judge_result.get("resolver_reason"),
        "video_sha256": video_sha256,
        "vlm_model_id": vlm_model_id,
        "adapter_version": ADAPTER_VERSION,
        "judge_result": judge_result,
    }
    if vlm_model_key is not None:
        details["vlm_model_key"] = vlm_model_key
    return {
        "evaluator": f"{vlm_model_id}:{ADAPTER_VERSION}",
        "assessment": str(assessment)[:4000],
        "score": score,
        "details": details,
        "evidence_kind": "visual_estimate",
    }


class DreamVideoEvaluator:
    def __init__(
        self,
        *,
        shared_output_root: Path,
        registry,
        judge_fn: Callable[..., dict[str, Any]] | None = None,
        enable_task_route: bool | None = None,
        generic_prompt_mode: str | None = None,
        # Legacy single-model kwargs kept for older call sites / tests.
        model=None,
        processor=None,
        vlm_model_id: str | None = None,
    ):
        from geniesim_dream_evaluation.vlm_registry import parse_judge_flags

        self.shared_output_root = Path(shared_output_root)
        self.registry = registry
        self.model = model
        self.processor = processor
        self.vlm_model_id = vlm_model_id or "unset"
        self._converter = _load_converter_module()
        if judge_fn is None:
            from geniesim_dream_evaluation.judge import judge_pair as default_judge

            judge_fn = default_judge
        self.judge_fn = judge_fn
        flags = (
            registry.judge_flags
            if registry is not None and hasattr(registry, "judge_flags")
            else parse_judge_flags({})
        )
        self.enable_task_route = (
            flags["enable_task_route"] if enable_task_route is None else enable_task_route
        )
        self.generic_prompt_mode = (
            flags["generic_prompt_mode"] if generic_prompt_mode is None else generic_prompt_mode
        )

    def evaluate(self, body: dict[str, Any]) -> dict[str, Any]:
        parsed = validate_video_evidence(body)
        evaluation_id = parsed["evaluation_id"]
        video_path = resolve_shared_video_path(parsed["video_path"], self.shared_output_root)
        actual_sha = hashlib.sha256(video_path.read_bytes()).hexdigest()
        if parsed["sha256"] and parsed["sha256"] != actual_sha:
            raise EvaluationError(
                "hash_mismatch",
                "video sha256 does not match evidence metadata",
                evaluation_id=evaluation_id or None,
            )
        instruction = parsed["instruction"]
        if self.registry is not None:
            family, bundle = self.registry.pick_for_instruction(instruction)
            model = bundle.model
            processor = bundle.processor
            vlm_model_id = bundle.vlm_model_id
            vlm_model_key = bundle.key
        else:
            family = None
            model = self.model
            processor = self.processor
            vlm_model_id = self.vlm_model_id
            vlm_model_key = None
        with tempfile.TemporaryDirectory(prefix="geniesim-dream-vlm-") as temporary:
            triplet = self._convert(video_path, Path(temporary))
            pair = {
                "episode": parsed["session_id"] or evaluation_id or video_path.stem,
                "wm_path": str(triplet),
                "instruction": instruction,
            }
            judge_result = self.judge_fn(
                model,
                processor,
                pair,
                enable_task_route=self.enable_task_route,
                generic_prompt_mode=self.generic_prompt_mode,
            )
            if not isinstance(judge_result, dict):
                raise EvaluationError(
                    "invalid_judge_result",
                    "judge_pair did not return a dict",
                    evaluation_id=evaluation_id or None,
                    http_status=500,
                )
        # The judge consumed the converted video, which is now gone. Keep only
        # the durable source path and conversion version in the saved report.
        persisted_judge_result = {
            key: value for key, value in judge_result.items() if key != "wm_path"
        }
        result = map_judge_result(
            persisted_judge_result,
            evaluation_id=evaluation_id,
            video_sha256=actual_sha,
            vlm_model_id=vlm_model_id,
            vlm_model_key=vlm_model_key,
            task_family=family if family is not None else judge_result.get("task_family"),
        )
        result["details"]["source_video_path"] = str(video_path)
        result["details"]["converter_version"] = self._converter.CONVERTER_VERSION
        return result

    def _convert(self, video_path: Path, out_dir: Path) -> Path:
        destination = out_dir / f"{video_path.stem}_top_left_right.mp4"
        status = self._converter.convert_one(video_path, destination, self._converter.DEFAULT_CRF, True)
        if status not in {"ok", "skip"}:
            raise EvaluationError("convert_failed", f"unexpected converter status {status}", http_status=500)
        return destination
