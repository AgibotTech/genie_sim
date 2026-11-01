"""Request/response contracts for Dream video evaluation."""

from __future__ import annotations

from typing import Any

ADAPTER_VERSION = "dream-video-adapter-v1"
PROTOCOL = "dream.evaluation.video.v1"
EXPECTED_LAYOUT = "head_full_width_top__left_right_wrists_bottom"
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 600
EXPECTED_FPS = 10


class EvaluationError(Exception):
    def __init__(self, code: str, message: str, *, evaluation_id: str | None = None, http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.evaluation_id = evaluation_id
        self.http_status = http_status

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.evaluation_id:
            payload["evaluation_id"] = self.evaluation_id
        return {"error": payload}


def validate_video_evidence(body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise EvaluationError("invalid_request", "Request body must be an object")
    protocol = body.get("protocol")
    if protocol != PROTOCOL:
        raise EvaluationError("unsupported_protocol", f"Expected {PROTOCOL}, got {protocol!r}")
    evidence = body.get("evidence")
    if not isinstance(evidence, dict):
        raise EvaluationError("invalid_evidence", "evidence must be an object")
    evaluation_id = str(evidence.get("evaluation_id") or "")
    instruction = str(evidence.get("instruction") or "").strip()
    if not instruction:
        raise EvaluationError("empty_instruction", "instruction must be non-empty", evaluation_id=evaluation_id or None)
    video = evidence.get("video")
    if not isinstance(video, dict):
        raise EvaluationError("invalid_video", "evidence.video must be an object", evaluation_id=evaluation_id or None)
    path = str(video.get("path") or "").strip()
    if not path:
        raise EvaluationError("missing_video_path", "evidence.video.path is required", evaluation_id=evaluation_id or None)
    try:
        width = int(video.get("width"))
        height = int(video.get("height"))
        fps = float(video.get("fps"))
        frame_count = int(video.get("frame_count"))
    except (TypeError, ValueError) as error:
        raise EvaluationError(
            "invalid_video_metadata",
            f"video width/height/fps/frame_count invalid: {error}",
            evaluation_id=evaluation_id or None,
        ) from error
    layout = str(video.get("layout") or "")
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise EvaluationError(
            "unsupported_geometry",
            f"expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, got {width}x{height}",
            evaluation_id=evaluation_id or None,
        )
    if layout and layout != EXPECTED_LAYOUT:
        raise EvaluationError(
            "unsupported_layout",
            f"expected layout {EXPECTED_LAYOUT}, got {layout}",
            evaluation_id=evaluation_id or None,
        )
    if abs(fps - EXPECTED_FPS) > 1e-3:
        raise EvaluationError(
            "unsupported_fps",
            f"expected fps {EXPECTED_FPS}, got {fps}",
            evaluation_id=evaluation_id or None,
        )
    if frame_count < 1:
        raise EvaluationError("invalid_frame_count", "frame_count must be >= 1", evaluation_id=evaluation_id or None)
    return {
        "evaluation_id": evaluation_id,
        "session_id": str(evidence.get("session_id") or ""),
        "instruction": instruction,
        "video_path": path,
        "sha256": str(video.get("sha256") or ""),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "layout": layout or EXPECTED_LAYOUT,
        "domain": evidence.get("domain"),
        "fd_model_id": evidence.get("fd_model_id"),
        "steps": evidence.get("steps"),
        "backend": evidence.get("backend"),
    }
