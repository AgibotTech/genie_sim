from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActionChunk(StrictModel):
    action: list[list[float]]
    normalized: bool = False
    representation: Literal["stateend_29d_backward_framewise_rot6d"] = (
        "stateend_29d_backward_framewise_rot6d"
    )
    state: dict | None = None
    state_source: Literal["unavailable", "commanded", "estimated", "measured"] = "unavailable"

    @field_validator("action")
    @classmethod
    def valid_action(cls, value):
        # 32 = default closed-loop / original contract; 16 remains allowed
        # (DreamConfig.chunk_length selects which length a session runs at).
        if len(value) not in (16, 32) or any(len(row) != 29 for row in value):
            raise ValueError(f"action must have shape [16,29] or [32,29], got [{len(value)},29]")
        if any(not math.isfinite(item) for row in value for item in row):
            raise ValueError("action must contain only finite numbers")
        return value

    @model_validator(mode="after")
    def valid_state(self):
        if (self.state is None) != (self.state_source == "unavailable"):
            raise ValueError("state and state_source must explicitly agree")
        return self


class SessionRequest(StrictModel):
    example_id: str | None = None
    initial_image_png_b64: str | None = None
    domain: Literal["agibotworld_g1", "agibotworld_g2"] = "agibotworld_g1"
    caption: str | None = Field(default=None, max_length=16000)
    policy: Literal["replay", "external", "manual"] = "replay"
    # Per-session action chunk length.  32 = original contract; 16 = the G2
    # cloth closed-loop chunk (pi05 emits 50 actions; 16 frames x 3 fit).
    # The runtime, frame checks and observation all follow this choice
    # faithfully -- no silent resampling behind it.
    chunk_length: Literal[16, 32] = 32
    num_steps: int | None = Field(default=None, ge=1, le=200)
    seed: int = Field(default=42, ge=0)
    initial_state: dict | None = None
    state_source: Literal["unavailable", "commanded", "estimated", "measured"] = "unavailable"

    @model_validator(mode="after")
    def valid_initial(self):
        if bool(self.example_id) == bool(self.initial_image_png_b64):
            raise ValueError("Choose exactly one: example_id or initial_image_png_b64")
        if self.policy == "replay" and not self.example_id:
            raise ValueError("replay policy requires an example")
        if (self.initial_state is None) != (self.state_source == "unavailable"):
            raise ValueError("initial_state and state_source must explicitly agree")
        return self


class StepRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=100)
    expected_step: int = Field(ge=0)
    chunk: ActionChunk | None = None


class RunRequest(StepRequest):
    chunks: int = Field(default=1, ge=1, le=1000)


class EvaluationRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=100)
    question: str = Field(
        default="Evaluate task progress and visual consistency.", max_length=16000
    )
    instruction: str | None = Field(default=None, max_length=16000)


class ModelSelectionRequest(StrictModel):
    model_id: str = Field(min_length=1, max_length=100)


class EvaluationResult(StrictModel):
    evaluator: str
    assessment: str
    score: float | None = Field(default=None, ge=0, le=1)
    details: dict = Field(default_factory=dict)
    evidence_kind: Literal["visual_estimate"] = "visual_estimate"
