# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Service-side wire contract; simulator envelopes never enter core ports."""

from typing import Any, Protocol

from agent2act.domain.models import ActionChunk, Observation


class SimulatorCodec(Protocol):
    def decode(self, request: dict[str, Any]) -> Observation: ...
    def encode(
        self, chunk: ActionChunk, observation: Observation, *, history: dict, need_depth: bool
    ) -> dict: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
