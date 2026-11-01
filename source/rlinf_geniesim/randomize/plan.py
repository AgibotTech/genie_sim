# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Serializable randomization results for one environment and seed."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RandomizationPlan:
    """Resolved parameters and sampled changes for one environment and seed.

    physics carries engine reset thresholds and sampled PD multipliers; sensor
    carries image augmentation parameters; render carries scene changes/readback.
    The plan records values without applying them."""

    seed: int
    env_id: int
    physics: dict[str, Any] = field(default_factory=dict)
    render: dict[str, Any] = field(default_factory=dict)
    sensor: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "RandomizationPlan":
        return cls(
            seed=int(d["seed"]),
            env_id=int(d["env_id"]),
            physics=dict(d.get("physics") or {}),
            render=dict(d.get("render") or {}),
            sensor=dict(d.get("sensor") or {}),
        )

    @classmethod
    def from_json(cls, s: str) -> "RandomizationPlan":
        return cls.from_dict(json.loads(s))


__all__ = ["RandomizationPlan"]
