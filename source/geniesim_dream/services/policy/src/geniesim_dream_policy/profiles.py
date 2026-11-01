"""Explicit task/checkpoint identities shared by the bridge and pi05 server.

Never infer these from prompts: putting a doll in and taking it out use different
weights. Checkpoint directories include their own normalization assets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROFILE_KEY = "dream.policy_profile"
CHECKPOINT_KEY = "dream.policy_checkpoint"


@dataclass(frozen=True)
class PolicyProfile:
    id: str
    robot: str
    checkpoint_name: str

    @property
    def domain(self) -> str:
        return "agibotworld_g2" if self.robot == "g2" else "agibotworld_g1"

    @property
    def state_format(self) -> str:
        return f"dream_{self.robot}_wm_joints16"

    def checkpoint(self, asset_root: Path | None = None) -> Path:
        root = asset_root or os.environ.get("GENIESIM_DREAM_POLICY_ASSETS")
        if root is None:
            raise ValueError("Set GENIESIM_DREAM_POLICY_ASSETS to the policy weight directory")
        root = Path(root)
        return root / self.checkpoint_name


PROFILES = {
    p.id: p
    for p in (
        PolicyProfile("cloth", "g2", "g2-cloth-v1"),
        PolicyProfile("bread", "g01", "g1-bread-v1"),
        PolicyProfile(
            "towel",
            "g01",
            "g1-towel-v1",
        ),
        PolicyProfile("doll_in", "g01", "g1-doll-in-v1"),
        PolicyProfile("doll_out", "g01", "g1-doll-out-v1"),
    )
}


def get_profile(profile_id: str) -> PolicyProfile:
    if not isinstance(profile_id, str) or profile_id not in PROFILES:
        raise ValueError(f"Unknown pi05 task {profile_id!r}; choose one of {list(PROFILES)}")
    return PROFILES[profile_id]


def observation_profile(domain: str, state: dict | None) -> PolicyProfile:
    if state is None:
        raise ValueError("The selected example has no initial_state; export a matching seed")
    if not isinstance(state, dict):
        raise ValueError("observation.state must be an object")
    # Compatibility with existing G02 cloth examples only. G01 has four distinct
    # tasks, so an absent identity must never silently select a checkpoint.
    profile_id = state.get("policy_profile")
    if profile_id is None and domain == "agibotworld_g2":
        profile_id = "cloth"
    profile = get_profile(profile_id)
    if domain != profile.domain:
        raise ValueError(f"pi05 task {profile.id} requires domain {profile.domain}, got {domain}")
    if state.get("format") != profile.state_format:
        raise ValueError(f"pi05 task {profile.id} requires state format {profile.state_format}")
    return profile
