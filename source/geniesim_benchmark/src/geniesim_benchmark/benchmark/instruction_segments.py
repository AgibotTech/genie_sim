# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Select episode segment text independently of the policy instruction."""


def get_episode_segments(subtask_steps: list | None, episode_id: int) -> list[str]:
    """Return authored text without exposing checker or budget metadata."""
    if not subtask_steps:
        return []
    steps = subtask_steps
    if isinstance(steps[0], list):
        steps = steps[episode_id % len(steps)]
    return [
        step["instruction"].strip()
        for step in steps
        if isinstance(step, dict)
        and isinstance(step.get("instruction"), str)
        and step["instruction"].strip()
    ]
