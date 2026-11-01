# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Public scene provenance without private collection or execution receipts."""

import re

_INTERNAL_RECEIPT_FIELDS = frozenset(
    {
        "gs_episode_id",
        "gs_review_status",
        "replay_episode_id",
        "replay_job_id",
        "source_run",
        "source_task_result",
        "source_uuid",
        "replay_verified",
        "source_physics_passed",
    }
)


def public_catalog_metadata(value):
    """Keep scene identities, authoring hashes and generic training overlap."""
    if isinstance(value, list):
        return [public_catalog_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in _INTERNAL_RECEIPT_FIELDS:
            continue
        if re.fullmatch(r"in_dataset_\d+", key):
            key = "training_overlap"
        elif re.fullmatch(r"scene_content_in_dataset_\d+_before_robot_removal", key):
            key = "source_scene_training_overlap_before_robot_removal"
        elif re.fullmatch(r"source_run_in_dataset_\d+", key):
            key = "source_run_training_overlap"
        item = public_catalog_metadata(item)
        if key in result and result[key] != item:
            raise ValueError(f"Conflicting public catalog field: {key}")
        result[key] = item
    return result
