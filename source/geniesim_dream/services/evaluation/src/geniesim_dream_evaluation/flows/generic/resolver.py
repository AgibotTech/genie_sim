"""Thin resolver for generic three-stage: Stage3 completion_status only."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def resolve_task_completion(
    verify_state: Optional[Dict[str, Any]],
) -> Tuple[Optional[int], str, Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "resolution_mode": "generic_release_stage3_only",
        "hard_veto": None,
    }
    if not isinstance(verify_state, dict):
        return None, "missing_stage3", debug

    status = (verify_state.get("completion_status") or "").strip().lower()
    debug["completion_status"] = status
    if status == "success":
        return 1, "stage3_generic_success", debug
    if status == "failure":
        return 0, "stage3_generic_failure", debug
    if status == "uncertain":
        return 0, "stage3_generic_uncertain", debug
    return None, "stage3_generic_unparsed", debug
