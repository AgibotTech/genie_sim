"""Thin fold-towel resolver: endpoint hard vetoes, else Stage3 verdict."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .schema import FOLD_WRIST_RECHECK_ENABLED
from .semantics import (
    derive_end_completion_candidate,
    is_supported_fold_towel_instruction,
)


def _fill_stage3_debug(
    debug: Dict[str, Any],
    verify_state: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(verify_state, dict):
        return
    debug["stage3_completion_status"] = (
        str(verify_state.get("completion_status") or "").strip().lower() or None
    )
    debug["stage3_completion_confidence"] = verify_state.get(
        "completion_confidence"
    )
    debug["stage3_observed_action"] = verify_state.get("observed_action")
    debug["stage3_observed_fold_events"] = verify_state.get(
        "observed_fold_events"
    )
    debug["stage3_reopen_to_flat"] = verify_state.get("reopen_to_flat")
    debug["stage3_wm_collapse"] = verify_state.get("wm_collapse")


def resolve_task_completion(
    instruction: str,
    start_state: Optional[Dict[str, Any]],
    end_state: Optional[Dict[str, Any]],
    verify_state: Optional[Dict[str, Any]],
    wrist_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[int], str, Dict[str, Any]]:
    """Return (success 0/1/None, reason, debug).

    hard vetoes (priority over Stage3 success):
    - Stage2 final_fold_state == crumpled
    - Stage2 final_shape == triangle  (GT requires square/compact rectangle)
    Otherwise Stage3 completion_status maps to 1/0/0.
    """
    _ = start_state
    _ = wrist_state
    _ = FOLD_WRIST_RECHECK_ENABLED

    debug: Dict[str, Any] = {
        "resolution_mode": "fold_towel_release_endpoint_hard_veto_then_stage3",
        "hard_veto": None,
        "stage3_completion_status": None,
        "stage3_completion_confidence": None,
        "stage3_observed_action": None,
        "stage3_observed_fold_events": None,
        "stage3_reopen_to_flat": None,
        "stage3_wm_collapse": None,
        "stage2_final_fold_state": None,
        "stage2_final_shape": None,
        "stage2_footprint_ratio_category": None,
        "stage2_benchmark_completion_state": None,
        "stage2_endpoint_evidence": None,
        "stage2_endpoint_completion_evidence": None,
        "stage2_endpoint_candidate_debug_only": derive_end_completion_candidate(
            end_state
        ),
        "wrist_recheck_used": False,
    }

    if isinstance(end_state, dict):
        debug["stage2_final_fold_state"] = end_state.get("final_fold_state")
        debug["stage2_final_shape"] = end_state.get("final_shape")
        debug["stage2_footprint_ratio_category"] = end_state.get(
            "footprint_ratio_category"
        )
        debug["stage2_benchmark_completion_state"] = end_state.get(
            "benchmark_completion_state"
        )
        debug["stage2_endpoint_evidence"] = end_state.get(
            "endpoint_fold_evidence"
        )
        debug["stage2_endpoint_completion_evidence"] = end_state.get(
            "endpoint_completion_evidence"
        )

    if not is_supported_fold_towel_instruction(instruction):
        return None, "unsupported_fold_towel_instruction", debug

    if isinstance(end_state, dict):
        final_fold_state = str(
            end_state.get("final_fold_state") or ""
        ).strip().lower()
        final_shape = str(end_state.get("final_shape") or "").strip().lower()

        if final_fold_state == "crumpled":
            debug["hard_veto"] = "stage2_final_crumpled"
            _fill_stage3_debug(debug, verify_state)
            return 0, "stage2_final_crumpled_hard_fail", debug

        if final_shape == "triangle":
            debug["hard_veto"] = "stage2_final_triangle"
            _fill_stage3_debug(debug, verify_state)
            return 0, "stage2_final_triangle_hard_fail", debug

    if not isinstance(verify_state, dict):
        return None, "missing_stage3_fold_verdict", debug

    _fill_stage3_debug(debug, verify_state)
    status = debug["stage3_completion_status"] or ""

    if status == "success":
        return 1, "stage3_fold_success", debug
    if status == "failure":
        return 0, "stage3_fold_failure", debug
    if status == "uncertain":
        return 0, "stage3_fold_uncertain", debug
    return None, "invalid_stage3_fold_verdict", debug
