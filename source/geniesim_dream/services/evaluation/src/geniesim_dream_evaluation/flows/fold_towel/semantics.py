"""Fold towel instruction routing and VLM JSON normalizers."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .schema import (
    BENCHMARK_COMPLETION_STATES,
    COMPLETION_STATUSES,
    CONFIDENCE_LEVELS,
    ENDPOINT_COMPLETION_EVIDENCE,
    ENDPOINT_FOLD_EVIDENCE,
    FINAL_FOLD_STATES,
    FINAL_SHAPES,
    FOLD_ACTION_TYPES,
    FOLD_PROGRESS_STATES,
    FOOTPRINT_CHANGE,
    FOOTPRINT_RATIO_CATEGORIES,
    INITIAL_FOLD_STATES,
    OBSERVED_FOLD_EVENTS,
    REGULARITY,
    WM_COLLAPSE_LEVELS,
    YES_NO_UNKNOWN,
)


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def is_supported_fold_towel_instruction(instruction: str) -> bool:
    """Match explicit towel folding instructions; garments use fold_cloth."""
    text = (instruction or "").strip().lower()
    if not text:
        return False
    if any(_has_word(text, word) for word in ("unfold", "unfolding", "unfolded")):
        return False

    fold_like = (
        _has_word(text, "fold")
        or _has_word(text, "folding")
        or _has_word(text, "folded")
    )
    cloth_like = _has_word(text, "towel") or _has_word(text, "towels")
    return bool(fold_like and cloth_like)


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return None


def _norm_enum(value: Any, allowed: frozenset, default: str = "unclear") -> str:
    text = str(value or "").strip().lower()
    if text in allowed:
        return text
    return default


def _normalize_candidate(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    towel_like = _as_bool(out.get("is_towel_like"))
    if towel_like is not None:
        out["is_towel_like"] = towel_like
    out["initial_fold_state"] = _norm_enum(
        out.get("initial_fold_state"), INITIAL_FOLD_STATES, "unclear"
    )
    conf = str(out.get("identity_confidence") or "").strip().lower()
    if conf in CONFIDENCE_LEVELS:
        out["identity_confidence"] = conf
    return out


def normalize_start_state(
    start_state: Optional[Dict[str, Any]],
    instruction: str = "",
) -> Optional[Dict[str, Any]]:
    _ = instruction
    if not isinstance(start_state, dict):
        return start_state

    out = dict(start_state)
    out["task_family"] = "fold_towel"
    out["task_action"] = "fold"

    cands = out.get("target_candidates")
    if isinstance(cands, list):
        out["target_candidates"] = [
            _normalize_candidate(dict(c)) for c in cands if isinstance(c, dict)
        ]
    readable = _as_bool(out.get("scene_readable"))
    if readable is not None:
        out["scene_readable"] = readable
    return out


def normalize_end_state(
    end_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(end_state, dict):
        return end_state

    out = dict(end_state)
    out["final_fold_state"] = _norm_enum(
        out.get("final_fold_state"), FINAL_FOLD_STATES, "unclear"
    )
    out["final_shape"] = _norm_enum(
        out.get("final_shape"), FINAL_SHAPES, "unclear"
    )
    for key in (
        "fold_edge_visible",
        "double_or_multiple_layer_visible",
        "stacked_region_visible",
        "full_single_layer_sheet",
        "target_identity_consistent",
    ):
        if key in out:
            out[key] = _norm_enum(out.get(key), YES_NO_UNKNOWN, "unknown")
    out["footprint_change"] = _norm_enum(
        out.get("footprint_change"), FOOTPRINT_CHANGE, "unknown"
    )
    out["footprint_ratio_category"] = _norm_enum(
        out.get("footprint_ratio_category"),
        FOOTPRINT_RATIO_CATEGORIES,
        "unclear",
    )
    out["regularity"] = _norm_enum(
        out.get("regularity"), REGULARITY, "unknown"
    )
    out["benchmark_completion_state"] = _norm_enum(
        out.get("benchmark_completion_state"),
        BENCHMARK_COMPLETION_STATES,
        "unclear",
    )
    out["wm_collapse"] = _norm_enum(
        out.get("wm_collapse"), WM_COLLAPSE_LEVELS, "none"
    )
    out["endpoint_fold_evidence"] = _norm_enum(
        out.get("endpoint_fold_evidence"),
        ENDPOINT_FOLD_EVIDENCE,
        "uncertain",
    )
    out["endpoint_completion_evidence"] = _norm_enum(
        out.get("endpoint_completion_evidence"),
        ENDPOINT_COMPLETION_EVIDENCE,
        "uncertain",
    )
    readable = _as_bool(out.get("last_frame_readable"))
    if readable is not None:
        out["last_frame_readable"] = readable
    return out


def normalize_trajectory_state(
    verify_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(verify_state, dict):
        return verify_state

    out = dict(verify_state)
    out["observed_action"] = _norm_enum(
        out.get("observed_action"), FOLD_ACTION_TYPES, "unclear"
    )
    out["observed_fold_events"] = _norm_enum(
        out.get("observed_fold_events"), OBSERVED_FOLD_EVENTS, "unclear"
    )
    out["fold_progress"] = _norm_enum(
        out.get("fold_progress"), FOLD_PROGRESS_STATES, "unclear"
    )
    out["completion_status"] = _norm_enum(
        out.get("completion_status"), COMPLETION_STATUSES, "uncertain"
    )
    for key in (
        "same_target_throughout",
        "clear_layering_seen_mid",
        "panel_crossing_or_flip_seen",
        "self_overlap_established",
        "reopen_to_flat",
    ):
        if key in out:
            out[key] = _norm_enum(out.get(key), YES_NO_UNKNOWN, "unknown")
    out["wm_collapse"] = _norm_enum(
        out.get("wm_collapse"), WM_COLLAPSE_LEVELS, "none"
    )
    conf = str(out.get("completion_confidence") or "").strip().lower()
    out["completion_confidence"] = (
        conf if conf in CONFIDENCE_LEVELS else "low"
    )
    return out


def derive_end_completion_candidate(
    end_state: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Debug-only Stage2 benchmark-completion hint. Never overrides Stage3."""
    if not isinstance(end_state, dict):
        return None

    final_state = str(end_state.get("final_fold_state") or "").strip().lower()
    completion = str(
        end_state.get("benchmark_completion_state") or ""
    ).strip().lower()
    evidence = str(
        end_state.get("endpoint_completion_evidence") or ""
    ).strip().lower()

    if final_state in {"spread_flat", "crumpled"}:
        return 0
    if completion == "complete":
        return 1
    if completion in {"partial", "incorrect"}:
        return 0
    if evidence == "supports_complete":
        return 1
    if evidence in {"supports_partial", "contradicts_complete"}:
        return 0
    return None
