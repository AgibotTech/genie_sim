"""Light normalize helpers for generic three-stage JSON."""

from __future__ import annotations

from typing import Any, Dict, Optional


_YES_NO = frozenset({"yes", "no", "unknown"})
_VIS = frozenset({"clear", "partial", "occluded", "unreadable", "unknown"})
_WM = frozenset({"none", "mild", "severe"})
_ENDPOINT = frozenset(
    {"supports_complete", "contradicts_complete", "uncertain"}
)
_STATUS = frozenset({"success", "failure", "uncertain"})
_CONF = frozenset({"high", "medium", "low"})


def _norm_enum(value: Any, allowed: frozenset, default: str = "unknown") -> str:
    if not isinstance(value, str):
        return default
    v = value.strip().lower()
    return v if v in allowed else default


def normalize_start_state(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    out = dict(obj)
    if "task_family" not in out:
        out["task_family"] = "generic"
    return out


def normalize_end_state(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    out = dict(obj)
    out["target_identity_consistent"] = _norm_enum(
        out.get("target_identity_consistent"), _YES_NO
    )
    out["target_visibility"] = _norm_enum(out.get("target_visibility"), _VIS)
    out["endpoint_completion_evidence"] = _norm_enum(
        out.get("endpoint_completion_evidence"), _ENDPOINT, "uncertain"
    )
    out["wm_collapse"] = _norm_enum(out.get("wm_collapse"), _WM, "none")
    return out


def normalize_trajectory_state(
    obj: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    out = dict(obj)
    out["same_target_throughout"] = _norm_enum(
        out.get("same_target_throughout"), _YES_NO
    )
    out["wm_collapse"] = _norm_enum(out.get("wm_collapse"), _WM, "none")
    out["completion_status"] = _norm_enum(
        out.get("completion_status"), _STATUS, "uncertain"
    )
    out["completion_confidence"] = _norm_enum(
        out.get("completion_confidence"), _CONF, "low"
    )
    return out
