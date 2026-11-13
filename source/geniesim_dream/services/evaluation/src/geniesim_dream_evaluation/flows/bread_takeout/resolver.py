"""Deterministic bread take-out des_no_gripper resolver.

Resolver behavior: Stage2 static gripper/arm endpoint evidence
no longer hard-fails; Stage3 temporal still_held remains an explicit failure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .schema import (
    BREAD_WRIST_RECHECK_ENABLED,
    ENDPOINT_EVIDENCE_SOURCES,
    EXTRACTION_SUCCESS,
    OCCLUDED_VISIBILITY,
    SUCCESS_COMPATIBLE_TRAJECTORY_INTEGRITY,
)
from .semantics import is_supported_bread_takeout_instruction

# Stage2 endpoint off-plate locations that must NOT hard-veto alone.
_GRIPPER_ENDPOINT_LOCATION_ALIASES = frozenset(
    {
        "gripper",
        "robotic_arm",
        "robot_arm",
        "robot_arm_gripper",
        "held_by_gripper",
        "still_held",
        "in_gripper",
        "between_gripper_jaws",
    }
)


def _str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return default


def _appearance(obj: Dict[str, Any]) -> str:
    return _str(obj.get("appearance") or obj.get("appearance_at_end"))


def required_targets_from_start(
    start_state: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not isinstance(start_state, dict):
        return []
    cands = start_state.get("target_candidates")
    if not isinstance(cands, list):
        return []
    required: List[Dict[str, Any]] = []
    for obj in cands:
        if not isinstance(obj, dict):
            continue
        if _bool(obj.get("is_bread_like")) and _bool(obj.get("initially_in_toaster")):
            required.append(dict(obj))
    return required


def _recover_from_stage3(
    verify_state: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Conservative zero-target recovery: toaster-origin tracks only."""
    if not isinstance(verify_state, dict):
        return []
    tracks = verify_state.get("object_tracks")
    if not isinstance(tracks, list):
        return []
    recovered: List[Dict[str, Any]] = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        if not _bool(track.get("originated_from_toaster")):
            continue
        if _str(track.get("source_origin_confidence")).lower() != "high":
            continue
        if not _bool(track.get("is_bread_like"), default=True):
            continue
        oid = _str(track.get("object_id"))
        if not oid:
            continue
        recovered.append(
            {
                "id": oid,
                "appearance": _appearance(track),
                "is_bread_like": True,
                "initially_in_toaster": True,
            }
        )
    return recovered


def resolve_required_targets(
    start_state: Optional[Dict[str, Any]],
    verify_state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    required = required_targets_from_start(start_state)
    if required:
        return required, "stage1", []
    recovered = _recover_from_stage3(verify_state)
    if recovered:
        return recovered, "downstream_recovery", [
            _str(r.get("id")) for r in recovered
        ]
    return [], "none", []


def apply_source_endpoint_override(
    end_state: Optional[Dict[str, Any]],
    verify_state: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Return (remaining_target_bread, override_debug)."""
    remaining = "unknown"
    confidence = "low"
    visibility = "unknown"
    if isinstance(end_state, dict):
        remaining = _str(end_state.get("remaining_target_bread"), "unknown").lower()
        if remaining not in {"none", "present", "unknown"}:
            remaining = "unknown"
        confidence = _str(
            end_state.get("source_relation_confidence"), "low"
        ).lower()
        visibility = _str(
            end_state.get("source_toaster_visibility"), "unknown"
        ).lower()

    override_debug: Dict[str, Any] = {
        "applied": False,
        "stage2_remaining": remaining,
        "stage2_confidence": confidence,
        "stage2_visibility": visibility,
        "reason": None,
    }

    if not isinstance(verify_state, dict):
        return remaining, override_debug

    req = verify_state.get("source_endpoint_override_request")
    if not isinstance(req, dict) or not _bool(req.get("requested")):
        return remaining, override_debug

    weak_stage2 = (
        remaining == "unknown"
        or confidence == "low"
        or visibility in OCCLUDED_VISIBILITY
    )
    if not weak_stage2:
        override_debug["reason"] = "stage2_not_weak"
        return remaining, override_debug

    # Clear present with medium/high confidence cannot be flipped to none.
    if remaining == "present" and confidence in {"medium", "high"}:
        override_debug["reason"] = "clear_present_authoritative"
        return remaining, override_debug

    evidence = _str(req.get("evidence_source")).lower()
    if evidence not in ENDPOINT_EVIDENCE_SOURCES:
        override_debug["reason"] = "evidence_not_end_source"
        return remaining, override_debug

    proposed = _str(req.get("proposed_remaining_target_bread"), "unknown").lower()
    if proposed not in {"none", "present", "unknown"}:
        override_debug["reason"] = "invalid_proposed"
        return remaining, override_debug

    override_debug["applied"] = True
    override_debug["reason"] = "stage3_end_override"
    override_debug["proposed"] = proposed
    override_debug["evidence_source"] = evidence
    return proposed, override_debug


def maybe_clear_source_from_extractions(
    remaining: str,
    required: List[Dict[str, Any]],
    verify_state: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """If Stage2 still says present but every required track was extracted, clear.

    Destination / plate success is checked later; this only lifts the early
    source_toaster_still_contains_bread veto when Stage3 tracks unanimously
    report extraction_status=removed.
    """
    debug: Dict[str, Any] = {
        "applied": False,
        "reason": None,
        "cleared_ids": [],
    }
    if remaining != "present" or not required:
        return remaining, debug

    tracks: List[Dict[str, Any]] = []
    if isinstance(verify_state, dict):
        raw_tracks = verify_state.get("object_tracks")
        if isinstance(raw_tracks, list):
            tracks = [t for t in raw_tracks if isinstance(t, dict)]

    used: Set[str] = set()
    cleared_ids: List[str] = []
    single_target = len(required) == 1
    for target in required:
        track = _find_track(tracks, target, used)
        if track is None and single_target and len(tracks) == 1:
            cand = tracks[0]
            oid = _str(cand.get("object_id"))
            if oid and oid not in used:
                track = cand
        if track is None:
            debug["reason"] = "missing_required_track"
            return remaining, debug
        oid = _str(track.get("object_id"))
        if oid:
            used.add(oid)
        extraction = _str(track.get("extraction_status"), "unknown").lower()
        if extraction not in EXTRACTION_SUCCESS:
            debug["reason"] = "not_all_extracted"
            return remaining, debug
        cleared_ids.append(oid or _str(target.get("id") or target.get("object_id")))

    debug["applied"] = True
    debug["reason"] = "stage3_all_extracted_clears_source"
    debug["cleared_ids"] = cleared_ids
    return "none", debug


def derive_source_clear_candidate(
    end_state: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Source-toaster clear only: 1 if none, 0 if present, None if unknown."""
    if not isinstance(end_state, dict):
        return None
    remaining = _str(end_state.get("remaining_target_bread"), "unknown").lower()
    if remaining == "none":
        return 1
    if remaining == "present":
        return 0
    return None


def _normalize_endpoint_location(value: Any) -> str:
    text = _str(value).lower()
    text = text.replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def _is_gripper_endpoint_location(value: Any) -> bool:
    return _normalize_endpoint_location(value) in _GRIPPER_ENDPOINT_LOCATION_ALIASES


def _read_destination_fields(
    end_state: Optional[Dict[str, Any]],
) -> Tuple[str, List[str], str, Optional[int]]:
    """Return (off_state, off_locations, dest_status, destination_candidate)."""
    off_state = "unknown"
    off_locations: List[str] = []
    dest_status = "uncertain"
    if isinstance(end_state, dict):
        off = end_state.get("off_plate_target_bread")
        if isinstance(off, dict):
            off_state = _str(off.get("state"), "unknown").lower()
            if off_state not in {"present", "none", "unknown"}:
                off_state = "unknown"
            raw_locations = off.get("locations")
            if isinstance(raw_locations, list):
                off_locations = [
                    _normalize_endpoint_location(x)
                    for x in raw_locations
                    if _str(x)
                ]
        dest = end_state.get("destination_endpoint")
        if isinstance(dest, dict):
            dest_status = _str(dest.get("status"), "uncertain").lower()
            if dest_status not in {"supported", "contradicted", "uncertain"}:
                dest_status = "uncertain"
        elif end_state.get("destination_endpoint") is None:
            # Missing field: treat as uncertain (permissive; keeps
            # fixtures that only set toaster fields working).
            dest_status = "uncertain"

    if dest_status == "supported":
        cand: Optional[int] = 1
    elif dest_status == "contradicted":
        cand = 0
    else:
        cand = None
    return off_state, off_locations, dest_status, cand


def _find_track(
    tracks: List[Dict[str, Any]],
    target: Dict[str, Any],
    used: Set[str],
) -> Optional[Dict[str, Any]]:
    tid = _str(target.get("id") or target.get("object_id"))
    # exact ID
    if tid:
        for track in tracks:
            oid = _str(track.get("object_id"))
            if oid == tid and oid not in used:
                return track
    # unique appearance fallback
    app = _appearance(target).lower()
    if app:
        matches = []
        for track in tracks:
            oid = _str(track.get("object_id"))
            if oid in used:
                continue
            if _appearance(track).lower() == app:
                matches.append(track)
        if len(matches) == 1:
            return matches[0]
    # single unused track fallback when only one target/track remain
    unused = [
        t
        for t in tracks
        if _str(t.get("object_id")) and _str(t.get("object_id")) not in used
    ]
    if len(unused) == 1:
        return unused[0]
    return None


def _target_failure_reason(track: Dict[str, Any]) -> str:
    extraction = _str(track.get("extraction_status"), "unknown").lower()
    destination = _str(track.get("destination_outcome"), "unknown").lower()
    integrity = _str(track.get("trajectory_integrity"), "unclear").lower()
    if extraction != "removed":
        return f"extraction_{extraction}"
    if destination != "placed_on_plate":
        return f"destination_{destination}"
    if not _bool(track.get("plate_entry_confirmed")):
        return "plate_entry_not_confirmed"
    if _bool(track.get("later_undo")):
        return "later_undo"
    if integrity == "vanished_before_plate_entry":
        return "vanished_before_plate_entry"
    if integrity not in SUCCESS_COMPATIBLE_TRAJECTORY_INTEGRITY:
        return f"trajectory_integrity_{integrity}"
    if not _bool(track.get("target_match"), default=True):
        return "target_match_false"
    return "unknown_target_failure"


def _object_success(
    track: Optional[Dict[str, Any]],
) -> Tuple[Optional[bool], str, Dict[str, Any]]:
    """Return (success True/False/None, reason, check).

    None means missing track ignored under the current rule (not an explicit failure).
    """
    check: Dict[str, Any] = {
        "object_id": None,
        "track_found": False,
        "extraction_status": None,
        "plate_entry_confirmed": None,
        "destination_outcome": None,
        "destination_plate_id": None,
        "merge_after_plate_entry": None,
        "later_undo": None,
        "trajectory_integrity": None,
        "object_success": None,
        "failure_reason": None,
        "note": None,
    }
    if track is None:
        check["note"] = "missing_target_track_ignored"
        return None, "missing_target_track_ignored", check

    check["object_id"] = _str(track.get("object_id"))
    check["track_found"] = True
    check["extraction_status"] = _str(track.get("extraction_status"), "unknown")
    check["plate_entry_confirmed"] = _bool(track.get("plate_entry_confirmed"))
    check["destination_outcome"] = _str(track.get("destination_outcome"), "unknown")
    check["destination_plate_id"] = track.get("destination_plate_id")
    check["merge_after_plate_entry"] = _bool(track.get("merge_after_plate_entry"))
    check["later_undo"] = _bool(track.get("later_undo"))
    check["trajectory_integrity"] = _str(
        track.get("trajectory_integrity"), "unclear"
    )

    ok = (
        check["track_found"]
        and _bool(track.get("target_match"), default=True)
        and check["extraction_status"] in EXTRACTION_SUCCESS
        and check["plate_entry_confirmed"] is True
        and check["destination_outcome"] == "placed_on_plate"
        and check["later_undo"] is False
        and check["trajectory_integrity"] in SUCCESS_COMPATIBLE_TRAJECTORY_INTEGRITY
    )
    if ok:
        check["object_success"] = True
        return True, "ok", check

    reason = _target_failure_reason(track)
    check["object_success"] = False
    check["failure_reason"] = reason
    return False, reason, check


def resolve_task_completion(
    instruction: str,
    start_state: Optional[Dict[str, Any]],
    end_state: Optional[Dict[str, Any]],
    verify_state: Optional[Dict[str, Any]],
    wrist_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[int], str, Dict[str, Any]]:
    """Return (success 0/1/None, reason, debug)."""

    _ = wrist_state  # Stage4 disabled; kept for judge signature compat.
    _ = BREAD_WRIST_RECHECK_ENABLED

    debug: Dict[str, Any] = {
        "resolution_mode": "bread_takeout_release",
        "required_target_ids": [],
        "required_target_source": "none",
        "recovered_target_ids": [],
        "source_toaster_remaining_bread": "unknown",
        "source_toaster_clear": False,
        "source_override": {},
        "source_track_clear": {},
        "object_checks": [],
        "failed_target_ids": [],
        "explicit_failed_target_ids": [],
        "ignored_missing_target_ids": [],
        "missing_target_track_ignored": False,
        "all_required_targets_success": False,
        "off_plate_target_bread_state": "unknown",
        "off_plate_target_bread_locations": [],
        "ignored_gripper_endpoint_locations": [],
        "effective_off_plate_locations": [],
        "gripper_endpoint_ignored": False,
        "destination_endpoint_status": "uncertain",
        "destination_endpoint_status_raw": "uncertain",
        "destination_endpoint_status_effective": "uncertain",
        "destination_endpoint_candidate": None,
        "destination_endpoint_candidate_raw": None,
        "destination_endpoint_candidate_effective": None,
    }

    if not is_supported_bread_takeout_instruction(instruction):
        return None, "unsupported_bread_takeout_instruction", debug

    required, source, recovered_ids = resolve_required_targets(
        start_state, verify_state
    )
    debug["required_target_source"] = source
    debug["recovered_target_ids"] = recovered_ids
    debug["required_target_ids"] = [
        _str(t.get("id") or t.get("object_id")) for t in required
    ]

    if not required:
        return 0, "no_required_bread_targets", debug

    remaining, override_debug = apply_source_endpoint_override(end_state, verify_state)
    debug["source_override"] = override_debug
    remaining, track_clear_debug = maybe_clear_source_from_extractions(
        remaining, required, verify_state
    )
    debug["source_track_clear"] = track_clear_debug
    debug["source_toaster_remaining_bread"] = remaining

    if remaining == "present":
        debug["source_toaster_clear"] = False
        return 0, "source_toaster_still_contains_bread", debug
    if remaining == "unknown":
        debug["source_toaster_clear"] = False
        return 0, "source_toaster_state_uncertain", debug

    debug["source_toaster_clear"] = True

    off_state, off_locations, raw_dest_status, raw_dest_cand = (
        _read_destination_fields(end_state)
    )
    ignored_gripper_locations = [
        loc for loc in off_locations if _is_gripper_endpoint_location(loc)
    ]
    effective_off_locations = [
        loc for loc in off_locations if not _is_gripper_endpoint_location(loc)
    ]
    gripper_only_off_plate = (
        off_state == "present"
        and bool(ignored_gripper_locations)
        and not effective_off_locations
    )

    effective_dest_status = raw_dest_status
    if raw_dest_status == "contradicted" and gripper_only_off_plate:
        effective_dest_status = "uncertain"

    if effective_dest_status == "supported":
        effective_cand: Optional[int] = 1
    elif effective_dest_status == "contradicted":
        effective_cand = 0
    else:
        effective_cand = None

    debug["off_plate_target_bread_state"] = off_state
    debug["off_plate_target_bread_locations"] = off_locations
    debug["ignored_gripper_endpoint_locations"] = ignored_gripper_locations
    debug["effective_off_plate_locations"] = effective_off_locations
    debug["gripper_endpoint_ignored"] = gripper_only_off_plate
    # Legacy fields keep raw VLM status for analysis compatibility.
    debug["destination_endpoint_status"] = raw_dest_status
    debug["destination_endpoint_status_raw"] = raw_dest_status
    debug["destination_endpoint_status_effective"] = effective_dest_status
    debug["destination_endpoint_candidate"] = raw_dest_cand
    debug["destination_endpoint_candidate_raw"] = raw_dest_cand
    debug["destination_endpoint_candidate_effective"] = effective_cand

    # Stage2 hard veto: only real non-gripper endpoint evidence.
    if off_state == "present" and not gripper_only_off_plate:
        return 0, "explicit_off_plate_bread_at_end", debug
    if effective_dest_status == "contradicted":
        return 0, "destination_endpoint_contradicted", debug

    tracks: List[Dict[str, Any]] = []
    if isinstance(verify_state, dict):
        raw_tracks = verify_state.get("object_tracks")
        if isinstance(raw_tracks, list):
            tracks = [t for t in raw_tracks if isinstance(t, dict)]

    used: Set[str] = set()
    object_checks: List[Dict[str, Any]] = []
    explicit_failed: List[str] = []
    ignored_missing: List[str] = []

    single_target = len(required) == 1
    for target in required:
        tid = _str(target.get("id") or target.get("object_id"))
        track = _find_track(tracks, target, used)
        if track is None and single_target and len(tracks) == 1:
            cand = tracks[0]
            oid = _str(cand.get("object_id"))
            if oid and oid not in used:
                track = cand
        if track is not None:
            oid = _str(track.get("object_id"))
            if oid:
                used.add(oid)

        ok, reason, check = _object_success(track)
        if not check.get("object_id"):
            check["object_id"] = tid
        object_checks.append(check)
        if ok is False:
            explicit_failed.append(tid or "?")
        elif ok is None:
            ignored_missing.append(tid or "?")

    debug["object_checks"] = object_checks
    debug["explicit_failed_target_ids"] = explicit_failed
    debug["ignored_missing_target_ids"] = ignored_missing
    debug["failed_target_ids"] = list(explicit_failed)  # only explicit fails
    debug["missing_target_track_ignored"] = bool(ignored_missing)

    has_explicit_fail = len(explicit_failed) > 0
    all_tracked_ok = (not has_explicit_fail) and all(
        c.get("object_success") is not False for c in object_checks
    )
    debug["all_required_targets_success"] = (
        all_tracked_ok and len(ignored_missing) == 0
    )

    if has_explicit_fail:
        return 0, "explicit_required_bread_failure", debug

    # No explicit track failure. Missing tracks ignored.
    # toaster already none; effective destination supported|uncertain allowed.
    if effective_dest_status in {"supported", "uncertain"}:
        if ignored_missing:
            return 1, "bread_takeout_success_missing_tracks_ignored", debug
        return 1, "all_required_bread_transferred_to_plate", debug

    return 0, "bread_takeout_endpoint_not_satisfied", debug
