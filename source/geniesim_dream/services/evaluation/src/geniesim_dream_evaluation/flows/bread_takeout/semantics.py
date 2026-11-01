"""Bread take-out semantics, instruction routing, and normalizers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .schema import (
    BREAD_ARRIVAL_EVIDENCE,
    CONFIDENCE_LEVELS,
    DESTINATION_ENDPOINT_STATUSES,
    DESTINATION_STATES,
    EXTRACTION_STATES,
    INITIAL_BREAD_PRESENCE,
    OFF_PLATE_STATES,
    PLATE_BREAD_PRESENCE,
    PLATE_VISIBILITY,
    SOURCE_REMAINING_STATES,
    SOURCE_SLOT_STATES,
    SUCCESS_COMPATIBLE_TRAJECTORY_INTEGRITY,
)

_BREAD_SEMANTICS = """
TARGET BREAD SEMANTICS

The target is a separate physical slice of bread/toast, typically
yellow, light-yellow, beige, or lightly browned.

Color is only a cue. Do not call an arbitrary yellow rectangular object bread.

Valid target examples:
- sliced bread
- toast
- toasted bread slice
- sandwich-bread slice visibly inserted in a toaster

Not target bread:
- toaster body / slot / heating element
- plate / dish
- table / tray / cutting board
- cloth / napkin / sponge
- package graphics / printed bread image / reflection
"""

_PLATE_SEMANTICS = """
PLATE DESTINATION SEMANTICS

Valid plate:
- physical plate / dish
- shallow plate-like serving surface that can receive bread

Not plate:
- table
- toaster top
- tray (unless dataset semantics clearly treat it as a plate)
- cutting board
- bowl
- printed plate pattern
- gripper

Any valid physical plate is an acceptable destination.
Plate color is irrelevant and must not be used for hard filtering.
There may be multiple plates.
Different bread slices may be placed on different plates.
Do not choose one privileged target plate.
"""

_BREAD_POSITIVE_TERMS = (
    "bread",
    "toast",
    "toasted bread",
    "bread slice",
    "slice of bread",
    "sandwich bread",
    "sandwich-bread",
)

_STRONG_NON_BREAD_TERMS = (
    "toaster",
    "plate",
    "dish",
    "table",
    "tray",
    "cutting board",
    "cloth",
    "napkin",
    "sponge",
    "package",
    "printed",
    "picture",
    "photo",
    "image",
    "reflection",
    "gripper",
    "robot",
)


def bread_semantics_block() -> str:
    return _BREAD_SEMANTICS.strip()


def plate_semantics_block() -> str:
    return _PLATE_SEMANTICS.strip()


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


def _contains_whole_term(text: str, term: str) -> bool:
    return (
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
            text,
        )
        is not None
    )


def normalize_is_bread_like(
    appearance: Optional[str],
    model_value: Any,
) -> Optional[bool]:
    """strong explicit non-bread > explicit bread/toast term > model boolean."""

    text = (appearance or "").strip().lower()
    model_bool = _as_bool(model_value)

    if text:
        if any(_contains_whole_term(text, term) for term in _STRONG_NON_BREAD_TERMS):
            # "bread on plate" still contains bread; only reject if the
            # appearance is clearly only a non-bread object.
            if not any(
                _contains_whole_term(text, term) for term in _BREAD_POSITIVE_TERMS
            ):
                return False

        if any(_contains_whole_term(text, term) for term in _BREAD_POSITIVE_TERMS):
            return True

    return model_bool


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def is_supported_bread_takeout_instruction(instruction: str) -> bool:
    return bread_task_family(instruction) == "bread_takeout"


def bread_task_family(instruction: str) -> Optional[str]:
    """Return 'bread_takeout' or None.

    Curated EN labels are mostly transfer-to-plate phrasing, e.g.:
      Place bread from toaster onto plate
      Put the bread in the toaster into the plate
    not only classic "Take the bread out of the toaster."
    """

    text = (instruction or "").strip().lower()
    if not text:
        return None

    bread_noun = any(
        _has_word(text, w) for w in ("bread", "toast")
    ) or ("bread slice" in text) or ("slice of bread" in text)
    has_toaster = _has_word(text, "toaster")
    if not (bread_noun and has_toaster):
        return None

    has_plate = _has_word(text, "plate")
    take_like = any(
        _has_word(text, w) for w in ("take", "remove", "pull", "extract")
    ) or ("pull out" in text) or ("take out" in text)

    # Reject put/place/insert bread INTO toaster with no plate destination.
    if not has_plate and not take_like:
        return None

    # bread + toaster + (plate transfer OR take-out verb)
    if has_plate or take_like:
        return "bread_takeout"
    return None


def _norm_enum(value: Any, allowed: frozenset, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if text in allowed:
        return text
    return default


def _normalize_candidate(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    appearance = out.get("appearance") or ""
    if isinstance(appearance, str):
        out["is_bread_like"] = normalize_is_bread_like(
            appearance, out.get("is_bread_like")
        )

    initially = _as_bool(out.get("initially_in_toaster"))
    if initially is None:
        rel = str(out.get("initial_relation_to_toaster") or "").strip().lower()
        if rel in ("inserted", "partially_inserted"):
            initially = True
        elif rel in ("outside",):
            initially = False
    if initially is not None:
        out["initially_in_toaster"] = initially

    return out


def _normalize_plate(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    plate_like = _as_bool(out.get("is_plate_like"))
    if plate_like is not None:
        out["is_plate_like"] = plate_like
    conf = str(out.get("identity_confidence") or "").strip().lower()
    if conf in CONFIDENCE_LEVELS:
        out["identity_confidence"] = conf
    anchor = str(out.get("spatial_anchor") or "").strip()
    if anchor:
        out["spatial_anchor"] = anchor
    else:
        out["spatial_anchor"] = "unknown"
    out["initial_bread_presence"] = _norm_enum(
        out.get("initial_bread_presence"),
        INITIAL_BREAD_PRESENCE,
        "unknown",
    )
    return out


def _normalize_destination_plate(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    if "plate_id" not in out and out.get("id"):
        out["plate_id"] = out.get("id")
    consistent = _as_bool(out.get("identity_consistent"))
    if consistent is not None:
        out["identity_consistent"] = consistent
    out["visibility"] = _norm_enum(
        out.get("visibility"), PLATE_VISIBILITY, "unknown"
    )
    out["end_bread_presence"] = _norm_enum(
        out.get("end_bread_presence"), PLATE_BREAD_PRESENCE, "unknown"
    )
    out["bread_arrival_evidence"] = _norm_enum(
        out.get("bread_arrival_evidence"), BREAD_ARRIVAL_EVIDENCE, "unknown"
    )
    return out


def _normalize_off_plate(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "state": "unknown",
            "locations": [],
            "confidence": "low",
            "reason": None,
        }
    out = dict(obj)
    out["state"] = _norm_enum(out.get("state"), OFF_PLATE_STATES, "unknown")
    locs = out.get("locations")
    if isinstance(locs, list):
        out["locations"] = [str(x).strip().lower() for x in locs if str(x).strip()]
    else:
        out["locations"] = []
    conf = str(out.get("confidence") or "").strip().lower()
    out["confidence"] = conf if conf in CONFIDENCE_LEVELS else "low"
    return out


def _normalize_destination_endpoint(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "status": "uncertain",
            "confidence": "low",
            "supporting_plate_ids": [],
            "reason": None,
        }
    out = dict(obj)
    out["status"] = _norm_enum(
        out.get("status"), DESTINATION_ENDPOINT_STATUSES, "uncertain"
    )
    conf = str(out.get("confidence") or "").strip().lower()
    out["confidence"] = conf if conf in CONFIDENCE_LEVELS else "low"
    ids = out.get("supporting_plate_ids")
    if isinstance(ids, list):
        out["supporting_plate_ids"] = [
            str(x).strip() for x in ids if str(x).strip()
        ]
    else:
        out["supporting_plate_ids"] = []
    return out


def normalize_start_state(
    start_state: Optional[Dict[str, Any]],
    instruction: str = "",
) -> Optional[Dict[str, Any]]:
    if not isinstance(start_state, dict):
        return start_state

    out = dict(start_state)
    out["task_family"] = "bread_takeout"
    out["target_quantifier"] = "all_initial_toaster_bread"

    cands = out.get("target_candidates")
    if isinstance(cands, list):
        out["target_candidates"] = [
            _normalize_candidate(dict(c)) for c in cands if isinstance(c, dict)
        ]

    plates = out.get("plate_candidates")
    if isinstance(plates, list):
        out["plate_candidates"] = [
            _normalize_plate(dict(p)) for p in plates if isinstance(p, dict)
        ]

    return out


def normalize_end_state(
    end_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(end_state, dict):
        return end_state

    out = dict(end_state)
    out["remaining_target_bread"] = _norm_enum(
        out.get("remaining_target_bread"),
        SOURCE_REMAINING_STATES,
        "unknown",
    )

    slots = out.get("slot_states")
    if isinstance(slots, list):
        fixed_slots: List[Dict[str, Any]] = []
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            s = dict(slot)
            s["state"] = _norm_enum(s.get("state"), SOURCE_SLOT_STATES, "unknown")
            fixed_slots.append(s)
        out["slot_states"] = fixed_slots

    dest_plates = out.get("destination_plates")
    if isinstance(dest_plates, list):
        out["destination_plates"] = [
            _normalize_destination_plate(dict(p))
            for p in dest_plates
            if isinstance(p, dict)
        ]
    else:
        out["destination_plates"] = []

    out["off_plate_target_bread"] = _normalize_off_plate(
        out.get("off_plate_target_bread")
    )
    out["destination_endpoint"] = _normalize_destination_endpoint(
        out.get("destination_endpoint")
    )

    return out


def _normalize_track(track: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(track)
    appearance = out.get("appearance") or ""
    if isinstance(appearance, str):
        out["is_bread_like"] = normalize_is_bread_like(
            appearance, out.get("is_bread_like")
        )

    tm = _as_bool(out.get("target_match"))
    if tm is not None:
        out["target_match"] = tm

    pec = _as_bool(out.get("plate_entry_confirmed"))
    if pec is not None:
        out["plate_entry_confirmed"] = pec

    merge = _as_bool(out.get("merge_after_plate_entry"))
    if merge is not None:
        out["merge_after_plate_entry"] = merge

    lost = _as_bool(out.get("identity_lost_after_plate_entry"))
    if lost is not None:
        out["identity_lost_after_plate_entry"] = lost

    later = _as_bool(out.get("later_undo"))
    if later is not None:
        out["later_undo"] = later

    origin = _as_bool(out.get("originated_from_toaster"))
    if origin is not None:
        out["originated_from_toaster"] = origin

    out["extraction_status"] = _norm_enum(
        out.get("extraction_status"), EXTRACTION_STATES, "unknown"
    )
    out["destination_outcome"] = _norm_enum(
        out.get("destination_outcome"), DESTINATION_STATES, "unknown"
    )

    integrity = str(out.get("trajectory_integrity") or "").strip().lower()
    if integrity in SUCCESS_COMPATIBLE_TRAJECTORY_INTEGRITY or integrity in {
        "vanished_before_plate_entry",
        "unclear",
    }:
        out["trajectory_integrity"] = integrity
    elif integrity:
        out["trajectory_integrity"] = integrity
    else:
        out["trajectory_integrity"] = "unclear"

    return out


def normalize_trajectory_state(
    verify_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(verify_state, dict):
        return verify_state

    out = dict(verify_state)
    tracks = out.get("object_tracks")
    if isinstance(tracks, list):
        out["object_tracks"] = [
            _normalize_track(dict(t)) for t in tracks if isinstance(t, dict)
        ]

    override = out.get("source_endpoint_override_request")
    if isinstance(override, dict):
        ov = dict(override)
        requested = _as_bool(ov.get("requested"))
        if requested is not None:
            ov["requested"] = requested
        if "proposed_remaining_target_bread" in ov:
            ov["proposed_remaining_target_bread"] = _norm_enum(
                ov.get("proposed_remaining_target_bread"),
                SOURCE_REMAINING_STATES,
                "unknown",
            )
        out["source_endpoint_override_request"] = ov

    return out
