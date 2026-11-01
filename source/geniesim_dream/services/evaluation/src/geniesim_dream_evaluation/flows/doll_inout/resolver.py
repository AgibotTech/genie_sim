"""Deterministic final 0/1 resolver for interleave v5a2_doll_v4.

v5a2_doll_v4 = v5a1_no_gripper architecture
    + v4 doll recognition decision boundary
    + bowtie phantom_mat rescue (from v5a1_no_gripper_bowtie)

Gripper open/closed state is intentionally not used.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .schema import (
    ENDPOINT_EVIDENCE_PREFIX,
    FAIL_RELATIONS,
    NOT_MANIPULATED_ACTION,
    OCCLUDED_VISIBILITY,
    PHANTOM_MAT_ANIMAL_TOKENS,
    PHANTOM_MAT_BOW_TOKENS,
    PHANTOM_MAT_REQUIRED_TOKENS,
    PUT_WRIST_RECHECK_EXCLUDED_STATES,
    PUT_WRIST_RECHECK_RELATIONS,
    PUT_WRIST_RECHECK_VISIBILITY,
    PUT_SUPPORTING_ACTIONS,
    RECOVERY_ACTIONS,
    RESCUE_OCCLUSION_SOURCES,
    SUCCESS_RELATIONS,
    TAKE_SUCCESS_RELATIONS,
    TAKE_SUPPORTING_ACTIONS,
    TAKE_WRIST_RECHECK_ENABLED,
    TAKE_WRIST_RECHECK_RELATIONS,
    TAKE_WRIST_RECHECK_VISIBILITY,
    WRIST_KNOWN_RELATIONS,
    WRIST_NEGATIVE_OVERRIDE_CONFIDENCE,
    WRIST_POSITIVE_OVERRIDE_CONFIDENCE,
)

from .semantics import (
    instruction_action,
    normalize_is_doll_like,
)


def _str(
    value: Any,
    default: str = "",
) -> str:

    if value is None:
        return default

    return str(value).strip()


def _bool(
    value: Any,
    default: bool = False,
) -> bool:

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        low = value.strip().lower()

        if low in ("true", "yes", "1"):
            return True

        if low in ("false", "no", "0"):
            return False

    return default


def _container_usable(
    end_state: Dict[str, Any],
    container_id: Any,
) -> bool:

    containers = end_state.get("containers")

    if not isinstance(containers, list) or not containers:
        return True

    cid = _str(container_id)

    for c in containers:

        if not isinstance(c, dict):
            continue

        if cid and _str(c.get("id")) != cid:
            continue

        integrity = _str(
            c.get("container_integrity"),
            "usable",
        ).lower()

        if integrity == "collapsed_unusable":
            return False

        return True

    # Preserve v4 behavior:
    # unmatched container ID alone does not hard-fail.
    return True


def _override_allowed(
    end_obj: Dict[str, Any],
) -> bool:

    relation = _str(
        end_obj.get("final_relation"),
        "unknown",
    ).lower()

    visibility = _str(
        end_obj.get("visibility"),
        "unknown",
    ).lower()

    confidence = _str(
        end_obj.get("final_relation_confidence"),
        "low",
    ).lower()

    if (
        relation in FAIL_RELATIONS
        and visibility == "visible"
        and confidence in {"high", "medium"}
    ):
        return False

    if (
        visibility == "vanished"
        and confidence in {"high", "medium"}
    ):
        return False

    return (
        relation == "unknown"
        or visibility in OCCLUDED_VISIBILITY
        or confidence == "low"
    )


def apply_endpoint_override(
    end_obj: Dict[str, Any],
    override: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    updated = dict(end_obj)

    debug: Dict[str, Any] = {
        "override_requested": False,
        "override_applied": False,
        "override_reason": "none",
        "original_relation": _str(
            end_obj.get("final_relation"),
            "unknown",
        ),
        "proposed_relation": None,
        "evidence_source": None,
    }

    if (
        not isinstance(override, dict)
        or not _bool(override.get("requested"))
    ):
        return updated, debug

    debug["override_requested"] = True

    proposed = _str(
        override.get("proposed_relation")
    )

    source = _str(
        override.get("evidence_source")
    ).lower()

    debug["proposed_relation"] = (
        proposed or None
    )

    debug["evidence_source"] = (
        source or None
    )

    oid = _str(
        override.get("object_id")
    )

    end_id = _str(
        end_obj.get("id")
    )

    if oid and end_id and oid != end_id:
        debug["override_reason"] = (
            "object_id_mismatch"
        )

        return updated, debug

    if not source.startswith(
        ENDPOINT_EVIDENCE_PREFIX
    ):
        debug["override_reason"] = (
            "non_endpoint_evidence_rejected"
        )

        return updated, debug

    if not _override_allowed(end_obj):
        debug["override_reason"] = (
            "clear_stage2_endpoint_is_authoritative"
        )

        return updated, debug

    if not proposed:
        debug["override_reason"] = (
            "missing_proposed_relation"
        )

        return updated, debug

    updated["final_relation"] = proposed
    updated["final_relation_source"] = source

    debug["override_applied"] = True
    debug["override_reason"] = (
        "endpoint_override_applied"
    )

    return updated, debug


def _appearance(
    obj: Dict[str, Any],
) -> str:

    return _str(
        obj.get("appearance_at_end")
        or obj.get("appearance")
    )


def _required_targets(
    start_state: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    v5 target set comes from START, never from END or trajectory.

    Current benchmark uses generic doll instructions:
    every doll-like target_candidate is required.
    """

    if not isinstance(start_state, dict):
        return []

    raw = start_state.get("target_candidates")

    if not isinstance(raw, list):
        return []

    required: List[Dict[str, Any]] = []

    for obj in raw:

        if not isinstance(obj, dict):
            continue

        obj = dict(obj)

        appearance = _appearance(obj)

        doll_like = normalize_is_doll_like(
            appearance,
            obj.get("is_doll_like"),
        )

        obj["is_doll_like"] = doll_like

        if doll_like is True:
            required.append(obj)

    return required


def _recover_required_targets(
    end_state: Optional[Dict[str, Any]],
    verify_state: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Conservative fallback used ONLY when Stage1 found zero
    required targets.

    Stage2 END objects are the primary recovery source.
    Stage3 may supplement manipulated doll-like objects that
    Stage2 omitted.
    """

    recovered: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    recovery_debug: Dict[str, Any] = {
        "stage2_recovered_ids": [],
        "stage3_recovered_ids": [],
    }

    if isinstance(end_state, dict):
        raw_objects = end_state.get("objects")
        if isinstance(raw_objects, list):
            for obj in raw_objects:
                if not isinstance(obj, dict):
                    continue

                appearance = (
                    _str(obj.get("appearance_at_end"))
                    or _appearance(obj)
                )
                doll_like = normalize_is_doll_like(
                    appearance,
                    obj.get("is_doll_like"),
                )
                if doll_like is not True:
                    continue

                object_id = _str(obj.get("id"))
                if not object_id or object_id in seen_ids:
                    continue

                recovered.append(
                    {
                        "id": object_id,
                        "appearance": appearance,
                        "is_doll_like": True,
                        "initial_relation": "unknown",
                        "nearest_container_id": obj.get(
                            "final_container_id"
                        ),
                        "recovered_target": True,
                        "recovered_from": "stage2_end",
                    }
                )
                seen_ids.add(object_id)
                recovery_debug["stage2_recovered_ids"].append(
                    object_id
                )

    if isinstance(verify_state, dict):
        raw_tracks = verify_state.get("object_tracks")
        if isinstance(raw_tracks, list):
            for track in raw_tracks:
                if not isinstance(track, dict):
                    continue

                appearance = _appearance(track)
                doll_like = normalize_is_doll_like(
                    appearance,
                    track.get("is_doll_like"),
                )
                if doll_like is not True:
                    continue

                object_id = _str(track.get("object_id"))
                if not object_id or object_id in seen_ids:
                    continue

                net_action = _str(
                    track.get("net_action")
                ).lower()
                if net_action not in RECOVERY_ACTIONS:
                    continue

                recovered.append(
                    {
                        "id": object_id,
                        "appearance": appearance,
                        "is_doll_like": True,
                        "initial_relation": "unknown",
                        "nearest_container_id": track.get(
                            "destination_container_id"
                        ),
                        "recovered_target": True,
                        "recovered_from": "stage3_positive_track",
                    }
                )
                seen_ids.add(object_id)
                recovery_debug["stage3_recovered_ids"].append(
                    object_id
                )

    return recovered, recovery_debug


def _resolve_required_targets(
    start_state: Optional[Dict[str, Any]],
    end_state: Optional[Dict[str, Any]] = None,
    verify_state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Prefer Stage1 inventory.

    Only when Stage1 finds zero required targets, recover from
    Stage2 (primary) and Stage3 positive tracks (supplementary).
    Never expand the set when Stage1 already found >=1 target.
    """

    stage1 = _required_targets(start_state)
    meta: Dict[str, Any] = {
        "required_target_source": "stage1",
        "stage1_required_target_ids": [
            _str(obj.get("id")) for obj in stage1
        ],
        "recovered_target_ids": [],
        "recovery": None,
    }

    if stage1:
        return stage1, meta

    recovered, recovery_debug = _recover_required_targets(
        end_state=end_state,
        verify_state=verify_state,
    )
    meta["recovery"] = recovery_debug
    meta["recovered_target_ids"] = [
        _str(obj.get("id")) for obj in recovered
    ]

    if recovered:
        meta["required_target_source"] = "downstream_recovery"
        return recovered, meta

    meta["required_target_source"] = "none"
    return [], meta


def _appearance_matches(
    a: str,
    b: str,
) -> bool:

    a = a.strip().lower()
    b = b.strip().lower()

    if not a or not b:
        return False

    return (
        a in b
        or b in a
    )


def _find_unused_end_object(
    end_objects: List[Dict[str, Any]],
    required_obj: Dict[str, Any],
    used_indices: Set[int],
    allow_single_fallback: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[int], str]:

    required_id = _str(
        required_obj.get("id")
    )

    required_appearance = _appearance(
        required_obj
    )

    # First: exact stable object ID.
    if required_id:

        for idx, obj in enumerate(end_objects):

            if idx in used_indices:
                continue

            if _str(obj.get("id")) == required_id:
                return obj, idx, "object_id"

    # Second: unique appearance continuity fallback.
    matches: List[Tuple[int, Dict[str, Any]]] = []

    if required_appearance:

        for idx, obj in enumerate(end_objects):

            if idx in used_indices:
                continue

            if _appearance_matches(
                required_appearance,
                _appearance(obj),
            ):
                matches.append(
                    (idx, obj)
                )

    if len(matches) == 1:
        idx, obj = matches[0]

        return obj, idx, "appearance"

    # Preserve v4's useful single-target mismatch fallback,
    # but NEVER use it for multi-target tasks.
    if allow_single_fallback:

        remaining = [
            (idx, obj)
            for idx, obj in enumerate(end_objects)
            if idx not in used_indices
        ]

        if len(remaining) == 1:
            idx, obj = remaining[0]

            return (
                obj,
                idx,
                "single_target_fallback",
            )

    return None, None, "not_found"


def _find_unused_track(
    tracks: List[Dict[str, Any]],
    required_obj: Dict[str, Any],
    end_obj: Optional[Dict[str, Any]],
    used_indices: Set[int],
    allow_single_fallback: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[int], str]:

    required_id = _str(
        required_obj.get("id")
    )

    required_appearance = _appearance(
        required_obj
    )

    end_appearance = (
        _appearance(end_obj)
        if isinstance(end_obj, dict)
        else ""
    )

    # Exact ID first.
    if required_id:

        for idx, track in enumerate(tracks):

            if idx in used_indices:
                continue

            if (
                _str(track.get("object_id"))
                == required_id
            ):
                return (
                    track,
                    idx,
                    "object_id",
                )

    # Then appearance continuity.
    appearance_matches: List[
        Tuple[int, Dict[str, Any]]
    ] = []

    for idx, track in enumerate(tracks):

        if idx in used_indices:
            continue

        track_app = _appearance(track)

        if (
            _appearance_matches(
                required_appearance,
                track_app,
            )
            or _appearance_matches(
                end_appearance,
                track_app,
            )
        ):
            appearance_matches.append(
                (idx, track)
            )

    if len(appearance_matches) == 1:
        idx, track = appearance_matches[0]

        return (
            track,
            idx,
            "appearance",
        )

    # Important v5 change:
    # one track may NOT be reused for multiple required targets.
    if allow_single_fallback:

        remaining = [
            (idx, track)
            for idx, track in enumerate(tracks)
            if idx not in used_indices
        ]

        if len(remaining) == 1:
            idx, track = remaining[0]

            return (
                track,
                idx,
                "single_target_fallback",
            )

    return None, None, "not_found"


def _fragment_relations(
    obj: Dict[str, Any],
) -> List[str]:

    raw = obj.get("fragment_relations")

    if not isinstance(raw, list):
        return []

    relations: List[str] = []

    for frag in raw:

        if not isinstance(frag, dict):
            continue

        rel = _str(
            frag.get("relation")
        ).lower()

        if rel:
            relations.append(rel)

    return relations


def _fragment_goal_satisfied(
    obj: Dict[str, Any],
    action: str,
) -> bool:

    integrity = _str(
        obj.get("target_integrity"),
        "unknown",
    ).lower()

    if integrity != "fragmented":
        return True

    if not _bool(
        obj.get(
            "all_fragments_accounted_for"
        )
    ):
        return False

    relations = _fragment_relations(obj)

    if not relations:
        return False

    if action == "put":

        return all(
            rel in SUCCESS_RELATIONS
            for rel in relations
        )

    if action == "take":

        return all(
            rel in TAKE_SUCCESS_RELATIONS
            for rel in relations
        )

    return False


def _endpoint_success_put(
    end_obj: Dict[str, Any],
    end_state: Dict[str, Any],
) -> bool:

    if end_obj.get("is_doll_like") is not True:
        return False

    visibility = _str(
        end_obj.get("visibility")
    ).lower()

    if visibility == "vanished":
        return False

    integrity = _str(
        end_obj.get("target_integrity"),
        "unknown",
    ).lower()

    if integrity in {
        "merged",
        "vanished",
    }:
        return False

    # Fragment-level relation is authoritative for fragmented targets.
    if integrity == "fragmented":

        if not _fragment_goal_satisfied(
            end_obj,
            "put",
        ):
            return False

    else:

        if _bool(
            end_obj.get(
                "task_invalidating_integrity"
            )
        ):
            return False

        relation = _str(
            end_obj.get("final_relation")
        ).lower()

        if relation not in SUCCESS_RELATIONS:
            return False

    if not _container_usable(
        end_state,
        end_obj.get(
            "final_container_id"
        ),
    ):
        return False

    return True


def _endpoint_success_take(
    end_obj: Dict[str, Any],
) -> bool:

    if end_obj.get("is_doll_like") is not True:
        return False

    visibility = _str(
        end_obj.get("visibility")
    ).lower()

    if visibility == "vanished":
        return False

    integrity = _str(
        end_obj.get("target_integrity"),
        "unknown",
    ).lower()

    if integrity in {
        "merged",
        "vanished",
    }:
        return False

    if integrity == "fragmented":

        if not _fragment_goal_satisfied(
            end_obj,
            "take",
        ):
            return False

    else:

        if _bool(
            end_obj.get(
                "task_invalidating_integrity"
            )
        ):
            return False

        relation = _str(
            end_obj.get("final_relation")
        ).lower()

        if relation not in TAKE_SUCCESS_RELATIONS:
            return False

    return True


def _track_supports_action(
    track: Optional[Dict[str, Any]],
    action: str,
) -> bool:

    if track is None:
        return False

    if _bool(
        track.get("later_undo")
    ):
        return False

    integrity = _str(
        track.get(
            "trajectory_integrity"
        ),
        "valid",
    ).lower()

    # v5 change:
    # fragmented no longer automatically invalidates the action.
    # Final fragment placement is checked at endpoint.
    if integrity == "vanished":
        return False

    net = _str(
        track.get("net_action")
    ).lower()

    if net == NOT_MANIPULATED_ACTION:
        return False

    if action == "put":
        return (
            net in PUT_SUPPORTING_ACTIONS
        )

    if action == "take":
        return (
            net in TAKE_SUPPORTING_ACTIONS
        )

    return False


def _start_already_satisfies_goal(
    start_obj: Dict[str, Any],
    action: str,
) -> bool:

    relation = _str(
        start_obj.get("initial_relation")
    ).lower()

    if action == "put":
        return (
            relation in SUCCESS_RELATIONS
        )

    if action == "take":
        return (
            relation in TAKE_SUCCESS_RELATIONS
        )

    return False


def _strong_endpoint_success(
    end_obj: Dict[str, Any],
    action: str,
) -> bool:
    """
    Conservative END evidence strong enough to outweigh a Stage3
    not_manipulated label under sparse temporal sampling.
    """

    relation = _str(
        end_obj.get("final_relation"),
        "unknown",
    ).lower()

    confidence = _str(
        end_obj.get("final_relation_confidence"),
        "low",
    ).lower()

    visibility = _str(
        end_obj.get("visibility"),
        "unknown",
    ).lower()

    integrity = _str(
        end_obj.get("target_integrity"),
        "unknown",
    ).lower()

    identity_match = _str(
        end_obj.get("identity_match"),
        "unknown",
    ).lower()

    if confidence != "high":
        return False

    if visibility != "visible":
        return False

    if integrity not in {
        "intact",
        "deformed_trackable",
    }:
        return False

    if _bool(end_obj.get("task_invalidating_integrity")):
        return False

    if identity_match not in {
        "same",
        "likely_same",
    }:
        return False

    if action == "put":
        return relation in SUCCESS_RELATIONS

    if action == "take":
        return relation in TAKE_SUCCESS_RELATIONS

    return False


def _can_occlusion_rescue(
    end_obj: Dict[str, Any],
    track: Optional[Dict[str, Any]],
    action: str,
) -> bool:

    if track is None:
        return False

    relation = _str(
        end_obj.get("final_relation")
    ).lower()

    visibility = _str(
        end_obj.get("visibility")
    ).lower()

    occ = _str(
        end_obj.get(
            "occlusion_source"
        ),
        "none",
    ).lower()

    if relation != "unknown":
        return False

    if visibility not in OCCLUDED_VISIBILITY:
        return False

    if occ not in RESCUE_OCCLUSION_SOURCES:
        return False

    if _bool(
        track.get("later_undo")
    ):
        return False

    integrity = _str(
        track.get(
            "trajectory_integrity"
        ),
        "valid",
    ).lower()

    # Keep v4's conservative occlusion rescue.
    if integrity != "valid":
        return False

    net = _str(
        track.get("net_action")
    ).lower()

    if action == "put":
        return net == "put_completed"

    if action == "take":
        return net == "take_completed"

    return False


def _can_hovering_wrist_occlusion_rescue(
    end_obj: Dict[str, Any],
    wrist_obj: Optional[Dict[str, Any]],
    track: Optional[Dict[str, Any]],
    action: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Object-level rescue for Stage2 hovering + Stage4 unknown with
    explicit wrist occlusion. Does not rewrite final_relation.
    """

    debug: Dict[str, Any] = {
        "eligible": False,
        "applied": False,
        "reason": "none",
        "stage2_relation": _str(
            end_obj.get("final_relation"),
            "unknown",
        ).lower(),
        "wrist_final_relation": None,
        "left_wrist_visibility": None,
        "right_wrist_visibility": None,
        "trajectory_net_action": (
            _str(track.get("net_action")).lower()
            if isinstance(track, dict)
            else None
        ),
        "trajectory_integrity": (
            _str(
                track.get("trajectory_integrity"),
                "unknown",
            ).lower()
            if isinstance(track, dict)
            else None
        ),
        "target_identity_consistent": (
            wrist_obj.get("target_identity_consistent")
            if isinstance(wrist_obj, dict)
            else None
        ),
        "container_identity_consistent": (
            wrist_obj.get("container_identity_consistent")
            if isinstance(wrist_obj, dict)
            else None
        ),
    }

    if action != "put":
        debug["reason"] = "not_put_action"
        return False, debug

    if debug["stage2_relation"] != "hovering":
        debug["reason"] = "stage2_not_hovering"
        return False, debug

    if not isinstance(wrist_obj, dict):
        debug["reason"] = "no_exact_id_wrist_result"
        return False, debug

    wrist_relation = _str(
        wrist_obj.get("wrist_final_relation"),
        "unknown",
    ).lower()

    left_visibility = _str(
        wrist_obj.get("left_wrist_visibility")
    ).lower()

    right_visibility = _str(
        wrist_obj.get("right_wrist_visibility")
    ).lower()

    debug["wrist_final_relation"] = wrist_relation
    debug["left_wrist_visibility"] = left_visibility
    debug["right_wrist_visibility"] = right_visibility

    if wrist_relation != "unknown":
        debug["reason"] = "wrist_relation_not_unknown"
        return False, debug

    if (
        left_visibility != "occluded"
        and right_visibility != "occluded"
    ):
        debug["reason"] = "no_explicit_wrist_occlusion"
        return False, debug

    if not _track_supports_action(
        track,
        action,
    ):
        debug["reason"] = "trajectory_does_not_support_put"
        return False, debug

    trajectory_integrity = _str(
        track.get("trajectory_integrity"),
        "unknown",
    ).lower()

    if trajectory_integrity != "valid":
        debug["reason"] = "trajectory_integrity_not_valid"
        return False, debug

    stage2_integrity = _str(
        end_obj.get("target_integrity"),
        "unknown",
    ).lower()

    if stage2_integrity in {
        "fragmented",
        "merged",
        "vanished",
    }:
        debug["reason"] = "stage2_target_integrity_invalid"
        return False, debug

    if _bool(
        end_obj.get("task_invalidating_integrity")
    ):
        debug["reason"] = "stage2_task_invalidating_integrity"
        return False, debug

    if _str(
        end_obj.get("visibility")
    ).lower() == "vanished":
        debug["reason"] = "stage2_target_vanished"
        return False, debug

    wrist_integrity = _str(
        wrist_obj.get("wrist_target_integrity"),
        "unknown",
    ).lower()

    if wrist_integrity in {
        "fragmented",
        "merged",
        "vanished",
    }:
        debug["reason"] = "wrist_target_integrity_invalid"
        return False, debug

    debug["eligible"] = True
    debug["applied"] = True
    debug["reason"] = (
        "stage2_hovering_plus_wrist_unknown_with_explicit_occlusion"
    )

    return True, debug


def _normalize_appearance(text: str) -> str:
    return re.sub(r"\s+", " ", _str(text).lower()).strip()


def _is_phantom_mat_tablecloth_target(
    appearance: str,
) -> bool:
    """
    Printed mat / tablecloth pattern misread as a second plush target.

    Requires plush + (bear|dog) + blue bow/bowtie (large not required).
    Handles tablecloth detections that do not correspond to a physical object.
    """

    normalized = _normalize_appearance(appearance)

    if not normalized:
        return False

    if not all(
        token in normalized
        for token in PHANTOM_MAT_REQUIRED_TOKENS
    ):
        return False

    if not any(
        animal in normalized
        for animal in PHANTOM_MAT_ANIMAL_TOKENS
    ):
        return False

    return any(
        bow in normalized
        for bow in PHANTOM_MAT_BOW_TOKENS
    )


def _phantom_mat_target_eligible(
    check: Dict[str, Any],
    action: str,
    multi_target: bool,
) -> bool:

    if action != "put" or not multi_target:
        return False

    appearance = _str(check.get("appearance"))

    if not _is_phantom_mat_tablecloth_target(
        appearance
    ):
        return False

    if check.get("endpoint_success"):
        return False

    net_action = _str(
        check.get("track_net_action")
    ).lower()

    if net_action != NOT_MANIPULATED_ACTION:
        return False

    return True


def _try_phantom_mat_episode_rescue(
    object_checks: List[Dict[str, Any]],
    action: str,
    multi_target: bool,
) -> bool:
    """
    Episode rescue when every failure is a phantom mat misidentification
    and every other required target succeeded.
    """

    if not object_checks:
        return False

    failed = [
        check
        for check in object_checks
        if not check.get("object_success")
    ]

    succeeded = [
        check
        for check in object_checks
        if check.get("object_success")
    ]

    if not failed or not succeeded:
        return False

    for check in failed:
        if not _phantom_mat_target_eligible(
            check,
            action,
            multi_target,
        ):
            return False

    return True


def _resolve_end_objects_with_override(
    end_state: Dict[str, Any],
    verify_state: Optional[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:

    objects = end_state.get("objects")

    if not isinstance(objects, list):
        return [], {
            "override_requested": False,
            "override_applied": False,
            "override_reason": "none",
        }

    override = None

    if isinstance(verify_state, dict):
        override = verify_state.get(
            "endpoint_override_request"
        )

    target_override_id = ""

    if (
        isinstance(override, dict)
        and _bool(
            override.get("requested")
        )
    ):
        target_override_id = _str(
            override.get("object_id")
        )

    resolved: List[Dict[str, Any]] = []

    override_debug: Dict[str, Any] = {
        "override_requested": False,
        "override_applied": False,
        "override_reason": "none",
    }

    for raw_obj in objects:

        if not isinstance(raw_obj, dict):
            continue

        obj = dict(raw_obj)

        appearance = _appearance(obj)

        obj["is_doll_like"] = (
            normalize_is_doll_like(
                appearance,
                obj.get("is_doll_like"),
            )
        )

        if (
            isinstance(override, dict)
            and _bool(
                override.get("requested")
            )
            and (
                not target_override_id
                or target_override_id
                == _str(obj.get("id"))
            )
        ):
            obj, od = apply_endpoint_override(
                obj,
                override,
            )

            override_debug = od

            # Only one v4-compatible override request.
            target_override_id = "__applied__"

        resolved.append(obj)

    return resolved, override_debug


_HOLDING_ABOVE_PATTERNS = (
    "holding above",
    "held above",
    "still holding",
    "gripper holding",
    "being held above",
    "held over the box",
    "held over the container",
    "holding the target above",
)


def _stage2_holding_above_cue(
    end_state: Optional[Dict[str, Any]],
) -> bool:
    """Detect Stage2 textual cues that the arm is still holding above."""

    if not isinstance(end_state, dict):
        return False

    text = " ".join(
        [
            _str(end_state.get("end_description")),
            _str(end_state.get("endpoint_reason")),
        ]
    ).lower()

    if not text.strip():
        return False

    if any(phrase in text for phrase in _HOLDING_ABOVE_PATTERNS):
        return True

    if re.search(r"\bholding\b.{0,80}\babove\b", text):
        return True

    if re.search(r"\bheld\b.{0,80}\babove\b", text):
        return True

    return False


def _wrist_trigger_reasons(
    end_obj: Dict[str, Any],
    track: Optional[Dict[str, Any]],
    action: str,
    holding_cue: bool,
) -> List[str]:
    """Compact reasons explaining why Stage4 was selected."""

    reasons: List[str] = []

    relation = _str(
        end_obj.get("final_relation"),
        "unknown",
    ).lower()

    visibility = _str(
        end_obj.get("visibility"),
        "unknown",
    ).lower()

    confidence = _str(
        end_obj.get("final_relation_confidence"),
        "low",
    ).lower()

    if action == "put":
        relation_triggers = PUT_WRIST_RECHECK_RELATIONS
        visibility_triggers = PUT_WRIST_RECHECK_VISIBILITY
    elif action == "take":
        relation_triggers = TAKE_WRIST_RECHECK_RELATIONS
        visibility_triggers = TAKE_WRIST_RECHECK_VISIBILITY
    else:
        relation_triggers = frozenset()
        visibility_triggers = frozenset()

    if relation in relation_triggers:
        reasons.append(f"stage2_{relation}")

    if visibility in visibility_triggers:
        reasons.append("stage2_occluded")

    if confidence == "low":
        reasons.append("stage2_low_confidence")

    if (
        holding_cue
        and relation not in SUCCESS_RELATIONS
    ):
        reasons.append("stage2_holding_above_text")

    if isinstance(track, dict):
        net_action = _str(track.get("net_action")).lower()
        if net_action:
            reasons.append(f"trajectory_{net_action}")

    return reasons


def _wrist_recheck_eligible(
    end_obj: Dict[str, Any],
    track: Optional[Dict[str, Any]],
    action: str,
    stage2_holding_cue: bool = False,
) -> bool:
    """
    Decide whether one target deserves a dedicated FINAL wrist recheck.

    v5a3_2D_no_outside:
    hovering / rim / unknown are relation-based Stage4 triggers.

    A clear outside relation is not a trigger by itself.
    Outside can still be rechecked through independent uncertainty
    signals such as low confidence, occlusion, or holding-above text.

    All Stage4 paths remain gated by trajectory support.
    """

    if track is None:
        return False

    if not _track_supports_action(
        track,
        action,
    ):
        return False

    if (
        action == "take"
        and not TAKE_WRIST_RECHECK_ENABLED
    ):
        # TAKE wrist recheck is intentionally disabled.
        # Disabled triggers:
        # - relation in TAKE_WRIST_RECHECK_RELATIONS
        # - visibility in TAKE_WRIST_RECHECK_VISIBILITY
        # - low Stage2 relation confidence
        # - Stage2 holding-above cue
        return False

    relation = _str(
        end_obj.get(
            "final_relation"
        ),
        "unknown",
    ).lower()

    visibility = _str(
        end_obj.get(
            "visibility"
        ),
        "unknown",
    ).lower()

    confidence = _str(
        end_obj.get(
            "final_relation_confidence"
        ),
        "low",
    ).lower()

    if action == "put":
        relation_triggers = PUT_WRIST_RECHECK_RELATIONS
        visibility_triggers = PUT_WRIST_RECHECK_VISIBILITY
    elif action == "take":
        relation_triggers = TAKE_WRIST_RECHECK_RELATIONS
        visibility_triggers = TAKE_WRIST_RECHECK_VISIBILITY
    else:
        return False

    if (
        action == "put"
        and (relation, visibility)
        in PUT_WRIST_RECHECK_EXCLUDED_STATES
    ):
        return False

    # Relation-based Stage4 trigger set. `outside` is intentionally excluded.
    if relation in relation_triggers:
        return True

    # Explicit endpoint visibility problem.
    if visibility in visibility_triggers:
        return True

    # Low-confidence Stage2 endpoint.
    if confidence == "low":
        return True

    # Textual holding-above cue on a non-success Stage2 relation.
    if (
        stage2_holding_cue
        and relation not in SUCCESS_RELATIONS
    ):
        return True

    return False


def select_wrist_recheck_targets(
    instruction: str,
    start_state: Optional[Dict[str, Any]],
    end_state: Optional[Dict[str, Any]],
    verify_state: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return compact per-target contexts that should be sent to Stage 4.

    Only targets with:
        ambiguous/failure-like endpoint
        +
        trajectory evidence supporting the instructed action

    are selected.
    """

    action = instruction_action(
        instruction
    )

    if action is None:
        return []

    if not isinstance(start_state, dict):
        return []

    if not isinstance(end_state, dict):
        return []

    holding_cue = (
        _stage2_holding_above_cue(end_state)
        if action == "put"
        else False
    )

    required, required_meta = _resolve_required_targets(
        start_state=start_state,
        end_state=end_state,
        verify_state=verify_state,
    )

    raw_end = end_state.get(
        "objects"
    )

    if not isinstance(raw_end, list):
        return []

    end_objects = [
        dict(obj)
        for obj in raw_end
        if isinstance(obj, dict)
    ]

    tracks: List[Dict[str, Any]] = []

    if isinstance(verify_state, dict):

        raw_tracks = verify_state.get(
            "object_tracks"
        )

        if isinstance(raw_tracks, list):
            tracks = [
                dict(track)
                for track in raw_tracks
                if isinstance(track, dict)
            ]

    used_end_indices: Set[int] = set()
    used_track_indices: Set[int] = set()

    allow_single_fallback = (
        len(required) == 1
    )

    selected: List[
        Dict[str, Any]
    ] = []

    for required_obj in required:

        end_obj, end_idx, _ = (
            _find_unused_end_object(
                end_objects,
                required_obj,
                used_end_indices,
                allow_single_fallback,
            )
        )

        if end_obj is None:
            continue

        if end_idx is not None:
            used_end_indices.add(
                end_idx
            )

        track, track_idx, _ = (
            _find_unused_track(
                tracks,
                required_obj,
                end_obj,
                used_track_indices,
                allow_single_fallback,
            )
        )

        if track_idx is not None:
            used_track_indices.add(
                track_idx
            )

        if not _wrist_recheck_eligible(
            end_obj,
            track,
            action,
            stage2_holding_cue=holding_cue,
        ):
            continue

        selected.append(
            {
                "object_id": _str(
                    required_obj.get("id")
                ),

                "appearance": (
                    _appearance(required_obj)
                    or _appearance(end_obj)
                ),

                "destination_container_id": (
                    end_obj.get(
                        "final_container_id"
                    )
                    or (
                        track.get(
                            "destination_container_id"
                        )
                        if isinstance(
                            track,
                            dict,
                        )
                        else None
                    )
                ),

                "stage2_final_relation": _str(
                    end_obj.get(
                        "final_relation"
                    ),
                    "unknown",
                ),

                "stage2_visibility": _str(
                    end_obj.get(
                        "visibility"
                    ),
                    "unknown",
                ),

                "stage2_confidence": _str(
                    end_obj.get(
                        "final_relation_confidence"
                    ),
                    "low",
                ),

                "stage2_occlusion_source": _str(
                    end_obj.get(
                        "occlusion_source"
                    ),
                    "none",
                ),

                "trajectory_net_action": (
                    _str(
                        track.get(
                            "net_action"
                        )
                    )
                    if isinstance(
                        track,
                        dict,
                    )
                    else None
                ),

                "stage2_holding_above_cue": (
                    holding_cue
                ),

                "wrist_recheck_trigger_reasons": (
                    _wrist_trigger_reasons(
                        end_obj,
                        track,
                        action,
                        holding_cue,
                    )
                ),
            }
        )

    return selected

def _find_wrist_result(
    wrist_state: Optional[Dict[str, Any]],
    object_id: str,
) -> Optional[Dict[str, Any]]:

    if not isinstance(
        wrist_state,
        dict,
    ):
        return None

    raw = wrist_state.get(
        "objects"
    )

    if not isinstance(raw, list):
        return None

    for obj in raw:

        if not isinstance(obj, dict):
            continue

        if (
            _str(
                obj.get("object_id")
            )
            == object_id
        ):
            return obj

    return None


def apply_wrist_recheck(
    end_obj: Dict[str, Any],
    wrist_obj: Optional[Dict[str, Any]],
    track: Optional[Dict[str, Any]],
    action: str,
    stage2_holding_cue: bool = False,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:
    """
    Apply a dedicated FINAL wrist observation to one Stage2 object.

    This is intentionally stricter than normal VLM reasoning:
    a failure-like endpoint can become success only with direct,
    high-confidence wrist spatial evidence.
    """

    updated = dict(end_obj)

    debug: Dict[str, Any] = {
        "available": (
            isinstance(
                wrist_obj,
                dict,
            )
        ),
        "eligible": False,
        "applied": False,
        "reason": "none",

        "stage2_relation": _str(
            end_obj.get(
                "final_relation"
            ),
            "unknown",
        ),

        "wrist_relation": None,
        "wrist_confidence": None,
        "evidence_view": None,
    }

    if not isinstance(
        wrist_obj,
        dict,
    ):
        debug["reason"] = (
            "no_wrist_result"
        )

        return updated, debug

    if not _wrist_recheck_eligible(
        end_obj,
        track,
        action,
        stage2_holding_cue=stage2_holding_cue,
    ):
        debug["reason"] = (
            "stage2_not_recheck_eligible"
        )

        return updated, debug


    debug["eligible"] = True

    # Never let wrist erase Stage2 task-invalidating fragmentation/merge.
    old_integrity = _str(
        end_obj.get("target_integrity"),
        "unknown",
    ).lower()
    debug["stage2_target_integrity"] = old_integrity
    debug["stage2_task_invalidating_integrity"] = _bool(
        end_obj.get("task_invalidating_integrity")
    )
    if (
        _bool(end_obj.get("task_invalidating_integrity"))
        and old_integrity in {"fragmented", "merged"}
    ):
        debug["reason"] = "stage2_invalidating_integrity_preserved"
        return updated, debug

    target_ok = _bool(
        wrist_obj.get(
            "target_identity_consistent"
        )
    )

    if not target_ok:
        debug["reason"] = (
            "target_identity_not_confirmed"
        )

        return updated, debug

    if action == "put":

        container_ok = _bool(
            wrist_obj.get(
                "container_identity_consistent"
            )
        )

        if not container_ok:
            debug["reason"] = (
                "container_identity_not_confirmed"
            )

            return updated, debug

    relation = _str(
        wrist_obj.get(
            "wrist_final_relation"
        ),
        "unknown",
    ).lower()

    confidence = _str(
        wrist_obj.get(
            "wrist_relation_confidence"
        ),
        "low",
    ).lower()

    evidence_view = _str(
        wrist_obj.get(
            "evidence_view"
        ),
        "none",
    ).lower()

    debug["wrist_relation"] = relation
    debug["wrist_confidence"] = confidence
    debug["evidence_view"] = evidence_view

    if relation not in WRIST_KNOWN_RELATIONS:

        debug["reason"] = (
            "wrist_relation_unknown"
        )

        return updated, debug

    if evidence_view == "none":

        debug["reason"] = (
            "no_direct_wrist_evidence"
        )

        return updated, debug

    if action == "put":
        wrist_success = (
            relation in SUCCESS_RELATIONS
        )

    elif action == "take":
        wrist_success = (
            relation in TAKE_SUCCESS_RELATIONS
        )

    else:
        wrist_success = False

    # --------------------------------------------------------
    # Positive override:
    # Stage2 failure-like -> wrist success
    #
    # Must be HIGH confidence.
    # --------------------------------------------------------

    if wrist_success:

        if (
            confidence
            not in WRIST_POSITIVE_OVERRIDE_CONFIDENCE
        ):
            debug["reason"] = (
                "positive_wrist_evidence_not_high_confidence"
            )

            return updated, debug

        if not _track_supports_action(
            track,
            action,
        ):
            debug["reason"] = (
                "trajectory_does_not_support_override"
            )

            return updated, debug

    # --------------------------------------------------------
    # Negative confirmation / correction:
    # ambiguous Stage2 -> wrist failure relation
    # --------------------------------------------------------

    else:

        if (
            confidence
            not in WRIST_NEGATIVE_OVERRIDE_CONFIDENCE
        ):
            debug["reason"] = (
                "negative_wrist_evidence_too_weak"
            )

            return updated, debug

    updated[
        "final_relation"
    ] = relation

    updated[
        "final_relation_confidence"
    ] = confidence

    if evidence_view == "left_wrist":

        updated[
            "final_relation_source"
        ] = "end_left_wrist"

    elif evidence_view == "right_wrist":

        updated[
            "final_relation_source"
        ] = "end_right_wrist"

    else:

        updated[
            "final_relation_source"
        ] = "end_wrist_both"

    # If the dedicated FINAL wrist crop directly sees the target,
    # a Stage2 "vanished" observation can be repaired.
    left_vis = _str(
        wrist_obj.get(
            "left_wrist_visibility"
        )
    ).lower()

    right_vis = _str(
        wrist_obj.get(
            "right_wrist_visibility"
        )
    ).lower()

    if (
        left_vis == "clear"
        or right_vis == "clear"
    ):
        updated[
            "visibility"
        ] = "visible"

        updated[
            "occlusion_source"
        ] = "none"

    # After a successful positive spatial override, clear stale Stage2
    # integrity flags that were caused only by the old spatial judgement.
    # Genuine fragmented/merged cases never reach here (early return above).
    repair_debug = _repair_stale_integrity_after_wrist_override(
        updated=updated,
        wrist_obj=wrist_obj,
        wrist_success=wrist_success,
        confidence=confidence,
    )
    debug.update(repair_debug)

    debug["applied"] = True
    debug["reason"] = (
        "dedicated_final_wrist_evidence_applied"
    )

    return updated, debug


def _repair_stale_integrity_after_wrist_override(
    updated: Dict[str, Any],
    wrist_obj: Dict[str, Any],
    wrist_success: bool,
    confidence: str,
) -> Dict[str, Any]:
    """
    Clear Stage2 integrity flags that were only tied to a spatial failure,
    once Stage4 has applied a high-confidence success relation and confirms
    the target itself is intact / deformed_trackable.

    Never clears genuine fragmented / merged structural failures
    (those are blocked earlier in apply_wrist_recheck).
    """

    repair_debug: Dict[str, Any] = {
        "integrity_repair_applied": False,
        "integrity_repair_reason": "none",
        "wrist_target_integrity": None,
        "post_wrist_task_invalidating_integrity": _bool(
            updated.get("task_invalidating_integrity")
        ),
    }

    wrist_integrity = _str(
        wrist_obj.get("wrist_target_integrity"),
        "unknown",
    ).lower()
    repair_debug["wrist_target_integrity"] = wrist_integrity

    old_integrity = _str(
        updated.get("target_integrity"),
        "unknown",
    ).lower()

    if not wrist_success:
        repair_debug["integrity_repair_reason"] = "not_wrist_success"
        return repair_debug

    if confidence not in WRIST_POSITIVE_OVERRIDE_CONFIDENCE:
        repair_debug["integrity_repair_reason"] = (
            "confidence_not_positive_override"
        )
        return repair_debug

    if wrist_integrity not in {"intact", "deformed_trackable"}:
        repair_debug["integrity_repair_reason"] = (
            "wrist_integrity_not_clear"
        )
        return repair_debug

    if old_integrity in {"fragmented", "merged"}:
        repair_debug["integrity_repair_reason"] = (
            "stage2_structural_integrity_preserved"
        )
        return repair_debug

    # Expand repair beyond vanished/unknown so intact+stale
    # task_invalidating_integrity (spatial mislabel) can be cleared.
    updated["target_integrity"] = wrist_integrity
    updated["task_invalidating_integrity"] = False
    updated["integrity_reason"] = "none"

    repair_debug["integrity_repair_applied"] = True
    repair_debug["integrity_repair_reason"] = (
        "high_confidence_wrist_confirms_intact_target"
    )
    repair_debug["post_wrist_task_invalidating_integrity"] = False
    return repair_debug


def derive_end_completion_candidate(
    instruction: str,
    start_state: Optional[Dict[str, Any]],
    end_state: Optional[Dict[str, Any]],
) -> Optional[int]:
    """
    Debug-only deterministic END candidate.

    v5 uses ALL required START targets, not any END object.
    """

    action = instruction_action(instruction)

    if action is None:
        return None

    if not isinstance(start_state, dict):
        return None

    if not isinstance(end_state, dict):
        return None

    required, _ = _resolve_required_targets(
        start_state=start_state,
        end_state=end_state,
        verify_state=None,
    )

    if not required:
        return None

    raw_end = end_state.get("objects")

    if not isinstance(raw_end, list):
        return 0

    end_objects = [
        dict(o)
        for o in raw_end
        if isinstance(o, dict)
    ]

    used_end_indices: Set[int] = set()

    allow_single_fallback = (
        len(required) == 1
    )

    for required_obj in required:

        end_obj, end_idx, _ = (
            _find_unused_end_object(
                end_objects,
                required_obj,
                used_end_indices,
                allow_single_fallback,
            )
        )

        if end_obj is None:
            return 0

        if end_idx is not None:
            used_end_indices.add(
                end_idx
            )

        end_obj = dict(end_obj)

        # Semantic identity inheritance:
        # this END object is matched to a known required START doll.
        end_obj["is_doll_like"] = True

        if action == "put":

            ok = _endpoint_success_put(
                end_obj,
                end_state,
            )

        else:

            ok = _endpoint_success_take(
                end_obj
            )

        if not ok:
            return 0

    return 1


def resolve_task_completion(
    instruction: str,
    start_state: Optional[Dict[str, Any]],
    end_state: Optional[Dict[str, Any]],
    verify_state: Optional[Dict[str, Any]],
    wrist_state: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Optional[int],
    str,
    Dict[str, Any],
]:

    debug: Dict[str, Any] = {
        "resolution_mode": "unresolved",
        "action": None,
        "target_quantifier": None,
        "required_target_ids": [],
        "all_required_targets_success": False,
        "failed_target_ids": [],
        "override": {
            "override_requested": False,
            "override_applied": False,
            "override_reason": "none",
        },
        "object_checks": [],
    }

    action = instruction_action(
        instruction
    )

    debug["action"] = action

    if action is None:

        return (
            None,
            "unknown_instruction_action",
            debug,
        )

    if not isinstance(start_state, dict):

        return (
            0,
            "missing_start_state",
            debug,
        )

    debug["target_quantifier"] = (
        _str(
            start_state.get(
                "target_quantifier"
            )
        )
    )

    if not isinstance(end_state, dict):

        return (
            0,
            "missing_end_state",
            debug,
        )

    required, required_meta = _resolve_required_targets(
        start_state=start_state,
        end_state=end_state,
        verify_state=verify_state,
    )

    debug["required_target_source"] = required_meta.get(
        "required_target_source"
    )
    debug["stage1_required_target_ids"] = required_meta.get(
        "stage1_required_target_ids"
    )
    debug["recovered_target_ids"] = required_meta.get(
        "recovered_target_ids"
    )
    debug["target_recovery"] = required_meta.get(
        "recovery"
    )

    debug["required_target_ids"] = [
        _str(obj.get("id"))
        for obj in required
    ]

    if not required:

        return (
            0,
            "no_required_target_objects_after_recovery",
            debug,
        )

    stage2_holding_cue = (
        _stage2_holding_above_cue(end_state)
        if action == "put"
        else False
    )
    debug["stage2_holding_above_cue"] = (
        stage2_holding_cue
    )

    resolved_end_objects, override_debug = (
        _resolve_end_objects_with_override(
            end_state,
            verify_state,
        )
    )

    debug["override"] = (
        override_debug
    )

    if not resolved_end_objects:

        return (
            0,
            "no_end_objects",
            debug,
        )

    tracks: List[Dict[str, Any]] = []

    if isinstance(verify_state, dict):

        raw_tracks = verify_state.get(
            "object_tracks"
        )

        if isinstance(raw_tracks, list):

            tracks = [
                dict(t)
                for t in raw_tracks
                if isinstance(t, dict)
            ]

    used_end_indices: Set[int] = set()
    used_track_indices: Set[int] = set()

    allow_single_fallback = (
        len(required) == 1
    )
    multi_target = len(required) >= 2

    for required_obj in required:

        required_id = _str(
            required_obj.get("id")
        )

        required_appearance = (
            _appearance(required_obj)
        )

        end_obj, end_idx, end_match = (
            _find_unused_end_object(
                resolved_end_objects,
                required_obj,
                used_end_indices,
                allow_single_fallback,
            )
        )

        check: Dict[str, Any] = {
            "object_id": (
                required_id or None
            ),
            "appearance": (
                required_appearance
                or None
            ),
            "required_target": True,
            "end_object_found": (
                end_obj is not None
            ),
            "end_match_method": (
                end_match
            ),
            "track_found": False,
            "track_match_method": None,
            "is_doll_like": True,
            "final_relation": None,
            "visibility": None,
            "endpoint_success": False,
            "already_satisfied_at_start": False,
            "track_net_action": None,
            "track_supports_action": False,
            "already_satisfied_rescue": False,
            "not_manipulated_endpoint_rescue": False,
            "strong_endpoint_success": False,
            "occlusion_rescue": False,
            "hovering_wrist_occlusion_rescue": False,
            "hovering_wrist_occlusion_rescue_debug": None,
            "stage2_holding_above_cue": False,
            "phantom_mat_target": False,
            "phantom_mat_rescue_eligible": False,
            "object_success": False,
            "failure_reason": None,
        }

        if end_obj is None:

            check["failure_reason"] = (
                "required_target_missing_from_end_state"
            )

            debug["object_checks"].append(
                check
            )

            continue

        if end_idx is not None:
            used_end_indices.add(
                end_idx
            )

        end_obj = dict(end_obj)

        # Critical v5 semantic inheritance:
        # a matched required START target remains doll-like.
        end_obj["is_doll_like"] = True

        track, track_idx, track_match = (
            _find_unused_track(
                tracks,
                required_obj,
                end_obj,
                used_track_indices,
                allow_single_fallback,
            )
        )

        if track_idx is not None:
            used_track_indices.add(
                track_idx
            )

        check["track_found"] = (
            track is not None
        )

        check["track_match_method"] = (
            track_match
        )

        if track is not None:

            # Same stable target identity.
            track = dict(track)
            track["is_doll_like"] = True

            check["track_net_action"] = (
                _str(
                    track.get(
                        "net_action"
                    )
                )
            )

        # v5a: dedicated FINAL wrist correction before endpoint scoring.
        check["pre_wrist_relation"] = _str(
            end_obj.get("final_relation"),
            "unknown",
        )

        wrist_obj = _find_wrist_result(
            wrist_state,
            required_id,
        )

        # P1: inspect original Stage2 before apply_wrist_recheck may mutate it.
        hovering_wrist_rescue, hovering_wrist_debug = (
            _can_hovering_wrist_occlusion_rescue(
                end_obj=end_obj,
                wrist_obj=wrist_obj,
                track=track,
                action=action,
            )
        )

        check["hovering_wrist_occlusion_rescue"] = (
            hovering_wrist_rescue
        )

        check["hovering_wrist_occlusion_rescue_debug"] = (
            hovering_wrist_debug
        )

        end_obj, wrist_debug = apply_wrist_recheck(
            end_obj=end_obj,
            wrist_obj=wrist_obj,
            track=track,
            action=action,
            stage2_holding_cue=stage2_holding_cue,
        )

        check["wrist_recheck"] = wrist_debug

        check["final_relation"] = (
            _str(
                end_obj.get(
                    "final_relation"
                ),
                "unknown",
            )
        )

        check["visibility"] = (
            _str(
                end_obj.get(
                    "visibility"
                ),
                "unknown",
            )
        )

        if action == "put":

            endpoint_ok = (
                _endpoint_success_put(
                    end_obj,
                    end_state,
                )
            )

        else:

            endpoint_ok = (
                _endpoint_success_take(
                    end_obj
                )
            )

        check["endpoint_success"] = (
            endpoint_ok
        )

        already_satisfied = (
            _start_already_satisfies_goal(
                required_obj,
                action,
            )
        )

        check[
            "already_satisfied_at_start"
        ] = already_satisfied

        track_ok = (
            _track_supports_action(
                track,
                action,
            )
        )

        check["track_supports_action"] = (
            track_ok
        )

        net_action = (
            _str(track.get("net_action")).lower()
            if isinstance(track, dict)
            else ""
        )

        # Rescue 1: already satisfied at START; no manipulation required.
        already_satisfied_rescue = (
            endpoint_ok
            and already_satisfied
            and (
                track is None
                or net_action == NOT_MANIPULATED_ACTION
            )
        )
        check["already_satisfied_rescue"] = (
            already_satisfied_rescue
        )

        # Rescue 2: Stage3 not_manipulated under sparse sampling, but
        # Stage2 originally already had a clear success endpoint.
        # Stage4-only success must NOT unlock this path.
        pre_wrist_relation = _str(
            check.get("pre_wrist_relation"),
            "unknown",
        ).lower()

        if action == "put":
            pre_wrist_endpoint_success = (
                pre_wrist_relation in SUCCESS_RELATIONS
            )
        elif action == "take":
            pre_wrist_endpoint_success = (
                pre_wrist_relation in TAKE_SUCCESS_RELATIONS
            )
        else:
            pre_wrist_endpoint_success = False

        strong_endpoint_ok = (
            pre_wrist_endpoint_success
            and _strong_endpoint_success(
                end_obj,
                action,
            )
        )
        check["strong_endpoint_success"] = (
            strong_endpoint_ok
        )

        not_manipulated_endpoint_rescue = (
            endpoint_ok
            and not already_satisfied
            and net_action == NOT_MANIPULATED_ACTION
            and strong_endpoint_ok
        )
        check["not_manipulated_endpoint_rescue"] = (
            not_manipulated_endpoint_rescue
        )

        rescue = _can_occlusion_rescue(
            end_obj,
            track,
            action,
        )

        check["occlusion_rescue"] = (
            rescue
        )

        check["stage2_holding_above_cue"] = (
            stage2_holding_cue
        )

        object_success = (
            (endpoint_ok and track_ok)
            or already_satisfied_rescue
            or not_manipulated_endpoint_rescue
            or rescue
            or hovering_wrist_rescue
        )

        check["object_success"] = (
            object_success
        )

        check["phantom_mat_target"] = (
            _is_phantom_mat_tablecloth_target(
                _str(check.get("appearance"))
            )
        )

        check["phantom_mat_rescue_eligible"] = (
            _phantom_mat_target_eligible(
                check,
                action,
                multi_target,
            )
        )

        if not object_success:

            if not endpoint_ok:
                check["failure_reason"] = (
                    "endpoint_not_satisfied"
                )

            elif not track_ok:
                check["failure_reason"] = (
                    "trajectory_not_supported"
                )

        debug["object_checks"].append(
            check
        )

    # IMPORTANT:
    # Never return success from inside the object loop.
    # Evaluate the complete required target set.

    if len(
        debug["object_checks"]
    ) != len(required):

        debug["resolution_mode"] = (
            "incomplete_target_accounting"
        )

        return (
            0,
            "required_target_accounting_incomplete",
            debug,
        )

    failed = [
        check
        for check in debug[
            "object_checks"
        ]
        if not check[
            "object_success"
        ]
    ]

    if failed:

        failed_ids = [
            check["object_id"]
            for check in failed
        ]

        debug["failed_target_ids"] = (
            failed_ids
        )

        debug["all_required_targets_success"] = False

        if _try_phantom_mat_episode_rescue(
            debug["object_checks"],
            action,
            multi_target,
        ):
            debug["phantom_mat_rescue_applied"] = True
            debug["all_required_targets_success"] = True
            debug["resolution_mode"] = (
                "phantom_mat_tablecloth_rescue"
            )

            return (
                1,
                "phantom_mat_tablecloth_rescue",
                debug,
            )

        debug["phantom_mat_rescue_applied"] = False

        debug["resolution_mode"] = (
            "all_targets_not_satisfied"
        )

        return (
            0,
            "not_all_required_targets_satisfy_goal",
            debug,
        )

    debug["all_required_targets_success"] = True

    debug["resolution_mode"] = (
        "all_targets_satisfied"
    )

    return (
        1,
        "all_required_targets_satisfy_goal",
        debug,
    )
