"""Shared task semantics + deterministic normalizers for interleave v5."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_DOLL_SEMANTICS = """
TASK OBJECT SEMANTICS — IMPORTANT

For this benchmark, "doll" is a BROAD semantic category.

Treat the following as DOLL-LIKE when they are recognizable soft
toy/character objects:
- doll
- plush toy / plushie
- stuffed toy / stuffed animal
- plush animal or character
- plush dinosaur, dog, wolf, pig, bear, whale, bird, rabbit, etc.
- soft/fluffy character toy with a recognizable body

Examples that SHOULD normally be doll-like:
- "green dinosaur plush"
- "blue plush toy"
- "blue dinosaur plush"
- "pink spherical plush toy"
- "pink stuffed toy"
- "pink and white plush toy" / pink-and-white plush rabbit or animal
- "plush wolf" / plush husky / plush dog
- "stuffed animal"

Do NOT require the noun to literally be "doll".

Normally NOT doll-like:
- loose cloth, rag, towel, mat, sheet
- bag
- flower
- fruit
- cup
- box/container/lid
- an unrecognizable material blob with no stable toy identity

Consistency rule:
If you describe an object yourself as "plush", "stuffed", or a
recognizable soft animal/character toy, you should normally set
is_doll_like=true.

If world-model deformation makes identity uncertain, use
unknown/low confidence instead of confidently declaring a
recognizable plush toy to be non-doll.
"""


_PUT_SPATIAL_RULES = """
PUT spatial states:

inside:
    The doll is clearly contained by the box.

mostly_inside:
    The central body / majority of the doll is inside the box.
    Some limbs/head may protrude because the doll is large.
    This CAN count as successful placement.

rim_or_minor_inside:
    Only a small portion is inside, or the doll is perched/stuck
    on the rim. This does NOT count as successful placement.

hovering:
    Held above the box without meaningful insertion.

outside:
    Clearly outside.

unknown:
    Cannot determine because of occlusion or severe ambiguity.

IMPORTANT:
A robot gripper still touching or holding the doll does NOT by
itself mean failure.

Judge spatial placement, not whether the robot arm has fully
retracted.

A large doll can be successfully placed even if it does not fit
perfectly, provided its main body is clearly inside the container.

FRAGMENTATION RULE:

If one original doll-like object splits into multiple visible fragments,
all fragments still belong to the SAME target object.

For PUT:
the fragmented target satisfies the spatial goal ONLY IF every
identifiable fragment is inside or mostly_inside the destination container.

Examples:

head inside + body outside
=> NOT successful

head inside + body inside
=> may count as successful

one fragment inside + another fragment unknown/missing
=> NOT successful

Do not judge only the largest or easiest-to-see fragment.

Fragmentation itself is not automatically a failure if ALL fragments
are accounted for and ALL fragments satisfy the spatial goal.
"""


_DOLL_POSITIVE_TERMS = (
    "doll",
    "plush",
    "plushie",
    "stuffed",
    "soft toy",
    "pink spherical",
    "toy animal",
    "toy character",
)

_STRONG_NON_DOLL_TERMS = (
    "cloth",
    "rag",
    "towel",
    "mat",
    "sheet",
    "bag",
    "flower",
    "bouquet",
    "fruit",
    "apple",
    "cup",
    "mug",
    # "box",
    # "container",
    "lid",
)


def doll_semantics_block() -> str:
    return _DOLL_SEMANTICS.strip()


def put_spatial_rules_block() -> str:
    return _PUT_SPATIAL_RULES.strip()



_TAKE_SPATIAL_RULES = """
TAKE spatial states:

outside:
    The original target body has been fully removed from the container
    volume. It may still be held by the gripper, may be above the
    container, or may be beside the container. If the target body is
    fully clear of the container/rim, this is outside and counts as
    successful extraction.

rim_or_minor_inside:
    The target is only partially extracted / caught on the rim, or a
    small but meaningful part remains within the container volume.
    This does not count as successful TAKE.

mostly_inside:
    Most of the target body remains inside the container.
    This does not count as successful TAKE.

inside:
    The target remains contained in the container. Or the target(s) is/are surrounded by the container walls.
    This does not count as successful TAKE.

hovering:
    For TAKE, do not use hovering merely because the gripper holds the
    target above the box. If the target body is fully clear of the
    container, classify it as outside.
    Use hovering only when the spatial relation is genuinely ambiguous
    and it cannot be established whether extraction cleared the
    container volume.

unknown:
    Endpoint relation cannot be determined because of occlusion,
    deformation, identity uncertainty, or insufficient depth evidence.

IMPORTANT:
Being held above the box after extraction is compatible with successful
TAKE.

gripper still holding target
    ≠ failure

target above container
    ≠ automatically hovering

fully extracted body above container
    = outside

FRAGMENTATION RULE:

If one original doll-like object splits into multiple visible fragments,
all fragments still belong to the SAME target object.

For TAKE:
the fragmented target satisfies the spatial goal ONLY IF every
identifiable fragment is outside the source container.

Examples:

head outside + body inside
=> NOT successful

head outside + body outside
=> may count as successful

one fragment outside + another fragment unknown/missing
=> NOT successful

Do not judge only the largest or easiest-to-see fragment.

Fragmentation itself is not automatically a failure if ALL fragments
are accounted for and ALL fragments satisfy the spatial goal.
"""


def take_spatial_rules_block() -> str:
    return _TAKE_SPATIAL_RULES.strip()


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
    """Match a semantic term without matching it inside another word."""

    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
        text,
    ) is not None


def normalize_is_doll_like(
    appearance: Optional[str],
    model_value: Any,
) -> Optional[bool]:
    """v4 doll decision boundary: strong non-doll > positive > model."""

    text = (appearance or "").strip().lower()
    model_bool = _as_bool(model_value)

    if text:
        if any(
            _contains_whole_term(text, term)
            for term in _STRONG_NON_DOLL_TERMS
        ):
            return False

        if any(
            _contains_whole_term(text, term)
            for term in _DOLL_POSITIVE_TERMS
        ):
            return True

    return model_bool


def _instruction_is_specific_target(instruction: str) -> bool:
    """
    Detect a simple explicitly qualified target such as:
        Put the blue doll into the box.
        Take the pink doll out of the box.

    Current benchmark uses generic "the doll", so generic instructions
    should resolve to all_matching_instances.
    """

    text = (instruction or "").strip().lower()

    # "the doll" / "a doll" is generic.
    if re.search(r"\b(?:the|a)\s+doll\b", text):
        return False

    # One or more qualifiers before "doll":
    # "the blue doll", "the light blue doll", etc.
    if re.search(
        r"\b(?:the|a)\s+(?!doll\b)"
        r"(?:[a-z0-9_-]+\s+){1,3}doll\b",
        text,
    ):
        return True

    return False


def parse_target_quantifier(
    instruction: str,
    model_value: Any = None,
) -> str:
    """
    v5 semantics:

    Generic "Put/Take the doll ..." means ALL matching target
    instances in the START scene are required.

    Do not allow an old v4 model output such as
    any_matching_instance to restore existential semantics.
    """

    if _instruction_is_specific_target(instruction):
        return "specific_instance"

    return "all_matching_instances"


def instruction_action(instruction: str) -> Optional[str]:
    """Return 'take', 'put', or None from the instruction text."""

    text = (instruction or "").strip().lower()

    if not text:
        return None

    def has_word(word: str) -> bool:
        return (
            re.search(
                rf"\b{re.escape(word)}\b",
                text,
            )
            is not None
        )

    take = any(
        has_word(w)
        for w in ("take", "remove")
    )

    put = any(
        has_word(w)
        for w in ("put", "place", "insert")
    )

    if take and not put:
        return "take"

    if put and not take:
        return "put"

    return None


def _normalize_object_fields(
    obj: Dict[str, Any],
) -> Dict[str, Any]:

    appearance = (
        obj.get("appearance_at_end")
        or obj.get("appearance")
        or ""
    )

    if isinstance(appearance, str):
        obj["is_doll_like"] = normalize_is_doll_like(
            appearance,
            obj.get("is_doll_like"),
        )

    inv = _as_bool(
        obj.get("task_invalidating_integrity")
    )

    if inv is not None:
        obj["task_invalidating_integrity"] = inv

    later = _as_bool(
        obj.get("later_undo")
    )

    if later is not None:
        obj["later_undo"] = later

    tm = _as_bool(
        obj.get("target_match")
    )

    if tm is not None:
        obj["target_match"] = tm

    fragments_accounted = _as_bool(
        obj.get("all_fragments_accounted_for")
    )

    if fragments_accounted is not None:
        obj["all_fragments_accounted_for"] = (
            fragments_accounted
        )

    fragments_goal = _as_bool(
        obj.get("all_fragments_satisfy_goal")
    )

    if fragments_goal is not None:
        obj["all_fragments_satisfy_goal"] = (
            fragments_goal
        )

    return obj


def normalize_start_state(
    start_state: Optional[Dict[str, Any]],
    instruction: str = "",
) -> Optional[Dict[str, Any]]:

    if not isinstance(start_state, dict):
        return start_state

    out = dict(start_state)

    out["target_quantifier"] = (
        parse_target_quantifier(
            instruction,
            out.get("target_quantifier"),
        )
    )

    cands = out.get("target_candidates")

    if isinstance(cands, list):
        fixed: List[Dict[str, Any]] = []

        for c in cands:
            if isinstance(c, dict):
                fixed.append(
                    _normalize_object_fields(
                        dict(c)
                    )
                )

        out["target_candidates"] = fixed

    return out


def normalize_end_state(
    end_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:

    if not isinstance(end_state, dict):
        return end_state

    out = dict(end_state)

    objs = out.get("objects")

    if isinstance(objs, list):
        fixed = []

        for o in objs:
            if isinstance(o, dict):
                fixed.append(
                    _normalize_object_fields(
                        dict(o)
                    )
                )

        out["objects"] = fixed

    # Legacy single-object fallback.
    if "objects" not in out and (
        "final_relation" in out
        or "appearance_at_end" in out
    ):
        legacy = _normalize_object_fields(
            {
                "id": out.get("id") or "obj_1",
                "appearance_at_end": (
                    out.get("appearance_at_end")
                    or out.get("appearance")
                    or out.get("target_object")
                    or ""
                ),
                "is_doll_like": out.get(
                    "is_doll_like"
                ),
                "visibility": (
                    out.get("visibility")
                    or "unknown"
                ),
                "occlusion_source": (
                    out.get("occlusion_source")
                    or "none"
                ),
                "final_relation": (
                    out.get("final_relation")
                    or "unknown"
                ),
                "final_relation_confidence": (
                    out.get(
                        "final_relation_confidence",
                        "low",
                    )
                ),
                "final_relation_source": (
                    out.get(
                        "final_relation_source",
                        "end_top",
                    )
                ),
                "target_integrity": (
                    out.get(
                        "target_integrity",
                        "unknown",
                    )
                ),
                "task_invalidating_integrity": (
                    out.get(
                        "task_invalidating_integrity",
                        False,
                    )
                ),
                "integrity_reason": (
                    out.get(
                        "integrity_reason",
                        "none",
                    )
                ),
                "fragment_count": out.get(
                    "fragment_count"
                ),
                "fragment_relations": out.get(
                    "fragment_relations",
                    [],
                ),
                "all_fragments_accounted_for": (
                    out.get(
                        "all_fragments_accounted_for"
                    )
                ),
                "all_fragments_satisfy_goal": (
                    out.get(
                        "all_fragments_satisfy_goal"
                    )
                ),
            }
        )

        out["objects"] = [legacy]

    return out


def normalize_trajectory_state(
    verify_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:

    if not isinstance(verify_state, dict):
        return verify_state

    out = dict(verify_state)

    tracks = out.get("object_tracks")

    if isinstance(tracks, list):
        fixed = []

        for t in tracks:
            if isinstance(t, dict):
                fixed.append(
                    _normalize_object_fields(
                        dict(t)
                    )
                )

        out["object_tracks"] = fixed

    return out


def normalize_wrist_state(
    wrist_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize dedicated v5a wrist-recheck JSON."""

    if not isinstance(wrist_state, dict):
        return wrist_state

    out = dict(wrist_state)

    raw_objects = out.get("objects")

    if not isinstance(raw_objects, list):
        return out

    fixed: List[Dict[str, Any]] = []

    for raw in raw_objects:

        if not isinstance(raw, dict):
            continue

        obj = dict(raw)

        target_ok = _as_bool(
            obj.get(
                "target_identity_consistent"
            )
        )

        if target_ok is not None:
            obj[
                "target_identity_consistent"
            ] = target_ok

        container_ok = _as_bool(
            obj.get(
                "container_identity_consistent"
            )
        )

        if container_ok is not None:
            obj[
                "container_identity_consistent"
            ] = container_ok

        contact = _as_bool(
            obj.get("gripper_contact")
        )

        if contact is not None:
            obj[
                "gripper_contact"
            ] = contact

        fixed.append(obj)

    out["objects"] = fixed

    return out
