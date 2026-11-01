"""Shared enums / constants for interleave v5 schemas."""

# PUT endpoint success.
# Keep v4 semantics unchanged:
# a large doll may count as successful when its main body is mostly inside.
SUCCESS_RELATIONS = frozenset({
    "inside",
    "mostly_inside",
})

FAIL_RELATIONS = frozenset({
    "rim_or_minor_inside",
    "hovering",
    "outside",
})

UNKNOWN_RELATIONS = frozenset({
    "unknown",
})

# TAKE endpoint success.
TAKE_SUCCESS_RELATIONS = frozenset({
    "outside",
})

OCCLUDED_VISIBILITY = frozenset({
    "fully_occluded",
    "partially_occluded",
})

RESCUE_OCCLUSION_SOURCES = frozenset({
    "robot_arm",
    "gripper",
    "container_wall",
    "other_object",
    "other",
})

PUT_SUPPORTING_ACTIONS = frozenset({
    "put_completed",
    "put_attempt",
})

TAKE_SUPPORTING_ACTIONS = frozenset({
    "take_completed",
    "take_attempt",
})

# Stage1 zero-target recovery may only add Stage3-only objects with
# positive manipulation evidence.
RECOVERY_ACTIONS = frozenset({
    "put_completed",
    "put_attempt",
    "take_completed",
    "take_attempt",
})

# Stage 3 may report this for a required target that never moved.
NOT_MANIPULATED_ACTION = "not_manipulated"

ENDPOINT_EVIDENCE_PREFIX = "end_"

# ============================================================
# v5a: dedicated FINAL wrist-view endpoint recheck
# ============================================================

# These ambiguous/failure-like relations are worth rechecking
# with enlarged wrist views.
#
# `outside` is intentionally NOT included:
# a clear Stage2 outside judgement must not trigger Stage4
# merely because its relation is outside. It may still trigger
# through independent uncertainty cues such as low confidence
# or endpoint occlusion.
PUT_WRIST_RECHECK_RELATIONS = frozenset({
    "hovering",
    "rim_or_minor_inside",
    "unknown",
})

TAKE_WRIST_RECHECK_RELATIONS = frozenset({
    "hovering",
    "rim_or_minor_inside",
    "unknown",
})

TAKE_WRIST_RECHECK_ENABLED = False

# These visibility states indicate that the global END judgement
# may have been affected by occlusion / disappearance.
PUT_WRIST_RECHECK_VISIBILITY = frozenset({
    "partially_occluded",
    "fully_occluded",
    "vanished",
})

# Exact PUT Stage2 states that should never trigger a wrist recheck,
# even when another generic trigger (such as low confidence) is present.
PUT_WRIST_RECHECK_EXCLUDED_STATES = frozenset({
    ("inside", "partially_occluded"),
})

TAKE_WRIST_RECHECK_VISIBILITY = frozenset({
    "partially_occluded",
    "fully_occluded",
    "vanished",
})

# A positive wrist correction is deliberately conservative:
# converting a failure-like Stage2 endpoint into success requires
# HIGH wrist confidence.
WRIST_POSITIVE_OVERRIDE_CONFIDENCE = frozenset({
    "high",
})

# A dedicated wrist view may confirm a failure with medium/high
# confidence when Stage2 itself was ambiguous.
WRIST_NEGATIVE_OVERRIDE_CONFIDENCE = frozenset({
    "high",
    "medium",
})

WRIST_KNOWN_RELATIONS = frozenset({
    "inside",
    "mostly_inside",
    "rim_or_minor_inside",
    "hovering",
    "outside",
})

# Printed-mat / tablecloth pattern misread as a second doll target.
# Explicit: plush + (bear|dog) + blue bow/bowtie (no "large" required).
PHANTOM_MAT_REQUIRED_TOKENS = ("plush",)
PHANTOM_MAT_ANIMAL_TOKENS = ("bear", "dog")
# Any one is enough; "blue bow" also covers "blue bowtie" / "blue bow tie".
PHANTOM_MAT_BOW_TOKENS = (
    "blue bowtie",
    "blue bow tie",
    "blue bow",
)
