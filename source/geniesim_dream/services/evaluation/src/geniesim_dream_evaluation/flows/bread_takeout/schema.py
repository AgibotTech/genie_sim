"""Bread take-out constants (independent from doll schema)."""

from __future__ import annotations

SOURCE_REMAINING_STATES = frozenset({"none", "present", "unknown"})
SOURCE_SLOT_STATES = frozenset({"empty", "occupied", "partial", "unknown"})

EXTRACTION_SUCCESS = frozenset({"removed"})
EXTRACTION_STATES = frozenset({"removed", "not_removed", "unknown"})

DESTINATION_SUCCESS = frozenset({"placed_on_plate"})
DESTINATION_FAILURES = frozenset(
    {
        "dropped_on_table",
        "stuck_on_toaster",
        "still_held",
        "moved_elsewhere",
        "still_in_toaster",
        "not_on_plate",
    }
)
DESTINATION_STATES = DESTINATION_SUCCESS | DESTINATION_FAILURES | {"unknown"}

SUCCESS_COMPATIBLE_TRAJECTORY_INTEGRITY = frozenset(
    {
        "valid",
        "fused_after_plate_entry",
    }
)

BREAD_WRIST_RECHECK_ENABLED = False

OCCLUDED_VISIBILITY = frozenset(
    {
        "partially_occluded",
        "fully_occluded",
        "occluded",
        "partial",
    }
)

ENDPOINT_EVIDENCE_SOURCES = frozenset(
    {
        "end_top",
        "end_left_arm",
        "end_right_arm",
        "end_multi_view",
        "multi_view",
    }
)

# Stage2 destination endpoint
DESTINATION_ENDPOINT_STATUSES = frozenset(
    {"supported", "contradicted", "uncertain"}
)
OFF_PLATE_STATES = frozenset({"present", "none", "unknown"})
PLATE_BREAD_PRESENCE = frozenset({"present", "none", "unknown"})
BREAD_ARRIVAL_EVIDENCE = frozenset({"yes", "no", "unknown"})
PLATE_VISIBILITY = frozenset(
    {"clear", "partial", "not_visible", "unknown"}
)
INITIAL_BREAD_PRESENCE = frozenset({"none", "present", "unknown"})
CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
