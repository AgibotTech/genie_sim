"""Fold towel schema constants."""

from __future__ import annotations

INITIAL_FOLD_STATES = frozenset(
    {
        "spread_flat",
        "partially_folded",
        "crumpled",
        "folded_layered",
        "unclear",
    }
)

FINAL_FOLD_STATES = frozenset(
    {
        "spread_flat",
        "crumpled",
        "folded_layered",
        "unclear",
    }
)

FOLD_ACTION_TYPES = frozenset(
    {
        "fold",
        "fold_attempt",
        "lift_only",
        "slide_only",
        "crumple_or_scrunch",
        "other",
        "unclear",
    }
)

FOLD_PROGRESS_STATES = frozenset(
    {
        "none",
        "in_progress",
        "completed",
        "reopened",
        "unclear",
    }
)

COMPLETION_STATUSES = frozenset(
    {
        "success",
        "failure",
        "uncertain",
    }
)

YES_NO_UNKNOWN = frozenset({"yes", "no", "unknown"})

ENDPOINT_FOLD_EVIDENCE = frozenset(
    {
        "supports_fold",
        "contradicts_fold",
        "uncertain",
    }
)

FINAL_SHAPES = frozenset(
    {
        "square",
        "compact_rectangle",
        "long_strip",
        "triangle",
        "irregular",
        "unclear",
    }
)

FOOTPRINT_RATIO_CATEGORIES = frozenset(
    {
        "near_original",
        "around_half",
        "around_quarter_or_smaller",
        "unclear",
    }
)

BENCHMARK_COMPLETION_STATES = frozenset(
    {
        "complete",
        "partial",
        "incorrect",
        "unclear",
    }
)

ENDPOINT_COMPLETION_EVIDENCE = frozenset(
    {
        "supports_complete",
        "supports_partial",
        "contradicts_complete",
        "uncertain",
    }
)

OBSERVED_FOLD_EVENTS = frozenset(
    {
        "none",
        "one_or_more",
        "multiple",
        "unclear",
    }
)

WM_COLLAPSE_LEVELS = frozenset({"none", "mild", "severe"})

FOOTPRINT_CHANGE = frozenset(
    {"smaller", "similar", "larger", "unknown"}
)

REGULARITY = frozenset(
    {"roughly_regular", "irregular", "unknown"}
)

CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})

# Stage4 wrist recheck reserved but disabled in fold_towel_release.
FOLD_WRIST_RECHECK_ENABLED = False
