"""Top-level task family router for multi-flow evaluator.

Order: bread takeout → fold towel → fold cloth → doll put/take.
Bread is matched first so toaster TAKE is not stolen by generic TAKE.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class TaskFamily(str, Enum):
    DOLL_INOUT = "doll_inout"
    BREAD_TAKEOUT = "bread_takeout"
    FOLD_TOWEL = "fold_towel"
    FOLD_CLOTH = "fold_cloth"


class UnsupportedTaskError(ValueError):
    """Instruction does not match any supported task family."""


def classify_task_family(instruction: str) -> TaskFamily:
    """Return task family or raise UnsupportedTaskError."""
    text = (instruction or "").strip()
    if not text:
        raise UnsupportedTaskError("empty_instruction")

    # Local import avoids circular import with flow judges.
    from geniesim_dream_evaluation.flows.bread_takeout.semantics import is_supported_bread_takeout_instruction
    from geniesim_dream_evaluation.flows.doll_inout.semantics import instruction_action
    from geniesim_dream_evaluation.flows.fold_towel.semantics import is_supported_fold_towel_instruction
    from geniesim_dream_evaluation.flows.fold_cloth.semantics import is_supported_fold_cloth_instruction

    if is_supported_bread_takeout_instruction(text):
        return TaskFamily.BREAD_TAKEOUT

    low = text.lower()
    # Bread+toaster that is not a supported takeout (e.g. put bread into toaster)
    # must not fall through to doll PUT via generic put/place verbs.
    if ("bread" in low or "toast" in low) and "toaster" in low:
        raise UnsupportedTaskError(f"unsupported_bread_instruction: {text!r}")

    if is_supported_fold_towel_instruction(text):
        return TaskFamily.FOLD_TOWEL

    if is_supported_fold_cloth_instruction(text):
        return TaskFamily.FOLD_CLOTH

    action = instruction_action(text)
    if action in ("put", "take"):
        return TaskFamily.DOLL_INOUT

    raise UnsupportedTaskError(f"unsupported_instruction: {text!r}")


def try_classify_task_family(instruction: str) -> Optional[TaskFamily]:
    try:
        return classify_task_family(instruction)
    except UnsupportedTaskError:
        return None
