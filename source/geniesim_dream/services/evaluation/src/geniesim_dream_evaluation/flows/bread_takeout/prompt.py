"""All Stage1/2/3 prompts for bread take-out V2. No doll prompt imports."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from .semantics import bread_semantics_block, plate_semantics_block


_STAGE1 = """\
You are Stage1 for a BREAD TAKE-OUT FROM TOASTER evaluator.

Instruction: {instruction}

{bread_semantics}

{plate_semantics}

INPUT: START_TOP only. Do NOT judge task success.

TASK:
1. Identify the source toaster.
2. Enumerate every physical bread/toast slice initially inserted or
   meaningfully engaged in the toaster. Assign stable IDs bread_1, bread_2, ...
3. Identify ALL visible valid plates as plate_1, plate_2, ... Do NOT choose
   one privileged target plate.
4. Do NOT include bread already outside the toaster (already on a plate,
   table, or elsewhere) in required targets.

REQUIRED TARGET SET:
Every bread slice initially inserted or meaningfully engaged in the toaster
is a required target.

Bread already on a plate, table, or elsewhere outside the toaster at START
is not a required target.

PLATE / DESTINATION INVENTORY:
Identify every valid physical plate visible in the scene.
There may be multiple plates.
Any valid plate is an acceptable destination for the bread.
Different bread slices may be placed on different plates.
Plate color is irrelevant.
Assign stable IDs: plate_1, plate_2, ...
Record appearance, coarse spatial location relative to the toaster, and
whether bread is already present on the plate at START.
Bread already present on a plate at START is NOT a required target.

MULTI-PLATE RULE:
Any valid physical plate is an acceptable destination.
Plate color is irrelevant.
Different target slices may use different plates.

2D/GRAPHIC RULE:
A printed/painted/photo bread image is not a physical target.

Color cue: target bread is typically yellow / light-yellow / beige /
lightly browned sliced bread. Color alone is not sufficient.

Return ONLY one JSON object:
{{
  "start_description": "...",
  "task_family": "bread_takeout",
  "target_quantifier": "all_initial_toaster_bread",
  "source_toaster": {{
    "id": "toaster_1",
    "appearance": "...",
    "visibility": "clear|partial|unknown"
  }},
  "target_candidates": [
    {{
      "id": "bread_1",
      "appearance": "yellow sliced bread",
      "is_bread_like": true,
      "identity_confidence": "high|medium|low",
      "initial_relation_to_toaster": "inserted|partially_inserted|outside|unknown",
      "initially_in_toaster": true,
      "source_slot_id": "slot_1|slot_2|unknown"
    }}
  ],
  "plate_candidates": [
    {{
      "id": "plate_1",
      "appearance": "white round ceramic plate",
      "is_plate_like": true,
      "identity_confidence": "high|medium|low",
      "spatial_anchor": "right_of_toaster|left_of_toaster|in_front_of_toaster|behind_toaster|unknown",
      "initial_bread_presence": "none|present|unknown"
    }}
  ],
  "initial_toaster_occupancy": "nonempty|empty|unknown",
  "initial_bread_count_estimate": 0,
  "initial_bread_count_confidence": "high|medium|low",
  "scene_readable": true
}}

initial_bread_count_estimate is debug metadata only.
"""

_STAGE2 = """\
You are Stage2 (SOURCE + DESTINATION ENDPOINT AUTHORITY) for a BREAD
TAKE-OUT FROM TOASTER evaluator.

Instruction: {instruction}

{bread_semantics}

{plate_semantics}

INPUTS: START_TOP, END_TOP, END_TRIPLET, and parsed START_STATE.

START_STATE:
{start_state}

Answer TWO authoritative questions:
A. At END, does any required bread remain inserted or partially inserted
   in the source toaster?
B. At END, is there evidence that bread reached any valid plate, OR is
   there explicit target bread remaining off all plates?

DO NOT COUNT BREAD PIECES ON PLATES.
Generated bread slices may:
- merge
- collapse
- overlap
- deform
- become one blob

Therefore two START bread slices do NOT require two separately visible
bread slices on plates at END.
Judge whether the destination condition is supported or contradicted,
not the exact number of visible bread instances.

SOURCE TOASTER:
remaining_target_bread = present
if any meaningful portion of a target bread slice remains engaged in a
toaster slot.

A slice resting on top of / against the toaster exterior but no longer in
a slot means the source may be cleared, but destination may still fail
if that slice is off-plate.

A slice held above the toaster likewise may mean source cleared, but does
not imply task success.

Slot semantics:
empty    = no meaningful target bread remains in the slot
occupied = target bread remains inserted
partial  = target bread is partly extracted but a meaningful portion remains engaged
unknown  = occlusion/generation artifact prevents judgement

DESTINATION PLATES:
For each valid plate from START_STATE (and any newly visible valid plate),
report end_bread_presence and bread_arrival_evidence.
If a plate already had bread at START (initial_bread_presence=present),
do NOT treat END bread alone as proof of transfer; compare START vs END.
If comparison is unclear, prefer destination_endpoint=uncertain rather
than contradicted.

destination_endpoint.status:
- supported: clear evidence bread is on any valid plate (or START->END
  arrival), and no explicit off-plate failure
- contradicted: explicit bread on table / toaster exterior / gripper /
  other non-plate surface
- uncertain: bread/plate collapsed, fused, deformed, or unrecognizable,
  with NO explicit failure evidence. uncertain is NOT failure.

off_plate_target_bread.state = present
when END clearly shows target bread on table, toaster exterior, gripper,
or another non-plate surface.

Do NOT output per-plate bread counts. Do NOT output task answer.

Return ONLY one JSON object:
{{
  "end_description": "...",
  "source_toaster_id": "toaster_1",
  "source_toaster_visibility": "clear|partially_occluded|fully_occluded|unknown",
  "remaining_target_bread": "none|present|unknown",
  "remaining_bread_count_estimate": 0,
  "slot_states": [
    {{
      "slot_id": "slot_1",
      "state": "empty|occupied|partial|unknown",
      "confidence": "high|medium|low"
    }}
  ],
  "source_relation_confidence": "high|medium|low",
  "source_relation_source": "end_top|end_left_arm|end_right_arm|multi_view",
  "source_endpoint_candidate": "cleared|not_cleared|uncertain",
  "source_endpoint_reason": "...",
  "destination_plates": [
    {{
      "plate_id": "plate_1",
      "identity_consistent": true,
      "visibility": "clear|partial|not_visible|unknown",
      "end_bread_presence": "present|none|unknown",
      "bread_arrival_evidence": "yes|no|unknown",
      "reason": "..."
    }}
  ],
  "off_plate_target_bread": {{
    "state": "present|none|unknown",
    "locations": ["table|toaster_exterior|gripper"],
    "confidence": "high|medium|low",
    "reason": "..."
  }},
  "destination_endpoint": {{
    "status": "supported|contradicted|uncertain",
    "confidence": "high|medium|low",
    "supporting_plate_ids": [],
    "reason": "..."
  }},
  "last_frame_readable": true
}}
"""

_STAGE3 = """\
You are Stage3 (PER-TARGET TRANSFER VERIFIER) for a BREAD TAKE-OUT FROM
TOASTER evaluator. You do NOT give the final task answer.

Instruction: {instruction}

{bread_semantics}

{plate_semantics}

START_STATE:
{start_state}

END_STATE (source + destination endpoints):
{end_state}

INPUTS: MIDDLE FRAME 1/2/3 triplets and END_TRIPLET.

For EVERY required START target ID (bread initially in toaster), answer:
A. Was it fully removed from the toaster?
B. Was it deposited onto any valid plate?

MULTI-PLATE RULE:
There may be multiple plates of arbitrary color.
Any valid physical plate is acceptable.
Different bread slices may be placed on different plates.
Do not require a specific plate ID.
Plate color is irrelevant.

FUSION RULE:
Once bread_i is clearly deposited onto a valid plate, mark
plate_entry_confirmed=true.
After that moment, loss of individual identity because generated bread
slices merge/fuse with other bread on the plate is NOT failure.
Do not later require bread_i to remain separately countable.
Do not infer failure merely because two successful slices look like one
blob at END.

WRONG-DESTINATION RULE:
Extraction alone is not completion.
If bread_i is extracted but ends up on the table, on/caught against the
toaster, still held by the gripper, on another non-plate surface, or at
an unknown destination, then bread_i is not complete.
dropped_on_table / stuck_on_toaster / still_held are failures.

LATER-UNDO RULE:
If bread_i is clearly placed on a plate and later clearly falls/leaves
that plate, later_undo=true and the target fails.
Identity loss after plate fusion is NOT evidence of later_undo.
Only explicit visual evidence of leaving the plate should set later_undo=true.

Try to include every required START target in object_tracks[].
If generation collapse / fusion makes a target untrackable, omit it;
the resolver may ignore missing tracks. Do not invent a track.

Optional source endpoint override (END evidence only):
If Stage2 source state is weak/occluded/unknown, you may propose an
override of remaining_target_bread using END views only. Never use MID
frames to override a clear Stage2 source endpoint.

Return ONLY one JSON object:
{{
  "object_tracks": [
    {{
      "object_id": "bread_1",
      "appearance": "yellow sliced bread",
      "is_bread_like": true,
      "target_match": true,
      "target_match_confidence": "high|medium|low",
      "originated_from_toaster": true,
      "source_origin_confidence": "high|medium|low",
      "events": [
        "grasp",
        "lift_from_toaster",
        "clear_toaster_slot",
        "transport_to_plate",
        "deposit_on_plate"
      ],
      "extraction_status": "removed|not_removed|unknown",
      "plate_entry_confirmed": true,
      "destination_outcome": "placed_on_plate|dropped_on_table|stuck_on_toaster|still_held|moved_elsewhere|still_in_toaster|not_on_plate|unknown",
      "destination_plate_id": "plate_2|untracked_plate|null",
      "destination_confidence": "high|medium|low",
      "merge_after_plate_entry": false,
      "identity_lost_after_plate_entry": false,
      "trajectory_integrity": "valid|fused_after_plate_entry|vanished_before_plate_entry|unclear",
      "later_undo": false
    }}
  ],
  "source_endpoint_override_request": {{
    "requested": false,
    "proposed_remaining_target_bread": "none|present|unknown",
    "evidence_source": "end_top|end_left_arm|end_right_arm|end_multi_view|null",
    "reason": null
  }},
  "trajectory_summary": "short factual per-target summary"
}}
"""


def _state_to_text(state: Union[str, Dict[str, Any], None]) -> str:
    if state is None:
        return "(unavailable)"
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def build_start_content(
    instruction: str,
    start_frame: Any,
) -> List[Dict[str, Any]]:
    prompt = _STAGE1.format(
        instruction=instruction or "",
        bread_semantics=bread_semantics_block(),
        plate_semantics=plate_semantics_block(),
    )
    return [
        {"type": "text", "text": prompt},
        {"type": "image", "image": start_frame},
    ]


def build_end_content(
    instruction: str,
    start_frame: Any,
    end_frame: Any,
    end_triplet: Any,
    start_state: Union[str, Dict[str, Any], None],
) -> List[Dict[str, Any]]:
    prompt = _STAGE2.format(
        instruction=instruction or "",
        bread_semantics=bread_semantics_block(),
        plate_semantics=plate_semantics_block(),
        start_state=_state_to_text(start_state),
    )
    return [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "START_TOP:"},
        {"type": "image", "image": start_frame},
        {"type": "text", "text": "END_TOP:"},
        {"type": "image", "image": end_frame},
        {"type": "text", "text": "END_TRIPLET (same final time step):"},
        {"type": "image", "image": end_triplet},
    ]


def build_verify_content(
    instruction: str,
    mid_triplets: List[Any],
    end_triplet: Any,
    start_state: Union[str, Dict[str, Any], None],
    end_state: Union[str, Dict[str, Any], None],
) -> List[Dict[str, Any]]:
    if len(mid_triplets) < 3:
        raise ValueError(
            f"need 3 mid triplets for bread takeout v2, got {len(mid_triplets)}"
        )
    mid1, mid2, mid3 = mid_triplets[0], mid_triplets[1], mid_triplets[2]
    prompt = _STAGE3.format(
        instruction=instruction or "",
        bread_semantics=bread_semantics_block(),
        plate_semantics=plate_semantics_block(),
        start_state=_state_to_text(start_state),
        end_state=_state_to_text(end_state),
    )
    return [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "MIDDLE FRAME 1 (triplet):"},
        {"type": "image", "image": mid1},
        {"type": "text", "text": "MIDDLE FRAME 2 (triplet):"},
        {"type": "image", "image": mid2},
        {"type": "text", "text": "MIDDLE FRAME 3 (triplet):"},
        {"type": "image", "image": mid3},
        {"type": "text", "text": "END_TRIPLET (endpoint cross-check only):"},
        {"type": "image", "image": end_triplet},
    ]
