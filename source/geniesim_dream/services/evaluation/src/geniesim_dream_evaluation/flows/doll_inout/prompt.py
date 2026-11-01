"""VLM judge prompts for dream_doll_inout_release.

PUT Stage2/3/4 prompts are renamed from v5a4_hover_res without body edits.
TAKE Stage2/3/4 are additive branches dispatched by instruction_action().
Stage1 remains shared.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

from .semantics import (
    doll_semantics_block,
    instruction_action,
    put_spatial_rules_block,
    take_spatial_rules_block,
)

_CONTAINER_VERBS = ("take", "put", "place", "insert", "remove")
_CONTAINER_NOUNS = ("box", "basket", "container", "bin", "bag", "doll")
_FOLD_TERMS = ("fold", "unfold")


_CONTAINER_START = """You are Stage 1 of a robot task-completion evaluator.

Your ONLY job is to construct an inventory of the INITIAL scene.
DO NOT judge task success and DO NOT predict later motion.

Task instruction:
{instruction}

{doll_semantics}

2D SURFACE GRAPHIC RULE:

A clearly flat printed or painted animal, doll, face, or character graphic
that is part of a mat, tablecloth, carpet, or other supporting/background
surface is NOT a target object.

Only treat such an appearance as a doll-like target when the image supports
that it is a separate physical object rather than part of the surface.

TARGET BINDING RULE:

If the instruction says only "the doll" or "a doll" without a visual
qualifier such as color, type, side, or location, DO NOT choose one
unique doll prematurely.

Treat ALL visually plausible doll-like objects as eligible target
candidates.

Likewise, if the instruction says only "the box" without a qualifier
and multiple box-like containers exist, keep all plausible container
candidates instead of choosing one too early.

Give every relevant object a stable ID.

Use obj_1, obj_2, ... for target candidates and box_1, box_2, ...
for containers.

Later stages should preserve these IDs when possible.

For a generic instruction such as
"Put the doll into the box",
set target_quantifier="all_matching_instances".

This field describes how downstream task completion is resolved.
It does NOT change the visual criteria for whether an object is
doll-like.

Only if the instruction explicitly identifies one unique target,
for example "Put the blue doll into the box", should
target_quantifier be "specific_instance".


The next image is START_TOP.

It is the overhead/head camera view only.
It is NOT a three-view stitch.

Identify:
1. every plausible doll-like target object
2. every plausible box/container
3. stable visual attributes for tracking
4. each target's initial relation to a nearby container

Give stable IDs:
obj_1, obj_2, ...
box_1, box_2, ...

Output EXACTLY one JSON object and nothing else:

{{
  "start_description": "short factual visual description",

  "expected_action": "put|take|other",

  "target_semantic_category": "doll_like|other|unknown",

  "target_quantifier": "all_matching_instances|specific_instance",

  "target_candidates": [
    {{
      "id": "obj_1",
      "appearance": "short stable visual description",
      "is_doll_like": true,
      "identity_confidence": "high|medium|low",
      "initial_relation": "inside|outside|rim|unknown",
      "nearest_container_id": "box_1"
    }}
  ],

  "container_candidates": [
    {{
      "id": "box_1",
      "appearance": "pink rectangular plastic box",
      "is_box_like": true
    }}
  ],

  "scene_readable": true
}}

Do NOT output success/failure.
Do NOT output answer=0 or answer=1.
"""



_CONTAINER_END_PUT = """You are Stage 2, the ENDPOINT AUTHORITY of a robot task-completion evaluator.

Task instruction:
{instruction}

{doll_semantics}

{put_spatial_rules}

A previous model produced the following START_STATE:

{start_state}

IMPORTANT:

START_STATE defines the REQUIRED TARGET SET.

For target_quantifier="all_matching_instances", EVERY doll-like target
listed in START_STATE.target_candidates[] must be independently checked
at END.

START_STATE is still a provisional visual observation.
Verify object identity against START_TOP, but do NOT silently remove a
required target merely because it was untouched or difficult to locate.


You will receive:

1. START_TOP
2. END_TOP
3. END_TRIPLET

END_TRIPLET is the same final time step:

HEAD/TOP | LEFT ARM | RIGHT ARM


Your job is the FINAL spatial state at END.

DO NOT analyze the whole trajectory.
DO NOT use a middle-frame event as proof of final completion.


MULTI-TARGET COMPLETION RULE — CRITICAL:

For target_quantifier="all_matching_instances":

EVERY required target must satisfy the final task condition.

For PUT:

obj_1 = inside
obj_2 = inside
=> endpoint_candidate may be success

obj_1 = inside
obj_2 = mostly_inside
=> endpoint_candidate may be success

obj_1 = inside
obj_2 = outside
=> failure

obj_1 = inside
obj_2 = hovering
=> failure

obj_1 = inside
obj_2 = rim_or_minor_inside
=> failure

obj_1 = inside
obj_2 = unknown
=> uncertain, NEVER success

One successful target can NEVER compensate for another required
target that is outside, hovering, unresolved, missing, or failed.

For TAKE, apply the symmetric rule:
EVERY required target must be outside the container.


TARGET COVERAGE RULE:

For EVERY required target ID from START_STATE.target_candidates[],
produce one corresponding entry in objects[].

Preserve the same object ID whenever possible.

If a target is not visible at END, DO NOT omit it.

Instead output that target with an appropriate state such as:
- visibility = fully_occluded
- visibility = vanished
- final_relation = unknown

A required target disappearing from objects[] is NOT allowed.


VIEW RULE:

END_TOP is primary global spatial evidence.

END_TRIPLET is auxiliary evidence for:
- insertion depth
- box wall occlusion
- robot-arm occlusion
- gripper contact

For END_TRIPLET:
left third = HEAD/TOP
middle third = LEFT ARM
right third = RIGHT ARM


OCCLUSION VS VANISHING:

occluded:
a plausible physical occluder explains why the target cannot be seen,
such as robot arm, gripper, box wall, or another object.

vanished:
the target should be visible, no physical occluder explains its
disappearance, and it cannot be located in any END view.


SEMANTIC IDENTITY RULE:

If START_STATE identifies obj_X as doll-like and the END object is the
same tracked obj_X, preserve its semantic identity when object continuity
remains visually reliable.

If deformation makes identity genuinely uncertain, use the available
unknown / low-confidence fields instead of inventing a different object.


FRAGMENTATION RULE:

If one original target splits into multiple fragments, keep all fragments
under the SAME original object ID.

Do NOT turn fragments into new independent target IDs.

For PUT:

all fragments inside/mostly_inside
=> fragmented target may satisfy the spatial goal

head inside + body outside
=> target does NOT satisfy the goal

one fragment inside + one fragment unknown/missing
=> target does NOT satisfy the goal

For TAKE:

all fragments must be outside.

Fragmentation itself is not automatically a task failure.
The important question is whether ALL fragments of that original target
satisfy the spatial goal.


============================================================
INTEGRITY FIELD SEMANTICS — CRITICAL
============================================================

"target_integrity" and "task_invalidating_integrity" describe
the PHYSICAL / IDENTITY INTEGRITY of the target object.

They do NOT describe whether the final PUT/TAKE spatial relation
satisfies the task.

Set task_invalidating_integrity=true ONLY when a genuine object
integrity problem prevents reliable task completion, for example:

- the target is fragmented and one or more fragments are outside,
  missing, or unresolved
- the target is merged with another object so that the original
  target cannot be reliably evaluated
- the target has vanished
- severe deformation destroys stable target identity
- some physically required portion of the original target cannot
  be accounted for

DO NOT set task_invalidating_integrity=true merely because:

- final_relation = outside
- final_relation = hovering
- final_relation = rim_or_minor_inside
- final_relation = unknown
- the object is not fully inside the container
- the PUT / TAKE goal is spatially incomplete

Those are SPATIAL TASK FAILURES and must be represented using:

- final_relation
- endpoint_candidate
- endpoint_reason

not task_invalidating_integrity.

If:

target_integrity = "intact"

then:

task_invalidating_integrity should normally be false.

Example:

final_relation = "rim_or_minor_inside"
target_integrity = "intact"

must normally produce:

task_invalidating_integrity = false

because the object itself is intact; only the spatial task goal is
not yet satisfied.


Before returning JSON, verify:

1. If target_integrity="intact", do not set
   task_invalidating_integrity=true merely due to spatial placement.

2. final_relation and task_invalidating_integrity represent different
   concepts:
   - final_relation = spatial state
   - task_invalidating_integrity = structural / identity failure

3. A target may have:
   final_relation="rim_or_minor_inside"
   and
   task_invalidating_integrity=false.

4. A target may fail the task spatially while remaining physically intact.


Output EXACTLY one JSON object and nothing else:

{{
  "end_description": "short factual description of every required target",

  "objects": [
    {{
      "id": "obj_1",

      "appearance_at_end": "green dinosaur plush",

      "identity_match": "same|likely_same|unknown",

      "is_doll_like": true,

      "visibility": "visible|partially_occluded|fully_occluded|vanished|unknown",

      "occlusion_source": "none|robot_arm|gripper|container_wall|other_object|other",

      "final_container_id": "box_1",

      "final_relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",

      "final_relation_confidence": "high|medium|low",

      "final_relation_source": "end_top|end_left_arm|end_right_arm|multi_view",

      "target_integrity": "intact|deformed_trackable|fragmented|merged|vanished|unknown",

      "fragment_count": 1,

      "fragment_relations": [
        {{
          "fragment": "head/upper part",
          "relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown"
        }}
      ],

      "all_fragments_accounted_for": true,

      "all_fragments_satisfy_goal": true,

      "task_invalidating_integrity": false,

      "integrity_reason": "none"
    }}
  ],

  "containers": [
    {{
      "id": "box_1",
      "container_integrity": "usable|deformed_but_usable|collapsed_unusable|unknown"
    }}
  ],

  "view_evidence": {{
    "top": {{
      "visibility": "clear|occluded|not_visible",
      "relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",
      "confidence": "high|medium|low"
    }},

    "left_arm": {{
      "visibility": "clear|occluded|not_visible",
      "relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",
      "confidence": "high|medium|low"
    }},

    "right_arm": {{
      "visibility": "clear|occluded|not_visible",
      "relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",
      "confidence": "high|medium|low"
    }}
  }},

  "endpoint_candidate": "success|failure|uncertain",

  "endpoint_reason": "explain whether ALL required targets satisfy the endpoint",

  "last_frame_readable": true
}}

For intact/non-fragmented targets:
- fragment_count may be 1
- fragment_relations may be []
- all_fragments_accounted_for should normally be true
- all_fragments_satisfy_goal should reflect the object's spatial state

endpoint_candidate="success" is allowed ONLY when EVERY required target
satisfies the endpoint requirement.

Do NOT output answer=0 or answer=1.
"""




_CONTAINER_END_TAKE = """You are Stage 2, the ENDPOINT AUTHORITY of a robot task-completion evaluator.

Task instruction:
{instruction}

{doll_semantics}

{take_spatial_rules}

A previous model produced the following START_STATE:

{start_state}

IMPORTANT:

START_STATE defines the REQUIRED TARGET SET.

For target_quantifier="all_matching_instances", EVERY doll-like target
listed in START_STATE.target_candidates[] must be independently checked
at END.

START_STATE is still a provisional visual observation.
Verify object identity against START_TOP, but do NOT silently remove a
required target merely because it was untouched or difficult to locate.

Use START initial_relation only as INITIAL-STATE context.
Judge the FINAL spatial state from END views.
Do NOT infer success merely because the instruction says TAKE.
A target that was already outside at START does NOT mean the video failed.


You will receive:

1. START_TOP
2. END_TOP
3. END_TRIPLET

END_TRIPLET is the same final time step:

HEAD/TOP | LEFT ARM | RIGHT ARM


Your job is the FINAL spatial state at END.

DO NOT analyze the whole trajectory.
DO NOT use a middle-frame event as proof of final completion.


MULTI-TARGET COMPLETION RULE — CRITICAL:

For target_quantifier="all_matching_instances":

EVERY required target must satisfy the final TAKE condition.

For TAKE:

final_relation=outside
=> spatial TAKE endpoint can succeed for that target

final_relation=inside
=> failure

final_relation=mostly_inside
=> failure

final_relation=rim_or_minor_inside
=> failure

final_relation=unknown
=> uncertain, NEVER success

final_relation=hovering
=> do not treat this as successful by default;
   however, do not call a fully extracted held-above target "hovering".
   Fully clear of the container volume should be "outside".

Examples:

obj_1 = outside
obj_2 = outside
=> endpoint_candidate may be success

obj_1 = outside
obj_2 = inside
=> failure

obj_1 = outside
obj_2 = mostly_inside
=> failure

obj_1 = outside
obj_2 = rim_or_minor_inside
=> failure

obj_1 = outside
obj_2 = unknown
=> uncertain, NEVER success

One successful target can NEVER compensate for another required
target that remains inside, mostly_inside, rim-bound, unresolved,
missing, or failed.


TARGET COVERAGE RULE:

For EVERY required target ID from START_STATE.target_candidates[],
produce one corresponding entry in objects[].

Preserve the same object ID whenever possible.

If a target is not visible at END, DO NOT omit it.

Instead output that target with an appropriate state such as:
- visibility = fully_occluded
- visibility = vanished
- final_relation = unknown

A required target disappearing from objects[] is NOT allowed.


VIEW RULE:

END_TOP is primary global spatial evidence.

END_TRIPLET is auxiliary evidence for:
- extraction clearance
- box wall occlusion
- robot-arm occlusion
- gripper contact

For END_TRIPLET:
left third = HEAD/TOP
middle third = LEFT ARM
right third = RIGHT ARM


OCCLUSION VS VANISHING:

occluded:
a plausible physical occluder explains why the target cannot be seen,
such as robot arm, gripper, box wall, or another object.

vanished:
the target should be visible, no physical occluder explains its
disappearance, and it cannot be located in any END view.


SEMANTIC IDENTITY RULE:

If START_STATE identifies obj_X as doll-like and the END object is the
same tracked obj_X, preserve its semantic identity when object continuity
remains visually reliable.

If deformation makes identity genuinely uncertain, use the available
unknown / low-confidence fields instead of inventing a different object.


FRAGMENTATION RULE:

If one original target splits into multiple fragments, keep all fragments
under the SAME original object ID.

Do NOT turn fragments into new independent target IDs.

For TAKE:

all fragments outside
=> fragmented target may satisfy the spatial goal

head outside + body inside
=> target does NOT satisfy the goal

one fragment outside + one fragment unknown/missing
=> target does NOT satisfy the goal

Fragmentation itself is not automatically a task failure.
The important question is whether ALL fragments of that original target
satisfy the spatial goal.


============================================================
INTEGRITY FIELD SEMANTICS — CRITICAL
============================================================

"target_integrity" and "task_invalidating_integrity" describe
the PHYSICAL / IDENTITY INTEGRITY of the target object.

They do NOT describe whether the final TAKE spatial relation
satisfies the task.

Set task_invalidating_integrity=true ONLY when a genuine object
integrity problem prevents reliable task completion, for example:

- the target is fragmented and one or more fragments are outside,
  missing, or unresolved
- the target is merged with another object so that the original
  target cannot be reliably evaluated
- the target has vanished
- severe deformation destroys stable target identity
- some physically required portion of the original target cannot
  be accounted for

DO NOT set task_invalidating_integrity=true merely because:

- final_relation = outside
- final_relation = hovering
- final_relation = rim_or_minor_inside
- final_relation = unknown
- the object remains inside the container
- the TAKE goal is spatially incomplete

Those are SPATIAL TASK FAILURES and must be represented using:

- final_relation
- endpoint_candidate
- endpoint_reason

not task_invalidating_integrity.

If:

target_integrity = "intact"

then:

task_invalidating_integrity should normally be false.

Example:

final_relation = "rim_or_minor_inside"
target_integrity = "intact"

must normally produce:

task_invalidating_integrity = false

because the object itself is intact; only the spatial task goal is
not yet satisfied.

A collapsed or deformed container does NOT automatically invalidate a
clearly extracted, trackable TAKE target. Still report container_integrity.


Before returning JSON, verify:

1. If target_integrity="intact", do not set
   task_invalidating_integrity=true merely due to spatial placement.

2. final_relation and task_invalidating_integrity represent different
   concepts:
   - final_relation = spatial state
   - task_invalidating_integrity = structural / identity failure

3. A target may have:
   final_relation="rim_or_minor_inside"
   and
   task_invalidating_integrity=false.

4. A target may fail the task spatially while remaining physically intact.


Output EXACTLY one JSON object and nothing else:

{{
  "end_description": "short factual description of every required target",

  "objects": [
    {{
      "id": "obj_1",

      "appearance_at_end": "green dinosaur plush",

      "identity_match": "same|likely_same|unknown",

      "is_doll_like": true,

      "visibility": "visible|partially_occluded|fully_occluded|vanished|unknown",

      "occlusion_source": "none|robot_arm|gripper|container_wall|other_object|other",

      "final_container_id": "box_1",

      "final_relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",

      "final_relation_confidence": "high|medium|low",

      "final_relation_source": "end_top|end_left_arm|end_right_arm|multi_view",

      "target_integrity": "intact|deformed_trackable|fragmented|merged|vanished|unknown",

      "fragment_count": 1,

      "fragment_relations": [
        {{
          "fragment": "head/upper part",
          "relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown"
        }}
      ],

      "all_fragments_accounted_for": true,

      "all_fragments_satisfy_goal": true,

      "task_invalidating_integrity": false,

      "integrity_reason": "none"
    }}
  ],

  "containers": [
    {{
      "id": "box_1",
      "container_integrity": "usable|deformed_but_usable|collapsed_unusable|unknown"
    }}
  ],

  "view_evidence": {{
    "top": {{
      "visibility": "clear|occluded|not_visible",
      "relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",
      "confidence": "high|medium|low"
    }},

    "left_arm": {{
      "visibility": "clear|occluded|not_visible",
      "relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",
      "confidence": "high|medium|low"
    }},

    "right_arm": {{
      "visibility": "clear|occluded|not_visible",
      "relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",
      "confidence": "high|medium|low"
    }}
  }},

  "endpoint_candidate": "success|failure|uncertain",

  "endpoint_reason": "explain whether ALL required targets satisfy the TAKE endpoint",

  "last_frame_readable": true
}}

For intact/non-fragmented targets:
- fragment_count may be 1
- fragment_relations may be []
- all_fragments_accounted_for should normally be true
- all_fragments_satisfy_goal should reflect the object's spatial state

endpoint_candidate="success" is allowed ONLY when EVERY required target
satisfies the endpoint requirement.

Do NOT output answer=0 or answer=1.
"""


_CONTAINER_VERIFY_PUT = """You are Stage 3, the TRAJECTORY verifier.

Task instruction:
{instruction}

{doll_semantics}

START_STATE:
{start_state}

END_STATE:
{end_state}


IMPORTANT:

Stage 2 is authoritative for FINAL spatial state.

Your main job is NOT to decide whether the whole episode succeeded.

Your job is to determine, FOR EACH REQUIRED TARGET:
- whether it was manipulated
- what motion/action occurred
- whether the intended action was completed
- whether the action was later undone
- whether the object's semantic identity remained consistent


You receive:

- MID1 triplet
- MID2 triplet
- MID3 triplet
- END triplet for endpoint cross-check only

Each triplet is:

HEAD/TOP | LEFT ARM | RIGHT ARM


REQUIRED TARGET COVERAGE RULE — CRITICAL:

For target_quantifier="all_matching_instances", EVERY target in
START_STATE.target_candidates[] matters.

object_tracks[] must account for EVERY required target ID.

Do NOT only output objects that were successfully manipulated.

Do NOT stop after finding one successful PUT/TAKE trajectory.

If a required target was never manipulated, still output a track:

{{
  "object_id": "obj_2",
  "net_action": "not_manipulated"
}}

An untouched required target must NEVER silently disappear from
object_tracks[].


MULTI-TARGET RULE:

Example:

START:
obj_1 = blue plush doll
obj_2 = pink plush doll

Trajectory:
obj_1 is untouched
obj_2 is successfully inserted

You must report BOTH:

obj_1 -> not_manipulated
obj_2 -> put_completed

Do NOT describe this as "task completed because one doll was placed".


SEMANTIC IDENTITY INHERITANCE:

Object identity is temporally stable when visual continuity remains
reliable.

If obj_1 is identified as doll-like in START_STATE or END_STATE and the
middle-frame object is the same obj_1 by motion continuity, appearance,
color, markings, and location, preserve that tracked identity.

If world-model deformation makes identity genuinely uncertain, report
uncertainty rather than inventing a new semantic category.


ACTION RULE:

For PUT:

grasp
-> lift
-> move toward box
-> lower
-> insert

is a PUT trajectory.

Initial lifting does NOT mean TAKE.

For TAKE:
use TAKE only when the net motion removes the target OUT OF / AWAY FROM
the container.


FRAGMENTATION RULE:

If an original target fragments, keep the SAME object_id.

trajectory_integrity may be "fragmented".

Fragmentation does not automatically mean that the manipulation action
was absent.

Whether the fragmented target ultimately satisfies the spatial goal is
determined by Stage 2 using ALL fragments.


ENDPOINT AUTHORITY RULE:

Do NOT replace a clear Stage-2 final relation with a middle-frame event.

Middle frames cannot prove final state.

An endpoint override may only be requested when Stage 2 is:
- unknown
- occluded
- low confidence

AND END_TRIPLET itself provides stronger endpoint evidence.

evidence_source MUST be one of:

end_top
end_left_arm
end_right_arm
end_multi_view

Never use a MID frame as endpoint override evidence.


Output EXACTLY one JSON object and nothing else:

{{
  "object_tracks": [
    {{
      "object_id": "obj_1",

      "appearance": "description",

      "is_doll_like": true,

      "target_match": true,

      "target_match_reason": "same required target object as START_STATE",

      "events": [
        "grasp",
        "lift",
        "transport_toward_box_1",
        "insert"
      ],

      "destination_container_id": "box_1",

      "net_action": "put_completed|put_attempt|take_completed|take_attempt|not_manipulated|other|unclear",

      "trajectory_integrity": "valid|fragmented|vanished|unclear",

      "later_undo": false
    }}
  ],

  "endpoint_override_request": {{
    "requested": false,
    "object_id": null,
    "proposed_relation": null,
    "evidence_source": null,
    "reason": null
  }},

  "trajectory_summary": "short factual per-target trajectory summary"
}}

Before returning JSON, verify that every required START target ID has
been accounted for in object_tracks[].

Do NOT output final task success.
Do NOT output answer 0/1.
Do NOT output corrected_final_relation.
"""




_CONTAINER_VERIFY_TAKE = """You are Stage 3, the TRAJECTORY verifier.

Task instruction:
{instruction}

{doll_semantics}

START_STATE:
{start_state}

END_STATE:
{end_state}


IMPORTANT:

Stage 2 is authoritative for FINAL spatial state.

Your main job is NOT to decide whether the whole episode succeeded.

Your job is to determine, FOR EACH REQUIRED TARGET:
- whether it was manipulated
- what motion/action occurred
- whether the intended TAKE / extraction was completed
- whether the action was later undone
- whether the object's semantic identity remained consistent


You receive:

- MID1 triplet
- MID2 triplet
- MID3 triplet
- END triplet for endpoint cross-check only

Each triplet is:

HEAD/TOP | LEFT ARM | RIGHT ARM


REQUIRED TARGET COVERAGE RULE — CRITICAL:

For target_quantifier="all_matching_instances", EVERY target in
START_STATE.target_candidates[] matters.

object_tracks[] must account for EVERY required target ID.

Do NOT only output objects that were successfully manipulated.

Do NOT stop after finding one successful TAKE trajectory.

If a required target was never manipulated, still output a track:

{{
  "object_id": "obj_2",
  "net_action": "not_manipulated"
}}

An untouched required target must NEVER silently disappear from
object_tracks[].


MULTI-TARGET RULE:

Example:

START:
obj_1 = blue plush doll (already outside)
obj_2 = pink plush doll (inside)

Trajectory:
obj_1 is untouched
obj_2 is successfully extracted

You must report BOTH:

obj_1 -> not_manipulated
obj_2 -> take_completed

Do NOT describe this as "task completed because one doll was removed".


SEMANTIC IDENTITY INHERITANCE:

Object identity is temporally stable when visual continuity remains
reliable.

If obj_1 is identified as doll-like in START_STATE or END_STATE and the
middle-frame object is the same obj_1 by motion continuity, appearance,
color, markings, and location, preserve that tracked identity.

If world-model deformation makes identity genuinely uncertain, report
uncertainty rather than inventing a new semantic category.


ACTION RULE:

For TAKE, distinguish extraction from generic lifting.

Typical TAKE trajectory:

target begins inside / partly inside container
    -> grasp/contact
    -> lift / pull upward
    -> clear container rim / container volume
    -> transport outside / away
    => take_completed

take_completed requires evidence of NET REMOVAL out of the container,
not merely grasping.

Examples:

grasp + lift while target remains in container
=> take_attempt

target rises but meaningful body remains inside/rim
=> take_attempt

target clears the rim/container volume
=> take_completed

target is extracted, then later returned into the box
=> later_undo=true

CRITICAL ANTI-FALSE-POSITIVE:

An object that was already outside the container and is simply moved
around must NOT be labeled take_completed solely because it was lifted.

If it was already outside and untouched:
not_manipulated
is acceptable.


FRAGMENTATION RULE:

If an original target fragments, keep the SAME object_id.

trajectory_integrity may be "fragmented".

Fragmentation does not automatically mean that the manipulation action
was absent.

Whether the fragmented target ultimately satisfies the spatial goal is
determined by Stage 2 using ALL fragments.


ENDPOINT AUTHORITY RULE:

Do NOT replace a clear Stage-2 final relation with a middle-frame event.

Middle frames cannot prove final state.

Do NOT automatically rewrite Stage2 inside -> outside merely because a
TAKE trajectory looks complete.

An endpoint override may only be requested when Stage 2 is:
- unknown
- occluded
- low confidence

AND END_TRIPLET itself provides stronger endpoint evidence.

evidence_source MUST be one of:

end_top
end_left_arm
end_right_arm
end_multi_view

Never use a MID frame as endpoint override evidence.


Output EXACTLY one JSON object and nothing else:

{{
  "object_tracks": [
    {{
      "object_id": "obj_1",

      "appearance": "description",

      "is_doll_like": true,

      "target_match": true,

      "target_match_reason": "same required target object as START_STATE",

      "events": [
        "grasp",
        "lift",
        "extract_from_box_1",
        "clear_rim",
        "transport_away"
      ],

      "destination_container_id": "box_1",

      "net_action": "put_completed|put_attempt|take_completed|take_attempt|not_manipulated|other|unclear",

      "trajectory_integrity": "valid|fragmented|vanished|unclear",

      "later_undo": false
    }}
  ],

  "endpoint_override_request": {{
    "requested": false,
    "object_id": null,
    "proposed_relation": null,
    "evidence_source": null,
    "reason": null
  }},

  "trajectory_summary": "short factual per-target trajectory summary"
}}

Before returning JSON, verify that every required START target ID has
been accounted for in object_tracks[].

Do NOT output final task success.
Do NOT output answer 0/1.
Do NOT output corrected_final_relation.
"""


_CONTAINER_WRIST_RECHECK_PUT = """You are Stage 4, a DEDICATED FINAL WRIST-VIEW endpoint rechecker.

This is NOT a trajectory judgement.
This is NOT a task-success judgement.

Your ONLY job is to re-check the FINAL spatial relation between
specific target objects and their destination container using
enlarged FINAL left-arm and right-arm camera views.

Task instruction:
{instruction}

The following targets were selected for wrist recheck because
Stage 2 was ambiguous or potentially confused by robot-arm /
gripper occlusion:

{recheck_targets}

NOTE about trigger fields in recheck_targets:
wrist_recheck_trigger_reasons / stage2_holding_above_cue explain WHY
this target was selected. They are NOT ground truth.
Do NOT assume the target is inside merely because Stage3 says
put_completed.


You will receive exactly TWO images from the SAME FINAL timestep:

1. END_LEFT_ARM
2. END_RIGHT_ARM

These are enlarged crops from the final triplet.
There are NO middle frames in this stage.


WHY THIS RECHECK EXISTS:

The overhead endpoint judge may incorrectly call a target
"hovering", "outside", or "unknown" when:

- the gripper covers the target,
- the robot arm covers the box,
- a transparent box wall hides depth,
- the target is large relative to the container,
- the target fills most of the wrist camera view.

Use the wrist views to directly inspect insertion depth.


============================================================
ROBOT-ARM / GRIPPER CONTACT RULE
============================================================

Robot-arm or gripper proximity/contact is NOT itself a spatial relation.

Do NOT classify a target as "hovering" merely because:
- a robot arm is still near the target,
- a gripper overlaps part of the target,
- the robot has not fully retracted.

Judge the TARGET BODY relative to the CONTAINER VOLUME.

The key question is spatial:

Is the target body inside, mostly inside, on the rim, above, or outside
the destination container?

Do NOT try to infer task completion from gripper open/closed state.

For PUT:

inside:
    target body is clearly contained in the box.

mostly_inside:
    most / central body is inside.
    Some head, limb, tail, or end may protrude.
    This counts as successful spatial placement.

rim_or_minor_inside:
    only a small portion enters the box, or the target is perched
    on the rim.

hovering:
    target is clearly above the container with no meaningful
    insertion.

outside:
    target is clearly outside the container.

unknown:
    wrist views do not provide enough evidence.


TRANSPARENT-CONTAINER RULE:

For transparent boxes, infer containment using:
- container rim,
- container walls,
- target body depth,
- whether the target body lies within the container footprint.

Do NOT assume "outside" just because the target can be seen through
a transparent wall.


LARGE-PLUSH RULE:

A large plush may fill the box or protrude above the rim.

If its main body is already inside the box volume, classify it as
mostly_inside rather than hovering.


CROSS-VIEW RULE:

Use both wrist views independently.

If one view is clear and the other is occluded, trust the clear view.

If the two views genuinely conflict and cannot be reconciled,
return wrist_final_relation="unknown" with low confidence.

Do NOT invent visibility that is not present.


IDENTITY RULE:

Only re-evaluate the target object IDs listed in recheck_targets.

Confirm that the visible object matches the requested target by
appearance, color, markings, shape, and context.

Do NOT use a different doll or unrelated object as evidence.

For PUT, also confirm that the visible container is consistent with
the destination container.


FRAGMENTATION SAFETY RULE:

Do NOT use wrist recheck to erase clear evidence of fragmentation.

If the target clearly appears fragmented in the wrist views, report
wrist_target_integrity="fragmented".

A success-like spatial relation must not be inferred from only one
fragment while another visible fragment remains outside.


Output EXACTLY one JSON object and nothing else:

{{
  "objects": [
    {{
      "object_id": "obj_1",

      "target_identity_consistent": true,

      "container_identity_consistent": true,

      "left_wrist_visibility": "clear|occluded|not_visible",

      "left_wrist_relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",

      "right_wrist_visibility": "clear|occluded|not_visible",

      "right_wrist_relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",

      "wrist_final_relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",

      "wrist_relation_confidence": "high|medium|low",

      "evidence_view": "left_wrist|right_wrist|both|none",

      "gripper_contact": true,

      "wrist_target_integrity": "intact|deformed_trackable|fragmented|unknown",

      "reason": "short factual explanation"
    }}
  ],

  "wrist_summary": "short factual summary"
}}

IMPORTANT:

Do NOT output task success.
Do NOT output answer 0/1.
Do NOT reason from earlier middle-frame success.
Use ONLY the two FINAL wrist views for the new spatial relation.
"""



_CONTAINER_WRIST_RECHECK_TAKE = """You are Stage 4, a DEDICATED FINAL WRIST-VIEW endpoint rechecker.

This is NOT a trajectory judgement.
This is NOT a task-success judgement.

Your ONLY job is to re-check whether specific target objects are fully
clear of the source container using enlarged FINAL left-arm and
right-arm camera views.

Task instruction:
{instruction}

The following targets were selected for wrist recheck because
Stage 2 was ambiguous or potentially confused by robot-arm /
gripper occlusion:

{recheck_targets}

NOTE about trigger fields in recheck_targets:
wrist_recheck_trigger_reasons / stage2_holding_above_cue explain WHY
this target was selected. They are NOT ground truth.
Do NOT assume the target is outside merely because Stage3 says
take_completed.


You will receive exactly TWO images from the SAME FINAL timestep:

1. END_LEFT_ARM
2. END_RIGHT_ARM

These are enlarged crops from the final triplet.
There are NO middle frames in this stage.


WHY THIS RECHECK EXISTS:

The overhead endpoint judge may incorrectly call a target
"hovering", "inside", "mostly_inside", or "unknown" when:

- the gripper covers the target,
- the robot arm covers the box,
- a transparent box wall hides depth,
- the target is large relative to the container,
- the target fills most of the wrist camera view.

Use the wrist views to directly inspect extraction / clearance.


============================================================
ROBOT-ARM / GRIPPER CONTACT RULE
============================================================

Robot-arm or gripper proximity/contact is NOT itself a spatial relation.

Do NOT classify a fully extracted target as "hovering" merely because:
- a robot arm is still near the target,
- a gripper still holds the target,
- the robot has not fully retracted.

Gripper contact/holding is NOT negative evidence for TAKE.

Judge the TARGET BODY relative to the CONTAINER VOLUME.

The key TAKE question is:

At the FINAL timestep, is the requested target fully clear of the
source container?


For TAKE:

outside:
    target body is fully clear of container volume/rim.
    Target may still be held by the gripper.
    Target may be directly above the box.
    This is successful TAKE spatial evidence.

rim_or_minor_inside:
    some meaningful portion remains caught in / below the rim.

mostly_inside:
    most of target remains inside.

inside:
    target remains in container.

hovering:
    Do NOT classify a fully extracted target as hovering merely because
    it is being held above the container.
    If the target body is fully clear of the container volume, use outside.
    Use hovering only when clearance is genuinely ambiguous.

unknown:
    wrist views do not provide enough evidence.


TRANSPARENT-CONTAINER RULE:

Visibility through a transparent wall does not mean the target remains
inside.

Use:
- rim position
- wall boundaries
- target depth
- whether any target body remains within the container footprint/volume

If the target body is fully above/outside the physical container volume,
classify outside.


CROSS-VIEW RULE:

Use both wrist views independently.

If one view is clear and the other is occluded, trust the clear view.

If the two views genuinely conflict and cannot be reconciled,
return wrist_final_relation="unknown" with low confidence.

Do NOT invent visibility that is not present.


IDENTITY RULE:

Only re-evaluate the target object IDs listed in recheck_targets.

Confirm that the visible object matches the requested target by
appearance, color, markings, shape, and context.

Do NOT use a different doll or unrelated object as evidence.


FRAGMENTATION SAFETY RULE:

Do NOT use wrist recheck to erase clear evidence of fragmentation.

If the target clearly appears fragmented in the wrist views, report
wrist_target_integrity="fragmented".

One visible fragment outside is not enough if another visible fragment
remains inside or unresolved.


Output EXACTLY one JSON object and nothing else:

{{
  "objects": [
    {{
      "object_id": "obj_1",

      "target_identity_consistent": true,

      "container_identity_consistent": true,

      "left_wrist_visibility": "clear|occluded|not_visible",

      "left_wrist_relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",

      "right_wrist_visibility": "clear|occluded|not_visible",

      "right_wrist_relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",

      "wrist_final_relation": "inside|mostly_inside|rim_or_minor_inside|hovering|outside|unknown",

      "wrist_relation_confidence": "high|medium|low",

      "evidence_view": "left_wrist|right_wrist|both|none",

      "gripper_contact": true,

      "wrist_target_integrity": "intact|deformed_trackable|fragmented|unknown",

      "reason": "short factual explanation"
    }}
  ],

  "wrist_summary": "short factual summary"
}}

IMPORTANT:

Do NOT output task success.
Do NOT output answer 0/1.
Do NOT reason from earlier middle-frame success.
Use ONLY the two FINAL wrist views for the new spatial relation.
"""


def build_wrist_recheck_content(
    instruction: str,
    left_wrist: Any,
    right_wrist: Any,
    recheck_targets: Union[
        str,
        List[Dict[str, Any]],
        Dict[str, Any],
    ],
    schema_object_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Stage 4: dedicated FINAL wrist endpoint recheck (v5a4 semantics).

    schema_object_id, when set, only rewrites the first JSON-schema example
    object_id so a split multi-target call does not hard-code obj_1.
    """

    action = instruction_action(instruction)
    template = (
        _CONTAINER_WRIST_RECHECK_TAKE
        if action == "take"
        else _CONTAINER_WRIST_RECHECK_PUT
    )
    prompt = template.format(
        instruction=instruction or "",
        recheck_targets=_state_to_text(recheck_targets),
    )

    if schema_object_id is not None:
        schema_object_id = str(schema_object_id).strip()
        if not schema_object_id:
            raise ValueError("schema_object_id must be non-empty")
        prompt = prompt.replace(
            '"object_id": "obj_1"',
            f'"object_id": "{schema_object_id}"',
            1,
        )

    return [
        {
            "type": "text",
            "text": prompt,
        },

        {
            "type": "text",
            "text": "END_LEFT_ARM:",
        },
        {
            "type": "image",
            "image": left_wrist,
        },

        {
            "type": "text",
            "text": "END_RIGHT_ARM:",
        },
        {
            "type": "image",
            "image": right_wrist,
        },
    ]


def classify_instruction(instruction: str) -> str:
    """Return task family: fold | container | generic."""
    text = (instruction or "").strip().lower()
    if not text:
        return "generic"

    def has_word(word: str) -> bool:
        return re.search(rf"\b{re.escape(word)}\b", text) is not None

    has_verb = any(has_word(v) for v in _CONTAINER_VERBS)
    has_noun = any(has_word(n) for n in _CONTAINER_NOUNS)
    if has_verb and has_noun:
        return "container"

    dest_nouns = ("box", "basket", "container", "bin", "bag")
    if has_word("doll") and any(has_word(n) for n in dest_nouns):
        return "container"

    if any(has_word(t) for t in _FOLD_TERMS):
        return "fold"

    return "generic"


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
    """Stage 1: initial-scene inventory (v4 doll binding)."""
    prompt = _CONTAINER_START.format(
        instruction=instruction or "",
        doll_semantics=doll_semantics_block(),
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
    """Stage 2: endpoint authority with END_TOP + END_TRIPLET."""
    action = instruction_action(instruction)
    if action == "take":
        prompt = _CONTAINER_END_TAKE.format(
            instruction=instruction or "",
            doll_semantics=doll_semantics_block(),
            take_spatial_rules=take_spatial_rules_block(),
            start_state=_state_to_text(start_state),
        )
    else:
        prompt = _CONTAINER_END_PUT.format(
            instruction=instruction or "",
            doll_semantics=doll_semantics_block(),
            put_spatial_rules=put_spatial_rules_block(),
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
    """Stage 3: trajectory verifier (no final answer).

    mid_triplets: [mid1, mid2, mid3] full stitches
    end_triplet: final full stitch for endpoint cross-check only
    """
    if len(mid_triplets) < 3:
        raise ValueError(
            f"need 3 mid triplets for interleave v5, got {len(mid_triplets)}"
        )
    mid1, mid2, mid3 = mid_triplets[0], mid_triplets[1], mid_triplets[2]
    action = instruction_action(instruction)
    template = (
        _CONTAINER_VERIFY_TAKE
        if action == "take"
        else _CONTAINER_VERIFY_PUT
    )
    prompt = template.format(
        instruction=instruction or "",
        doll_semantics=doll_semantics_block(),
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
