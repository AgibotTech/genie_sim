"""Stage1/2/3 prompts for folding garments into a compact rectangle or square.

Integrated into the release task router and three-stage judge."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union


_STAGE1 = """\
You are Stage1 for a FOLD CLOTH (GARMENT) evaluator.

Instruction: {instruction}

INPUT: START_TOP only. Do NOT judge task success.

In this task, "cloth" means an item of clothing / a garment, not necessarily
an initially rectangular towel or fabric sheet.
A garment is deformable and may have sleeves, trouser legs, a collar, or a hood.
Its natural open silhouette may be T-shaped or otherwise non-rectangular.
These garment parts and sewn seams are not evidence of folding by themselves.
Wrinkles or small local bends do not automatically mean folded.
A spread garment can be imperfectly flat.
A crumpled / scrunched garment is not the same as a deliberately folded garment.
Do not use the instruction itself as evidence that the garment is folded.

TASK:
1. Identify physical garment / clothing candidates.
2. Choose the instruction-relevant primary target.
3. Describe initial appearance, garment type, body region, and visible sleeves / legs / other extensions.
4. Set initial_fold_state.

initial_fold_state meanings:
- spread_flat: mostly open garment with no deliberate panel-over-panel fold (can be wrinkled)
- partially_folded: some folding started but not a clear final layered fold
- crumpled: irregular multi-direction bunching / wad
- folded_layered: already shows coherent layered fold
- unclear: cannot tell

Vision-first: judge from the image, not from the instruction wording.

Return ONLY one JSON object:
{{
  "start_description": "...",
  "task_family": "fold_cloth",
  "task_action": "fold",
  "target_candidates": [
    {{
      "id": "garment_1",
      "appearance": "...",
      "garment_type": "shirt|pants|other|unclear",
      "visible_garment_parts": "...",
      "is_garment_like": true,
      "identity_confidence": "high|medium|low",
      "initial_fold_state": "spread_flat|partially_folded|crumpled|folded_layered|unclear",
      "initial_visibility": "clear|partial|unknown",
      "spatial_anchor": "..."
    }}
  ],
  "primary_target_id": "garment_1|null",
  "scene_readable": true
}}
"""

_STAGE2 = """\
You are Stage2 (FINAL FOLD GEOMETRY) for a FOLD CLOTH (GARMENT) evaluator.

Instruction: {instruction}

INPUTS: START_TOP, END_TOP, END_TRIPLET, and parsed START_STATE.

START_STATE:
{start_state}

PRIMARY VIEW POLICY:
TOP-HEAD is primary for final folding geometry.
Left/right wrist views in the END_TRIPLET are secondary evidence only.
Do NOT let a close-up bent garment in a wrist crop override the head-view
overall geometry.

Do NOT output final task success/failure here. Analyze endpoint geometry and
benchmark completion level only.

final_fold_state (structural taxonomy):
- spread_flat: essentially an open garment at END, with no deliberate folding overlap
- crumpled: irregular multi-direction wrinkles / scrunched wad / collapse
- folded_layered: coherent self-overlap / layering (one panel over another)
- unclear: cannot tell

Normal surface wrinkles on a coherently folded bundle do not make it crumpled.
full_open_garment means the garment remains open without deliberate folding;
it does not mean the garment has only one physical fabric layer. Shirts and
trousers naturally have front / back panels even when completely unfolded.

TASK COMPLETION CRITERIA — FOLDING CLOTHING:

The intended completed END state is:
- the correct garment folded into a compact, approximately rectangular or
  square bundle;
- coherent deliberate self-overlap / multiple folded layers;
- sleeves, trouser legs, or other large extensions, when present, folded into
  or aligned within the main bundle rather than substantially spread outside it.

Judge the WHOLE garment silhouette, not only a rectangular torso region while
sleeves or legs remain extended.
Small uneven corners, a collar, a little hood material, modest edge offsets,
and normal wrinkles are acceptable when the overall bundle is still roughly
rectangular or square. Do not require perfect symmetry or sharply aligned edges.

Garments have different starting shapes and folding methods. Do NOT assume
that START is a rectangular sheet. Do NOT impose a fixed 2–3 half-fold sequence
or a mandatory END/START area ratio of one quarter.
Folding sleeves inward, overlapping body panels, aligning trouser legs, and
folding the body or legs across their length are possible actions, not a
mandatory procedure. Judge the visible final result.

A valid fold is only PARTIAL progress when the remaining garment is still
open / substantially extended or the result is only an unfinished long strip.
A single observed fold is not automatically partial: it may finish a garment
that was already partly folded at START.
A compact rectangle or square is not sufficient by itself if it is still an
open garment with no deliberate overlap, a rolled cylinder, or a scrunched wad.

final_fold_state answers:
"Is the garment structurally open / crumpled / deliberately layered?"

benchmark_completion_state answers:
"Is the whole garment folded into the intended compact rectangle or square?"

Examples:
A shirt with sleeves folded inward but the body still largely open:
- final_fold_state = folded_layered
- benchmark_completion_state = partial
- endpoint_fold_evidence = supports_fold
- endpoint_completion_evidence = supports_partial

A shirt folded into a compact layered rectangle with sleeves contained:
- final_fold_state = folded_layered
- final_shape = compact_rectangle
- garment_parts_contained = yes
- benchmark_completion_state = complete
- endpoint_completion_evidence = supports_complete

Trousers with legs aligned but still extended as an unfinished long strip:
- final_fold_state = folded_layered
- final_shape = long_strip
- benchmark_completion_state = partial

A roughly square torso region with substantial sleeves still protruding:
- final_shape = irregular (judge the whole silhouette)
- garment_parts_contained = no
- benchmark_completion_state = partial

A coherent layered triangular bundle:
- final_fold_state may still be folded_layered
- final_shape = triangle
- benchmark_completion_state = incorrect
- endpoint_completion_evidence = contradicts_complete

Decision order:
1. Decide final_fold_state from deliberate overlap and fold structure.
2. Decide final_shape from the WHOLE garment silhouette.
3. Check whether substantial garment extensions are contained / aligned.
4. Compare END with START qualitatively, accounting for the initial fold state.
5. Decide benchmark_completion_state from layering + overall shape + containment.

Do NOT infer completion from visible layers or reduced footprint alone.
Sewn hems, cuffs, collars, and naturally overlapping garment parts are not
sufficient evidence of deliberate folding by themselves.

final_shape:
- square: compact overall near-square bundle
- compact_rectangle: compact overall near-rectangle, with the garment folded
  onto itself and no substantial open sleeves / legs outside the bundle
- long_strip: elongated unfinished strip with substantial length still unfolded
- triangle: predominantly triangular overall bundle, not the intended completion
- irregular: irregular overall silhouette, including substantial protrusions;
  not necessarily crumpled
- unclear

A rectangular garment is not required to become near-square. Do not classify
an otherwise completed rectangular bundle as long_strip merely because one
side is longer; look for substantial remaining unfolded length.

garment_parts_contained:
- yes: substantial sleeves / legs / extensions are folded into or aligned
  within the main bundle; small natural details may protrude
- no: substantial garment parts remain extended outside the main bundle
- unknown: occlusion / readability prevents a reliable judgment

footprint_ratio_category (visual estimate, supporting evidence only):
- near_original / around_half / around_quarter_or_smaller / unclear
Use START_TOP vs END_TOP. Perspective, pose, initial partial folds, and WM
deformation make exact area measurement unreliable. The ratio alone neither
proves nor rules out completion. Use unclear when these coarse categories do
not describe the observed change reliably.

benchmark_completion_state:
- complete: coherent deliberate folded layers + compact_rectangle/square +
  substantial garment parts contained; no substantial unfinished open region
- partial: credible folding progress but still open / extended / an unfinished
  long strip / substantial sleeves or legs protruding
- incorrect: visibly wrong final result, such as a triangular bundle, rolled
  cylinder, scrunched wad, or garment still open without folding progress
- unclear: cannot reliably assess structure, whole shape, or containment

STRONG positive folding evidence (for endpoint_fold_evidence only):
- visible fold edge / deliberately stacked layers / one garment panel over another

WEAK evidence (never sufficient alone for complete):
- footprint smaller / more compact / held in gripper / local bend

Surface color/pattern change after flip is normal and is NOT object switch
by itself. WM collapse / melt / vanish differs from temporary arm occlusion.

Return ONLY one JSON object:
{{
  "end_description": "...",
  "target_id": "garment_1",
  "target_identity_consistent": "yes|no|unknown",
  "target_visibility": "clear|partial|occluded|unreadable",
  "final_fold_state": "spread_flat|crumpled|folded_layered|unclear",
  "final_shape": "square|compact_rectangle|long_strip|triangle|irregular|unclear",
  "fold_edge_visible": "yes|no|unknown",
  "double_or_multiple_layer_visible": "yes|no|unknown",
  "stacked_region_visible": "yes|no|unknown",
  "full_open_garment": "yes|no|unknown",
  "footprint_change": "smaller|similar|larger|unknown",
  "footprint_ratio_category": "near_original|around_half|around_quarter_or_smaller|unclear",
  "regularity": "roughly_regular|irregular|unknown",
  "garment_parts_contained": "yes|no|unknown",
  "remaining_unfolded_parts": "...",
  "benchmark_completion_state": "complete|partial|incorrect|unclear",
  "wm_collapse": "none|mild|severe",
  "endpoint_fold_evidence": "supports_fold|contradicts_fold|uncertain",
  "endpoint_completion_evidence": "supports_complete|supports_partial|contradicts_complete|uncertain",
  "endpoint_reason": "...",
  "last_frame_readable": true
}}
"""

_STAGE3 = """\
You are Stage3 (TEMPORAL FOLD JUDGE + FINAL VERDICT) for a FOLD CLOTH (GARMENT)
evaluator. You give the final task completion decision.

Instruction: {instruction}

START_STATE:
{start_state}

END_STATE (final geometry analysis):
{end_state}

INPUTS: MIDDLE FRAME 1/2/3 triplets and END_TRIPLET.

You must answer:
1) What deformable manipulation happened over time?
2) Was the benchmark Fold Cloth task completed?

VERY IMPORTANT — SPARSE FRAME SAMPLING:

You only see three uniformly sampled middle frames plus END.
Real folding actions can occur between sampled frames.
Do NOT require observing every sleeve, leg, or body fold, a specific sequence,
or a fixed number of folds.
Use END_STATE and the final imagery as primary evidence of final geometry;
use temporal evidence to verify manipulation, target continuity, and whether
folding was undone.

observed_fold_events:
- none: no credible fold event in sampled frames
- one_or_more: at least one real fold is visible; this does not mean only one
  fold occurred in the whole video
- multiple: clearly multiple distinct fold events / stages are visible
- unclear: cannot tell

fold_progress = completed means the WHOLE garment has reached the intended
compact rectangular / square folded state, not merely that a fold occurred.

If END_STATE benchmark_completion_state = partial, completion_status = failure
and fold_progress = in_progress, even when a valid fold and self-overlap are
visible. If END_STATE benchmark_completion_state = incorrect,
completion_status = failure.

SUCCESS is allowed when ALL hold:
1. The correct garment is manipulated and its identity remains consistent;
2. Temporal evidence is consistent with deliberate folding, rather than only
   lifting / sliding / scrunching / rolling;
3. END_STATE indicates complete, or END imagery clearly supports coherent
   folded layers in an approximately rectangular / square compact bundle with
   substantial sleeves / legs / extensions contained;
4. No clear unfolding of the completed result, wrong-object manipulation,
   severe WM collapse, or substantial remaining unfinished garment region.

Only one visible fold event can be enough when the final result is complete.
No sampled fold event is not automatically a failure if START-to-END evidence
supports a genuine folding change and MID frames are consistent with it.
An already folded garment that is merely lifted / moved / held does not prove
completion of a new folding task. If START is already partly folded, a further
fold that completes the target is sufficient; no fixed area reduction applies.

FAIL when any of:
- only lifting / sliding / dragging / pinching an open or already folded garment
- random wrinkling / scrunching / crumpling / rolling instead of folding
- temporary completion followed by unfolding back to an open or partial state
- wrong object / clear object switch
- severe target melt / vanish / WM collapse
- END remains partial, triangular, rolled, or otherwise clearly inconsistent
  with a compact rectangular / square layered garment

If severe occlusion or unreadable imagery prevents assessment of the whole
endpoint or target identity, use completion_status = uncertain rather than
inventing hidden folds. A plausible silhouette under occlusion is insufficient.
Temporary arm occlusion, small wrinkles, and minor protruding details are
acceptable when the completion evidence is otherwise clear.
Do NOT require perfect edges or gripper release.
For reopen_to_flat, record a return to an open garment. If folding is partially
undone without becoming fully open, explain it in trajectory_summary and judge
the actual END state. Earlier reopening followed by successful refolding does
not invalidate a clearly completed final result.

Return ONLY one JSON object:
{{
  "target_id": "garment_1",
  "observed_action": "fold|fold_attempt|lift_only|slide_only|crumple_or_scrunch|other|unclear",
  "observed_fold_events": "none|one_or_more|multiple|unclear",
  "same_target_throughout": "yes|no|unknown",
  "clear_layering_seen_mid": "yes|no|unknown",
  "panel_crossing_or_flip_seen": "yes|no|unknown",
  "self_overlap_established": "yes|no|unknown",
  "reopen_to_flat": "yes|no|unknown",
  "fold_progress": "none|in_progress|completed|reopened|unclear",
  "wm_collapse": "none|mild|severe",
  "completion_status": "success|failure|uncertain",
  "completion_confidence": "high|medium|low",
  "completion_reason": "...",
  "trajectory_summary": "..."
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
    prompt = _STAGE1.format(instruction=instruction or "")
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
            f"need 3 mid triplets for fold_cloth prompts, got {len(mid_triplets)}"
        )
    mid1, mid2, mid3 = mid_triplets[0], mid_triplets[1], mid_triplets[2]
    prompt = _STAGE3.format(
        instruction=instruction or "",
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
        {"type": "text", "text": "END_TRIPLET:"},
        {"type": "image", "image": end_triplet},
    ]
