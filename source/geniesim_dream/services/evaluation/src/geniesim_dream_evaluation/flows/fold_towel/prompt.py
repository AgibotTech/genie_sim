"""Stage1/2/3 prompts for fold towel V2. No bread/doll prompt imports."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union


_STAGE1 = """\
You are Stage1 for a FOLD TOWEL evaluator.

Instruction: {instruction}

INPUT: START_TOP only. Do NOT judge task success.

A towel/cloth is deformable.
Wrinkles or small local bends do not automatically mean folded.
A spread towel can be imperfectly flat.
A crumpled / scrunched towel is not the same as a deliberately folded towel.
Do not use the instruction itself as evidence that the towel is folded.

TASK:
1. Identify physical towel / cloth candidates.
2. Choose the instruction-relevant primary target.
3. Describe initial appearance and geometry.
4. Set initial_fold_state.

initial_fold_state meanings:
- spread_flat: mostly single-layer open sheet (can be imperfectly flat)
- partially_folded: some folding started but not a clear final layered fold
- crumpled: irregular multi-direction bunching / wad
- folded_layered: already shows coherent layered fold
- unclear: cannot tell

Vision-first: judge from the image, not from the instruction wording.

Return ONLY one JSON object:
{{
  "start_description": "...",
  "task_family": "fold_towel",
  "task_action": "fold",
  "target_candidates": [
    {{
      "id": "towel_1",
      "appearance": "...",
      "is_towel_like": true,
      "identity_confidence": "high|medium|low",
      "initial_fold_state": "spread_flat|partially_folded|crumpled|folded_layered|unclear",
      "initial_visibility": "clear|partial|unknown",
      "spatial_anchor": "..."
    }}
  ],
  "primary_target_id": "towel_1|null",
  "scene_readable": true
}}
"""

_STAGE2 = """\
You are Stage2 (FINAL FOLD GEOMETRY) for a FOLD TOWEL evaluator.

Instruction: {instruction}

INPUTS: START_TOP, END_TOP, END_TRIPLET, and parsed START_STATE.

START_STATE:
{start_state}

PRIMARY VIEW POLICY:
TOP-HEAD is primary for final folding geometry.
Left/right wrist views in the END_TRIPLET are secondary evidence only.
Do NOT let a close-up bent towel in a wrist crop override the head-view
overall geometry.

Do NOT output final task success/failure here. Analyze endpoint geometry and
benchmark completion level only.

final_fold_state (structural taxonomy):
- spread_flat: essentially a full/single-layer open sheet at END
- crumpled: irregular multi-direction wrinkles / scrunched wad / collapse
- folded_layered: coherent self-overlap / layering (one panel over another)
- unclear: cannot tell

BENCHMARK-SPECIFIC GT TASK PRIOR:

The instruction only says "Fold Towel", but in this benchmark the intended
GT task has a specific multi-step folding procedure.

The towel starts as a spread rectangular sheet.

The reference execution performs 2 to 3 half-fold operations:
1. First, fold approximately top-to-bottom along one axis of the towel
   (relative to the towel layout on the table, not camera axes).
2. Then, fold approximately left-to-right along the other axis.
3. Some cases contain one additional half-fold along one of these axes.

Because the five sampled frames may skip intermediate actions, do NOT require
the exact fold sequence to be visually observed here. Stage2 judges the END
geometry.

The intended completed END state is:
- a compact rectangle or square;
- a coherent multi-layer structure;
- roughly around one quarter of the initial START footprint after two
  half-folds, or smaller if an additional fold occurred.

This is a qualitative visual ratio, NOT an exact pixel measurement.
Use START_TOP vs END_TOP comparison; perspective and WM deformation make
exact area measurement unreliable.

IMPORTANT:
- One valid half-fold is only PARTIAL progress, not completion.
- A long strip / large double-layer rectangle occupying roughly half of the
  START footprint is typically a one-fold partial state.
- A triangular folded shape is NOT the intended completed GT state.
- Crumpled / scrunched / irregular wad is NOT successful folding.
- Do not call a state complete merely because it is layered.

final_fold_state answers:
"Is the towel structurally spread / crumpled / layered?"

benchmark_completion_state answers:
"Has the benchmark-specific 2–3-half-fold target been completed?"

A towel may be:
final_fold_state = folded_layered
but
benchmark_completion_state = partial

Examples:
A spread towel folded once into a long double-layer strip:
- final_fold_state = folded_layered
- final_shape = long_strip
- footprint_ratio_category = around_half
- benchmark_completion_state = partial
- endpoint_fold_evidence = supports_fold
- endpoint_completion_evidence = supports_partial

A towel ending as a compact square/rectangle around one-quarter of START:
- final_fold_state = folded_layered
- final_shape = square or compact_rectangle
- footprint_ratio_category = around_quarter_or_smaller
- benchmark_completion_state = complete
- endpoint_completion_evidence = supports_complete

A layered triangular endpoint:
- final_fold_state may still be folded_layered
- final_shape = triangle
- benchmark_completion_state = incorrect
- endpoint_completion_evidence = contradicts_complete

Decision order:
1. First decide final_fold_state.
2. Then decide final_shape.
3. Compare END footprint qualitatively against START
   (footprint_ratio_category).
4. Then decide benchmark_completion_state from shape + ratio + layering.
Do NOT infer benchmark completion only from visible layers.
Do NOT treat footprint_ratio_category as a hard numeric threshold alone.

final_shape:
- square: overall near-square
- compact_rectangle: compact rectangle with BOTH axes clearly shrunk vs START
  (NOT a single-axis half-fold strip)
- long_strip: elongated one-fold-like strip, often ~half footprint
- triangle: triangular / diagonal fold (not GT complete)
- irregular: irregular but not necessarily crumpled
- unclear

footprint_ratio_category (visual estimate only):
- near_original / around_half / around_quarter_or_smaller / unclear

benchmark_completion_state:
- complete: coherent folded_layered + compact_rectangle/square + roughly
  around_quarter_or_smaller, consistent with multiple half-folds
- partial: real fold exists but one-fold-like (long_strip / large layered
  rectangle / around_half)
- incorrect: folded but wrong GT geometry (e.g. triangle)
- unclear: cannot tell

STRONG positive folding evidence (for endpoint_fold_evidence only):
- visible fold edge / double-or-multiple layers / panel over panel

WEAK evidence (never sufficient alone for complete):
- footprint smaller / more compact / held in gripper / local bend

Surface color/pattern change after flip is normal and is NOT object switch
by itself. WM collapse / melt / vanish differs from temporary arm occlusion.

Return ONLY one JSON object:
{{
  "end_description": "...",
  "target_id": "towel_1",
  "target_identity_consistent": "yes|no|unknown",
  "target_visibility": "clear|partial|occluded|unreadable",
  "final_fold_state": "spread_flat|crumpled|folded_layered|unclear",
  "final_shape": "square|compact_rectangle|long_strip|triangle|irregular|unclear",
  "fold_edge_visible": "yes|no|unknown",
  "double_or_multiple_layer_visible": "yes|no|unknown",
  "stacked_region_visible": "yes|no|unknown",
  "full_single_layer_sheet": "yes|no|unknown",
  "footprint_change": "smaller|similar|larger|unknown",
  "footprint_ratio_category": "near_original|around_half|around_quarter_or_smaller|unclear",
  "regularity": "roughly_regular|irregular|unknown",
  "benchmark_completion_state": "complete|partial|incorrect|unclear",
  "wm_collapse": "none|mild|severe",
  "endpoint_fold_evidence": "supports_fold|contradicts_fold|uncertain",
  "endpoint_completion_evidence": "supports_complete|supports_partial|contradicts_complete|uncertain",
  "endpoint_reason": "...",
  "last_frame_readable": true
}}
"""

_STAGE3 = """\
You are Stage3 (TEMPORAL FOLD JUDGE + FINAL VERDICT) for a FOLD TOWEL
evaluator. You give the final task completion decision.

Instruction: {instruction}

START_STATE:
{start_state}

END_STATE (final geometry analysis):
{end_state}

INPUTS: MIDDLE FRAME 1/2/3 triplets and END_TRIPLET.

You must answer:
1) What deformable manipulation happened over time?
2) Was the benchmark Fold Towel task completed?

VERY IMPORTANT — SPARSE FRAME SAMPLING:

You only see three uniformly sampled middle frames plus END.
The real video may contain multiple half-fold operations between sampled frames.

Therefore:
- If multiple distinct fold events are visible, this is strong positive evidence.
- If only one valid fold event is visible, this is STILL compatible with a
  completed multi-fold task.
- Do NOT conclude that "only one fold occurred in the whole video" merely
  because only one fold event is visible in the sampled frames.
- Do NOT fail solely because the top-to-bottom and left-to-right folds are not
  both explicitly visible.
- Use END_STATE benchmark_completion_state and final geometry as the primary
  evidence for whether the multi-fold target was completed.

observed_fold_events:
- none: no credible fold event in sampled frames
- one_or_more: at least one real fold visible; this is NOT negative evidence
  and does NOT mean the whole video had only one fold
- multiple: clearly multiple distinct fold events/stages in sampled frames
- unclear: cannot tell

fold_progress = completed means the FULL benchmark Fold Towel task is done
(trajectory + END geometry), NOT merely that one half-fold action occurred.

If END_STATE benchmark_completion_state = partial (e.g. long_strip /
around_half one-fold-like), then:
- fold_progress = in_progress
- completion_status = failure
even if observed_action = fold and self_overlap_established = yes.
Valid folding action != full benchmark completion.

If END_STATE benchmark_completion_state = incorrect (e.g. triangle),
completion_status = failure.

If END_STATE benchmark_completion_state = complete (or END imagery clearly
supports the compact ~quarter multi-layer GT endpoint), SUCCESS is allowed
even when observed_fold_events = one_or_more.

SUCCESS is allowed when ALL hold:
1. Correct towel is manipulated;
2. Temporal evidence is consistent with folding rather than only lift/slide/
   crumple;
3. END_STATE benchmark_completion_state is complete OR the END imagery itself
   clearly supports the benchmark-complete endpoint;
4. No clear reopen-to-flat / wrong-object / severe WM collapse contradiction.

Seeing only one sampled fold event does NOT prevent success.

FAIL when any of:
- only lifting / sliding / dragging / pinching
- random wrinkling / scrunching into a wad / crumpling
- temporary fold followed by reopen-to-flat
- wrong object / clear object switch
- severe target melt / vanish / WM collapse
- END is only one-fold partial / incorrect GT geometry (even if layered)

Temporary arm occlusion is tolerable. Severe WM collapse is not.
Do NOT require neat edges or gripper release.
Do NOT require observing every half-fold event in the MID frames.

Return ONLY one JSON object:
{{
  "target_id": "towel_1",
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
            f"need 3 mid triplets for fold_towel v2, got {len(mid_triplets)}"
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
