"""Generic one-shot and three-stage prompts (no task-specific checklists)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union


_ONE_SHOT = """\
You are a robot task success judge. Decide ONLY from the images and the instruction.
Do NOT assume the instruction is true; verify it against what you see in the frames.
Do NOT invent task-specific checklists; judge only whether the instruction's
intended outcome is achieved in the video.

**Task Instruction**: {instruction}

**INPUT DESCRIPTION:**
- 5 images in time order (first → last) from the predicted / rollout video ONLY \
(no GT reference).
- Uniformly sampled: first frame, 3 middle frames, last frame.
- Each image is a horizontal stitch of three views: top head | left arm | right arm.
- Use the LEFT third (top head) as the primary view for scene understanding and \
task-completion judgment.
- Use the middle/right (left/right arm) views only to verify which object is \
being grasped or contacted.

**STEP 1 — Observe first (ignore the instruction wording initially):**
From the frames alone (prefer TOP-HEAD), identify:
1. Primary object(s) being grasped or contacted
2. Action type
3. Start → end state change
4. Mid-frame object switch, vanishing, or WM collapse

**STEP 2 — Match to the instruction (ALL must hold for success = 1):**
1. Target object MUST match the instruction. Near-miss objects = FAILURE.
2. Action intent must match the instruction.
3. Outcome: last frame shows the instructed completion state.
**Do NOT answer 1 merely because an action is in progress.**

**JUDGE AS FAILURE (0) if any of:**
- Wrong target object
- Wrong action type
- No completed outcome in the last frame
- Instruction-only hallucination
- Last-frame target unreadable due to severe WM collapse without earlier \
completion evidence

**OUTPUT FORMAT REQUIREMENTS:**
Please respond in this format:
thinking: [1. grasped/contacted object  2. action  3. collapse/object-switch?  \
4. object+action match?  5. last-frame outcome, not merely in-progress]
answer: [0 or 1]
"""


_STAGE1 = """\
You are Stage1 for a GENERIC robot task evaluator.

Instruction: {instruction}

INPUT: START_TOP only. Do NOT judge task success.

TASK:
1. Identify physical object candidates that may be manipulated.
2. Choose the instruction-relevant primary target (if any).
3. Describe initial appearance, location, and visibility.
4. Note initial scene readability.

Vision-first: judge from the image, not from the instruction wording alone.

Return ONLY one JSON object:
{{
  "start_description": "...",
  "task_family": "generic",
  "target_candidates": [
    {{
      "id": "obj_1",
      "appearance": "...",
      "identity_confidence": "high|medium|low",
      "initial_visibility": "clear|partial|unknown",
      "spatial_anchor": "..."
    }}
  ],
  "primary_target_id": "obj_1|null",
  "scene_readable": true
}}
"""


_STAGE2 = """\
You are Stage2 (FINAL ENDPOINT STATE) for a GENERIC robot task evaluator.

Instruction: {instruction}

INPUTS: START_TOP, END_TOP, END_TRIPLET, and parsed START_STATE.

START_STATE:
{start_state}

PRIMARY VIEW POLICY:
TOP-HEAD is primary for final scene / outcome geometry.
Left/right wrist views in the END_TRIPLET are secondary evidence only.

Do NOT output final task success/failure here. Analyze endpoint state only.

Focus on:
1. Whether the same primary target is still identifiable at END.
2. What changed from START to END (pose, location, contact, containment, etc.).
3. Whether the END frame is readable or degraded by WM collapse.
4. Endpoint evidence for / against instruction completion (not a final verdict).

WM collapse / melt / vanish differs from temporary arm occlusion.

Return ONLY one JSON object:
{{
  "end_description": "...",
  "target_id": "obj_1|null",
  "target_identity_consistent": "yes|no|unknown",
  "target_visibility": "clear|partial|occluded|unreadable",
  "start_to_end_change": "...",
  "endpoint_completion_evidence": \
"supports_complete|contradicts_complete|uncertain",
  "wm_collapse": "none|mild|severe",
  "endpoint_reason": "...",
  "last_frame_readable": true
}}
"""


_STAGE3 = """\
You are Stage3 (TEMPORAL JUDGE + FINAL VERDICT) for a GENERIC robot task \
evaluator. You give the final task completion decision.

Instruction: {instruction}

START_STATE:
{start_state}

END_STATE (endpoint analysis):
{end_state}

INPUTS: MIDDLE FRAME 1/2/3 triplets and END_TRIPLET.

You must answer:
1) What manipulation / interaction happened over time?
2) Was the instructed task completed?

VERY IMPORTANT — SPARSE FRAME SAMPLING:
You only see three uniformly sampled middle frames plus END.
Do NOT require every intermediate action to be visible.
Use END_STATE and last-frame outcome as primary evidence for completion.
Do NOT answer success merely because an action is in progress mid-video.

SUCCESS is allowed when ALL hold:
1. Correct target object relative to the instruction;
2. Action intent matches the instruction;
3. Last-frame / END evidence supports the instructed completion state;
4. No clear wrong-object / reopen-of-failure / severe WM collapse contradiction.

FAIL when any of:
- wrong target object
- wrong action type
- no completed outcome in the last frame
- instruction-only hallucination
- severe target melt / vanish / WM collapse without earlier completion evidence

Temporary arm occlusion is tolerable. Severe WM collapse is not.

Return ONLY one JSON object:
{{
  "target_id": "obj_1|null",
  "observed_action": "...",
  "same_target_throughout": "yes|no|unknown",
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


def build_one_shot_content(
    instruction: str,
    frames: List[Any],
) -> List[Dict[str, Any]]:
    if len(frames) < 5:
        raise ValueError(f"need 5 frames for generic one_shot, got {len(frames)}")
    prompt = _ONE_SHOT.format(instruction=instruction or "")
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, frame in enumerate(frames[:5]):
        content.append({"type": "text", "text": f"FRAME {i} (triplet):"})
        content.append({"type": "image", "image": frame})
    return content


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
            f"need 3 mid triplets for generic three_stage, got {len(mid_triplets)}"
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
