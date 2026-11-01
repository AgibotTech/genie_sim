"""Judge one WM video with dream_doll_inout_release.

Mirrored from v5a4_hover_res with instruction-routed PUT/TAKE prompts.
Same Stage1-4 topology and Stage4 multi-target orchestration.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from geniesim_dream_evaluation.frames import (
    crop_start_end_top,
    sample_five_frames,
    split_triplet_views,
)
from geniesim_dream_evaluation.parse import parse_json_object
from .prompt import (
    build_end_content,
    build_start_content,
    build_verify_content,
    build_wrist_recheck_content,
)
from .resolver import (
    derive_end_completion_candidate,
    resolve_task_completion,
    select_wrist_recheck_targets,
)
from .semantics import (
    instruction_action,
    normalize_end_state,
    normalize_start_state,
    normalize_trajectory_state,
    normalize_wrist_state,
)
from geniesim_dream_evaluation.vlm_backend import run_qwen_vl_interleaved

STAGE1_MAX_NEW_TOKENS = 768
# Allow enough output tokens to avoid truncating the end-state JSON.
STAGE2_MAX_NEW_TOKENS = 2048
STAGE3_MAX_NEW_TOKENS = 1024
# Keep unchanged for controlled comparison with the previous v5a1 run.
STAGE4_MAX_NEW_TOKENS = 1024


def _next_state(parsed: Optional[Dict[str, Any]], raw: str):
    if parsed is not None:
        return parsed
    return raw or "(unavailable)"


def _extract_requested_wrist_object(
    wrist_state: Optional[Dict[str, Any]],
    requested_object_id: str,
) -> Optional[Dict[str, Any]]:
    """Return the unique Stage4 object matching requested_object_id."""
    if not isinstance(wrist_state, dict):
        return None

    raw_objects = wrist_state.get("objects")
    if not isinstance(raw_objects, list):
        return None

    matches: List[Dict[str, Any]] = []
    for obj in raw_objects:
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("object_id") or "").strip()
        if object_id == requested_object_id:
            matches.append(dict(obj))

    if len(matches) != 1:
        return None
    return matches[0]


def _format_raw_stage4_by_target(raw_by_target: Dict[str, str]) -> str:
    chunks = []
    for object_id, raw in raw_by_target.items():
        chunks.append(f"===== {object_id} =====\n{raw}")
    return "\n\n".join(chunks)


def _run_wrist_rechecks_per_target(
    model,
    processor,
    instruction: str,
    left_wrist: Any,
    right_wrist: Any,
    recheck_targets: List[Dict[str, Any]],
) -> Tuple[
    Optional[Dict[str, Any]],
    Dict[str, str],
    Dict[str, str],
]:
    """Run one Stage4 VLM call per selected wrist target; aggregate results."""
    collected_objects: List[Dict[str, Any]] = []
    raw_by_target: Dict[str, str] = {}
    errors_by_target: Dict[str, str] = {}

    for target in recheck_targets:
        if not isinstance(target, dict):
            continue

        object_id = str(target.get("object_id") or "").strip()
        if not object_id:
            continue

        try:
            raw4 = run_qwen_vl_interleaved(
                model,
                processor,
                build_wrist_recheck_content(
                    instruction=instruction,
                    left_wrist=left_wrist,
                    right_wrist=right_wrist,
                    recheck_targets=[target],
                    schema_object_id=object_id,
                ),
                max_new_tokens=STAGE4_MAX_NEW_TOKENS,
            )
        except Exception as e:
            errors_by_target[object_id] = f"stage4_call_failed: {e}"
            continue

        raw_by_target[object_id] = raw4 if isinstance(raw4, str) else str(raw4)

        parsed = parse_json_object(raw4)
        normalized = normalize_wrist_state(parsed)
        wrist_obj = _extract_requested_wrist_object(normalized, object_id)
        if wrist_obj is None:
            errors_by_target[object_id] = (
                "missing_or_duplicate_requested_object_id"
            )
            continue

        collected_objects.append(wrist_obj)

    if not collected_objects:
        return None, raw_by_target, errors_by_target

    wrist_state = {
        "objects": collected_objects,
        "wrist_summary": (
            "Aggregated independent per-target Stage4 wrist rechecks."
        ),
    }
    return wrist_state, raw_by_target, errors_by_target


def _run_selected_wrist_rechecks(
    model,
    processor,
    instruction: str,
    left_wrist: Any,
    right_wrist: Any,
    recheck_targets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Dispatch Stage4: baseline single-target vs per-target multi-target."""
    if len(recheck_targets) == 1:
        object_id = str(
            recheck_targets[0].get("object_id") or ""
        ).strip()
        raw4 = run_qwen_vl_interleaved(
            model,
            processor,
            build_wrist_recheck_content(
                instruction=instruction,
                left_wrist=left_wrist,
                right_wrist=right_wrist,
                recheck_targets=recheck_targets,
                schema_object_id=None,
            ),
            max_new_tokens=STAGE4_MAX_NEW_TOKENS,
        )
        wrist_state = normalize_wrist_state(parse_json_object(raw4))
        raw_by_target = {object_id: raw4} if object_id else {}
        errors_by_target: Dict[str, str] = {}
        completed: List[str] = []
        if object_id and isinstance(wrist_state, dict):
            for obj in wrist_state.get("objects") or []:
                if (
                    isinstance(obj, dict)
                    and str(obj.get("object_id") or "").strip() == object_id
                ):
                    completed.append(object_id)
                    break
            if not completed:
                errors_by_target[object_id] = (
                    "requested_object_id_not_present_in_single_target_stage4_output"
                )
        return {
            "wrist_state": wrist_state,
            "raw_stage4": raw4,
            "raw_by_target": raw_by_target,
            "errors_by_target": errors_by_target,
            "completed_target_ids": completed,
        }

    wrist_state, raw_by_target, errors_by_target = (
        _run_wrist_rechecks_per_target(
            model=model,
            processor=processor,
            instruction=instruction,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            recheck_targets=recheck_targets,
        )
    )
    completed = []
    if isinstance(wrist_state, dict):
        completed = [
            str(o.get("object_id"))
            for o in (wrist_state.get("objects") or [])
            if isinstance(o, dict) and o.get("object_id")
        ]
    return {
        "wrist_state": wrist_state,
        "raw_stage4": _format_raw_stage4_by_target(raw_by_target),
        "raw_by_target": raw_by_target,
        "errors_by_target": errors_by_target,
        "completed_target_ids": completed,
    }


def judge_pair(model, processor, pair: Dict[str, Any]) -> Dict[str, Any]:
    """
    Three greedy VLM stages (+ optional Stage4 wrist) then Python resolver.

    Same 5-frame cache as v5. Stage4 only runs when select_wrist_recheck_targets
    returns a non-empty list. One selected target uses the v5a4 baseline Stage4
    path; multiple selected targets get independent Stage4 calls.
    """
    result: Dict[str, Any] = {
        "episode": pair.get("episode"),
        "wm_path": pair.get("wm_path"),
        "instruction": pair.get("instruction", ""),
        "frame_cache": None,
        "success": None,
        "thinking": "",
        "raw_response": "",
        "raw_stage1": "",
        "raw_stage2": "",
        "raw_stage3": "",
        "raw_stage4": "",
        "raw_stage4_by_target": {},
        "start_state": None,
        "end_state": None,
        "verify": None,
        "end_completion_candidate": None,
        "wrist_recheck_triggered": False,
        "wrist_recheck_target_ids": [],
        "wrist_recheck_completed_target_ids": [],
        "wrist_recheck_errors_by_target": {},
        "wrist_recheck": None,
        "wrist_recheck_error": None,
        "resolver_reason": None,
        "resolver_debug": None,
        "error": None,
        "prompt_family": "container",
        "judge_version": "dream_doll_inout_release",
        "task_family": "doll_inout",
        "task_action": None,
    }

    wm_path = pair.get("wm_path")

    if not wm_path or not os.path.isfile(wm_path):
        result["error"] = "wm_missing"
        return result

    wm_frames = sample_five_frames(wm_path)
    result["frame_cache"] = None
    if not wm_frames:
        result["error"] = "wm_no_frames"
        return result
    if len(wm_frames) < 5:
        result["error"] = "wm_too_few_frames"
        return result

    try:
        start_top, end_top = crop_start_end_top(wm_frames[0], wm_frames[-1])
    except Exception as e:
        result["error"] = f"top_crop_failed: {e}"
        return result

    mid_triplets = [wm_frames[1], wm_frames[2], wm_frames[3]]
    end_triplet = wm_frames[-1]
    instruction = pair.get("instruction", "") or ""
    result["task_action"] = instruction_action(instruction)

    try:
        raw1 = run_qwen_vl_interleaved(
            model,
            processor,
            build_start_content(instruction, start_top),
            max_new_tokens=STAGE1_MAX_NEW_TOKENS,
        )
    except Exception as e:
        result["error"] = f"stage1: {e}"
        return result
    result["raw_stage1"] = raw1
    start_state = normalize_start_state(parse_json_object(raw1), instruction)
    result["start_state"] = start_state

    try:
        raw2 = run_qwen_vl_interleaved(
            model,
            processor,
            build_end_content(
                instruction=instruction,
                start_frame=start_top,
                end_frame=end_top,
                end_triplet=end_triplet,
                start_state=_next_state(start_state, raw1),
            ),
            max_new_tokens=STAGE2_MAX_NEW_TOKENS,
        )
    except Exception as e:
        result["error"] = f"stage2: {e}"
        return result
    result["raw_stage2"] = raw2
    end_state = normalize_end_state(parse_json_object(raw2))
    result["end_state"] = end_state
    result["end_completion_candidate"] = derive_end_completion_candidate(
        instruction=instruction,
        start_state=start_state if isinstance(start_state, dict) else None,
        end_state=end_state if isinstance(end_state, dict) else None,
    )

    try:
        raw3 = run_qwen_vl_interleaved(
            model,
            processor,
            build_verify_content(
                instruction=instruction,
                mid_triplets=mid_triplets,
                end_triplet=end_triplet,
                start_state=_next_state(start_state, raw1),
                end_state=_next_state(end_state, raw2),
            ),
            max_new_tokens=STAGE3_MAX_NEW_TOKENS,
        )
    except Exception as e:
        result["error"] = f"stage3: {e}"
        return result
    result["raw_stage3"] = raw3
    result["raw_response"] = raw3
    verify = normalize_trajectory_state(parse_json_object(raw3))
    result["verify"] = verify

    if isinstance(verify, dict):
        summary = verify.get("trajectory_summary")
        if isinstance(summary, str) and summary.strip():
            result["thinking"] = summary.strip()
        else:
            result["thinking"] = json.dumps(verify, ensure_ascii=False)
    else:
        result["thinking"] = raw3 or ""

    # --- optional Stage4 ---
    wrist_state = None
    recheck_targets = select_wrist_recheck_targets(
        instruction=instruction,
        start_state=start_state if isinstance(start_state, dict) else None,
        end_state=end_state if isinstance(end_state, dict) else None,
        verify_state=verify if isinstance(verify, dict) else None,
    )
    result["wrist_recheck_target_ids"] = [
        str(t.get("object_id"))
        for t in recheck_targets
        if isinstance(t, dict) and t.get("object_id")
    ]

    if recheck_targets:
        result["wrist_recheck_triggered"] = True
        try:
            _, left_wrist, right_wrist = split_triplet_views(end_triplet)
        except Exception as e:
            result["wrist_recheck_error"] = f"wrist_view_load_failed: {e}"
            wrist_state = None
        else:
            try:
                stage4 = _run_selected_wrist_rechecks(
                    model=model,
                    processor=processor,
                    instruction=instruction,
                    left_wrist=left_wrist,
                    right_wrist=right_wrist,
                    recheck_targets=recheck_targets,
                )
            except Exception as e:
                result["wrist_recheck_error"] = f"stage4: {e}"
                wrist_state = None
            else:
                wrist_state = stage4["wrist_state"]
                result["raw_stage4"] = stage4["raw_stage4"]
                result["raw_stage4_by_target"] = stage4["raw_by_target"]
                result["wrist_recheck_errors_by_target"] = stage4[
                    "errors_by_target"
                ]
                result["wrist_recheck_completed_target_ids"] = stage4[
                    "completed_target_ids"
                ]
                result["wrist_recheck"] = wrist_state
                errors_by_target = stage4["errors_by_target"]
                if len(recheck_targets) > 1:
                    if errors_by_target and not wrist_state:
                        result["wrist_recheck_error"] = (
                            "all_stage4_targets_failed: "
                            + json.dumps(errors_by_target, ensure_ascii=False)
                        )
                    elif errors_by_target:
                        result["wrist_recheck_error"] = (
                            "partial_stage4_failures: "
                            + json.dumps(errors_by_target, ensure_ascii=False)
                        )

    success, reason, resolver_debug = resolve_task_completion(
        instruction=instruction,
        start_state=start_state if isinstance(start_state, dict) else None,
        end_state=end_state if isinstance(end_state, dict) else None,
        verify_state=verify if isinstance(verify, dict) else None,
        wrist_state=wrist_state if isinstance(wrist_state, dict) else None,
    )
    result["success"] = success
    result["resolver_reason"] = reason
    result["resolver_debug"] = resolver_debug
    if isinstance(resolver_debug, dict):
        result["required_target_source"] = resolver_debug.get(
            "required_target_source"
        )
        result["recovered_target_ids"] = resolver_debug.get(
            "recovered_target_ids"
        )
    if success is None:
        result["error"] = reason
    return result
