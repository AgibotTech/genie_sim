# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Join exact-frame RGB and target evidence; no model calls or simulator access."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from agent2act.adapters.planners.tool_center import ToolCenter

from .projection import image_projection


def _images(snapshot: dict[str, Any], key: str, resolve: Callable[[str], Path]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    camera = None
    for item in snapshot.get("content", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "input_text":
            text = item.get("text", "")
            camera = text.removeprefix("current camera=") if text.startswith("current camera=") else camera
            try:
                metadata = json.loads(text)
            except ValueError:
                continue
            if isinstance(metadata, dict) and "calibration" in metadata and found:
                if metadata.get("camera") == found[-1]["camera"] and metadata.get("image_role") == "current":
                    found[-1]["calibration"] = metadata["calibration"]
        elif item.get("type") == "input_image" and camera and " " not in camera:
            ref = item.get("image_url")
            if isinstance(ref, dict) and "$artifact" in ref:
                found.append({"camera": camera, "ref": ref["$artifact"], "calibration": {}})
    # Policy/continuation/old recordings can still show the source RGB without pretending it was sent.
    if not found:
        for camera, item in snapshot.get("images", {}).items():
            if isinstance(item, dict) and item.get("encoding") == "JPEG":
                ref = item.get("image_data")
                if isinstance(ref, dict) and "$artifact" in ref:
                    calibration = item.get("calibration", {})
                    if (
                        isinstance(calibration, dict)
                        and (calibration.get("snapshot_id") or item.get("snapshot_id"))
                        and calibration.get("snapshot_id") != item.get("snapshot_id")
                    ):
                        calibration = {
                            "available": False,
                            "reason": "RGB and calibration snapshot_id do not match",
                        }
                    found.append(
                        {
                            "camera": camera,
                            "ref": ref["$artifact"],
                            "calibration": calibration,
                        }
                    )
    result = []
    for item in found:
        try:
            relative = f"{key}/{item.pop('ref')}"
            with Image.open(resolve(relative)) as image:
                size = image.size
            result.append({**item, "path": relative, "size": size})
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
    return result


def _pose(entry: Any) -> Any:
    if isinstance(entry, dict):
        p, q = entry.get("position"), entry.get("orientation")
        return p + q if isinstance(p, list) and isinstance(q, list) else None
    return None


def build_views(
    records: list[dict[str, Any]], key: str, resolve: Callable[[str], Path]
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    observation: dict[str, Any] = {}
    view: dict[str, Any] | None = None
    has_model_snapshot = any(r["kind"] == "observer-input" for r in records)
    for record in records:
        kind, value = record["kind"], record["value"]
        if not isinstance(value, dict):
            continue
        if kind == "observer-observation":
            observation = value
        if kind == "simulator-request" and not observation:
            params = value.get("params", {})
            observation = {
                "images": params.get("images", {}),
                "end_pose": params.get("states", {}).get("end_pose", {}),
            }
        if kind == "observer-input" or (view is None and kind == "runtime-event" and not has_model_snapshot):
            snapshot = value if kind == "observer-input" else observation
            view = {
                "id": record["name"],
                "attempt": snapshot.get("planning_attempt"),
                "snapshot": snapshot,
                "decision": None,
                "execution_decision": None,
                "issued": None,
                "images": _images(snapshot, key, resolve),
                "reference": snapshot.get("reference", "wrist"),
                "image_source": "model_input" if kind == "observer-input" else "observation",
            }
            views.append(view)
        if view is None:
            continue
        if kind == "planner-output-decision":
            view["decision"] = value.get("decision")
        elif kind == "gripper-center-execution-decision":
            view["execution_decision"] = value
            view["converted_model_decision"] = value
        elif kind == "runtime-event" and value.get("status") == "planned":
            view["execution_decision"] = value.get("decision")
            if view["decision"] is None:
                view["decision"] = value.get("decision")
        elif kind == "observer-action":
            view["issued"] = {**value, "source": record["name"]}
        elif kind == "runtime-event" and value.get("status") == "issued":
            view["issued_event"] = value
        elif kind == "simulator-response" and view.get("issued") is None:
            # Older traces keep compiler FK in response metadata (not in wire joint values).
            event = view.get("issued_event", {})
            metadata = value.get("result", {}).get("agent2act", {})
            path, steps = metadata.get("cartesian_path"), event.get("issued_steps")
            if (
                event.get("kind") == "JOINT_ABS"
                and isinstance(path, dict)
                and type(steps) is int
                and steps > 0
            ):
                view["issued"] = {
                    "source": record["name"],
                    "kind": "JOINT_ABS",
                    "frame": event.get("frame"),
                    "steps": steps,
                    "endpoint_fk": {
                        arm: poses[steps - 1]
                        for arm, poses in path.items()
                        if isinstance(poses, list) and len(poses) >= steps
                    },
                }
    if not views and observation:
        views.append(
            {
                "id": "observation",
                "attempt": None,
                "snapshot": observation,
                "decision": None,
                "execution_decision": None,
                "issued": None,
                "images": _images(observation, key, resolve),
                "reference": "wrist",
                "image_source": "observation",
            }
        )
    for view in views:
        _markers(view)
        view.pop("snapshot")
    return views


def _markers(view: dict[str, Any]) -> None:
    snapshot = view["snapshot"]
    center = None
    if view["reference"] == "gripper_center":
        try:
            center = ToolCenter(snapshot.get("gripper_center_offset_m"))
        except (ValueError, TypeError):
            pass
    points: list[dict[str, Any]] = []

    def add(
        kind: str,
        arm: str,
        pose: Any,
        frame: str | None,
        source: str,
        *,
        wrist: bool = False,
        index: int | None = None,
    ) -> None:
        if not isinstance(pose, list) or len(pose) != 7:
            return
        try:
            if wrist and view["reference"] == "gripper_center":
                if center is None:
                    return
                pose = center.convert(pose, arm)
        except ValueError:
            return
        points.append(
            {"kind": kind, "arm": arm, "pose": pose, "frame": frame, "source": source, "waypoint": index}
        )

    for frame, arms in snapshot.get("end_pose", {}).items():
        for arm in ("left", "right"):
            add("current", arm, _pose(arms.get(arm + "_arm")), frame, view["id"])
    # Legacy traces without an observer snapshot have wrist observations. Use the
    # recorded conversion, not raw model TCP poses, for the common-reference overlay.
    converted = view["reference"] == "wrist" and view.get("converted_model_decision")
    decision = converted or view.get("decision") or {}
    target_source = "gripper-center-execution-decision (wrist)" if converted else "planner-output-decision"
    motion = decision.get("motion") or {}
    candidate_mode = "candidates" in motion
    for index, waypoint in enumerate(motion.get("waypoints", motion.get("candidates", [])), 1):
        for arm in ("left", "right"):
            add(
                "candidate" if candidate_mode else "target",
                arm,
                waypoint.get(arm + "_pose"),
                motion.get("frame"),
                target_source,
                index=index,
            )
    if candidate_mode:
        selected = (view.get("execution_decision") or {}).get("motion") or {}
        for index, waypoint in enumerate(selected.get("waypoints", []), 1):
            for arm in ("left", "right"):
                add(
                    "target",
                    arm,
                    waypoint.get(arm + "_pose"),
                    selected.get("frame"),
                    "runtime planned decision",
                    wrist=True,
                    index=index,
                )
    issued = view.get("issued")
    if issued:
        for arm in ("left", "right"):
            poses = (
                issued.get("endpoint", {})
                if issued.get("kind") == "EEF_ABS"
                else issued.get("endpoint_fk", {})
            )
            add(
                "issued",
                arm,
                poses.get(arm),
                issued.get("frame"),
                str(issued.get("source", "observer-action"))
                + (" / recorded FK" if issued.get("kind") != "EEF_ABS" else ""),
                wrist=True,
            )
    elif view.get("issued_event", {}).get("kind") == "EEF_ABS":
        event = view["issued_event"]
        for arm in ("left", "right"):
            add(
                "issued",
                arm,
                event.get("issued_endpoint", {}).get(arm + "_arm"),
                event.get("frame"),
                "runtime issued",
                wrist=True,
            )
    for image in view["images"]:
        image["markers"] = [
            {
                **point,
                **(
                    image_projection(image["calibration"], point["pose"], tuple(image["size"]))
                    if point["frame"] == "base_link"
                    else {"pixel": None, "reason": "no transform from this reference frame to the camera yet"}
                ),
            }
            for point in points
        ]
    view["points"] = points
