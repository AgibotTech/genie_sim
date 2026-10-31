# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build a readable local HTML report from JSONL events and referenced trace artifacts."""

import argparse
import html
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import quote

from agent2act.service.observer.projection import project_base_point


def image_references(value):
    """Find persisted image references in Responses and Chat Completions requests."""
    if isinstance(value, list):
        for child in value:
            yield from image_references(child)
    elif isinstance(value, dict):
        for key, child in value.items():
            if key in ("image_url", "url") and isinstance(child, dict) and "$artifact" in child:
                yield child
            else:
                yield from image_references(child)


def xyz(values):
    """Show XYZ only for a complete finite-looking position, never fill unknown axes."""
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return "unknown"
    if any(type(value) not in (int, float) for value in values[:3]):
        return "unknown"
    return ", ".join(f"{value:.3f}" for value in values[:3]) + " m"


def object_location(report):
    location = report.get("object_location") if isinstance(report, dict) else None
    if not isinstance(location, dict):
        return "not reported"
    coordinates = [location.get(axis) for axis in ("x_m", "y_m", "z_m")]
    result = f"{location.get('label') or 'unknown object'}: "
    result += f"{xyz(coordinates)} ({location.get('frame') or 'frame unknown'}; {location.get('basis') or 'basis unknown'})"
    if location.get("camera") and all(
        type(location.get(axis)) in (int, float) for axis in ("u_norm", "v_norm")
    ):
        result += f"; {location['camera']} image ({location['u_norm']:.0f}, {location['v_norm']:.0f}) / 1000"
    result += f"; source: {location.get('source') or 'unknown'}"
    return result


def motion_target(decision):
    motion = decision.get("motion") if isinstance(decision, dict) else None
    if not isinstance(motion, dict):
        return "no motion target"
    waypoints = motion.get("waypoints")
    if not isinstance(waypoints, list):
        return "no waypoints"
    frame = motion.get("frame") or "frame unknown"
    parts = []
    for index, waypoint in enumerate(waypoints, 1):
        if not isinstance(waypoint, dict):
            continue
        arms = [
            f"{arm}: {xyz(waypoint[f'{arm}_pose'])}" for arm in ("left", "right") if f"{arm}_pose" in waypoint
        ]
        parts.append(f"waypoint {index} ({frame}): " + ("; ".join(arms) or "hold arm poses"))
    return " | ".join(parts) or "no arm pose in waypoints"


def image_marker(location, manifest_item):
    """Place a claim only on the exact current RGB input cited by the model."""
    if not isinstance(location, dict) or not isinstance(manifest_item, dict):
        return None
    if manifest_item.get("media_type") != "image/jpeg":
        return None
    if manifest_item.get("label") != f"current camera={location.get('camera')}":
        return None
    if manifest_item.get("id") not in re.findall(r"input:\d+", str(location.get("source", ""))):
        return None
    coordinates = [location.get("u_norm"), location.get("v_norm")]
    for value in coordinates:
        if type(value) not in (int, float):
            return None
        try:
            if not math.isfinite(value) or not 0 <= value <= 1000:
                return None
        except OverflowError:
            return None
    return coordinates


def image_caption(manifest_item, fallback):
    if not isinstance(manifest_item, dict):
        return fallback
    label = manifest_item.get("label", "")
    if isinstance(label, str) and label.startswith("current camera="):
        camera = label.removeprefix("current camera=")
        name = {
            "head": "head camera",
            "hand_left": "left-hand camera",
            "hand_right": "right-hand camera",
        }.get(camera)
        if name:
            return f"Current {name} ({manifest_item.get('id', '?')})"
    return f"{manifest_item.get('id', '?')} {label}"


def matching_calibration(manifest, image_item):
    """Use only the calibration text immediately following this current RGB input."""
    if not isinstance(image_item, dict) or not image_item.get("label", "").startswith("current camera="):
        return None
    try:
        next_id = f"input:{int(image_item['id'].split(':')[1]) + 1}"
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    camera = image_item["label"].removeprefix("current camera=")
    matches = [
        item.get("calibration")
        for item in manifest
        if isinstance(item, dict)
        and item.get("id") == next_id
        and item.get("camera") == camera
        and item.get("image_role") == "current"
        and isinstance(item.get("calibration"), dict)
    ]
    return matches[0] if len(matches) == 1 else None


def llm_waypoint_marker(claim, manifest, image_item, decision):
    """Show the model's pixel claim only with its exact image, calibration and waypoint."""
    if not isinstance(claim, dict) or claim.get("method") == "unavailable":
        return None
    if claim.get("arm") not in ("left", "right") or model_point(decision, claim["arm"]) is None:
        return None
    image_source = claim.get("image_source")
    calibration_source = claim.get("calibration_source")
    if not isinstance(image_source, str) or not isinstance(calibration_source, str):
        return None
    coordinates = image_marker(
        {
            "camera": claim.get("camera"),
            "source": image_source,
            "u_norm": claim.get("u_norm"),
            "v_norm": claim.get("v_norm"),
        },
        image_item,
    )
    if coordinates is None or matching_calibration(manifest, image_item) is None:
        return None
    try:
        next_id = f"input:{int(image_item['id'].split(':')[1]) + 1}"
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if next_id not in re.findall(r"input:\d+", calibration_source) or claim.get("method") not in (
        "calibration_projection",
        "visual_estimate",
    ):
        return None
    return coordinates


def model_point(decision, arm):
    """Read the first requested waypoint in base_link, without using execution conversion."""
    motion = decision.get("motion") if isinstance(decision, dict) else None
    if not isinstance(motion, dict) or motion.get("frame") != "base_link":
        return None
    waypoints = motion.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints or not isinstance(waypoints[0], dict):
        return None
    return waypoints[0].get(f"{arm}_pose")


def projected_marker(coordinates, label, color):
    side = " label-left" if coordinates[0] > 700 else ""
    vertical = " label-below" if coordinates[1] < 130 else ""
    return (
        f'<span class="projected-marker {color}{side}{vertical}" '
        f'style="left:{coordinates[0] / 10:.1f}%;top:{coordinates[1] / 10:.1f}%" '
        f'aria-label="{html.escape(label)}" title="{html.escape(label)}">'
        f'<span class="marker-label">{html.escape(label)}</span></span>'
    )


def head_calibration_for_report(calibration, states, configured):
    """Reconstruct a missing head extrinsic from recorded joints and explicit config."""
    if (
        configured is None
        or not isinstance(calibration, dict)
        or not isinstance(states, dict)
        or "optical_to_base_link" in calibration
        or calibration.get("camera_to_mount_usd") != configured.entries["head"]["camera_to_mount"]
        or calibration.get("intrinsics") != configured.entries["head"]["intrinsics"]
    ):
        return calibration
    try:
        pose = configured.head_pose(states)
    except ValueError:
        return calibration
    return {
        **calibration,
        "optical_to_base_link": configured._optical_to_base(configured.entries["head"], pose),
        "optical_axes": "+x right, +y down, +z forward",
        "extrinsic_basis": "offline reconstruction from this request's measured waist/head joints and configured URDF",
        "metric_projection_ready": False,
    }


def render(log: Path, output: Path, calibration_config: Path | None = None) -> None:
    configured = None
    if calibration_config is not None:
        from agent2act.adapters.planners.static_rgb_calibration import StaticRgbCalibration

        config = json.loads(calibration_config.read_text(encoding="utf-8"))
        configured = StaticRgbCalibration(config["planner"]["rgb_camera_calibration"])
    groups = {}
    malformed = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if isinstance(event, dict) and event.get("inference_id"):
            groups.setdefault(event["inference_id"], []).append(event)

    def esc(value):
        return html.escape(str(value))

    def link(path):
        return quote(os.path.relpath(path, output.parent), safe="/")

    cards = []
    for request, events in groups.items():
        end = next(
            (
                e
                for e in reversed(events)
                if e.get("phase") == "inference_request" and e.get("message") == "phase_finished"
            ),
            {},
        )
        issued = next((e for e in reversed(events) if e.get("status") == "issued"), {})
        task = next((e["episode"][-1] for e in events if e.get("episode")), "Unknown task")
        outcome = end.get("outcome", "unfinished")
        errors = [e["error"] for e in events if e.get("status") == "rejected"]
        http = [e for e in events if e.get("message") == "llm_http_finished"]
        http_s = sum(e.get("duration_s", 0) for e in http)
        detail = []
        view_attempts = []
        active_view = None
        diagnostics = []
        position_rows = []
        latest_diagnostic = None
        latest_model_decision = None
        latest_states = None
        for e in events:
            if e.get("status") == "planned":
                position_rows.append(
                    {
                        "object": object_location(latest_diagnostic),
                        "target": (
                            motion_target(latest_model_decision)
                            if latest_model_decision is not None
                            else "not recorded in this log"
                        ),
                        "execution": motion_target(e.get("decision")),
                        "issued": "not issued",
                    }
                )
            elif e.get("status") == "issued" and position_rows:
                endpoint = e.get("issued_endpoint", {})
                frame = e.get("frame") or "frame unknown"
                if e.get("kind") == "EEF_ABS" and isinstance(endpoint, dict):
                    position_rows[-1]["issued"] = f"{frame}: " + "; ".join(
                        f"{arm}: {xyz(endpoint.get(f'{arm}_arm'))}" for arm in ("left", "right")
                    )
                else:
                    position_rows[-1]["issued"] = "issued action is not a Cartesian pose"
            if e.get("message") != "artifact_saved":
                continue
            path = Path(e["artifact_path"])
            detail.append(f'<li><a href="{link(path)}">{esc(path.name)}</a></li>')
            kind = e.get("artifact_kind")
            if kind == "simulator-request" and path.exists():
                request_data = json.loads(path.read_text())
                latest_states = request_data.get("params", {}).get("states")
            if kind == "planner-output-decision" and path.exists():
                latest_model_decision = json.loads(path.read_text()).get("decision")
                if active_view is not None:
                    active_view["decision"] = latest_model_decision
            if kind in ("diagnostic-input", "decision-diagnostics", "reasoning-summary") and path.exists():
                data = json.loads(path.read_text())
                if kind == "diagnostic-input":
                    active_view = {
                        "manifest": data.get("input_manifest", []),
                        "pictures": [],
                        "claim": None,
                        "camera_projections": [],
                        "decision": None,
                        "states": latest_states,
                    }
                    view_attempts.append(active_view)
                elif kind == "decision-diagnostics":
                    latest_diagnostic = data.get("report") if data.get("status") == "available" else None
                    if active_view is not None and isinstance(latest_diagnostic, dict):
                        active_view["claim"] = latest_diagnostic.get("object_location")
                        active_view["camera_projections"] = latest_diagnostic.get("camera_projections", [])
                title = {
                    "diagnostic-input": "Actual input manifest",
                    "decision-diagnostics": "Model-reported evidence (unverified)",
                    "reasoning-summary": "Provider reasoning summary",
                }[kind]
                diagnostics.append(
                    f"<details><summary>{title}</summary>"
                    f"<pre>{esc(json.dumps(data, ensure_ascii=False, indent=2))}</pre></details>"
                )
            if e.get("artifact_kind") == "llm-request" and path.exists():
                if active_view is None:
                    active_view = {
                        "manifest": [],
                        "pictures": [],
                        "claim": None,
                        "camera_projections": [],
                        "decision": None,
                        "states": latest_states,
                    }
                    view_attempts.append(active_view)
                data = json.loads(path.read_text())
                seen_images = {image for image, _ in active_view["pictures"]}
                for ref in image_references(data):
                    image = path.parent / ref["$artifact"]
                    if image in seen_images:
                        continue
                    seen_images.add(image)
                    active_view["pictures"].append((image, ref.get("sha256")))
        picture_html = []
        marked = False
        projected = False
        llm_projected = False
        llm_projection_requested = False
        for attempt_index, view in enumerate(view_attempts, 1):
            manifest = view["manifest"]
            image_inputs = {}
            for item in manifest:
                if isinstance(item, dict) and isinstance(item.get("sha256"), str):
                    image_inputs.setdefault(item["sha256"], []).append(item)
            for image, digest in view["pictures"]:
                matches = image_inputs.get(digest, [])
                manifest_item = matches[0] if len(matches) == 1 else None
                caption = image_caption(manifest_item, image.name)
                if len(view_attempts) > 1:
                    caption = f"Attempt {attempt_index} · {caption}"
                overlays = []
                claim = view["claim"]
                coordinates = image_marker(claim, manifest_item)
                if coordinates is not None:
                    marked = True
                    name = str(claim.get("label") or "unknown object")
                    title = f"LLM estimate: {name}; source: {claim.get('source') or 'unknown'}"
                    side = " label-left" if coordinates[0] > 700 else ""
                    vertical = " label-below" if coordinates[1] < 130 else ""
                    overlays.append(
                        f'<span class="model-marker{side}{vertical}" '
                        f'style="left:{coordinates[0] / 10:.1f}%;top:{coordinates[1] / 10:.1f}%" '
                        f'aria-label="{esc(title)}" title="{esc(title)}">'
                        f'<span class="marker-label">LLM estimate: {esc(name)}</span></span>'
                    )
                calibration = matching_calibration(manifest, manifest_item)
                projection_note = ""
                if calibration is not None and manifest_item is not None:
                    llm_projection_requested = True
                    claims = view.get("camera_projections") or []
                    seen_arms = set()
                    for target_claim in claims:
                        if not isinstance(target_claim, dict) or target_claim.get("arm") in seen_arms:
                            continue
                        if (
                            target_claim.get("method") == "unavailable"
                            and target_claim.get("camera")
                            == manifest_item.get("label", "").removeprefix("current camera=")
                            and manifest_item["id"]
                            in re.findall(r"input:\d+", str(target_claim.get("image_source", "")))
                        ):
                            arm_name = {"left": "left", "right": "right"}.get(
                                target_claim.get("arm"), "unknown"
                            )
                            projection_note += f"; LLM {arm_name} hand could not be projected: {target_claim.get('note') or 'no reason given'}"
                            seen_arms.add(target_claim["arm"])
                            continue
                        llm_pixel = llm_waypoint_marker(
                            target_claim, manifest, manifest_item, view["decision"]
                        )
                        if llm_pixel is None:
                            continue
                        seen_arms.add(target_claim["arm"])
                        llm_projected = True
                        arm_name = {"left": "left", "right": "right"}[target_claim["arm"]]
                        method_name = (
                            "self-computed calibrated projection"
                            if target_claim["method"] == "calibration_projection"
                            else "visual estimate"
                        )
                        label = f"LLM {method_name}: {arm_name} hand ({llm_pixel[0]:.0f}, {llm_pixel[1]:.0f})"
                        overlays.append(projected_marker(llm_pixel, label, "llm-target-marker"))
                        projection_note += (
                            f"; LLM {arm_name} hand {method_name} ({llm_pixel[0]:.0f}, {llm_pixel[1]:.0f})/1000"
                            f" [{target_claim['image_source']}, {target_claim['calibration_source']}]"
                        )
                        if target_claim.get("note"):
                            projection_note += f"：{target_claim['note']}"
                    if manifest_item.get("label") == "current camera=head":
                        calibration = head_calibration_for_report(calibration, view["states"], configured)
                        if str(calibration.get("extrinsic_basis", "")).startswith("offline reconstruction"):
                            projection_note += "; head extrinsics rebuilt offline from logged joint states; the LLM of an older run did not receive them"
                    if "optical_to_base_link" not in calibration and "optical_to_frame" not in calibration:
                        projection_note += "; this image lacks same-round base_link camera extrinsics, so the 3D projection cannot be shown"
                    elif calibration.get("available") is not True:
                        projection_note += (
                            "; camera calibration is unavailable for this image, so no 3D projection is shown"
                        )
                    else:
                        target_points = []
                        target_pixels = []
                        for arm, name in (("left", "left"), ("right", "right")):
                            point = model_point(view["decision"], arm)
                            if point is not None:
                                target_points.append(point)
                            pixel = project_base_point(calibration, point)
                            if pixel is not None:
                                target_pixels.append(pixel)
                                projected = True
                                overlays.append(
                                    projected_marker(
                                        pixel, f"Code check: {name} hand next target", "target-marker"
                                    )
                                )
                                model_pixel = next(
                                    (
                                        candidate
                                        for item in (view.get("camera_projections") or [])
                                        if isinstance(item, dict) and item.get("arm") == arm
                                        if (
                                            candidate := llm_waypoint_marker(
                                                item, manifest, manifest_item, view["decision"]
                                            )
                                        )
                                        is not None
                                    ),
                                    None,
                                )
                                if model_pixel is not None:
                                    delta_u, delta_v = pixel[0] - model_pixel[0], pixel[1] - model_pixel[1]
                                    projection_note += f"; {name} hand LLM/code landing offset Δu={delta_u:+.0f}, Δv={delta_v:+.0f} (0..1000)"
                        if target_points and not target_pixels:
                            projection_note += "; the model's next target is not projected onto this image (possibly outside the frame or behind the camera)"
                        if isinstance(claim, dict) and claim.get("frame") == "base_link":
                            point = [claim.get(axis) for axis in ("x_m", "y_m", "z_m")]
                            pixel = project_base_point(calibration, point)
                            if pixel is not None:
                                projected = True
                                overlays.append(
                                    projected_marker(
                                        pixel,
                                        "Back-projection of the model's 3D object point",
                                        "object3d-marker",
                                    )
                                )
                picture_html.append(
                    f'<figure><div class="image-frame"><a href="{link(image)}">'
                    f'<img loading="lazy" src="{link(image)}" alt="{esc(caption)}"></a>'
                    f"{''.join(overlays)}</div><figcaption>{esc(caption + projection_note)}</figcaption></figure>"
                )
        marker_note = (
            '<p class="marker-note">The red hollow circle is the object image position reported by the LLM, labeled with the object name it believes this is. '
            "It is not an arm target and does not mean the object has been grasped.</p>"
            if marked
            else ""
        )
        if projected:
            marker_note += (
                '<p class="projection-note">The yellow square is the model’s next waypoint projected back onto the image by code, using the current camera calibration after the model replied. '
                "The blue diamond is the back-projection of the 3D object point the model reported. Neither is a pixel point given directly by the LLM, "
                "and neither proves the model used the calibration, the target is reachable, or the object is grasped. The static calibration has not been verified against the current rendering, "
                "so projections are for troubleshooting only.</p>"
            )
        if llm_projected:
            marker_note += (
                '<p class="llm-projection-note">The purple hollow square is the position the LLM itself reports for its next waypoint in each camera image; '
                "the label distinguishes whether it claims a calibrated computation or a visual estimate. The offset from the yellow square can be used to check the model's computation, "
                "but the model's own statement does not prove it actually used the calibration.</p>"
            )
        elif llm_projection_requested:
            marker_note += '<p class="llm-projection-note">No LLM-reported target projection is available; rerun older logs, or check the null values and reasons in the model diagnostics.</p>'
        tracking = [e for e in events if e.get("status") == "observed_tracking"]
        feedback = ""
        for e in tracking:
            for arm in ("left", "right"):
                error = e.get(arm, {}).get("translation_error_m")
                if error is not None:
                    feedback += f"{arm}: {error * 1000:.1f} mm endpoint error. "
        rows = "".join(
            f"<tr><td>{esc(e.get('attempt'))}</td><td>{esc(e.get('http_status'))}</td>"
            f"<td>{e.get('request_bytes', 0) / 1024:.1f} KiB</td><td>{e.get('duration_s', 0):.2f}s</td>"
            f"<td>{esc(e.get('input_tokens', '?'))}</td><td>{esc(e.get('output_tokens', '?'))}</td></tr>"
            for e in http
        )
        positions = "".join(
            "<tr>"
            + "".join(f"<td>{esc(row[key])}</td>" for key in ("object", "target", "execution", "issued"))
            + "</tr>"
            for row in position_rows
        )
        position_table = (
            "<h3>Object estimates and arm targets</h3>"
            "<p>The first column is the object position reported by the model; the second is the waypoint the model asks the gripper to reach. "
            "The third is the execution waypoint after adapter conversion; the fourth is the endpoint of the action actually issued. "
            "Compare values only when the frame and control point match. An issued action does not mean it was reached or grasped.</p>"
            "<table><tr><th>Model-estimated object position</th><th>Model target waypoint XYZ</th>"
            "<th>Execution target waypoint XYZ</th>"
            f"<th>Issued endpoint XYZ</th></tr>{positions}</table>"
            if position_rows
            else ""
        )
        cards.append(f"""<article><h2>{esc(task)}</h2>
<p>{esc(events[0].get("timestamp"))} · {esc(request[:8])} · <b>{esc(outcome)}</b></p>
<p>Total: {esc(end.get("duration_s", "?"))}s · HTTP: {http_s:.2f}s / {len(http)} calls ·
Issued: {issued.get("issued_steps", 0)} steps · Discarded: {issued.get("discarded_steps", "?")} steps</p>
<p>{esc(issued.get("instruction", "No action issued"))}</p><p>{esc(feedback)}</p>
<p class="error">{esc("; ".join(errors))}</p>{marker_note}<div class="images">{"".join(picture_html)}</div>
{position_table}
{"".join(diagnostics)}
<details><summary>HTTP calls</summary><table><tr><th>Attempt</th><th>Status</th><th>Request</th><th>Time</th><th>Input tokens</th><th>Output tokens</th></tr>{rows}</table></details>
<details><summary>Inputs, raw answers, actions and images ({len(detail)} artifacts)</summary><ul>{"".join(detail)}</ul></details>
<details><summary>Full events</summary><pre>{esc(json.dumps(events, ensure_ascii=False, indent=2))}</pre></details></article>""")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Agent2Act run report</title>
<style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;background:#f3f5f8;color:#17283b}article{background:white;padding:24px;margin:20px 0;border-radius:12px}h2{font-size:20px}.error{color:#a02424}pre{white-space:pre-wrap;overflow-wrap:anywhere}.images{display:flex;flex-wrap:wrap}.image-frame{position:relative;width:400px;max-width:100%}.image-frame img{display:block;width:100%}.model-marker,.projected-marker{position:absolute;transform:translate(-50%,-50%);pointer-events:none}.model-marker{width:26px;height:26px;border-radius:50%;border:3px solid #d62828;background:transparent;box-shadow:0 0 0 2px white,inset 0 0 0 1px white}.projected-marker{width:24px;height:24px;border:3px solid #b77900;background:#ffd34d77;box-shadow:0 0 0 2px white}.llm-target-marker{border:3px solid #7434bb;background:transparent;box-shadow:0 0 0 2px white}.object3d-marker{border:0;background:none;box-shadow:none;color:#0867bd;font-size:32px;line-height:24px;text-shadow:0 0 3px white}.object3d-marker::before{content:"◆"}.marker-label{position:absolute;left:31px;bottom:24px;white-space:nowrap;padding:3px 7px;background:#fff;color:#a01010;border:2px solid #d62828;border-radius:5px;font:bold 13px system-ui;box-shadow:0 1px 5px #3338}.projected-marker .marker-label{color:#765000;border-color:#b77900}.object3d-marker .marker-label{color:#08599e;border-color:#0867bd}.llm-target-marker .marker-label{color:#5a278e;border-color:#7434bb}.llm-projection-note{color:#5a278e;font-weight:600}.label-left .marker-label{left:auto;right:31px}.label-below .marker-label{bottom:auto;top:24px}.marker-note{color:#9d2020;font-weight:600}.projection-note{color:#654900;font-weight:600}td,th{padding:8px;text-align:left}summary{cursor:pointer;padding:8px}a{color:#1261a0}</style>
</head><body><h1>Agent2Act run report</h1><p>Issued actions are not observed task success. HTTP is included in planning/total time. Timestamps retain their recorded timezone.</p>"""
        + f"<p>{len(groups)} requests; {malformed} malformed lines skipped.</p>"
        + "".join(cards)
        + "</body></html>",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="Matching planner JSON; reconstruct head camera extrinsics in older RGB-only traces",
    )
    args = parser.parse_args()
    output = args.output or args.log.with_suffix(".html")
    render(
        args.log.resolve(),
        output.resolve(),
        args.config.resolve() if args.config is not None else None,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
