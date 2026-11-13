# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Serialized episode boundaries on the physics owner; YAML owns every plan."""
from __future__ import annotations

import copy
import json
import math
import threading


def sequence_entries(scene):
    entries = scene.get("libuipc", {}).get("episode_sequence")
    if entries is None:
        return None
    if not isinstance(entries, list) or not entries:
        raise ValueError("libuipc.episode_sequence must be a nonempty list")
    paths = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) not in (
            {"state_recording", "material"},
            {"state_recording", "material", "robot_base_pose"},
        ):
            raise ValueError(
                "episode entries require state_recording/material and an optional preloaded robot_base_pose"
            )
        if "robot_base_pose" in entry:
            pose = entry["robot_base_pose"]
            initial = scene.get("robot", {}).get("init_base_pose")
            if not isinstance(pose, dict) or not isinstance(initial, dict) or set(pose) != set(initial):
                raise ValueError("episode robot_base_pose must match the initial pose fields")
            if not {"x", "y"} <= set(pose) or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in pose.values()
            ):
                raise ValueError("episode robot_base_pose requires finite numeric x/y and pose fields")
            if any(pose[k] != initial[k] for k in pose if k not in ("x", "y")):
                raise ValueError("only preloaded base x/y changes are supported")
        from .physical_state_recording import PhysicalStateRecordingConfig

        cfg = PhysicalStateRecordingConfig.from_scene({"libuipc": entry})
        if cfg is None:
            raise ValueError("every sequence episode requires a fresh physical archive")
        material = entry["material"]
        if not isinstance(material, dict) or set(material) != {"episode_plan", "receipt_path"}:
            raise ValueError("episode material requires episode_plan and receipt_path")
        paths.extend((cfg.output_dir, material["receipt_path"]))
    if len(set(paths)) != len(paths):
        raise ValueError("episode archive and receipt paths must be unique")
    first = entries[0]
    if any(("robot_base_pose" in e) != ("robot_base_pose" in first) for e in entries):
        raise ValueError("base pose plans must be provided for every sequence entry")
    if "robot_base_pose" in first and first["robot_base_pose"] != scene["robot"]["init_base_pose"]:
        raise ValueError("initial robot base pose must match episode_sequence[0]")
    if (
        scene["libuipc"].get("state_recording") != first["state_recording"]
        or scene.get("generalization", {}).get("material") != first["material"]
    ):
        raise ValueError("initial recording/material must match episode_sequence[0]")
    return copy.deepcopy(entries)


class InvalidEpisodeOperation(ValueError):
    """A client lifecycle request rejected before any native state mutation."""


class EpisodeRequests:
    """Callbacks enqueue plain data; drain executes once between native steps.

    Exact duplicate IDs return the original ACK, including after epoch change.
    Unknown old identities and conflicting retries never perform another reset.
    """

    def __init__(self, engine):
        self.engine = engine
        self._lock = threading.RLock()
        self._payloads, self._responses, self._pending = {}, {}, []

    def request(self, packet):
        with self._lock:
            reply = (
                {key: packet.get(key) for key in ("request_id", "source_id", "epoch")}
                if isinstance(packet, dict)
                else {}
            )
            try:
                if not isinstance(packet, dict) or set(packet) != {
                    "version",
                    "request_id",
                    "source_id",
                    "epoch",
                    "operation",
                    "index",
                }:
                    raise ValueError("invalid episode request structure")
                if (
                    type(packet["version"]) is not int
                    or packet["version"] != 1
                    or not isinstance(packet["request_id"], str)
                    or not packet["request_id"]
                ):
                    raise ValueError("invalid episode request identity")
                if (
                    type(packet["epoch"]) is not int
                    or type(packet["index"]) is not int
                    or packet["index"] < 0
                    or packet["operation"] not in ("begin", "finish", "reset")
                ):
                    raise ValueError("invalid episode operation/index")
                key = (packet["source_id"], packet["epoch"], packet["request_id"])
                canonical = json.dumps(packet, sort_keys=True, allow_nan=False)
                if key in self._payloads:
                    if self._payloads[key] != canonical:
                        raise ValueError("episode request_id payload conflict")
                    return copy.deepcopy(self._responses.get(key))
                if packet["source_id"] != self.engine._source_id or packet["epoch"] != self.engine._epoch:
                    raise ValueError("stale episode source/epoch")
                self._payloads[key] = canonical
                self._pending.append((key, copy.deepcopy(packet)))
                return None
            except (ValueError, TypeError) as exc:
                return dict(reply, status="rejected", error=str(exc))

    def drain(self, publish=None):
        replies = []
        with self._lock:
            while self._pending:
                key, packet = self._pending.pop(0)
                reply = {field: packet[field] for field in ("request_id", "source_id", "epoch", "operation", "index")}
                try:
                    if packet["epoch"] != self.engine._epoch:
                        reply.update(status="rejected", error="stale queued epoch")
                    else:
                        identity = self.engine.sequence_operation(packet["operation"], packet["index"])
                        reply.update(status="completed", identity=identity)
                except InvalidEpisodeOperation as exc:
                    reply.update(status="rejected", error=str(exc))
                except BaseException as exc:
                    reply.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    self._responses[key] = reply
                    if publish is not None:
                        publish(reply)
                    raise
                self._responses[key] = reply
                replies.append(copy.deepcopy(reply))
                if publish is not None:
                    publish(reply)
        return replies


class EpisodeSource:
    def __init__(self, engine, node, topic):
        from std_msgs.msg import String

        self.engine, self.node, self.String = engine, node, String
        self.requests = EpisodeRequests(engine)
        self.publisher = node.create_publisher(String, topic + "/episode/done", 10)
        self.subscription = node.create_subscription(String, topic + "/episode/request", self.receive, 10)

    def publish(self, reply):
        try:
            self.publisher.publish(self.String(data=json.dumps(reply, allow_nan=False)))
        except BaseException as exc:
            self.engine.report_transport_failure(exc)
            raise

    def receive(self, message):
        try:
            packet = json.loads(message.data)
        except (ValueError, TypeError):
            packet = None
        reply = self.requests.request(packet)
        if reply is not None:
            self.publish(reply)

    def drain(self):
        self.requests.drain(self.publish)

    def close(self):
        try:
            self.node.destroy_subscription(self.subscription)
        finally:
            self.node.destroy_publisher(self.publisher)
