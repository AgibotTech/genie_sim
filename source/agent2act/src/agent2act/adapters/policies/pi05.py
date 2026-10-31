# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Configurable OpenPI/π0.5 checkpoint adapter.

The OpenPI server owns checkpoint (un)normalization; this adapter handles camera/state selection
and physical action mapping."""

import logging
import math
from dataclasses import replace
from io import BytesIO

from agent2act.adapters.telemetry import record_artifact
from agent2act.adapters.transports.openpi import OpenPiTransport
from agent2act.domain.actions import gripper_start
from agent2act.domain.models import FRAMES, ActionChunk, ContractError, Observation, number, vector
from agent2act.domain.validation import ActionValidator


class Pi05Adapter:
    def __init__(self, spec, config, client_factory=None):
        self.spec, self.config, self.client = spec, dict(config), None
        self.client_factory = client_factory or self._openpi_client
        required = {
            "robot_type",
            "checkpoint",
            "skills",
            "host",
            "port",
            "camera_map",
            "image_size",
            "state_fields",
            "state_key",
            "prompt_key",
            "action_key",
            "action_dim",
            "left_indices",
            "right_indices",
            "gripper_indices",
            "kind",
            "frame",
            "dt_s",
            "action_scale",
            "action_offset",
            "state_scale",
            "state_offset",
        }
        optional = {
            "joint_delta",
            "timeout_s",
            "gripper_clip_tolerance",
            "retime_joint_actions",
            "rate_limit_joint_actions",
            "expected_training_config",
            "capabilities",
            "body_indices",
        }
        if set(config) - required - optional or required - set(config):
            raise ContractError(f"pi05 requires fields {sorted(required)}")
        self.expected_training_config = config.get("expected_training_config")
        if self.expected_training_config is not None and (
            not isinstance(self.expected_training_config, str) or not self.expected_training_config.strip()
        ):
            raise ContractError("expected_training_config must be a nonempty string")
        self.retime_joint_actions = config.get("retime_joint_actions", False)
        if type(self.retime_joint_actions) is not bool:
            raise ContractError("retime_joint_actions must be boolean")
        if self.retime_joint_actions and config["kind"] != "JOINT_ABS":
            raise ContractError("retime_joint_actions requires JOINT_ABS")
        self.rate_limit_joint_actions = config.get("rate_limit_joint_actions", False)
        if type(self.rate_limit_joint_actions) is not bool:
            raise ContractError("rate_limit_joint_actions must be boolean")
        if self.rate_limit_joint_actions and config["kind"] != "JOINT_ABS":
            raise ContractError("rate_limit_joint_actions requires JOINT_ABS")
        if self.rate_limit_joint_actions and self.retime_joint_actions:
            raise ContractError("choose either retime_joint_actions or rate_limit_joint_actions")
        self.capabilities = self._parse_capabilities(config.get("capabilities", []))
        self.gripper_clip_tolerance = number(
            config.get("gripper_clip_tolerance", 0), "pi05 gripper_clip_tolerance"
        )
        if self.gripper_clip_tolerance < 0:
            raise ContractError("pi05 gripper_clip_tolerance must be nonnegative")
        if number(config.get("timeout_s", 30), "pi05 timeout_s") <= 0:
            raise ContractError("pi05 timeout_s must be positive")
        if config["robot_type"] != spec.robot_type:
            raise ContractError("pi05 checkpoint robot_type mismatch")
        if config["kind"] not in ("JOINT_ABS", "EEF_ABS") or config["frame"] not in FRAMES:
            raise ContractError("unsupported pi05 action kind/frame")
        if config.get("joint_delta", False) and config["kind"] != "JOINT_ABS":
            raise ContractError("only observation-relative joint delta conversion is supported")
        if type(config.get("joint_delta", False)) is not bool:
            raise ContractError("joint_delta must be boolean")
        if number(config["dt_s"], "pi05 dt_s") != spec.dt_s:
            raise ContractError("pi05 action period must equal simulator action-step period")
        dim = config["action_dim"]
        if type(dim) is not int or dim < 1:
            raise ContractError("action_dim must be a positive integer")
        widths = (len(spec.left_joints), len(spec.right_joints)) if config["kind"] == "JOINT_ABS" else (7, 7)
        indices = []
        for key, width in zip(("left_indices", "right_indices", "gripper_indices"), (*widths, 2)):
            values = config[key]
            if not isinstance(values, list) or len(values) != width:
                raise ContractError(f"{key} has wrong width")
            if any(type(v) is not int or not 0 <= v < dim for v in values):
                raise ContractError(f"{key} contains invalid indices")
            indices.extend(values)
        if len(set(indices)) != len(indices):
            raise ContractError("pi05 output mappings overlap")
        body_indices = config.get("body_indices", [])
        if not isinstance(body_indices, list) or len(body_indices) != len(spec.body_joints):
            raise ContractError("body_indices must match configured body joints")
        if any(i is not None and (type(i) is not int or not 0 <= i < dim) for i in body_indices):
            raise ContractError("invalid body_indices")
        mapped_body = [i for i in body_indices if i is not None]
        if len(set(indices + mapped_body)) != len(indices + mapped_body):
            raise ContractError("pi05 body output mappings overlap")
        if body_indices and config["kind"] != "JOINT_ABS":
            raise ContractError("pi05 body mapping requires JOINT_ABS")
        vector(config["action_scale"], dim, "action_scale")
        vector(config["action_offset"], dim, "action_offset")
        if not isinstance(config["state_fields"], list) or not config["state_fields"]:
            raise ContractError("state_fields must be nonempty")
        if not isinstance(config["camera_map"], dict) or not config["camera_map"]:
            raise ContractError("camera_map must map model keys to CoRobot camera names")
        size = config["image_size"]
        if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
            raise ContractError("image_size must be [width,height] positive integers")
        keys = list(config["camera_map"]) + [config["state_key"], config["prompt_key"]]
        if any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != len(keys):
            raise ContractError("model input keys must be distinct nonempty strings")

    def _openpi_client(self):
        try:
            return OpenPiTransport(self.config["host"], self.config["port"], self.config.get("timeout_s", 30))
        except ImportError as exc:
            raise ContractError("install the locked pi05 extra with: uv sync --locked --extra pi05") from exc

    @staticmethod
    def _parse_capabilities(value):
        if not isinstance(value, list):
            raise ContractError("pi05 capabilities must be a list")
        result = []
        names = set()
        required = {"name", "description", "instruction_scope", "max_policy_chunks"}
        for capability in value:
            if not isinstance(capability, dict) or set(capability) != required:
                raise ContractError(f"pi05 capability requires exactly {sorted(required)}")
            name = capability["name"]
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ContractError("pi05 capability names must be unique nonempty text")
            if not isinstance(capability["description"], str) or not capability["description"].strip():
                raise ContractError("pi05 capability description must be nonempty text")
            if capability["instruction_scope"] not in ("local_subtask", "original_task"):
                raise ContractError("pi05 capability instruction_scope is invalid")
            limit = capability["max_policy_chunks"]
            if type(limit) is not int or not 1 <= limit <= 32:
                raise ContractError("pi05 capability max_policy_chunks must be an integer in [1,32]")
            result.append(dict(capability))
            names.add(name)
        return result

    def describe(self):
        return {
            "backend": "pi05",
            "robot_type": self.spec.robot_type,
            "checkpoint": self.config["checkpoint"],
            "skills": self.config["skills"],
            "kind": self.config["kind"],
            "frame": self.config["frame"],
            "gripper_clip_tolerance": self.gripper_clip_tolerance,
            "retime_joint_actions": self.retime_joint_actions,
            "rate_limit_joint_actions": self.rate_limit_joint_actions,
            "capabilities": self.capabilities,
        }

    def reset(self):
        if self.client is not None and hasattr(self.client, "reset"):
            self.client.reset()

    def close(self):
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()
        self.client = None

    def encode_observation(self, observation, instruction):
        import numpy as np
        from PIL import Image, ImageOps

        c = self.config
        state: list[float] = []
        for key in c["state_fields"]:
            values = observation.state_groups.get(key)
            if not isinstance(values, list):
                raise ContractError(f"pi05 missing flat state field: {key}")
            state.extend(number(v, key) for v in values)
        scale = vector(c["state_scale"], len(state), "state_scale")
        offset = vector(c["state_offset"], len(state), "state_offset")
        result = {
            c["state_key"]: np.asarray(
                [v * a + b for v, a, b in zip(state, scale, offset)], dtype=np.float32
            ),
            c["prompt_key"]: instruction,
        }
        for target, source in c["camera_map"].items():
            image = observation.images.get(source)
            if not isinstance(image, dict) or image.get("encoding") != "JPEG":
                raise ContractError(f"pi05 missing JPEG camera {source}")
            with Image.open(BytesIO(image["image_data"])) as decoded:
                # RGB uint8, preserve aspect ratio with centred black padding.
                rgb = ImageOps.pad(
                    decoded.convert("RGB"),
                    tuple(c["image_size"]),
                    method=Image.Resampling.BILINEAR,
                    color=(0, 0, 0),
                )
                result[target] = np.asarray(rgb, dtype=np.uint8)
        return result

    def decode_actions(self, response, observation):
        c = self.config
        if not isinstance(response, dict) or c["action_key"] not in response:
            raise ContractError("pi05 response missing configured action_key")
        values = response[c["action_key"]]
        values = values.tolist() if hasattr(values, "tolist") else values
        # Capture before mapping/validation so rejected predictions remain inspectable.
        record_artifact("policy-actions", {"backend": "pi05", "actions": values})
        if not isinstance(values, list) or not 1 <= len(values) <= self.spec.max_chunk_steps:
            raise ContractError("pi05 returned empty or oversized action horizon")
        chunk = ActionChunk(c["kind"], c["frame"], c["dt_s"], [], [], [])
        corrections = []
        n = len(self.spec.left_joints)
        for step, raw in enumerate(values):
            row = vector(raw, c["action_dim"], "pi05 action")
            row = [v * a + b for v, a, b in zip(row, c["action_scale"], c["action_offset"])]
            left, right = [row[i] for i in c["left_indices"]], [row[i] for i in c["right_indices"]]
            if c.get("joint_delta", False):
                # Each row is relative to the same observation, not cumulatively integrated.
                left = [a + b for a, b in zip(left, observation.joints[:n])]
                right = [a + b for a, b in zip(right, observation.joints[n:])]
            chunk.left_arm.append(left)
            chunk.right_arm.append(right)
            grippers = [row[i] for i in c["gripper_indices"]]
            if self.gripper_clip_tolerance > 0:
                for side, (lower, upper) in enumerate(self.spec.gripper_bounds):
                    original = number(grippers[side], "pi05 mapped gripper")
                    corrected = min(upper, max(lower, original))
                    correction = corrected - original
                    if abs(correction) > self.gripper_clip_tolerance:
                        raise ContractError(
                            f"pi05 gripper bounds exceeded beyond clip tolerance at step {step}: "
                            f"side={side}, value={original}, bounds={[lower, upper]}, "
                            f"tolerance={self.gripper_clip_tolerance}"
                        )
                    if correction:
                        grippers[side] = corrected
                        corrections.append(
                            {
                                "step": step,
                                "side": "left" if side == 0 else "right",
                                "original": original,
                                "corrected": corrected,
                                "delta": correction,
                            }
                        )
            chunk.grippers.append(grippers)
            if self.spec.body_joints:
                # Null explicitly holds an unsupervised joint at its measured value.
                # This is used for checkpoint action-mask channels, not arbitrary clipping.
                body = [
                    observed if i is None else row[i] + (observed if c.get("joint_delta", False) else 0)
                    for i, observed in zip(c["body_indices"], observation.body)
                ]
                chunk.body.append(body)
        if corrections:
            chunk.metadata["gripper_clips"] = corrections
            logging.getLogger(__name__).info(
                "pi05_gripper_clipped",
                extra={"event": {"tolerance": self.gripper_clip_tolerance, "corrections": corrections}},
            )
        if self.rate_limit_joint_actions:
            return self._rate_limit(chunk, observation)
        return self._retime(chunk, observation) if self.retime_joint_actions else chunk

    def _validate_joint_targets(self, chunk: ActionChunk, observation: Observation) -> None:
        # Conditioning may change speeds, never legalize an invalid raw target.
        limits = replace(
            self.spec,
            max_joint_speed=1e300,
            max_gripper_speed=1e300,
            body_speed_limits=[1e300] * len(self.spec.body_joints),
        )
        ActionValidator(limits).validate(chunk, observation)

    def _rate_limit(self, chunk: ActionChunk, observation: Observation) -> ActionChunk:
        """Bound each channel's increments without extending the sampled horizon.

        Unlike waypoint retiming this changes the reference path; unreached targets are not replayed."""
        self._validate_joint_targets(chunk, observation)
        grippers, correction = gripper_start(observation, self.spec)
        previous = observation.joints + grippers + (observation.body if chunk.body else [])
        deltas = [
            rate * chunk.dt_s
            for rate in (
                [self.spec.max_joint_speed] * len(observation.joints)
                + [self.spec.max_gripper_speed] * 2
                + (self.spec.body_speed_limits if chunk.body else [])
            )
        ]
        result = ActionChunk(chunk.kind, chunk.frame, chunk.dt_s, [], [], [], dict(chunk.metadata))
        result.metadata.update(correction)
        n, m = len(self.spec.left_joints), len(observation.joints)
        max_errors = [0.0] * len(previous)
        limited_steps = 0
        endpoint_error = [0.0] * len(previous)
        for i, (left, right, grips) in enumerate(zip(chunk.left_arm, chunk.right_arm, chunk.grippers)):
            target = left + right + grips + (chunk.body[i] if chunk.body else [])
            row = [min(a + limit, max(a - limit, b)) for a, b, limit in zip(previous, target, deltas)]
            endpoint_error = [b - actual for b, actual in zip(target, row)]
            limited_steps += any(error != 0 for error in endpoint_error)
            max_errors = [max(old, abs(error)) for old, error in zip(max_errors, endpoint_error)]
            result.left_arm.append(row[:n])
            result.right_arm.append(row[n:m])
            result.grippers.append(row[m : m + 2])
            if chunk.body:
                result.body.append(row[m + 2 :])
            previous = row
        result.metadata["conditioning"] = {
            "method": "per_channel_rate_limit",
            "original_steps": chunk.steps,
            "conditioned_steps": result.steps,
            "limited_steps": limited_steps,
            "channels": self.spec.left_joints
            + self.spec.right_joints
            + ["left_gripper", "right_gripper"]
            + (self.spec.body_joints if chunk.body else []),
            "max_target_error": max_errors,
            "endpoint_target_minus_command": endpoint_error,
            "meaning": "Rate-limited command path, not exact raw policy waypoints. Errors are reference changes, not observed tracking or task success. FK previews this conditioned path.",
        }
        ActionValidator(self.spec).validate(result, observation)
        record_artifact("policy-conditioning", result.metadata["conditioning"])
        return result

    def _retime(self, chunk: ActionChunk, observation: Observation) -> ActionChunk:
        self._validate_joint_targets(chunk, observation)
        result = ActionChunk(chunk.kind, chunk.frame, chunk.dt_s, [], [], [], dict(chunk.metadata))
        grippers, correction = gripper_start(observation, self.spec)
        result.metadata.update(correction)
        previous = observation.joints + grippers + observation.body
        rates = (
            [self.spec.max_joint_speed] * len(observation.joints)
            + [self.spec.max_gripper_speed] * 2
            + self.spec.body_speed_limits
        )
        n = len(self.spec.left_joints)
        m = len(observation.joints)
        for index, (left, right, grippers) in enumerate(zip(chunk.left_arm, chunk.right_arm, chunk.grippers)):
            target = left + right + grippers + (chunk.body[index] if chunk.body else [])
            steps = max(
                1, math.ceil(max(abs(b - a) / (r * chunk.dt_s) for a, b, r in zip(previous, target, rates)))
            )
            if result.steps + steps > self.spec.max_chunk_steps:
                raise ContractError("pi05 retimed trajectory exceeds max_chunk_steps")
            for index in range(1, steps + 1):
                row = (
                    target
                    if index == steps
                    else [a + (b - a) * index / steps for a, b in zip(previous, target)]
                )
                result.left_arm.append(row[:n])
                result.right_arm.append(row[n:m])
                result.grippers.append(row[m : m + 2])
                if chunk.body:
                    result.body.append(row[m + 2 :])
            previous = target
        result.metadata["retiming"] = {
            "original_steps": chunk.steps,
            "retimed_steps": result.steps,
            "inserted_steps": result.steps - chunk.steps,
        }
        ActionValidator(self.spec).validate(result, observation)
        record_artifact("policy-retiming", result.metadata["retiming"])
        return result

    def generate(self, observation, instruction):
        model_input = self.encode_observation(observation, instruction)
        if self.client is None:
            self.client = self.client_factory()
        try:
            if self.expected_training_config is not None:
                metadata = getattr(self.client, "metadata", None)
                actual = metadata.get("training_config") if isinstance(metadata, dict) else None
                if actual != self.expected_training_config:
                    raise ContractError(
                        f"pi05 training config mismatch: expected={self.expected_training_config}, "
                        f"reported={actual}; start the matching model server"
                    )
            return self.decode_actions(self.client.infer(model_input), observation)
        except Exception:
            # Never consume a late reply to a timed-out request as the next chunk.
            self.close()
            raise
