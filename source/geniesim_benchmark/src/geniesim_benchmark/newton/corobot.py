# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CoRobot wire protocol for a caller-owned Newton evaluation loop.

Wuji observations and actions use the legacy joint order and native radians.
Fixed scene joints appear in observations but accept only their initial hold
position. No legacy policy, simulator, IK solver or task runtime is imported.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import math
import time
from typing import Any

import cv2
import msgpack
import numpy as np
from websockets.sync.client import connect


@dataclass(frozen=True)
class JointGroups:
    """Wire order for the G2 Wuji embodiment, independent of engine ordering."""

    left_arm: tuple[str, ...] = tuple(f"idx2{j}_arm_l_joint{j}" for j in range(1, 8))
    right_arm: tuple[str, ...] = tuple(f"idx6{j}_arm_r_joint{j}" for j in range(1, 8))
    left_effector: tuple[str, ...] = tuple(
        f"hand_l_{finger}_joint{j}" for finger in ("thumb", "index", "middle", "ring", "pinky") for j in range(1, 5)
    )
    right_effector: tuple[str, ...] = tuple(
        f"hand_r_{finger}_joint{j}" for finger in ("thumb", "index", "middle", "ring", "pinky") for j in range(1, 5)
    )
    waist: tuple[str, ...] = tuple(f"idx0{j}_body_joint{j}" for j in range(1, 6))
    head: tuple[str, ...] = tuple(f"idx1{j}_head_joint{j}" for j in range(1, 4))

    def __post_init__(self) -> None:
        for key, dimension in (("left_arm", 7), ("right_arm", 7), ("left_effector", 20), ("right_effector", 20)):
            if len(getattr(self, key)) != dimension:
                raise ValueError(f"{key} requires {dimension} joint names")
        names = self.all_joints
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("joint groups require unique nonempty names")

    @property
    def all_joints(self) -> tuple[str, ...]:
        return self.left_arm + self.right_arm + self.left_effector + self.right_effector + self.waist + self.head


WUJI_GROUPS = JointGroups()


@dataclass(frozen=True)
class ActionChunk:
    """Validated full targets; a server stop is separate from task success."""

    targets: tuple[dict[str, float], ...]
    server_episode_done: bool = False
    history_interval: int = 0


def _finite_positions(positions: Mapping[str, float], label: str) -> dict[str, float]:
    result = {}
    for name, value in positions.items():
        if isinstance(value, (str, bytes, bool)):
            raise ValueError(f"{label}[{name!r}] must be a finite number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{label}[{name!r}] must be a finite number")
        result[name] = converted
    return result


def encode_image(image_rgb: np.ndarray, *, quality: int = 95) -> dict[str, Any]:
    """Encode exactly the legacy RGB-to-JPEG camera dictionary."""
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("camera image must be a nonempty HWC RGB uint8 array")
    success, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("JPEG encoding failed")
    return {"encoding": "JPEG", "image_data": buffer.tobytes(), "height": image.shape[0], "width": image.shape[1]}


def _history_interval(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("hist_frame_interval must be a nonnegative integer")
    return value


def build_payload(
    joint_positions: Mapping[str, float],
    images: Mapping[str, np.ndarray],
    prompt: str,
    *,
    groups: JointGroups = WUJI_GROUPS,
    task_name: str = "",
    episode_idx: int = 0,
    episode_done: bool = False,
    task_progress: Sequence[Mapping[str, Any]] = (),
    end_pose: Mapping[str, Any] | None = None,
    history_interval: int = 0,
    history_frames: Sequence[Mapping[str, Any]] = (),
    robot_type: str = "g2a_sim",
    timestamp_ns: int | None = None,
) -> dict[str, Any]:
    """Build the existing CoRobot infer envelope without adding wire fields.

    ``joint_positions`` includes scene-initial holds for fixed/absent joints.
    ``task_progress`` is accepted for caller compatibility but stays local.
    History frames are the already JPEG-encoded ``{"head": image_dict}``
    entries captured by the caller during the preceding action chunk.
    """
    positions = _finite_positions(joint_positions, "joint_positions")

    def values(names: tuple[str, ...]) -> list[float]:
        missing = [name for name in names if name not in positions]
        if missing:
            raise ValueError(f"missing observation joints: {missing}")
        return [positions[name] for name in names]

    cameras = {}
    for wire_name, observation_name in (("head", "head"), ("hand_left", "left_hand"), ("hand_right", "right_hand")):
        camera = images.get(observation_name)
        if camera is None:
            camera = images.get(wire_name)
        if camera is None:
            raise ValueError(f"missing camera image: {wire_name}")
        cameras[wire_name] = encode_image(camera)
    stamp = time.time_ns() if timestamp_ns is None else timestamp_ns
    params = {
        "timestamps": {"head": stamp, "states": stamp},
        "images": cameras,
        "states": {
            "head_joint_states": values(groups.head),
            "arm_joint_states": values(groups.left_arm + groups.right_arm),
            "waist_joint_states": values(groups.waist),
            "gripper_states": values(groups.left_effector + groups.right_effector),
            "end_pose": end_pose,
        },
        "prompt": prompt,
        "robot_type": robot_type,
        "task_name": task_name,
        "episode_idx": episode_idx,
        "episode_done": episode_done,
    }
    interval = _history_interval(history_interval)
    if interval:
        params["history"] = {"interval": interval, "images": list(history_frames)}
    return {"method": "infer", "params": params}


def _chunk_values(value: Any, key: str, dimension: int, horizon: int | None) -> np.ndarray:
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "fiu":
            raise ValueError("nonnumeric values")
        array = array.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key}.values must be a numeric chunk") from error
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != dimension:
        raise ValueError(f"{key}.values must have shape (H, {dimension}) with H > 0")
    if horizon is not None and len(array) != horizon:
        raise ValueError(f"{key}.values horizon differs from left_arm")
    if not np.isfinite(array).all():
        raise ValueError(f"{key}.values must be finite")
    return array


def parse_result(
    reply: Mapping[str, Any],
    *,
    controllable_joints: Collection[str],
    held_positions: Mapping[str, float],
    initial_positions: Mapping[str, float],
    groups: JointGroups = WUJI_GROUPS,
) -> ActionChunk:
    """Validate a whole reply before exposing any executable target.

    Engine-fixed/missing joints can only hold their scene-initial value. The
    returned dictionaries include every controllable joint, holding previous
    targets for optional groups the server omitted. EEF and chassis control
    are deliberately unsupported on this path.
    """
    if not isinstance(reply, Mapping):
        raise ValueError("CoRobot reply must be a mapping")
    if reply.get("error"):
        raise RuntimeError(f"inference server error: {reply['error']}")
    result = reply.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("CoRobot reply requires a result mapping")
    interval = _history_interval(result.get("hist_frame_interval", 0) or 0)
    done = result.get("episode_done", False)
    if not isinstance(done, bool):
        raise ValueError("episode_done must be a boolean")
    if done:
        return ActionChunk((), server_episode_done=True, history_interval=interval)
    if result.get("wheels") is not None:
        raise ValueError("wheels/chassis control is unsupported by Newton evaluation")
    held = _finite_positions(held_positions, "held_positions")
    initial = _finite_positions(initial_positions, "initial_positions")
    controlled = tuple(controllable_joints)
    if len(set(controlled)) != len(controlled):
        raise ValueError("controllable joint names must be unique")
    missing = [name for name in controlled if name not in held]
    if missing:
        raise ValueError(f"missing held controllable joints: {missing}")
    baseline = {name: held[name] for name in controlled}
    controlled_set = set(controlled)
    arrays = {}
    horizon = None
    for key in ("left_arm", "right_arm", "left_effector", "right_effector", "waist", "head"):
        block = result.get(key)
        required = key in {"left_arm", "right_arm", "left_effector", "right_effector"}
        if block is None:
            if required:
                raise ValueError(f"CoRobot result requires {key}")
            continue
        if key in {"left_arm", "right_arm", "waist"}:
            if not isinstance(block, Mapping):
                raise ValueError(f"{key} must be a kind/values mapping")
            kind = block.get("kind", "JOINT_ABS")
            accepted = {"JOINT_ABS", "ABS_JOINT"} if key == "waist" else {"JOINT_ABS"}
            if kind not in accepted:
                raise ValueError(f"unsupported {key} action kind: {kind}; only JOINT_ABS is supported")
            values = block.get("values")
        else:
            if isinstance(block, Mapping) and block.get("kind", "JOINT_ABS") != "JOINT_ABS":
                raise ValueError(f"unsupported {key} action kind; only JOINT_ABS is supported")
            values = block.get("values") if isinstance(block, Mapping) else block
        if not required and (values is None or np.asarray(values).size == 0):
            continue
        array = _chunk_values(values, key, len(getattr(groups, key)), horizon)
        if horizon is None:
            horizon = len(array)
        arrays[key] = array
    targets = [baseline.copy() for _ in range(horizon)]
    for key, array in arrays.items():
        for column, name in enumerate(getattr(groups, key)):
            if name in controlled_set:
                for index, target in enumerate(targets):
                    target[name] = float(array[index, column])
            else:
                if name not in initial:
                    raise ValueError(f"fixed/absent joint requires a scene-initial hold: {name}")
                if not np.allclose(array[:, column], initial[name], rtol=0, atol=1e-6):
                    raise ValueError(f"cannot command fixed/absent joint away from its initial hold: {name}")
    return ActionChunk(tuple(targets), history_interval=interval)


class CorobotClient:
    """Synchronous CoRobot binary WebSocket client with bounded network waits."""

    def __init__(self, host: str, port: int | None = None, *, timeout_s: float = 30.0) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}"
        if port is not None:
            if host.startswith(("ws://", "wss://")):
                raise ValueError("pass either a complete WebSocket URI or host and port")
            self.uri += f":{port}"
        self.timeout_s = timeout_s
        self._connection = None

    def infer(
        self,
        payload: Mapping[str, Any],
        *,
        controllable_joints: Collection[str],
        held_positions: Mapping[str, float],
        initial_positions: Mapping[str, float],
        groups: JointGroups = WUJI_GROUPS,
    ) -> ActionChunk:
        """Send one infer envelope; errors close the socket and are never replayed."""
        if payload.get("method") != "infer" or not isinstance(payload.get("params"), Mapping):
            raise ValueError("CoRobot request requires method=infer and params")
        try:
            if self._connection is None:
                self._connection = connect(
                    self.uri, compression=None, max_size=None, open_timeout=self.timeout_s, close_timeout=self.timeout_s
                )
            self._connection.send(msgpack.packb(dict(payload)))
            response = self._connection.recv(timeout=self.timeout_s)
            if isinstance(response, str):
                raise RuntimeError(f"inference server returned text: {response}")
            return parse_result(
                msgpack.unpackb(response, raw=False),
                controllable_joints=controllable_joints,
                held_positions=held_positions,
                initial_positions=initial_positions,
                groups=groups,
            )
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def reset(self) -> None:
        """Drop the prior episode's connection without sending a new wire method."""
        self.close()

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def __enter__(self) -> CorobotClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
