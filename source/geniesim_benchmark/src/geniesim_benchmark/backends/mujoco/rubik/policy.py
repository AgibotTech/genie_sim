# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CoRobot-compatible wire interface with an explicit 54-joint action contract."""

from contextlib import ExitStack
from inspect import signature
from io import BytesIO
import time
import msgpack
import numpy as np
from PIL import Image


class PolicyError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


def payload_for(env, episode, images, prompt="Solve the Rubik's Cube"):
    now = time.time_ns()
    encoded = {}
    for name, image in images.items():
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=95)
        encoded[name] = {
            "encoding": "JPEG",
            "image_data": buffer.getvalue(),
            "height": image.shape[0],
            "width": image.shape[1],
        }
    return {
        "method": "infer",
        "params": {
            "timestamps": {"head": now, "states": now},
            "simulation_time": float(env.data.time),
            "images": encoded,
            "states": env.states(),
            "robot_type": "g2a_sim",
            "ee_type": "wuji",
            "prompt": prompt,
            "task_name": getattr(env, "task_name", "rubik_cube_solve_1turn"),
            "episode_idx": episode,
            "episode_start": env.steps == 0,
            "episode_done": False,
            "task_progress": [],
            "action_contract": env.action_contract(),
        },
    }


def joint_array(value):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError("Joint targets must contain numeric values, excluding booleans and strings")
    return array.astype(float)


def parse_actions(reply, contract, max_chunk_size, hand_target_tolerance=0.0, projection_log=None):
    """Reject unsupported kinds, shapes, limits and ambiguous action encodings."""
    if not np.isfinite(hand_target_tolerance) or not 0 <= hand_target_tolerance <= 0.05:
        raise PolicyError("Hand target tolerance must be in [0,0.05] radians")
    if not isinstance(reply, dict) or reply.get("error"):
        raise PolicyError(
            f"Policy error reply: {reply.get('error') if isinstance(reply, dict) else type(reply).__name__}"
        )
    result = reply.get("result")
    if not isinstance(result, dict):
        raise PolicyError("Reply must contain a result mapping")
    done = result.get("episode_done", False)
    if type(done) is not bool:
        raise PolicyError("episode_done must be boolean")
    if result.get("head") or result.get("waist") or result.get("wheels"):
        raise PolicyError("The reference model has a fixed head, torso and chassis")
    if result.get("hist_frame_interval", 0):
        raise PolicyError("History images are not implemented in the native Rubik interface")
    try:
        if "joint_targets" in result:
            if any(k in result for k in ("left_arm", "right_arm", "left_effector", "right_effector")):
                raise ValueError("Ambiguous joint_targets and CoRobot actions")
            if result.get("kind") != "JOINT_ABS":
                raise ValueError("joint_targets requires kind=JOINT_ABS")
            actions = joint_array(result["joint_targets"])
        else:
            if done and not any(k in result for k in ("left_arm", "right_arm", "left_effector", "right_effector")):
                return np.empty((0, 54)), done
            arrays = []
            for name, width in (("left_arm", 7), ("right_arm", 7), ("left_effector", 20), ("right_effector", 20)):
                value = result[name]
                if name.endswith("arm"):
                    if not isinstance(value, dict) or value.get("kind") != "JOINT_ABS":
                        raise ValueError(f"{name} requires kind=JOINT_ABS")
                    value = value["values"]
                arr = joint_array(value)
                if arr.ndim != 2 or arr.shape[1] != width:
                    raise ValueError(f"{name} must have shape [T,{width}]")
                arrays.append(arr)
            actions = np.concatenate(arrays, axis=1)
        if actions.ndim != 2 or actions.shape[1] != 54 or not 1 <= len(actions) <= max_chunk_size:
            raise ValueError(f"Expected joint chunk [T,54], 1<=T<={max_chunk_size}")
        bounds = np.asarray(contract["bounds"])
        if not np.isfinite(actions).all():
            raise ValueError("Joint targets must be finite")
        violation = np.maximum(np.maximum(bounds[:, 0] - actions, actions - bounds[:, 1]), 0)
        if np.any(violation[:, :14] > 1e-6):
            raise ValueError("Arm targets violate action_contract bounds")
        if np.any(violation[:, 14:] > hand_target_tolerance + 1e-6):
            raise ValueError(
                f"Hand targets violate action_contract bounds: maximum excess {violation[:, 14:].max():.6f} rad"
            )
        if hand_target_tolerance:
            projected = np.clip(actions[:, 14:], bounds[14:, 0], bounds[14:, 1])
            difference = np.abs(actions[:, 14:] - projected)
            if projection_log is not None:
                projection_log.update(
                    changed_values=int(np.count_nonzero(difference)), maximum_projection_radians=float(difference.max())
                )
            actions[:, 14:] = projected
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise PolicyError(str(error)) from error
    return actions, done


class WebsocketPolicy:
    def __init__(self, host, timeout):
        self.host, self.timeout, self.connection = host, timeout, None
        self.stack = ExitStack()

    def reset(self):
        self.close()

    def infer(self, payload):
        from websockets.sync.client import connect

        try:
            if self.connection is None:
                parameters = signature(connect).parameters
                # In 14.x, **kwargs are socket options, not keepalive support.
                keepalive = {
                    name: value
                    for name, value in (("ping_interval", 20), ("ping_timeout", 60))
                    if name in parameters
                }
                self.connection = self.stack.enter_context(
                    connect(
                        "ws://" + self.host,
                        compression=None,
                        max_size=64 * 1024 * 1024,
                        open_timeout=self.timeout,
                        close_timeout=2,
                        **keepalive,
                    )
                )
            self.connection.send(msgpack.packb(payload, use_bin_type=True))
            response = self.connection.recv(timeout=self.timeout)
        except Exception as error:
            self.close()
            raise InferenceError(f"{type(error).__name__}: {error}") from error
        if not isinstance(response, bytes):
            raise PolicyError("Policy must return a binary MessagePack reply")
        try:
            return msgpack.unpackb(response, raw=False)
        except (ValueError, msgpack.ExtraData, msgpack.FormatError) as error:
            raise PolicyError("Malformed MessagePack reply") from error

    def close(self):
        if self.connection is not None:
            try:
                self.stack.close()
            finally:
                self.connection = None


class HoldPolicy:
    """Interface smoke policy; cannot solve a scrambled cube."""

    def reset(self):
        self.target = None

    def infer(self, payload):
        if self.target is None:
            state = payload["params"]["states"]
            self.target = state["arm_joint_states"] + state["gripper_states"]
        return {"result": {"kind": "JOINT_ABS", "joint_targets": [self.target] * 8}}

    def close(self):
        pass
