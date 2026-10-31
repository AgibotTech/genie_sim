# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Adapter for the existing CoRobotPolicy msgpack request/result envelope."""

from agent2act.domain.models import ActionChunk, ContractError, Observation, RobotSpec, vector


class CoRobotCodec:
    def __init__(self, spec: RobotSpec, options: dict) -> None:
        if options:
            raise ContractError("corobot codec accepts no options")
        self.spec = spec

    def decode(self, request: dict) -> Observation:
        return decode_request(request, self.spec)

    def encode(
        self, chunk: ActionChunk, observation: Observation, *, history: dict, need_depth: bool
    ) -> dict:
        return encode_result(chunk, history=history, need_depth=need_depth)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


def decode_request(request: dict, spec: RobotSpec) -> Observation:
    if not isinstance(request, dict) or request.get("method") != "infer":
        raise ContractError("expected method=infer")
    p = request.get("params")
    if not isinstance(p, dict) or p.get("robot_type") != spec.robot_type:
        raise ContractError("request robot_type must match configured robot profile")
    states = p.get("states")
    if not isinstance(states, dict):
        raise ContractError("missing states")
    if not isinstance(p.get("prompt"), str):
        raise ContractError("prompt must be a string")
    if type(p.get("episode_done", False)) is not bool:
        raise ContractError("episode_done must be boolean")
    task, episode = p.get("task_name", ""), p.get("episode_idx", 0)
    if not isinstance(task, str) or type(episode) is not int:
        raise ContractError("invalid task_name or episode_idx")
    images = p.get("images", {})
    if not isinstance(images, dict):
        raise ContractError("images must be an object")
    history = p.get("history") or {}
    if not isinstance(history, dict) or not isinstance(history.get("images", []), list):
        raise ContractError("history.images must be an array")
    return Observation(
        robot_type=p["robot_type"],
        joints=vector(states.get("arm_joint_states"), len(spec.left_joints) + len(spec.right_joints), "arms"),
        grippers=vector(states.get("gripper_states"), 2, "grippers"),
        end_pose=states.get("end_pose") or {},
        images=images,
        instruction=p["prompt"],
        episode_key=(task, episode),
        done=p.get("episode_done", False),
        state_groups={key: value for key, value in states.items() if isinstance(value, list)},
        history=history.get("images", []),
        body=(
            vector(states.get("waist_joint_states"), len(spec.body_joints), "body")
            if spec.body_joints
            else []
        ),
    )


def encode_result(chunk: ActionChunk, *, history: dict, need_depth: bool) -> dict:
    return {
        "result": {
            "left_arm": {"kind": chunk.kind, "base_link": chunk.frame, "values": chunk.left_arm},
            "right_arm": {"kind": chunk.kind, "base_link": chunk.frame, "values": chunk.right_arm},
            "left_effector": [[g[0]] for g in chunk.grippers],
            "right_effector": [[g[1]] for g in chunk.grippers],
            **({"waist": {"kind": "JOINT_ABS", "values": chunk.body}} if chunk.body else {}),
            "history": history,
            "need_depth": need_depth,
            **({"episode_done": True} if chunk.metadata.get("episode_terminal") is True else {}),
            # Old clients ignore these. dt_s is a declared assumption, NOT enforced by CoRobotPolicy.
            "agent2act": {"protocol_version": 1, "dt_s": chunk.dt_s, **chunk.metadata},
        }
    }
