#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run finite, model-free chunk/reset checks against an explicitly selected ZMQ server.

Use --attach SUB_TASK_NAME for an initialized server, or --task-init FILE to
initialize it from a task YAML. The default action holds the reset joint pose.
The client detaches on exit and leaves the server running.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from rlinf_geniesim.envs.vec_env import GenieSimVecEnv


def make_actions(hold: np.ndarray, *, chunk: int) -> np.ndarray:
    """Repeat the current episode's controlled joint targets across a chunk."""
    return np.repeat(hold[:, None, :], chunk, axis=1)


def validate_observation(obs: dict, num_envs: int, state_dim: int) -> None:
    """Check the resolved state/instruction contract and any supplied camera frames."""
    states = np.asarray(obs["states"])
    assert states.shape == (num_envs, state_dim), f"Expected states {(num_envs, state_dim)}, found {states.shape}"
    assert states.dtype == np.float32 and np.isfinite(states).all(), "Expected finite float32 states"
    assert len(obs["task_descriptions"]) == num_envs, "Instruction count differs from environment count"
    for key in ("main_images", "wrist_images", "wrist_images_right"):
        if key not in obs or obs[key] is None:
            continue
        image = np.asarray(obs[key])
        assert (
            image.ndim == 4 and image.shape[0] == num_envs and image.shape[-1] == 3
        ), f"Expected {key} shape (N, H, W, 3), found {image.shape}"
        assert image.dtype == np.uint8 and image.size > 0, f"Expected nonempty uint8 {key}"


def validate_chunk(rewards, terms, truncs, infos, elapsed, *, num_envs: int, chunk: int, max_steps: int) -> np.ndarray:
    """Validate chunk boundaries and the final-observation auto-reset protocol."""
    rewards, terms, truncs = np.asarray(rewards), np.asarray(terms), np.asarray(truncs)
    for name, array in (("rewards", rewards), ("terminations", terms), ("truncations", truncs)):
        assert array.shape == (num_envs, chunk), f"Expected {name} {(num_envs, chunk)}, found {array.shape}"
    assert np.isfinite(rewards).all(), "Rewards contain non-finite values"
    assert terms.dtype == bool and truncs.dtype == bool, "Expected boolean done flags"
    assert not terms[:, :-1].any() and not truncs[:, :-1].any(), "Done flags appeared before the chunk end"
    dones = terms[:, -1] | truncs[:, -1]
    if dones.any():
        assert "final_observation" in infos and "final_info" in infos, "Auto-reset omitted final observation/info"
        assert (np.asarray(elapsed)[dones] == 0).all(), "Auto-reset did not clear elapsed steps"
        lengths = np.asarray(infos["final_info"]["episode"]["length"])
        assert lengths.shape == (num_envs,), f"Expected episode lengths {(num_envs,)}, found {lengths.shape}"
        # A time limit inside a chunk takes effect at its end, without an intermediate reset.
        assert (
            (lengths[dones] > 0) & (lengths[dones] <= max_steps + chunk - 1)
        ).all(), f"Episode lengths exceed the chunk/time-limit contract: {lengths[dones]}"
    return dones


def initialize_env(env: GenieSimVecEnv, task_init: dict, *, timeout_s: float) -> None:
    """Initialize using the task YAML's explicit backend and workload configuration."""
    task = dict(task_init["task"])
    backends = dict(task_init["backends"])
    num_envs = backends.pop("num_envs", None)
    if num_envs is not None:
        if "num_envs" in task and task["num_envs"] != num_envs:
            raise ValueError("task.num_envs and backends.num_envs disagree")
        task["num_envs"] = num_envs
    env.client.set_backends(backends)
    config = {key: value for key, value in task_init.items() if key not in ("task", "backends")}
    env.init(task, config=config, timeout_s=timeout_s)


def run_smoke(
    client,
    *,
    attach: str = "",
    task_init: dict | None = None,
    chunks: int = 15,
    chunk: int = 4,
    max_steps: int = 12,
    timeout_s: float = 900.0,
) -> dict:
    """Exercise the current VecEnv API; the caller owns and closes the connection."""
    env = GenieSimVecEnv(client, to_torch=False, max_episode_steps=max_steps)
    try:
        if task_init is not None:
            initialize_env(env, task_init, timeout_s=timeout_s)
        else:
            env.attach_task(attach)
        obs, _ = env.reset()
        validate_observation(obs, env.num_envs, env.state_dim)
        # Controlled joints are the arm/gripper/(optional waist) prefix of states.
        hold = np.asarray(obs["states"])[:, : env.action_dim].copy()
        episodes = 0
        start = time.monotonic()
        for _ in range(chunks):
            actions = make_actions(hold, chunk=chunk)
            obs, rewards, terms, truncs, infos = env.step(actions)
            validate_observation(obs, env.num_envs, env.state_dim)
            dones = validate_chunk(
                rewards,
                terms,
                truncs,
                infos,
                env.elapsed_steps,
                num_envs=env.num_envs,
                chunk=chunk,
                max_steps=max_steps,
            )
            if dones.any():
                validate_observation(infos["final_observation"], env.num_envs, env.state_dim)
                hold[dones] = np.asarray(obs["states"])[dones, : env.action_dim]
                episodes += int(dones.sum())
        minimum_episodes = env.num_envs * (chunks // math.ceil(max_steps / chunk))
        assert episodes >= minimum_episodes, f"Expected at least {minimum_episodes} auto-resets, found {episodes}"
        return {
            "num_envs": env.num_envs,
            "steps": chunks * chunk,
            "episodes": episodes,
            "wall_s": time.monotonic() - start,
        }
    finally:
        env.close()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number")
    return parsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--address", required=True, help="ZMQ endpoint of the selected server")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--attach", metavar="SUB_TASK_NAME", help="Adopt the task already loaded on the server")
    mode.add_argument(
        "--task-init", type=Path, metavar="YAML", help="Initialize a task from explicit task/backend settings"
    )
    parser.add_argument("--chunks", type=_positive_int, default=15)
    parser.add_argument("--chunk", type=_positive_int, default=4)
    parser.add_argument("--max-steps", type=_positive_int, default=12)
    parser.add_argument("--timeout-s", type=_positive_float, default=900.0)
    args = parser.parse_args(argv)
    if not args.address.strip():
        parser.error("--address must name a server endpoint")
    task_init = None
    if args.task_init is not None:
        import yaml

        try:
            task_init = yaml.safe_load(args.task_init.read_text())
            if not isinstance(task_init, dict) or not isinstance(task_init.get("task"), dict):
                raise ValueError("Task YAML must contain a task mapping")
            if (
                not isinstance(task_init.get("backends"), dict)
                or not task_init["backends"].get("physics")
                or not task_init["backends"].get("render")
            ):
                raise ValueError("Task YAML must declare backends.physics and backends.render")
            if not task_init["task"].get("sub_task_name"):
                raise ValueError("Task YAML must declare task.sub_task_name")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            parser.error(str(exc))
    from rlinf_geniesim.comm import make_client

    client = None
    status = 0
    try:
        client = make_client("zmq", address=args.address, timeout_s=args.timeout_s)
        result = run_smoke(
            client,
            attach=args.attach or "",
            task_init=task_init,
            chunks=args.chunks,
            chunk=args.chunk,
            max_steps=args.max_steps,
            timeout_s=args.timeout_s,
        )
    except Exception as exc:
        print(f"[smoke] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        status = 1
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                print(f"[smoke] FAIL closing connection: {exc}", file=sys.stderr, flush=True)
                status = 1
    if status == 0:
        print(f"[smoke] PASS: {result}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
