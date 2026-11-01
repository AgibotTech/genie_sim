#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise G2 policy chunks, observations, and auto-reset against a ZMQ server.

This integration check perturbs G2 arm targets and toggles its grippers. It
expects the G2 24-state/16-action layout; use scripts/smoke_test.py for a
robot-layout-aware hold check. Optional torch output exercises the client device
boundary. The task is initialized from the command-line arguments.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from rlinf_geniesim.envs.vec_env import GenieSimVecEnv

try:
    import torch

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None
    _HAS_TORCH = False


class FakeChunkPolicy:
    """G2 arm perturbation and gripper toggling around the reset pose."""

    def __init__(self, num_envs: int, action_dim: int, hold: np.ndarray, chunk: int, seed: int = 0):
        self.n, self.a, self.chunk = num_envs, action_dim, chunk
        self.hold = hold.copy()  # (N, A) reset targets
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0, 2 * np.pi, (num_envs, 14)).astype(np.float32)
        self.grip_period = (6 + rng.integers(0, 6, num_envs)).astype(np.int64)
        self.t = 0

    def act(self, obs) -> np.ndarray:
        # a real policy would consume obs["main_images"/"states"/...]; the fake
        # one only asserts they exist (contract check happens in the driver).
        assert "states" in obs and "task_descriptions" in obs
        out = np.empty((self.n, self.chunk, self.a), np.float32)
        for k in range(self.chunk):
            t = self.t + k
            a = self.hold.copy()
            a[:, :14] += 0.08 * np.sin(0.3 * t + self.phase)
            closed = (t // self.grip_period) % 2 == 1
            # gripper master: hold(init)=0.785 rad OPEN, 0.0 = closed
            a[:, 14:16] = np.where(closed[:, None], 0.0, self.hold[:, 14:16])
            out[:, k] = a
        self.t += self.chunk
        return out


def _np(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "bootstrap",
        nargs="?",
        default="/tmp/boot.json",
        help="[unused] pre-migration SHM bootstrap-file path; the "
        "client+env.init() construction below only ever "
        "reads --address now — kept so an old invocation "
        "that still passes this positional doesn't error on "
        "an unrecognized argument",
    )
    p.add_argument(
        "--comm",
        default="shm",
        help="'zmq' is the only supported value — SHM has no "
        "init()/task_metadata verb (ShmClient.init() raises "
        "by design) so it cannot serve env.init(); default "
        "stays 'shm' (this script's historical default) "
        "purely so a no-flags invocation fails LOUD here "
        "instead of silently degrading deep inside a half "
        "connection",
    )
    p.add_argument(
        "--address", default="", help="zmq address (tcp://<sim-host>:<port>) of the " "serving sim; implies --comm zmq"
    )
    p.add_argument("--task-name", default="table_task_1_g2_op", help="benchmark task_name (background+robot combo)")
    p.add_argument("--sub-task-name", default="pick_block_color", help="benchmark sub_task_name (the actual task)")
    p.add_argument(
        "--task-desc",
        default="",
        help="task instruction override (remote machines without " "the benchmark config files read an empty one)",
    )
    p.add_argument("--physics", default="newton_mujoco_warp")
    p.add_argument("--physics-solver", default="mujoco-warp")
    p.add_argument("--physics-solver-substep", type=int, default=10)
    p.add_argument("--render", default="isaac_tiled")
    p.add_argument("--sim-device", default="cuda:0", help="Server device within its CUDA_VISIBLE_DEVICES selection")
    p.add_argument(
        "--no-set-backends",
        action="store_true",
        help="Retain the server backend settings locked at its first initialization",
    )
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--chunks", type=int, default=15)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--device", default="cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu")
    args = p.parse_args()

    comm = "zmq" if args.address else args.comm
    if comm != "zmq":
        # SHM has no task initialization or metadata channel.
        raise SystemExit(
            f"--comm {comm!r} not supported: GenieSimVecEnv now always "
            f"builds via client + env.init(), and SHM has no init()/"
            f"task_metadata verb to serve that. Run this against an "
            f"ALREADY-SERVING sim over zmq instead: pass "
            f"--address tcp://<sim-host>:<port> (zmq only) — e.g. start "
            f"the sim with sim_server.py --comm-config ... then build "
            f"the task (scripts/init_task.py) "
            f"and point --address at its bound port, then re-run with "
            f"--address (no --comm/positional bootstrap file needed)."
        )

    from rlinf_geniesim.comm import make_client

    client = make_client("zmq", address=args.address)
    # Backends lock at first initialization; reuse them when requested.
    if not args.no_set_backends:
        client.set_backends(
            {
                "physics": args.physics,
                "physics_solver": args.physics_solver,
                "physics_solver_substep": args.physics_solver_substep,
                "render": args.render,
                "device": args.sim_device,
            }
        )
    env = GenieSimVecEnv(
        client,
        to_torch=_HAS_TORCH,
        device=args.device,
        max_episode_steps=args.max_steps,
        task_description=args.task_desc,
    )
    env.init({"task_name": args.task_name, "sub_task_name": args.sub_task_name, "num_envs": args.num_envs})
    N, A = env.num_envs, env.action_dim
    print(
        f"[rollout] env up: {N} env(s), action_dim={A}, " f"comm=zmq @{args.address}, torch={_HAS_TORCH}({args.device})"
    )

    obs, _ = env.reset()
    # --- contract check 1: obs dict ---
    # Cameras share a common WIDTH but keep their own benchmark aspect/height:
    # head 640x400 (1.60), wrists 640x528 (1.21, native 1280x1056 downscaled).
    for key, want in (
        ("states", (N, 24)),
        ("main_images", (N, 400, 640, 3)),
        ("wrist_images", (N, 528, 640, 3)),
        ("wrist_images_right", (N, 528, 640, 3)),
    ):
        arr = _np(obs[key])
        assert arr.shape == want, f"{key} shape {arr.shape} != {want}"
    assert _np(obs["main_images"]).dtype == np.uint8
    if _HAS_TORCH:
        assert isinstance(obs["states"], torch.Tensor)
        assert str(obs["states"].device).startswith(args.device)
    assert len(obs["task_descriptions"]) == N
    if not obs["task_descriptions"][0]:
        print(
            "[rollout] NOTE: empty instruction (benchmark configs absent "
            "on this machine — pass --task-desc to set one)"
        )
    print(f"[rollout] obs contract OK; instruction: " f"{obs['task_descriptions'][0][:50]!r}")

    policy = FakeChunkPolicy(N, A, _np(env._hold_targets)[:, env._act_idx], args.chunk)
    episodes_done = 0
    lengths: list[int] = []
    t0 = time.monotonic()
    for ci in range(args.chunks):
        chunk_actions = policy.act(obs)
        obs, rewards, terms, truncs, infos = env.step(chunk_actions)

        r, te, tr = _np(rewards), _np(terms).astype(bool), _np(truncs).astype(bool)
        assert r.shape == (N, args.chunk) and te.shape == (N, args.chunk)
        # --- contract check 2: dones only on the LAST chunk column ---
        assert not te[:, :-1].any() and not tr[:, :-1].any(), "mid-chunk dones leaked (auto_reset semantics broken)"
        dones = te[:, -1] | tr[:, -1]
        if dones.any():
            fi = infos
            assert "final_observation" in fi and "final_info" in fi
            fl = _np(fi["final_info"]["episode"]["length"])
            lengths += [int(v) for v in fl[dones]]
            episodes_done += int(dones.sum())
            # --- contract check 3: auto-reset zeroed the counters ---
            el = _np(env.elapsed_steps)
            assert (el[dones] == 0).all(), "elapsed not reset after done"
    dt = time.monotonic() - t0
    steps = args.chunks * args.chunk
    print(
        f"[rollout] {args.chunks} chunks x {args.chunk} = {steps} steps in "
        f"{dt:.1f}s ({steps / dt:.2f} steps/s, {steps * N / dt:.1f} env-steps/s)"
    )
    print(
        f"[rollout] episodes finished: {episodes_done} "
        f"(expected ~{N * steps // args.max_steps}); "
        f"episode lengths: {sorted(set(lengths))}"
    )
    assert episodes_done >= N * steps // args.max_steps - N, "too few episodes — auto-reset not cycling"
    assert all(
        v == args.max_steps for v in lengths
    ), f"episode length != max_steps (truncation accounting broken): {lengths}"
    env.close()
    print("[rollout] RLINF-INTERACTION SMOKE: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
