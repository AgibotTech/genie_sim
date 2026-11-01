#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""A stand-in inference server that keeps a benchmark run moving without a model.

CoRobotPolicy (source/geniesim/benchmark/policy/corobotpolicy.py) blocks the
simulation loop on every inference round-trip, so with no server listening the
run stalls in connect/retry. This server answers every request with a
JOINT_ABS chunk that repeats the robot's current joint state, so the robot
holds still, the physics keeps stepping, and checkers, timeouts and episode
resets all run as usual.

    python3 scripts/fake_infer_server.py --listen 0.0.0.0:8999

    # separate terminal
    geniesim benchmark run --config g2crsb90d_lego.yaml --benchmark.infer_host=127.0.0.1:8999

Each request is logged on one line: episode, subtask stage, prompt and which
goal images came with it. That makes the log a quick way to check when
instruction_mode=subtask switches stage.

Since the robot never moves, no checker passes. A subtask-mode episode ends
when its first step's step_out runs out; a full-mode episode ends at its
problems.json StepOut.

Gripper values are echoed back unchanged, so the gripper stays where it is for
robots whose process_gripper_action is a passthrough (G2_crsB_90d included).
For robots that relabel the gripper on the way back, pass --gripper to send a
fixed value instead.
"""

import argparse
import asyncio
import time

import msgpack
import websockets


def hold_still_response(params, chunk, gripper):
    """A chunk of `chunk` identical waypoints at the current joint state."""
    states = params.get("states") or {}
    arm = list(states.get("arm_joint_states") or [])
    grip = list(states.get("gripper_states") or [])
    if gripper is not None:
        grip = [gripper] * len(grip)
    half_arm, half_grip = len(arm) // 2, len(grip) // 2
    return {
        "result": {
            "left_arm": {"kind": "JOINT_ABS", "values": [arm[:half_arm]] * chunk},
            "right_arm": {"kind": "JOINT_ABS", "values": [arm[half_arm:]] * chunk},
            "left_effector": [grip[:half_grip]] * chunk,
            "right_effector": [grip[half_grip:]] * chunk,
        }
    }


def describe(params):
    subtasks = params.get("subtask_list") or {}
    stage = subtasks.get("current_index")
    total = len(subtasks.get("subtasks") or [])
    goal = params.get("subgoal_images") or {}
    goal_desc = ",".join(f"{cam}:{img.get('width')}x{img.get('height')}" for cam, img in goal.items()) or "none"
    return (
        f"task={params.get('task_name')} scene={params.get('scene_instance_id')} "
        f"ep={params.get('episode_idx')} reset={params.get('on_reset')} "
        f"stage={stage}/{total} goal_images={goal_desc} prompt={params.get('prompt')!r}"
    )


async def handler(websocket, args):
    peer = websocket.remote_address
    print(f"[fake] client connected: {peer}", flush=True)
    count = 0
    try:
        async for message in websocket:
            payload = msgpack.unpackb(message, raw=False)
            params = payload.get("params") or {}
            count += 1
            print(f"[fake] #{count} {time.strftime('%H:%M:%S')} {describe(params)}", flush=True)
            if args.delay > 0:
                await asyncio.sleep(args.delay)
            await websocket.send(msgpack.packb(hold_still_response(params, args.chunk, args.gripper)))
    except websockets.ConnectionClosed:
        pass
    print(f"[fake] client disconnected: {peer} after {count} request(s)", flush=True)


async def serve(host, port, args):
    async with websockets.serve(lambda ws: handler(ws, args), host, port, compression=None, max_size=None):
        print(f"[fake] listening on ws://{host}:{port} (chunk={args.chunk}, delay={args.delay}s)", flush=True)
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="0.0.0.0:8999", help="host:port to serve on")
    ap.add_argument("--chunk", type=int, default=30, help="waypoints per reply; larger means fewer round-trips")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait before replying, to mimic model latency")
    ap.add_argument("--gripper", type=float, default=None, help="send this gripper value instead of echoing the state")
    args = ap.parse_args()
    if args.chunk < 1:
        ap.error("--chunk must be >= 1")
    host, port = args.listen.rsplit(":", 1)
    asyncio.run(serve(host, int(port), args))


if __name__ == "__main__":
    main()
