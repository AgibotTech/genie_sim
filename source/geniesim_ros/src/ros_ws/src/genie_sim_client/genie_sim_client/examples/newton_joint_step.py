# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Reset a Newton scene, step joint targets and print the final state.

    python3 -m genie_sim_client.examples.newton_joint_step \
        --launcher /path/launcher.yaml --manifest /path/stage/manifest.json \
        --steps 10

Without --joint/--target, hold the post-reset joint positions. This example
reports state only; it does not evaluate a benchmark task or prove exact K ticks.
"""

from __future__ import annotations

import argparse
import json
import math

from genie_sim_client import ClientConfig, SimulationClient


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--steps", type=int, default=10, help="explicit shared-client step requests")
    parser.add_argument("--joint", default=None)
    parser.add_argument("--target", type=float, default=None, help="radians for revolute, metres for prismatic joints")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if (args.joint is None) != (args.target is None):
        parser.error("--joint and --target must be supplied together")
    if args.target is not None and not math.isfinite(args.target):
        parser.error("--target must be finite")

    config = ClientConfig.from_files(args.launcher, args.manifest)
    with SimulationClient(config) as client:
        snapshot = client.reset()
        targets = snapshot.joint_positions.copy()
        if args.joint is not None:
            if args.joint not in snapshot.schema.joint_names:
                parser.error(f"unknown engine joint: {args.joint}")
            targets[:, snapshot.schema.joint_names.index(args.joint)] = args.target
        for _ in range(args.steps):
            snapshot = client.step(targets).snapshot

    state = {
        "completion": snapshot.completion,
        "step_requests": args.steps,
        "joint_names": list(snapshot.schema.joint_names),
        "joint_positions": snapshot.joint_positions.tolist(),
        "reset_epoch": snapshot.reset_epoch,
    }
    print(json.dumps(state, indent=2))
    return state


if __name__ == "__main__":
    main()
