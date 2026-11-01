#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Exercise FusedIsaacBackend against the real Isaac node.

Requires the simulator runtime and a manifest. Build one with fused_smoke.py
or pass --manifest. The default manifest is /tmp/fused_smoke/manifest.json.
Checks startup, schema, step, and idempotent close through the backend API.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "source").is_dir():
        sys.path.insert(0, str(_p / "source"))
        break

from rlinf_geniesim.physics.backends import make_engine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="/tmp/fused_smoke/manifest.json")
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    if not Path(args.manifest).is_file():
        print(f"[integ] no manifest at {args.manifest} -- run fused_smoke.py " f"first, or pass --manifest", flush=True)
        return 2

    be = make_engine(
        "isaac_fused",
        scene_yaml="",
        robot_urdf="",
        num_envs=args.num_envs,
        physics_hz=120.0,
        control_hz=30.0,
        stage_manifest=args.manifest,
        cameras=[{"name": "head_front_Camera", "image_hw": [400, 640]}],
        env_spacing=15.0,
    )

    try:
        print("[integ] starting backend (launches the real Isaac node)...", flush=True)
        be.start()
        print(f"[integ] started. action_dim={be.action_dim} " f"state_dim={be.state_dim}", flush=True)

        schema = be.get_scene_schema()
        print(
            f"[integ] schema: {len(schema['joint_names'])} joints, "
            f"{len(schema['body_paths'])} bodies, "
            f"{len(schema['object_names'])} objects",
            flush=True,
        )
        if not schema["joint_names"]:
            print("[integ] FAIL: schema has no joints", flush=True)
            return 1

        be.reset()
        js0 = be.get_joint_states().copy()
        # Command the CURRENT positions so any motion is physics, not our action.
        hold = js0[:, :, 0].astype(np.float32)
        for i in range(args.steps):
            be.step(hold)
            print(f"[integ] step {i} ok sim_time={be.get_sim_time()[0]:.4f}", flush=True)
        delta = float(np.abs(be.get_joint_states() - js0).max())
        print(f"[integ] joint delta over {args.steps} steps: {delta:.3e}", flush=True)
    finally:
        be.close()

    # Shared engine/render ownership requires repeated close calls to be safe.
    be.close()
    print("[integ] ===== BACKEND <-> REAL NODE OK =====", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
