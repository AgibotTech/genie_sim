#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Check the fused Isaac node through its raw wire protocol.

Builds a manifest, launches the node, and checks schema, finite state,
physics motion, and nonconstant rendered frames. Run in the simulator runtime:
    python3 source/rlinf_geniesim/tests/integration/fused_smoke.py --num-envs 2 --steps 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "source").is_dir():
        sys.path.insert(0, str(_p / "source"))
        break

from rlinf_geniesim.comm.shm.sem import SemPair  # noqa: E402
from rlinf_geniesim.physics.fused_shm import (  # noqa: E402
    CMD_RESET,
    CMD_SHUTDOWN,
    CMD_STEP,
    STATUS_OK,
    FusedStateShm,
)
from rlinf_geniesim.render.tiled_shm import TiledFrameShm  # noqa: E402

_NODE = "source/geniesim_ros/src/ros_ws/src/genie_sim_render/scripts/" "isaacsim_fused_sim.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task-name", default="table_task_1_g2_op")
    p.add_argument("--sub-task", default="pick_block_color")
    p.add_argument("--instance", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--env-spacing", type=float, default=15.0)
    p.add_argument("--physics-hz", type=float, default=120.0)
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--workdir", default="/tmp/fused_smoke")
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--save-dir", default="/tmp/fused_smoke/shots")
    p.add_argument("--keep-manifest", default="", help="reuse this manifest instead of building one")
    return p.parse_args()


def build_manifest(args, work: Path) -> str:
    if args.keep_manifest:
        return args.keep_manifest
    from rlinf_geniesim.task_loader import build_stage_manifest

    # ``prim_path`` (not just ``link``) is what decides the manifest's camera
    # path: task_loader does ``c.get("prim_path") or c.get("link") or name``,
    # so passing only ``link`` yields path="head_link3" and the stage build
    # then looks for a camera prim literally named "head_link3" and fails.
    cameras = [
        {
            "name": "head_front_Camera",
            "prim_path": "head_link3/head_front_Camera",
            "link": "head_link3",
            "image_hw": [400, 640],
        }
    ]
    out = str(work / "manifest.json")
    # num_envs=1 ON PURPOSE. The manifest describes ONE env; the node's
    # GridCloner is what replicates it to N. Passing N here instead builds a
    # manifest that ALREADY contains /World/envs/env_*/... and the cloner
    # then clones that again, doubling every prim under a nested
    # /World/envs/env_0/World/envs/env_0/... path. sim_server's
    # _split_cfg_pieces states this split: "PHYSICS always keeps the
    # single-env manifest".
    # author_render=True is what the real server passes when rendering is on
    # (sim_server.py's build_stage_manifest call sites use author_render=
    # render_on). It authors the RENDER LAYER, and the render layer is where
    # the LIGHTS live: a scene whose own USD has no light prims of its own
    # renders pure black on every camera without this.
    path = build_stage_manifest(
        args.task_name,
        args.sub_task,
        args.instance,
        cameras,
        out,
        num_envs=1,
        env_spacing=args.env_spacing,
        author_render=True,
    )
    print(f"[smoke] manifest: {path}", flush=True)
    return path


def main() -> int:
    args = parse_args()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args, work)
    schema_out = work / "schema.json"
    ready_file = work / "ready"
    for f in (schema_out, ready_file):
        if f.exists():
            f.unlink()

    tag = str(os.getpid())
    state_name = f"fused_smoke_state_{tag}"
    tiled_name = f"fused_smoke_tiled_{tag}"
    req_sem, resp_sem = f"/fused_smoke_req_{tag}", f"/fused_smoke_resp_{tag}"

    # Client owns ONLY the semaphores -- the node sizes and creates both SHM
    # segments from the built stage. See fused_shm.FusedStateShm's docstring.
    sems = SemPair(req_sem, resp_sem, create=True)

    repo = _HERE.parents[4]
    cmd = [
        "python3",
        str(repo / _NODE),
        "--manifest",
        manifest,
        "--shm-name",
        tiled_name,
        "--state-shm",
        state_name,
        "--req-sem",
        req_sem,
        "--resp-sem",
        resp_sem,
        "--schema-out",
        str(schema_out),
        "--ready-file",
        str(ready_file),
        "--num-envs",
        str(args.num_envs),
        "--env-spacing",
        str(args.env_spacing),
        "--physics-hz",
        str(args.physics_hz),
        "--control-hz",
        str(args.control_hz),
    ]
    print(f"[smoke] launching: {' '.join(cmd)}", flush=True)
    env = dict(os.environ, SIM_REPO_ROOT=str(repo))
    proc = subprocess.Popen(cmd, env=env)

    state = None
    tiled = None
    failures: list[str] = []
    try:
        # 1. wait for ready
        t0 = time.monotonic()
        while not ready_file.exists():
            if proc.poll() is not None:
                print(
                    f"[smoke] FAIL: node exited early (rc={proc.returncode}) " f"before writing {ready_file}",
                    flush=True,
                )
                return 1
            if time.monotonic() - t0 > args.timeout_s:
                print(f"[smoke] FAIL: node not ready after {args.timeout_s}s", flush=True)
                proc.kill()
                return 1
            time.sleep(0.5)
        print(f"[smoke] node ready in {time.monotonic() - t0:.1f}s", flush=True)

        # 2. schema sanity
        schema = json.loads(schema_out.read_text())
        n_joints = len(schema["joint_names"])
        n_bodies = len(schema["body_paths"])
        n_objects = len(schema["object_names"])
        print(f"[smoke] schema: {n_joints} joint(s), {n_bodies} body(ies), " f"{n_objects} object(s)", flush=True)
        if n_joints == 0:
            failures.append(
                "schema has zero joints -- the articulation view " "found no DOFs, so nothing can be actuated"
            )
        if n_bodies == 0:
            failures.append(
                "schema has zero bodies -- no rigid bodies were " "discovered, so physics has nothing to simulate"
            )

        state = FusedStateShm(
            state_name,
            num_envs=args.num_envs,
            action_dim=n_joints,
            num_joints=n_joints,
            num_bodies=n_bodies,
            num_objects=n_objects,
            create=False,
        )

        def _roundtrip(cmd_id: int, label: str) -> bool:
            state.set_command(cmd_id, reset_mask_valid=False)
            seq_before = state.seq
            sems.post_request()
            sems.wait_response(timeout=args.timeout_s)
            if state.seq == seq_before:
                failures.append(f"{label}: seq did not advance")
                return False
            if state.status != STATUS_OK:
                failures.append(f"{label}: node reported error: " f"{state.read_error()}")
                return False
            return True

        # 3. reset, then step, recording state each time
        if not _roundtrip(CMD_RESET, "reset"):
            raise SystemExit(1)
        js_hist, bp_hist = [], []
        # Hold the arm still: command the CURRENT joint positions every step, so
        # anything that moves does so because PHYSICS moved it (gravity, contact,
        # settling) -- not because we drove it there. That is what makes step 4
        # a real test of physics rather than of our own action plumbing.
        hold = state.joint_states[:, :, 0].copy()
        for i in range(args.steps):
            state.actions[:] = hold
            t_s = time.monotonic()
            if not _roundtrip(CMD_STEP, f"step {i}"):
                break
            js_hist.append(state.joint_states.copy())
            bp_hist.append(state.body_poses.copy())
            print(
                f"[smoke] step {i}: {(time.monotonic() - t_s) * 1000:.0f}ms " f"sim_time={state.sim_time[0]:.4f}",
                flush=True,
            )

        if js_hist:
            if not np.isfinite(js_hist[-1]).all():
                failures.append("joint_states contain non-finite values")
            if not np.isfinite(bp_hist[-1]).all():
                failures.append("body_poses contain non-finite values")

        # 4. physics actually advanced
        if len(js_hist) >= 2:
            d_joint = float(np.abs(js_hist[-1] - js_hist[0]).max())
            d_body = float(np.abs(bp_hist[-1] - bp_hist[0]).max())
            print(
                f"[smoke] max |delta| over {len(js_hist)} steps: " f"joints={d_joint:.6g} bodies={d_body:.6g}",
                flush=True,
            )
            if d_joint == 0.0 and d_body == 0.0:
                failures.append(
                    "state is byte-identical across steps -- physics is NOT "
                    "running (a stage whose physics APIs were stripped looks "
                    "exactly like this)"
                )
        else:
            failures.append("fewer than 2 successful steps; cannot tell " "whether physics advanced")

        # 5. render actually rendered
        # TiledFrameShm needs dims up front to map its view, and only the node
        # knows the tile geometry it chose (tiles = envs x cameras, height =
        # the tallest camera). Read the header raw first, then map properly.
        from multiprocessing.shared_memory import SharedMemory

        raw = SharedMemory(name=tiled_name, create=False)
        hdr = np.ndarray((8,), dtype=np.int64, buffer=raw.buf[:64])
        t_envs, t_h, t_w = int(hdr[1]), int(hdr[2]), int(hdr[3])
        frame_id = int(hdr[5])
        raw.close()
        print(f"[smoke] tiled SHM: {t_envs} tile(s) {t_h}x{t_w}, " f"frame_id={frame_id}", flush=True)
        if frame_id == 0:
            failures.append("tiled frame_id is 0 -- the node never published a " "rendered frame")
        tiled = TiledFrameShm(tiled_name, num_envs=t_envs, height=t_h, width=t_w, create=False)
        img = tiled.rgb.copy()
        for t in range(t_envs):
            tile = img[t]
            lo, hi, mean = int(tile.min()), int(tile.max()), float(tile.mean())
            print(f"[smoke] tile {t}: min={lo} max={hi} mean={mean:.1f}", flush=True)
            if lo == hi:
                failures.append(
                    f"tile {t} is a single constant value ({lo}) -- the render " f"pipeline produced nothing"
                )
        try:
            from PIL import Image

            for t in range(t_envs):
                Image.fromarray(img[t]).save(f"{args.save_dir}/tile_{t}.png")
            print(f"[smoke] wrote {t_envs} PNG(s) to {args.save_dir}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[smoke] PNG dump skipped: {exc!r}", flush=True)

        # 6. shut the node down over the protocol (not SIGKILL) so its own
        #    teardown path gets exercised too.
        state.set_command(CMD_SHUTDOWN)
        sems.post_request()
        try:
            sems.wait_response(timeout=30.0)
        except Exception:  # noqa: BLE001
            pass
    finally:
        for obj, kw in ((state, {}), (tiled, {})):
            if obj is not None:
                try:
                    obj.close(**kw)
                except Exception:  # noqa: BLE001
                    pass
        try:
            proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            proc.kill()
        try:
            sems.close(unlink=True)
        except Exception:  # noqa: BLE001
            pass

    if failures:
        print("\n[smoke] ===== FAILED =====", flush=True)
        for f in failures:
            print(f"  * {f}", flush=True)
        return 1
    print("\n[smoke] ===== ALL CHECKS PASSED =====", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
