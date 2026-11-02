#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Capture completed particle frames for one declarative deformable object.

Requires a built ROS overlay, an assembled manifest and the native runtime.
All engine settings come from the selected launcher YAML. The default is the
installed launcher_libuipc.yaml; select a custom YAML to change pacing or rates.
Output must be a new directory. NPZ frames and frames.json describe one run.
This is an offline example using package-private scene hooks, not a recorder or
another render backend. See examples/README.md for invocation and limitations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="assembled scene manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="new directory for this capture")
    parser.add_argument("--ticks", type=int, default=240, help="positive number of completed physics ticks")
    parser.add_argument("--launcher", type=Path, help="launcher YAML; defaults to installed launcher_libuipc.yaml")
    args = parser.parse_args(argv)
    if args.ticks <= 0:
        parser.error("--ticks must be positive")
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    if args.out.exists() or args.out.is_symlink():
        parser.error("--out must be a new directory; existing captures are never overwritten")
    launcher = args.launcher
    if launcher is None:
        from ament_index_python.packages import get_package_share_directory

        launcher = Path(get_package_share_directory("genie_sim_bringup")) / "config/launcher_libuipc.yaml"
    config = yaml.safe_load(launcher.read_text())
    params = dict(config["genie_sim_engine"]["ros__parameters"])
    if params.get("physics_engine") != "libuipc" or params.get("physics_solver") != "ipc":
        parser.error("--launcher must select physics_engine: libuipc and physics_solver: ipc")
    params["stage_manifest"] = str(args.manifest.resolve())

    from genie_sim_engine_libuipc.session import LibuipcSession, SimpleLogger

    logger = SimpleLogger("dump_fold_cloth")
    session = None
    scene = None
    original_post_step = None
    frame_times = []
    try:
        session = LibuipcSession(
            node_name="dump_fold_cloth", params=params, simulation_app=None, physics_engine="libuipc", logger=logger
        )
        session.startup(headless=True)
        sim = session.sim
        scene = sim._scene
        objects = [obj for obj in scene.objects if obj["kind"] in ("cloth_mesh", "tet_mesh")]
        if len(objects) != 1:
            raise ValueError("this example requires exactly one cloth_mesh or tet_mesh object")
        obj = objects[0]
        start, count = scene._ranges[obj["mesh_prim_path"]]
        expected = obj.get("expected_particle_count")
        if expected is not None and int(expected) != count:
            raise ValueError(f"particle range has {count} points; scene declares {expected}")
        # mkdir is exclusive even if another capture races the preflight check.
        args.out.mkdir(parents=True)
        frames = args.out / "deformable" / obj["name"]
        frames.mkdir(parents=True)
        original_post_step = scene.on_post_step

        def capture(state, sim_time, dt, ctx):
            original_post_step(state, sim_time, dt, ctx)
            points = np.asarray(state.particle_q.numpy()[start : start + count], dtype=np.float32)
            if points.shape != (count, 3) or not np.isfinite(points).all():
                raise RuntimeError(f"invalid particle state at capture tick {len(frame_times)}")
            np.savez(frames / f"{len(frame_times):06d}.npz", points=points)
            frame_times.append(sim_time)

        scene.on_post_step = capture
        session.run(render_hook=None, exit_check=lambda: len(frame_times) < args.ticks)
        if len(frame_times) != args.ticks:
            raise RuntimeError(f"capture stopped after {len(frame_times)} of {args.ticks} ticks")
        (args.out / "frames.json").write_text(
            json.dumps(
                {
                    "object_name": obj["name"],
                    "cloth_prim_path": obj["mesh_prim_path"],
                    "frame_pattern": f"deformable/{obj['name']}/%06d.npz",
                    "n_particles": count,
                    "physics_hz": session.physics_hz,
                    "n_frames": len(frame_times),
                    "sim_time": frame_times,
                },
                indent=2,
            )
            + "\n"
        )
        logger.info(f"wrote {len(frame_times)} completed frames to {args.out}")
        return 0
    finally:
        if scene is not None and original_post_step is not None:
            scene.on_post_step = original_post_step
        if session is not None:
            session.sim.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
