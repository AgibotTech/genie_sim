# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Initialize a task on a running server, then detach without stopping it.

The server's task metadata supplies the official background/robot pairing when
``--task-name`` is omitted. ``--config`` accepts additional task-init sections,
including physics, render, and sensor randomization. Task and backend selection
remain CLI arguments; the server owns transport configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def resolve_task_name(client, sub_task_name: str) -> str:
    """Select the official suite pairing, preferring the instruction-following suite."""
    from rlinf_geniesim.task_pairing import official_task_name  # noqa: PLC0415

    name = official_task_name(client.task_metadata(sub_task_name), sub_task_name)
    if name is None:
        raise ValueError(f"No official task pairing for {sub_task_name!r}; specify --task-name")
    return name


def backends_from_args(args) -> dict:
    """Declare the engine section locked by the first successful server init.

    An omitted render or substep count follows the physics backend: fused
    Isaac PhysX renders itself with 4 substeps; other backends use Isaac tiled
    rendering with 10."""
    fused = args.physics == "isaac_fused"
    render = args.render if args.render is not None else ("isaac_fused" if fused else "isaac_tiled")
    substep = args.substep if args.substep is not None else (4 if fused else 10)
    if args.physics == "isaac_fused" and render != "isaac_fused":
        raise ValueError("--physics=isaac_fused requires --render=isaac_fused")
    if render == "isaac_fused" and args.physics != "isaac_fused":
        raise ValueError("--render=isaac_fused requires --physics=isaac_fused")
    # Runtime solvers belong to backend classes; this field also participates
    # in the server lock. Keep the default declaration aligned with the client.
    backend = {
        "physics": args.physics,
        "physics_solver": {
            "isaac_fused": "physx",
            "isaac_physx": "physx",
            "newton_avbd": "avbd",
            "newton_featherstone": "featherstone",
        }.get(args.physics, "mujoco-warp"),
        "physics_solver_substep": substep,
        "render": render,
        "device": args.device,
    }
    if getattr(args, "njmax", 0):
        backend["mujoco_njmax"] = int(args.njmax)
    return backend


def build_parser() -> argparse.ArgumentParser:
    """Build the task initialization CLI without importing simulator dependencies."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--addr", default="tcp://127.0.0.1:5601", help="server ZMQ address")
    parser.add_argument("--sub", required=True, help="benchmark sub_task_name")
    parser.add_argument("--task-name", default="", help="eval task; defaults to the official suite pairing")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--render", default=None, help="render backend; default follows --physics")
    parser.add_argument("--physics", default="isaac_fused")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--substep", type=int, default=None, help="physics solver substeps; default 4 for isaac_fused, otherwise 10"
    )
    parser.add_argument("--spacing", type=float, default=None, help="distance between environment origins in meters")
    parser.add_argument("--background-default", action="store_true", help="include a background for unlisted tasks")
    parser.add_argument("--instances", default="", help="comma-separated initial instances")
    parser.add_argument("--superset", default=None, help="switchable instances; an empty string disables switching")
    parser.add_argument("--seed", type=int, default=None, help="build-time randomization seed; -1 requests replay")
    parser.add_argument("--config", type=Path, help="YAML/JSON file of additional task-init sections")
    parser.add_argument("--njmax", type=int, default=0, help="MuJoCo constraint capacity; 0 uses the engine default")
    parser.add_argument("--num-bodies", type=int, default=None, help="explicit body-pose buffer capacity")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    return parser


def config_from_args(args) -> dict:
    """Read extra init sections and apply explicitly supplied CLI overrides."""
    config = {}
    if args.config is not None:
        import yaml

        try:
            loaded = yaml.safe_load(args.config.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid --config YAML: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("--config must contain a mapping of task-init sections")
        forbidden = set(loaded) & {"task", "backends", "comm"}
        if forbidden:
            raise ValueError(f"--config must not contain {sorted(forbidden)}; use task/backend CLI arguments")
        config = loaded
    overrides = (
        ("multi_env", "env_spacing", args.spacing),
        ("manifest", "use_background_default", True if args.background_default else None),
        ("scene", "num_bodies_estimate", args.num_bodies),
    )
    for section, key, value in overrides:
        if value is not None:
            target = config.setdefault(section, {})
            if not isinstance(target, dict):
                raise ValueError(f"--config section {section!r} must be a mapping")
            target[key] = value
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.envs < 1 or (args.substep is not None and args.substep < 1) or args.timeout_s <= 0:
        parser.error("--envs, --substep, and --timeout-s must be positive")
    if args.njmax < 0 or (args.num_bodies is not None and args.num_bodies < 1):
        parser.error("--njmax must be nonnegative and --num-bodies must be positive")
    try:
        backends = backends_from_args(args)
        config = config_from_args(args)
        task = {"sub_task_name": args.sub, "num_envs": args.envs}
        if args.instances:
            task["instances"] = [int(value) for value in args.instances.split(",")]
        if args.superset is not None:
            task["superset_instances"] = [int(value) for value in args.superset.split(",")] if args.superset else []
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    from rlinf_geniesim.comm import make_client
    from rlinf_geniesim.envs.vec_env import GenieSimVecEnv

    client = make_client("zmq", address=args.addr, timeout_s=args.timeout_s)
    try:
        task["task_name"] = args.task_name or resolve_task_name(client, args.sub)
        client.set_backends(backends)
        env = GenieSimVecEnv(client, to_torch=False)
        env.init(task, config=config or None, timeout_s=args.timeout_s)
        print(f"[init-task] {args.sub}: num_envs={env.num_envs} action_dim={env.action_dim}", flush=True)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
