# geniesim_ros — repository-level dispatcher

> 🧭 **Canonical source**: [`source/geniesim_ros/AGENTS.md`](../source/geniesim_ros/AGENTS.md) — the per-package guide is the source of truth for the ROS 2 package table, the wheel layout / `_ros_install.tar.gz` contract, the AS2 ↔ AS3 dispatch, and the manifest schema.

This file is a 30-second pointer. Do not duplicate content here that lives at the canonical source — duplication is what makes dispatchers rot.

---

## What it is

**Genie Sim RT Engine** — realtime, interactive ROS 2 simulation. `geniesim_ros` ships a 10-package colcon workspace plus a pip-installable wheel that bundles the pre-built install tree. Teleop, MoveIt, ros2_control, RViz, and physics all share one `sim_time`.

Three physics backends coexist as switchable launcher choices:
- `launcher_ovrtx_isaac_physx` — Isaac Sim PhysX (stable, default)
- `launcher_ovrtx_isaac_newton` — Isaac Sim Newton wrapper (experimental, rigid only)
- `launcher_newton_*` — Kit-free Newton-standalone (mjwarp / featherstone+VBD / …) — cloth + soft-body experimental path

## Where to look

| Topic | File |
|---|---|
| Canonical package table + invariants | [`source/geniesim_ros/AGENTS.md`](../source/geniesim_ros/AGENTS.md) |
| User-facing engine overview + scene × launcher matrix | [`source/geniesim_ros/README.md`](../source/geniesim_ros/README.md) |
| Agent skills (build, launch, MoveIt, teleop bridge, record, debug, material) | [`source/geniesim_ros/skills/`](../source/geniesim_ros/skills/) |
| Engine internals (deep-dives) | [`source/geniesim_ros/src/ros_ws/src/genie_sim_engine_core/docs/`](../source/geniesim_ros/src/ros_ws/src/genie_sim_engine_core/docs/) |
| Launch + scene/launcher yamls | [`source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/`](../source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/) |

## Invariants the rest of the repo relies on

- Scene yamls (`scene_*.yaml`) and launcher yamls (`launcher_*.yaml`) are **orthogonal axes**. Scene = robot + task; launcher = physics backend + renderer. Any scene × any launcher.
- The canonical engine ids are `isaac_physx`, `isaac_newton`, `newton_standalone`. Bare `physx` / `newton` are **rejected** by `runtime.bootstrap._validate_engine_id`.
- `assemble_robot.py` is **read-only** after first write of `robot.usda`; `assemble_scene.py` never modifies it. Cache-gated by `manifest.json`.
- `init_base_pose` / `init_joint_pos` are non-physics teleport on every backend — apply *before* the first solver tick, so the first PD step sees zero error.
- Every ROS package ships **both** `README.md` and `AGENTS.md` — enforced by `geniesim tool docs --scope ros`.
- The pip wheel ships one payload — `geniesim_ros/_ros_install.tar.gz` — with deterministic mode-preserving tar to keep `lib/<pkg>/<entrypoint>.py` executable. Do **not** revert to raw `package_data`.
