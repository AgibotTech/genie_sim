# geniesim_teleop — repository-level dispatcher

> 🧭 **Canonical source**: [`source/geniesim_teleop/AGENTS.md`](../source/geniesim_teleop/AGENTS.md) — the per-package guide is the source of truth for the device-driver layer, gRPC client, robot configs, and rosbag → HDF5 pipeline.

This file is a 30-second pointer. Do not duplicate content here that lives at the canonical source — duplication is what makes dispatchers rot.

---

## What it is

VR / Pico teleoperation and Wuji glove collection for Isaac Sim. The host launchers coordinate the simulator and controls; the bridge merges motion-controller and input commands into `/joint_command`. The `geniesim teleop` CLI verb (owned by [`geniesim_cli`](geniesim_cli.md)) exposes individual control processes. Follow the package skill for complete collection and post-processing.

## Where to look

| Topic | File |
|---|---|
| Canonical CLI surface + device protocol + robot configs | [`source/geniesim_teleop/AGENTS.md`](../source/geniesim_teleop/AGENTS.md) |
| User-facing intro | [`source/geniesim_teleop/README.md`](../source/geniesim_teleop/README.md) |
| Legacy agent skill (5.1, E.O.L.) | [`source/geniesim_teleop/skills/run-teleop/SKILL.md`](../source/geniesim_teleop/skills/run-teleop/SKILL.md) |
| Bridging into the RT Engine | [`source/geniesim_ros/skills/teleop-bridge/SKILL.md`](../source/geniesim_ros/skills/teleop-bridge/SKILL.md) |

## Workflow boundaries

Use the [package README](../source/geniesim_teleop/README.md) for current Isaac
Sim 6.2 startup, Pico/Wuji selection, networking and recording post-processing.
The retained `run-teleop` skill describes the deprecated (E.O.L.) 5.1 workflow;
it is not the current startup recipe. Runtime and data contracts live in the
canonical package guide above. The separate RT Engine bridge skill describes its own
MoveIt workflow; it is not a prerequisite for Isaac Sim + MC collection.
