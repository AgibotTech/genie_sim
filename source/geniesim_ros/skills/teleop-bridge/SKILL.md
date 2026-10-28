---
name: teleop-bridge
description: >
  Wire a `geniesim_teleop` VR/Pico session into a running Genie Sim RT
  Engine scene — pick the right scene yaml, launch `wbc.launch.py`
  with `use_ros2_control:=false` so move_group serves `/compute_ik`
  while the teleop publisher owns `/joint_command`, and confirm no
  topic fighting.
  Trigger: When the user asks to "connect teleop to the sim", "drive the engine with VR", "use teleop instead of
  ros2_control", "stop the controllers from fighting teleop", or has
  a teleop loop ready and a scene up but the robot jitters / flicks.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:
  - geniesim_ros:launch-scene
  - geniesim_teleop:run-teleop
inputs:
  - name: scene_gripper
    desc: "Gripper variant of the running scene; the MoveIt `gripper:=` must match (e.g. `omnipicker` for `scene_pnp_g2_op`)"
    required: true
outputs:
  - desc: Exactly one publisher on `/joint_command` (the teleop node); no topic fighting from the controller manager
---

## When to Use

- VR teleop loop (`geniesim_teleop` skill `run-teleop`) and an RT
  Engine scene (`launch-scene` skill) need to share `/joint_command`
  without the ros2_control hardware interface rebroadcasting it at
  the controller-manager update rate.
- User sees "the arm jitters / payloads get flung" — that's the
  classic two-publisher symptom.
- External motion script (e.g. `wok_flip_cmds.py`) and MoveIt are
  both interested in the same topic.

Do **not** use for:
- Starting the teleop loop itself → `run-teleop` skill in
  `geniesim_teleop`.
- MoveIt planning with the simulator's controllers driving →
  `moveit-wbc` skill (default mode).
- Recording the teleop session → `record-episode` skill.

## Critical Patterns

1. **`use_ros2_control:=false` is the bridge switch.** With it,
   `wbc.launch.py` starts `move_group` only (`/compute_ik`,
   `/compute_fk`) and skips the ros2_control node + controller
   spawners. `/joint_command` then has exactly one source: teleop.
2. **Match scene gripper to teleop config.** `run-teleop` defaults
   to `G2_omnipicker.json`. Use `scene_pnp_g2_op` (or any
   `scene_*_g2_op`) so the engine is built with the same gripper
   the teleop loop expects. Mismatched grippers produce a robot
   MoveIt can plan against but the engine refuses to drive.
3. **Three processes, three shells.** Scene + MoveIt + teleop each
   run as a long-lived ROS node. Don't try to background them in one
   shell — the engine and teleop both want stdin / SIGINT semantics.
4. **`use_sim_time:=true` is engine-side**, but the teleop loop
   publishes wall-clock by default. The bridge uses sim time inside
   the scene; the publisher rate adapts. Don't override unless you
   know what you're doing.

## Workflow

### Step 1 — Confirm the prerequisites

- Scene is up via `launch-scene` skill (e.g.
  `scene_pnp_g2_op` × `launcher_ovrtx_isaac_physx`).
- Teleop config matches the scene's gripper.
- VR device reachable, default port `8080`.

### Step 2 — Launch MoveIt without controllers

In a fresh shell:

```bash
source devel/setup.bash

ros2 launch genie_sim_moveit wbc.launch.py \
  arm:=crsB \
  gripper:=omnipicker \
  use_ros2_control:=false
```

Console banner should print:

```
[wbc.launch.py] use_ros2_control=false → move_group only (no ros2_control
node / controllers); drive the arm via /joint_command.
```

### Step 3 — Start teleop

In another shell (`geniesim docker into`):

```bash
geniesim teleop run \
  --device_type=pico \
  --port=8080 \
  --robot_config=G2_omnipicker.json
```

### Step 4 — Verify single-publisher on `/joint_command`

```bash
ros2 topic info /joint_command -v
# Publishers should list exactly one: the teleop node.
```

If you see two publishers, you forgot `use_ros2_control:=false` —
stop MoveIt and relaunch with the flag.

### Step 5 — Plan-then-execute combo (optional)

Even without ros2_control, `/compute_ik` is still available. So you
can:
- Use MoveIt's RViz markers to plan and call `/compute_ik` for
  manual goals.
- Hand the resulting joint target to your teleop publisher (or skip
  MoveIt entirely for direct VR drive).

## Commands (copy-paste summary for the user)

```bash
# Shell 1 — scene
source devel/setup.bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_pnp_g2_op \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false   # local workstation with a screen; flip to true on a remote/headless host

# Shell 2 — MoveIt, no controllers
source devel/setup.bash
ros2 launch genie_sim_moveit wbc.launch.py \
  arm:=crsB gripper:=omnipicker use_ros2_control:=false

# Shell 3 — teleop loop
geniesim teleop run --device_type=pico --port=8080 \
  --robot_config=G2_omnipicker.json

# Verify single publisher
ros2 topic info /joint_command -v
```

## Notes

- `simple_body_controller` is configured but not auto-spawned even in
  the controller mode — it conflicts with
  `simple_waist_controller` + `simple_torso_controller` on the same
  body-joint resources. The teleop bridge avoids that ambiguity by
  not spawning any of them.
- For wok-flip / scripted motion (`wok_flip_cmds.py`,
  benchmark replay scripts), the same `use_ros2_control:=false`
  flag applies — anything that owns `/joint_command` directly needs
  the controllers out of the way.
- The teleop loop has its own per-episode recording hooks
  (`--record-dir`). For raw-topic recording on top, see the
  `record-episode` skill.

## Resources

- **Teleop CLI + skill**: [source/geniesim_teleop/skills/run-teleop/SKILL.md](../../../geniesim_teleop/skills/run-teleop/SKILL.md)
- **MoveIt launch with the bridge switch**: [source/geniesim_ros/src/ros_ws/src/genie_sim_moveit/launch/wbc.launch.py](../../src/ros_ws/src/genie_sim_moveit/launch/wbc.launch.py)
- **Related skills**: `launch-scene`, `moveit-wbc`, `record-episode`
