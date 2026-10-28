---
name: launch-scene
description: >
  Launch a `genie_sim_bringup` scene against a chosen physics + render
  backend, using `ros2 launch genie_sim_bringup app.launch.py`. Covers
  the scene × launcher matrix (pick-and-place / whole-body-control /
  flat-table demos × Isaac PhysX / Newton-standalone backends) and the
  optional MoveIt 2 + WBC RViz overlay.
  Trigger: When the user asks to "launch a scene", "run pnp",
  "run wbc", "start the simulator", names any `scene_*.yaml` or
  `launcher_*.yaml`, or wants to bring up the RT Engine in interactive
  mode (with or without rviz / moveit).
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:
  - geniesim_ros:build-workspace
inputs:
  - name: scene
    desc: "scene yaml basename (e.g. `scene_pnp_g2_op`)"
    required: true
  - name: launcher_config
    desc: launcher yaml basename
    required: false
    default: launcher_ovrtx_isaac_physx
  - name: headless
    desc: GUI off
    required: false
    default: "false"
outputs:
  - desc: Live ROS 2 graph publishing `/clock`, `/tf`, `/joint_states`, `/joint_command`, camera topics
---

## When to Use

- User wants to bring up the Genie Sim RT Engine interactively —
  physics + render + ROS topics on one shared `sim_time`.
- User references any scene yaml under
  `source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/config/scene_*.yaml`
  or launcher yaml `launcher_*.yaml`.
- User wants MoveIt 2 + WBC RViz on top of a running scene (Genie G2
  only).

Do **not** use for:
- Running a benchmark task → `run-benchmark` (under
  `source/geniesim_benchmark/skills/`).
- Probing an inference server → `check-inference`.
- Building the workspace first → `build-workspace`. Run that *before*
  this skill.

## Critical Patterns

1. **Scene = robot + task; Launcher = physics engine + renderer.**
   They're orthogonal axes — pair any scene with any launcher.
2. **Stable path is `launcher_ovrtx_isaac_physx`.** Every
   `launcher_newton_*` and `launcher_ovrtx_isaac_newton` row is
   research / preview — physics behaviour, perf, and yaml schema can
   break between commits. Use the stable launcher unless the user
   explicitly wants the experimental rig (cloth / soft body / mjwarp).
3. **MoveIt is Genie G2 only.** Other robots run on the engine but
   don't have a packaged MoveIt config.
4. **Inside the container.** Every command below assumes
   `geniesim docker into` and `source devel/setup.bash` (see
   `build-workspace` skill).

## The matrix

### Scenes

| Scene | Robot | What it showcases |
|---|---|---|
| `scene_pnp_g2_op` | Genie G2 + **omnipicker** | Pick-and-place workflow |
| `scene_wbc_g2_sp` | Genie G2 + **swiftpicker** | Whole-body control workflow |
| `scene_flat_g2_sp` | Genie G2 + swiftpicker, flat table | Baseline tabletop |
| `scene_flat_acone` | ARX acone | Reference robot (tabletop) |
| `scene_flat_aloha` | Aloha | Reference robot (tabletop) |
| `scene_flat_fr3` | Franka FR3 | Reference robot (tabletop) |
| `scene_flat_ur5_robotiq_140` / `_85` | UR5 + Robotiq | Reference robot (tabletop) |

### Launchers

| Launcher | Physics engine | Renderer | Status |
|---|---|---|---|
| `launcher_ovrtx_isaac_physx` | Isaac Sim PhysX | Standalone OVRtx node | ✅ **Stable — start here** |
| `launcher_ovrtx_isaac_newton` | Isaac Sim Newton (wrapper) | Standalone OVRtx node | 🧪 EXPERIMENTAL |
| `launcher_newton_mjwarp` | Newton-standalone + mujoco-warp | Inline OVRtx | ✅ Stable (rigid only) |
| `launcher_newton_fsvbd` | Newton-standalone + Featherstone + VBD | Inline OVRtx | 🧪 EXPERIMENTAL — cloth / soft body |
| `launcher_newton_avbd` / `_mjvbd` / `_mjxpbd` | Newton-standalone, mixed solvers | Inline OVRtx | 🧪 EXPERIMENTAL |

## Workflow

### Step 1 — Collect inputs

Ask the user via `AskUserQuestion` if not given:
- **Scene**: free-text basename (e.g. `scene_pnp_g2_op`). Auto-complete
  from the table above.
- **Launcher**: default `launcher_ovrtx_isaac_physx` (stable). Only
  pick a `launcher_newton_*` if the user explicitly wants the
  experimental backend or cloth / soft body.
- **Headless?** Default `false` (GUI window opens on `$DISPLAY`). Pick
  the value by **where you're running**:
  - **Local workstation with a screen** → `headless:=false`. The Isaac
    Sim viewport renders into a window you can see.
  - **Remote / SSH / headless server / CI** → `headless:=true`. No GUI,
    no `$DISPLAY` requirement. Physics + render still run; cameras
    publish over ROS exactly the same. Use this on machines without a
    display, when running over `ssh` without X-forwarding, or in
    container / batch jobs.
  - **Remote machine, want the GUI?** SSH with X-forwarding (`ssh -X`)
    or set `$DISPLAY` to a Xvfb / VNC server, then use
    `headless:=false`. Otherwise the launcher errors on the missing
    display rather than silently failing late.
- **MoveIt + RViz?** Default `false`. Only meaningful for `scene_*_g2_*`.

### Step 2 — Make sure the overlay is sourced

```bash
source devel/setup.bash                   # see `build-workspace` skill if missing
ros2 pkg list | grep genie_sim_bringup    # must list the package
```

### Step 3 — Launch the scene

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=<SCENE> \
  launcher_config:=<LAUNCHER> \
  headless:=<true|false>
```

Recommended starting line for new users (P&P, stable physics, GUI):

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_pnp_g2_op \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false
```

### Step 4 — (optional) MoveIt 2 + WBC RViz

In a second container shell:

```bash
source devel/setup.bash

# default Genie G2: crsB arm + swiftpicker gripper
ros2 launch genie_sim_moveit wbc.launch.py

# different arm/gripper combo (match whatever scene_*_g2_{op,sp} you launched):
ros2 launch genie_sim_moveit wbc.launch.py arm:=crs gripper:=omnipicker
```

`arm:={crs,crsB}` and `gripper:={swiftpicker,omnipicker,none}` are
independent — the launch wires both into the URDF filename and the
SRDF xacro mappings.

### Step 5 — Verify topics

```bash
ros2 topic list                            # /tf, /joint_states, /joint_command, /clock, …
ros2 topic hz /clock                       # should print sim-time pacing
```

## Commands (copy-paste summary for the user)

```bash
# Inside the container, after `geniesim ros build dev`:
source devel/setup.bash

# Pick-and-place + stable Isaac PhysX:
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_pnp_g2_op \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false   # local workstation with a screen; flip to true on a remote/headless host

# Whole-body control + Newton-standalone (experimental):
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_wbc_g2_sp \
  launcher_config:=launcher_newton_mjwarp \
  headless:=false   # local workstation with a screen; flip to true on a remote/headless host

# Optional MoveIt 2 + WBC RViz (second shell):
ros2 launch genie_sim_moveit wbc.launch.py
```

## Notes

- The same `app.launch.py` covers every scene × launcher combo —
  there is no per-scene launch file. Don't try to invoke a yaml
  directly.
- Newton GL viewer pose is configurable per scene: every
  `scene_*.yaml` carries a `viewer_camera:` block with `pos:` /
  `lookat:`.
- For cloth / soft-body work, you need a `launcher_newton_*` that
  pairs with a VBD-family solver and a scene yaml that opts into the
  deformable. Treat any cloth / soft-body result as a preview, not a
  benchmark.
- `init_base_pose` and `init_joint_pos` apply as a non-physics
  teleport on every backend — the robot spawns at the configured
  pose without an initial PD swing.

## Resources

- **Bringup package**: [source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/](../../src/ros_ws/src/genie_sim_bringup/)
- **Scene + launcher yamls**: [source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/config/](../../src/ros_ws/src/genie_sim_bringup/config/)
- **Engine overview**: [source/geniesim_ros/README.md](../../README.md)
- **MoveIt config**: [source/geniesim_ros/src/ros_ws/src/genie_sim_moveit/](../../src/ros_ws/src/genie_sim_moveit/)
- **Package routing**: [source/geniesim_ros/AGENTS.md](../../AGENTS.md)
