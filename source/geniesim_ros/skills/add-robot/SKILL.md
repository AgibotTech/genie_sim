---
name: add-robot
description: >
  Bring a custom robot into the Genie Sim RT Engine — author / fix a
  xacro / URDF in `genie_sim_robot_model`, prep meshes with the
  offline tools (`normalize_obj_names.py`, `diagnose_urdf.py`,
  `recompute_inertia.py`, `fix_dae_units.py`, `copy_dae_material.py`),
  stage assets into the AS3 layout (`robot.usda` +
  `payloads/Physics/{physics,physx,mujoco}.usda`) that the engine
  consumes, and wire the result into a `scene_*.yaml`.
  Trigger: When the user asks to "add a new robot", "import a robot",
  "support <vendor> in geniesim", names a URDF / xacro not currently
  in `genie_sim_robot_model/robots/`, or wants to convert a third-party
  mesh pack into AS3-ready USD.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:
  - geniesim_ros:build-workspace
inputs:
  - name: urdf_path
    desc: Path to the URDF / xacro file to integrate
    required: true
  - name: robot_name
    desc: "Robot name in `<vendor>_<model>[_<gripper>]` form"
    required: true
  - name: mesh_dir
    desc: Directory containing the robot's meshes (for the mesh-prep tools)
    required: false
outputs:
  - desc: "URDF passes `diagnose_urdf.py`; new `scene_flat_<robot>.yaml`; the engine stages AS3 USD layout on first launch"
---

## When to Use

- User wants to run an off-tree robot on the RT Engine.
- User has a URDF / xacro that fails to import (missing inertia,
  unit mismatch, DAE materials lost, OBJ names colliding) and needs
  the offline mesh-prep tools.
- User wants to add the new robot to a `scene_*.yaml` and launch it
  via the `launch-scene` skill.

Do **not** use for:
- Tweaking an existing Genie G2 variant → just edit the relevant
  URDF / xacro and rebuild via `build-workspace`.
- Authoring a MoveIt config for the new robot → that's a separate,
  larger job (port SRDF + `coupled_constraints.yaml`).
- Adding a benchmark task that uses the new robot → `run-benchmark`
  in `geniesim_benchmark/skills/`.

## Critical Patterns

1. **Tier 1 only validates Genie G2.** Reference robots (Franka,
   UR5, Aloha, ARX, Agilex) have correct URDFs and import cleanly,
   but scene yamls and physics tuning may be stale. Treat them as
   starting points, not certified configs.
2. **AS3 layout is mandatory.** The engine expects the per-robot
   directory `robot.usda` + `payloads/Physics/{physics,physx,mujoco}.usda`
   regardless of which backend is active — switching physics never
   rewrites the asset.
3. **Fix the URDF before importing.** `diagnose_urdf.py` catches
   missing inertia, zero-mass links, collision/visual mismatches,
   and bad OBJ name collisions — every one of these breaks the
   URDF→USD importer downstream.
4. **Mesh names must be unique.** USD's flat namespace will silently
   merge two meshes that share an OBJ name. `normalize_obj_names.py`
   rewrites names to be prefix-unique.
5. **Mimic joints handled at runtime.** Don't model gripper mimics
   inside the URDF — the engine bridges them via the controller
   layer (see `86308f040` in git history). URDF stays plain.
6. **Robots live in `genie_sim_robot_model`, not in `geniesim_assets`.**
   Meshes can move to `geniesim_assets` once stabilised, but URDFs +
   xacros stay in the ROS package.

## Workflow

### Step 1 — Stage the URDF / xacro

Drop the file under
`source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/<vendor>/<model>/`
next to the reference robots. Name follows the convention
`<model>[_<gripper>].urdf.xacro`, with meshes in a sibling `meshes/`.

```bash
ls source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/
# agilex/ arx/ franka/ genie/ universal_robots/
ls source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/franka/fr3/
# fr3.urdf.xacro, meshes/
```

### Step 2 — Diagnose

```bash
cd source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model
python scripts/diagnose_urdf.py robots/<vendor>/<model>/<your_robot>.urdf
```

Look for: missing `<inertial>`, zero mass, visual/collision link
mismatches, mesh path resolution failures.

### Step 3 — Prep meshes

For each issue the diagnose tool flagged:

```bash
# OBJ name collisions:
python scripts/normalize_obj_names.py path/to/meshes/

# DAE units off (mm vs m):
python scripts/fix_dae_units.py path/to/meshes/<file>.dae

# DAE materials missing after a re-export:
python scripts/copy_dae_material.py source.dae target.dae

# Missing or wrong inertials (uniform-density approximation):
python scripts/recompute_inertia.py urdf/<your_robot>.urdf
```

### Step 4 — Confirm the URDF imports

Re-run `diagnose_urdf.py` until it's clean, then build the workspace
(`build-workspace` skill) and let the engine's URDF→USD importer
stage the AS3 layout on first run. The importer writes:

```
assets/scenes/<scene>/
├── manifest.json
├── scene.usda
└── robot/
    ├── robot.usda
    └── payloads/
        └── Physics/
            ├── physics.usda
            ├── physx.usda
            └── mujoco.usda
```

`manifest.json` presence = cache hit. To force a regenerate:

```bash
rm -rf assets/scenes/<scene>/
# or pass:
ros2 launch genie_sim_bringup app.launch.py … always_regenerate_robot_usd:=true
```

### Step 5 — Wire into a scene yaml

Create `genie_sim_bringup/config/scene_flat_<your_robot>.yaml`
modelled on one of the reference scenes (e.g. `scene_flat_fr3.yaml`).
Required keys:

```yaml
robot:
  robot_name: <your_robot>_manifold
  robot_prefix: <your_robot>
  robot_source:                      # composed by xacro/robot.xacro providers
    arm: <model>                     # e.g. fr3 — selects the provider macro
    body: T2                         # case-sensitive; "" drops the body
    gripper: ""
    robot_model: <vendor>            # e.g. franka — picks xacro/providers/<vendor>.xacro
    package: genie_sim_robot_model
    pin_base_to_world: true
    urdf: {}                         # per-robot xacro args, {} for none
  init_base_pose:                    # non-physics teleport at spawn
    x: 0.0
    y: 0.0
    z: 0.0
    theta: 0.0
  init_joint_pos:                    # optional, per-joint dict
    joint_name: <value>

viewer_camera:                       # Newton GL viewer default cam
  pos: [1.6, -1.6, 1.2]
  lookat: [0.0, 0.0, 0.8]

scene:
  base_path: <relative_to_geniesim_assets>
  scene_usda: <stage>.usda           # or omit for an empty scene
```

### Step 6 — Launch via `launch-scene`

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_<your_robot> \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false   # local workstation with a screen; flip to true on a remote/headless host
```

If anything goes wrong, fall back to the experimental
`launcher_newton_mjwarp` to see whether the issue is PhysX-specific
or applies to all backends.

## Commands (copy-paste summary for the user)

```bash
# Inside the container:
cd source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model

# 1. Diagnose
python scripts/diagnose_urdf.py robots/<vendor>/<model>/<your_robot>.urdf

# 2. Mesh prep (only the ones the diagnose tool flagged)
python scripts/normalize_obj_names.py path/to/meshes/
python scripts/fix_dae_units.py path/to/meshes/<file>.dae
python scripts/copy_dae_material.py src.dae dst.dae
python scripts/recompute_inertia.py urdf/<your_robot>.urdf

# 3. Build, then let the engine stage AS3 on first run
cd /workspace
geniesim ros build dev && source devel/setup.bash

# 4. Author scene_flat_<your_robot>.yaml, then:
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_<your_robot> \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false   # local workstation with a screen; flip to true on a remote/headless host
```

## Notes

- The AS3 layout is the same regardless of backend, so a robot that
  works under Isaac PhysX also works under Newton-standalone — only
  the physics tuning (gains, contact compliance) may need
  per-backend trimming.
- `recompute_inertia.py` assumes uniform density and is only a
  starting point. For Tier 1 quality, source inertials from the
  vendor's CAD.
- The URDF→USD importer is cache-gated by `manifest.json`. Editing
  `init_joint_pos` does **not** bust the cache (it's forwarded as a
  JSON-encoded launch param, not a manifest field) — only changes
  that affect topology (URDF, xacro args, mimic, fixed-base
  selection) require a regenerate.
- For non-G2 robots, MoveIt is **not** packaged. The engine runs
  fine without it; you just don't get planning or RViz interactive
  markers out of the box.

## Resources

- **Robot package**: [source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/](../../src/ros_ws/src/genie_sim_robot_model/)
- **Mesh-prep scripts**: [source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/scripts/](../../src/ros_ws/src/genie_sim_robot_model/scripts/)
- **Reference URDFs**: [source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/](../../src/ros_ws/src/genie_sim_robot_model/robots/)
- **Scene yamls**: [source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/config/](../../src/ros_ws/src/genie_sim_bringup/config/)
- **Engine overview**: [source/geniesim_ros/README.md](../../README.md)
- **Package routing**: [source/geniesim_ros/AGENTS.md](../../AGENTS.md)
