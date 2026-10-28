---
name: debug-physics
description: >
  Diagnose physics misbehaviour in the Genie Sim RT Engine — robot
  swings on spawn, contacts tunnel, joints drift past their limits,
  cloth blows up, the convex-hull proxy renders instead of the
  visual mesh, or the wrong physics backend is active. Walks the
  user through the engine's debug toggles (visualizers, marker
  array, GL viewer, `init_*` teleport, backend swap) and the common
  failure-mode fixes.
  Trigger: When the user reports "robot swings at start", "objects
  float / sink into the floor", "contact tunnelling", "joint went
  past limit", "robot vibrates / explodes", "shelf looks like a
  convex hull", "wrong gripper poses", "newton vs physx vs mjwarp
  difference", or asks to "debug the physics".
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:
  - geniesim_ros:launch-scene
inputs:
  - name: symptom
    desc: Free-text description of the misbehaviour observed
    required: true
  - name: launcher_config
    desc: Current launcher (used for backend bisection)
    required: false
outputs:
  - desc: Identified failure mode (init, contact, render proxy, backend-specific) + targeted fix or workaround
---

## When to Use

- Visual / kinematic glitch on launch (swing, drift, float, sink).
- Contact behaviour wrong (tunnelling, no contact, no friction).
- Solver-specific symptoms — works on PhysX but not Newton, or vice
  versa.
- Render proxy showing instead of visual mesh (convex-hull look).
- Joint exceeds URDF `<limit>` and MoveIt rejects the start state.

Do **not** use for:
- Launch errors before physics even runs → check
  `build-workspace` skill output first.
- MoveIt planning failures with sane physics → that's `moveit-wbc`,
  in particular the `fix_start_state` + `start_state_max_bounds_error`
  knobs.
- Performance tuning (low FPS, JIT compile time) — different topic.

## Critical Patterns

1. **`init_base_pose` / `init_joint_pos` are non-physics teleport.**
   The engine writes them to USD attributes *before* `world.reset()`
   on PhysX and to `model.joint_q` + `control.joint_target_pos`
   *before* the first solver tick on Newton — drive error is zero at
   t=0, so no startup swing. If you see a swing, your launcher /
   scene yaml is bypassing this path.
2. **`init_joint_pos` is JSON-encoded, not a launch list.**
   `launch_ros` mangles list-typed params; the engine reads
   `init_joint_pos_json` and decodes. Always edit it in scene yaml,
   not on the CLI.
3. **The convex-hull look means the visual mesh was stripped.** The
   engine blocks the `physics:approximation` token when
   `collisionEnabled=False`, so the GL viewer renders the visual
   mesh. If you still see the proxy, the USD set
   `collisionEnabled=True` with a convexDecomposition approximation
   *and* the visual mesh is missing.
4. **Backend swap is one-arg.** Whenever you suspect engine bugs,
   re-launch with the other launcher: `launcher_ovrtx_isaac_physx`
   (stable) vs `launcher_newton_mjwarp` (rigid, mujoco-warp). If
   only one backend reproduces, the issue is solver-side, not
   asset-side.
5. **Use the GL viewer for ground truth.** Newton-standalone +
   inline OVRtx can mask geometry issues with photoreal lighting.
   The Newton GL viewer renders raw visual + collision meshes, so
   it's the right tool for "is the asset actually there".

## Decision tree

### Symptom: robot swings violently on spawn

| Check | What to do |
|---|---|
| Was `init_joint_pos` set? | Add to scene yaml under the robot block. |
| Did the swing happen on PhysX? | Engine seeds `state:angular/linear:physics:position` *before* `world.reset()`. Inspect the seed attributes, selected gains, and reset ordering before the first step. |
| Did the swing happen on Newton? | Engine writes `model.joint_q` + `control.joint_target_pos` + syncs `state_0.joint_q` + calls `eval_fk` in `_phase_finalize_init_state` *before* `_warmup`. Confirm with `cat source/geniesim_ros/src/ros_ws/src/genie_sim_engine_newton/genie_sim_engine_newton/engine/newton/setup/init_pose.py` |
| Mimic followers still swing? | Mimic states are trivial — they ride the constraint. If they swing, the **master** joint init pose is wrong, not the follower. |

### Symptom: contact tunnelling / objects sink through floor

| Check | What to do |
|---|---|
| Backend? | Try the other launcher — Isaac PhysX vs MuJoCo-Warp have very different defaults. |
| Was `collisionEnabled=False` set on the floor / object? | The engine strips the `physics:approximation` token only; if the whole `CollisionAPI` is disabled it'll fall through. Re-enable in the USD. |
| Soft joint limit drift? | MoveIt URDF limits are padded ±0.01 rad / ±0.001 m at launch (see `wbc.launch.py`). For the engine itself, raise the URDF `<limit>` if real contact load is pushing past it. |
| MuJoCo soft contact? | mjwarp uses a softer contact model than PhysX; inspect `newton.mjc_contact` in scene YAML and the substep stability warning. |

### Symptom: shelf / object renders as a convex hull

| Check | What to do |
|---|---|
| GL viewer or OVRtx? | OVRtx renders visual mesh; GL viewer renders both visual + collision. If only GL shows the hull, that's expected — toggle off the collision layer. |
| `collisionEnabled` value on the prim | If `False`, the engine should not be authoring a collision shape at all. If a hull renders, the USD still has `physics:approximation=convexDecomposition`. The engine blocks that token at load — rebuild via `build-workspace` if not. |
| Recently merged scene? | Re-run the `isaac_scene` assemble stage (`rm -rf assets/scenes/<scene>/` + relaunch). |

### Symptom: gripper poses wrong / EEF off

Common one for omnipicker / swiftpicker — verify:

- Scene yaml `gripper:` matches the URDF you built MoveIt for.
- MoveIt's `(arm, gripper)` matches the engine's. See `moveit-wbc`.
- `EEF_ABS` payloads use `arm_base_link` framing.

## Toggles & visualizers

```bash
# Newton GL viewer (rigid only, kit-free) — see raw geometry:
ros2 launch genie_sim_bringup app.launch.py \
  scene:=<SCENE> \
  launcher_config:=launcher_newton_mjwarp \
  headless:=false   # local workstation with a screen; flip to true on a remote/headless host

# Debug marker / pointcloud publishers (enabled in the launcher yaml):
ros2 topic list | grep -E "marker|pointcloud|debug"
ros2 topic echo /debug/contacts                      # contact normals + impulses
ros2 topic echo /debug/init_pose                     # confirms init_* applied

# Compare runtime USD against source:
cat assets/scenes/<scene>/robot_runtime.usda | head  # post-strip / post-overrides
```

Per-scene viewer-camera pose is configurable in the scene yaml:

```yaml
viewer_camera:
  pos:    [1.6, -1.6, 1.2]
  lookat: [0.0,  0.0, 0.8]
```

## Backend swap as a bisection tool

When in doubt, run the same scene through both stable launchers and
diff the behaviour:

```bash
ros2 launch genie_sim_bringup app.launch.py scene:=<S> launcher_config:=launcher_ovrtx_isaac_physx
ros2 launch genie_sim_bringup app.launch.py scene:=<S> launcher_config:=launcher_newton_mjwarp
```

- Same misbehaviour on both → **asset / scene yaml** bug (URDF
  inertia, init_pose, collision flags).
- Different misbehaviour → **solver-tuning** bug (gains, contact
  compliance, integrator step).

## Commands (copy-paste summary for the user)

```bash
# 1. Force scene regenerate to rule out a stale cache
rm -rf assets/scenes/<scene>/
# OR:
ros2 launch genie_sim_bringup app.launch.py scene:=<S> launcher_config:=launcher_ovrtx_isaac_physx always_regenerate_robot_usd:=true

# 2. Try the other backend (bisection) — flip headless to true on a remote/headless host
ros2 launch genie_sim_bringup app.launch.py scene:=<S> launcher_config:=launcher_newton_mjwarp headless:=false

# 3. Inspect debug topics
ros2 topic list | grep -E "marker|debug|contacts"
ros2 topic echo /debug/contacts

# 4. Inspect the runtime USD
cat assets/scenes/<scene>/robot_runtime.usda
```

## Notes

- All `launcher_newton_*` rows except `launcher_newton_mjwarp` are
  experimental — physics, perf, and yaml schema can break between
  commits. Use them when you specifically need cloth / soft body or
  are intentionally testing a solver, not as a general fallback.
- The `_apply_init_base_pose` helper writes to the session layer
  with a unique suffix, so re-launching doesn't accumulate
  xformOpOrder ops. Safe to relaunch repeatedly.
- The OVRtx first-frame banner is one-shot, not a heartbeat — shader
  compile holds the GIL inside a C extension, so there's nothing to
  tick. If the banner sits without progress for >2 min, the compile
  cache is missing (rebuild the docker image's shader cache mount).

## Resources

- **Engine packages**: [genie_sim_engine_core](../../src/ros_ws/src/genie_sim_engine_core/) (ABC, ROS bridge, assemble), [genie_sim_engine_newton](../../src/ros_ws/src/genie_sim_engine_newton/), [genie_sim_engine_isaacsim](../../src/ros_ws/src/genie_sim_engine_isaacsim/)
- **Init pose pipeline**: `genie_sim_engine_newton/engine/newton/setup/init_pose.py`, `genie_sim_engine_isaacsim/kit/stage.py`
- **Scene yamls**: [source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/config/](../../src/ros_ws/src/genie_sim_bringup/config/)
- **Engine overview**: [source/geniesim_ros/README.md](../../README.md)
- **Related skills**: `launch-scene`, `moveit-wbc`, `add-robot`
