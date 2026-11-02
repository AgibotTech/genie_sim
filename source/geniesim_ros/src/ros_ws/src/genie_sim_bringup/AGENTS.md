# genie_sim_bringup

Launch orchestration and configuration for the GenieSim ROS 2 stack.

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/](.)
License: [Mozilla Public License Version 2.0](../../../../LICENSE)

**Maintenance contract**: when you add a launch arg, a new launcher YAML key,
or a new config file, update this file in the same diff.
Keep the [README scene test plan](README.md#scene-testing) in sync when adding,
removing or changing scenes, launcher pairings, prerequisites, or assembly/cache
behavior. This package owns the canonical scene test checklist, grouped by
Tier 1 (G2 Isaac/Newton scenes) and Tier 2 (reference robots). Use scene/backend and pass-criteria matrices with shared
launch templates instead of repeating per-scene commands.
List physics backends and render/viewer choices in separate README columns,
with launcher keys identifying the configured pairings.
Keep the Isaac Newton rigid wrapper out of README supported-backend lists and
the main scene/backend matrix; list its opt-in checks separately as experimental.

`scene_flat_g2_sp`, `scene_flat_g2_op` and `scene_flat_g2_wuji` are self-contained
mobile G2 robot tests with a generated floor/light and a configured head camera.
OP is G2 crs + Omnipicker and retains source/class baseline parameters because
no compatible sysid selection is shipped. SP is G2 crsB + Swiftpicker; Wuji also
uses crsB. Retain their compatible arm and gripper/hand sysid selections for
assembly and runtime. Never apply the crsB arm overlay to crs.
`scene_pnp_g2_op` and both `scene_wbc_g2_*` cases require
the external `scenes/GDK/demo.usda` world and its dependencies under the asset
root. Preserve that prerequisite; a missing world must not silently fall back
to a flat scene or count as demo-task acceptance.

**Support tier**: G2 is the only Tier-1 robot in this stack.  Scene yamls,
launchers, and physics tuning are validated end-to-end on
`scene_flat_g2_*.yaml` + `launcher_newton_*.yaml` against the
`genie/g2/*` description.  Every other robot — `agilex/aloha`,
`agilex/piper`, `arx/x5`, `arx/acone`, `franka/fr3`,
`universal_robots/ur5` — is Tier 2: buildable but not continuously
validated through the bringup, drive-gain, and physics-tuning paths
described below.  See
[`genie_sim_robot_model/AGENTS.md → Support tiers`](../genie_sim_robot_model/AGENTS.md#support-tiers)
for the full policy.

---

## Layout

```
genie_sim_bringup/
├── README.md                      ← overview and canonical scene/assembly tests by support tier
├── launch/
│   ├── app.launch.py              ← top-level entry point (requires launcher_config)
│   ├── physics_backend.launch.py ← physics sub-launch
│   └── utils.py                   ← shared launch helpers (pure I/O, no ROS state)
├── config/
│   ├── launcher_ovrtx_isaac_physx.yaml  ← Isaac Sim 5.1 + PhysX + OVRtx renderer (stable default)
│   ├── launcher_ovrtx_isaac_newton.yaml ← Isaac Sim 6.x + Isaac's Newton wrapper + OVRtx (rigid only, experimental)
│   ├── launcher_newton_mjwarp.yaml      ← newton-standalone, MuJoCo-Warp rigid (no cloth)
│   ├── launcher_newton_mjvbd.yaml       ← newton-standalone, MuJoCo-Warp rigid + VBD cloth
│   ├── launcher_newton_mjxpbd.yaml      ← newton-standalone, MuJoCo-Warp rigid + XPBD cloth
│   ├── launcher_newton_fsvbd.yaml       ← newton-standalone, Featherstone rigid + VBD cloth
│   ├── launcher_newton_avbd.yaml        ← newton-standalone, Featherstone rigid + augmented-VBD cloth
│   └── scene_*.yaml                     ← scene configs (robot + cameras + optional newton + optional debug)
├── scripts/
│   ├── gripper_cmds.py                  ← test driver for /joint_command gripper setpoints
│   └── wbc_cmds.py                      ← whole-body-control command driver
└── rviz/                          ← RViz2 config files
```

> **Newton standalone is experimental** — `launcher_newton_*` and
> `launcher_ovrtx_isaac_newton` print a yellow `[ENTER]` gate before
> startup. The stable rigid-body path is `launcher_ovrtx_isaac_physx`.

---

## Command-driver coverage

`scripts/wbc_cmds.py` discovers movable non-mimic joints from the live URDF.
Wuji's `hand_l_*` / `hand_r_*` names form the `hand` group, with independent
`--hand-deg` and `--vel-hand` controls; `gripper` remains the parallel-jaw group.
An unmatched filter reports `NO_MATCHING_JOINTS`. Initial commands require
feedback for every selected joint, and missing feedback cannot count as reached.
Timeout records include each stuck joint's final goal, published command,
measured position and signed error; preserve this distinction for ramp tests.
See the [README execution checks](README.md#command-and-robot-execution) for
copyable commands. `test/test_wbc_cmds.py` expands the shipped Wuji xacro and
checks 40-joint discovery, limits, ramp speed and missing-feedback behavior;
run it with `python3 -m pytest test/test_wbc_cmds.py` in a sourced ROS runtime.

## Runtime joint parameters

Compatible G2 crsB and ARX Isaac/Newton scenes select `robot.joint_overlays`
(package URIs) alongside the sysid xacro selector. Keep both selections when
copying a scene. Runtime precedence is per field: last overlay, then existing
class/importer defaults. Explicit zero overrides; mimic followers remain
undriven. PhysX passive parameters use Isaac 6.2 native joint-axis properties;
older PhysX requires an explicit baseline selection. Newton's MuJoCo-Warp
adapters apply SI values after class/mimic writes and before solver construction.
Do not fold passive damping into drive damping or map Coulomb torque to the
legacy load-dependent joint-friction coefficient. See the README coverage and
verification matrix.

## Launch architecture

### Entry point: `app.launch.py`

Requires `launcher_config` (no default — exits with a clear error if missing).
Reads the launcher YAML to determine the physics engine, then delegates to one
of two sub-launches:

| `launcher.physics.engine` | Sub-launch |
|---|---|
| `genie_sim_engine_isaacsim` (default) | `physics_backend.launch.py` |
| downstream engine declared in launcher YAML | `physics_backend.launch.py` |

Common args declared here and forwarded to sub-launches:
`scene`, `physics_hz`, `render_hz`, `headless`, `always_regenerate_robot_usd`,
`physics_engine`, `physics_solver`, `robot_model`, `body`, `arm`, `gripper`,
`use_sim_time`, `remap_tf`, `fake_slam`, `log_level`, `interaction_tools`,
`launcher_config`.

### `physics_backend.launch.py`

Owns the physics pipeline. Declares its own args:
`physics_hz`, `render_hz`, `headless`, `always_regenerate_robot_usd`,
`physics_engine` (`isaac_physx` | `isaac_newton` | `newton`),
`physics_solver` (`mujoco-warp` — the only supported value; pinned at
`choices=["mujoco-warp"]` so wrong values can't be typed. Honored by
`isaac_newton`; ignored by `isaac_physx` (PhysX has its own scheme) and
`newton` (newton-standalone hardcodes Featherstone for rigid and reads
its cloth solver from the scene yaml's `newton.solver.prefer`)),
`physics_solver_substep`, `physics_solver_iterations`, `render_mode`
(`raster` | `pathtrace` | `offline`).

Isaac/Newton orchestration order (all gated behind the assemble pipeline):
```
[urdf_to_usd?] → [isaac_scene] → [newton_extras] → [physics engine node + industrial_bridge? + renders?]
```

Downstream backends select assembly without changing this package. A backend
with its own executable uses `assemble_style: standalone_hook`; a backend
registered with `genie_sim_engine_core.assemble` uses
`assemble_style: registered_pipeline` and an `assemble_pipeline` mapping with
`name`, `modules`, and optional `robot_stage`. When `robot_stage` is present,
bringup uses that stage name for cache-hit skipping; otherwise the backend's
pipeline owns its stage-level skip policy.

`assemble_newton` belongs to the Isaac/Newton pipeline and self-gates
on whether the scene yaml carries a non-empty `newton.entries` block. A
downstream registered pipeline controls its own stage list and does not run
`assemble_newton` unless it explicitly registers such a stage.

`physics_engine` / `physics_solver` / `physics_solver_substep` /
`physics_solver_iterations` / `render_mode` are forwarded as Node
params to the active engine node — **do not rename these keys** without
updating the engine entry points
(`genie_sim_engine_isaacsim/scripts/genie_sim_engine_isaacsim_node.py` /
`genie_sim_engine_newton/scripts/genie_sim_engine_newton_node.py`) and
`genie_sim_engine_core`'s `common/params.py`, which parses them.

### `utils.py`

Pure helper module — no ROS state, no side effects. Key functions:

| Function | Purpose |
|---|---|
| `perform(context, name)` | Safe `LaunchConfiguration` resolver |
| `resolve_scene_yaml_robot_params()` | Merge scene YAML robot block with CLI overrides |
| `resolve_bringup_config_file()` | Resolve a config name to an absolute path |
| `load_launcher_yaml()` | Parse launcher YAML → `(launcher_section, plug_yaml, params_path)` |
| `resolve_physics_engine()` | Extract ROS engine id / package / executable from launcher YAML |
| `make_assemble_pipeline()` | Emit the single `python -m genie_sim_engine_core.assemble` process the runtime nodes gate on |
| `make_industrial_bridge_node()` | Construct the industrial_bridge Node |
| `make_render_ovrtx_node()` | Construct the render_ovrtx Node |
| `make_render_isaacsim_node()` | Construct the render_isaacsim Node |
| `resolve_active_renders()` | Filter render list to installed packages |
| `resolve_assets_folder()` | Resolve assets by priority: `./assets` → `/geniesim_assets` → `geniesim_assets.ASSETS_PATH` fallback |

`make_render_ovrtx_node()` resolves the active `ovrtx/bin` without importing
`ovrtx`, then prepends it to `LD_LIBRARY_PATH` through the render `Node`'s
`additional_env`. Keep this environment process-local: globally importing or
exporting OVRTX pollutes other app/assembly processes with its USD plugins.

When a required scene name cannot resolve, `resolve_scene_yaml_robot_params()`
reports the requested name as `scene YAML not found`; an omitted argument keeps
the `scene is required` message. New YAML files need a bringup rebuild to be
installed into an existing symlink-install config directory.

`make_render_isaacsim_node()` honors `GENIESIM_PY_CMD`, otherwise uses the launch
interpreter when its installed metadata includes Isaac. Preserve the ROS underlay
for that same-interpreter case (system Python 3.12 with Isaac 6.2, or 3.10 with
Isaac 4.5). Only bundled `omni_python` filters incompatible system native paths.

---

## Launcher YAML schema

```yaml
launcher:
  physics:
    engine: genie_sim_engine_isaacsim   # ROS engine id (selector key)
    engines:
      genie_sim_engine_isaacsim:
        package: genie_sim_engine_isaacsim
        executable: genie_sim_engine_isaacsim_node.py
        name: genie_sim_engine
        # Downstream registered-pipeline example:
        # assemble_style: registered_pipeline
        # assemble_pipeline:
        #   name: backend_pipeline_name
        #   modules: [backend_package.assemble]
        #   robot_stage: urdf_to_usd
    industrial_bridge: ""           # optional node name
  renders:
    - render_ovrtx                  # or render_isaacsim

render_ovrtx:                       # params block for the render node
  ros__parameters:
    plugin:
      - genie_sim_render/RosImagePublisherPlugin
```

`physics_engine` / `physics_solver` / `physics_solver_substep` /
`physics_solver_iterations` / `render_mode` are **launch CLI args
with launcher-YAML defaults** — each launcher YAML's
`genie_sim_engine.ros__parameters` block sets the engine-appropriate
defaults (e.g. `launcher_newton_fsvbd.yaml` defaults to
`physics_engine: newton, physics_hz: 60, physics_solver_substep: 10,
physics_solver_iterations: 5`), and any `key:=value` on the
`ros2 launch` CLI overrides on top. Precedence: CLI > launcher YAML >
`DeclareLaunchArgument` default. See `_OVERRIDABLE_FROM_YAML` in
[physics_backend.launch.py](launch/physics_backend.launch.py) for
the full overridable set.

Newton inline OVRtx may additionally set
`genie_sim_engine.ros__parameters.ovrtx_camera_zoom_topic`. The shipped
`launcher_newton_mjxpbd.yaml` maps it to
`/genie_sim_engine/viewer/camera_zoom`, matching `view_robot.rviz`. This is an
additive `std_msgs/Float64` focal-length multiplier channel; the external
`render_ovrtx` process used by g2sim does not consume it.

---

## Scene YAML robot block

```yaml
robot:
  robot_model: genie
  body: default
  arm: default
  gripper: default
  robot_prefix: genie
  init_joint_pos:
    arm_l_joint1: -45.0   # degrees for revolute, metres for prismatic
  robot_source:
    package: genie_sim_robot_model
    urdf:                 # presence triggers URDF→USD route
      xacro_entry: xacro/robot.xacro
```

URDF mimic joints are handled by the URDF→USD importer (PhysX
`PhysxMimicJointAPI` / Newton `NewtonMimicAPI`); there is no scene-yaml
`mimic:` block to author. The Newton paths (`isaac_newton` and
`newton`) additionally broadcast each master target across its
followers in software (`engine/_mimic.expand_targets`) — Newton's
`apply_action` only drives DOFs whose names are passed in. See
[engine_core/docs/pipeline.md → Mimic joints](../genie_sim_engine_core/docs/pipeline.md#mimic-joints)
for the design.

`init_joint_pos` is forwarded as `init_joint_pos_json` (JSON-encoded) to the
physics node — deliberately kept out of the stage manifest so editing it does
not invalidate the assemble cache.

### Assembly parameter source

Shipped scenes select identified parameters by default wherever a compatible
set exists: G2 `crsB` scenes set `robot.robot_source.urdf.inertia_source: sysid`;
ARX `acone` / `x5` scenes set `robot.robot_source.urdf.param_source: sysid`.
`build_urdf_xacro_mappings()` passes these existing YAML keys to xacro for
`/robot_description`, which supplies the assembly input and URDF hash. Use
`default` explicitly in that scene mapping for a baseline comparison. New
scenes for these robots must retain the sysid selection. G2 `crs` and other
reference robots have no compatible selector and keep their source baseline.

This selects the transitional sysid xacro parameters, not the sparse overlays
or all runtime drive gains. Backend class tables still override some families;
see [identified-parameter limits](../genie_sim_robot_model/robots/SYSID_PARAMS.md).
Keep this distinction in assembly acceptance evidence. Raw xacro/CMake defaults
and authored visual USDs are independent of these scene defaults.

### `robot_source.prebaked` — hand-authored robot USD

```yaml
robot_source:
  urdf: {}                          # staged URDF→USD (physics topology)
  prebaked:
    usda: robot/G2_crsB_wuji/robot.usda   # relative to assets folder, or absolute
    visual_only: true               # optional; default false
```

| Key | Meaning |
|---|---|
| `usda` (only) | Prebaked USD drives **both** physics and rendering; the URDF→USD stage is skipped. For scenes that ship a hand-authored robot whose topology already matches the physics engine. |
| `usda` + `visual_only: true` | Prebaked USD drives **rendering only** (`robot_visual.usda`); physics still uses the staged URDF (requires `urdf:` present). Use when the authored visual asset has a **different joint namespace** from the URDF the physics engine needs — e.g. IPC cloth scenes where the prebaked visual is authored under `/genie/...` but the CPU-MuJoCo/IPC physics is staged from the URDF under `/robot/...`. Without this split, feeding the `/genie/...` USD to physics raises PhysX "no bodies defined". |

`visual_only` is honored in `_scene_impl.py` (`assemble/isaac_scene` stage):
it opens the prebaked USD as a separate stage and exports `robot_visual.usda`
as a thin reference to it, while `robot_usda` in the manifest keeps pointing at
the staged URDF output.

---

## Scene YAML `newton:` block

Active under `physics_engine:=newton` (newton-standalone) or
`physics_engine:=isaac_newton` (Isaac's Newton wrapper). Ignored by
`isaac_physx`. Defines Newton-native particle features (cloth, soft
body, static box colliders) that the rigid-body USD bridge doesn't
cover.

The `newton_extras` assemble stage reads this block, validates it, extracts mesh
geometry from the referenced USD files, and writes
`newton_solvers.json` into the stage directory. The runtime
cloth-inject hook reads that sidecar to materialize particles in the
Newton `ModelBuilder` between `add_usd` and `finalize`.
The stage is unconditionally chained in the launch graph — it
self-gates on a non-empty `newton.entries` block, so scenes without
particle features pass through silently.

```yaml
newton:
  solver:
    # Required when entries contain cloth/particles. Pick explicitly:
    #   "vbd"     — unconditionally stable, garment-quality (newton-standalone only)
    #   "xpbd"    — PBD; the only particle solver the isaac_newton wrapper exposes
    #   "style3d" — higher-quality garment drape at greater per-step cost
    #               (newton-standalone only)
    # No "auto" — every path declares its solver explicitly so the yaml matches
    # the runtime.
    prefer: vbd

  contact:
    # Cloth ↔ rigid contact stiffness/damping/friction.
    # Model-global contact settings; no per-entry override.
    # VBD-only — XPBD ignores these (kept for cross-launcher parity).
    soft_ke: 1.0e4
    soft_kd: 1.0e-2
    soft_mu: 0.25

  entries:
    - kind: cloth                    # required; "cloth" or "box"
      name: tshirt                   # required; unique per scene — used as the USD prim
                                     # name and as the runtime particle-writeback key
      mesh_usd: scenes/blank/Female_T_Shirt.usd
                                     # relative to --base-path (assets folder), or absolute
      pose: [tx, ty, tz, qx, qy, qz, qw]  # spawn pose in world frame; default: identity
      vel:  [vx, vy, vz]             # initial linear velocity; default: [0,0,0]
      params:
        density:        0.02         # kg/m^2
        tri_ke:         1.0e4        # stretch stiffness
        tri_ka:         1.0e4        # area preservation
        tri_kd:         1.5e-6       # stretch damping
        edge_ke:        5.0          # bend stiffness
        edge_kd:        1.0e-2       # bend damping
        particle_radius: 0.008       # metres; contact + self-collision offset
```

All `params` keys for `kind: cloth` are **required** — `assemble_newton`
exits non-zero with a clear error if any is missing. `solver` and
`contact` sub-blocks are optional (shown defaults apply when absent).

**Static / dynamic split** (same rule as the rest of the manifest):

| Field | Baked into `newton_solvers.json` | Live from scene yaml at runtime |
|---|---|---|
| vertices + indices (from `mesh_usd`) | yes — extracted once | no |
| `pose` / `vel` | yes | no |
| `params` (density, stiffness, …) | **no** | **yes — re-read on every launch** |
| `contact` block | advisory only | re-read live |

Iterating cloth stiffness / damping doesn't require re-running
`assemble_newton` — change the yaml and relaunch. The geometry itself
(vertex positions, topology) is baked; changing `mesh_usd` requires
deleting the stage cache.

### Chef USDA object integration

`scene_flat_g2_sp_chef_wuji.yaml` enables `newton.chef.usd_objects`. Its
plugin uses the pre-import `on_stage(stage, ctx)` hook to author an anonymous
session-layer physics overlay. Visual selection is authored in the source
USDA itself.

| Key | Contract |
|---|---|
| `enabled` | Select the imported-pot path; false uses the plugin's procedural fallback. |
| `objects_root` | Must resolve to lowercase `/World/objects`. |
| `pot_body_prim_path` / `pot_mesh_prim_path` | Exact body and authored Mesh bindings for the pot. |
| `expected_food_count` | Fail-closed inventory guard; currently 42 retained objects (16 noodles + 26 side dishes). Import binding checks these exact object paths; each object may produce multiple Newton shapes, all disabled before its soft proxy takes over. |
| `expected_pot_mass` | Fail-closed composed-mass guard; currently 1.3 kg. |
| `particle_proxy_thickness` | Hidden particle-only bowl thickness; 0.080 m preserves the leak-resistant tuning. |
| `procedural_food_transition` | Selects authored food proxies instead of procedural food. |

`newton.chef.usd_side_dishes.migrated_families` selects the six retained,
configured families for FEM compilation. `procedural_fallback: false`
prevents procedural side dishes from being emitted beside those authored objects.

`newton.chef.startup_hand_motion` is a Wuji-only, finite startup trajectory.
The robot must begin from the collision-safe values in `robot.init_joint_pos`;
after `delay_s` of simulation time the scene plugin publishes a smoothstep
ramp to `target_joint_pos` through `/joint_command`, holds the final target for
`hold_s`, then stops publishing so teleoperation can take over. Keep all values
in both YAML mappings in degrees; ROS commands are converted to radians.

After `add_usd`, the plugin reuses the imported pot body, remaps the procedural
principal inertia onto the fitted USDA bowl axes, clears particle collision
from its convex pieces, sets their rigid gap to zero, and adds one hidden
particle-only thick shell.  It does not add a second pot, handle, or stove.

### Chef semantic food recording

`scene_flat_g2_sp_chef_wuji.yaml` enables a plugin-owned, replay-oriented
particle stream under `newton.chef.food_state`.  The scene plugin publishes:

| Topic | Type | Contract |
|---|---|---|
| `/genie_sim_engine/chef_food/info` | `std_msgs/String` | JSON schema, topology hash, food-type table, and stable particle ranges for every food instance; transient-local and repeated at 1 Hz |
| `/genie_sim_engine/chef_food/state` | `sensor_msgs/PointCloud2` | 30 Hz sim-time XYZ plus `food_type`, `instance_id`, `local_index`, and Newton `global_index` |

The plugin's `record_snapshot_enabled()` query is true only while the food
publisher is live. Disabling `newton.chef.food_state.enabled`, or a publisher
initialization/runtime failure, disables snapshot callbacks and particle host
copies. No separate mirror setting is required.

The PointCloud2 stream contains raw FEM particles, not the Catmull-Rom visual
tube vertices.  Replay/render consumers must use the `info` mapping and the
same fixed reconstruction topology.  Always record `/clock` with both topics.

---

## Routing rules

- Launch arg declarations → `app.launch.py` (common) or `physics_backend.launch.py` (own)
- Launcher YAML parsing → `utils.py:load_launcher_yaml`
- Scene YAML parsing → `utils.py:resolve_scene_yaml_robot_params`
- Assemble pipeline wiring → `utils.py:make_assemble_pipeline` (one CLI process; stage list comes from the selected registered pipeline)
- Physics engine resolution → `utils.py:resolve_physics_engine`
- Newton extras assembly → `genie_sim_engine_newton/assemble/newton_extras_stage.py`
- Gripper test commands → `scripts/gripper_cmds.py`

## G2 camera pose acceptance

`scene_pnp_g2_op.yaml` and `scene_flat_g2_sp.yaml` use the front-camera pose
from `robot/G2_omnipicker/robot.usda`'s `head_link3/head_front_Camera`, in USD
graphics coordinates. An identity extrinsic places the synthesized
camera inside the head housing and returns black RGB with 2–3 cm depth.
Keep the pose in scene YAML; the assembler writes it verbatim. Fresh AS3
conversion supplies no baked camera and must synthesize this pose.

## Wuji evaluation pairing

`scene_flat_g2_sp_chef_wuji_eval.yaml` plus `launcher_newton_wuji_eval.yaml` is the explicit locally validated procedural baseline: default inertias, pregrasp pose, 46 procedural foods, authored visual background only, no startup hand publisher. The default `scene_flat_g2_sp_chef_wuji` retains its authored physics, sysid selection and food recording. Evaluation uses `step_mode: lockstep`, one requested tick per target, frozen-state cameras, optional video, and launcher-only `wok_evaluation_json` thresholds. See the scene and launcher YAML for the evaluation configuration.


## Optional libuipc bringup

The additional `launcher_libuipc*` configurations select the independently
registered `libuipc` assembly pipeline and `physics_engine: libuipc` /
`physics_solver: ipc`. Existing engine/solver choices and runtime cache paths
remain valid. Native scene settings live in the scene YAML's `libuipc:` block;
[the backend guide](../genie_sim_engine_libuipc/AGENTS.md) owns that contract.
`scripts/fold_cloth_infer.py` is an optional policy adapter for the new cloth
scene. It does not alter existing Newton or Isaac task launchers.
