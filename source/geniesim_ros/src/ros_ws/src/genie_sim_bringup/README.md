# genie_sim_bringup

Launch orchestration for the GenieSim ROS 2 stack. `app.launch.py` reads a
launcher YAML, selects physics/rendering and composes the ROS nodes.

| Location | Purpose |
|---|---|
| `launch/app.launch.py` | Entry point; requires `launcher_config:=…` |
| `launch/physics_backend.launch.py`, `launch/utils.py` | Assembly, engine startup and shared launch helpers |
| `config/scene_*.yaml` | Robot, initial pose, cameras, background and task features |
| `config/launcher_*.yaml` | Physics backend, solver and rendering configuration |
| `rviz/` | RViz layouts |

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_g2_sp \
  launcher_config:=launcher_ovrtx_isaac_physx
```

See the [workspace quick start](../../../../README.md) for setup and
[AGENTS.md](AGENTS.md) for architecture and YAML/CLI precedence.

## Scene testing

This README owns the manual scene test plan. Update the matrices, prerequisites
and checks alongside scene, launcher or assembly/cache changes. Inventory:
**16 scene YAMLs**: 15 matrix cases and one separate Wuji evaluation scene. **No scene has completed every
checklist item**; R01 component smoke results are recorded below. Coverage
symbols describe required checks; they do not establish passing results.

| Priority | Cases | Count | Scope |
|---|---|---:|---|
| [Tier 1 — G2 Isaac/Newton](#tier-1--g2-isaacnewton) | R01–R04, R12–R13, N01, N03 | 8 | G2 rigid, cloth and chef scenes |
| [Tier 2 — reference robots](#tier-2--reference-robots) | R05–R11 | 7 | Reference robot scenes |

Run Tier 1 first, then Tier 2 reference robots. Robot family support follows
[robot-model policy](../genie_sim_robot_model/AGENTS.md#support-tiers).
Tier 1 does not imply that experimental cloth/chef solvers have completed
runtime acceptance. The separate `scene_flat_g2_sp_chef_wuji_eval` scene uses
`launcher_newton_wuji_eval` for procedural evaluation and frozen observations;
it is outside this manual scene matrix.

### Tier 1 — G2 Isaac/Newton

Base: **free** = mobile, **pinned** = fixed root, **welded** = pinned root plus
configured base/head/body subtrees converted to fixed joints. Cameras count
configured sensors, not images already validated.

**Physics backends:** Isaac PhysX, standalone Newton MuJoCo-Warp,
MuJoCo-Warp + VBD, and MuJoCo-Warp + XPBD. The rigid MuJoCo-Warp path is
maintained; the cloth/soft-body combinations are experimental. **Render/viewer:** external OVRtx for the PhysX runs, Newton GL
for the rigid Newton comparisons, and inline OVRtx for the cloth/chef runs.
The two columns below show the configured pairing. Parenthesized keys identify
launchers; **bold physics entries mark primary runs**. See the
[launcher key](#scene--backend-matrix) for exact names. These are test-plan
routes, not completed acceptance.

**Experimental only:** Isaac Newton rigid wrapper (**IN**) is outside the
supported-backend list and main matrix. Its optional checks are listed under
[additional launcher tests](#additional-launcher-tests).

| ID | Scene (`scene:=`) | Physics backends | Render/viewer | Robot / base | Cameras | Distinct checks |
|---|---|---|---|---|---|---|
| R01 | `scene_flat_g2_sp` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | G2 crsB + swiftpicker / free | 1 RGB-D | Both arms/grippers; front-camera pose |
| R02 | `scene_pnp_g2_op` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | G2 crs + omnipicker / pinned | 1 RGB-D | Demo props; gripper collider/mimic behavior |
| R03 | `scene_wbc_g2_sp` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | G2 crsB + swiftpicker / free | 1 RGB-D | Wheels/body/arms; demo scene; inspect camera pose |
| R04 | `scene_wbc_g2_wuji` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | G2 crsB + Wuji / free | 1 RGB-D | Both hands; finger geometry/articulation; inspect camera pose |
| R12 | `scene_flat_g2_wuji` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | G2 crsB + Wuji / free | 1 RGB-D | Wheels/body/arms and both hands on generated floor; front-camera pose; no GDK dependency |
| R13 | `scene_flat_g2_op` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | G2 crs + omnipicker / free | 1 RGB-D | Both arms/jaws and mobile base on generated floor; source baseline parameters; no GDK dependency |
| N01 | `scene_flat_g2_sp_shirt` | **Newton MJ-Warp + VBD (V)** | Inline OVRtx (V) | G2 swiftpicker / welded | — | Shirt mesh, finite drape and cloth contact |
| N03 | `scene_flat_g2_sp_chef_wuji` | **Newton MJ-Warp + XPBD (X)** | Inline OVRtx (X) | G2 Wuji / welded | — | Authored wok; 16 noodles + 26 side dishes; startup hand motion |

### Tier 2 — reference robots

**Physics backends:** reference robots use Isaac PhysX and experimental
standalone Newton MuJoCo-Warp comparisons. **Render/viewer:** reference runs
pair PhysX with external OVRtx and Newton with Newton GL. Matching launcher keys
in both columns identify pairings. Bold physics entries mark primary runs.

**Experimental only:** Isaac Newton rigid wrapper (**IN**) comparisons for
reference robots are separate from supported backends; see
[additional launcher tests](#additional-launcher-tests).

| ID | Scene (`scene:=`) | Physics backends | Render/viewer | Robot / base | Cameras | Distinct checks |
|---|---|---|---|---|---|---|
| R05 | `scene_flat_acone` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | ARX acone / pinned | 3 RGB-D | Both arms/grippers; head and two wrist views |
| R06 | `scene_flat_x5` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | ARX X5A / pinned | 2 RGB-D | Arm/default gripper; head and wrist views |
| R07 | `scene_flat_piper` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | Agilex Piper / pinned | 2 RGB-D | Arm with_gripper; head and wrist views |
| R08 | `scene_flat_aloha` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | Agilex Aloha / pinned | 1 RGB-D | All four configured arm chains |
| R09 | `scene_flat_fr3` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | Franka FR3 / pinned | 1 RGB-D | Seven joints; initial elbow/wrist pose |
| R10 | `scene_flat_ur5_robotiq_85` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | UR5 + Robotiq 85 / pinned | 1 RGB-D | Correct gripper geometry and coupled finger motion |
| R11 | `scene_flat_ur5_robotiq_140` | **Isaac PhysX (P)**; Newton MJ-Warp (M) | External OVRtx (P); Newton GL (M) | UR5 + Robotiq 140 / pinned | 1 RGB-D | Correct gripper geometry and coupled finger motion |

## Scene × backend matrix

**● Primary** = the scene's first test. **○ Additional** = separate rigid-backend
comparison. **— Outside this plan** = no task-level acceptance claimed for that
pair. One ● per scene gives **15 primary runs**; the ○ cells add 13, for
**28 main scene/backend runs**. The 13 optional experimental Isaac Newton
wrapper comparisons below bring the combined count to **41 runs**. Keep the
case ID and record the backend separately.

| Cases | P | M | V | X |
|---|:---:|:---:|:---:|:---:|
| R01–R04, R12–R13 · Tier 1 G2 rigid | ● | ○ | — | — |
| N01 · Tier 1 shirt | — | — | ● | — |
| N03 · Tier 1 chef | — | — | — | ● |
| R05–R11 · Tier 2 reference robots | ● | ○ | — | — |

Use the exact launcher name below as `launcher_config:=…`.

| Key | Launcher | Physics backend | Render/viewer for this plan |
|---|---|---|---|
| P | `launcher_ovrtx_isaac_physx` | Isaac PhysX | External OVRtx |
| M | `launcher_newton_mjwarp` | Standalone Newton / MuJoCo-Warp (MJ-Warp) | Newton GL |
| V | `launcher_newton_mjvbd` | Standalone Newton / MuJoCo-Warp + VBD | Inline OVRtx (explicit override below) |
| X | `launcher_newton_mjxpbd` | Standalone Newton / MuJoCo-Warp + XPBD | Inline OVRtx (explicit override below) |

For N01 and N03, select `physics_engine_visualizer:=ovrtx` in the shared command
below. Other standalone Newton runs use their launcher's default viewer unless
you explicitly choose another. Isaac Newton's rigid wrapper does not validate
standalone Newton cloth/chef behavior.

## Prepare the runtime

On the host, from this checkout with the asset package available to the CLI:

```bash
# Build when the current image is missing or needs updating:
geniesim docker6.2 build
geniesim docker6.2 up
geniesim docker6.2 into
```

Inside the container:

```bash
cd /workspace
geniesim ros build dev
source /opt/ros/jazzy/setup.bash
source devel/setup.bash
mkdir -p test-results/scenes
```

| Runtime | Test scope / prerequisite |
|---|---|
| `docker6.2` | Isaac 6.2 fresh conversion and primary Isaac/Newton checks |
| `docker` / `docker5.1` | Runtime compatibility comparison; bare `docker` still selects 5.1 |
| `docker4.5` | Runtime compatibility comparison in Python 3.10 / Humble |

Build a separate ROS overlay for each Python/ROS runtime. Cold URDF conversion
needs the Isaac importer. Missing prerequisites are **Blocked**. Deformable
cases are not blanket compatibility test targets.

### Flat robot tests and GDK world prerequisite

R02–R04 require `scenes/GDK/demo.usda` under the resolved asset root, including
its referenced assets. With `/geniesim_assets` as the root, check before launch:

```bash
test -f /geniesim_assets/scenes/GDK/demo.usda
```

If this world is missing, record R02–R04 as **Blocked** until the GDK asset pack
is available at that path. Regenerating the robot USD does not supply the world.
Do not replace a missing demo with an unrelated scene and count the demo test
as passing.

For robot-motion tests without GDK, use R01 (`scene_flat_g2_sp`, Swiftpicker),
R12 (`scene_flat_g2_wuji`, Wuji) or R13 (`scene_flat_g2_op`, Omnipicker).
These generate their own floor and light and keep the base mobile. SP and Wuji
use crsB arms with compatible sysid selections for assembly and runtime. OP uses
crs arms with Omnipicker and retains source/class baseline parameters because
no compatible sysid selection is shipped for that combination.
Run one at a time inside the prepared container:

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_g2_sp \
  launcher_config:=launcher_ovrtx_isaac_physx \
  always_regenerate_robot_usd:=true headless:=true rviz:=true
```

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_g2_wuji \
  launcher_config:=launcher_ovrtx_isaac_physx \
  always_regenerate_robot_usd:=true headless:=true rviz:=true
```

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_g2_op \
  launcher_config:=launcher_ovrtx_isaac_physx \
  always_regenerate_robot_usd:=true headless:=true rviz:=true
```

If a newly added scene reports `scene YAML not found`, rebuild bringup with
`geniesim ros build dev` and source `devel/setup.bash` again. An installed
symlink directory does not automatically include newly created YAML files.

Use the [command/execution checks](#command-and-robot-execution) below for
arms, body and hands. Flat-case results cover those robots on a floor; GDK
props, contacts and task layout remain separate checks.

### Sysid parameters are the assembly default

The shipped scenes explicitly select the compatible identified parameter set
in `robot.robot_source.urdf`; no extra launch argument is needed.

| Scenes | Default YAML selector | Assembly coverage |
|---|---|---|
| R01, R03–R04, R12, N01, N03 · G2 crsB | `inertia_source: sysid` | Identified arm link inertials and URDF joint damping/friction |
| R05–R06 · ARX acone / X5 | `param_source: sysid` | Identified URDF drive and passive joint parameters |
| R02, R13 · G2 crs + Omnipicker; R07–R11 · other reference robots | Source baseline | No compatible sysid xacro selector is shipped |

For an intentional baseline comparison, change the applicable selector to
`default` in a separate scene YAML. Include the selector and source revision in
the run evidence. For a new compatible scene, copy the selector as well as the
robot configuration. Raw xacro/CMake defaults are independent of scene defaults.
On the first run after changing selection, use `always_regenerate_robot_usd:=true`
as below and verify the logged `xacro_args (description)` plus the staged
`robot_raw.urdf` values against the selected parameter source.

Runtime selection is separate and explicit: the compatible Isaac/Newton scenes
also ship `robot.joint_overlays`. For example, `scene_flat_g2_sp` selects:

```yaml
robot:
  joint_overlays:
    - package://genie_sim_robot_model/robots/genie/g2/overlays/g2_crsB.sysid-2026-09-22.yaml
    - package://genie_sim_robot_model/robots/genie/g2/overlays/g2_swiftpicker.sysid-2026-09-18.yaml
```

Wuji scenes select the Wuji hand overlay instead of Swiftpicker; the ARX scenes
select their corresponding arm overlay. R02 and R13 use crs arms with Omnipicker
and have no compatible overlay; do not apply crsB or Swiftpicker overlays to
these scenes. Last overlay wins per field, including
explicit zero. Class defaults fill omitted fields. Startup prints each winning
value and source under `[joint_params]`; missing joints and mismatched robot
selectors abort startup rather than silently using defaults. An absent/empty
`joint_overlays` list retains the original class behavior. For a complete
baseline comparison, set both the xacro selector to `default` and
`robot.joint_overlays: []`.

| Runtime field | Isaac PhysX 6.2 | Newton MuJoCo-Warp (rigid, VBD or XPBD composition) |
|---|---|---|
| Drive stiffness / damping | USD seed plus SI articulation tensor gains | `joint_target_ke` / `joint_target_kd` before solver compilation |
| Passive damping | Native per-axis viscous friction, separate from drive damping | `joint_damping`, SI, with no degree compensation |
| Coulomb friction | Native static and dynamic friction efforts | `joint_friction` → MuJoCo `dof_frictionloss` |
| Armature | Native per-axis armature, plus the shared USD attribute | `joint_armature` |
| Effort when supplied | Articulation tensor cap | Model effort cap |
| Velocity when supplied | Not implemented; rejects selection | Model velocity limit |
| Command delay / torque constant | Unimplemented; explicit startup warning | Unimplemented; explicit startup warning |

In Isaac Sim, inspect **Physics → Joint → Advanced** for native passive values.
Angular USD viscosity and drive gains are per degree: multiply the SI value by
`pi/180`; tensor readback is per radian. For `idx21_arm_l_joint1`, expect SI
passive damping **4.15966580248**, static/dynamic friction **5.24682837521**, and
armature **1.51643658513**. For Swiftpicker masters, expect runtime stiffness
**202.026337** and drive damping **13.345311**. Followers remain undriven while
retaining identified passive parameters.

The full PhysX passive path requires **Isaac Sim 6.2** (`PhysxJointAxisAPI`).
Older PhysX versions fail clearly when overlays are selected; use an explicit
empty overlay list for their baseline tests. Newton Featherstone/AVBD do not
consume these overlays and reject a nonempty selection. The experimental Isaac
Newton wrapper also preserves the selected values.
Link inertials still use the existing sysid xacro path; a general overlay-to-URDF
patch is not implemented. Authored visual USDs retain their own values.

Runtime regression checks (from the repository root, inside the prepared runtime):

```bash
python3 -m pytest -q source/geniesim_ros/src/ros_ws/src/genie_sim_engine_isaacsim/tests/test_runtime_joint_params.py
# Native PhysX schema, live tensor readback, and compiled Newton model checks:
python3 source/geniesim_ros/src/ros_ws/src/genie_sim_engine_isaacsim/tests/isaac62_joint_params_smoke.py
```

## One command for every matrix cell

Choose a scene and launcher from the matrices. Run one case at a time; stop it
with Ctrl-C before starting another. Change `run_id` for each run to retain logs.

```bash
run_id=R01-P-fresh-01
scene=scene_flat_g2_sp
launcher=launcher_ovrtx_isaac_physx
fresh=true
viewer_args=()
# For N01 and N03, enable the inline OVRtx viewer:
# viewer_args=(physics_engine_visualizer:=ovrtx)

set -o pipefail
ros2 launch genie_sim_bringup app.launch.py \
  scene:="$scene" \
  launcher_config:="$launcher" \
  always_regenerate_robot_usd:="$fresh" \
  headless:=false \
  rviz:=false \
  "${viewer_args[@]}" \
  2>&1 | tee "test-results/scenes/${run_id}.log"
```

The GUI run needs a display. For remote physics/topic checks change to
`headless:=true`; complete visual QA separately. Newton inline OVRtx requires
`headless:=false`. For the unchanged cache-reuse run, set `fresh=false`, choose
a new `run_id` and repeat the same launch command.

The force flag rebuilds xacro → URDF → USD from existing sources. It does not
download a new asset pack, normalize meshes, regenerate authored visual assets
or enable a missing backend. `scene_flat_g2_sp_chef.py` is the shared chef
plugin, not another launchable scene.

## Pass criteria matrix

Apply every relevant row to each scene/backend run. Suggested smoke duration:
**60 seconds after warmup**; this is a manual test-plan choice, not a performance
acceptance threshold.

| Check | Procedure / pass condition | Evidence |
|---|---|---|
| Fresh assembly | Correct sysid selector (where available); `urdf_to_usd` actually runs successfully before engine startup; a cache/hash message alone is insufficient | Logged xacro mappings and selected values in `robot_raw.urdf`; fresh `robot.urdf`, `robot.usda`, `manifest.json` under the printed `<assets_folder>/scenes/<scene>/` |
| Geometry / topology | Correct model/gripper; all intended limbs, meshes, movable joints and references; on AS3 inspect physics payloads/variant | Screenshot; staged URDF and manifest; archive the complete AS3 directory when retaining reopenable assets |
| Initial state | Pose and free/pinned/welded behavior match scene YAML | Initial joint/root readback and screenshot |
| Command response | Use the [command/execution checks](#command-and-robot-execution) to move arms and grippers within URDF limits; expected DOFs and mimic followers respond | Joint-state trace or video; scene angles are degrees, ROS commands radians; do not command welded joints |
| Stability / ROS | Advancing `/clock`; finite, continuously published `/joint_states`; coherent geometry; no NaNs or solver explosion | Logs and measured topic rates; record pacing separately from correctness |
| Cameras | Each configured sensor has correct topic/dimensions, advancing stamps, plausible RGB/depth and attachment during motion | Camera samples and stamps; no configured camera means N/A |
| Deformables / plugin | Check the scene-specific inventory/contact behavior in the tier tables and notes below | Screenshot/video plus particle/plugin diagnostics |
| Cache reuse / shutdown | Stop cleanly; unchanged scene/launcher with `fresh=false` reuses robot conversion and starts successfully; scene assembly still runs | Separate relaunch log and matching appearance/joint inventory |

In a second shell in the same container/ROS domain, source the same overlay:

```bash
ros2 node list
ros2 topic list
ros2 topic echo /joint_states --once
ros2 topic hz /joint_states
ros2 topic hz /clock
```

Stop `topic hz` with Ctrl-C. Save evidence before another launch overwrites the
stage directory. A published camera message alone is not usable-image acceptance;
a requested 100 Hz is not proof that a backend achieves it.

For OVRTX cameras, wait for `OVRTX first frame complete — renderer is live.`
before measuring image delivery. Cold shader/JIT caches can take several minutes;
the image topics are advertised during that wait. During chassis motion, check
that the chassis shell and wheel assemblies follow the same base pose. AS3
`manifest.robot.body_paths` must map `base_link` and the other links to their
physics-owned Xforms, not same-named mesh wrappers. Scene assembly refreshes
this map even when robot conversion is reused.

### Command and robot execution

The existing drivers are [wbc_cmds.py](scripts/wbc_cmds.py) for joint motion
and feedback, and [gripper_cmds.py](scripts/gripper_cmds.py) for parallel jaws.
Both publish `sensor_msgs/JointState` on `/joint_command` directly. This tests
the command → engine → robot → `/joint_states` path. MoveIt Plan/Execute and
ros2_control trajectory execution need separate tests.

Start the scene using the shared launch template, then run the driver in a
second shell in the same container with `source devel/setup.bash`. Wait for
assembly/engine startup first: the driver's `--duration` includes discovery
time. Run one command producer at a time; stop teleop/other motion drivers,
wait for chef startup hand motion to finish, and inspect
`ros2 topic info /joint_command --verbose`. If MoveIt is also needed for IK,
launch it with `use_ros2_control:=false` to avoid competing commands.

| Check | Scene scope | Driver / options | Expected evidence |
|---|---|---|---|
| C01 · smooth arms | G2 scenes, selected movable arm joints | WBC shared command below: `--cmd-mode ramp` | Correct discovered joint count; commands and measured motion; goals reached within tolerance |
| C02 · Swiftpicker jaws | R01, R03, N01, after matching the live leader names | `ros2 run genie_sim_bringup gripper_cmds.py both toggle --toggle-period 2` | Both jaws open/close; mimic followers move coherently; Ctrl-C to stop |
| C03 · step response | G2 rigid scenes after C01 | Repeat WBC command with `--cmd-mode step` | Measured overshoot, settling and persistent error under instantaneous targets |
| C04 · head/body | G2 scenes with these joints still movable | Repeat WBC command with `--joints head body --body-deg 1 --head-deg 3` | Selected joints respond; welded joints excluded |
| C05 · Wuji fingers | R04, R12, N03 with movable hand joints | WBC Wuji command below: `--joints hand_l_ hand_r_ --hand-deg 30 --vel-hand 30` | `groups: {"hand": 40}`; both hands' five fingers move, with feedback for all 40 joints |
| C06 · Omnipicker jaws | R02, R13 | WBC Omnipicker command below: `--joints gripper --cmd-mode ramp --vel-gripper 30` | Two commanded masters; URDF-limited travel within −45° to 0°; all 14 mimic followers move coherently |

Start with a small, reproducible arm excursion; this command excludes grippers,
body and chassis by selecting only the arm joint names:

```bash
mkdir -p test-results/scenes
set -o pipefail
ros2 run genie_sim_bringup wbc_cmds.py \
  --cmd-mode ramp \
  --joints arm_l_joint arm_r_joint \
  --arm-deg 5 --vel-arm 5 \
  --duration 60 --seed 42 \
  --tol 0.05 --goal-timeout 8 \
  2>&1 | tee test-results/scenes/R01-P-C01.log
```

For Wuji, select the hand names explicitly. The arm-only command above excludes
fingers, and `--joints gripper` does not match Wuji. Wuji uses a separate `hand`
group with a default ±20° excursion from the initial measured pose and a 30°/s
ramp speed. This check increases the excursion to ±30°, clipped to URDF limits:

```bash
ros2 run genie_sim_bringup wbc_cmds.py \
  --joints hand_l_ hand_r_ \
  --cmd-mode ramp --hand-deg 30 --vel-hand 30 \
  --duration 60 --seed 42 --tol 0.03 --goal-timeout 8 \
  2>&1 | tee test-results/scenes/R12-P-C05.log
```

Expect 40 active joints (20 per hand). Use `--joints hand_l_` or
`--joints hand_r_` to isolate one hand, or a substring such as
`hand_l_index_joint` to test one finger. These are independent joint sweeps;
they do not test coordinated grasping or object retention.

For Omnipicker, use the WBC driver's discovered URDF limits instead of the
Swiftpicker driver's fixed close position:

```bash
ros2 run genie_sim_bringup wbc_cmds.py \
  --joints gripper --cmd-mode ramp --vel-gripper 30 \
  --duration 60 --seed 42 --tol 0.05 --goal-timeout 8
```

Expect `groups: {"gripper": 2}`. Only masters are commanded; inspect the
seven mimic followers per jaw separately.

Use a distinct log filename for each scene/backend/check. In another shell,
record the command and measured response before running the driver:

```bash
ros2 bag record -o test-results/scenes/R01-P-C01-bag \
  /clock /joint_command /joint_states
```

Stop recording with Ctrl-C after the driver finishes. Check the bag and robot
view together; topic publication alone does not establish physical execution.

| Observation | Interpretation / action |
|---|---|
| `WAITING_FOR_DESCRIPTION` | Check the scene's `/robot_description` publisher and ROS domain |
| `WAITING_FOR_STATES` | Check engine startup and `/joint_states` transport |
| `NO_MATCHING_JOINTS`, `no_matching_joints` | Filter/exclusions matched no movable joints; Wuji needs `hand_l_` / `hand_r_`, not `gripper` |
| `description_received`, `joint_trace`, `MOVING` | Verify every intended joint is selected **and has fresh measured feedback**; inspect matching command/state names |
| `goal_timeout` | `stuck_detail` gives each stuck joint's final `goal`, last published `command`, measured `state` and signed `error` (rad/m), plus mode/tolerance. A ramp command still short of its goal suggests travel time; a settled position error needs contact/torque/gain inspection |
| `done`, exit 0, or `goals_completed` | End of driver run, **not an automated pass**; the count includes generated/timed-out goals |

The WBC driver waits for all selected joints' initial feedback, reports
`missing_feedback` names, and never counts a missing joint as reached. Validate
fresh feedback independently; received positions are retained between messages.
It skips fixed, continuous and mimic joints; mimic motion needs visual/feedback
inspection. The `--tol` value is rad for
revolute joints and metres for prismatic joints. Excursion/velocity options
use degree-to-radian conversion, so adapt them before testing prismatic joints
or reference robots; G2 filters and defaults are not universal presets.

The gripper driver's active preset commands:
`idx31_gripper_l_inner_joint1`, `idx71_gripper_r_inner_joint1`, open `0.0`,
close `-0.91` rad. This close position matches Swiftpicker. Omnipicker's lower
limit is `-pi/4` (−45°), so use C06 instead of this preset's close/toggle action.
Check the live names/limits after fresh assembly; there is no CLI robot-preset
selector. The gripper driver does not drive Wuji fingers or provide
reference-robot presets. Its source notes mirrored master readback on a
measured Newton scene: inspect jaw separation and mimic behavior before
interpreting a sign difference as failure. Prefer repeated commands over
`--once`, which can race ROS discovery.

#### Wuji thumb tracking under gravity

The selected sysid parameters can produce a steady offset on both
`hand_*_thumb_joint1` joints under gravity. The runtime uses stiffness
**0.65 N·m/rad** and drive damping **0.05 N·m·s/rad**, zero passive friction,
and the URDF effort cap **0.4452 N·m**. The position drive has no integral or
gravity feedforward in this path. At rest, gravity is balanced by the drive
torque, so the expected position offset is approximately `gravity_torque / kp`.

Using the assembled URDF masses/COMs, measured joint configuration and base
orientation gave the following comparison with the first logged thumb goal:

| Joint | Calculated gravity torque | Predicted offset | Measured offset |
|---|---:|---:|---:|
| `hand_l_thumb_joint1` | 0.02563 N·m | 2.26° | 2.26° |
| `hand_r_thumb_joint1` | 0.03255 N·m | 2.87° | 2.88° |

Both exceed `--tol 0.03` (1.72°), so an 8-second timeout is expected even after
the ramp command reaches its final target. Increasing the timeout cannot remove
this steady offset. This diagnosis is for the sampled pose; contact, object
loads and changed finger poses require their own checks.

For an unloaded motion smoke check at this pose, `--tol 0.06` (3.44°) allows
the observed gravity deflection. Record that tolerance explicitly; it does not
pass a 0.03-rad precision requirement. Tighter tracking requires a separately
validated controller change such as gravity compensation or an explicit
operator gain overlay. Preserve the measured sysid overlay. This targeted
diagnostic does not complete the full R12 scene acceptance matrix.

### Scene-specific prerequisites and interpretation

| Cases | Additional condition |
|---|---|
| R01, R12–R13 | Generated floor/light; no external scene USD required; mobile base; SP/Wuji use crsB sysid, OP uses crs source baseline |
| R02–R04 | Require `scenes/GDK/demo.usda` and its dependencies under the asset root; missing world blocks scene assembly after robot conversion |
| N01 | Scene selects VBD and `scenes/blank/T_Shirt_fold.usd`, with `scenes/blank/blank_cloth.usda` as background |
| N03 | Chef scene plugin requires standalone MJ/XPBD; a rigid Isaac launch does not test food/particles. Authored inventory: 42 food objects, 1.3 kg pot guard, finite startup hand trajectory; inspect configured chef food `info`/`state` topics for semantic recording |
| R03–R04, some reference scenes | Identity camera extrinsics remain in YAML; report bad views independently from conversion success |

## Additional launcher tests

These cover the remaining launcher YAMLs and are outside the 15-case,
28-run main matrix. The 13 optional IN comparisons are included in the combined
41-run count above; the other launcher tests below are outside that count.
They are validation targets, not assumed-compatible passing combinations.

| Launcher | Target | Required evidence |
|---|---|---|
| `launcher_ovrtx_isaac_newton` · **IN, experimental** | R01–R04, R12–R13 (Tier 1 scenes), R05–R11 (Tier 2 scenes); 13 optional comparisons | Isaac 6.2 only; explicitly select and acknowledge startup gate; validate rigid response and achieved pacing; outside supported-backend lists |
| `launcher_newton_fsvbd` | N01 shirt | Experimental Featherstone/VBD behavior compared with N01 |
| `launcher_newton_avbd` | N01 shirt | Experimental AVBD compatibility and behavior |
| `launcher_newton_mjwarp_multi_env` | Separate replicated-world test | World count, per-world state/reset isolation and RLinf profile; a single-world launch is insufficient |

The Isaac Newton wrapper is not enabled by default. Validate its live solver,
rigid response and achieved pacing separately; a startup banner does not establish
per-scene acceptance. The wrapper has no standalone Newton cloth/chef coverage.

## Results matrix

Copy one row per scene/backend/runtime/run. Record commit, asset revision,
image ID, Python/SDK versions, display mode, resolved assets path and artifact
locations alongside each run. Use **NR** (Not run), **P** (Pass), **F** (Fail),
**B** (Blocked) or **—** (N/A). These result codes are independent of launcher keys.

| Run ID | Tier | Scene / launcher / runtime | Assemble | Robot / motion | Physics / ROS | Camera | Cache | Overall | Evidence / issue |
|---|:---:|---|:---:|:---:|:---:|:---:|---|
| R01-P | 1 | `scene_flat_g2_sp` / P / selected runtime | NR | NR | NR | NR | NR | NR | Record artifacts after running the full checklist |
| R01-M | 1 | `scene_flat_g2_sp` / M / selected runtime | NR | NR | NR | NR | NR | NR | Record artifacts after running the full checklist |

Overall Pass requires every applicable check to pass. This plan tests engine
and assembly integration; it does not establish autonomous task completion.

## References

- [Launch and scene contracts](AGENTS.md)
- [Assembly pipeline](../genie_sim_engine_core/docs/pipeline.md)
- [Build workspace](../../../../skills/build-workspace/SKILL.md)
- [Launch scene](../../../../skills/launch-scene/SKILL.md)
- [Wuji chef startup](../../../../skills/run-wuji-chef-newton/SKILL.md)


## Optional native IPC scenes

The added launchers are `launcher_libuipc`, `launcher_libuipc_ovrtx`,
`launcher_libuipc_isaacsim` and `launcher_libuipc_wuji_eval`. Added Wuji scenes
cover cloth folding, cloth hanging, drawstring bags and paper folding. They use
external task/background assets named in their scene YAML. Existing scene
configurations and launchers are retained.

For example, after installing the native runtime and building the ROS workspace:

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_g2_wuji_ipc_fold_cloth launcher_config:=launcher_libuipc
```

The [backend guide](../genie_sim_engine_libuipc/README.md) lists configuration
and runtime requirements. The three-task evaluation launcher enables
[synchronized observations](../genie_sim_client/IPC_CLIENT.md); successful
assembly or CPU checks do not establish physical task or camera acceptance.
