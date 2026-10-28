# Joint parameters: baseline URDF and identification overlays

Joint overlays are sparse SI parameter records. The read-only schema CLI inspects
and validates them; `JointParamResolver` applies selected joint values at runtime
on Isaac PhysX 6.2 and Newton MuJoCo-Warp. Link inertials use xacro selection.

## Ownership

| Change | Source | Consumer |
| --- | --- | --- |
| Baseline link inertials and joint limits | Robot `*.params.default.xacro` | URDF importer |
| Identified link inertials | Sysid xacro selected by `inertia_source` / `param_source` | URDF importer |
| Identified joint physics and gains | `robots/<vendor>/<model>/overlays/<id>.yaml` | Runtime resolver and backend application helpers |
| Class fallback gains and passive properties | `genie_sim_engine_core/config/physics_params*.yaml` | Backend class tuning |
| Selected overlay stack | Scene `robot.joint_overlays` | Backend path resolution, then core resolver |

Core owns the [schema](../src/ros_ws/src/genie_sim_engine_core/genie_sim_engine_core/schema/joint_params.py)
and [resolver](../src/ros_ws/src/genie_sim_engine_core/genie_sim_engine_core/common/joint_params.py).
Robot model owns the [overlay data](../src/ros_ws/src/genie_sim_robot_model/robots/OVERLAYS.md).
Core takes filesystem paths and imports neither robot assets nor backend runtimes.

## Runtime selection

Compatible scenes explicitly select an ordered list of package URIs or absolute
paths. For example, the G2 crsB Swiftpicker stack is:

```yaml
robot:
  joint_overlays:
    - package://genie_sim_robot_model/robots/genie/g2/overlays/g2_crsB.sysid-2026-09-22.yaml
    - package://genie_sim_robot_model/robots/genie/g2/overlays/g2_swiftpicker.sysid-2026-09-18.yaml
```

Last overlay wins per field. Explicit zero overrides a class value; an omitted
field retains the class fallback. `applies_to` must match the robot, and joint
names must match the solver's robot joints. Nonzero follower drive gains fail:
mimic followers remain constraint-driven, with their passive properties applied.

For a complete baseline comparison, use `robot.joint_overlays: []` and select
xacro `inertia_source: default` or `param_source: default` as appropriate.
See [bringup defaults](../src/ros_ws/src/genie_sim_bringup/README.md#sysid-parameters-are-the-assembly-default).

## Backend application

| Property | Newton MuJoCo-Warp | Isaac Newton wrapper (experimental) | Isaac PhysX 6.2 |
| --- | --- | --- | --- |
| Passive damping | SI `joint_damping` | SI `joint_damping` after class writes | Native axis viscosity |
| Coulomb friction | `joint_friction` | `joint_friction` after class writes | Native static/dynamic friction efforts |
| Armature | `joint_armature` | `joint_armature` after class writes | Native axis armature and USD armature attribute |
| Drive gains | Final model values before solver compilation | USD seeds, SI tensor gains, final model overrides | USD seeds and SI tensor gains |
| Link inertials | Selected sysid xacro | Selected sysid xacro | Selected sysid xacro |
| Command delay, torque constant | Not consumed; logged | Not consumed; logged | Not consumed; logged |

Older PhysX runtimes reject selected passive overlays because they lack the
required `PhysxJointAxisAPI`. Newton Featherstone/AVBD reject selected runtime
overlays. Libuipc does not apply the joint overlay stack. PhysX overlay velocity
limits are also rejected. Choose an explicit baseline configuration for these
paths rather than assuming selected fields reach the solver.

The runtime keeps passive damping separate from drive damping. Angular USD
DriveAPI gains and PhysX viscosity use `SI * pi/180`; tensor/model values remain
SI. PhysX static and dynamic friction efforts both receive `frictionloss`.
Readback verifies parameter delivery, not equal response across solvers.

## Schema v1

The emitted JSON Schema uses [JSON Schema draft 2020-12](https://json-schema.org/draft/2020-12/schema).

```yaml
schema: genie_sim.joint_params/1
id: g2_crsB.operator-example
applies_to:
  robot_model: genie
  body: g2
  arm: crsB
meta:
  kind: operator
  caveats: Single-joint experiment; no identification claim.
joints:
  idx21_arm_l_joint1:
    damping: 0.0
    armature: 1.51643658513
```

| Group | Fields | Units |
| --- | --- | --- |
| Link inertials | `mass`, `com`, `com_rpy`, `inertia.{ixx,iyy,izz,ixy,ixz,iyz}` | kg, m, rad, kg·m² |
| Passive joint physics | `damping`, `frictionloss`, `armature` | N·m·s/rad, N·m, kg·m² for revolute joints |
| Drive | `stiffness`, `damping_drive`, `max_effort` | N·m/rad, N·m·s/rad, N·m for revolute joints |
| Limits | `effort`, `velocity` | N·m, rad/s for revolute joints |
| Recorded only | `command_delay`, `torque_constant` | s, N·m/A |

Prismatic equivalents use linear SI units. A field's presence in the schema does
not imply backend support; use the application table above.

Scalar values accept either a number or `{value: ..., note: ...}` with optional
`confidence` and `frozen_to_cad` annotations. Only `stiffness`, `damping_drive`,
`damping` and `frictionloss` accept engine maps keyed `mujoco`, `newton`, `physx`.
An engine map cannot mix `value` with branch keys. Missing branches do not inherit
a sibling's value; runtime selection fails on an unresolved field.

The schema engine keys differ from registered backend names. The in-tree helper
selects `mujoco` for standalone MuJoCo-Warp and the Isaac Newton wrapper, and
`physx` for Isaac PhysX. The schema's `newton` branch does not establish
Featherstone/AVBD runtime support.

`meta.kind` is `sysid`, `correction` or `operator`. A sysid overlay with drive
gains requires `meta.identified_on_engine`: `mujoco_pd`, `isaacsim_physx` or
`unrecorded`. This records the fitted control law. `meta.not_identified` names
unmeasured fields; publishing one of those fields in the same overlay is an error.
Unknown fields, nonfinite numbers and names absent from a supplied URDF fail
validation. Keys use exact URDF names, including `idxNN_` prefixes.

Do not edit a `kind: sysid` measurement for an experiment. Add an operator overlay
and append it. Do not fill missing measurements with baseline numbers or encode
unit compensation as an engine-specific fit. The shipped baseline includes both
CAD and simulation choices; it is not a claim of vendor identification.

## Read-only tools

Run after building and sourcing the workspace. Paths below are from the repo root.

```bash
M=genie_sim_engine_core.schema.joint_params
R=source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model
OVERLAY="$R/robots/genie/g2/overlays/g2_crsB.sysid-2026-09-22.yaml"
URDF="$R/urdf/genie_g2_crsB_swiftpicker.urdf"
python3 -m "$M" validate "$OVERLAY" --urdf "$URDF"
python3 -m "$M" resolve --overlay "$OVERLAY" --urdf "$URDF" \
  --engine mujoco --joint idx21_arm_l_joint1
python3 -m "$M" diff --overlay "$OVERLAY" --urdf "$URDF" --engine mujoco
python3 -m "$M" bilateral --overlay "$OVERLAY" --engine mujoco
python3 -m "$M" emit-schema --out joint_params.schema.json
```

`resolve`, `diff` and `bilateral` accept repeated `--overlay` options. Reports
identify the winning source. Their A/B channel labels classify fields; they do
not confirm a URDF patch or runtime write. No general overlay-to-URDF patch is
implemented.

`convert --to physx` is a read-only drive-only conversion analysis: it folds
passive damping into drive damping and reports unsupported friction or follower
cases. It does **not** describe native PhysX 6.2 application and must not be used
to pre-fold values selected by the current runtime. `convert` does not edit
artifacts or configure the engine.

## Verification

```bash
python3 -m pytest -q \
  source/geniesim_ros/src/ros_ws/src/genie_sim_engine_core/tests/schema/test_joint_params.py
```

Runtime checks are owned by
[Isaac sysid verification](../src/ros_ws/src/genie_sim_engine_isaacsim/AGENTS.md#runtime-sysid-precedence)
and the [bringup matrix](../src/ros_ws/src/genie_sim_bringup/README.md#scene-testing).
The [identified parameter reference](../src/ros_ws/src/genie_sim_robot_model/robots/SYSID_PARAMS.md)
records measurement limits. Unimplemented application work belongs in the
[roadmap](ROADMAP.md).
