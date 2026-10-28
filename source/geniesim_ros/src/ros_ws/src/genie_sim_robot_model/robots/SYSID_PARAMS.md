# Identified parameter reference

The five shipped sysid overlays contain 610 fields. Their YAML is the source of
truth for numeric values and measurement caveats. This reference describes the
current selection and the limits on interpreting those values.

See [overlay conventions](OVERLAYS.md), the [schema and CLI](../../genie_sim_engine_core/genie_sim_engine_core/schema/joint_params.py)
and [bringup guide](../../genie_sim_bringup/README.md).
Paths in the table are relative to this `robots/` directory.

## 1. The five sets

| Overlay | Target | Links / joints / fields | Fitted law | Measurement limits |
| --- | --- | --- | --- | --- |
| [g2_crsB.sysid-2026-09-22](genie/g2/overlays/g2_crsB.sysid-2026-09-22.yaml) | G2 crsB arm | 14 / 14 / 182 | No drive gains | Offline torque regression; some masses frozen to CAD, passive values not refit in the inertial revision |
| [g2_wuji.sysid-2026-09-24](genie/g2/overlays/g2_wuji.sysid-2026-09-24.yaml) | G2 crsB Wuji hand | 0 / 40 / 200 | `mujoco_pd` | Uniform hand parameters; accepted replay evidence used different armature on Newton, so it does not establish agreement at the shipped inertia |
| [g2_swiftpicker.sysid-2026-09-18](genie/g2/overlays/g2_swiftpicker.sysid-2026-09-18.yaml) | G2 crsB Swiftpicker | 0 / 12 / 60 | `mujoco_pd` | Position-only fit of a coupled chain; passive terms and gains are jointly identified |
| [ac1.sysid-2026-09-30](arx/acone/overlays/ac1.sysid-2026-09-30.yaml) | ARX acone | 0 / 12 / 72 | `mujoco_pd` | Left-arm response parameters reused on both arms, except independently fitted delays |
| [x5a.sysid-2026-09-30](arx/x5/overlays/x5a.sysid-2026-09-30.yaml) | ARX x5 | 0 / 6 / 36 | `mujoco_pd` | Transplant from AC1 left arm; no independent X5A sweep |

All shipped controller fields are flat values rather than per-engine fits.
A shared number does not establish cross-engine agreement. No set identifies
PhysX drive gains. Readback on PhysX verifies delivery, not identification quality.
For the ARX sets, the date identifies publication rather than a recorded capture.

## 2. Selection and runtime application

| Input | Selection | What it establishes |
| --- | --- | --- |
| G2 crsB link inertials, passive damping and friction | Xacro `inertia_source:=sysid` | Values authored in the generated URDF |
| ARX drive gains and passive parameters | Xacro `param_source:=sysid` | Values authored in the generated URDF |
| Sparse joint/link overlays | Core schema CLI or `JointParamResolver` | Validated records and resolved values, independent of any solver |

The active `genie_sim_engine` launcher does not consume `robot.joint_overlays`.
Its class tuning can overwrite imported URDF joint parameters. A generated
`*_sysid.urdf`, successful schema validation, or a resolver result therefore does
not prove runtime application. Armature has no standard URDF slot. Command
delay and torque constant remain unconsumed.

For a baseline comparison, select xacro `default`. Preserve the original
identification records and inspect initialized solver values before drawing
conclusions about the selected parameters.

## 3. Units and provenance

| Fields | Revolute-joint units | Notes |
| --- | --- | --- |
| `damping`, `damping_drive` | N·m·s/rad | Passive viscosity and velocity-error drive gain are separate quantities |
| `frictionloss` | N·m | Coulomb effort, not a load-dependent friction coefficient |
| `armature` | kg·m² | Reflected inertia |
| `stiffness` | N·m/rad | Position drive gain |
| `effort`, `max_effort` | N·m | URDF rating and runtime clamp; neither is identified by the shipped sets |
| `velocity` | rad/s | URDF limit; not identified here |
| `command_delay`, `torque_constant` | s, N·m/A | Recorded fields without runtime consumers |

Link fields use kg, m, rad and kg·m². Joint/link keys are exact URDF names,
including `idxNN_` prefixes. Only four fields can branch by schema engine key:
`stiffness`, `damping_drive`, `damping`, `frictionloss`. The schema CLI accepts an engine key to inspect engine-specific values;
that selection does not configure a simulation runtime.

## 4. Measurement caveats

| Set | Caveats that affect testing |
| --- | --- |
| crsB | L2/L3 torque sensitivity to mass is about 7e-10 N·m/kg; CAD-frozen masses are not independent estimates. The friction basis zeros below 0.02 rad/s, affecting 42.2% of J2 training points. Training/holdout overlap is unknown. Right L3 distribution extends beyond its collision shell. No drive gain or effort limit was fitted. |
| Wuji | Stiffness 0.65 and drive damping 0.05 are response choices. Armature is 0.0005; replay evidence at 0.03 cannot validate that response at the shipped value. Passive damping/friction are explicit zero. The fit's uniform 0.5 N·m clamp exceeds some per-joint URDF ratings and is not published as a limit. Gravity can produce a steady position offset; measure position error against the applied gravity load. |
| Swiftpicker | Each side is one master plus five followers. Shared passive values describe a lumped chain, not six independent estimates. Bilateral equality is authored, not independently replicated. The fit used an 18 N·m master clamp and MuJoCo equality `solref -10000 -100`, `solimp 0.9 0.95 0.001 0.5 2.0`, without an input filter. These conditions are not all encoded in the joint overlay. The URDF's master effort 50 is a simulation setting, not an identified vendor rating. |
| AC1 | Delay supplied about 90% of the reported tracking improvement and remains unconsumed. J1 armature, J3 damping and J4 friction reached search-box edges. Right-arm passive values and gains reuse the left fit; delays are 0.085 s left and 0.089 s right. The 27/7 N·m harness clamps differ from URDF limits and are not applied by selecting this overlay. |
| X5A | Inherits AC1's fit limits and left-arm 0.085 s delay. Shared kinematics support use as a starting point, not a claim of independent identification. |

Do not edit immutable sysid records to compensate for an adapter. Their caveats
include statements about the fitting environment; current runtime behavior is
defined by section 2 and the implementation, not by those run descriptions.

## 5. Verification

Build and source the workspace, then run from the repo root:

```bash
M=genie_sim_engine_core.schema.joint_params
R=source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model
python3 -m "$M" validate \
  "$R/robots/genie/g2/overlays/g2_crsB.sysid-2026-09-22.yaml" \
  --urdf "$R/urdf/genie_g2_crsB_swiftpicker_sysid.urdf"
python3 -m "$M" resolve \
  --overlay "$R/robots/genie/g2/overlays/g2_crsB.sysid-2026-09-22.yaml" \
  --urdf "$R/urdf/genie_g2_crsB_swiftpicker_sysid.urdf" \
  --engine mujoco --joint idx21_arm_l_joint1
```

Repeat validation against each matching generated URDF. Inspect the reported
source, then the initialized solver readback. Schema validation alone does not
establish runtime application or hardware matching. Use the
[bringup guide](../../genie_sim_bringup/README.md) for
fresh/cached assembly, commands, state, cameras and stability. Record overlay IDs,
image/source identity, initial pose, clamp and measured tolerance with each run.
