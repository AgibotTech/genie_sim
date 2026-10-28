# Newton integration constraints

The current system runtime pins Newton **1.6.1** with MuJoCo/MuJoCo-Warp **3.12.0**.
This reference records active integration constraints and the code that handles
them. Numeric section IDs are stable because implementation comments cite them.
Verify imported versions and paths before applying a result to another runtime.

| ID | Constraint | Current handling |
| --- | --- | --- |
| 2 | Joint and actuator force caps are independent | Inspect both compiled caps |
| 4 | Mimic equality export omits custom solref/solimp | Apply shared MJCF postprocessing and live equality settings |
| 5 | Angular intermediate values differ from final MJCF units | Validate the compiled model, not an intermediate writer dictionary |
| 6 | Principal-axis inertia interpretation affects model import | Check imported body inertia against authored link inertia |
| 7 | Root weld anchors have branch-dependent frame interpretation | Normalize misplacing welds in the session layer before import |
| 8 | `mjc:damping` requires Mjc schema resolution and SI authoring | Include `SchemaResolverMjc`; author per-radian values |
| 9 | Placeholder inertials become real rigid-body mass | Treat them as simulation approximations, not massless frames |

## 2. Joint and actuator force caps

`joint_effort_limit` and actuator `forcerange` can be authored independently.
The USD joint attributes `mjc:actuatorfrcrange:min/max` and
`mjc:actuatorfrclimited` do not imply the same cap as actuator custom attributes.
A joint cap is also possible without any actuator, so it is not evidence that a
position servo exists. Inspect both joint and actuator caps in the compiled
model when diagnosing saturation.

## 4. Mimic equality compliance

Newton's mimic equality writer does not carry `eq_solref` / `eq_solimp` into the
dumped equality, although other equality writers do. The live model update path
can consume those arrays, so a dump and running model can differ.

`common/mjcf_postprocess.py` applies
`augment_mjcf_gripper_from_params` through the shared postprocess pipeline.
The Isaac wrapper's `_stiffen_live_mimic_equalities` applies live values from the
same `usd_drive_api.gripper.mimic_eq_*` profile block. Both paths must retain the
same gripper scope. Do not extend the policy to unrelated loop constraints.
Compliance is timestep-sensitive; a stiffer equality is not automatically stable.

## 5. Angular units through MjSpec

The MuJoCo writer uses `rad2deg` for angular range/ref/springref values passed to
`MjSpec`; the serialized MJCF declares radians. The intermediate dictionary is
not the final unit contract. Validate a known URDF limit against the compiled
MJCF and solver range rather than removing a conversion based on its name.
Linear limits use metres. Read-side USD angular limits and DriveAPI gains have
separate conversions.

## 6. Principal-axis inertia

Newton's USD importer composes `R @ diag(D) @ R.T` from principal axes and
principal inertias. The integration's inertia checks compare the resulting
body-frame tensor with the URDF, particularly for near-degenerate eigenvalues
and cyclic axis orderings. Model import affects every solver, so an MJCF-only
correction cannot establish correctness of Newton model arrays.

The robot-model tool
[recompute_g2_inertia.py](../../genie_sim_robot_model/scripts/recompute_g2_inertia.py)
spreads near-degenerate pairs and avoids cyclic sorting. This is a modelling
approximation; inspect its resulting tensor rather than describing it as an
unchanged vendor inertia. Tiny RPY changes and equal eigenvalues do not establish
that the imported tensor is correct.

## 7. Root world-weld anchors

Root `FixedJoint` handling depends on whether `body0` resolves to a non-rigid prim
or an empty world path. A nonidentity `localPose0` can be composed twice or lose
an ancestor offset. `body_q` immediately after import is insufficient evidence:
check the pose after FK or a solver step, when the weld anchor takes effect.

`engine/newton/setup/stage.py::_repair_root_weld_anchor` runs after initial base
placement and before `add_usd`. For a misplacing weld it clears `body0`, writes
the child body's composed world transform to `localPose0`, sets `localPose1` to
identity and records `newton:weldAnchorIsWorld` for idempotence. It edits only the
session layer and leaves a correctly authored weld untouched.

Only `pin_base_mode: weld` retains the root joint. `free` and `mocap` deactivate
it; the schema's [base-pin contract](../../genie_sim_engine_core/docs/pipeline.md#base-pin-runtime-toggle)
defines the resolved mode. Preserve diagnostics and post-FK readback.

## 8. Damping namespaces and resolver selection

| Attribute | Angular units | Consumer contract |
| --- | --- | --- |
| `mjc:damping` | SI per radian | `SchemaResolverMjc`, then `joint_damping` |
| `mjc:stiffness` | SI per radian | MuJoCo passive stiffness attribute |
| `drive:angular:physics:stiffness/damping` | Per degree | USD/PhysX resolver converts to SI model gains |
| `newton:damping` | Per degree | Newton schema resolver |

`setup/model.py` passes Newton, PhysX and Mjc resolvers together. Omitting the
Mjc resolver can discard `mjc:damping` on Newton 1.6.1. The assembly fallback
`_HAND_JOINT_DAMPING_USD` is **0.05**, authored verbatim; dividing it by pi/180
would increase the intended SI damping by about 57.3 times.

For selected sysid stacks, runtime name-keyed values apply after class fallbacks
and take precedence over assembly seeds. Check
[JOINT_PARAMS.md](../../../../../docs/JOINT_PARAMS.md) for native PhysX 6.2 and
Newton application. Regenerate robot USD after changing assembly authoring;
editing a generator does not update an existing cache.

## 9. Placeholder frame inertials

Isaac 6.2 conversion treats a frame with an authored placeholder inertial as a
rigid body with that mass. It is not a massless kinematic frame. Keep the URDF
root massless for KDL and diagnose non-root missing/placeholder inertials with
[robot-model tools](../../genie_sim_robot_model/AGENTS.md#mesh-developer-tools).
Load-bearing links need measured or justified inertials. Body counts alone do
not prove a physically sound articulation.

## Verification

Use the [Newton tools guide](../genie_sim_engine_newton/tools/AGENTS.md) for
assembly snapshots, compiled MJCF, PD readback and solver checks. Run fresh robot
assembly and initialized joint-parameter checks from the
[bringup matrix](../../genie_sim_bringup/README.md#scene-testing). Retain the
complete USD payload tree with each result; a root USDA file alone is incomplete.
