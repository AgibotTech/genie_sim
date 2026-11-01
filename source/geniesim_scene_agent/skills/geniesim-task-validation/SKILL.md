---
name: geniesim-task-validation
description: Validate GenieSim data_gen.py task contracts, RobotAPI usage, physical completion checks, and collection recordings on fixed scenes and intended variants.
---

# Validate automatic collection tasks

Use this skill after authoring or changing task orchestration, or when a task
fails. Read the local [complete API reference](../geniesim-task-orchestration/references/robot_api_reference.md)
for exact signatures and the [task contract](../geniesim-task-orchestration/references/task-contract.md)
for inputs and execution ownership. Work from evidence about the failing
phase; change the smallest relevant selection, geometry, or motion assumption.

## Check code and scene contracts

Parse or compile `data_gen.py` without importing the robot runtime. Verify the
entrypoint, explicit imports, argument names, tuple unpacking, and required
action descriptions. Check that every success path includes the requested
physical condition, and that failures cannot fall through to `True`.

Review several actual scene instances: selectors must return the intended
objects and counts, fields such as `target_letter` must be valid, asset IDs
must resolve, and needed prims, interaction labels, and joints must exist.
Validate the composed USD with the bundled scene tools. Inspect grasp files
with the [grasp tool](../geniesim-task-orchestration/references/grasp-selection.md)
and use mesh/support tools for actual surfaces, interior space, and clearance.
Unknown asset IDs may be skipped by runtime object registration; a mesh that
renders is not necessarily registered for RobotAPI manipulation.

Review [execution tips](../geniesim-task-orchestration/references/execution-tips.md)
and [effective Config extras](../geniesim-task-orchestration/references/config-extra-params.md).
Check that contact phases use the intended planner exclusion while preserving
required physical collision, and that their cleanup is explicit. Check ignored
extras, unconditional success returns, logged-but-unused checkers, and use of
truthy tuples as booleans.

Check meters, xyzw/wxyz conversion, degree/radian boundaries, object/gripper
pose semantics, base/goal offset composition, and valid candidate transforms. Check that protected objects
are stationary by task definition, not objects expected to settle or move.

Use a recording adapter or mock when useful to exercise target selection and
action order across metadata variants. This verifies code decisions, not
contacts, IK, collision avoidance, or physical completion.

For code-only requests, finish with static contract checks and report physical
execution/recording as unverified. Installing a collection module/image is not
a prerequisite for those checks. Apply the runtime sections below when actual
robot execution or physical validation is requested.

## Execute a fixed scene first

Apply [data collection](../geniesim-data-collection/SKILL.md) to launch the
configured collection runtime. Select one explicit scene and one run before
using random selection. Record the task/scene, robot, initial-state noise,
seed controls actually available, first failing phase, and observed final state.

```bash
geniesim data-collection run "$TASK_NAME" \
  --task-root "$TASK_ROOT" --assets-root "$ASSETS_ROOT" \
  --scene "$SCENE_ID" --run-times 1 --render --output-dir "$RUN_DIR"
```

The high-level CLI enables recording and headless execution by default.
`--render` records configured camera videos directly during task execution;
replay is optional. Add `--no-headless` for a GUI with rendering, or choose
`--no-record` for a non-recorded physical/debug check. Verify actual video
outputs as well as the episode status. Read the bundled
[CLI reference](../geniesim-data-collection/references/cli.md) for preparation,
camera settings, outputs, timeouts and single-run restrictions.

The CLI manages the server and client, copies the selected task to an isolated
run directory, and rebinds that copy's USD assets. A direct legacy launcher
requires an existing server and may rewrite its input USD; its recording flag
is off by default in the inspected current version. Use explicit recording
arguments and separate task processes when choosing that lower-level path.

If a required pose/path repeatedly has no IK solution, follow
[IK recovery](../geniesim-task-orchestration/references/ik-recovery.md): verify
the contract and try bounded task-valid alternatives, then present a concrete
layout correction if geometry/reach appears to be the cause. Obtain user
confirmation before moving scene objects/supports or the robot base. After an
approved change, verify scene metadata/USD, preview the single scene and retest
the failing phase before assessing the complete task.

## Check physical completion

- Grasp: object follows the intended hand and remains stable through the next
  motion; a closed gripper alone does not prove a grasp.
- Container: object lies in the usable interior at a valid height and remains
  there after release and settling. XY overlap alone is insufficient.
- Stack/shelf: object rests on the intended support, with the required
  orientation and preserved surrounding objects.
- Articulation: the intended joint reaches the requested interval/direction,
  and the cabinet or appliance remains stable.
- Insertion: alignment, insertion depth, and connection state match the goal.
- Writing: requested glyph, paper-frame placement, pen contact, stroke order,
  and pen-up gaps are correct. Waypoint completion alone does not prove ink.
- Cloth: measure the intended fold/node relationships after releasing pins.

Read checker values as well as the action return. Check the scene after the
last release, reset motion, or other action that could undo completion.
Explicitly restore temporary planner/attachment/pin/collider state. Pinning
objects or disabling physical colliders changes simulation behavior and must
remain consistent with the task's intended collection semantics.

## Cover variants and recording

After the fixed scene passes, cover approved variation boundaries; obtain or
reuse the collection scope before executing a variant batch:
different assets and counts, smallest clearances, mirrored positions, scene
task fields, and robot initial-pose noise. Use reproducible selections where
supported. Summarize successes and failures with a denominator and scene IDs;
do not infer a rate from an unmeasured collection run.

When recording is part of the requested output, verify the overall instruction,
per-action descriptions, camera names, recording start/stop, final status, and
the configured quality metrics. A valid robot trajectory and a usable recorded
episode are separate checks.

Store test outputs in `.cache/` or the selected output directory. Report the
actual checks performed, unresolved failures, and whether physical execution
and recording were available. Do not describe an unavailable runtime test as
passed.

Present the evidence and remaining limitations for user acceptance. After the
task is accepted, complete the
[self-improvement closeout](../geniesim-task-orchestration/references/task-self-improvement.md):
summarize lessons, update relevant skills and retain the approved task/scenes
with the actual validation level. Keep user acceptance distinct from measured
physical/recording results.
