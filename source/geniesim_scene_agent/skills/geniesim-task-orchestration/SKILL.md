---
name: geniesim-task-orchestration
description: Author or adapt GenieSim automatic collection tasks in data_gen.py using RobotAPI, scene metadata, grasp annotations or geometry-derived grasps, and physical completion checks.
---

# Author automatic collection tasks

Create task code that works across the intended scene variants. Use the
installed collection runtime for robot execution; the authoring references,
inspection tool, and reference tasks are bundled in this repository.
Asset data stays in the configured external assets root. No other source
checkout, prompt service, or example MCP server is required.

## Read only the relevant references

- [Task contract](references/task-contract.md): input/output files, object
  selection, metadata conventions, task descriptions, and execution ownership.
- [Complete RobotAPI reference](references/robot_api_reference.md): all public
  methods, full parameter tables, configuration constructors, supported
  extensions, return values, and implementation limitations. Search by method
  or parameter name; load the relevant sections rather than the entire manual.
- [Config extra_params](references/config-extra-params.md): per-Config keys,
  defaults, units, merge order, nested schemas, and options that are ignored
  or broken in the inspected execution path. Read before adding extensions.
- [Grasp selection](references/grasp-selection.md): where to find annotated
  poses, how to inspect them, and how to derive task-specific candidates.
- [Motion patterns](references/motion-patterns.md): shelves, articulated
  objects, insertion, held containers, bimanual motion, writing, and cloth.
- [Execution tips](references/execution-tips.md): deliberate contact and collision
  scope, frame composition, extraction, precision feedback, ownership, checkers,
  rollback and known source traps. Read the matching recipe before coding.
- [Per-task source review](references/example-review.md): all bundled examples,
  reusable decisions and limitations to address when adapting each task.
- [Repeated IK failure](references/ik-recovery.md): diagnose persistent
  infeasibility, propose specific layout changes, obtain user confirmation,
  and validate the revised scene and motion.
- [Task-driven self-improvement](references/task-self-improvement.md): use user
  acceptance to close a task, summarize lessons, update skills and retain the
  accepted implementation as a searchable example.
- [Task examples](../../examples/README.md): select existing implementations with
  [find-example](../find-example/SKILL.md), then adapt their scene selectors,
  grasp decisions, and physical checkers to the task.

## Design before coding

Start with scene preparation. Apply
[scene generation](../geniesim-scene-generation/SKILL.md) to generate and review
the required scene, or inspect and reuse a supplied/accepted scene. Keep object
roles and scene metadata ready before implementing motion. Code authoring and
static checks use the bundled references without installing a collection image;
robot execution is an optional downstream stage determined by the request.

Use the bundled [find-example skill](../find-example/SKILL.md). Compare several
examples by action sequence, robot profile, object contract, and completion
check. Read their implementations and scene metadata. Each task's README and
manifest list statically observed interface tags; these describe calls in the
code, not proof that every branch ran successfully.

Write a short plan with object roles and selection rules, ordered phases,
per-phase inputs and completion checks, stationary objects, and any temporary
attachments or planner obstacle changes. Identify whether the task requires
annotated placement, explicit object poses, gripper poses, or ordered paths.
Resolve material ambiguity with the user while continuing independent work.

Use [scene generation](../geniesim-scene-generation/SKILL.md) when the layout
needs to change during task development. For a revision, keep recipe/patch,
USD, metadata and selectors consistent, preview the revised scene and retest
affected phases when execution is requested. Keep its foreground default, preview process, and batch
approval requirements. Scene instance count and collected episode count are
separate settings.

## Choose grasps explicitly

Inspect the actual asset's grasp annotations before choosing an action.

1. With usable annotations and no special grasp requirement, prefer the
   runtime's annotated candidates.
2. With a required contact position, approach direction, or articulated part,
   inspect and filter annotations against that requirement. Derive custom
   candidates when none satisfy it.
3. With missing or empty annotations, infer grasp regions from mesh geometry,
   current pose, gripper width, obstacles, and the subsequent motion. Supply
   custom `GraspConfig.grasp_pose`, `grasp_pose_frame`, and `grasp_width`.

Generate several valid candidates when the geometry permits. A single precise
trajectory may be appropriate for tight contact tasks. An IK result alone
does not establish a collision-free approach or a stable grasp. Never replace
failed grasp reasoning with an unrelated random pose.

## Implement the task

Read the relevant [execution tips](references/execution-tips.md) and the selected
example's [review note](references/example-review.md). For deliberate contact,
plan approach, scoped planner exclusion, bounded contact, a live physical
checker, withdrawal and restoration. Keep physical collision when it drives
the requested interaction. Derive relative offsets in the planner's actual
base/goal frame, and verify options against the selected Config and dispatch.

- Export `execute_task(robot: RobotAPI) -> bool`. Use explicit imports.
- Select objects dynamically from `robot.scene_info`; validate cardinality and
  roles. Read current poses after interactions move objects. Metadata poses
  describe the initial layout.
- Set the concrete task description before the first robot action. Match the
  instruction to the selected objects and actual execution order. For writing,
  read `scene_info.target_letter` once and use it for both text and strokes.
- Confirm interaction labels against `robot.interactions[asset_id]`. Use
  actor-to-actor placement only when the annotations express the intended
  relationship; otherwise calculate explicit object target poses.
- Derive surfaces and clearances from geometry. Prim origins need not be box
  centers, and an overall shelf bound does not locate individual shelves.
- Supply task-valid pose candidates with bounded position/orientation ranges.
  Preserve object-to-gripper transforms when converting object goals for IK.
- Check every required action result. Unpack `dual_act` into `success, state`;
  a tuple itself is truthy even when its success field is false.
- Use zero-argument `extra_checkers` that read current simulation state. Add a
  final settle check when later motion or release can invalidate a subtask.
- Track and restore temporary obstacle exclusions, attachments, FEM/rigid pins,
  and collider changes on success and failure. Restore only state introduced
  by this task phase; global detach/release operations affect other groups.
- Return failure when a required phase fails. Let the runtime handle its
  bounded candidate attempts and rollback; task-specific recovery should
  have an explicit state and stopping condition.

Write all comments and explanatory documentation in English. Runtime language
keys such as `zh` and `en` remain part of the API schema; preserve supplied
localized task text when requested without embedding it in this skill.

## Recover from persistent IK failure

Follow [IK recovery](references/ik-recovery.md) when a required grasp, endpoint
or trajectory remains unsolvable after bounded, task-valid alternatives.
Check pose frames, units, gripper conventions, candidate filters, contact
obstacle policy and robot reach before attributing the failure to the layout.

When the layout appears to prevent a feasible approach or motion, propose the
smallest concrete change to object/support positions, orientations or spacing
that preserves the user's task. Show the current failure and proposed changes
with a render, workspace markers or a position comparison. Obtain user
confirmation of the specific changes before editing the scene. Then update
the generated USD and metadata together, preview the revised scene and test
the failing phase again. Existing approval of those exact changes remains
valid. A refused layout change leaves the approved layout fixed; report the
remaining limitation and pursue only alternatives within it.

## Validate and hand off

Apply [task validation](../geniesim-task-validation/SKILL.md) for static checks.
For code-only requests, deliver the code and matching scene with physical
execution explicitly unverified; no collection image is required. When robot
execution or an action demonstration is requested, add the compatible runtime
and apply [data collection](../geniesim-data-collection/SKILL.md) for a
fixed-scene execution with directly recorded camera video (`--render`). Use
[the complete workflow](../geniesim-scene-to-data/SKILL.md) when the request
also covers scene generalization and bulk collection. Report the task files,
selected scenes, runtime, validation results and output videos.

## Close the task and improve the next iteration

The user's requirements and acceptance feedback define whether the task meets
their needs. Present the concrete result and actual validation evidence for
that decision. Once accepted, apply
[task-driven self-improvement](references/task-self-improvement.md):

1. Summarize pitfalls, ineffective attempts, causes, working changes and
   useful shortcuts from this task, separating observed facts from hypotheses.
2. Merge reusable lessons into the relevant skill/reference. Keep asset- or
   task-specific constants in the task guide and correct superseded advice.
3. Add the accepted task and representative approved scenes to `examples/tasks/`,
   with English documentation, the actual validation level, consistent metadata,
   file inventories and externally bound assets.
4. Refresh interface tags and retrieval tables, validate the changed artifacts,
   and report what was learned and where it was retained. The next task reads
   these updated skills and searches the expanded example library.

Perform this local maintenance as part of the authorized workflow without
requesting a second approval for an already accepted task. Keep unresolved
tasks and speculative changes in draft form; record useful diagnosed failures
without presenting the draft as a completed example. A repository commit or
remote publication follows the user's separate Git instructions.
