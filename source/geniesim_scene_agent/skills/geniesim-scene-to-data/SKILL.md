---
name: geniesim-scene-to-data
description: Coordinate GenieSim requests across scene design, optional task orchestration and optional data collection, preserving scene approvals and routing only the requested stages.
---

# From scene design to collected data

Use this skill for a request spanning scene creation, robot task authoring,
and collection, or to continue a task across those stages. Reuse completed
work and existing approvals. Select the relevant stage skills below; do not
restart an accepted scene or ask for the same decision again.

## Select the requested scope

| Deliverable | Required stages | Installation |
| --- | --- | --- |
| Scenes only | Scene preparation, review, optional approved variants | Scene tools and configured Isaac Sim renderer or scene-agent image |
| Scene and task code | Scene preparation, orchestration, static validation | Scene tools and bundled RobotAPI references/examples |
| Executed task or collected episodes | Scene preparation, orchestration, physical validation; collection only if requested | Add the compatible collection runtime/image |

Scene generation is a standalone workflow. Use
[scene generation](../geniesim-scene-generation/SKILL.md) directly for scene-only
requests. Install/configure data collection only when execution is needed.
Code authoring can finish without execution; report physical checks as
unverified. A requested action demonstration needs the robot runtime even
without a requested dataset. Complete the requested stages and their approved
variants; do not automatically advance to episodes.

## Maintain the scene and optional task contract

Record the English scene identity, object roles, appearance constraints,
approved layout, and actual validation evidence. For robot tasks, add behavior,
robot, and final physical conditions. Keep separate decisions for the scene,
demonstrated task behavior, scene generalization, and collection scope.
Scene acceptance is not evidence of motion success.

Scene-only output uses the shared generator contract:

```text
<output_root>/<scene_id>/
  generalization.json
  batch_report.json
  <numeric_instance>/
    scene.usda
    scene_info.json
    generation.json
    validation.json
    renders/{front,side,top}.png
```

For newly generated robot tasks, use the task ID as the scene ID and add
`data_gen.py` and complete `task_info.json` at the scene root. These task files
are not required for scene-only output. Keep the recipe, optional patch,
reviewed dimension specification, and approval/evidence notes with the writable
project. Assets remain external. Collection outputs and runtime USD copies
live outside the scene/task library. Retained task examples add manifests/tags
at closeout.

## 1. Find references and build one foreground scene

Scene preparation is the first stage of every task request. Inspect and reuse
a supplied or previously accepted scene when suitable; generate missing scenes
before coding the task. Apply
[scene generation](../geniesim-scene-generation/SKILL.md), search similar
scenes first, and inspect several candidates with
[asset search](../geniesim-asset-search/SKILL.md). Include only requested
foreground objects/supports unless background was explicitly requested.

Prefer DSL; use a DSL plus USD patch for unsupported properties such as a
hinge, and direct Python USD authoring for unsupported layouts. All routes
share the same output contract. Use mesh/support tools for bounds, shelves,
interiors and clearance. When a robot task is requested, account for the
intended robot workspace without claiming reachability before execution.

Render front/side/top views, inspect the final USD and metadata, and show the
actual scene. Revise through user feedback until the single scene is accepted.
Do not use the static scene preview as an action demonstration.

## 2. Agree on scene generalization

After scene approval, ask whether variants are wanted. If yes, confirm concrete
dimensions, ranges, fixed relations and total count. A usual suggestion of
1000 means **scene instances**, including the base instance, not episodes.
Record the approved spec; defer expensive full generation until the base
task has been physically validated when the request includes robot execution.
If the user explicitly wants scene-only generation, complete that approved
scope without requiring a robot task. Route scene-only requests directly to
step 5 for approved batches, or deliver the accepted single scene. Continue
to step 3 only when orchestration is requested.

## 3. Author and validate orchestration

Apply [find-example](../find-example/SKILL.md) to compare several task flows,
interfaces and scene contracts, then
[task orchestration](../geniesim-task-orchestration/SKILL.md). Author
`execute_task(robot) -> bool` with robust role selection, accurate descriptions,
ordered action phases, cleanup, and live physical completion checks.

Inspect asset grasp annotations. Prefer compatible precomputed candidates
when there is no special grasp constraint; infer/filter custom candidates
when annotations are absent or do not match the required direction/contact.
Inspect interaction labels, config extras, frames and deliberate-contact
collision policy using the bundled references. Select compatible robot and
camera settings in `task_info.json`.

Apply [task validation](../geniesim-task-validation/SKILL.md) for static checks
and, when execution is requested, runtime checks. For code-only delivery,
complete the approved scene variants if any and hand off scene/code artifacts
with the actual static validation level; do not require a collection image. Persistent IK failure uses the diagnosis/reach workflow;
review revised layouts through a single-scene preview and retest the failing phase.

## Scene revisions during orchestration

User feedback, contact requirements, clearance checks, and reach diagnosis may
require a scene revision during task development. Review revised layouts and
batch ranges with the user before execution. Apply the scene skill and update
recipe/patch parameters, USD and `scene_info.json` together, keeping role IDs
and selectors consistent. Generate a revised draft in a fresh output location,
validate it and show its renders. Check task goals, grasp candidates and camera
framing, then retest the affected phases.

## 4. Execute and show the complete task

For requested robot execution, apply
[data collection](../geniesim-data-collection/SKILL.md). Use one explicit
scene and one run with `--render` and recording enabled. The collection runtime
executes the policy and records camera videos directly; replay is optional.
Check the final physical state, returned status, recording metrics, trajectory
artifacts and videos before showing the action video to the user.

Iterate on the user's behavioral feedback while preserving the accepted scene
contract. Record task acceptance separately from measured physical/recording
results. Reuse acceptance already given in the session. An unavailable runtime
check remains explicitly unverified and cannot satisfy the execution gate.

## 5. Generate the approved variants and test coverage

If variants were not requested, retain the accepted fixed scene. Otherwise,
generate only the approved dimensions/count after scene acceptance. Requests
including robot execution also require accepted, physically validated base
behavior before the expensive batch. Scene-only and code-only batches do not
require that runtime gate; retain their actual validation level. Use a fresh output root: the generator refuses existing
instances or a changed `generalization.json`. Preserve the accepted base
parameters through each dimension's `base`, seed and recipe/patch, and compare
the resulting instance `0` to the approved scene before treating it as the
same design. Do not modify the approved draft in place.

The current generator requires the same layout/background instance-key sets
across a batch. Do not promise object-count changes that violate this check.
The recipe must actually consume every approved parameter. Validate all USDs,
metadata and planned relationships; render the batch under the scene skill's
contract and inspect representative extremes. Retain rejected-sample evidence.

For orchestration requests, carry `data_gen.py` and task metadata into the
variant task root, preserving the actual test level. For scene-only delivery,
finish with the validated scene artifacts and previews. For code-only delivery,
finish with the statically checked code and matching scene contracts.
When execution is requested, execute selected boundary and representative scene IDs with direct recording.
Check grasping, clearance, selectors, final conditions and camera coverage.
If a dimension fails, diagnose and revise within approval; confirm any changed
dimension/range/layout before regenerating. Do not extrapolate a success rate
from one base scene.

## 6. Confirm and execute collection scope

Enter this stage only when the user requests episode collection. An accepted
scene or task demonstration does not authorize a data production batch.
Before bulk execution, confirm scene IDs/coverage, episodes per scene or total
episodes, robot noise, camera/render settings, failure/retry limits, and output
budget. Reuse explicitly approved details. Keep scene count and episode count
separate; do not silently interpret 1000 scenes as 1000 successful episodes.

For coverage of every scene, enumerate explicit IDs. `--scene random` samples
with replacement and cannot guarantee coverage. Ordinary `--run-times N`
runs episodes serially; deformable and attached-behavior tasks require separate
single runs. Keep an index of attempts, scenes, outcomes, episode paths and
failure phases. Stop on a common repeated failure instead of wasting the batch.

Verify physical outcomes, final task/metric status, trajectory conversion and
requested camera videos. Report attempted/successful/failed counts with a
denominator, actual scene coverage, and data/output paths. Keep failures visible
and preserve useful diagnostics. A dataset upload/publication is a separate
authorized action.

## 7. Retain accepted work and lessons

Apply [task-driven self-improvement](../geniesim-task-orchestration/references/task-self-improvement.md)
after task acceptance, and update it with later collection evidence. Summarize
observed pitfalls and working corrections, update the relevant skills, and
retain the accepted task plus representative approved scenes as searchable
examples. Refresh interface tags, retrieval tables and file inventories, and
validate changed artifacts. Keep all skill text and explanatory comments
English; preserve required runtime language keys.

Use [the phase checklist](references/phase-checklist.md) to hand off an
in-progress task or review its completion without losing approvals and evidence.
