# Recovery from repeated IK failure

Use this guide when a required grasp, motion endpoint or ordered trajectory
remains infeasible after checking the inputs and trying bounded alternatives
that still satisfy the task. Record the first failing phase and stop repeating
an identical unsolvable request. A runtime exception, empty filter result or
failed physical checker needs its own diagnosis before being called IK failure.

## Diagnose before changing the scene

| Check | Evidence to collect | Possible correction |
| --- | --- | --- |
| Target contract | Current actor/part pose, EE pose, finite target matrices or waypoint arrays, candidate counts | Fix object-versus-gripper targets, transform composition, units, wxyz/xyzw order and stale poses. |
| Grasp filtering | Candidate counts before/after flips, percentiles, posture and task-direction filters | Retain several task-compatible candidates; check custom-gripper conventions and measured width. An empty filter pool is not an IK result. |
| Frame and constraints | Base/goal frame, free axes, intended translation/rotation and relative offset | Correct the frame and path weights. False goal-frame flag uses base axes in cuRobo; review offset composition. |
| Endpoint versus path | IK mask for each meaningful candidate and the first failing waypoint; planner failure information when available | Separate unreachable pose, blocked approach, invalid start state and later path infeasibility. Reachable endpoints do not establish a feasible path. |
| Intended contact | Actual contact prim, gripper/tool geometry, planner obstacles and physical colliders | Use a scoped planning exclusion for required contact, retaining the physical interaction. Keep unrelated geometry in the planner. |
| Robot/profile | Actual robot, EE axes, joint limits, selected arm and base pose | Test appropriate arms and task-valid wrist alternatives. Position-based arm selection alone is insufficient. |
| Clearance and reach | Mesh-derived surfaces, compartment ceiling, fingers/tool envelope, receiver opening and neighboring objects | Reduce a valid lift, extract along a clear axis, split phases or derive a different compatible grip. |

Read [Config extensions](config-extra-params.md) for ignored/broken branches and
[execution tips](execution-tips.md) for contact and frame behavior. Correct an
input/API problem before treating the scene as the cause. Use a finite candidate
or recovery budget; widening noise must not change the required contact region,
stroke, insertion axis or requested final relationship.

## Propose a layout change to the user

If the diagnosed layout leaves no feasible task-valid motion, consider moving
or rotating the manipulated object, receiver or support, increasing spacing,
changing support height, or moving a target into the robot's reachable workspace.
Choose the smallest change supported by the evidence. Preserve the task's object
roles, counts and spatial constraints. A robot-base relocation is also a layout
change requiring explicit confirmation; changing robot type or task meaning
requires its own agreed requirement change.

Before editing the scene, present:

- The failing phase and attempted valid alternatives, with actual IK/planner
  results. State the suspected cause and its confidence.
- The specific objects/parts affected, current and proposed xyz/orientation,
  frame and units; derive proposed values from current geometry.
- A current render with relevant markers, a reachable-workspace visualization
  when available, or a compact position comparison that makes the change clear.
- Which requested relationships remain satisfied, and any unavoidable change
  to the user's original scene requirements.

Ask for confirmation of this concrete proposal. The generic ability to consider
layout changes does not authorize an unspecified rearrangement. Previously
approved exact changes may be applied without asking again. While awaiting
confirmation, continue source/geometry analysis and task-valid candidate work
that does not depend on the changed layout. If the user declines, keep the
approved layout and explain any remaining infeasibility.

## Apply and verify the approved change

Use [scene generation](../../geniesim-scene-generation/SKILL.md) and its tools.
Patch or regenerate the task's generated scene; preserve the external source
assets. Keep the foreground-only default, stable role IDs and common output
structure. Update composed USD and `scene_info.json` consistently; update the
recipe/parameters when present so regeneration reproduces the approved change.
Recompute dependent targets from the new live poses and geometry.

Inspect USD transforms, support, overlap, aperture/hand clearance and asset
references. Render the revised single scene and show it to the user for scene
acceptance. Test the previously failing phase in the configured robot runtime,
then the complete task, including final release/settling checks when available.
Report render, IK and physical-execution results separately.

Review variation dimensions, ranges and count through the batch workflow before collection.
Once the task is accepted, record the layout-related lesson through
[task-driven self-improvement](task-self-improvement.md), including the robot,
geometry and task conditions under which it applies.
