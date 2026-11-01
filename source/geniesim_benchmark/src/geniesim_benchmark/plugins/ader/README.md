# 🧠 Benchmark Evaluation Framework

Use ADER (Action Domain Evaluation Rule) to configure task evaluation.

## Overview

ADER expresses task conditions as a tree of actions. Sequencing primitives such
as `ActionList`, `ActionSetWaitAny` and `ActionSetWaitAll` combine predicates such
as `Ontop`, `Inside` and `PickUpOnGripper`. The runtime evaluates the tree against
simulator state and reports success, cancellation or progress.

## Authoring a configuration

Each scene instance supplies `problems.json`. The parser
([`action/action_parsing.py`](action/action_parsing.py) → `parse_action`) reads
`problems["problem<i>"]["Acts"][0]`. Use that parser and the task configuration
for supported arguments and condition values.

Write the configuration directly or generate it with the LLM evaluator:

```bash
python -m geniesim_benchmark.evaluator.generators.eval_gen --scene_dir <llm_task/<task>/<i>>
```

Generation uses [`ADER_SYSTEM_PROMPT.md`](../../evaluator/prompts/ADER_SYSTEM_PROMPT.md),
`PROMPT_EVAL.txt` and the scene's `instructions.json`. Register new predicates in
`action_parsing.py` and update the generator's specification alongside their API.

## Example structure

```json
{
  "Acts": [
    {
      "ActionList": [
        {"PickUpOnGripper": "beverage_bottle_002|right_gripper"},
        {"Ontop": "beverage_bottle_002|table_000"}
      ]
    }
  ],
  "Init": [],
  "Objects": [],
  "Problem": "example_task"
}
```

## Control and composite actions

| Action | Purpose | Value form |
|---|---|---|
| ActionList | Run children in order | `[ {…}, {…} ]` |
| ActionSetWaitAny | Finish when any child finishes; stop the others | `[ {…}, {…} ]` |
| ActionSetWaitAll | Finish when all children finish | `[ {…}, {…} ]` |
| ActionSetWaitSome_N | Finish after the number of children named by the suffix | `[ {…}, {…} ]` |
| ActionWaitForTime | Wait without blocking the evaluation loop | Number of seconds |
| Timeout | Cancel after the configured elapsed time | Number of seconds |
| StepOut | Cancel after the configured step limit | Number of steps |

Pair cancellation guards with success predicates inside `ActionSetWaitAny` to
express a task that can finish or time out. Task files supply the limits.

## Predicate arguments

Arguments are pipe-separated unless a JSON object is specified. Brackets in
signatures mark optional arguments. Spatial distances are metres unless noted.

### Spatial relationships

| Predicate | Syntax | Purpose |
|---|---|---|
| Ontop | `"active_obj\|passive_obj"` | Check the active object's placement above the reference surface |
| Inside | `"active_obj\|passive_obj\|scale"` | Check the active center against the reference's scaled AABB |
| InBBox | `"obj_id\|cx,cy,cz\|lx,ly,lz"` | Check a world-axis-aligned box defined by center and full side lengths |
| Cover | `"active_obj\|passive_obj"` | Check the active object's coverage of the reference |
| Stack | `"[a,b,c]\|[x,y]"` | Check alignment of the listed objects relative to the first |
| OnShelf | `"obj\|target\|[xmin,xmax,ymin,ymax,zmin,zmax]\|height"` | Check the object in a target-relative offset region with a Z height adjustment |
| Onfloor | `"obj\|ref_z"` | Cancel when the object reaches the configured floor region |
| Upright | `"obj\|tilt_deg[\|allow_flipped]"` | Compare the object's local +Y axis with world +Z; optionally accept the flipped direction |
| RelativePosition | `"objA\|objB\|relation[\|threshold]"` | Compare objects in the robot body frame using `leftof`, `rightof`, `topof`, `bottomof` or `aligned_x/y/z` |

### Grasping and manipulation

| Predicate | Syntax | Purpose |
|---|---|---|
| PickUpOnGripper | `"object\|gripper_id"` | Check lift relative to the first observed object height and proximity to the gripper reference |
| StableGrasp | `"object\|gripper_id[\|dist[\|pos_diff[\|rot_diff_rad]]]"` | Check relative object/gripper motion; angular differences use radians |
| Follow | `"object\|[x,y,z]\|gripper_id"` | Check the gripper center in a box around the object |
| Approach | `"x\|y\|z"` | Check the right gripper against a world-space target point |
| GripperPassing | `"obj_prim\|reverse"` | Check handover in the selected direction between grippers |
| LiftUp | `"object\|lift_threshold"` | Check lift relative to the object's first observed height |

`PickUpOnGripper` uses the robot configuration's `end_effector_prim_path` or its
family fallback as the gripper reference. `PickUpOnGripper` and `Follow` accept
comma-separated object candidates. Gripper identifiers containing `right`
select the right hand; other identifiers select the left.

### Joint, state and navigation

| Predicate | Syntax | Purpose |
|---|---|---|
| PushPull | `"obj\|min\|max[\|joint_index]"` | Check the selected prismatic joint within the configured interval |
| TriggerAction | `"prim_path\|expected"` | Compare `get_trigger_action(prim_path)` with the expected value |
| ChassisAtTarget | `"[x,y,yaw]\|[x_th,y_th,yaw_th]"` | Check a planar chassis target; yaw is in degrees |

### Other predicates

| Predicate | Syntax | Purpose |
|---|---|---|
| FluidInside | `"container\|object_info_dir"` | Check fluid-particle occupancy of the container AABB from the object metadata |
| CheckParticleInBBox | `"threshold\|xmin,ymin,zmin,xmax,ymax,zmax"` | Compare the particle population in the configured box with its limit |
| CheckStainClean | `"stain_prim\|threshold"` | Compare the visible stain population with its limit |
| PlaceOnRivet | `"active\|passive\|rx,ry,rz\|qw,qx,qy,qz\|[xy_tol]\|[z_tol]\|[orient_tol]\|[still_thresh]\|[still_steps]"` | Check target-relative placement and stillness |
| VLM | `"task_id\|interval"` | Evaluate the configured task description at the selected update interval |

For `PlaceOnRivet`, positional tolerances use metres, angular tolerances use
radians, and speed uses metres per simulation second. `still_steps` counts
checker evaluations. The quaternion order is `wxyz`.

## Combining predicates

`MixedRules` accepts a JSON object containing named sub-checkers and their
parameters:

```json
{
  "MixedRules": {
    "rules": [
      {"name": "Ontop", "params": "cup|tray"},
      {"name": "RelativePosition", "params": "cup|bowl|leftof"}
    ],
    "mode": "and"
  }
}
```

OR mode permits partial credit; AND mode requires all configured sub-checkers.
Supported names include `Inside`, `PushPull`, `Upright`, `RelativePosition`,
`Ontop`, `Onfloor`, `Cover`, `LiftUp`, `InBBox`, `Stack` and `StableGrasp`.
Use the task configuration for its selected mode and condition settings.

## Argument conventions

- Use object IDs directly in pipe arguments. List arguments are list literals;
  numeric and boolean arguments remain plain text within the pipe string.
- Supported object arguments can use `{@placeholder_name}` to reference a value
  resolved at runtime, such as the object selected by an earlier pickup.
- `Timeout`, `StepOut` and `Onfloor` cancel the evaluation; they are distinct from
  success predicates.
- JSON key `Timeout` maps to class `TimeOut`; `RelativePosition` maps to
  `RelativePositionChecker`.
