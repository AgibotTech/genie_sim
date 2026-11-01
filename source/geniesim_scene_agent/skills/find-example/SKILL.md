---
name: find-example
description: Find and compare existing GenieSim task examples by scene type, action, RobotAPI calls, and object keywords before authoring or changing data_gen.py.
---

# Find task examples

Use this skill before designing the orchestration in `data_gen.py`. Start with
the [task family tables](references/index.md), which describe every retained
task in `examples/tasks/`. The table rows link to the actual code and list
scene/action tags, statically observed interfaces, configuration classes, and
a compact sample of literal scene keywords.
Read the [tag vocabulary](references/tags.md) when choosing scene, action,
lookup, motion, or validation terms.

## Find a shortlist

Translate the user's intent into English action, object, scene, and control
terms. Include requirements such as a held container, articulated door,
sequence observation, placement relative to another object, or contact motion.
Keep the user's requested robot profile when it is known.

Read the relevant family table and select several plausible examples. Use the
Python search to rank its rows; it returns up to five different implementations
by default, subject to matching relevance:

```bash
python3 -m scene_agent_scene.find_example search "open microwave door" --limit 5
python3 -m scene_agent_scene.find_example search "fruit shelf move middle layer" --robot G2
python3 -m scene_agent_scene.find_example search "hold bowl place fruit" --tag held_container
python3 -m scene_agent_scene.find_example search "write letter pen target_letter" --api follow_trajectory
```

`--api` and `--tag` are exact filters and can be repeated. Matching is lexical;
try action synonyms or a different family when the first shortlist misses the
user's intended flow. If a strict filter returns too few results, relax it and
compare the remaining requirements explicitly. There may be only one writing
task family; its 26 letter scenes share one implementation.

Prefer examples with a compatible action sequence, object lookup contract,
motion constraints, and physical success check. Use several examples when one
provides the grasp approach and another provides the target pose or checker.
Scene appearance alone is insufficient for choosing orchestration code.

## Read the implementations before adapting

For each serious candidate, read:

- `README.md` and `manifest.json` interface tags: observed RobotAPI and client
  calls, configuration classes, and links to every scene instance.
- `data_gen.py`: object selection, phases, configuration arguments, checkers,
  recovery, and `execute_task` return conditions.
- `task_info.json`: robot profile, initial pose, task description, and metrics.
- The selected `scene_info.json`: object IDs, asset IDs, keywords, poses, and
  any task-specific fields such as `target_letter`.
- The composed `scene.usda`: actual prim paths, articulations, geometry, and
  support or containment relationships.
- `manifest.json`: the file inventory, task family, and scene-specific fields.
  Read an accompanying `LLM_RESULT_*.py` when layout logic affects the motion
  recipe.
- Accompanying task documents, when present: review their INPUT, Task flow,
  and OUTPUT sections alongside the current code and metadata.
- `lessons.md`, when present, or the experience section in the task guide:
  user acceptance scope, pitfalls, ineffective attempts, working corrections
  and the actual validation level.

The search returns these paths. Review other instances when target selection
or asset variation matters. Check the [complete RobotAPI parameter reference](../geniesim-task-orchestration/references/robot_api_reference.md)
and use [task orchestration](../geniesim-task-orchestration/SKILL.md) to adapt
the chosen flows. Static
calls in an example need to agree with the configured runtime signatures.

Read the selected task's [source review note](../geniesim-task-orchestration/references/example-review.md)
and the matching [execution tips](../geniesim-task-orchestration/references/execution-tips.md),
especially deliberate-contact collision scope, effective extras and final
checker limitations. The example is a reference to adapt, not an instruction
to preserve known weak checks or leaked temporary state.

Before writing code, summarize the best few candidates: what matches, what
differs, and which parts to reuse. Map object roles and literal keyword filters
to the new scene; derive target poses from current geometry. Preserve explicit
quaternion conventions and check physical completion before returning `True`.
For writing, the requested letter comes from `scene_info.target_letter`.

Use the [scene-generation skill](../geniesim-scene-generation/SKILL.md) when
creating or changing the layout, rendering it, or validating USD and mesh
relationships. Exercise adapted `data_gen.py` code in the configured robot
runtime.

## Maintain the tables

After adding, removing, or changing example tasks, refresh the tables:

```bash
python3 -m scene_agent_scene.find_example rebuild --task-root examples/tasks
python3 -m scene_agent_scene.find_example tag --task-root examples/tasks
```

The builder reads local code and scene metadata without importing RobotAPI or
Isaac Sim. `rebuild` writes English Markdown tables and produces no JSON catalog.
`tag` refreshes each task's generated README, interface/configuration/scene-action
tags in its existing manifest, the README file inventory hash, and the task
interface table in the example library's existing parent README. Interface
tags are static observations, not runtime coverage or successful execution.

For a newly accepted task, follow the
[self-improvement closeout](../geniesim-task-orchestration/references/task-self-improvement.md)
to retain representative approved scenes and durable lessons, maintain file
hashes, and refresh these tables. Library maintenance does not authorize a
new scene-variant batch.
