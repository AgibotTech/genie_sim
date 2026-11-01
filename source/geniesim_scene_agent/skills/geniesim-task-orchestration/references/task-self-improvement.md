# User-guided task self-improvement

Close each task orchestration by turning its execution experience and user
feedback into better authoring instructions and reusable examples. The updated
skills and library become the starting point of the next task. This persistent
cycle is the intended self-improvement mechanism for this agent.

## Acceptance and evidence

The user defines the desired task, acceptable scene, quality and tradeoffs.
Use their acceptance feedback to decide whether the result meets those needs.
Present the concrete task code, scene preview and actual validation results
before asking for acceptance when it has not already been given.

Record separately:

- What the user requested and accepted, including relevant corrections.
- What was checked: source/static contracts, geometry/render, mock execution,
  physical task execution and recording, with actual outcomes.
- What remains untested or unresolved, including hypotheses about failures.

User acceptance of a design does not imply an unperformed simulator run passed.
Use the accepted level of verification when describing the example. Keep the
physical checker faithful to the task; changing a checker or requirement to
accept a different result needs the user's agreement and a clear description.
No automatic score or presumed universal quality threshold replaces this
acceptance process.

## Summarize the task experience

At the end of a task attempt, review its material failures, corrections and
user feedback. After acceptance, complete the following closeout as part of
the workflow. Keep the useful experience compact; ordinary uneventful steps
do not need a lesson. A diagnosed failed attempt can contribute a factual
lesson while its unfinished task remains a draft.

For each meaningful lesson, capture:

| Field | Content |
| --- | --- |
| Context | Action, robot profile, relevant geometry/asset or interface, and applicable frame conventions. |
| Symptom | Concrete failure or user-observed shortcoming; identify the phase. |
| Attempts | Important alternatives tried and why they failed or were rejected. |
| Cause | Supported explanation; label unconfirmed explanations as hypotheses. |
| Working change | The accepted correction and its scope, including an approved layout change when applicable. |
| Evidence | Actual observations/checks and the user acceptance scope; include remaining limits. |
| Next-time guidance | A specific condition -> action rule, diagnostic shortcut or reference example. |

Summarize this in the task's existing English guide, or create
`examples/tasks/<task_name>/lessons.md` when no suitable guide exists. The
generated task README is refreshed by `find_example tag`; keep durable lessons
in a separate inventoried guide rather than manually appending them to that
generated file. Keep logs, temporary renders and failed drafts in `.cache/`.
Published guides use repository-relative references and externally configured
assets; omit private platform links, identities and machine-specific paths.

## Update the relevant skill

Read the current guidance before editing it. Find the closest existing rule,
then merge, refine or replace it instead of appending a duplicate after every
task. A single observation supports a conditional lesson, not a universal rule.

| Lesson scope | Destination |
| --- | --- |
| Common authoring decision | Orchestration SKILL workflow or `execution-tips.md`. |
| Motion-specific method | `motion-patterns.md` or the relevant focused reference. |
| Actual API behavior or parameter limitation | `robot_api_reference.md` / `config-extra-params.md`, verified against compatible implementation. |
| Grasp annotations or geometry-derived selection | `grasp-selection.md`. |
| Reachability-driven layout correction | `ik-recovery.md` and relevant scene-generation guidance. |
| Task- or asset-specific constants, offsets or workarounds | Task guide/`lessons.md`, linked from a relevant reference when useful. |
| Unconfirmed hypothesis | Draft notes with its missing evidence; avoid presenting it as established guidance. |

Keep the user-defined task constraints and actual applicability conditions.
Correct outdated advice and remove conflicts. For example, an approved button
press may establish a scoped planner-exclusion recipe; it does not establish
that every contact task should disable physical colliders. If a task-specific
tool/recipe change is retained, exercise its observable behavior before treating
it as a reusable fix. Report a reusable runtime bug rather than documenting a
broken branch as supported.

## Retain an accepted task as an example

Use `examples/tasks/<task_name>/` with a stable English snake_case
identifier. Adapt an existing matching directory/manifest schema. Include:

- `data_gen.py` with the `execute_task` entrypoint and English comments.
- `task_info.json`, meaningful task instructions and the actual robot profile.
- One or a few representative **approved** scene directories containing
  `scene.usda` and `scene_info.json`, plus required scene-local scripts or layers.
  Select from already generated scenes; do not create a batch or pad the library
  to a fixed scene count for this closeout.
- A task guide or `lessons.md` with the acceptance scope, actual validation level,
  reusable lessons and asset/geometry limitations. Keep the required rendering
  evidence when it is part of the accepted scene deliverable.
- `object_urls.json` or other task-local files when the task requires them.
- `manifest.json` with `schema_version`, `task_name`, `english_task_name`,
  `task_family`, `scene_count`, root `files` (`path` and SHA-256), and `scenes`
  (`scene_id` and relative-path-to-SHA-256 `files` mapping). Inventory the new
  guide and required local dependencies; exclude the manifest's own hash.

Keep task identity/title consistent between directory, manifest, task config
and scene metadata. Retain task-specific fields such as `target_letter`.
Use a supported family from the existing index. Bind asset references through
`examples/assets`; retain local dependencies and use portable relative paths.
Do not copy the external asset library into the example. Avoid overwriting an
existing task with a different contract under the same identifier.

Static/physical validation confirms the documented evidence, while the user
determines acceptance. An accepted example may state that physical execution
was unavailable; do not present it as an executed success. Keep unaccepted or
incomplete candidates outside the completed example library.

## Refresh, verify and report

Update affected file hashes **after** editing code, guides or scenes. Refresh
the generated interface tags, task README/overview and retrieval family tables:

```bash
python3 -m scene_agent_scene.find_example tag --task-root examples/tasks
python3 -m scene_agent_scene.find_example rebuild --task-root examples/tasks
python3 examples/validate.py
```

`tag` refreshes the generated README hash, but does not recompute arbitrary task
or scene file hashes. Update those in the manifest explicitly. Read
[example usage](../../../examples/README.md) for the external asset binding
needed by scene dependency validation. Exercise a relevant `find_example search`
and check that the retained task can be found and its lesson guide can be read.
When adding an example, extend `example-review.md` with a concise applicable
recipe/limitation note and refresh any manually stated library totals. Keep
historical review scope explicit when retaining an earlier source-review snapshot.

Verify English skill text, links and consistency. Run relevant tool tests only
when tool behavior changed; perform physical/recording checks at the level
actually available. Report which lessons changed which references and which
task/scenes were retained, along with outstanding limitations.

Keep skill and example-library changes within the current task's authorized
scope. Search the updated examples on later orchestration tasks and revise
lessons when new evidence changes their applicability.
