# Workflow artifacts and handoffs

This checklist records progress; it does not replace the user's requirements
or impose every stage on a scene-only or task-only request. Update the current
phase and evidence as work proceeds. A rendered initial scene, a returned task
boolean, user acceptance and a verified recording are distinct facts.

| Phase | Input | Reviewable output | Decision/evidence needed to advance |
| --- | --- | --- | --- |
| Scene preparation | User description or reusable accepted scene, similar scenes, verified assets | One foreground scene, recipe/patch, metadata, front/side/top renders | Scene accepted; static USD and layout checks reported |
| Variant planning | Accepted scene | Dimensions, ranges, fixed relations, count, base parameters | User confirms variants; 1000 refers to scene instances |
| Orchestration (optional) | Prepared scene contract, comparable task implementations | `data_gen.py`, robot/task/camera metadata, physical checkers | Static contracts checked; ready for fixed-scene execution |
| Action demonstration (optional) | One fixed scene and execution runtime | Direct recorded video, logs, status, final-state checks | Physical/recording checks reported; behavior accepted |
| Scene batch (optional) | Accepted scene and approved dimensions; validated base behavior if execution is requested | Numeric scenes, generation/validation reports, renders | Approved count; representative/boundary variants inspected, and executed when in scope |
| Collection planning (optional) | Validated scenes/task for requested episodes | Scene coverage, episode counts, noise/cameras, retries, resource budget | User-approved collection scope, separate from scene count |
| Collection (optional) | Fixed approved task/library and runtime | Episodes, per-attempt index, metrics, logs, video verification | Measured attempts/outcomes/coverage and retained failure evidence |
| Scene revision | User feedback or motion/layout evidence, approved concrete changes | Updated recipe/patch, USD, metadata, previews and affected task contract | Revised scene accepted; affected motion retested when execution is in scope |
| Closeout | User acceptance and actual evidence | Scene artifacts, optional task code or demonstrated task; applicable example/lesson updates | Actual validation level and remaining limitations recorded |

Scene-only requests finish after scene preparation and any approved scene
batch, without task code or collection installation. Code-only requests add
orchestration and static checks, retaining unverified physical status. A robot
demonstration requires the optional runtime; bulk episodes remain a separate
request. Scene revisions return to scene validation/rendered review, then to
affected task phases. Reuse unchanged approvals.

## Command routing

Run authoring Python commands from a writable task project with the scene-agent
package installed. Run collection CLI commands on the host with the configured
main CLI/collection module and runtime image. Use absolute task/output paths
when switching working directories. The skill/resource library can be located
with `python -m geniesim_scene_agent resources` after wheel installation.

```bash
# Scene authoring: select a similar layout.
python -m scene_agent_scene.similar "user scene description" --task-root "$EXAMPLE_ROOT" --limit 5
# Optional orchestration: compare matching task implementations.
python -m scene_agent_scene.find_example search "user action description" --limit 5

# Authoring: a prepared DSL recipe; use --patch for hybrid or --mode direct as appropriate.
python -m scene_agent_scene build --mode dsl --recipe "$RECIPE" \
  --scene-id "$TASK_NAME" --output-root "$DRAFT_ROOT" \
  --assets-root "$ASSETS_ROOT" --isaac-python "$ISAACSIM_PYTHON"
python -m scene_agent_scene validate "$DRAFT_ROOT/$TASK_NAME/0" --require-renders

# Optional robot execution only: TASK_ROOT contains the prepared task metadata/code and numeric scene folders.
geniesim data-collection llm show "$TASK_NAME" --task-root "$TASK_ROOT"
geniesim data-collection run "$TASK_NAME" \
  --task-root "$TASK_ROOT" --assets-root "$ASSETS_ROOT" \
  --scene 0 --run-times 1 --render --output-dir "$DEMO_RUN"

# Authoring: only after the dimensions and count have been approved.
python -m scene_agent_scene build --mode dsl --recipe "$RECIPE" \
  --scene-id "$TASK_NAME" --output-root "$VARIANT_ROOT" \
  --generalization "$APPROVED_SPEC" --count "$APPROVED_SCENE_COUNT" \
  --seed "$APPROVED_SEED" --assets-root "$ASSETS_ROOT" \
  --isaac-python "$ISAACSIM_PYTHON"
```

Set these variables to actual prepared files/approved values; they are not
automatic defaults. Copy the tested code/metadata into the generated variant
task directory when orchestration is requested, before any execution. Keep collection outputs outside both task
roots. Use fresh output roots and retain local USD dependencies when staging
task projects. Runtime binding rewrites the copied scene asset references.

## Coverage-driven collection

For one episode per approved scene, enumerate the IDs explicitly. Each
`geniesim data-collection run` selects one ID, uses `--run-times 1 --render`,
and writes a new output run. For several episodes per scene, use the approved
episode count when the runtime policy permits it; otherwise use separate
single runs. Store one row per attempt with task/scene ID, task revision,
runtime/image, noise controls, output run, episode path, return/metric status,
video checks, and any first failing phase. Record retries instead of replacing
failed attempts with later successes.

The current CLI copies a whole task for every run. For many scene instances,
use staged single-scene input libraries when appropriate, preserving task-level
files and required local USD/script dependencies. Include render/input-copy
storage and per-episode server startup in the budget. Do not hardlink writable
USD files back to the approved scene library.

Random sampling is appropriate only when explicitly chosen in the collection
plan. Do not report full coverage because the number of runs equals the
number of scenes. A batch dispatcher should serialize these commands unless
resource-safe concurrency was deliberately configured; the current CLI is a
single-task serial runner, not an automatic all-scenes scheduler.

## Approval and evidence notes

Retain the user's decisions in the task project or session handoff: accepted
scene revision, approved background choice, dimension ranges/count, accepted
behavior video/run, specific confirmed layout corrections, and collection
scope. Save actual preview/render/static/runtime/recording evidence separately.
Record the current phase and next unmet condition. Prior approvals remain
valid for their exact scope; a changed layout/dimension/task definition needs
review of that concrete change. Do not use this checklist to request another
approval for an action already authorized in the session.
