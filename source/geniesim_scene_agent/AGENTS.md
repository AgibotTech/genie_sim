# geniesim_scene_agent — Agent Development Guide

Release version: the local `VERSION` links to the repository root `VERSION`
(**3.5.0**). Wheel metadata and runtime `__version__` use that release value.

> Maintenance contract: update this guide, the README, and the skill index when
> package entry points, output contracts, dependencies, or skill workflows change.

## 1. What this is

English agent skills and Python tools for asset search, foreground scene design,
DSL/direct/hybrid USD generation, example retrieval, task orchestration, and
optional collection execution workflows with direct video demonstrations.
Scene generation is a complete standalone deliverable.
The only external project data required for scene generation is the asset tree.
Isaac Sim rendering and robot task execution use their configured runtimes.

## 2. Layout

| Path | Responsibility |
| --- | --- |
| `src/geniesim_scene_agent/` | Lightweight package identity, CLI, and resource location |
| `scene_agent_assets/` | Asset catalogs, optional video embeddings, rendering, geometry |
| `scene_agent_scene/` | Scene builder, DSL, validation, support surfaces, example search |
| `skills/` | English workflows and complete API references |
| `examples/` | Task library, binding helper, and validator |
| `docker/` | Official Isaac Sim 6.2 image and isolated authoring Python |

## 3. CLI and installation

Install this optional peer from a monorepo root with
`python -m pip install -e 'source/geniesim_scene_agent[scene]'`.
The umbrella extra is `[scene_agent]`; default bootstrap does not install it.
`geniesim deploy geniesim_scene_agent` builds its wheel after editable installation.
The README's [monorepo quick start](README.md#monorepo-quick-start) includes
scene-only installation, resource checks, and optional GPU retrieval.
[Task execution and collection](README.md#optional-task-execution-and-data-collection)
are separate opt-in steps for host CLI/path setup and the runtime image.
Pip installs Python dependencies/resources only; it does not pull/build images
or start containers. Deploy builds a wheel only. Collection locator installation
also leaves its runtime image separate. Pip bundles skills but does not automatically register them in an agent's global skill directory.

| Command | Purpose |
| --- | --- |
| `python -m geniesim_scene_agent assets-root` | Locate external assets through CLI/package/relative discovery |
| `python -m geniesim_scene_agent resources` | Locate the installed skill/example library |
| `python -m geniesim_scene_agent build ...` | Generate a scene with the scene CLI |
| `python -m scene_agent_assets ...` | Index, search, render, or inspect assets |
| `python -m scene_agent_scene ...` | Generate, render, validate, or inspect supports |
| `python -m scene_agent_scene.find_example search ...` | Find task orchestration references |
| `python -m scene_agent_scene.similar ...` | Find comparable scene layouts |
| `geniesim data-collection run <task> ... --render` | Execute and directly record camera videos with the configured collection runtime |
| `geniesim data-collection replay <episode> ...` | Optionally render already recorded trajectories |

The package owns `geniesim-scene-agent`, `scene-assets`, and `scene-scenes`.
The umbrella `geniesim` console script remains owned by `geniesim_cli`.
Read [installation and usage](README.md) and [container commands](docker/README.md).

Build images from the checkout with `python3 docker/build_image.py`, adding
`--target vl`, `--china`/`--mirrors NAME`, or explicit URL build arguments as needed.
`docker/Dockerfile.6.2` uses the same official Isaac Sim `6.2.0.1` base release
as the root Dockerfile. It keeps scene authoring and optional VL dependencies
in their own Python 3.12 venv; they do not replace the bundled SDK stack.
The build-only script reuses sibling CLI sources and the shared mirror presets;
it does not add a runtime dependency on that CLI. BuildKit keeps the module as
the main context and adds `geniesim_mirrors` for the shared helper. Empty APT/pip
arguments inherit parent settings; explicit presets/arguments persist through
`/etc/pip.conf` into both image targets. Preserve the isolated authoring venv and
the bundled Isaac environment when changing mirror support.

## 4. Authoring rules

Route by the requested deliverable. Scene-only requests use asset search and scene generation; they do not require `data_gen.py`, `task_info.json`, or installation of the data-collection module/image. Scene rendering uses a configured Isaac Sim environment or the independent scene-agent image. Task code can be authored with the bundled references without a collection runtime; report physical execution as unverified. Add the collection runtime only for requested robot execution, physical validation, action demonstrations or episodes.

Every orchestration request begins with scene preparation: generate and review the required scene, or inspect and reuse a suitable supplied/accepted scene. During task development, return to scene generation for layout corrections. Keep recipe/patch parameters, USD, metadata and task selectors consistent, validate and preview the revised single scene, and retest affected phases when execution is requested. Obtain confirmation for changes to an accepted layout and specific persistent-IK corrections, reusing authorization already given.

When finding GenieSim assets for a scene, read and apply [the local asset-search skill](skills/geniesim-asset-search/SKILL.md). Its Python tools are in `scene_agent_assets/`; run them from this workspace or install the package. Use `GENIESIM_ASSETS` to point at the external assets directory. Keep generated catalogs, vectors, and renders in `.cache/` or another writable directory outside the assets tree.

When designing or generating GenieSim scenes, read and apply [the local scene-generation skill](skills/geniesim-scene-generation/SKILL.md). Its Python tools are in `scene_agent_scene/`. Search a configured existing-scene library for similar tasks before each new design using `python3 -m scene_agent_scene.similar`; review the returned scene metadata and USD before borrowing a layout. Preview a single foreground scene with the user before discussing batch generalization. Do not start a variant batch before the user has approved the scene, dimensions, ranges, and count.

The bundled reference library is `examples/tasks/`; use `--task-root examples/tasks` when no other library is configured. Read `examples/README.md` for external asset bindings and usage. When borrowing task orchestration, review `data_gen.py`, `task_info.json`, and the chosen scene's `scene_info.json` together. Validate adapted task code in the configured robot runtime when execution is requested; code-only delivery retains its actual static validation level.

Before authoring or modifying `data_gen.py`, read and apply [the find-example skill](skills/find-example/SKILL.md). Use its Markdown task tables and `python3 -m scene_agent_scene.find_example search` to compare several references by action, scene type, RobotAPI calls, and literal object keywords. Review the selected implementations and scene contracts before adapting them.

For automatic collection task orchestration, apply [the task-orchestration skill](skills/geniesim-task-orchestration/SKILL.md) and its complete RobotAPI parameter reference. Inspect asset grasp annotations; use them when compatible with the task, and derive custom grasps from geometry when annotations are missing or do not meet a required contact/direction. Apply [task validation](skills/geniesim-task-validation/SKILL.md) before reporting physical task success. Each bundled task's README and manifest expose interface tags; refresh them with `python3 -m scene_agent_scene.find_example tag` after changing task code.

For execution and user-facing action demonstrations, apply [data collection](skills/geniesim-data-collection/SKILL.md). Prefer one fixed scene and one recorded run with `--render`; replay is optional. Verify physical results, recording conversion and camera videos, then show the actual video to the user. The host CLI manages the collection server/client and isolates task copies. It does not add runtime dependencies to this authoring package.

For requests spanning design through collection, apply [the complete scene-to-data workflow](skills/geniesim-scene-to-data/SKILL.md). Allow completion at scene delivery, code delivery or demonstrated-task delivery according to scope; bulk episodes are optional. Preserve separate scene and behavior acceptance, approved variant dimensions/count, and approved collection coverage/episode counts. Random scene sampling does not guarantee coverage; enumerate explicit scene IDs when every scene must be collected. Default foreground generation and existing batch/layout approval rules continue to apply.

For persistent IK failure, diagnose pose contracts, candidate filters, contact policy and reach, then propose a concrete layout correction when appropriate. Obtain user confirmation of specific object/support/base changes before editing the scene, and preview/retest the revised single scene. Follow [IK recovery](skills/geniesim-task-orchestration/references/ik-recovery.md).

User requirements and acceptance feedback define task quality. After a task is accepted, follow [task-driven self-improvement](skills/geniesim-task-orchestration/references/task-self-improvement.md): summarize pitfalls and working corrections, merge reusable lessons into skills, retain the accepted task and representative approved scenes in the example library, refresh tags/indexes and validate affected artifacts. Record the actual validation level separately from acceptance; keep unfinished tasks as drafts. This maintenance does not authorize a new variant batch or remote publication.

## 5. Skills and workflows

Read [the skill index](skills/README.md) and the appropriate local skill before
using its tools. Scene design, example retrieval, task authoring, and validation
have separate workflows; keep every accepted scene/task contract consistent.

## 6. Troubleshooting

| Symptom | Check |
| --- | --- |
| `pxr` cannot be imported | Install `[scene]` in the authoring environment |
| Isaac Sim render cannot start | Configure `ISAACSIM_PYTHON` and the GPU runtime |
| Asset references do not resolve | Run `assets-root`, then auto-bind the external tree for example USD references |
| Installed example index is missing | Use `resources` and verify the wheel includes its library |
| Repeated IK failure | Apply the task skill's IK recovery workflow |

## 7. Verification

Validate generated scene instances with `python -m scene_agent_scene validate
<instance> --require-renders`. After binding assets, run
`python examples/validate.py` for the example inventory and USD dependencies.
Use [container output checks](docker/README.md#validate-generated-scenes) for
rendered artifacts. Task execution needs the compatible collection runtime and
its physical checks. Keep repository test suites and local
development artifacts out of this module; retain production validation tools.

## 8. Do not

- Write generated data, vector indexes, or previews into the source assets tree.
- Add backgrounds, approve batches, or edit a repeatedly unreachable layout
  without the user decisions required by the skills.
- Add GPU/Isaac Sim imports to the lightweight package identity module.
- Treat static checks or a rendered initial scene as robot task success.
