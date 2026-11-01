# Scene agent skills

| Skill | Workflow |
| --- | --- |
| [geniesim-asset-search](geniesim-asset-search/SKILL.md) | Find several asset candidates and verify images/geometry |
| [geniesim-scene-generation](geniesim-scene-generation/SKILL.md) | Design, render, validate, and approve a foreground scene before batches |
| [find-example](find-example/SKILL.md) | Compare task examples by actions, interfaces, and scene contracts |
| [geniesim-task-orchestration](geniesim-task-orchestration/SKILL.md) | Author data_gen.py, select grasps, recover from IK failures, retain accepted lessons |
| [geniesim-task-validation](geniesim-task-validation/SKILL.md) | Check static contracts and physical task/recording evidence |
| [geniesim-data-collection](geniesim-data-collection/SKILL.md) | Execute tasks, record camera videos directly, verify episodes, optionally replay |
| [geniesim-scene-to-data](geniesim-scene-to-data/SKILL.md) | Route scene-first requests through optional orchestration, execution and collection; track revisions and approvals |

Python tools, references, and examples are bundled in this module and its wheel.
See [the agent guide](../AGENTS.md) and [the README](../README.md) for entry points.

The optional image build uses `python3 docker/build_image.py` from the module
root. Shared presets, custom repository URLs, and direct BuildKit commands are
documented in [the container guide](../docker/README.md#build).

Scene generation is standalone: use asset search and scene generation, then
finish with the accepted scene and any approved variants. No collection module
or image is needed; use the configured Isaac Sim renderer or scene-agent image.
Pip installs tools/resources only. Container builds and image acquisition are
separate explicit steps; they are not package-install hooks.
For orchestration, prepare or reuse the scene first and apply find-example and
task orchestration. Code-only delivery needs static validation and an explicit
unverified physical status. Add data collection only for requested robot
execution/demonstrations or episodes. During task development, scene adjustments
return to scene generation, consistent metadata and rendered review, then to
affected motion checks when execution is requested.

For requests covering the full pipeline, start with `geniesim-scene-to-data`.
Its [phase checklist](geniesim-scene-to-data/references/phase-checklist.md)
records handoffs and decisions. The self-contained
[collection CLI reference](geniesim-data-collection/references/cli.md) describes
execution, direct recording, outputs, coverage and optional replay.
