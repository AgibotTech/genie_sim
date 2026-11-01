---
name: geniesim-scene-generation
description: Design, generate, preview, validate, and optionally generalize GenieSim scenes using the existing DSL, a USD patch, or direct Python USD authoring.
---

# GenieSim scene generation

Scene generation is a complete standalone workflow. Deliver accepted scene
artifacts and any approved variants without requiring task code, robot/task
metadata, or a data-collection module/image. Rendering needs a configured Isaac
Sim environment or the independent scene-agent image. For orchestration
requests, scene preparation comes first; inspect/reuse an accepted existing
scene when suitable instead of regenerating it.

For a request spanning scene design, orchestration and data collection, follow
[the complete workflow](../geniesim-scene-to-data/SKILL.md) to retain approvals
and keep scene instance counts separate from episode counts.

When the request includes robot task orchestration in `data_gen.py`, use the
[find-example skill](../find-example/SKILL.md) to compare task flow, RobotAPI
interfaces, and object keyword contracts before implementing the task.
Use [task orchestration](../geniesim-task-orchestration/SKILL.md) for the
RobotAPI implementation and [task validation](../geniesim-task-validation/SKILL.md)
for physical execution and recording checks.

Before each new scene design, search for comparable existing scenes using the bundled Python tool and review the best matches. Follow [similar-scene search](references/similar-scenes.md); record which examples influenced the plan and what must differ. Then use the asset-search skill to verify assets before writing a scene. Keep draft output outside the GenieSim source asset tree. Read [tool commands and recipe protocol](references/commands.md) before building.

## Design with the user

Start with one foreground scene. Include only requested objects and necessary support surfaces; do not add a room, floor, backdrop, decor, or environment geometry unless the user explicitly requests background. Rendering cameras and lights are temporary and must not be saved in `scene.usda`.

Generate an initial scene, inspect the USD and mesh bounds, render front, side, and top views in Isaac Sim, and show the images to the user. Ask for feedback on the actual scene and revise as needed. For articulated objects, inspect joint paths and limits before patching. Verify a requested open or closed state visually in Isaac Sim; authored joint attributes alone are insufficient evidence.

If the scene will feed a robot task, read the [RobotAPI quick reference](references/robot_api.md) before fixing object roles and keywords. Check that task-relevant asset IDs resolve in the target `ASSETS_INDEX`, that composed object prims have the expected `/World/Objects/<id>/entity` paths, and that grasp/place targets are reachable. Scene rendering alone does not establish task success; validate task execution in the configured robot runtime after task files are available.

During orchestration, feedback, contact requirements, clearance or reach may
require a scene adjustment. Review concrete changes with the user when they
alter an accepted layout, reusing authorization already given. Update the
recipe/patch, USD and metadata together, preserve object-role IDs, validate and
show the revised scene, then retest affected phases when execution is in scope.
Use a fresh draft output location so the accepted revision remains reviewable.

When downstream orchestration repeatedly fails IK, apply
[IK recovery](../geniesim-task-orchestration/references/ik-recovery.md). Propose
specific object/support/base changes using reach and clearance evidence, obtain
user confirmation before editing the layout, and preview/retest the revised
single scene. Keep USD, metadata and recipe parameters consistent.

For shelves, cabinets, or multi-level supports, read [surface placement guidance](references/surfaces.md). Use reviewed `layers.json` data when available, transform its local corners into the composed scene frame, and inspect marker renders. If metadata is absent, inspect upward horizontal mesh faces and confirm the real usable levels visually. Use the actual asset bottom and support polygon when placing objects; the overall furniture bounding box does not identify individual levels.

Prefer the GenieSim DSL for ordinary layouts. Use a patch hook after DSL construction for properties it cannot set, such as a microwave hinge. Use direct Python authoring when the DSL cannot represent the scene. All routes use the same `SceneBuilder`, output layout, object naming, and validation. Patch only the generated stage, never a source asset.

## Write and run a DSL recipe

Use the bundled `helper` API and asset IDs from the external assets directory. Define `root_scene()` and return a `Shape`. A minimal table and block scene is in [table_block_dsl.py](examples/table_block_dsl.py):

```python
from helper import *

@register()
def root_scene(params=None) -> Shape:
    params = params or {}
    table = library_call("usd", oid="table_000", keywords=["table"])
    block = library_call("usd", oid="benchmark_building_blocks_007", keywords=["block"])
    top_z = get_object_info(table)["max"][2]
    block = transform_shape(block, translation_matrix((params.get("block_x", 0.0), 0.0, top_z)))
    return concat_shapes(table, block)
```

`library_call("usd", oid=...)` loads an asset ID, `translation_matrix` uses meters, `rotation_matrix` uses radians, and Z is up. For shelves and cabinets, use the support-surface tool instead of treating the whole furniture bound as a level. Run the recipe with `python3 -m scene_agent_scene build --mode dsl --recipe <recipe.py> --assets-root <assets> --scene-id <name> --output-root <output> --isaac-python <python.sh>`. `--assets-root` may be omitted to use automatic CLI/package/relative asset discovery (see the [discovery order](../../README.md#external-asset-discovery)). The only external project directory required is `<assets>`; the DSL runtime, rendering code, and asset-search implementation live in this workspace. Read [DSL authoring details](references/dsl.md) for transforms, parameters, preview, and validation commands.

## After scene approval

Only after the user approves the single scene, ask whether to generate variants. If yes, propose concrete dimensions, ranges, fixed relationships, and a count; 1000 total instances including instance `0` is a useful default, not an automatic choice. Confirm them with the user before running a batch. Show representative variants when the proposed dimensions materially change the appearance.

Each instance goes in `<output_root>/<scene_id>/<numeric_id>/` and contains `scene.usda`, `scene_info.json`, `generation.json`, `validation.json`, and `renders/{front,side,top}.png`. The scene root also has `generalization.json` and `batch_report.json`. Object IDs denote roles and stay stable across variants. Use deterministic seeds and inspect failures; never call an unrendered or invalid scene complete. These are scene instances, not collected robot episodes. Run task integration checks in a compatible robot runtime when appropriate.
