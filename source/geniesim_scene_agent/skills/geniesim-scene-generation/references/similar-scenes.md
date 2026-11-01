# Find comparable existing scenes first

Before designing each new scene, search an existing task library. This project bundles the search code; the library is optional reference data, not a Python dependency. Set `GENIESIM_SCENE_LIBRARY` to one or more task roots separated by the OS path separator, or pass repeated `--task-root` flags. A task root contains task directories; each task must have at least one child directory with `scene.usda` and `scene_info.json`.

```bash
python3 -m scene_agent_scene.similar "fruit on each shelf level" \
  --task-root /path/to/llm_tasks_data_collection --limit 5
```

Search the user's Chinese wording and useful English synonyms separately if the first results are weak. The search ranks task names, task descriptions, asset IDs, and `layout.keywords`. It returns one representative scene per task and suppresses near-identical robot variants. This is a lexical shortlist, not geometric proof of similarity; if results are empty, report that and proceed with asset search.

For each promising result, read the returned `scene_info.json` and, when available, its `task_info.json`, `LLM_RESULT_*.py`, `data_gen.py`, and preview images. Compare object roles, asset IDs, dimensions, support surfaces, robot-relative placement, and generalization rules. Render or inspect the USD if the visual layout is important or the snapshots are missing. Cite the examples used in the scene plan. Reuse validated ideas and asset IDs, but write a new recipe for the user's actual constraints; do not copy unreviewed poses or assume a source scene is physically correct.

The representative instance is only for discovery. If a reference task varies asset choices or layout across instances, inspect more than one instance before adopting its approach.
