---
name: geniesim-asset-search
description: Find and verify assets in a local GenieSim asset library for scene generation, using optional Qwen3-VL video embeddings and a directory-index fallback.
---

# GenieSim asset search

Use the Python CLI in this project to return a real `asset_id` and USD path for each requested object. Inspect the discovered asset directory with `python -m scene_agent_assets assets-root`: tools use explicit paths, `GENIESIM_ASSETS`, optional CLI configuration, the installed asset package, then relative `assets/` layouts. Set `GENIESIM_ASSETS` when an override is needed. Keep the catalog and vector DB in a writable cache outside the asset tree. Read [tool commands](references/commands.md) when running the tools.

For a request such as "a yellow building block", extract the object category, color, shape, size, and any required interaction properties. Translate useful search terms into English for the current VL model and file descriptions.

1. If a VL vector DB exists, search it first. The default returns up to 20 candidates. Check its `coverage` for the relevant folder. Review at least three plausible candidates when available, including their actual snapshots or renders and `inspect-usd` dimensions; never accept the first result solely because it ranks first. Reject candidates that do not meet the stated attributes.
2. If the VL DB is absent, its coverage is incomplete, or verified candidates do not fit, use `catalog-search` to locate likely category folders. Browse the small table for that folder, run `render` to make a contact sheet when existing images are insufficient, and use `inspect-usd` for geometry and size. Review at least three candidates when available, and search neighboring folders or synonyms if the first category is empty.
3. Return the chosen `asset_id`, USD path, and concrete evidence from its description, image, or USD inspection. If no asset fits, say which folders and candidates were checked. Do not invent an asset ID.

`vl-build` is optional and uses the bundled Qwen3-VL multi-view **video** embedding implementation. Users may build all assets or only a folder. Explain that first-time GPU model setup and full indexing can take a long time; the catalog route remains usable without them.
