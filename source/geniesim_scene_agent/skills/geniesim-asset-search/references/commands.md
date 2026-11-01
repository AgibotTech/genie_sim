# Local asset commands

Run these from the `scene_agent` project after installing it, or replace `scene-assets` with `python -m scene_agent_assets`. `GENIESIM_ASSETS` is the external assets directory containing `objects/`.

```bash
export GENIESIM_ASSETS=/path/to/geniesim/assets
scene-assets catalog-build --assets-root "$GENIESIM_ASSETS" --catalog .cache/catalog
scene-assets catalog-search 'building block' --catalog .cache/catalog
scene-assets catalog-search yellow --catalog .cache/catalog --folder robolab/yellow_block
```

Optional GPU path, using the bundled `Qwen3VLEmbedder` on each asset's `merged.mp4`:

```bash
modelscope download --model qwen/Qwen3-VL-Embedding-2B --local_dir .models/Qwen3-VL-Embedding-2B
scene-assets vl-build --catalog .cache/catalog --db .cache/vl.sqlite --folder benchmark/beer
scene-assets vl-build --catalog .cache/catalog --db .cache/vl.sqlite
scene-assets vl-search 'beer can' --catalog .cache/catalog --db .cache/vl.sqlite --folder benchmark/beer
```

The first build may spend substantial time loading the model and processing each video. The command commits vectors asset by asset, so rerunning it skips unchanged videos. The default weights path is `.models/Qwen3-VL-Embedding-2B` in this workspace; set `QWEN3_VL_MODEL` or pass `--model-path` to change it. No other GenieSim source code is loaded.

Search returns up to 20 candidates by default. Use `--top-k` to change this. Compare multiple candidates before choosing one.

Folder fallback and verification:

```bash
scene-assets render --catalog .cache/catalog --folder robolab/yellow_block --output-dir .cache/renders --isaac-python /path/to/isaacsim/python
scene-assets inspect-usd /path/to/asset/Aligned.usda
```

`render` reuses existing snapshots when available. For missing images it uses the scene agent's bundled Isaac Sim renderer on temporary scenes in the output cache, leaving the source asset untouched. Inspect the returned `contact_sheet` and candidate image paths. `inspect-usd` reports mesh bounds in metres; a box-shaped bound alone does not prove semantic shape or color.
