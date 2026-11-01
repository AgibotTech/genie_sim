# Scene tool commands and Python hooks

Set `GENIESIM_ASSETS` to the external GenieSim assets directory and `ISAACSIM_PYTHON` to Isaac Sim's `python.sh`. Run tools from this workspace or install the package to use `scene-scenes`. The DSL implementation and renderer are bundled here. Read [DSL authoring](dsl.md) before writing a recipe.

## DSL scene

```bash
python3 -m scene_agent_scene build --mode dsl --recipe /path/to/scene_dsl.py \
  --scene-id table_scene --output-root .cache/scenes \
  --assets-root "$GENIESIM_ASSETS" --isaac-python "$ISAACSIM_PYTHON"
```

The DSL file defines `root_scene()` and uses `from helper import *`. The bundled runtime resolves asset IDs from `--assets-root`, assigns stable role IDs, and passes objects through the common builder. See [the executable example](../examples/table_block_dsl.py).

## DSL plus USD patch

Add `--patch /path/to/patch.py` to the DSL command. The patch file defines:

```python
def patch_scene(builder, rng, params):
    builder.set_revolute_joint_state(
        "microwave_oven_000", "entity/part_00/RevoluteJoint", -90,
        damping=1.0, friction=2.5,
    )
```

Inspect the composed joint path and its degree limits before using this helper. Confirm the physical state in Isaac Sim after generation.

## Direct Python scene

The recipe file defines `build_scene(builder, rng, params)`. For example:

```python
def build_scene(builder, rng, params):
    builder.add_asset("table", "/path/to/table/Aligned.usda", xyz=(0, 0, 0.37))
    builder.add_asset("building block", "/path/to/block/Aligned.usda",
                      xyz=(params.get("x", 0.0), 0, 0.75), keywords=["yellow"])
```

Use `builder.add_box(...)` for simple new geometry. More complex USD changes can use `builder.stage`, but put scene objects under `/World/Objects` and keep `builder.layout` in sync. A direct recipe may define `check_scene(builder, params)` and raise `ValueError` when an instance violates a scene-specific constraint.
Use `builder.add_background(...)` only when the user explicitly requested background geometry. To place an object on a verified shelf level, use `builder.place_asset_on_surface(name, asset_usd, surface, x=..., y=..., margin=...)`.

```bash
python3 -m scene_agent_scene build --mode direct --recipe /path/to/scene_recipe.py \
  --scene-id table_scene --output-root .cache/scenes --isaac-python "$ISAACSIM_PYTHON"
```

## Generalization

An approved JSON specification lists explicit parameter dimensions. The recipe or patch reads them from `params`.

```json
{
  "dimensions": {
    "x": {"type": "uniform", "min": -0.05, "max": 0.05, "base": 0.0},
    "block_asset": {"type": "choice", "values": ["/path/to/a.usda", "/path/to/b.usda"], "base": "/path/to/a.usda"}
  }
}
```

`uniform`, `int`, and `choice` are supported. `base` supplies the approved instance `0` value. Confirm ranges, fixed relations, and total count before calling:

```bash
python3 -m scene_agent_scene build --mode direct --recipe /path/to/scene_recipe.py \
  --scene-id table_scene --output-root .cache/scenes --generalization /path/to/approved.json \
  --count 1000 --seed 42 --isaac-python "$ISAACSIM_PYTHON"
```

The renderer handles the batch in one Isaac Sim process. A failed build keeps `.staging_<id>` for diagnosis and does not publish numeric instance directories. A recipe may reject an invalid sampled placement with `from scene_agent_scene.generate import SceneRejected`; the batch resamples that instance up to `max_attempts` (default 5) and records rejected attempts in `batch_report.json`.

## Inspect, render, validate

```bash
python3 -m scene_agent_scene validate .cache/scenes/table_scene/0 --require-renders
python3 -m scene_agent_scene render .cache/scenes/table_scene/0/scene.usda --isaac-python "$ISAACSIM_PYTHON"
python3 -m scene_agent_assets inspect-usd /path/to/asset/Aligned.usda
python3 -m scene_agent_scene supports .cache/scenes/table_scene/0/scene.usda --prim /World/Objects/shelf_000 --markers .cache/shelf_markers.usda
```

`--no-render` on `build` is for fast USD drafts; render and validate before presenting a scene as final.
