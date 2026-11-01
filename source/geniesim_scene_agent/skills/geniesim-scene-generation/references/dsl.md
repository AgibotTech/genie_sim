# Writing and executing a DSL scene

The scene agent includes its own GenieSim-style DSL runtime. The recipe is a Python file and does not import code from another GenieSim checkout. `from helper import *` is supplied by the scene agent while the recipe runs. The only external project tree needed is the assets directory containing `objects/`, each asset's `item.py`, `description.py`, and `Aligned.usda`.

## Recipe structure

Start from [the working table and block example](../examples/table_block_dsl.py). Export a callable `root_scene() -> Shape`; use `@register()` for consistency with existing DSL scripts. The optional `params` dictionary contains approved generalization values. Return `concat_shapes(...)` of foreground objects and requested support surfaces. `root_scene()` without `params` also works for a single scene or seed-driven scripts. A batch with explicit dimension names requires `root_scene(params=None)` so sampled values cannot be silently ignored.

Use `library_call("usd", oid="asset_id", keywords=[...])` with an ID from `item.py`, not a USD path. The runtime resolves that ID in `--assets-root`. If two folders claim the same ID, the folder whose name exactly matches the ID is preferred; remaining ambiguity is an error. Verify chosen assets with the asset-search skill before using them.

The local helper provides `register`, `library_call`, `Shape`, `concat_shapes`, `transform_shape`, `translation_matrix`, `rotation_matrix`, `quaternion_to_angle_direction`, `scale_matrix`, `get_object_info`, `get_subpart_info`, `compute_shape_min`, `compute_shape_max`, `compute_shape_center`, `compute_shape_sizes`, `loop`, `if_else`, `np`, `math`, and `random`. Translation is in meters, angles are in radians, and Z is up. For yaw around an object's own center:

```python
import math

block = library_call("usd", oid="benchmark_building_blocks_007", keywords=["block"])
block = transform_shape(block, rotation_matrix(math.pi / 4, direction=(0, 0, 1),
                                               point=compute_shape_center(block)))
```

`get_object_info(shape)` returns `min`, `max`, `size`, and `center` for the current transformed shape. A table's overall top can be a starting height, but a shelf or cabinet needs a selected usable support polygon. Inspect it with `scene-scenes supports`, review markers, and use a direct recipe or patch when a placement needs mesh-level precision.

The DSL currently covers asset composition and rigid transforms. Joint state, material edits, custom mesh topology, and articulated parts belong in a USD patch or a direct recipe. The current local helper does not implement every historical GenieSim DSL helper; inspect the supported API above before reusing an old script. Unsupported imports from `geniesim.*` must be rewritten to the bundled API.

## Build and review one instance

From the `scene_agent` workspace:

```bash
export GENIESIM_ASSETS=/path/to/geniesim/assets
export ISAACSIM_PYTHON=/path/to/isaac-sim/python.sh
python3 -m scene_agent_scene build --mode dsl \
  --recipe skills/geniesim-scene-generation/examples/table_block_dsl.py \
  --assets-root "$GENIESIM_ASSETS" --scene-id table_block \
  --output-root .cache/scenes --seed 42 --isaac-python "$ISAACSIM_PYTHON"
python3 -m scene_agent_scene validate .cache/scenes/table_block/0 --require-renders
```

The output is `.cache/scenes/table_block/0/scene.usda`, `scene_info.json`, `generation.json`, `validation.json`, and `renders/front.png`, `side.png`, `top.png`. Show the renders to the user and revise the recipe until the scene is approved. Use `--no-render` only for a temporary USD draft; render it before calling it complete.

## Generalize after approval

For the example recipe, save an approved specification such as:

```json
{"dimensions": {"block_x": {"type": "uniform", "min": -0.1, "max": 0.1, "base": 0.0}}}
```

Then run `build --mode dsl` with `--generalization /path/to/approved.json --count <approved_count>`. Instance `0` uses the approved `base` value; later instances sample the given range. Keep role IDs and support relationships fixed, inspect representative variants, and use a new output root if a previous batch already exists. The `--patch` hook can also read `params` when a scene needs a DSL layout plus edited USD attributes.
