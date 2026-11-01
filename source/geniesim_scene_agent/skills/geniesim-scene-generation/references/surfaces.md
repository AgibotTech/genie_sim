# Shelf and cabinet placement surfaces

Inspect the asset snapshots first to count usable levels and distinguish solid planks from baskets. A cabinet top, side wall, or back panel may produce a broad bounding box without being a usable placement surface.

Run `scene-scenes supports <scene.usda> --prim /World/Objects/<shelf_id>`. The tool prefers the asset's existing `assets/interaction/<asset_id>/layers.json` when available. It transforms `corners_local` from the declared reference prim through the actual composed USD transform into world coordinates, including shelf rotation and translation. Review each `height_m`, `corners_world_m`, and `kind` against the rendered scene. Existing layers may intentionally exclude a top cap.

When no `layers.json` exists, the tool reports upward, nearly horizontal mesh faces grouped by height. These are **candidates**: thin rails, a cabinet top, and internal planks may all appear. Inspect the asset visually, compare candidate area and height, and select the usable levels. The method currently assumes horizontal support faces; for sloped or basket floors, use a more specific mesh or physics inspection and explicitly author the placement height.

Add `--markers /path/to/markers.usda` to create a separate overlay with colored corner cubes. Render that overlay with `scene-scenes render`; marker geometry must stay out of the final `scene.usda`. The marker stage keeps Z up and meters, matching the scene.

After choosing a solid plank, call `builder.place_asset_on_surface(...)`. It computes the payload's actual lower bound, puts that bound on the level, and rejects footprints outside the four-corner support polygon with the requested margin. For baskets, determine the interior support height before placing an object; the rim height is not a floor height. Verify the placement in Isaac Sim from multiple views.
