# Motion patterns and local examples

All example links below point into the bundled library. Read their task README,
code, selected scene metadata, and USD before adapting. Constants and asset
part names describe each example's geometry; they are not universal defaults.
Read [execution tips](execution-tips.md) for planner/physical collision scope,
frames, cleanup and measured completion, and [per-task review](example-review.md)
for the selected reference's adaptation limits.

## Ordinary pick and place

Use a combined pick/place API when no intermediate inspection or extraction is
needed. Use separate phases when the object must first clear an enclosure or
be reoriented. With actor-to-actor placement inspect receiving annotations and
the manipulated object's placement semantics. For pose placement generate
object target matrices and check the usable support polygon, orientation,
stability, and gripper clearance. Protect genuinely stationary objects and
completed structures. Do not protect objects expected to shift inside a pile.

References: [stacking and assembly](../../find-example/references/stacking-and-assembly.md),
[sorting](../../find-example/references/storage-and-sorting.md).

## Shelves and cabinets

Use reviewed asset `layers.json` when present, or the bundled mesh/support tools
to find actual shelf surfaces and compartment openings. Transform local
surface coordinates with the current furniture transform and any instance
scale. Determine the lowest/top/middle layer by the task's rule and geometry,
not a copied index. Plan a grasp with a lift that fits the compartment/reach,
pull outward to clear the front, travel to the target compartment, then place.
Account for the ceiling, plate rim or container walls, and hand/finger volume.

Examples: [fruit onto a shelf plate](../../../examples/tasks/move_shelf_fruit_onto_plate_g2/data_gen.py),
[beverages between shelves](../../../examples/tasks/move_beverage_from_top_to_middle_shelf_g2/data_gen.py).
The first example's particular layer indices and small lift are asset-specific.
Use the [surface guide](../../geniesim-scene-generation/references/surfaces.md)
for geometry extraction and marker rendering.

## Articulated doors and sliding drawers

Resolve the actual body, handle, joint path, axis, and limits. Derive handle
grasps from its current part pose. Grasp/close at the handle, permit only the
intended contact in the planner, and execute the joint-compatible movement.
`open` requests a **revolute** trajectory; a prismatic drawer needs a linear
pull along its actual joint direction. Re-read joint state and check the
requested interval and movement direction. Keep the appliance body stable.

Example: [microwave door](../../../examples/tasks/open_microwave_door_g2/data_gen.py).
Its `part_00`/`part_01` names and opening direction must be re-derived for other
microwaves. Related drawer recipes are indexed in
[drawers and shelves](../../find-example/references/drawers-and-shelves.md).

## Precision insertion

Check receiving geometry, aperture clearance, axis and task-specific depth.
Use compatible active/passive placement primitives when they encode that
relationship. A nonzero `PlaceConfig.pre_insert_offset` selects InsertStage;
it does not by itself identify a correct insertion axis or depth. Use approach,
alignment, insertion, release, and withdrawal phases appropriate to geometry.
Candidate perturbations must remain within aperture tolerance. Validate actual
depth and connection state; bounding-box overlap can pass a bad insertion.

References: [insertion tasks](../../find-example/references/insertion.md).

## Held containers

Assign a holding arm and a manipulating arm. Grasp and lift the container,
re-read its pose, and derive insertion/drop targets in its current frame.
Check containment, interior height, and container tilt. When returning the
container to the table, re-check the contents after release and settling.
The attachment and obstacle policy must represent the requested physics;
attaching contents to the container changes contact dynamics.

Example: [fruit into a held bowl](../../../examples/tasks/place_fruit_into_held_bowl_g2/data_gen.py).
Its XY-distance checks are useful structure, but precise general containment
also needs usable interior geometry and height checks.

## Two distinct bimanual modes

`dual_act` synchronizes separate per-arm stages using `DualPick`,
`DualPlaceToActor`, `DualPlaceToPose`, `DualMoveToPose`, or `DualRotate`. Configure
per-arm checkers/protected objects in each config, and shared checks at the
dual call. Unpack its `(success, store_name)` result.

`bimanual_rigid_move` keeps the initial relative end-effector transform while
the leader moves. Establish both contacts/attachments first and check both
arm IK paths. Build targets for the **leader gripper**, not the carried object.
Detach/release in the intended order and verify the final support.

Examples: [lift a pot](../../../examples/tasks/lift_pot_g2/data_gen.py),
[bimanual and held-container family](../../find-example/references/bimanual-and-held-containers.md).

## Buttons and deliberate contact

Approach with the appropriate finger/closed-gripper contact geometry. Temporarily
exclude only the intended contact prim from planner obstacles if necessary.
That changes planning, not physical colliders. Execute along the actual press
axis, verify the relevant part/joint/state changed, then withdraw and restore
planner state. A successful move to a nearby point does not prove a press.

Example: [press a button](../../../examples/tasks/press_button_g2/data_gen.py).

## Writing and ordered contact trajectories

Read `scene_info.target_letter`; use the same letter for instruction and glyph.
Use the current paper pose/geometry to map glyph coordinates into world space,
with the actual pen tip-to-object transform. Generate ordered stroke paths,
move above the first point, lower to contact, follow the path, lift between
strokes, then withdraw. Preserve paper and table; use bounded IK noise that
cannot change the glyph or lose contact. Verify writing/ink as well as motion.

Example: [A-Z writing task](../../../examples/tasks/write_uppercase_letter_with_pen_g2/data_gen.py),
whose 26 instances specify their target letter explicitly.

## Cloth and pins

Query actual FEM node positions to choose cloth corners/edges; a rigid-object
pose does not describe deformed geometry. Approach and close the grippers,
create intended pin groups, execute coordinated folds, release groups, and
measure final node/surface relationships. If temporary collider changes are
needed, restore them after release. Pinning is an explicit simulation mechanism,
not proof of friction-based grasping. Avoid releasing unrelated groups.

Example: [double-layer T-shirt fold](../../../examples/tasks/fold_double_layer_t_shirt_g2/data_gen.py).

## Completion check design

Zero-argument checker closures query live state and return `bool`. Capture
loop variables by value when creating per-object checkers. Use geometric
relationships with appropriate tolerances, and distinguish contact geometry,
support polygons, and hollow interiors from overall bounding boxes.

`check_box_containment` uses wxyz poses and local box sizes; it is an overlap
approximation. Ignoring Z can be useful for open containers only when combined
with vertical/interior and stable-state checks. The current implementation
converts boxes to world AABBs, so rotated boxes and concave geometry require
more precise checks. Final task checks should include objects that can move
after an earlier action checker passes.
