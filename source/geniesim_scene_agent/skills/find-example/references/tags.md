# Task retrieval vocabulary

Use these tags alongside an object's name and the intended action. API and
Config identifiers are also searchable. The tables list observed code calls
and scene keywords; tag classification helps locate comparable task flows.

## Scene types

| Tag | Meaning |
| --- | --- |
| `table_only` | Objects placed on a table without a selected container scene type |
| `table_with_containers` | Tabletop containers such as bowls, cups, boxes, bags, or holders |
| `table_with_articulation` | A task involving a door, drawer, button, or trash-can mechanism |
| `shelf_scene` | Furniture with shelf levels; compare support height, clearance, and approach direction |
| `drawer_scene` | A drawer-based layout; compare hidden targets, handles, and obstacle updates |
| `multi_table` | The implementation uses body turning to transfer between work areas |
| `deformable_cloth` | Cloth node positions and FEM constraints affect the task |
| `held_container` | One hand holds the receiving container during the task |

## Actions and task logic

| Tag | Meaning |
| --- | --- |
| `pick_and_place` | A combined pick/place call or a sequence with explicit placement |
| `pick` | Picking is observed; inspect subsequent moves and release behavior |
| `stack` / `sort` / `swap` | Stack objects, order/group objects, or exchange positions |
| `open_action` | Opening a mechanism; some examples press a pedal or move a gripper instead of calling `open` |
| `press` | Button or pedal pressing; inspect the direction and path constraints |
| `pour` / `shake` | Container tilt or shaking phases with a physical completion check |
| `assembly` / `disassembly` / `insertion` | Fit, remove, or insert objects using geometry and interaction annotations |
| `capture` | Move a captured chess piece aside and move the capturing piece to its square |
| `demo_reference` | Derive target order or layout from a demonstration |
| `relative_position` | Compute a target pose relative to another object |
| `sequential_inspection` | Inspect candidate drawers in sequence to find a hidden target |
| `observation_event` / `sequential_press` | Observe a cue sequence, then press buttons in the corresponding order |
| `bimanual` | The code uses `dual_act` or `bimanual_rigid_move`; inspect synchronization and attachment |
| `fold` | Deformable folding with grasp, transport, release, and pin management |
| `contact_trajectory` / `follow_trajectory` | Follow a contact path, such as pen strokes on paper |

## Object lookup and metadata

| Term | Meaning |
| --- | --- |
| `keywords` | Read literal role markers from `scene_info.layout[object_id].keywords` |
| `usd` / `usd_match` | Match asset IDs; review the actual code for category or reference matching |
| `target` / `target_object` / `target_beverage` | Example role keywords for the manipulated target |
| `on_top_layer` / `on_middle_layer` | Literal shelf placement markers in the selected shelf examples |
| `hidden_block` / `inside_drawer` / `contains_block` | Hidden-target or containment markers |
| `fixed` / `reference` / `on_desk` | Common role markers; use the exact words in the chosen scene |
| `scene_info_target_letter` | The writing task reads `scene_info.target_letter`, one uppercase letter A-Z |

Keyword examples here are search hints. Each row's keyword sample and the
actual `scene_info.json` determine the applicable object contract.

## Motion and verification

| Term | Meaning |
| --- | --- |
| `intermediate_motion` | The code calls `move_actor_to_target_pose`; inspect waypoint clearance and release phases |
| `path_constraint` | A motion direction or path is constrained through a Config argument |
| `extra_checkers` | The code supplies additional physical success checks |
| `unmovable_objects` | The code marks objects that should remain fixed while planning |
| `GraspConfig` | Grasp direction, primitives, obstacles, and grasp checks |
| `PlaceConfig` | Placement poses, primitives, constraints, and completion checks |
| `MoveConfig` | Waypoint motion and goal offsets |
| `OpenConfig` | Articulation opening options |
| `FollowTrajectoryConfig` | Contact trajectory motion options |
| `DualPick` / `DualMoveToPose` | Specs supplied to synchronized dual-arm actions |

Compare the selected task's actual arguments and checker code. Read the
[RobotAPI quick reference](../../geniesim-scene-generation/references/robot_api.md)
for pose conventions and the basic interface contract.
