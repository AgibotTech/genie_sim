# Write an uppercase letter with a pen

Task: `write_uppercase_letter_with_pen_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.follow_trajectory` `robot.get_object_pose` `robot.get_object_prim_path` `robot.get_object_size_in_world_frame` `robot.move_actor_to_target_pose` `robot.pick_actor` `robot.remove_objs_from_obstacle`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`FollowTrajectoryConfig` `GraspConfig` `MoveConfig`

## Scene and action tags

`contact_trajectory` `follow_trajectory` `intermediate_motion` `keywords` `pick` `scene_info_target_letter` `table_only` `unmovable_objects`

## Scene keyword sample

`paper` `pen` `pen_holder` `table` `writing_pen` `writing_table` `center` `dynamic` `inside_pen_holder` `kinematic` `non_kinematic` `on_table` `table_side` `workspace` `writing_surface`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `0` | [scene_info.json](0/scene_info.json) | [scene.usda](0/scene.usda) |
| `1` | [scene_info.json](1/scene_info.json) | [scene.usda](1/scene.usda) |
| `10` | [scene_info.json](10/scene_info.json) | [scene.usda](10/scene.usda) |
| `11` | [scene_info.json](11/scene_info.json) | [scene.usda](11/scene.usda) |
| `12` | [scene_info.json](12/scene_info.json) | [scene.usda](12/scene.usda) |
| `13` | [scene_info.json](13/scene_info.json) | [scene.usda](13/scene.usda) |
| `14` | [scene_info.json](14/scene_info.json) | [scene.usda](14/scene.usda) |
| `15` | [scene_info.json](15/scene_info.json) | [scene.usda](15/scene.usda) |
| `16` | [scene_info.json](16/scene_info.json) | [scene.usda](16/scene.usda) |
| `17` | [scene_info.json](17/scene_info.json) | [scene.usda](17/scene.usda) |
| `18` | [scene_info.json](18/scene_info.json) | [scene.usda](18/scene.usda) |
| `19` | [scene_info.json](19/scene_info.json) | [scene.usda](19/scene.usda) |
| `2` | [scene_info.json](2/scene_info.json) | [scene.usda](2/scene.usda) |
| `20` | [scene_info.json](20/scene_info.json) | [scene.usda](20/scene.usda) |
| `21` | [scene_info.json](21/scene_info.json) | [scene.usda](21/scene.usda) |
| `22` | [scene_info.json](22/scene_info.json) | [scene.usda](22/scene.usda) |
| `23` | [scene_info.json](23/scene_info.json) | [scene.usda](23/scene.usda) |
| `24` | [scene_info.json](24/scene_info.json) | [scene.usda](24/scene.usda) |
| `25` | [scene_info.json](25/scene_info.json) | [scene.usda](25/scene.usda) |
| `3` | [scene_info.json](3/scene_info.json) | [scene.usda](3/scene.usda) |
| `4` | [scene_info.json](4/scene_info.json) | [scene.usda](4/scene.usda) |
| `5` | [scene_info.json](5/scene_info.json) | [scene.usda](5/scene.usda) |
| `6` | [scene_info.json](6/scene_info.json) | [scene.usda](6/scene.usda) |
| `7` | [scene_info.json](7/scene_info.json) | [scene.usda](7/scene.usda) |
| `8` | [scene_info.json](8/scene_info.json) | [scene.usda](8/scene.usda) |
| `9` | [scene_info.json](9/scene_info.json) | [scene.usda](9/scene.usda) |

The writing instances specify `scene_info.target_letter` (A-Z).
The task reads this field for both the instruction and the contact strokes.
