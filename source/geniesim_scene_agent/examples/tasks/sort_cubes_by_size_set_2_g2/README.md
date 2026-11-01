# Sort cubes by size (scene set 2)

Task: `sort_cubes_by_size_set_2_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.pick_and_place_actor_to_target_pose`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `keywords` `pick_and_place` `sort` `table_only` `unmovable_objects`

## Scene keyword sample

`building_block` `cube` `table` `desktop_with_legs` `green` `interaction` `small_cube` `up_axis_y` `white` `workspace_table` `blue` `large_cube_blue` `medium_cube_blue` `medium_cube_reddish_brown` `reddish_brown` `large_cube_red` `red`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1285440727_a95ccddf` | [scene_info.json](1285440727_a95ccddf/scene_info.json) | [scene.usda](1285440727_a95ccddf/scene.usda) |
| `1388281965_a95ccddf` | [scene_info.json](1388281965_a95ccddf/scene_info.json) | [scene.usda](1388281965_a95ccddf/scene.usda) |
| `2501987137_a95ccddf` | [scene_info.json](2501987137_a95ccddf/scene_info.json) | [scene.usda](2501987137_a95ccddf/scene.usda) |
| `2566328155_a95ccddf` | [scene_info.json](2566328155_a95ccddf/scene_info.json) | [scene.usda](2566328155_a95ccddf/scene.usda) |
| `2605312312_a95ccddf` | [scene_info.json](2605312312_a95ccddf/scene_info.json) | [scene.usda](2605312312_a95ccddf/scene.usda) |
| `2842905069_a95ccddf` | [scene_info.json](2842905069_a95ccddf/scene_info.json) | [scene.usda](2842905069_a95ccddf/scene.usda) |
| `332698610_a95ccddf` | [scene_info.json](332698610_a95ccddf/scene_info.json) | [scene.usda](332698610_a95ccddf/scene.usda) |
| `3743609721_a95ccddf` | [scene_info.json](3743609721_a95ccddf/scene_info.json) | [scene.usda](3743609721_a95ccddf/scene.usda) |
| `905491390_a95ccddf` | [scene_info.json](905491390_a95ccddf/scene_info.json) | [scene.usda](905491390_a95ccddf/scene.usda) |
| `932299086_a95ccddf` | [scene_info.json](932299086_a95ccddf/scene_info.json) | [scene.usda](932299086_a95ccddf/scene.usda) |
