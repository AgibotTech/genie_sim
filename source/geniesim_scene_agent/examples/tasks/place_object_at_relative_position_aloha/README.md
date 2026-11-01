# Place an object at a relative position

Task: `place_object_at_relative_position_aloha`. Robot: `Aloha`.

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

`extra_checkers` `keywords` `pick_and_place` `relative_position` `table_only` `unmovable_objects`

## Scene keyword sample

`on_desk` `table` `left` `apple` `benchmark_apple_002` `right` `benchmark_medicine_bottle_000` `medicine bottle` `benchmark_building_blocks_072` `benchmark_orange_004` `building blocks` `orange` `benchmark_rubik_cube_001` `rubik cube` `desktop_with_legs` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1368987191_5ab08bfe` | [scene_info.json](1368987191_5ab08bfe/scene_info.json) | [scene.usda](1368987191_5ab08bfe/scene.usda) |
| `1587341418_5ab08bfe` | [scene_info.json](1587341418_5ab08bfe/scene_info.json) | [scene.usda](1587341418_5ab08bfe/scene.usda) |
| `2004548525_5ab08bfe` | [scene_info.json](2004548525_5ab08bfe/scene_info.json) | [scene.usda](2004548525_5ab08bfe/scene.usda) |
| `2632071579_5ab08bfe` | [scene_info.json](2632071579_5ab08bfe/scene_info.json) | [scene.usda](2632071579_5ab08bfe/scene.usda) |
| `2824621783_5ab08bfe` | [scene_info.json](2824621783_5ab08bfe/scene_info.json) | [scene.usda](2824621783_5ab08bfe/scene.usda) |
| `2960417496_5ab08bfe` | [scene_info.json](2960417496_5ab08bfe/scene_info.json) | [scene.usda](2960417496_5ab08bfe/scene.usda) |
| `3503447270_5ab08bfe` | [scene_info.json](3503447270_5ab08bfe/scene_info.json) | [scene.usda](3503447270_5ab08bfe/scene.usda) |
| `3650269533_5ab08bfe` | [scene_info.json](3650269533_5ab08bfe/scene_info.json) | [scene.usda](3650269533_5ab08bfe/scene.usda) |
| `490978501_5ab08bfe` | [scene_info.json](490978501_5ab08bfe/scene_info.json) | [scene.usda](490978501_5ab08bfe/scene.usda) |
| `543309969_5ab08bfe` | [scene_info.json](543309969_5ab08bfe/scene_info.json) | [scene.usda](543309969_5ab08bfe/scene.usda) |
