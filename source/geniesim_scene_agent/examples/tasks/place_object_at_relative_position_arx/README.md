# Place an object at a relative position

Task: `place_object_at_relative_position_arx`. Robot: `ARX`.

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

`left` `on_desk` `table` `right` `benchmark_building_blocks_072` `building blocks` `benchmark_orange_004` `orange` `apple` `benchmark_apple_002` `benchmark_rubik_cube_001` `rubik cube` `benchmark_medicine_bottle_000` `medicine bottle` `desktop_with_legs` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1109945818_5ab08bfe` | [scene_info.json](1109945818_5ab08bfe/scene_info.json) | [scene.usda](1109945818_5ab08bfe/scene.usda) |
| `1566709514_5ab08bfe` | [scene_info.json](1566709514_5ab08bfe/scene_info.json) | [scene.usda](1566709514_5ab08bfe/scene.usda) |
| `2266987751_5ab08bfe` | [scene_info.json](2266987751_5ab08bfe/scene_info.json) | [scene.usda](2266987751_5ab08bfe/scene.usda) |
| `3524388043_5ab08bfe` | [scene_info.json](3524388043_5ab08bfe/scene_info.json) | [scene.usda](3524388043_5ab08bfe/scene.usda) |
| `3604943335_5ab08bfe` | [scene_info.json](3604943335_5ab08bfe/scene_info.json) | [scene.usda](3604943335_5ab08bfe/scene.usda) |
| `394848580_5ab08bfe` | [scene_info.json](394848580_5ab08bfe/scene_info.json) | [scene.usda](394848580_5ab08bfe/scene.usda) |
| `4020398234_5ab08bfe` | [scene_info.json](4020398234_5ab08bfe/scene_info.json) | [scene.usda](4020398234_5ab08bfe/scene.usda) |
| `490978501_5ab08bfe` | [scene_info.json](490978501_5ab08bfe/scene_info.json) | [scene.usda](490978501_5ab08bfe/scene.usda) |
| `547364813_5ab08bfe` | [scene_info.json](547364813_5ab08bfe/scene_info.json) | [scene.usda](547364813_5ab08bfe/scene.usda) |
| `73889972_5ab08bfe` | [scene_info.json](73889972_5ab08bfe/scene_info.json) | [scene.usda](73889972_5ab08bfe/scene.usda) |
