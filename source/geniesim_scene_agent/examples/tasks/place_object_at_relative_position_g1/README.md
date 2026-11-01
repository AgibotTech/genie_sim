# Place an object at a relative position

Task: `place_object_at_relative_position_g1`. Robot: `G1`.

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

`extra_checkers` `keywords` `pick_and_place` `relative_position` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`left` `on_desk` `right` `table` `benchmark_peach_021` `peach` `benchmark_beverage_bottle_003` `benchmark_beverage_bottle_004` `benchmark_building_blocks_072` `building blocks` `fanta` `oolong tea beverage` `apple` `benchmark_apple_002` `benchmark_bottle_017` `benchmark_bowl_sponge_000` `benchmark_cup_043` `benchmark_cup_105` `benchmark_green_apple_001` `benchmark_lemon_030`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1122712314_5697c473` | [scene_info.json](1122712314_5697c473/scene_info.json) | [scene.usda](1122712314_5697c473/scene.usda) |
| `1420762626_5697c473` | [scene_info.json](1420762626_5697c473/scene_info.json) | [scene.usda](1420762626_5697c473/scene.usda) |
| `1675594207_5697c473` | [scene_info.json](1675594207_5697c473/scene_info.json) | [scene.usda](1675594207_5697c473/scene.usda) |
| `2316003831_5697c473` | [scene_info.json](2316003831_5697c473/scene_info.json) | [scene.usda](2316003831_5697c473/scene.usda) |
| `328279711_5697c473` | [scene_info.json](328279711_5697c473/scene_info.json) | [scene.usda](328279711_5697c473/scene.usda) |
| `3530510582_5697c473` | [scene_info.json](3530510582_5697c473/scene_info.json) | [scene.usda](3530510582_5697c473/scene.usda) |
| `4005951819_5697c473` | [scene_info.json](4005951819_5697c473/scene_info.json) | [scene.usda](4005951819_5697c473/scene.usda) |
| `4178323869_5697c473` | [scene_info.json](4178323869_5697c473/scene_info.json) | [scene.usda](4178323869_5697c473/scene.usda) |
| `488232453_5697c473` | [scene_info.json](488232453_5697c473/scene_info.json) | [scene.usda](488232453_5697c473/scene.usda) |
| `641924667_5697c473` | [scene_info.json](641924667_5697c473/scene_info.json) | [scene.usda](641924667_5697c473/scene.usda) |
