# Place an object at a relative position

Task: `place_object_at_relative_position_g2`. Robot: `G2`.

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

`left` `on_desk` `right` `table` `benchmark_clock_044` `benchmark_garage_kit_064` `benchmark_orange_004` `charmander figure` `clock` `orange` `benchmark_beverage_bottle_001` `benchmark_beverage_bottle_004` `benchmark_bottle_017` `benchmark_toy_plane_029` `coke cola` `oolong tea beverage` `toy plane` `water bottle` `apple` `benchmark_apple_002`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `2636699503_5697c473` | [scene_info.json](2636699503_5697c473/scene_info.json) | [scene.usda](2636699503_5697c473/scene.usda) |
| `301013754_5697c473` | [scene_info.json](301013754_5697c473/scene_info.json) | [scene.usda](301013754_5697c473/scene.usda) |
| `3116205695_5697c473` | [scene_info.json](3116205695_5697c473/scene_info.json) | [scene.usda](3116205695_5697c473/scene.usda) |
| `3204777939_5697c473` | [scene_info.json](3204777939_5697c473/scene_info.json) | [scene.usda](3204777939_5697c473/scene.usda) |
| `3551842323_5697c473` | [scene_info.json](3551842323_5697c473/scene_info.json) | [scene.usda](3551842323_5697c473/scene.usda) |
| `3863375487_5697c473` | [scene_info.json](3863375487_5697c473/scene_info.json) | [scene.usda](3863375487_5697c473/scene.usda) |
| `3871350824_5697c473` | [scene_info.json](3871350824_5697c473/scene_info.json) | [scene.usda](3871350824_5697c473/scene.usda) |
| `4005951819_5697c473` | [scene_info.json](4005951819_5697c473/scene_info.json) | [scene.usda](4005951819_5697c473/scene.usda) |
| `4292336273_5697c473` | [scene_info.json](4292336273_5697c473/scene_info.json) | [scene.usda](4292336273_5697c473/scene.usda) |
| `921021561_5697c473` | [scene_info.json](921021561_5697c473/scene_info.json) | [scene.usda](921021561_5697c473/scene.usda) |
