# Stack bowls

Task: `stack_bowls_g2`. Robot: `G2`.

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

`extra_checkers` `keywords` `pick_and_place` `stack` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`dining_table` `table` `bowl_1` `bowl_2` `bowl_3` `ceramic` `furniture` `kitchenware` `on_table` `white` `wood`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1273360758_7193c6cb` | [scene_info.json](1273360758_7193c6cb/scene_info.json) | [scene.usda](1273360758_7193c6cb/scene.usda) |
| `1294728184_7193c6cb` | [scene_info.json](1294728184_7193c6cb/scene_info.json) | [scene.usda](1294728184_7193c6cb/scene.usda) |
| `1892447765_7193c6cb` | [scene_info.json](1892447765_7193c6cb/scene_info.json) | [scene.usda](1892447765_7193c6cb/scene.usda) |
| `2008980183_7193c6cb` | [scene_info.json](2008980183_7193c6cb/scene_info.json) | [scene.usda](2008980183_7193c6cb/scene.usda) |
| `2608655455_7193c6cb` | [scene_info.json](2608655455_7193c6cb/scene_info.json) | [scene.usda](2608655455_7193c6cb/scene.usda) |
| `2683610120_7193c6cb` | [scene_info.json](2683610120_7193c6cb/scene_info.json) | [scene.usda](2683610120_7193c6cb/scene.usda) |
| `2901126356_7193c6cb` | [scene_info.json](2901126356_7193c6cb/scene_info.json) | [scene.usda](2901126356_7193c6cb/scene.usda) |
| `3689906980_7193c6cb` | [scene_info.json](3689906980_7193c6cb/scene_info.json) | [scene.usda](3689906980_7193c6cb/scene.usda) |
| `4016187823_7193c6cb` | [scene_info.json](4016187823_7193c6cb/scene_info.json) | [scene.usda](4016187823_7193c6cb/scene.usda) |
| `731232738_7193c6cb` | [scene_info.json](731232738_7193c6cb/scene_info.json) | [scene.usda](731232738_7193c6cb/scene.usda) |
