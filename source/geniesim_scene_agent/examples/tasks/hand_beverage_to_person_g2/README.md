# Hand a beverage to a person

Task: `hand_beverage_to_person_g2`. Robot: `G2`.

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

`extra_checkers` `keywords` `pick_and_place` `table_only` `unmovable_objects`

## Scene keyword sample

`dining_table` `human` `person` `table` `water_bottle` `cylinder` `drinking` `four_legs` `in_front_of_table` `on_table` `sitting` `transparent` `wooden`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `117356952_0965bdf3` | [scene_info.json](117356952_0965bdf3/scene_info.json) | [scene.usda](117356952_0965bdf3/scene.usda) |
| `1405548753_0965bdf3` | [scene_info.json](1405548753_0965bdf3/scene_info.json) | [scene.usda](1405548753_0965bdf3/scene.usda) |
| `1475427239_0965bdf3` | [scene_info.json](1475427239_0965bdf3/scene_info.json) | [scene.usda](1475427239_0965bdf3/scene.usda) |
| `3154043893_0965bdf3` | [scene_info.json](3154043893_0965bdf3/scene_info.json) | [scene.usda](3154043893_0965bdf3/scene.usda) |
| `3382715868_0965bdf3` | [scene_info.json](3382715868_0965bdf3/scene_info.json) | [scene.usda](3382715868_0965bdf3/scene.usda) |
| `3726094312_0965bdf3` | [scene_info.json](3726094312_0965bdf3/scene_info.json) | [scene.usda](3726094312_0965bdf3/scene.usda) |
| `3747590881_0965bdf3` | [scene_info.json](3747590881_0965bdf3/scene_info.json) | [scene.usda](3747590881_0965bdf3/scene.usda) |
| `4290879545_0965bdf3` | [scene_info.json](4290879545_0965bdf3/scene_info.json) | [scene.usda](4290879545_0965bdf3/scene.usda) |
| `453710254_0965bdf3` | [scene_info.json](453710254_0965bdf3/scene_info.json) | [scene.usda](453710254_0965bdf3/scene.usda) |
| `834966897_0965bdf3` | [scene_info.json](834966897_0965bdf3/scene_info.json) | [scene.usda](834966897_0965bdf3/scene.usda) |
