# Swap two building blocks

Task: `swap_two_building_blocks_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.judge_arm_to_use` `robot.pick_and_place_actor_to_target_pose`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`keywords` `pick_and_place` `swap` `table_only` `unmovable_objects`

## Scene keyword sample

`block_A` `block_B` `building_block` `table` `colored` `desktop_with_legs` `left` `on_desk` `right` `wood` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1971100705_b1632919` | [scene_info.json](1971100705_b1632919/scene_info.json) | [scene.usda](1971100705_b1632919/scene.usda) |
| `2263701847_b1632919` | [scene_info.json](2263701847_b1632919/scene_info.json) | [scene.usda](2263701847_b1632919/scene.usda) |
| `3002181964_b1632919` | [scene_info.json](3002181964_b1632919/scene_info.json) | [scene.usda](3002181964_b1632919/scene.usda) |
| `3404011142_b1632919` | [scene_info.json](3404011142_b1632919/scene_info.json) | [scene.usda](3404011142_b1632919/scene.usda) |
| `3493538129_b1632919` | [scene_info.json](3493538129_b1632919/scene_info.json) | [scene.usda](3493538129_b1632919/scene.usda) |
| `365463903_b1632919` | [scene_info.json](365463903_b1632919/scene_info.json) | [scene.usda](365463903_b1632919/scene.usda) |
| `3784070174_b1632919` | [scene_info.json](3784070174_b1632919/scene_info.json) | [scene.usda](3784070174_b1632919/scene.usda) |
| `4131051562_b1632919` | [scene_info.json](4131051562_b1632919/scene_info.json) | [scene.usda](4131051562_b1632919/scene.usda) |
| `757411000_b1632919` | [scene_info.json](757411000_b1632919/scene_info.json) | [scene.usda](757411000_b1632919/scene.usda) |
| `830962134_b1632919` | [scene_info.json](830962134_b1632919/scene_info.json) | [scene.usda](830962134_b1632919/scene.usda) |
