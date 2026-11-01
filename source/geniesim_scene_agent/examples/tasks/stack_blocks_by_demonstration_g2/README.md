# Stack blocks by demonstration

Task: `stack_blocks_by_demonstration_g2`. Robot: `G2`.

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

`demo_reference` `extra_checkers` `keywords` `pick_and_place` `stack` `table_only` `unmovable_objects`

## Scene keyword sample

`bottom` `middle` `scattered` `stacked_demo` `table` `top` `building_block` `cube` `desktop_with_legs` `purple` `purple_block` `red` `red_block` `stacked_bottom` `stacked_middle` `stacked_top` `white` `workspace_table` `yellow` `yellow_block`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1734401138_a0c48ce8` | [scene_info.json](1734401138_a0c48ce8/scene_info.json) | [scene.usda](1734401138_a0c48ce8/scene.usda) |
| `231696214_a0c48ce8` | [scene_info.json](231696214_a0c48ce8/scene_info.json) | [scene.usda](231696214_a0c48ce8/scene.usda) |
| `2419178089_a0c48ce8` | [scene_info.json](2419178089_a0c48ce8/scene_info.json) | [scene.usda](2419178089_a0c48ce8/scene.usda) |
| `2648724934_a0c48ce8` | [scene_info.json](2648724934_a0c48ce8/scene_info.json) | [scene.usda](2648724934_a0c48ce8/scene.usda) |
| `3234463675_a0c48ce8` | [scene_info.json](3234463675_a0c48ce8/scene_info.json) | [scene.usda](3234463675_a0c48ce8/scene.usda) |
| `3347645310_a0c48ce8` | [scene_info.json](3347645310_a0c48ce8/scene_info.json) | [scene.usda](3347645310_a0c48ce8/scene.usda) |
| `3401618238_a0c48ce8` | [scene_info.json](3401618238_a0c48ce8/scene_info.json) | [scene.usda](3401618238_a0c48ce8/scene.usda) |
| `647887024_a0c48ce8` | [scene_info.json](647887024_a0c48ce8/scene_info.json) | [scene.usda](647887024_a0c48ce8/scene.usda) |
| `697671655_a0c48ce8` | [scene_info.json](697671655_a0c48ce8/scene_info.json) | [scene.usda](697671655_a0c48ce8/scene.usda) |
| `814089944_a0c48ce8` | [scene_info.json](814089944_a0c48ce8/scene_info.json) | [scene.usda](814089944_a0c48ce8/scene.usda) |
