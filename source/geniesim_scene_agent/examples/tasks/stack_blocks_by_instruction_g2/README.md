# Stack blocks by instruction

Task: `stack_blocks_by_instruction_g2`. Robot: `G2`.

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

`extra_checkers` `keywords` `pick_and_place` `stack` `table_only` `unmovable_objects`

## Scene keyword sample

`purple` `red` `scattered` `table` `yellow` `building_block` `cube` `desktop_with_legs` `purple_block` `red_block` `white` `workspace_table` `yellow_block`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1657201473_bbf9b7dd` | [scene_info.json](1657201473_bbf9b7dd/scene_info.json) | [scene.usda](1657201473_bbf9b7dd/scene.usda) |
| `2068337255_bbf9b7dd` | [scene_info.json](2068337255_bbf9b7dd/scene_info.json) | [scene.usda](2068337255_bbf9b7dd/scene.usda) |
| `2105261499_bbf9b7dd` | [scene_info.json](2105261499_bbf9b7dd/scene_info.json) | [scene.usda](2105261499_bbf9b7dd/scene.usda) |
| `2248040595_bbf9b7dd` | [scene_info.json](2248040595_bbf9b7dd/scene_info.json) | [scene.usda](2248040595_bbf9b7dd/scene.usda) |
| `2697944058_bbf9b7dd` | [scene_info.json](2697944058_bbf9b7dd/scene_info.json) | [scene.usda](2697944058_bbf9b7dd/scene.usda) |
| `2800931699_bbf9b7dd` | [scene_info.json](2800931699_bbf9b7dd/scene_info.json) | [scene.usda](2800931699_bbf9b7dd/scene.usda) |
| `294614174_bbf9b7dd` | [scene_info.json](294614174_bbf9b7dd/scene_info.json) | [scene.usda](294614174_bbf9b7dd/scene.usda) |
| `2951080838_bbf9b7dd` | [scene_info.json](2951080838_bbf9b7dd/scene_info.json) | [scene.usda](2951080838_bbf9b7dd/scene.usda) |
| `3531782056_bbf9b7dd` | [scene_info.json](3531782056_bbf9b7dd/scene_info.json) | [scene.usda](3531782056_bbf9b7dd/scene.usda) |
| `3886425344_bbf9b7dd` | [scene_info.json](3886425344_bbf9b7dd/scene_info.json) | [scene.usda](3886425344_bbf9b7dd/scene.usda) |
