# Stack three building blocks

Task: `stack_three_building_blocks_g2`. Robot: `G2`.

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

`building_block` `table` `cube` `desktop_with_legs` `furniture` `region_0` `region_1` `region_2` `scattered_block_0` `scattered_block_1` `scattered_block_2` `toy` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1036605177_01427688` | [scene_info.json](1036605177_01427688/scene_info.json) | [scene.usda](1036605177_01427688/scene.usda) |
| `1478248193_01427688` | [scene_info.json](1478248193_01427688/scene_info.json) | [scene.usda](1478248193_01427688/scene.usda) |
| `158314999_01427688` | [scene_info.json](158314999_01427688/scene_info.json) | [scene.usda](158314999_01427688/scene.usda) |
| `1653217364_01427688` | [scene_info.json](1653217364_01427688/scene_info.json) | [scene.usda](1653217364_01427688/scene.usda) |
| `2400858556_01427688` | [scene_info.json](2400858556_01427688/scene_info.json) | [scene.usda](2400858556_01427688/scene.usda) |
| `2777342085_01427688` | [scene_info.json](2777342085_01427688/scene_info.json) | [scene.usda](2777342085_01427688/scene.usda) |
| `3574715504_01427688` | [scene_info.json](3574715504_01427688/scene_info.json) | [scene.usda](3574715504_01427688/scene.usda) |
| `3708908900_01427688` | [scene_info.json](3708908900_01427688/scene_info.json) | [scene.usda](3708908900_01427688/scene.usda) |
| `402493233_01427688` | [scene_info.json](402493233_01427688/scene_info.json) | [scene.usda](402493233_01427688/scene.usda) |
| `423885271_01427688` | [scene_info.json](423885271_01427688/scene_info.json) | [scene.usda](423885271_01427688/scene.usda) |
