# Stack three building blocks

Task: `stack_three_building_blocks_arx`. Robot: `ARX`.

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
| `1059617336_01427688` | [scene_info.json](1059617336_01427688/scene_info.json) | [scene.usda](1059617336_01427688/scene.usda) |
| `1093065435_01427688` | [scene_info.json](1093065435_01427688/scene_info.json) | [scene.usda](1093065435_01427688/scene.usda) |
| `1169864891_01427688` | [scene_info.json](1169864891_01427688/scene_info.json) | [scene.usda](1169864891_01427688/scene.usda) |
| `1333717490_01427688` | [scene_info.json](1333717490_01427688/scene_info.json) | [scene.usda](1333717490_01427688/scene.usda) |
| `1759894551_01427688` | [scene_info.json](1759894551_01427688/scene_info.json) | [scene.usda](1759894551_01427688/scene.usda) |
| `1826676194_01427688` | [scene_info.json](1826676194_01427688/scene_info.json) | [scene.usda](1826676194_01427688/scene.usda) |
| `2120078103_01427688` | [scene_info.json](2120078103_01427688/scene_info.json) | [scene.usda](2120078103_01427688/scene.usda) |
| `3867495246_01427688` | [scene_info.json](3867495246_01427688/scene_info.json) | [scene.usda](3867495246_01427688/scene.usda) |
| `651345143_01427688` | [scene_info.json](651345143_01427688/scene_info.json) | [scene.usda](651345143_01427688/scene.usda) |
| `890422703_01427688` | [scene_info.json](890422703_01427688/scene_info.json) | [scene.usda](890422703_01427688/scene.usda) |
