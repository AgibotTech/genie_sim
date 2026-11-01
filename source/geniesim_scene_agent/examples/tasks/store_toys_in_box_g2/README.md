# Store toys in a box

Task: `store_toys_in_box_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_size_in_local_frame` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.pick_actor` `robot.place_actor_to_target_pose`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `keywords` `pick_and_place` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`table` `inside_storage` `box` `desktop_with_legs` `on_desk` `plastic` `plaything` `randomly_placed` `small` `storage_container` `toy_1` `white` `workspace_table` `toy_in_box_1` `toy_2` `toy_in_box_2` `toy_in_box_3`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1678894611_c5e1609e` | [scene_info.json](1678894611_c5e1609e/scene_info.json) | [scene.usda](1678894611_c5e1609e/scene.usda) |
| `2631235939_c5e1609e` | [scene_info.json](2631235939_c5e1609e/scene_info.json) | [scene.usda](2631235939_c5e1609e/scene.usda) |
| `3276661659_c5e1609e` | [scene_info.json](3276661659_c5e1609e/scene_info.json) | [scene.usda](3276661659_c5e1609e/scene.usda) |
| `33791984_c5e1609e` | [scene_info.json](33791984_c5e1609e/scene_info.json) | [scene.usda](33791984_c5e1609e/scene.usda) |
| `3415985182_c5e1609e` | [scene_info.json](3415985182_c5e1609e/scene_info.json) | [scene.usda](3415985182_c5e1609e/scene.usda) |
| `3433810526_c5e1609e` | [scene_info.json](3433810526_c5e1609e/scene_info.json) | [scene.usda](3433810526_c5e1609e/scene.usda) |
| `359965423_c5e1609e` | [scene_info.json](359965423_c5e1609e/scene_info.json) | [scene.usda](359965423_c5e1609e/scene.usda) |
| `3719229828_c5e1609e` | [scene_info.json](3719229828_c5e1609e/scene_info.json) | [scene.usda](3719229828_c5e1609e/scene.usda) |
| `506374001_c5e1609e` | [scene_info.json](506374001_c5e1609e/scene_info.json) | [scene.usda](506374001_c5e1609e/scene.usda) |
| `896498700_c5e1609e` | [scene_info.json](896498700_c5e1609e/scene_info.json) | [scene.usda](896498700_c5e1609e/scene.usda) |
