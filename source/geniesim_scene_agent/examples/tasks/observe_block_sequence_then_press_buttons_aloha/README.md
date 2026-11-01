# Observe a block sequence and press buttons in the same order

Task: `observe_block_sequence_then_press_buttons_aloha`. Robot: `Aloha`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.close_gripper` `robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_prim_path` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose` `robot.set_object_parameter_action`

### Client RPC

`robot.client.remove_objs_from_obstacle` `robot.client.set_task_task_basic_info`

### Configuration classes

`MoveConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `observation_event` `press` `sequential_press` `table_only` `table_with_articulation` `unmovable_objects`

## Scene keyword sample

`benchmark_button_000` `benchmark_button_012` `benchmark_button_013` `building_block` `button` `hint_block` `left` `middle` `right` `table` `blue` `control` `cube` `furniture` `green` `has_block_in_front` `left_button` `middle_button` `rectangular` `red`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1697560492_f407e1ae` | [scene_info.json](1697560492_f407e1ae/scene_info.json) | [scene.usda](1697560492_f407e1ae/scene.usda) |
| `1796590420_f407e1ae` | [scene_info.json](1796590420_f407e1ae/scene_info.json) | [scene.usda](1796590420_f407e1ae/scene.usda) |
| `183417345_f407e1ae` | [scene_info.json](183417345_f407e1ae/scene_info.json) | [scene.usda](183417345_f407e1ae/scene.usda) |
| `2266165320_f407e1ae` | [scene_info.json](2266165320_f407e1ae/scene_info.json) | [scene.usda](2266165320_f407e1ae/scene.usda) |
| `3006560034_f407e1ae` | [scene_info.json](3006560034_f407e1ae/scene_info.json) | [scene.usda](3006560034_f407e1ae/scene.usda) |
| `3131320093_f407e1ae` | [scene_info.json](3131320093_f407e1ae/scene_info.json) | [scene.usda](3131320093_f407e1ae/scene.usda) |
| `3603470249_f407e1ae` | [scene_info.json](3603470249_f407e1ae/scene_info.json) | [scene.usda](3603470249_f407e1ae/scene.usda) |
| `765627464_f407e1ae` | [scene_info.json](765627464_f407e1ae/scene_info.json) | [scene.usda](765627464_f407e1ae/scene.usda) |
| `892775274_f407e1ae` | [scene_info.json](892775274_f407e1ae/scene_info.json) | [scene.usda](892775274_f407e1ae/scene.usda) |
| `991403463_f407e1ae` | [scene_info.json](991403463_f407e1ae/scene_info.json) | [scene.usda](991403463_f407e1ae/scene.usda) |
