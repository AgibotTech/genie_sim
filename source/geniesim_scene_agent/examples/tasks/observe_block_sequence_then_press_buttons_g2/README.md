# Observe a block sequence and press buttons in the same order

Task: `observe_block_sequence_then_press_buttons_g2`. Robot: `G2`.

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
| `1375950062_f407e1ae` | [scene_info.json](1375950062_f407e1ae/scene_info.json) | [scene.usda](1375950062_f407e1ae/scene.usda) |
| `1388361966_f407e1ae` | [scene_info.json](1388361966_f407e1ae/scene_info.json) | [scene.usda](1388361966_f407e1ae/scene.usda) |
| `1538407005_f407e1ae` | [scene_info.json](1538407005_f407e1ae/scene_info.json) | [scene.usda](1538407005_f407e1ae/scene.usda) |
| `2404980675_f407e1ae` | [scene_info.json](2404980675_f407e1ae/scene_info.json) | [scene.usda](2404980675_f407e1ae/scene.usda) |
| `2911347005_f407e1ae` | [scene_info.json](2911347005_f407e1ae/scene_info.json) | [scene.usda](2911347005_f407e1ae/scene.usda) |
| `3300484755_f407e1ae` | [scene_info.json](3300484755_f407e1ae/scene_info.json) | [scene.usda](3300484755_f407e1ae/scene.usda) |
| `4065284896_f407e1ae` | [scene_info.json](4065284896_f407e1ae/scene_info.json) | [scene.usda](4065284896_f407e1ae/scene.usda) |
| `489633694_f407e1ae` | [scene_info.json](489633694_f407e1ae/scene_info.json) | [scene.usda](489633694_f407e1ae/scene.usda) |
| `846226688_f407e1ae` | [scene_info.json](846226688_f407e1ae/scene_info.json) | [scene.usda](846226688_f407e1ae/scene.usda) |
| `852581829_f407e1ae` | [scene_info.json](852581829_f407e1ae/scene_info.json) | [scene.usda](852581829_f407e1ae/scene.usda) |
