# Press a button

Task: `press_button_arx`. Robot: `ARX`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.close_gripper` `robot.get_object_pose` `robot.get_object_prim_path` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose`

### Client RPC

`robot.client.remove_objs_from_obstacle` `robot.client.set_task_task_basic_info`

### Configuration classes

`MoveConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `press` `table_only` `table_with_articulation` `unmovable_objects`

## Scene keyword sample

`button` `table` `benchmark_button_009` `benchmark_button_023` `benchmark_button_011` `benchmark_button_013` `benchmark_button_017` `benchmark_button_022` `benchmark_button_027` `benchmark_button_028` `control` `desktop` `furniture` `randomly_selected` `white`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1042092724_41f905e0` | [scene_info.json](1042092724_41f905e0/scene_info.json) | [scene.usda](1042092724_41f905e0/scene.usda) |
| `1782164330_41f905e0` | [scene_info.json](1782164330_41f905e0/scene_info.json) | [scene.usda](1782164330_41f905e0/scene.usda) |
| `1918779850_41f905e0` | [scene_info.json](1918779850_41f905e0/scene_info.json) | [scene.usda](1918779850_41f905e0/scene.usda) |
| `202850642_41f905e0` | [scene_info.json](202850642_41f905e0/scene_info.json) | [scene.usda](202850642_41f905e0/scene.usda) |
| `2497742135_41f905e0` | [scene_info.json](2497742135_41f905e0/scene_info.json) | [scene.usda](2497742135_41f905e0/scene.usda) |
| `284290284_41f905e0` | [scene_info.json](284290284_41f905e0/scene_info.json) | [scene.usda](284290284_41f905e0/scene.usda) |
| `3044984861_41f905e0` | [scene_info.json](3044984861_41f905e0/scene_info.json) | [scene.usda](3044984861_41f905e0/scene.usda) |
| `3420825328_41f905e0` | [scene_info.json](3420825328_41f905e0/scene_info.json) | [scene.usda](3420825328_41f905e0/scene.usda) |
| `3654531066_41f905e0` | [scene_info.json](3654531066_41f905e0/scene_info.json) | [scene.usda](3654531066_41f905e0/scene.usda) |
| `623902489_41f905e0` | [scene_info.json](623902489_41f905e0/scene_info.json) | [scene.usda](623902489_41f905e0/scene.usda) |
