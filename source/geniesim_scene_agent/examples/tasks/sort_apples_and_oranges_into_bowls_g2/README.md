# Sort apples and oranges into separate bowls

Task: `sort_apples_and_oranges_into_bowls_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.detach_obj` `robot.dual_act` `robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_prim_path` `robot.get_object_size_in_world_frame` `robot.remove_objs_from_obstacle` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`DualMoveToPose` `DualPick` `GraspConfig` `MoveConfig`

## Scene and action tags

`bimanual` `extra_checkers` `keywords` `sort` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`apple` `bowl` `left` `orange` `right` `table` `bowl_left` `bowl_right` `container` `fruit` `furniture` `on_desk` `workspace`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1051399485_460105e6` | [scene_info.json](1051399485_460105e6/scene_info.json) | [scene.usda](1051399485_460105e6/scene.usda) |
| `1887895703_460105e6` | [scene_info.json](1887895703_460105e6/scene_info.json) | [scene.usda](1887895703_460105e6/scene.usda) |
| `2768447884_460105e6` | [scene_info.json](2768447884_460105e6/scene_info.json) | [scene.usda](2768447884_460105e6/scene.usda) |
| `2972794749_460105e6` | [scene_info.json](2972794749_460105e6/scene_info.json) | [scene.usda](2972794749_460105e6/scene.usda) |
| `3079268752_460105e6` | [scene_info.json](3079268752_460105e6/scene_info.json) | [scene.usda](3079268752_460105e6/scene.usda) |
| `322211931_460105e6` | [scene_info.json](322211931_460105e6/scene_info.json) | [scene.usda](322211931_460105e6/scene.usda) |
| `677044041_460105e6` | [scene_info.json](677044041_460105e6/scene_info.json) | [scene.usda](677044041_460105e6/scene.usda) |
| `831741754_460105e6` | [scene_info.json](831741754_460105e6/scene_info.json) | [scene.usda](831741754_460105e6/scene.usda) |
| `894934236_460105e6` | [scene_info.json](894934236_460105e6/scene_info.json) | [scene.usda](894934236_460105e6/scene.usda) |
| `90554508_460105e6` | [scene_info.json](90554508_460105e6/scene_info.json) | [scene.usda](90554508_460105e6/scene.usda) |
