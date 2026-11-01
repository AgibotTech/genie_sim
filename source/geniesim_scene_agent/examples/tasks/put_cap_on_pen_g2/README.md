# Put a cap on a pen using both hands

Task: `put_cap_on_pen_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_prim_path` `robot.move_actor_to_target_pose` `robot.open_gripper` `robot.pick_actor` `robot.remove_objs_from_obstacle` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig`

## Scene and action tags

`intermediate_motion` `keywords` `pick` `table_only` `unmovable_objects`

## Scene keyword sample

`cap` `pen` `target_cap` `target_pen` `POSITION_TAG_left` `POSITION_TAG_right` `desktop_with_legs` `furniture` `graspable` `marker_pen` `pen_cap` `rod` `stationery` `table` `target` `workspace` `writing_tool`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1047100030_97771c6d` | [scene_info.json](1047100030_97771c6d/scene_info.json) | [scene.usda](1047100030_97771c6d/scene.usda) |
| `1527194599_97771c6d` | [scene_info.json](1527194599_97771c6d/scene_info.json) | [scene.usda](1527194599_97771c6d/scene.usda) |
| `1644178067_97771c6d` | [scene_info.json](1644178067_97771c6d/scene_info.json) | [scene.usda](1644178067_97771c6d/scene.usda) |
| `1941023757_97771c6d` | [scene_info.json](1941023757_97771c6d/scene_info.json) | [scene.usda](1941023757_97771c6d/scene.usda) |
| `2018483987_97771c6d` | [scene_info.json](2018483987_97771c6d/scene_info.json) | [scene.usda](2018483987_97771c6d/scene.usda) |
| `291527105_97771c6d` | [scene_info.json](291527105_97771c6d/scene_info.json) | [scene.usda](291527105_97771c6d/scene.usda) |
| `3306742919_97771c6d` | [scene_info.json](3306742919_97771c6d/scene_info.json) | [scene.usda](3306742919_97771c6d/scene.usda) |
| `3377653531_97771c6d` | [scene_info.json](3377653531_97771c6d/scene_info.json) | [scene.usda](3377653531_97771c6d/scene.usda) |
| `776005506_97771c6d` | [scene_info.json](776005506_97771c6d/scene_info.json) | [scene.usda](776005506_97771c6d/scene.usda) |
| `997310943_97771c6d` | [scene_info.json](997310943_97771c6d/scene_info.json) | [scene.usda](997310943_97771c6d/scene.usda) |
