# Insert a coin into a money box

Task: `insert_coin_into_money_box_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.detach_obj` `robot.follow_trajectory` `robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_size_in_world_frame` `robot.init_position` `robot.init_rotation` `robot.move_actor_to_target_pose` `robot.open_gripper` `robot.pick_actor` `robot.remove_objs_from_obstacle` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`FollowTrajectoryConfig` `GraspConfig` `MoveConfig`

## Scene and action tags

`follow_trajectory` `insertion` `intermediate_motion` `keywords` `pick` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`coin_box` `coin_target` `money_box` `POSITION_TAG_center` `coin` `coin_tray` `container` `currency` `desktop_with_legs` `furniture` `in_cell` `piggy_bank` `table` `target` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1129196055_7bcb1d98` | [scene_info.json](1129196055_7bcb1d98/scene_info.json) | [scene.usda](1129196055_7bcb1d98/scene.usda) |
| `1746238648_7bcb1d98` | [scene_info.json](1746238648_7bcb1d98/scene_info.json) | [scene.usda](1746238648_7bcb1d98/scene.usda) |
| `1811326769_7bcb1d98` | [scene_info.json](1811326769_7bcb1d98/scene_info.json) | [scene.usda](1811326769_7bcb1d98/scene.usda) |
| `2110417897_7bcb1d98` | [scene_info.json](2110417897_7bcb1d98/scene_info.json) | [scene.usda](2110417897_7bcb1d98/scene.usda) |
| `2545503639_7bcb1d98` | [scene_info.json](2545503639_7bcb1d98/scene_info.json) | [scene.usda](2545503639_7bcb1d98/scene.usda) |
| `3613267392_7bcb1d98` | [scene_info.json](3613267392_7bcb1d98/scene_info.json) | [scene.usda](3613267392_7bcb1d98/scene.usda) |
| `3694060330_7bcb1d98` | [scene_info.json](3694060330_7bcb1d98/scene_info.json) | [scene.usda](3694060330_7bcb1d98/scene.usda) |
| `3766224429_7bcb1d98` | [scene_info.json](3766224429_7bcb1d98/scene_info.json) | [scene.usda](3766224429_7bcb1d98/scene.usda) |
| `394896520_7bcb1d98` | [scene_info.json](394896520_7bcb1d98/scene_info.json) | [scene.usda](394896520_7bcb1d98/scene.usda) |
| `4024484185_7bcb1d98` | [scene_info.json](4024484185_7bcb1d98/scene_info.json) | [scene.usda](4024484185_7bcb1d98/scene.usda) |
