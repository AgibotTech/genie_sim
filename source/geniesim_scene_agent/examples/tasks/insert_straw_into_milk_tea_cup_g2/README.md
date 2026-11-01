# Insert a straw into a milk tea cup

Task: `insert_straw_into_milk_tea_cup_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.detach_obj` `robot.follow_trajectory` `robot.get_ee_pose` `robot.get_joint_positions` `robot.get_object_pose` `robot.get_object_prim_path` `robot.joint_to` `robot.move_actor_to_target_pose` `robot.open_gripper` `robot.pick_actor` `robot.release_rigid_pin` `robot.remove_objs_from_obstacle` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`FollowTrajectoryConfig` `GraspConfig` `MoveConfig`

## Scene and action tags

`follow_trajectory` `insertion` `intermediate_motion` `keywords` `path_constraint` `pick` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`source_paper_cup` `straw` `target_milk_tea_cup` `target_straw` `POSITION_TAG_left` `POSITION_TAG_right` `container` `cup` `desktop_with_legs` `drinking_straw` `furniture` `graspable` `lidded_cup` `milk_tea_cup` `paper_cup` `rod` `source` `table` `target` `target_container`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1457331657_484fde37` | [scene_info.json](1457331657_484fde37/scene_info.json) | [scene.usda](1457331657_484fde37/scene.usda) |
| `152649344_484fde37` | [scene_info.json](152649344_484fde37/scene_info.json) | [scene.usda](152649344_484fde37/scene.usda) |
| `166775060_484fde37` | [scene_info.json](166775060_484fde37/scene_info.json) | [scene.usda](166775060_484fde37/scene.usda) |
| `2248599970_484fde37` | [scene_info.json](2248599970_484fde37/scene_info.json) | [scene.usda](2248599970_484fde37/scene.usda) |
| `329873673_484fde37` | [scene_info.json](329873673_484fde37/scene_info.json) | [scene.usda](329873673_484fde37/scene.usda) |
| `3754806127_484fde37` | [scene_info.json](3754806127_484fde37/scene_info.json) | [scene.usda](3754806127_484fde37/scene.usda) |
| `3781270993_484fde37` | [scene_info.json](3781270993_484fde37/scene_info.json) | [scene.usda](3781270993_484fde37/scene.usda) |
| `3938377657_484fde37` | [scene_info.json](3938377657_484fde37/scene_info.json) | [scene.usda](3938377657_484fde37/scene.usda) |
| `722935818_484fde37` | [scene_info.json](722935818_484fde37/scene_info.json) | [scene.usda](722935818_484fde37/scene.usda) |
| `927790481_484fde37` | [scene_info.json](927790481_484fde37/scene_info.json) | [scene.usda](927790481_484fde37/scene.usda) |
