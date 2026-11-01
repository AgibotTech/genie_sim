# Find a hidden block in three drawers (scene set 3)

Task: `find_hidden_block_in_three_drawers_set_3_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_prim_path` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose` `robot.open_gripper` `robot.pick_actor` `robot.remove_objs_from_obstacle` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig`

## Scene and action tags

`drawer_scene` `extra_checkers` `intermediate_motion` `keywords` `pick` `sequential_inspection` `table_with_articulation` `unmovable_objects`

## Scene keyword sample

`building_block` `left` `middle` `right` `table` `container` `contains_block` `cube` `drawer` `furniture` `hidden_block` `inside_drawer` `rectangular` `regular_drawer_left` `regular_drawer_middle` `storage_box` `target` `target_drawer_right` `target_object` `wood`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1580732220_becc7bbc` | [scene_info.json](1580732220_becc7bbc/scene_info.json) | [scene.usda](1580732220_becc7bbc/scene.usda) |
| `2204737360_becc7bbc` | [scene_info.json](2204737360_becc7bbc/scene_info.json) | [scene.usda](2204737360_becc7bbc/scene.usda) |
| `2782889667_becc7bbc` | [scene_info.json](2782889667_becc7bbc/scene_info.json) | [scene.usda](2782889667_becc7bbc/scene.usda) |
| `2904808774_becc7bbc` | [scene_info.json](2904808774_becc7bbc/scene_info.json) | [scene.usda](2904808774_becc7bbc/scene.usda) |
| `3044239258_becc7bbc` | [scene_info.json](3044239258_becc7bbc/scene_info.json) | [scene.usda](3044239258_becc7bbc/scene.usda) |
| `3697819287_becc7bbc` | [scene_info.json](3697819287_becc7bbc/scene_info.json) | [scene.usda](3697819287_becc7bbc/scene.usda) |
| `3961776623_becc7bbc` | [scene_info.json](3961776623_becc7bbc/scene_info.json) | [scene.usda](3961776623_becc7bbc/scene.usda) |
| `4014574270_becc7bbc` | [scene_info.json](4014574270_becc7bbc/scene_info.json) | [scene.usda](4014574270_becc7bbc/scene.usda) |
| `4260858703_becc7bbc` | [scene_info.json](4260858703_becc7bbc/scene_info.json) | [scene.usda](4260858703_becc7bbc/scene.usda) |
| `697441168_becc7bbc` | [scene_info.json](697441168_becc7bbc/scene_info.json) | [scene.usda](697441168_becc7bbc/scene.usda) |
