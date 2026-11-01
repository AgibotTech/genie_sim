# Place a long block across two supports using both hands

Task: `place_long_block_across_two_supports_g1`. Robot: `G1`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.attach_obj` `robot.bimanual_rigid_move` `robot.dual_act` `robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_prim_path` `robot.get_object_size_in_world_frame` `robot.move_actor_to_target_pose` `robot.open_gripper` `robot.remove_objs_from_obstacle` `robot.wait`

### Client RPC

`robot.client.detach_obj` `robot.client.set_frame_state` `robot.client.set_task_task_basic_info`

### Configuration classes

`BimanualRigidMoveConfig` `DualMoveToPose` `GraspConfig` `MoveConfig`

## Scene and action tags

`bimanual` `intermediate_motion` `keywords` `path_constraint` `table_only` `unmovable_objects`

## Scene keyword sample

`bridge_support` `left` `lift_block` `right` `workspace_table` `long_block` `on_table` `table` `yellow`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `2026111601_166a0404` | [scene_info.json](2026111601_166a0404/scene_info.json) | [scene.usda](2026111601_166a0404/scene.usda) |
| `2026111613_166a0404` | [scene_info.json](2026111613_166a0404/scene_info.json) | [scene.usda](2026111613_166a0404/scene.usda) |
| `2026111779_166a0404` | [scene_info.json](2026111779_166a0404/scene_info.json) | [scene.usda](2026111779_166a0404/scene.usda) |
| `2026111846_166a0404` | [scene_info.json](2026111846_166a0404/scene_info.json) | [scene.usda](2026111846_166a0404/scene.usda) |
| `2026111920_166a0404` | [scene_info.json](2026111920_166a0404/scene_info.json) | [scene.usda](2026111920_166a0404/scene.usda) |
| `2026111974_166a0404` | [scene_info.json](2026111974_166a0404/scene_info.json) | [scene.usda](2026111974_166a0404/scene.usda) |
| `2026112076_166a0404` | [scene_info.json](2026112076_166a0404/scene_info.json) | [scene.usda](2026112076_166a0404/scene.usda) |
| `2026112241_166a0404` | [scene_info.json](2026112241_166a0404/scene_info.json) | [scene.usda](2026112241_166a0404/scene.usda) |
| `2026112404_166a0404` | [scene_info.json](2026112404_166a0404/scene_info.json) | [scene.usda](2026112404_166a0404/scene.usda) |
| `2026112462_166a0404` | [scene_info.json](2026112462_166a0404/scene_info.json) | [scene.usda](2026112462_166a0404/scene.usda) |
