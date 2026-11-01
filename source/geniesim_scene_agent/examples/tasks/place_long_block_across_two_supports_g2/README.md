# Place a long block across two supports using both hands

Task: `place_long_block_across_two_supports_g2`. Robot: `G2`.

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
| `2026121696_166a0404` | [scene_info.json](2026121696_166a0404/scene_info.json) | [scene.usda](2026121696_166a0404/scene.usda) |
| `2026121794_166a0404` | [scene_info.json](2026121794_166a0404/scene_info.json) | [scene.usda](2026121794_166a0404/scene.usda) |
| `2026121879_166a0404` | [scene_info.json](2026121879_166a0404/scene_info.json) | [scene.usda](2026121879_166a0404/scene.usda) |
| `2026121980_166a0404` | [scene_info.json](2026121980_166a0404/scene_info.json) | [scene.usda](2026121980_166a0404/scene.usda) |
| `2026122024_166a0404` | [scene_info.json](2026122024_166a0404/scene_info.json) | [scene.usda](2026122024_166a0404/scene.usda) |
| `2026122189_166a0404` | [scene_info.json](2026122189_166a0404/scene_info.json) | [scene.usda](2026122189_166a0404/scene.usda) |
| `2026122221_166a0404` | [scene_info.json](2026122221_166a0404/scene_info.json) | [scene.usda](2026122221_166a0404/scene.usda) |
| `2026122268_166a0404` | [scene_info.json](2026122268_166a0404/scene_info.json) | [scene.usda](2026122268_166a0404/scene.usda) |
| `2026122292_166a0404` | [scene_info.json](2026122292_166a0404/scene_info.json) | [scene.usda](2026122292_166a0404/scene.usda) |
| `2026122481_166a0404` | [scene_info.json](2026122481_166a0404/scene_info.json) | [scene.usda](2026122481_166a0404/scene.usda) |
