# Fold a double-layer T-shirt with both arms

Task: `fold_double_layer_t_shirt_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.disable_gripper_colliders` `robot.draw_debug_cube` `robot.dual_act` `robot.get_cloth_nodal_positions` `robot.get_ee_pose` `robot.get_object_prim_path` `robot.move_actor_to_target_pose` `robot.open_gripper` `robot.pick_actor` `robot.release_fem_pin` `robot.remove_objs_from_obstacle` `robot.set_fem_pin` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`DualMoveToPose` `DualPick` `GraspConfig` `MoveConfig`

## Scene and action tags

`bimanual` `deformable_cloth` `fold` `intermediate_motion` `keywords` `path_constraint` `pick` `table_only` `unmovable_objects`

## Scene keyword sample

`cloth` `deformable` `fem` `table` `workspace_table` `collision_proxy` `double_layer` `fold_target` `furniture` `motion_gen_obstacle` `rectangular` `table_collision_proxy` `target_cloth` `tshirt` `wood`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1108067013_de8f8c1b` | [scene_info.json](1108067013_de8f8c1b/scene_info.json) | [scene.usda](1108067013_de8f8c1b/scene.usda) |
| `1621888175_de8f8c1b` | [scene_info.json](1621888175_de8f8c1b/scene_info.json) | [scene.usda](1621888175_de8f8c1b/scene.usda) |
| `1780785630_de8f8c1b` | [scene_info.json](1780785630_de8f8c1b/scene_info.json) | [scene.usda](1780785630_de8f8c1b/scene.usda) |
| `2470982955_de8f8c1b` | [scene_info.json](2470982955_de8f8c1b/scene_info.json) | [scene.usda](2470982955_de8f8c1b/scene.usda) |
| `2501906260_de8f8c1b` | [scene_info.json](2501906260_de8f8c1b/scene_info.json) | [scene.usda](2501906260_de8f8c1b/scene.usda) |
| `3051625067_de8f8c1b` | [scene_info.json](3051625067_de8f8c1b/scene_info.json) | [scene.usda](3051625067_de8f8c1b/scene.usda) |
| `3912758451_de8f8c1b` | [scene_info.json](3912758451_de8f8c1b/scene_info.json) | [scene.usda](3912758451_de8f8c1b/scene.usda) |
| `659073756_de8f8c1b` | [scene_info.json](659073756_de8f8c1b/scene_info.json) | [scene.usda](659073756_de8f8c1b/scene.usda) |
| `752311345_de8f8c1b` | [scene_info.json](752311345_de8f8c1b/scene_info.json) | [scene.usda](752311345_de8f8c1b/scene.usda) |
| `87028019_de8f8c1b` | [scene_info.json](87028019_de8f8c1b/scene_info.json) | [scene.usda](87028019_de8f8c1b/scene.usda) |
