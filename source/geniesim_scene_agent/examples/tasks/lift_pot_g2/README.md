# Lift a pot

Task: `lift_pot_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.attach_obj` `robot.bimanual_rigid_move` `robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_prim_path` `robot.move_actor_to_target_pose`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`BimanualRigidMoveConfig` `MoveConfig`

## Scene and action tags

`bimanual` `extra_checkers` `intermediate_motion` `keywords` `table_only` `unmovable_objects`

## Scene keyword sample

`pot` `table` `cooking_pot` `furniture` `ground_level` `kitchenware` `on_table` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `2268254726_29bbebfd` | [scene_info.json](2268254726_29bbebfd/scene_info.json) | [scene.usda](2268254726_29bbebfd/scene.usda) |
| `2370829508_29bbebfd` | [scene_info.json](2370829508_29bbebfd/scene_info.json) | [scene.usda](2370829508_29bbebfd/scene.usda) |
| `2844886405_29bbebfd` | [scene_info.json](2844886405_29bbebfd/scene_info.json) | [scene.usda](2844886405_29bbebfd/scene.usda) |
| `3140210051_29bbebfd` | [scene_info.json](3140210051_29bbebfd/scene_info.json) | [scene.usda](3140210051_29bbebfd/scene.usda) |
| `3354305361_29bbebfd` | [scene_info.json](3354305361_29bbebfd/scene_info.json) | [scene.usda](3354305361_29bbebfd/scene.usda) |
| `3662456165_29bbebfd` | [scene_info.json](3662456165_29bbebfd/scene_info.json) | [scene.usda](3662456165_29bbebfd/scene.usda) |
| `3893686726_29bbebfd` | [scene_info.json](3893686726_29bbebfd/scene_info.json) | [scene.usda](3893686726_29bbebfd/scene.usda) |
| `4015002511_29bbebfd` | [scene_info.json](4015002511_29bbebfd/scene_info.json) | [scene.usda](4015002511_29bbebfd/scene.usda) |
| `4069802263_29bbebfd` | [scene_info.json](4069802263_29bbebfd/scene_info.json) | [scene.usda](4069802263_29bbebfd/scene.usda) |
| `63370768_29bbebfd` | [scene_info.json](63370768_29bbebfd/scene_info.json) | [scene.usda](63370768_29bbebfd/scene.usda) |
