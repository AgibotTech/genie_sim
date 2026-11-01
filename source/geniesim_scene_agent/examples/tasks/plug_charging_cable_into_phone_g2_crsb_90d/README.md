# Plug a charging cable into a phone

Task: `plug_charging_cable_into_phone_g2_crsb_90d`. Robot: `G2_crsB_90d`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.draw_debug_cube` `robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_prim_path` `robot.move_actor_to_target_pose` `robot.open_gripper` `robot.pick_actor` `robot.remove_objs_from_obstacle` `robot.solve_ik` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig`

## Scene and action tags

`insertion` `intermediate_motion` `keywords` `path_constraint` `pick` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`left` `left_table` `phone_holder` `power_strip_storage_box` `right` `target_phone` `target_plug` `furniture` `holder` `left_pick_target` `phone` `plug` `right_pick_target` `stand` `storage_box` `table` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1147954434_16d7e01c` | [scene_info.json](1147954434_16d7e01c/scene_info.json) | [scene.usda](1147954434_16d7e01c/scene.usda) |
| `1167859635_16d7e01c` | [scene_info.json](1167859635_16d7e01c/scene_info.json) | [scene.usda](1167859635_16d7e01c/scene.usda) |
| `1387499189_16d7e01c` | [scene_info.json](1387499189_16d7e01c/scene_info.json) | [scene.usda](1387499189_16d7e01c/scene.usda) |
| `2384340650_16d7e01c` | [scene_info.json](2384340650_16d7e01c/scene_info.json) | [scene.usda](2384340650_16d7e01c/scene.usda) |
| `28857805_16d7e01c` | [scene_info.json](28857805_16d7e01c/scene_info.json) | [scene.usda](28857805_16d7e01c/scene.usda) |
| `3590454526_16d7e01c` | [scene_info.json](3590454526_16d7e01c/scene_info.json) | [scene.usda](3590454526_16d7e01c/scene.usda) |
| `3643762820_16d7e01c` | [scene_info.json](3643762820_16d7e01c/scene_info.json) | [scene.usda](3643762820_16d7e01c/scene.usda) |
| `4183209837_16d7e01c` | [scene_info.json](4183209837_16d7e01c/scene_info.json) | [scene.usda](4183209837_16d7e01c/scene.usda) |
| `877737869_16d7e01c` | [scene_info.json](877737869_16d7e01c/scene_info.json) | [scene.usda](877737869_16d7e01c/scene.usda) |
| `955675969_16d7e01c` | [scene_info.json](955675969_16d7e01c/scene_info.json) | [scene.usda](955675969_16d7e01c/scene.usda) |
