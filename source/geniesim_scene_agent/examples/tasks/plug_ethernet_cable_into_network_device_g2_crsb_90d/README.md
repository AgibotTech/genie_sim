# Plug an Ethernet cable into a network device

Task: `plug_ethernet_cable_into_network_device_g2_crsb_90d`. Robot: `G2_crsB_90d`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.draw_debug_cube` `robot.follow_trajectory` `robot.get_object_pose` `robot.get_object_prim_path` `robot.move_actor_to_target_pose` `robot.pick_actor` `robot.remove_objs_from_obstacle` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`FollowTrajectoryConfig` `GraspConfig` `MoveConfig`

## Scene and action tags

`follow_trajectory` `insertion` `intermediate_motion` `keywords` `pick` `table_only` `unmovable_objects`

## Scene keyword sample

`ethernet_plug` `ethernet_router` `table` `left_hand` `dynamic` `fixed` `kinematic` `network_cable` `network_device` `workspace`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1811962896_c54a873c` | [scene_info.json](1811962896_c54a873c/scene_info.json) | [scene.usda](1811962896_c54a873c/scene.usda) |
| `2035569937_c54a873c` | [scene_info.json](2035569937_c54a873c/scene_info.json) | [scene.usda](2035569937_c54a873c/scene.usda) |
| `2506948011_c54a873c` | [scene_info.json](2506948011_c54a873c/scene_info.json) | [scene.usda](2506948011_c54a873c/scene.usda) |
| `3259999773_c54a873c` | [scene_info.json](3259999773_c54a873c/scene_info.json) | [scene.usda](3259999773_c54a873c/scene.usda) |
| `3312310963_c54a873c` | [scene_info.json](3312310963_c54a873c/scene_info.json) | [scene.usda](3312310963_c54a873c/scene.usda) |
| `3815914314_c54a873c` | [scene_info.json](3815914314_c54a873c/scene_info.json) | [scene.usda](3815914314_c54a873c/scene.usda) |
| `3872847706_c54a873c` | [scene_info.json](3872847706_c54a873c/scene_info.json) | [scene.usda](3872847706_c54a873c/scene.usda) |
| `3881106877_c54a873c` | [scene_info.json](3881106877_c54a873c/scene_info.json) | [scene.usda](3881106877_c54a873c/scene.usda) |
| `4231973193_c54a873c` | [scene_info.json](4231973193_c54a873c/scene_info.json) | [scene.usda](4231973193_c54a873c/scene.usda) |
| `678167304_c54a873c` | [scene_info.json](678167304_c54a873c/scene_info.json) | [scene.usda](678167304_c54a873c/scene.usda) |
