# Place a block into a drawer

Task: `place_block_into_drawer_g1`. Robot: `G1`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.close_gripper` `robot.get_object_pose` `robot.get_object_prim_path` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose` `robot.pick_actor` `robot.place_actor_to_target_pose` `robot.remove_objs_from_obstacle`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig` `PlaceConfig`

## Scene and action tags

`drawer_scene` `extra_checkers` `intermediate_motion` `keywords` `pick_and_place` `table_with_articulation` `unmovable_objects`

## Scene keyword sample

`building_block` `drawer_cabinet` `colorful` `cube` `furniture` `ground_level` `single` `small` `storage` `wood`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1244915871_633a39af` | [scene_info.json](1244915871_633a39af/scene_info.json) | [scene.usda](1244915871_633a39af/scene.usda) |
| `1454641522_633a39af` | [scene_info.json](1454641522_633a39af/scene_info.json) | [scene.usda](1454641522_633a39af/scene.usda) |
| `2116696317_633a39af` | [scene_info.json](2116696317_633a39af/scene_info.json) | [scene.usda](2116696317_633a39af/scene.usda) |
| `229926329_633a39af` | [scene_info.json](229926329_633a39af/scene_info.json) | [scene.usda](229926329_633a39af/scene.usda) |
| `2775982316_633a39af` | [scene_info.json](2775982316_633a39af/scene_info.json) | [scene.usda](2775982316_633a39af/scene.usda) |
| `2777220690_633a39af` | [scene_info.json](2777220690_633a39af/scene_info.json) | [scene.usda](2777220690_633a39af/scene.usda) |
| `2831605280_633a39af` | [scene_info.json](2831605280_633a39af/scene_info.json) | [scene.usda](2831605280_633a39af/scene.usda) |
| `3010332944_633a39af` | [scene_info.json](3010332944_633a39af/scene_info.json) | [scene.usda](3010332944_633a39af/scene.usda) |
| `3700831688_633a39af` | [scene_info.json](3700831688_633a39af/scene_info.json) | [scene.usda](3700831688_633a39af/scene.usda) |
| `3959900707_633a39af` | [scene_info.json](3959900707_633a39af/scene_info.json) | [scene.usda](3959900707_633a39af/scene.usda) |
