# Open a trash can

Task: `open_trash_can_g1`. Robot: `G1`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.close_gripper` `robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_prim_path` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose`

### Client RPC

`robot.client.remove_objs_from_obstacle` `robot.client.set_task_task_basic_info`

### Configuration classes

`MoveConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `open_action` `table_with_articulation` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`table` `trash_can` `black` `cylinder` `desktop_with_legs` `waste_container` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1175741158_872e3622` | [scene_info.json](1175741158_872e3622/scene_info.json) | [scene.usda](1175741158_872e3622/scene.usda) |
| `1877327591_872e3622` | [scene_info.json](1877327591_872e3622/scene_info.json) | [scene.usda](1877327591_872e3622/scene.usda) |
| `2126434178_872e3622` | [scene_info.json](2126434178_872e3622/scene_info.json) | [scene.usda](2126434178_872e3622/scene.usda) |
| `2315916732_872e3622` | [scene_info.json](2315916732_872e3622/scene_info.json) | [scene.usda](2315916732_872e3622/scene.usda) |
| `2666255249_872e3622` | [scene_info.json](2666255249_872e3622/scene_info.json) | [scene.usda](2666255249_872e3622/scene.usda) |
| `3438471883_872e3622` | [scene_info.json](3438471883_872e3622/scene_info.json) | [scene.usda](3438471883_872e3622/scene.usda) |
| `4011733461_872e3622` | [scene_info.json](4011733461_872e3622/scene_info.json) | [scene.usda](4011733461_872e3622/scene.usda) |
| `4197085419_872e3622` | [scene_info.json](4197085419_872e3622/scene_info.json) | [scene.usda](4197085419_872e3622/scene.usda) |
| `4725685_872e3622` | [scene_info.json](4725685_872e3622/scene_info.json) | [scene.usda](4725685_872e3622/scene.usda) |
| `754080744_872e3622` | [scene_info.json](754080744_872e3622/scene_info.json) | [scene.usda](754080744_872e3622/scene.usda) |
