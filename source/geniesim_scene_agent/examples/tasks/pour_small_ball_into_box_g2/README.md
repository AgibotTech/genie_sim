# Pour a small ball into a box

Task: `pour_small_ball_into_box_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.get_object_pose` `robot.get_object_size_in_local_frame` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose` `robot.pick_actor` `robot.rotate_actor`

### Client RPC

`robot.client.run_step` `robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig` `RotateConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `pick` `pour` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`ball` `container` `paper_cup` `storage_box` `cylinder` `furniture` `plastic` `rectangular` `small` `sphere` `table` `toy` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1812959663_ec9fa11c` | [scene_info.json](1812959663_ec9fa11c/scene_info.json) | [scene.usda](1812959663_ec9fa11c/scene.usda) |
| `1966889009_ec9fa11c` | [scene_info.json](1966889009_ec9fa11c/scene_info.json) | [scene.usda](1966889009_ec9fa11c/scene.usda) |
| `2927811337_ec9fa11c` | [scene_info.json](2927811337_ec9fa11c/scene_info.json) | [scene.usda](2927811337_ec9fa11c/scene.usda) |
| `3287041433_ec9fa11c` | [scene_info.json](3287041433_ec9fa11c/scene_info.json) | [scene.usda](3287041433_ec9fa11c/scene.usda) |
| `3600827969_ec9fa11c` | [scene_info.json](3600827969_ec9fa11c/scene_info.json) | [scene.usda](3600827969_ec9fa11c/scene.usda) |
| `3684263976_ec9fa11c` | [scene_info.json](3684263976_ec9fa11c/scene_info.json) | [scene.usda](3684263976_ec9fa11c/scene.usda) |
| `543167404_ec9fa11c` | [scene_info.json](543167404_ec9fa11c/scene_info.json) | [scene.usda](543167404_ec9fa11c/scene.usda) |
| `83075227_ec9fa11c` | [scene_info.json](83075227_ec9fa11c/scene_info.json) | [scene.usda](83075227_ec9fa11c/scene.usda) |
| `982383334_ec9fa11c` | [scene_info.json](982383334_ec9fa11c/scene_info.json) | [scene.usda](982383334_ec9fa11c/scene.usda) |
| `994128006_ec9fa11c` | [scene_info.json](994128006_ec9fa11c/scene_info.json) | [scene.usda](994128006_ec9fa11c/scene.usda) |
