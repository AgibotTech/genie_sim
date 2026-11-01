# Store billiard balls in a box

Task: `store_billiard_balls_in_box_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_size_in_local_frame` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.pick_and_place_actor_to_target_actor`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `keywords` `pick_and_place` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`billiard` `container` `storage_box` `table` `ball` `billiard_ball_0` `box` `desktop_with_legs` `on_desk` `target_container` `wood` `workspace_table` `right` `left` `billiard_ball_1`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1143779039_30f46f27` | [scene_info.json](1143779039_30f46f27/scene_info.json) | [scene.usda](1143779039_30f46f27/scene.usda) |
| `1497819141_30f46f27` | [scene_info.json](1497819141_30f46f27/scene_info.json) | [scene.usda](1497819141_30f46f27/scene.usda) |
| `1580850946_30f46f27` | [scene_info.json](1580850946_30f46f27/scene_info.json) | [scene.usda](1580850946_30f46f27/scene.usda) |
| `1656129646_30f46f27` | [scene_info.json](1656129646_30f46f27/scene_info.json) | [scene.usda](1656129646_30f46f27/scene.usda) |
| `1663821982_30f46f27` | [scene_info.json](1663821982_30f46f27/scene_info.json) | [scene.usda](1663821982_30f46f27/scene.usda) |
| `2025926442_30f46f27` | [scene_info.json](2025926442_30f46f27/scene_info.json) | [scene.usda](2025926442_30f46f27/scene.usda) |
| `2247063670_30f46f27` | [scene_info.json](2247063670_30f46f27/scene_info.json) | [scene.usda](2247063670_30f46f27/scene.usda) |
| `2483845684_30f46f27` | [scene_info.json](2483845684_30f46f27/scene_info.json) | [scene.usda](2483845684_30f46f27/scene.usda) |
| `3967372750_30f46f27` | [scene_info.json](3967372750_30f46f27/scene_info.json) | [scene.usda](3967372750_30f46f27/scene.usda) |
| `4282118106_30f46f27` | [scene_info.json](4282118106_30f46f27/scene_info.json) | [scene.usda](4282118106_30f46f27/scene.usda) |
