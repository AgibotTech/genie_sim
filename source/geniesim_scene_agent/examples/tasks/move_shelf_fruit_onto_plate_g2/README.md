# Move a fruit from the top shelf onto a plate on the middle shelf

Task: `move_shelf_fruit_onto_plate_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose` `robot.pick_actor` `robot.place_actor_to_target_pose` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `pick_and_place` `shelf_scene` `unmovable_objects`

## Scene keyword sample

`apple` `plate` `shelf` `target_plate` `target_shelf` `apple_0` `dish` `distractor` `edible` `fruit` `kitchenware` `left` `on_middle_layer` `on_top_layer` `orange` `orange_0` `storage_furniture` `supermarket_display` `tableware` `target`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1249073102_d30f7014` | [scene_info.json](1249073102_d30f7014/scene_info.json) | [scene.usda](1249073102_d30f7014/scene.usda) |
| `1494353374_d30f7014` | [scene_info.json](1494353374_d30f7014/scene_info.json) | [scene.usda](1494353374_d30f7014/scene.usda) |
| `215162856_d30f7014` | [scene_info.json](215162856_d30f7014/scene_info.json) | [scene.usda](215162856_d30f7014/scene.usda) |
| `2580756168_d30f7014` | [scene_info.json](2580756168_d30f7014/scene_info.json) | [scene.usda](2580756168_d30f7014/scene.usda) |
| `3746538490_d30f7014` | [scene_info.json](3746538490_d30f7014/scene_info.json) | [scene.usda](3746538490_d30f7014/scene.usda) |
| `3951237762_d30f7014` | [scene_info.json](3951237762_d30f7014/scene_info.json) | [scene.usda](3951237762_d30f7014/scene.usda) |
| `3953630936_d30f7014` | [scene_info.json](3953630936_d30f7014/scene_info.json) | [scene.usda](3953630936_d30f7014/scene.usda) |
| `474919839_d30f7014` | [scene_info.json](474919839_d30f7014/scene_info.json) | [scene.usda](474919839_d30f7014/scene.usda) |
| `614895317_d30f7014` | [scene_info.json](614895317_d30f7014/scene_info.json) | [scene.usda](614895317_d30f7014/scene.usda) |
| `695874243_d30f7014` | [scene_info.json](695874243_d30f7014/scene_info.json) | [scene.usda](695874243_d30f7014/scene.usda) |
