# Place a fruit into a bowl held by the other hand

Task: `place_fruit_into_held_bowl_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.attach_obj` `robot.back_to_origin` `robot.detach_obj` `robot.get_object_pose` `robot.get_object_prim_path` `robot.get_object_size_in_world_frame` `robot.move_actor_to_target_pose` `robot.pick_actor` `robot.place_actor_to_target_pose` `robot.set_object_parameter_action` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `held_container` `intermediate_motion` `keywords` `pick_and_place` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`bowl` `container` `fruit` `table` `bowl_025` `furniture` `on_desk` `target_container` `to_place` `workspace`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `2074590548_2a1d205b` | [scene_info.json](2074590548_2a1d205b/scene_info.json) | [scene.usda](2074590548_2a1d205b/scene.usda) |
| `2300368622_2a1d205b` | [scene_info.json](2300368622_2a1d205b/scene_info.json) | [scene.usda](2300368622_2a1d205b/scene.usda) |
| `240939207_2a1d205b` | [scene_info.json](240939207_2a1d205b/scene_info.json) | [scene.usda](240939207_2a1d205b/scene.usda) |
| `2581971614_2a1d205b` | [scene_info.json](2581971614_2a1d205b/scene_info.json) | [scene.usda](2581971614_2a1d205b/scene.usda) |
| `2800094862_2a1d205b` | [scene_info.json](2800094862_2a1d205b/scene_info.json) | [scene.usda](2800094862_2a1d205b/scene.usda) |
| `3615963191_2a1d205b` | [scene_info.json](3615963191_2a1d205b/scene_info.json) | [scene.usda](3615963191_2a1d205b/scene.usda) |
| `3648470162_2a1d205b` | [scene_info.json](3648470162_2a1d205b/scene_info.json) | [scene.usda](3648470162_2a1d205b/scene.usda) |
| `3857759727_2a1d205b` | [scene_info.json](3857759727_2a1d205b/scene_info.json) | [scene.usda](3857759727_2a1d205b/scene.usda) |
| `4045514100_2a1d205b` | [scene_info.json](4045514100_2a1d205b/scene_info.json) | [scene.usda](4045514100_2a1d205b/scene.usda) |
| `852533373_2a1d205b` | [scene_info.json](852533373_2a1d205b/scene_info.json) | [scene.usda](852533373_2a1d205b/scene.usda) |
