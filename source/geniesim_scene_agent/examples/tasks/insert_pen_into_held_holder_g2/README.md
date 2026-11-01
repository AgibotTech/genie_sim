# Insert a pen into a holder held by the other hand

Task: `insert_pen_into_held_holder_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.attach_obj` `robot.back_to_origin` `robot.detach_obj` `robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_prim_path` `robot.get_object_size_in_world_frame` `robot.move_actor_to_target_pose` `robot.pick_actor` `robot.place_actor_to_target_pose` `robot.remove_objs_from_obstacle` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `held_container` `insertion` `intermediate_motion` `keywords` `pick_and_place` `table_only` `unmovable_objects`

## Scene keyword sample

`container` `pen` `pen_holder` `table` `target_container` `furniture` `holder` `on_desk` `stationery` `stationery_021` `to_place` `workspace` `writing_tool`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1039481195_eb26cd0f` | [scene_info.json](1039481195_eb26cd0f/scene_info.json) | [scene.usda](1039481195_eb26cd0f/scene.usda) |
| `1428409552_eb26cd0f` | [scene_info.json](1428409552_eb26cd0f/scene_info.json) | [scene.usda](1428409552_eb26cd0f/scene.usda) |
| `1600618925_eb26cd0f` | [scene_info.json](1600618925_eb26cd0f/scene_info.json) | [scene.usda](1600618925_eb26cd0f/scene.usda) |
| `2410998007_eb26cd0f` | [scene_info.json](2410998007_eb26cd0f/scene_info.json) | [scene.usda](2410998007_eb26cd0f/scene.usda) |
| `2524695392_eb26cd0f` | [scene_info.json](2524695392_eb26cd0f/scene_info.json) | [scene.usda](2524695392_eb26cd0f/scene.usda) |
| `2913968121_eb26cd0f` | [scene_info.json](2913968121_eb26cd0f/scene_info.json) | [scene.usda](2913968121_eb26cd0f/scene.usda) |
| `2964259685_eb26cd0f` | [scene_info.json](2964259685_eb26cd0f/scene_info.json) | [scene.usda](2964259685_eb26cd0f/scene.usda) |
| `340236385_eb26cd0f` | [scene_info.json](340236385_eb26cd0f/scene_info.json) | [scene.usda](340236385_eb26cd0f/scene.usda) |
| `3623576849_eb26cd0f` | [scene_info.json](3623576849_eb26cd0f/scene_info.json) | [scene.usda](3623576849_eb26cd0f/scene.usda) |
| `766713074_eb26cd0f` | [scene_info.json](766713074_eb26cd0f/scene_info.json) | [scene.usda](766713074_eb26cd0f/scene.usda) |
