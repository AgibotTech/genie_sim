# Place stationery into a holder held by the other hand

Task: `place_stationery_into_held_holder_g2`. Robot: `G2`.

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

`extra_checkers` `held_container` `intermediate_motion` `keywords` `pick_and_place` `table_only` `unmovable_objects`

## Scene keyword sample

`container` `pen_holder` `table` `target_container` `glue_stick` `eraser` `glue_bottle` `figurine` `POSITION_TAG_left` `POSITION_TAG_right` `furniture` `on_desk` `stationery` `stationery_item_to_place` `to_place` `workspace`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1067574464_a2b2e45a` | [scene_info.json](1067574464_a2b2e45a/scene_info.json) | [scene.usda](1067574464_a2b2e45a/scene.usda) |
| `2612563869_a2b2e45a` | [scene_info.json](2612563869_a2b2e45a/scene_info.json) | [scene.usda](2612563869_a2b2e45a/scene.usda) |
| `2725255908_a2b2e45a` | [scene_info.json](2725255908_a2b2e45a/scene_info.json) | [scene.usda](2725255908_a2b2e45a/scene.usda) |
| `2970915539_a2b2e45a` | [scene_info.json](2970915539_a2b2e45a/scene_info.json) | [scene.usda](2970915539_a2b2e45a/scene.usda) |
| `3805558290_a2b2e45a` | [scene_info.json](3805558290_a2b2e45a/scene_info.json) | [scene.usda](3805558290_a2b2e45a/scene.usda) |
| `4123141667_a2b2e45a` | [scene_info.json](4123141667_a2b2e45a/scene_info.json) | [scene.usda](4123141667_a2b2e45a/scene.usda) |
| `4227476052_a2b2e45a` | [scene_info.json](4227476052_a2b2e45a/scene_info.json) | [scene.usda](4227476052_a2b2e45a/scene.usda) |
| `428309317_a2b2e45a` | [scene_info.json](428309317_a2b2e45a/scene_info.json) | [scene.usda](428309317_a2b2e45a/scene.usda) |
| `883719879_a2b2e45a` | [scene_info.json](883719879_a2b2e45a/scene_info.json) | [scene.usda](883719879_a2b2e45a/scene.usda) |
| `906386735_a2b2e45a` | [scene_info.json](906386735_a2b2e45a/scene_info.json) | [scene.usda](906386735_a2b2e45a/scene.usda) |
