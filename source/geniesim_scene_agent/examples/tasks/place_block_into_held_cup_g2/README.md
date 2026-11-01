# Place a block into a cup held by the other hand

Task: `place_block_into_held_cup_g2`. Robot: `G2`.

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

`extra_checkers` `held_container` `intermediate_motion` `keywords` `pick_and_place` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`block` `building_blocks` `container` `cup` `table` `cup_107` `furniture` `on_desk` `target_container` `to_place` `toy` `workspace`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1029009286_cceb0fc1` | [scene_info.json](1029009286_cceb0fc1/scene_info.json) | [scene.usda](1029009286_cceb0fc1/scene.usda) |
| `1070165371_cceb0fc1` | [scene_info.json](1070165371_cceb0fc1/scene_info.json) | [scene.usda](1070165371_cceb0fc1/scene.usda) |
| `1742347177_cceb0fc1` | [scene_info.json](1742347177_cceb0fc1/scene_info.json) | [scene.usda](1742347177_cceb0fc1/scene.usda) |
| `2092364774_cceb0fc1` | [scene_info.json](2092364774_cceb0fc1/scene_info.json) | [scene.usda](2092364774_cceb0fc1/scene.usda) |
| `2302960054_cceb0fc1` | [scene_info.json](2302960054_cceb0fc1/scene_info.json) | [scene.usda](2302960054_cceb0fc1/scene.usda) |
| `2675303453_cceb0fc1` | [scene_info.json](2675303453_cceb0fc1/scene_info.json) | [scene.usda](2675303453_cceb0fc1/scene.usda) |
| `2858822717_cceb0fc1` | [scene_info.json](2858822717_cceb0fc1/scene_info.json) | [scene.usda](2858822717_cceb0fc1/scene.usda) |
| `3306966362_cceb0fc1` | [scene_info.json](3306966362_cceb0fc1/scene_info.json) | [scene.usda](3306966362_cceb0fc1/scene.usda) |
| `3342920839_cceb0fc1` | [scene_info.json](3342920839_cceb0fc1/scene_info.json) | [scene.usda](3342920839_cceb0fc1/scene.usda) |
| `971063609_cceb0fc1` | [scene_info.json](971063609_cceb0fc1/scene_info.json) | [scene.usda](971063609_cceb0fc1/scene.usda) |
