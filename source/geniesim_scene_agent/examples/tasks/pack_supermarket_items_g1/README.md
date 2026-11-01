# Pack supermarket items

Task: `pack_supermarket_items_g1`. Robot: `G1`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_size_in_local_frame` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.pick_actor` `robot.place_actor_to_target_pose`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `keywords` `pick_and_place` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`handbag` `product` `table` `apple` `green` `orange` `bag` `desktop_with_legs` `left_area` `on_desk` `white` `workspace_table` `red` `red_bull` `green_apple`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1536131510_054ca88e` | [scene_info.json](1536131510_054ca88e/scene_info.json) | [scene.usda](1536131510_054ca88e/scene.usda) |
| `207425482_054ca88e` | [scene_info.json](207425482_054ca88e/scene_info.json) | [scene.usda](207425482_054ca88e/scene.usda) |
| `2265384835_054ca88e` | [scene_info.json](2265384835_054ca88e/scene_info.json) | [scene.usda](2265384835_054ca88e/scene.usda) |
| `2615183127_054ca88e` | [scene_info.json](2615183127_054ca88e/scene_info.json) | [scene.usda](2615183127_054ca88e/scene.usda) |
| `2646436831_054ca88e` | [scene_info.json](2646436831_054ca88e/scene_info.json) | [scene.usda](2646436831_054ca88e/scene.usda) |
| `3568902720_054ca88e` | [scene_info.json](3568902720_054ca88e/scene_info.json) | [scene.usda](3568902720_054ca88e/scene.usda) |
| `3621059068_054ca88e` | [scene_info.json](3621059068_054ca88e/scene_info.json) | [scene.usda](3621059068_054ca88e/scene.usda) |
| `4093725700_054ca88e` | [scene_info.json](4093725700_054ca88e/scene_info.json) | [scene.usda](4093725700_054ca88e/scene.usda) |
| `4284608049_054ca88e` | [scene_info.json](4284608049_054ca88e/scene_info.json) | [scene.usda](4284608049_054ca88e/scene.usda) |
| `546800895_054ca88e` | [scene_info.json](546800895_054ca88e/scene_info.json) | [scene.usda](546800895_054ca88e/scene.usda) |
