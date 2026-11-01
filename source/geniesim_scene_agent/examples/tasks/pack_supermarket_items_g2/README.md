# Pack supermarket items

Task: `pack_supermarket_items_g2`. Robot: `G2`.

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

`handbag` `product` `table` `orange` `apple` `green` `bag` `desktop_with_legs` `left_area` `on_desk` `white` `workspace_table` `red` `red_bull` `banana` `green_apple` `yellow`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1156137387_054ca88e` | [scene_info.json](1156137387_054ca88e/scene_info.json) | [scene.usda](1156137387_054ca88e/scene.usda) |
| `1703184253_054ca88e` | [scene_info.json](1703184253_054ca88e/scene_info.json) | [scene.usda](1703184253_054ca88e/scene.usda) |
| `2329326631_054ca88e` | [scene_info.json](2329326631_054ca88e/scene_info.json) | [scene.usda](2329326631_054ca88e/scene.usda) |
| `2617131227_054ca88e` | [scene_info.json](2617131227_054ca88e/scene_info.json) | [scene.usda](2617131227_054ca88e/scene.usda) |
| `278680090_054ca88e` | [scene_info.json](278680090_054ca88e/scene_info.json) | [scene.usda](278680090_054ca88e/scene.usda) |
| `294174434_054ca88e` | [scene_info.json](294174434_054ca88e/scene_info.json) | [scene.usda](294174434_054ca88e/scene.usda) |
| `3220279124_054ca88e` | [scene_info.json](3220279124_054ca88e/scene_info.json) | [scene.usda](3220279124_054ca88e/scene.usda) |
| `3527303139_054ca88e` | [scene_info.json](3527303139_054ca88e/scene_info.json) | [scene.usda](3527303139_054ca88e/scene.usda) |
| `515066684_054ca88e` | [scene_info.json](515066684_054ca88e/scene_info.json) | [scene.usda](515066684_054ca88e/scene.usda) |
| `695950752_054ca88e` | [scene_info.json](695950752_054ca88e/scene_info.json) | [scene.usda](695950752_054ca88e/scene.usda) |
