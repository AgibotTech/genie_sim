# Sort beverages by type

Task: `sort_beverages_by_type_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.pick_and_place_actor_to_target_pose`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `keywords` `pick_and_place` `sort` `table_only` `unmovable_objects`

## Scene keyword sample

`table` `sprite_can` `wang_lao_ji_can` `coca_cola_can` `fanta_can` `aluminum_can` `carbonated_drink` `rectangular_desk` `white` `workspace_table` `red` `green` `herbal_tea` `orange`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1833703629_52275988` | [scene_info.json](1833703629_52275988/scene_info.json) | [scene.usda](1833703629_52275988/scene.usda) |
| `1870203616_52275988` | [scene_info.json](1870203616_52275988/scene_info.json) | [scene.usda](1870203616_52275988/scene.usda) |
| `2064097925_52275988` | [scene_info.json](2064097925_52275988/scene_info.json) | [scene.usda](2064097925_52275988/scene.usda) |
| `2308233964_52275988` | [scene_info.json](2308233964_52275988/scene_info.json) | [scene.usda](2308233964_52275988/scene.usda) |
| `2462387493_52275988` | [scene_info.json](2462387493_52275988/scene_info.json) | [scene.usda](2462387493_52275988/scene.usda) |
| `2499961822_52275988` | [scene_info.json](2499961822_52275988/scene_info.json) | [scene.usda](2499961822_52275988/scene.usda) |
| `2658733902_52275988` | [scene_info.json](2658733902_52275988/scene_info.json) | [scene.usda](2658733902_52275988/scene.usda) |
| `4085998410_52275988` | [scene_info.json](4085998410_52275988/scene_info.json) | [scene.usda](4085998410_52275988/scene.usda) |
| `4104653121_52275988` | [scene_info.json](4104653121_52275988/scene_info.json) | [scene.usda](4104653121_52275988/scene.usda) |
| `473123092_52275988` | [scene_info.json](473123092_52275988/scene_info.json) | [scene.usda](473123092_52275988/scene.usda) |
