# Pass a chips bucket between hands and place it on the opposite side

Task: `hand_over_chips_bucket_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_ee_pose` `robot.get_object_pose` `robot.handover_ownership_of_attached_object` `robot.move_actor_to_target_pose` `robot.pick_actor` `robot.place_actor_to_target_pose` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `pick_and_place` `table_only`

## Scene keyword sample

`chips_bucket` `snack` `left` `right` `cylinder` `desktop_with_legs` `food` `table` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1441352262_6fb2e28d` | [scene_info.json](1441352262_6fb2e28d/scene_info.json) | [scene.usda](1441352262_6fb2e28d/scene.usda) |
| `1576563708_6fb2e28d` | [scene_info.json](1576563708_6fb2e28d/scene_info.json) | [scene.usda](1576563708_6fb2e28d/scene.usda) |
| `1812174037_6fb2e28d` | [scene_info.json](1812174037_6fb2e28d/scene_info.json) | [scene.usda](1812174037_6fb2e28d/scene.usda) |
| `2193731359_6fb2e28d` | [scene_info.json](2193731359_6fb2e28d/scene_info.json) | [scene.usda](2193731359_6fb2e28d/scene.usda) |
| `272235000_6fb2e28d` | [scene_info.json](272235000_6fb2e28d/scene_info.json) | [scene.usda](272235000_6fb2e28d/scene.usda) |
| `3241861309_6fb2e28d` | [scene_info.json](3241861309_6fb2e28d/scene_info.json) | [scene.usda](3241861309_6fb2e28d/scene.usda) |
| `3329839822_6fb2e28d` | [scene_info.json](3329839822_6fb2e28d/scene_info.json) | [scene.usda](3329839822_6fb2e28d/scene.usda) |
| `3387636579_6fb2e28d` | [scene_info.json](3387636579_6fb2e28d/scene_info.json) | [scene.usda](3387636579_6fb2e28d/scene.usda) |
| `3726451088_6fb2e28d` | [scene_info.json](3726451088_6fb2e28d/scene_info.json) | [scene.usda](3726451088_6fb2e28d/scene.usda) |
| `3981780417_6fb2e28d` | [scene_info.json](3981780417_6fb2e28d/scene_info.json) | [scene.usda](3981780417_6fb2e28d/scene.usda) |
