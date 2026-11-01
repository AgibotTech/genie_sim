# Pick a block matching the reference color

Task: `pick_block_matching_reference_color_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.judge_arm_to_use` `robot.pick_actor`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig`

## Scene and action tags

`keywords` `pick` `table_only` `unmovable_objects`

## Scene keyword sample

`building_block` `candidate_block` `coaster` `matching` `on_coaster` `reference_block` `table` `red` `blue` `yellow` `purple` `desktop_with_legs` `fixed` `on_desk` `reference_platform` `right` `round` `scattered` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1472563149_665eedea` | [scene_info.json](1472563149_665eedea/scene_info.json) | [scene.usda](1472563149_665eedea/scene.usda) |
| `2254043991_665eedea` | [scene_info.json](2254043991_665eedea/scene_info.json) | [scene.usda](2254043991_665eedea/scene.usda) |
| `2263678176_665eedea` | [scene_info.json](2263678176_665eedea/scene_info.json) | [scene.usda](2263678176_665eedea/scene.usda) |
| `3232967422_665eedea` | [scene_info.json](3232967422_665eedea/scene_info.json) | [scene.usda](3232967422_665eedea/scene.usda) |
| `3316656593_665eedea` | [scene_info.json](3316656593_665eedea/scene_info.json) | [scene.usda](3316656593_665eedea/scene.usda) |
| `361066645_665eedea` | [scene_info.json](361066645_665eedea/scene_info.json) | [scene.usda](361066645_665eedea/scene.usda) |
| `3759688164_665eedea` | [scene_info.json](3759688164_665eedea/scene_info.json) | [scene.usda](3759688164_665eedea/scene.usda) |
| `4168388649_665eedea` | [scene_info.json](4168388649_665eedea/scene_info.json) | [scene.usda](4168388649_665eedea/scene.usda) |
| `474207373_665eedea` | [scene_info.json](474207373_665eedea/scene_info.json) | [scene.usda](474207373_665eedea/scene.usda) |
| `799239243_665eedea` | [scene_info.json](799239243_665eedea/scene_info.json) | [scene.usda](799239243_665eedea/scene.usda) |
