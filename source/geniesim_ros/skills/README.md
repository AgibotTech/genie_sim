# geniesim_ros — skills index

Each `SKILL.md` is a self-contained recipe an agent (or human) can follow end-to-end. `cat` the file to read it.

| Skill | When to use |
|---|---|
| [build-workspace](build-workspace/SKILL.md) | After `geniesim docker into`, before any `ros2 launch …`. Builds the colcon overlay. |
| [launch-scene](launch-scene/SKILL.md) | Scene × launcher matrix — bring up `genie_sim_bringup` with a chosen physics backend. |
| [moveit-wbc](moveit-wbc/SKILL.md) | Plan + RViz on top of a running G2 scene. Arm/gripper matrix, IK plugin A/B, `/joint_command` mode. |
| [add-robot](add-robot/SKILL.md) | URDF / xacro intake, offline mesh-prep tools, AS3 layout, scene wiring. |
| [teleop-bridge](teleop-bridge/SKILL.md) | Wire `geniesim_teleop` into `/joint_command` without ros2_control fighting it. |
| [run-wuji-chef-newton](run-wuji-chef-newton/SKILL.md) | Start the Wuji chef scene with Newton XPBD: OpenGL by default, optional OVRtx with RViz or image-viewer instructions. |
| [record-episode](record-episode/SKILL.md) | `ros2 bag` canonical-topic capture and deformable replay conversion. |
| [debug-physics](debug-physics/SKILL.md) | Contact / init / backend cookbook — convex hull, init swing, tunnelling, backend bisection. |
| [material-override](material-override/SKILL.md) | Tune metallic / roughness PBR via `<material_override>` inside URDF `<visual>`. |

Recommended order for a new user: **build-workspace → launch-scene → (moveit-wbc | teleop-bridge | record-episode) → debug-physics if needed**.

For the canonical package table + AS2/AS3 dispatch + wheel layout, see [`../AGENTS.md`](../AGENTS.md).
For the engine overview + scene × launcher matrix, see [`../README.md`](../README.md).
