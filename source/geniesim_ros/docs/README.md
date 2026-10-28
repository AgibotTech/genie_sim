# Genie Sim ROS documentation

Current runtime contracts and verification. Backend details live with their
owning packages.

| Document | Scope |
| --- | --- |
| [Scene tests](../src/ros_ws/src/genie_sim_bringup/README.md#scene-testing) | Tier 1 G2 and Tier 2 reference cases, commands and pass criteria |
| [Runtime verification](ISAACSIM_ACCEPTANCE.md) | Measured checks and remaining acceptance limits |
| [Step modes](STEP_MODES.md) | Realtime, lockstep, freerun and lifecycle contracts |
| [Contact tuning](CONTACT_TUNING.md) | Newton MuJoCo contact configuration |
| [Joint parameters](JOINT_PARAMS.md) | Sysid schema, selection and runtime precedence |
| [Roadmap](ROADMAP.md) | Unimplemented work and acceptance gates |

Implementation: [core contracts](../src/ros_ws/src/genie_sim_engine_core/README.md),
[Isaac](../src/ros_ws/src/genie_sim_engine_isaacsim/README.md),
[Newton](../src/ros_ws/src/genie_sim_engine_newton/README.md), and
[rendering](../src/ros_ws/src/genie_sim_render/README.md).
