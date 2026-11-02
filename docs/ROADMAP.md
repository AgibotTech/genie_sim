# Platform roadmap

Current workflows are indexed in the [module map](../source/README.md). This file records proposals and open work, not supported interfaces or delivery commitments.

| Area | Open work | Current contract / owner |
|---|---|---|
| Runtime release | Complete image-specific Isaac 6.2 qualification, fresh SDK download reproducibility and publication; qualify bundled Isaac 5.1 independently | [Runtime acceptance](../source/geniesim_ros/docs/ISAACSIM_ACCEPTANCE.md) |
| Task coverage | Expand G2 WBC, navigation and long-run scene validation | [Bringup test matrix](../source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/README.md#scene-testing) |
| Physics | Broaden Newton solver coverage with explicit backend acceptance | [ROS roadmap](../source/geniesim_ros/docs/ROADMAP.md) |
| Benchmark runtime | Continue moving benchmark execution onto RT Engine client interfaces while retaining task-specific configuration contracts | [Benchmark usage](../source/geniesim_benchmark/benchmark_test_docs.md) |
| Recording | Improve recording beyond the current rosbag workflow | [Recording workflow](../source/geniesim_ros/skills/record-episode/SKILL.md) |
| World generation | Investigate text/sparse-image inputs and loading generated Gaussians through RT Engine scene YAML | [World pipeline](../source/geniesim_world/README.md) currently accepts panoramas and exports to its own tooling |
| Data distribution | Extend published asset/dataset availability | [Bootstrap guide](../source/geniesim_cli/AGENTS.md) |

Each implementation must update the owning current guide and record its validation before being described as supported. Roadmap items do not change existing configuration or compatibility contracts.
