# genie_sim_rviz_plugins

Custom RViz2 display plugins for the GenieSim stack.

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_rviz_plugins/](.)
License: [Mozilla Public License Version 2.0](../../../../LICENSE)

**Maintenance contract**: when you add a plugin, update `plugin_view_camera_pose.xml`
and this file in the same diff.

---

## Layout

```
genie_sim_rviz_plugins/
├── src/
│   └── view_camera_pose_publisher_display.cpp ← camera pose publisher display
├── include/
│   └── genie_sim_rviz_plugins/                ← C++ headers
└── plugin_view_camera_pose.xml                ← pluginlib plugin registration
```

---

## Plugins

### `ViewCameraPosePublisherDisplay`

RViz2 display panel that publishes the current RViz2 camera pose as a
`geometry_msgs/PoseStamped` on a configurable topic. Used by the render node
to drive the free-fly camera in the USD stage.

Published topic: configurable in the RViz2 panel (default:
`/genie_sim_engine/viewer/camera_pose`).

---

## Build

C++ CMake package. Depends on `rviz_common`, `rviz_rendering`,
`geometry_msgs`, `pluginlib`.
Shared CMake conventions come from `genie_sim_common` via `genie_sim_package()`.

```bash
colcon build --packages-select genie_sim_rviz_plugins
```

---

## Routing rules

- Camera pose publisher display → `src/view_camera_pose_publisher_display.cpp`
- Plugin registration → `plugin_view_camera_pose.xml`
- Free cam pose consumer → `../genie_sim_render/src/render_node.cpp`
