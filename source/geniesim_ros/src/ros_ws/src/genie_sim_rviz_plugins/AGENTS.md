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
├── test/test_camera_zoom.cpp                  ← Orbit policy and ROS zoom transport checks
└── plugin_view_camera_pose.xml                ← pluginlib plugin registration
```

---

## Plugins

### `ViewCameraPosePublisherDisplay`

RViz2 display panel that publishes the current RViz2 camera pose as a
`geometry_msgs/PoseStamped` on a configurable topic. Used by the render node
to drive the free-fly camera in the USD stage. It can also publish an optional
`std_msgs/Float64` optical-zoom factor for Newton inline OVRtx.

Published pose topic: configurable in the RViz2 panel (default:
`/genie_sim_engine/viewer/camera_pose`).

Published zoom topic: disabled by default (empty). Configure
`/genie_sim_engine/viewer/camera_zoom` for a renderer that consumes optical
zoom factors. Keep this channel additive: external
renderers that only consume pose must not acquire zoom semantics implicitly.
With `OVRtx Auto Zoom from Orbit` enabled, the display reads the active Orbit
controller's public `Distance` child property and accumulates
`previous_distance / current_distance` into the zoom factor. Do not cast to
`rviz_default_plugins::OrbitViewController`; that would add an unnecessary
runtime dependency on the implementation package.

---

## Build

C++ CMake package. Depends on `rviz_common`, `rviz_rendering`,
`geometry_msgs`, `std_msgs`, `pluginlib`.
Shared CMake conventions come from `genie_sim_common` via `genie_sim_package()`.

```bash
colcon build --packages-select genie_sim_rviz_plugins
```

With `BUILD_TESTING=ON`, `test_camera_zoom` checks the Orbit property contract,
factor limits, manual mode, topic isolation and transient-local ROS delivery.
It uses Qt offscreen and needs no Ogre camera or renderer. `publishZoom()` is
the send path shared by the display update and these transport checks.

---

## Routing rules

- Camera pose publisher display → `src/view_camera_pose_publisher_display.cpp`
- Plugin registration → `plugin_view_camera_pose.xml`
- Free cam pose consumer → `../genie_sim_render/src/render_node.cpp`
