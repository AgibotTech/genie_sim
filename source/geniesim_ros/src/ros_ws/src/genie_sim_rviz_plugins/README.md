# genie_sim_rviz_plugins

Custom RViz 2 display plugins for GenieSim — currently a free-camera
pose publisher that lets users drive the simulator's viewport camera
from RViz.

## Plugins

### `ViewCameraPosePublisherDisplay`

An RViz 2 display panel that publishes the current RViz camera pose
as `geometry_msgs/PoseStamped` on a configurable topic. The render
node (`genie_sim_render`) subscribes to it and drives the free-fly
camera in the USD stage accordingly.

Default topic: `/genie_sim_engine/viewer/camera_pose` (configurable in the RViz panel).

## When you'd touch this package

- Adding a new RViz display (registers via `pluginlib` and the
  `plugin_view_camera_pose.xml` manifest).
- Changing the free-camera pose publisher's topic schema or behaviour.
- Anything else that needs to surface in RViz with custom rendering.

## Mechanics

See [AGENTS.md](AGENTS.md) for the plugin registration pattern, the
free-cam pose contract, and where the consumer side lives
(`genie_sim_render`).
