# genie_sim_rviz_plugins

Custom RViz 2 display plugins for GenieSim — currently a free-camera
state publisher that lets users drive the simulator's viewport camera
from RViz.

## Plugins

### `ViewCameraPosePublisherDisplay`

An RViz 2 display panel that publishes the current RViz camera pose
as `geometry_msgs/PoseStamped` on a configurable topic. The render
node (`genie_sim_render`) subscribes to it and drives the free-fly
camera in the USD stage accordingly.

Default pose topic: `/genie_sim_engine/viewer/camera_pose`.

The optional `OVRtx Zoom Topic` publishes `std_msgs/Float64` focal-length
multipliers from the `OVRtx Zoom Factor` property. Its default is empty, so
existing external renderers remain pose-only. Set it to
`/genie_sim_engine/viewer/camera_zoom` when using a renderer that consumes
optical focal-length multipliers; supported factors are 0.25–8.0.

With `OVRtx Auto Zoom from Orbit` enabled (the shipped RViz default), mouse
wheel changes to the active Orbit controller's `Distance` property are
converted into the same optical-zoom factor. Disable it to edit the factor
manually. Other RViz view-controller types remain manual.

## When you'd touch this package

- Adding a new RViz display (registers via `pluginlib` and the
  `plugin_view_camera_pose.xml` manifest).
- Changing the free-camera pose publisher's topic schema or behaviour.
- Anything else that needs to surface in RViz with custom rendering.

## Mechanics

See [AGENTS.md](AGENTS.md) for the plugin registration pattern, the
free-cam pose contract, and where the consumer side lives
(`genie_sim_render`).

With testing enabled, `test_camera_zoom` verifies Orbit scaling and bounds,
manual control, topic changes and delivery of the latest zoom value to new
ROS subscribers. These checks use Qt offscreen without a renderer.
