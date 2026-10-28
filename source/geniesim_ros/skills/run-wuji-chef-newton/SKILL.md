---
name: run-wuji-chef-newton
description: >
  Start and verify the Newton engine for the G2 crsB + Wuji wok scene
  scene_flat_g2_sp_chef_wuji with launcher_newton_mjxpbd, using OpenGL by
  default or optional OVRtx rendering with ROS image viewing.
  Use when the user asks to launch, restart, view, or check the Newton chef
  scene, "start the wok scene", "switch the wok viewer to OVRtx", or "view wok images".
license: MPL-2.0
metadata:
  author: genie-sim
  version: "2.1"
prerequisites:
  - geniesim_ros:build-workspace
inputs:
  - name: container
    desc: Running Genie Sim container with the checkout mounted at /workspace
    required: false
    default: geniesim4
  - name: visualizer
    desc: OpenGL window (newton) or OVRtx ROS camera images (ovrtx)
    required: false
    default: newton
outputs:
  - desc: Newton chef scene running with an OpenGL window or OVRtx camera images, /clock, and /joint_states
---

## When to Use

Start or inspect the Newton engine for `scene_flat_g2_sp_chef_wuji`.
This recipe covers simulator startup, renderer selection, image viewing,
and ROS state feedback.

## Critical Patterns

1. Run the commands inside the selected container, from `/workspace`.
2. Pair this scene with `launcher_newton_mjxpbd`: MuJoCo-Warp handles rigid
   dynamics and XPBD handles the scene's FEM particles.
3. Default to OpenGL (`physics_engine_visualizer:=newton`) unless the user
   chooses OVRtx (`physics_engine_visualizer:=ovrtx`). Both use Newton physics
   and require `headless:=false` to enable the selected renderer.
4. OpenGL opens the foreground Newton `ViewerGL` window. OVRtx publishes
   camera images over ROS; view them in RViz or `rqt_image_view`.
   `rviz:=false` disables only RViz, leaving the selected renderer enabled.
5. A working display is required for the OpenGL window and image-viewing
   applications. Keep physics and scene settings in their existing launcher
   and scene YAMLs.

## Workflow

The container must already be running with the checkout and scene assets
available. If `devel/setup.bash` is missing or the ROS source has changed,
complete [build-workspace](../build-workspace/SKILL.md) first:

```bash
cd /workspace
geniesim ros build dev
```

Choose one rendering mode below. To switch modes, stop the current launch
with Ctrl-C before launching again. Use the same container and ROS domain
for the simulator and image-viewing terminals.

## Commands

### OpenGL — default

Inside the selected container, prepare the environment and launch:

```bash
cd /workspace
source /opt/ros/jazzy/setup.bash
source devel/setup.bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_g2_sp_chef_wuji \
  launcher_config:=launcher_newton_mjxpbd \
  physics_engine_visualizer:=newton \
  headless:=false rviz:=false
```

The Newton OpenGL window shows the live scene directly; RViz is optional.

### OVRtx — optional

In a terminal with the same environment sourced:

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_g2_sp_chef_wuji \
  launcher_config:=launcher_newton_mjxpbd \
  physics_engine_visualizer:=ovrtx \
  headless:=false rviz:=true
```

OVRtx images are available through either viewer below.

**RViz:**

1. In the left **Displays** panel, enable **Camera**; it is disabled in the
   default layout.
2. Its preset image topic is
   `/genie_sim/free_camera_rgb/image_raw/compressed`. The raw image topic is
   `/genie_sim/free_camera_rgb/image_raw`.
3. Keep **ViewCameraPosePublisher** and its **Publish TF** option enabled.
   Move the RViz 3D view to adjust the OVRtx FreeCam viewpoint.

**Image viewer:** open another terminal in the same container and ROS domain:

```bash
cd /workspace
source /opt/ros/jazzy/setup.bash
source devel/setup.bash
ros2 run rqt_image_view rqt_image_view /genie_sim/free_camera_rgb/image_raw
```

The image viewer displays the camera output. Use RViz to change the FreeCam
viewpoint. The camera-info topic is
`/genie_sim/free_camera_rgb/camera_info`.

## Notes

Check that the scene or camera image appears and keeps updating. In another
container terminal with the same ROS environment and domain sourced:

```bash
ros2 topic echo /clock --once
ros2 topic echo /joint_states --once
```

These topics provide simulation time and joint-state feedback. Stop the
scene with Ctrl-C in its launch terminal.

## Resources

- [Scene YAML](../../src/ros_ws/src/genie_sim_bringup/config/scene_flat_g2_sp_chef_wuji.yaml)
- [Newton MJXPBD launcher](../../src/ros_ws/src/genie_sim_bringup/config/launcher_newton_mjxpbd.yaml)
- [RViz layout](../../src/ros_ws/src/genie_sim_bringup/rviz/view_robot.rviz)
- [OVRtx visualizer](../../src/ros_ws/src/genie_sim_engine_newton/docs/ovrtx_sync.md)
- [Build workspace](../build-workspace/SKILL.md)
