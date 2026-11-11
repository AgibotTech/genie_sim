#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""genie_sim_engine_newton_node.py

Newton-direct physics engine entry point.  Kit-free by default.
ROS topics (clock, joint_states, body_tf, odom) are always published.

Visualizer is selected at launch via ``physics_engine_visualizer``:

  none   — headless, no window (default)
  newton — Newton ViewerGL: direct OpenGL render of state.body_q
  ovrtx  — in-process OVRtx photoreal render thread (InlineOvrtxVisualizer)
  isaacsim — owned Isaac Sim RTX worker, synchronous full-scene snapshots
  rerun  — placeholder; not implemented

For the Kit viewport with Newton physics, use
``physics_engine:=isaac_newton`` instead — that path runs Isaac Sim's
Newton wrapper inside Kit.

Scene assets are the same YAML / manifest as genie_sim_engine_isaacsim_node.py;
select this script in the launcher via the launcher YAML's
``launcher.physics.engines`` block.
"""

from __future__ import annotations

from contextlib import ExitStack
import os
import sys
import time

from genie_sim_engine_core.common.parent_lifetime import exit_with_parent

exit_with_parent()

# Early arg parsing — Kit-free, must run before any conditional bootstrap.
from genie_sim_engine_core.common.early_params import _early_param  # noqa: E402

# ---------------------------------------------------------------------------
# Resolve visualizer / solver BEFORE any omni import.
# ---------------------------------------------------------------------------

_VISUALIZER = _early_param(sys.argv[1:], "physics_engine_visualizer", "none").strip().lower()
_RENDER_MODE = _early_param(sys.argv[1:], "render_mode", "raster")
_SOLVER = _early_param(sys.argv[1:], "physics_solver", "fsvbd").strip().lower()

_VALID_VISUALIZERS = ("none", "newton", "rerun", "ovrtx", "isaacsim")
if _VISUALIZER not in _VALID_VISUALIZERS:
    print(
        f"[genie_sim_engine_newton] unknown physics_engine_visualizer={_VISUALIZER!r}; "
        f"valid: {_VALID_VISUALIZERS}. Falling back to 'none'.",
        flush=True,
    )
    _VISUALIZER = "none"

# ``headless:=true`` (the launcher default) wins over physics_engine_visualizer:
# no GUI, no Kit, no ViewerGL — even if the launcher yaml asks for one. The
# signal is the absence of ``--gui`` in argv (physics_backend.launch.py only
# appends ``--gui`` when ``headless != "true"``). This way the same launcher
# yaml works for both interactive bringup (`headless:=false rviz:=true`) and
# headless batch runs without two separate launcher files.
_HEADLESS = "--gui" not in sys.argv
if _HEADLESS and _VISUALIZER != "none":
    print(
        f"[genie_sim_engine_newton] headless=true → forcing physics_engine_visualizer=none "
        f"(launcher requested {_VISUALIZER!r}); pass headless:=false to enable a visualizer.",
        flush=True,
    )
    _VISUALIZER = "none"

# Pyglet's Linux window uses GLX. Select the matching PyOpenGL backend before
# warmup imports MuJoCo: an inherited MUJOCO_GL=egl otherwise locks PyOpenGL
# to EGL, so ViewerGL's ImGui renderer cannot see the window's GL context.
if _VISUALIZER == "newton" and sys.platform == "linux" and os.environ.get("DISPLAY"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

# ---------------------------------------------------------------------------
# Bootstrap — newton-standalone is Kit-free.  No SimulationApp, ever.
# Just warm up the Warp kernels so the first physics step doesn't pay
# JIT-compile latency.  Use ``physics_engine:=isaac_newton`` for the
# Kit-using path with Newton physics.
# ---------------------------------------------------------------------------

if _VISUALIZER == "rerun":
    # ``rerun`` is a placeholder visualizer name; the gRPC stream isn't
    # wired up, so we fall back to headless and warn the operator.
    print(
        "[genie_sim_engine_newton] physics_engine_visualizer=rerun not implemented; running headless.",
        flush=True,
    )
    _VISUALIZER = "none"

# ovrtx visualizer is implemented in engine.newton.ovrtx_visualizer.
# It is constructed in run() (after the engine builds and ``model.body_paths``
# is populated) so the module-level guard above can still downgrade
# ovrtx → none when ``headless:=true``.  See docs/ovrtx_sync.md.

# Newton GL viewer / ovrtx / headless — all Kit-free.  Warm Warp kernels.
from genie_sim_engine_newton.warmup import warmup_newton_kernels  # noqa: E402

warmup_newton_kernels(_SOLVER)
simulation_app = None

# ---------------------------------------------------------------------------
# Heavy imports (safe after bootstrap)
# ---------------------------------------------------------------------------

from genie_sim_engine_core.common.loop import _parse_args  # noqa: E402
from genie_sim_engine_newton.common.session import GenieSimEngineSession, SimpleLogger  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_newton_viewer(sim, render_hz: float, params, logger) -> object | None:
    """Instantiate ViewerGL after the engine is built.

    Returns the viewer, or None if no display is available or init fails.

    Reads optional ``viewer_camera_json`` from ``params`` -- a JSON-encoded
    ``{"pos":[x,y,z], "lookat":[x,y,z], "pitch":deg, "yaw":deg}`` originating
    in the scene YAML's ``viewer_camera`` block.  ``lookat`` (when present)
    takes precedence over pitch/yaw and also sets the orbit pivot, so RMB
    drag rotates around the chosen target.  Missing / empty / unparseable
    JSON keeps Newton's built-in default pose.

    The JSON-string indirection sidesteps ``launch_ros``'s parameter pipeline,
    which silently drops list-typed YAML values; the same trick is already
    used for ``init_joint_pos_json``.  See ``physics_backend.launch.py``.
    """
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
        logger.warn(
            "physics_engine_visualizer=newton requested but no $DISPLAY / $WAYLAND_DISPLAY detected. "
            "Running headless. Set DISPLAY=:0 or use a virtual framebuffer (Xvfb)."
        )
        return None

    def _xyz(seq):
        try:
            xyz = [float(v) for v in seq]
        except (TypeError, ValueError):
            return None
        return xyz if len(xyz) == 3 else None

    raw_json = str((params or {}).get("viewer_camera_json", "") or "").strip()
    cam_cfg: dict = {}
    if raw_json and raw_json != "{}":
        import json as _json

        try:
            parsed = _json.loads(raw_json)
            if isinstance(parsed, dict):
                cam_cfg = parsed
        except ValueError as exc:
            logger.warn(f"viewer_camera_json parse failed: {exc!r}; using default pose.")

    viewer = None
    try:
        from newton.viewer import ViewerGL
        from genie_sim_engine_newton.engine.newton.visualizers.gl_observation import GlObservationConfig

        rgb_config = GlObservationConfig.from_params(params or {})

        viewer = ViewerGL(
            width=rgb_config.width if rgb_config else 1280,
            height=rgb_config.height if rgb_config else 720,
            vsync=False,
            headless=False,
        )
        viewer.set_model(sim._model)

        cam_pos = _xyz(cam_cfg.get("pos", []))
        if cam_pos is not None:
            try:
                import warp as wp

                pos_vec = wp.vec3(cam_pos[0], cam_pos[1], cam_pos[2])
                lookat = _xyz(cam_cfg.get("lookat", []))

                if lookat is not None:
                    viewer.camera.pos = viewer.camera._as_vec3(pos_vec)
                    viewer.camera.look_at((lookat[0], lookat[1], lookat[2]))
                    logger.info(f"newton GL viewer cam pos={tuple(cam_pos)} lookat={tuple(lookat)}")
                else:
                    pitch = float(cam_cfg.get("pitch", 0.0))
                    yaw = float(cam_cfg.get("yaw", -180.0))
                    viewer.set_camera(pos=pos_vec, pitch=pitch, yaw=yaw)
                    logger.info(f"newton GL viewer cam pos={tuple(cam_pos)} pitch={pitch:.1f} yaw={yaw:.1f}")
            except Exception as exc:
                logger.warn(f"newton GL viewer set_camera failed: {exc!r}; using default pose.")
        elif cam_cfg:
            logger.warn(f"viewer_camera.pos missing or invalid in scene yaml ({cam_cfg!r}); using default pose.")

        logger.info(f"newton GL viewer ready (render_hz={render_hz:.0f})")
        return viewer
    except Exception as exc:
        if viewer is not None:
            viewer.close()
        logger.warn(f"newton GL viewer init failed: {exc}. Running headless.")
        return None


def _create_ovrtx_visualizer(session, params, logger):
    """Instantiate :class:`InlineOvrtxVisualizer` and wire it into the engine.

    Returns ``(viz, physics_event)`` or ``(None, None)`` on failure.
    The caller is responsible for calling ``viz.stop()`` at shutdown.

    Failure modes (all return ``None, None``):
      * ovrtx 0.4.x not installed
      * Renderer construction fails (USD load, RTX init, etc.)
      * No bodies in the Newton model (assemble step did not run)

    See docs/ovrtx_sync.md for the full design.
    """
    try:
        import warp as wp
        from genie_sim_engine_newton.engine.newton.visualizers.ovrtx import InlineOvrtxVisualizer
    except Exception as exc:
        logger.warn(f"physics_engine_visualizer=ovrtx import failed: {exc!r}. Running headless.")
        return None, None

    sim = session.sim
    body_paths = list(getattr(sim, "_body_paths", []) or [])
    if not body_paths:
        logger.warn(
            "physics_engine_visualizer=ovrtx requested but Newton model has no body paths. " "Running headless."
        )
        return None, None

    manifest_path = str(params.get("stage_manifest", ""))
    ovrtx_root = str(params.get("ovrtx_root", "")).strip() or None
    log_path = str(params.get("ovrtx_log_path", "/tmp/ovrtx_inline.log"))
    log_level = str(params.get("ovrtx_log_level", "info"))
    # FreeCam pose topic — defaults to the topic published by the RViz
    # view_camera_pose_publisher_display plugin (genie_sim_rviz_plugins).
    camera_pose_topic = str(params.get("ovrtx_camera_pose_topic", "/genie_sim_engine/viewer/camera_pose"))
    # Empty by default: optical zoom is additive and only launchers that opt
    # in should create a subscriber. Existing external/g2sim render paths keep
    # the camera-pose-only contract.
    camera_zoom_topic = str(params.get("ovrtx_camera_zoom_topic", ""))

    try:
        viz = InlineOvrtxVisualizer(
            scene_usda=sim._scene_usda,
            robot_usda=sim._robot_usda,
            render_layer_usda=sim._render_layer_usda,
            manifest_path=manifest_path,
            robot_prefix=sim._robot_prefix_str,
            device=str(wp.get_device()),
            ovrtx_root=ovrtx_root,
            render_fps=session.render_hz,
            realtime_factor=session.realtime_factor,
            log_path=log_path,
            log_level=log_level,
            camera_pose_topic=camera_pose_topic,
            camera_zoom_topic=camera_zoom_topic,
        )
    except Exception as exc:
        logger.warn(f"InlineOvrtxVisualizer init failed: {exc!r}. Running headless.")
        return None, None

    # Stable wp.array view of body paths for the visualizer's body-index map.
    # Use a closure as the body_q provider so we always read the *current*
    # state_0 (Newton swaps state_0 ↔ state_1 between substeps).
    def _body_q_provider():
        state = getattr(sim, "_state_0", None)
        return None if state is None else getattr(state, "body_q", None)

    # Same closure pattern for particle_q — the cloth animator on the
    # visualizer side reads this every render frame.
    def _particle_q_provider():
        state = getattr(sim, "_state_0", None)
        return None if state is None else getattr(state, "particle_q", None)

    # Cloth bookkeeping is set by engine.newton.cloth._inject_cloth on the
    # engine instance.  v1 supports a single cloth slot.
    cloth_prim = getattr(sim, "_cloth_usd_prim_path", "") or ""
    cloth_start = int(getattr(sim, "_cloth_particle_start", 0) or 0)
    cloth_end = int(getattr(sim, "_cloth_particle_end", 0) or 0)
    cloth_info = None
    if cloth_prim and cloth_end > cloth_start:
        cloth_info = {"prim_path": cloth_prim, "start": cloth_start, "end": cloth_end}

    # Scene plugins may expose authored Mesh paths plus fixed FEM-particle
    # reconstruction maps. The inline visualizer consumes these directly on
    # its CUDA stream; the physics thread only publishes its usual event.
    deformable_bindings = []
    try:
        deformable_bindings = sim._call_plugin("get_deformable_render_bindings") or []
    except AttributeError:
        pass

    try:
        # Build a tiny shim that exposes ``body_paths`` as a method, so the
        # visualizer can fall back to engine-side paths if model.body_paths
        # is not populated.
        class _Shim:
            def __init__(self, paths):
                self._paths = paths

            def body_paths(self):
                return self._paths

        # Newton's Model.body_paths may or may not exist depending on
        # version; pass the engine's authoritative list via the shim.
        # The visualizer uses model.body_paths first, then falls back to
        # body_q_provider.body_paths().
        viz.bind_newton_bodies(
            _Shim(body_paths),
            _body_q_provider,
            cloth_info=cloth_info,
            particle_q_provider=(_particle_q_provider if cloth_info or deformable_bindings else None),
            deformable_bindings=deformable_bindings,
        )
    except Exception as exc:
        logger.warn(f"InlineOvrtxVisualizer.bind_newton_bodies failed: {exc!r}. Running headless.")
        return None, None

    # Per-step CUDA event recorded by NewtonHeadlessEngine.tick_extras.
    physics_event = wp.Event(device=wp.get_device())
    viz.attach_physics_event(physics_event)
    sim.attach_physics_event(physics_event)

    try:
        viz.start()
    except Exception as exc:
        logger.warn(f"InlineOvrtxVisualizer.start failed: {exc!r}. Running headless.")
        sim.attach_physics_event(None)
        return None, None

    logger.info(
        f"inline ovrtx visualizer ready ({len(body_paths)} bodies, "
        f"{len(deformable_bindings)} deformable meshes, render_hz={session.render_hz:.0f})"
    )
    return viz, physics_event


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------


def run(params: dict) -> int:
    logger = SimpleLogger("genie_sim_engine_newton")
    from genie_sim_engine_newton.engine.newton.visualizers.gl_observation import (
        GlObservationConfig,
        GlObservationPublisher,
        body_indices,
        capture_views,
        fixed_render_size,
        mounted_camera_pose,
    )
    from genie_sim_engine_newton.engine.newton.visualizers.gl_video import GlVideoConfig, GlVideoRecorder

    rgb_config = GlObservationConfig.from_params(params)
    video_config = GlVideoConfig.from_params(params)
    wok_config = None
    if params.get("wok_evaluation_json") not in (None, ""):
        from genie_sim_engine_newton.engine.newton.wok_evaluation_ros import (
            WokEvaluationProducer,
            WokEvaluationRosConfig,
        )

        wok_config = WokEvaluationRosConfig.from_params(params)
    visualizer = params.get("physics_engine_visualizer", _VISUALIZER).strip().lower()
    # Re-apply the headless override here so a ros-param value can't
    # bypass the module-level guard (the ros-param is read by
    # EngineSession AFTER module init, and would otherwise mask
    # the ``_VISUALIZER = "none"`` set above).
    if _HEADLESS and visualizer != "none":
        logger.info(
            f"headless=true → ignoring ros-param physics_engine_visualizer={visualizer!r}; "
            f"running with no visualizer."
        )
        visualizer = "none"

    if rgb_config is not None and visualizer not in {"newton", "isaacsim"}:
        raise ValueError("newton_rgb_views_json requires a newton or isaacsim visualizer and headless=false")
    if video_config is not None and visualizer not in {"newton", "isaacsim"}:
        raise ValueError("newton_video_path requires a newton or isaacsim visualizer and headless=false")

    session = GenieSimEngineSession(
        node_name="genie_sim_engine_newton",
        params=params,
        simulation_app=simulation_app,
        physics_engine="newton",
        logger=logger,
    )

    logger.info(
        f"Newton engine ready — physics_hz={session.physics_hz:.0f}  "
        f"render_hz={session.render_hz:.0f}  rtf={session.realtime_factor}  "
        f"visualizer={visualizer!r}  scene={session.scene_usda}"
    )

    session.startup(headless=(visualizer != "newton"))
    session.bind_newton_body_topology()

    # Start the ROS reset listener. Must run AFTER session.startup() (which
    # builds the engine and composes all mixins) so the RosResetCoordinator
    # is instantiated and _snapshot_reset_state() has already captured the
    # settled init state.
    session.sim._lifecycle.start_listener()

    # Tell the engine the wall-clock RTF so its stats display shows
    # rtf-scaled target Hz instead of the raw simulation rate (which would
    # always look wrong by a factor of `rtf` on the actual-vs-target line).
    session.sim._rtf = session.realtime_factor

    # Visualizer init (after engine build so _model is available)
    viewer = None
    ovrtx_viz = None
    gl_resources = ExitStack()
    if visualizer == "newton":
        viewer = _create_newton_viewer(session.sim, session.render_hz, params, logger)
        if viewer is not None:
            gl_resources.callback(viewer.close)
    elif visualizer == "ovrtx":
        ovrtx_viz, _physics_event = _create_ovrtx_visualizer(session, params, logger)

    next_render = [time.monotonic()]  # mutable cell for closure
    dt = session.dt

    def _render_scene(sim_time):
        viewer.begin_frame(sim_time)
        viewer.log_state(session.sim._state_0)
        # The chef's smooth noodle and ham meshes are real ViewerGL geometry.
        # Invoke the same hook for every mounted camera, without stepping.
        session.sim._call_plugin("on_render", viewer, session.sim._state_0, sim_time, session.sim._plugin_ctx())
        viewer.end_frame()

    rgb_publisher = None
    video_recorder = None
    isaac_renderer = None
    isaac_scene = None
    wok_progress = None
    try:
        if wok_config is not None:
            lifecycle_node = session.sim._lifecycle._node
            if lifecycle_node is None:
                raise RuntimeError("configured wok evaluation requires the engine lifecycle ROS node")
            wok_progress = WokEvaluationProducer(lifecycle_node, session.sim, wok_config, dt=session.dt)
            gl_resources.callback(wok_progress.close)
            progress_hook = wok_progress.after_step
            session.post_step_hooks.append(progress_hook)
            gl_resources.callback(session.post_step_hooks.remove, progress_hook)
            wok_progress.refresh()
            logger.info(f"Wok evaluation progress: {wok_config.topic} (every completed physics tick)")
        if visualizer == "isaacsim":
            import json
            from pathlib import Path
            from ament_index_python.packages import get_package_prefix
            from genie_sim_engine_newton.engine.newton.visualizers.isaacsim import IsaacSimRenderer
            from genie_sim_engine_newton.engine.newton.visualizers.scene_snapshot import SceneSnapshot
            from genie_sim_engine_core.schema.manifest import Manifest

            if rgb_config is None:
                raise ValueError("Isaac Sim snapshot rendering requires explicit newton_rgb_views_json")
            worker = Path(get_package_prefix("genie_sim_render")) / "lib/genie_sim_render/isaacsim_snapshot_render.py"
            log_path = params.get("isaac_render_log_path", "")
            if not log_path:
                raise ValueError("Isaac Sim rendering requires an unused isaac_render_log_path")
            manifest = Manifest.read(Path(params["stage_manifest"]))
            visual = manifest.robot_visual_usda
            scene_options = {}
            authored_visuals = ((getattr(session.sim, "_scene_cfg", None) or {}).get("newton") or {}).get(
                "authored_visuals"
            )
            if authored_visuals is not None:
                descriptor, prefixes = SceneSnapshot.authored_visuals_from_config(
                    authored_visuals, assets_root=manifest.base_path
                )
                scene_options = {"authored_scene": descriptor, "hide_shape_prefixes": prefixes}
            isaac_renderer = IsaacSimRenderer(
                worker,
                Path(log_path),
                timeout_s=float(params.get("isaac_render_timeout_s", 120)),
                startup_timeout_s=float(params.get("isaac_render_startup_timeout_s", 600)),
            )
            gl_resources.callback(isaac_renderer.close)
            isaac_scene = SceneSnapshot(
                session.sim,
                required_meshes=json.loads(params.get("isaac_render_required_meshes_json", "[]")),
                authored_robot={"asset_path": str(visual)} if visual is not None else None,
                robot_bodies=manifest.robot.get("body_paths", {}),
                **scene_options,
            )
            gl_resources.callback(isaac_scene.close)
            observer = json.loads(params.get("viewer_camera_json", "{}"))
            observer_camera = {
                "name": "observer",
                "eye": observer.get("pos", [1.6, -1.6, 1.2]),
                "target": observer.get("lookat", [0, 0, 0.8]),
                "up": [0, 0, 1],
                "fov_degrees": 45.0,
            }
            indices = body_indices(session.sim._body_paths, rgb_config.views)

            def _isaac_capture(sim_time, *, sensors):
                cameras = [observer_camera]
                if sensors:
                    poses = session.sim._state_0.body_q.numpy()
                    cameras = []
                    for view, index in zip(rgb_config.views, indices, strict=True):
                        eye, target, up = mounted_camera_pose(poses[index], view)
                        cameras.append(
                            {
                                "name": view.name,
                                "eye": eye.tolist(),
                                "target": target.tolist(),
                                "up": up.tolist(),
                                "fov_degrees": view.fov_degrees,
                            }
                        )
                scene = isaac_scene.capture(sim_time)
                return isaac_renderer.render(scene, cameras, rgb_config.width, rgb_config.height)

            # Warm every real sensor before ordinary state publication makes
            # the engine attachable. No physics tick is integrated here.
            _isaac_capture(session.sim._sim_time_acc, sensors=True)
            logger.info("Isaac Sim RTX snapshot renderer ready; Newton physics remains authoritative")
        if video_config is not None:
            if viewer is None and isaac_renderer is None:
                raise RuntimeError("configured video recording requires an available Newton GL viewer")
            video_recorder = GlVideoRecorder(video_config, physics_hz=session.physics_hz)
            gl_resources.callback(video_recorder.close)
            if isaac_renderer is not None:
                video_config.path.parent.mkdir(parents=True, exist_ok=True)
                video_frames = video_config.path.with_suffix(".frames.jsonl").open("x", buffering=1)
                gl_resources.callback(video_frames.close)
            logger.info(f"Newton observer video: {video_config.path} ({video_config.fps:g} fps, simulation clock)")
        if rgb_config is not None:
            if viewer is None and isaac_renderer is None:
                raise RuntimeError("configured RGB capture requires an available Newton GL viewer")
            indices = body_indices(session.sim._body_paths, rgb_config.views)
            lifecycle_node = session.sim._lifecycle._node
            if lifecycle_node is None:
                raise RuntimeError("configured RGB capture requires the engine lifecycle ROS node")
            rgb_publisher = GlObservationPublisher(lifecycle_node, rgb_config)
            gl_resources.callback(rgb_publisher.close)

            def _capture_paused_observation():
                from genie_sim_engine_core.common.loop import _publish_tick

                sim_time = session.sim._sim_time_acc
                # pause_at_tick_boundary has synchronously filled both host
                # mirror slots. Publish that final state before the RGB batch.
                _publish_tick(session.sim, session.publish_plan, sim_time, session.core)
                if isaac_renderer is not None:
                    captured = _isaac_capture(sim_time, sensors=True)
                    images = tuple(captured[view.name] for view in rgb_config.views)
                else:
                    poses = session.sim._state_0.body_q.numpy()
                    images = capture_views(viewer, rgb_config, poses, indices, lambda: _render_scene(sim_time))
                rgb_publisher.publish(images)
                # Publish after the common image stamp is assigned/sent. A
                # consumer can join paused images and this task snapshot by
                # published_ns without adding scoring time or physics ticks.
                if wok_progress is not None:
                    wok_progress.refresh()

            session.sim._pause_observation_callback = _capture_paused_observation
        elif wok_progress is not None:
            session.sim._pause_observation_callback = wok_progress.refresh
    except BaseException:
        try:
            gl_resources.close()
        finally:
            try:
                session.sim.shutdown()
            finally:
                session.core.shutdown()
        raise

    render_failure = [None]

    def _render_hook(now, next_tick, sim_time):
        # Newton-standalone is Kit-free: the only in-loop viewport path
        # is the Newton GL ViewerGL.
        #
        # The ovrtx visualizer runs entirely on its own thread and is
        # gated on a CUDA event recorded by tick_extras; it does NOT
        # participate in this hook (no in-loop step needed).
        # Video samples the simulation clock, independently of network pauses
        # and wall pacing. Sensor capture calls _render_scene directly and
        # never enters this recording path, preserving the observer viewpoint.
        video_due = video_recorder is not None and video_recorder.due(sim_time)
        if isaac_renderer is not None:
            if not video_due:
                return 0.0, 0.0, False
            t_r0 = time.monotonic()
            try:
                captured = _isaac_capture(sim_time, sensors=False)
                if video_recorder.record_rgb(captured["observer"], sim_time):
                    video_frames.write(
                        json.dumps(
                            {
                                "frame": video_recorder.frame_count - 1,
                                "sim_time": sim_time,
                                "reset_epoch": int(session.sim._reset_epoch),
                            }
                        )
                        + "\n"
                    )
            except Exception as error:
                # Core treats ordinary render errors as recoverable step
                # failures and retains lockstep credit. This capture is a
                # terminal barrier failure: stop BEFORE another physics tick.
                render_failure[0] = error
                raise
            ms = (time.monotonic() - t_r0) * 1000.0
            return ms, ms, True
        if viewer is not None and (now >= next_render[0] or video_due):
            t_r0 = time.monotonic()
            if video_due:
                # Reuse the sensor FBO boundary for stable real video frames;
                # a WM resize changes only the restored interactive viewport.
                video_size = (rgb_config.width, rgb_config.height) if rgb_config else (1280, 720)
                try:
                    with fixed_render_size(viewer, *video_size):
                        _render_scene(sim_time)
                        video_recorder.record(viewer, sim_time)
                finally:
                    if viewer.is_running():
                        _render_scene(sim_time)
            else:
                _render_scene(sim_time)
            ms = (time.monotonic() - t_r0) * 1000.0
            render_period = 1.0 / (session.render_hz * session.realtime_factor)
            next_render[0] = now + render_period if video_recorder is not None else next_render[0] + render_period
            return ms, ms, True
        return 0.0, 0.0, False

    def _exit_check():
        if render_failure[0] is not None:
            return False
        if viewer is not None and not viewer.is_running():
            logger.info("newton GL window closed — shutting down.")
            return False
        return True

    # Spin (shutdown is handled inside spin's finally block)
    try:
        result = session.run(render_hook=_render_hook, exit_check=_exit_check)
        if render_failure[0] is not None:
            raise RuntimeError("Isaac Sim observer rendering failed; evaluation stopped") from render_failure[0]
        return result
    finally:
        session.sim._pause_observation_callback = None
        try:
            gl_resources.close()
            if video_recorder is not None:
                if video_recorder.frame_count:
                    logger.info(
                        f"Newton observer video finalized: {video_config.path}; {video_recorder.frame_count} frames"
                    )
                else:
                    logger.warn("Newton observer recording ended before its first GL frame; no video was created")
        finally:
            # Stop the OVRtx render thread cleanly. Joins within 5 s; safe
            # even if start() failed (stop() is idempotent).
            if ovrtx_viz is not None:
                try:
                    ovrtx_viz.stop()
                except Exception as exc:
                    logger.warn(f"ovrtx_viz.stop() raised: {exc!r}")


def main() -> int:
    params = _parse_args(sys.argv[1:], prefix="genie_sim_engine_newton")
    try:
        return run(params)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"[genie_sim_engine_newton] fatal: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # ``simulation_app`` is permanently None on Kit-free
        # newton-standalone; nothing to close.  Kept the try/except
        # shape for parity with the other entry-point scripts.
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
