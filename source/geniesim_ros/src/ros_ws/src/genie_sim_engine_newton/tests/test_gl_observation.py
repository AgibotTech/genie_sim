# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CPU tests for real-view capture ordering and paused readback; no ROS/GPU."""

from importlib.util import module_from_spec, spec_from_file_location
from importlib import import_module
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = spec_from_file_location(name, PACKAGE / "genie_sim_engine_newton/engine/newton" / relative)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gl = _load("tested_gl_observation", "visualizers/gl_observation.py")
sync = _load("tested_state_sync", "state_sync.py")


def _rows():
    return [
        dict(
            name=name,
            topic=f"/test/{name}/image_raw",
            body=body,
            eye=[0, 0, 0],
            look_at=[1, 0, 0],
            up=[0, 0, 1],
            fov_degrees=65,
        )
        for name, body in (
            ("head", "head_link3"),
            ("hand_left", "hand_l_base_link"),
            ("hand_right", "hand_r_base_link"),
        )
    ]


def _config(rows=None):
    return gl.GlObservationConfig.from_params(
        dict(newton_rgb_width=4, newton_rgb_height=2, newton_rgb_views_json=json.dumps(rows or _rows()))
    )


def test_launcher_preserves_chef_physics_and_has_three_explicit_views():
    bringup = PACKAGE.parent / "genie_sim_bringup/config"
    reference = yaml.safe_load((bringup / "launcher_newton_mjxpbd.yaml").read_text())["genie_sim_engine"][
        "ros__parameters"
    ]
    evaluation = yaml.safe_load((bringup / "launcher_newton_wuji_eval.yaml").read_text())["genie_sim_engine"][
        "ros__parameters"
    ]
    for key in (
        "physics_params_file",
        "physics_solver",
        "physics_hz",
        "physics_solver_substep",
        "physics_solver_iterations",
        "physics_solver_mass_matrix_interval",
        "realtime_factor",
    ):
        assert evaluation[key] == reference[key]
    config = gl.GlObservationConfig.from_params(evaluation)
    assert [view.name for view in config.views] == ["head", "hand_left", "hand_right"]
    assert len({view.body for view in config.views}) == 3
    assert [view.topic for view in config.views] == [
        f"/wuji_eval/{name}/image_raw" for name in ("head", "hand_left", "hand_right")
    ]


def test_actual_ros_argument_parser_preserves_rgb_launcher_and_scalar_overrides():
    core_path = PACKAGE.parent / "genie_sim_engine_core/genie_sim_engine_core"
    spec = spec_from_file_location(
        "gl_test_engine_core", core_path / "__init__.py", submodule_search_locations=[str(core_path)]
    )
    package = module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    loop = import_module("gl_test_engine_core.common.loop")
    launcher = PACKAGE.parent / "genie_sim_bringup/config/launcher_newton_wuji_eval.yaml"
    args = loop._parse_args(["--ros-args", "--params-file", str(launcher), "-p", "newton_rgb_width:=320"])
    assert args["newton_rgb_width"] == "320"
    assert args["newton_rgb_height"] == "480"
    config = gl.GlObservationConfig.from_params(args)
    assert (config.width, config.height) == (320, 480)
    assert len(config.views) == 3


@pytest.mark.parametrize("value", [True, "true", "4.0", 4.0, "-1", "0", "1e3", " 4 ", "004"])
def test_rgb_dimensions_reject_noncanonical_or_nonpositive_integers(value):
    with pytest.raises(ValueError, match="positive integer"):
        gl.GlObservationConfig.from_params(
            dict(newton_rgb_width=value, newton_rgb_height="2", newton_rgb_views_json=json.dumps(_rows()))
        )


@pytest.mark.parametrize(
    "change",
    [
        {"eye": [0, 0, 0.0], "look_at": [0, 0, 0.01]},
        {"eye": [float("nan"), 0, 0]},
        {"fov_degrees": 180},
        {"topic": "relative/image_raw"},
    ],
)
def test_bad_view_config_fails_before_rendering(change):
    rows = _rows()
    rows[0].update(change)
    with pytest.raises(ValueError):
        _config(rows)


def test_camera_mount_applies_parent_rotation_and_translation():
    view = _config().views[0]
    angle = np.sqrt(0.5)
    eye, target, up = gl.mounted_camera_pose(np.array([1, 2, 3, 0, 0, angle, angle]), view)
    np.testing.assert_allclose(eye, [1, 2, 3])
    np.testing.assert_allclose(target, [1, 3, 3])
    np.testing.assert_allclose(up, [0, 0, 1])


def test_wrist_mount_up_rotates_with_parent_instead_of_staying_world_upright():
    angle = np.sqrt(0.5)
    _, _, up = gl.mounted_camera_pose(np.array([0, 0, 0, angle, 0, 0, angle]), _config().views[0])
    np.testing.assert_allclose(up, [0, -1, 0], atol=1e-8)


def test_camera_parent_resolution_fails_for_missing_or_ambiguous_links():
    config = _config()
    with pytest.raises(ValueError, match="0 bodies"):
        gl.body_indices([], config.views)
    with pytest.raises(ValueError, match="2 bodies"):
        gl.body_indices(["/a/head_link3", "/b/head_link3"], config.views)


class Camera:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def look_at(self, target):
        self.target = target

    def update_screen_size(self, width, height):
        self.width, self.height = width, height

    @staticmethod
    def _as_vec3(vector):
        return tuple(vector)


class Renderer:
    def __init__(self, viewer, size=(4, 2)):
        self.native_size = size
        self.window = SimpleNamespace(get_size=lambda: self.native_size, get_framebuffer_size=lambda: self.native_size)
        self._screen_width, self._screen_height = size
        self._resize_callbacks = [viewer.on_resize]
        self.allocations = []
        self._setup_frame_buffer()

    def resize(self, *_args):
        self._screen_width, self._screen_height = self.window.get_framebuffer_size()
        self._setup_frame_buffer()

    def _setup_frame_buffer(self):
        self.framebuffer_shape = (self._screen_height, self._screen_width, 3)
        self.allocations.append(self.framebuffer_shape)

    def dispatch_resize(self, width, height):
        self.native_size = (width, height)
        self.resize(width, height)
        for callback in self._resize_callbacks:
            callback(width, height)


class Viewer:
    def __init__(self):
        self.camera = Camera(pos=(100, 100, 100), width=4, height=2)
        self.show_ui = True
        self.gui = object()
        self.events = []
        self.count = 0
        self.invalidations = 0
        self.renderer = Renderer(self)

    def on_resize(self, *_args):
        self.camera.update_screen_size(*self.renderer.window.get_framebuffer_size())
        self._invalidate_pbo()

    def _invalidate_pbo(self):
        self.invalidations += 1

    def is_running(self):
        return True

    def get_frame(self, *, render_ui):
        assert not render_ui
        self.events.append(("read", self.camera.pos, self.show_ui))
        self.count += 1
        return SimpleNamespace(numpy=lambda: np.full(self.renderer.framebuffer_shape, self.count, dtype=np.uint8))


def test_each_view_is_rendered_then_read_and_ui_is_restored():
    viewer = Viewer()
    original = viewer.camera
    original_gui = viewer.gui
    poses = np.array([[x, 0, 0, 0, 0, 0, 1] for x in (1, 2, 3)])

    def render():
        assert (viewer.gui is original_gui) if viewer.camera is original else (viewer.gui is None)
        viewer.events.append(("render", viewer.camera.pos, viewer.show_ui))

    images = gl.capture_views(viewer, _config(), poses, (0, 1, 2), render)
    assert [event[0] for event in viewer.events] == ["render", "read"] * 3 + ["render"]
    assert [tuple(event[1]) for event in viewer.events if event[0] == "read"] == [(1, 0, 0), (2, 0, 0), (3, 0, 0)]
    assert [int(image[0, 0, 0]) for image in images] == [1, 2, 3]
    assert viewer.camera is original
    assert viewer.show_ui
    assert viewer.gui is original_gui


def test_partial_capture_raises_and_restores_ui_camera():
    viewer = Viewer()
    original = viewer.camera

    def fail():
        if viewer.camera is not original:
            raise RuntimeError("GL readback unavailable")

    with pytest.raises(RuntimeError, match="GL readback"):
        gl.capture_views(viewer, _config(), np.tile([0, 0, 0, 0, 0, 0, 1], (3, 1)), (0, 1, 2), fail)
    assert viewer.camera is original
    assert viewer.show_ui


def test_sensor_frames_have_real_fixed_fbo_size_despite_async_native_window_resize():
    viewer = Viewer()
    original_camera = viewer.camera
    original_callbacks = viewer.renderer._resize_callbacks
    viewer.renderer.dispatch_resize(13, 7)
    resized = False
    render_shapes = []

    def render():
        nonlocal resized
        if viewer.camera is not original_camera:
            if not resized:
                # Mirrors the real ViewerGL.end_frame -> renderer.update ->
                # X/WM resize event that changed the live FBO to 1619x1058.
                viewer.renderer.dispatch_resize(1619, 1058)
                resized = True
            render_shapes.append(viewer.renderer.framebuffer_shape)

    images = gl.capture_views(viewer, _config(), np.tile([0, 0, 0, 0, 0, 0, 1], (3, 1)), (0, 1, 2), render)
    assert render_shapes == [(2, 4, 3)] * 3
    assert [image.shape for image in images] == [(2, 4, 3)] * 3
    assert viewer.renderer.native_size == (1619, 1058)
    assert viewer.renderer.framebuffer_shape == (1058, 1619, 3)
    assert viewer.renderer._resize_callbacks is original_callbacks
    assert viewer.camera is original_camera
    assert (viewer.camera.width, viewer.camera.height) == (1619, 1058)
    assert viewer.invalidations >= 3


def test_failed_sensor_capture_restores_native_framebuffer_and_resize_callbacks():
    viewer = Viewer()
    original = viewer.camera
    original_callbacks = viewer.renderer._resize_callbacks
    viewer.renderer.dispatch_resize(13, 7)

    def fail():
        if viewer.camera is not original:
            raise RuntimeError("GL render failed")

    with pytest.raises(RuntimeError, match="GL render failed"):
        gl.capture_views(viewer, _config(), np.tile([0, 0, 0, 0, 0, 0, 1], (3, 1)), (0, 1, 2), fail)
    assert viewer.renderer.framebuffer_shape == (7, 13, 3)
    assert viewer.renderer._resize_callbacks is original_callbacks
    assert viewer.camera is original


def _fake_ros(monkeypatch):
    class Image:
        def __init__(self):
            self.header = SimpleNamespace()

    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", SimpleNamespace(Image=Image))
    monkeypatch.setitem(
        sys.modules,
        "rclpy.qos",
        SimpleNamespace(
            DurabilityPolicy=SimpleNamespace(TRANSIENT_LOCAL="latched"),
            HistoryPolicy=SimpleNamespace(KEEP_LAST="last"),
            ReliabilityPolicy=SimpleNamespace(RELIABLE="reliable"),
            QoSProfile=lambda **kw: SimpleNamespace(**kw),
        ),
    )
    return Image


def test_complete_rgb_batch_has_one_stamp_and_latched_owned_publishers(monkeypatch):
    _fake_ros(monkeypatch)
    messages, destroyed, qos_values = [], [], []
    stamp = object()

    def create(_type, topic, qos):
        qos_values.append(qos)
        return SimpleNamespace(publish=lambda message: messages.append((topic, message)))

    node = SimpleNamespace(
        create_publisher=create,
        destroy_publisher=destroyed.append,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: stamp)),
    )
    publisher = gl.GlObservationPublisher(node, _config())
    publisher.publish([np.full((2, 4, 3), i, dtype=np.uint8) for i in range(3)])
    assert len(messages) == 3
    assert all(message.header.stamp is stamp for _, message in messages)
    assert all(message.encoding == "rgb8" and message.step == 12 for _, message in messages)
    assert all(qos.depth == 1 and qos.durability == "latched" for qos in qos_values)
    assert len({message.data for _, message in messages}) == 3
    publisher.close()
    publisher.close()
    assert len(destroyed) == 3


def test_publisher_construction_failure_releases_already_created_publisher(monkeypatch):
    _fake_ros(monkeypatch)
    resource, destroyed = object(), []
    count = 0

    def create(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("ROS creation failed")
        return resource

    with pytest.raises(RuntimeError, match="ROS creation"):
        gl.GlObservationPublisher(
            SimpleNamespace(create_publisher=create, destroy_publisher=destroyed.append), _config()
        )
    assert destroyed == [resource]


def _engine(monkeypatch, events):
    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(
            synchronize_device=lambda: events.append("sync"),
            get_stream=lambda device: device,
        ),
    )
    engine = SimpleNamespace(
        _paused=False,
        _update_record_snapshot_mirror=lambda: None,
        _logger=SimpleNamespace(info=lambda _: events.append("paused")),
        _state_0=SimpleNamespace(
            **{name: SimpleNamespace(device=name) for name in ("joint_q", "joint_qd", "body_q", "particle_q")}
        ),
    )
    for name in ("_jq_mirror", "_jqd_mirror", "_bq_mirror", "_pq_mirror"):
        setattr(engine, name, SimpleNamespace(refresh=lambda source, _stream: events.append(source.device)))
    return engine


def test_pause_joins_gpu_flushes_both_mirror_slots_and_captures_before_ack(monkeypatch):
    events = []
    engine = _engine(monkeypatch, events)

    def capture():
        assert not engine._paused
        events.append("capture")

    engine._pause_observation_callback = capture
    sync.pause_at_tick_boundary(engine)
    assert events == ["sync"] + [name for name in ("joint_q", "joint_qd", "body_q", "particle_q") for _ in range(2)] + [
        "sync",
        "capture",
        "paused",
    ]
    assert engine._paused


def test_failed_pause_capture_does_not_acknowledge_pause(monkeypatch):
    engine = _engine(monkeypatch, [])
    engine._pause_observation_callback = lambda: (_ for _ in ()).throw(RuntimeError("capture failed"))
    with pytest.raises(RuntimeError, match="capture failed"):
        sync.pause_at_tick_boundary(engine)
    assert not engine._paused


def test_already_paused_request_refreshes_observation_without_step(monkeypatch):
    events = []
    engine = _engine(monkeypatch, events)
    engine._paused = True
    engine._pause_observation_callback = lambda: events.append("capture")
    sync.pause_at_tick_boundary(engine)
    assert events[-1] == "capture"
    assert "paused" not in events


def test_pause_respects_disabled_particle_recording_mirror(monkeypatch):
    events = []
    engine = _engine(monkeypatch, events)
    engine._update_record_snapshot_mirror = lambda: setattr(engine, "_pq_mirror", None)
    sync.pause_at_tick_boundary(engine)
    assert "particle_q" not in events
    assert engine._paused
