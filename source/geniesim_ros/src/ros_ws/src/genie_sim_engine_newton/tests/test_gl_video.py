# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CPU checks for simulation-clock video sampling and owned encoder cleanup."""

from importlib.util import module_from_spec, spec_from_file_location
import ast
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "tested_gl_video", PACKAGE / "genie_sim_engine_newton/engine/newton/visualizers/gl_video.py"
)
video = module_from_spec(SPEC)
sys.modules[SPEC.name] = video
SPEC.loader.exec_module(video)
GL_SPEC = spec_from_file_location(
    "tested_video_fixed_gl", PACKAGE / "genie_sim_engine_newton/engine/newton/visualizers/gl_observation.py"
)
fixed_gl = module_from_spec(GL_SPEC)
sys.modules[GL_SPEC.name] = fixed_gl
GL_SPEC.loader.exec_module(fixed_gl)
SCENE_SPEC = spec_from_file_location(
    "tested_video_scene_snapshot", PACKAGE / "genie_sim_engine_newton/engine/newton/visualizers/scene_snapshot.py"
)
scene_snapshot = module_from_spec(SCENE_SPEC)
SCENE_SPEC.loader.exec_module(scene_snapshot)


class Writer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def write(self, pixels):
        self.frames.append(pixels)
        return len(pixels)

    def close(self):
        self.closed = True


class Encoder:
    def __init__(self):
        self.stdin = Writer()
        self.waited = False

    def poll(self):
        return None

    def wait(self, timeout):
        self.waited = True
        return 0


class Viewer:
    def __init__(self):
        self.reads = 0
        self.shape = (2, 4, 3)

    def is_running(self):
        return True

    def get_frame(self, *, render_ui):
        assert render_ui is False
        self.reads += 1
        return SimpleNamespace(numpy=lambda: np.full(self.shape, self.reads, dtype=np.uint8))


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    encoder = Encoder()
    commands = []
    monkeypatch.setattr(video.shutil, "which", lambda _: "/test/ffmpeg")

    def spawn(command, **kwargs):
        commands.append((command, kwargs))
        return encoder

    monkeypatch.setattr(video.subprocess, "Popen", spawn)
    instance = video.GlVideoRecorder(video.GlVideoConfig(tmp_path / "observer.mp4", 30), physics_hz=60)
    yield instance, encoder, commands
    instance.close()


def test_recording_is_disabled_by_default_and_accepts_ros_scalar_strings(tmp_path):
    assert video.GlVideoConfig.from_params({}) is None
    config = video.GlVideoConfig.from_params(
        dict(newton_video_path=str(tmp_path / "view.mp4"), newton_video_fps="30.0")
    )
    assert config.fps == 30
    assert config.path == tmp_path / "view.mp4"


@pytest.mark.parametrize("fps", [True, "bad", "nan", "inf", "0", -1])
def test_invalid_rate_fails_explicitly(tmp_path, fps):
    with pytest.raises(ValueError, match="finite and positive"):
        video.GlVideoConfig.from_params(dict(newton_video_path=str(tmp_path / "view.mp4"), newton_video_fps=fps))


def test_unsupported_rate_cannot_synthesize_repeated_frames(tmp_path):
    with pytest.raises(ValueError, match="must not exceed physics_hz"):
        video.GlVideoRecorder(video.GlVideoConfig(tmp_path / "view.mp4", 61), physics_hz=60)


def test_60_hz_simulation_records_exactly_one_real_frame_per_two_ticks(recorder):
    instance, encoder, commands = recorder
    viewer = Viewer()
    recorded = [tick for tick in range(1, 121) if instance.record(viewer, tick / 60)]
    assert recorded == list(range(1, 121, 2))
    assert instance.frame_count == viewer.reads == len(encoder.stdin.frames) == 60
    assert len(commands) == 1
    command, kwargs = commands[0]
    assert command[command.index("-pixel_format") + 1] == "rgb24"
    assert command[command.index("-framerate") + 1] == "30"
    assert kwargs["start_new_session"]


def test_paused_sim_time_never_adds_filler_frames(recorder):
    instance, encoder, _ = recorder
    viewer = Viewer()
    assert instance.record(viewer, 0.1)
    for _ in range(100):
        assert not instance.record(viewer, 0.1)
    assert viewer.reads == len(encoder.stdin.frames) == 1


def test_late_sample_does_not_duplicate_pixels_for_missed_times(recorder):
    instance, encoder, _ = recorder
    viewer = Viewer()
    instance.record(viewer, 0)
    instance.record(viewer, 10)
    assert len(encoder.stdin.frames) == 2
    assert not instance.record(viewer, 10)


def test_reset_sim_clock_starts_next_segment_without_waiting_for_old_time(recorder):
    instance, encoder, _ = recorder
    viewer = Viewer()
    instance.record(viewer, 2)
    instance.record(viewer, 0)
    assert len(encoder.stdin.frames) == 2
    assert not instance.record(viewer, 0)


def test_resize_fails_instead_of_corrupting_video(recorder):
    instance, _, _ = recorder
    viewer = Viewer()
    instance.record(viewer, 0)
    viewer.shape = (4, 4, 3)
    with pytest.raises(RuntimeError, match="resized"):
        instance.record(viewer, 1)


def test_encoder_is_finalized_and_reaped_once(recorder):
    instance, encoder, _ = recorder
    instance.record(Viewer(), 0)
    instance.close()
    instance.close()
    assert encoder.stdin.closed
    assert encoder.waited
    assert instance._log is None


def test_encoder_start_failure_closes_its_log(tmp_path, monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda _: "/test/ffmpeg")
    monkeypatch.setattr(
        video.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed"))
    )
    instance = video.GlVideoRecorder(video.GlVideoConfig(tmp_path / "view.mp4", 30), physics_hz=60)
    with pytest.raises(OSError, match="spawn failed"):
        instance.record(Viewer(), 0)
    assert instance._closed
    assert instance._log is None
    assert instance._process is None


def test_existing_video_is_never_overwritten(tmp_path):
    path = tmp_path / "old.mp4"
    path.write_bytes(b"old experiment")
    with pytest.raises(FileExistsError):
        video.GlVideoRecorder(video.GlVideoConfig(path, 30), physics_hz=60)
    assert path.read_bytes() == b"old experiment"


def test_encoder_failure_is_not_reported_as_successful_recording(recorder):
    instance, encoder, _ = recorder
    instance.record(Viewer(), 0)
    encoder.wait = lambda timeout: 1
    with pytest.raises(RuntimeError, match="encoder failed"):
        instance.close()
    assert encoder.stdin.closed
    assert instance._log is None


def test_real_backend_hook_records_observer_and_excludes_mounted_sensor_renders(recorder, monkeypatch):
    # Execute the real run() wiring without executing the script's heavy
    # Newton/GPU bootstrap. Existing render/get_frame tests exercise the
    # separate camera implementation; this protects the recording entry point.
    instance, encoder, _ = recorder
    viewer = Viewer()
    viewer.view = "observer"
    viewer.camera = SimpleNamespace(width=13, height=7)
    viewer.camera.update_screen_size = lambda w, h: viewer.camera.__dict__.update(width=w, height=h)
    viewer._invalidate_pbo = lambda: None

    class Renderer:
        def __init__(self):
            self.native = (13, 7)
            self.window = SimpleNamespace(get_size=lambda: self.native, get_framebuffer_size=lambda: self.native)
            self._screen_width, self._screen_height = self.native
            self._resize_callbacks = [lambda *_: viewer.camera.update_screen_size(*self.native)]
            self._setup_frame_buffer()
            self.resized = False

        def _setup_frame_buffer(self):
            viewer.shape = (self._screen_height, self._screen_width, 3)

        def resize(self, *_):
            self._screen_width, self._screen_height = self.native
            self._setup_frame_buffer()

        def wm_event(self):
            if not self.resized:
                self.resized = True
                self.native = (1619, 1058)
                self.resize(*self.native)
                for callback in self._resize_callbacks:
                    callback(*self.native)

    viewer.renderer = Renderer()
    read_views = []
    get_frame = viewer.get_frame
    viewer.get_frame = lambda **kwargs: read_views.append(viewer.view) or get_frame(**kwargs)
    viewer.begin_frame = lambda _: None
    viewer.log_state = lambda _: None
    viewer.end_frame = viewer.renderer.wm_event
    viewer.close = lambda: None
    sim = SimpleNamespace(
        _state_0=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: np.zeros((3, 7)))),
        _sim_time_acc=0,
        _body_paths=[],
        _lifecycle=SimpleNamespace(_node=object(), start_listener=lambda: None),
        start_joint_feedback_listener=lambda: None,
        start_deformable_publisher=lambda: None,
        _call_plugin=lambda *args: None,
        _plugin_ctx=lambda: None,
    )

    def sensors(_viewer, _config, _poses, _indices, render_scene):
        for name in ("head", "hand_left", "hand_right"):
            viewer.view = name
            render_scene()
        viewer.view = "observer"
        render_scene()
        return ()

    def run_loop(*, render_hook, exit_check):
        render_hook(1, 1, 0)
        sim._pause_observation_callback()
        # Even a wall-clock update with no simulation advancement adds no frame.
        render_hook(100, 100, 0)
        return 0

    session = SimpleNamespace(
        sim=sim,
        physics_hz=60,
        render_hz=30,
        realtime_factor=0.5,
        scene_usda="unused",
        dt=1 / 60,
        startup=lambda **kwargs: None,
        bind_newton_body_topology=lambda: None,
        publish_plan=object(),
        core=object(),
        run=run_loop,
    )
    monkeypatch.setitem(
        sys.modules,
        "genie_sim_engine_newton.engine.newton.visualizers.gl_observation",
        SimpleNamespace(
            GlObservationConfig=SimpleNamespace(from_params=lambda _: SimpleNamespace(views=(), width=4, height=2)),
            GlObservationPublisher=lambda *_: SimpleNamespace(publish=lambda _: None, close=lambda: None),
            body_indices=lambda *_: (),
            capture_views=sensors,
            fixed_render_size=fixed_gl.fixed_render_size,
            mounted_camera_pose=fixed_gl.mounted_camera_pose,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "genie_sim_engine_newton.engine.newton.visualizers.gl_video",
        SimpleNamespace(GlVideoConfig=video.GlVideoConfig, GlVideoRecorder=lambda *args, **kwargs: instance),
    )
    monkeypatch.setitem(
        sys.modules, "genie_sim_engine_core.common.loop", SimpleNamespace(_publish_tick=lambda *args: None)
    )
    script = PACKAGE / "scripts/genie_sim_engine_newton_node.py"
    tree = ast.parse(script.read_text())
    run_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    namespace = dict(
        SimpleLogger=lambda _: SimpleNamespace(info=lambda _: None, warn=lambda _: None),
        GenieSimEngineSession=lambda **kwargs: session,
        _VISUALIZER="newton",
        _HEADLESS=False,
        simulation_app=None,
        ExitStack=ExitStack,
        time=time,
        _create_newton_viewer=lambda *args: viewer,
    )
    exec(compile(ast.Module(body=[run_function], type_ignores=[]), str(script), "exec"), namespace)
    assert namespace["run"](dict(newton_video_path=str(instance.config.path), newton_video_fps="30")) == 0
    assert read_views == ["observer"]
    assert len(encoder.stdin.frames) == 1
    assert len(encoder.stdin.frames[0]) == 2 * 4 * 3
    assert viewer.renderer.native == (1619, 1058)
    assert viewer.shape == (1058, 1619, 3)


@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="optional CPU ffmpeg tools missing"
)
def test_cpu_encoder_output_is_playable_mp4_with_requested_rate_and_frame_count(tmp_path):
    # These are codec-test pixels, not simulator validation or a user artifact.
    path = tmp_path / "codec_test.mp4"
    instance = video.GlVideoRecorder(video.GlVideoConfig(path, 30), physics_hz=60)
    viewer = Viewer()
    viewer.shape = (16, 24, 3)
    try:
        for tick in range(12):
            instance.record(viewer, tick / 60)
    finally:
        instance.close()
    metadata = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )["streams"][0]
    assert metadata == dict(width=24, height=16, r_frame_rate="30/1", nb_read_frames="6")


def _isaac_run_wiring(
    recorder, monkeypatch, loop, *, render_error=None, authored_visuals=None, assets_root=None, snapshot_error=None
):
    instance, encoder, _ = recorder
    events = []
    views = tuple(SimpleNamespace(name=name, fov_degrees=65) for name in ("head", "hand_left", "hand_right"))
    poses = np.zeros((3, 7))
    poses[:, 6] = 1
    sim = SimpleNamespace(
        _state_0=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: poses)),
        _sim_time_acc=0,
        _body_paths=["head", "left", "right"],
        _lifecycle=SimpleNamespace(_node=object(), start_listener=lambda: None),
        start_joint_feedback_listener=lambda: None,
        start_deformable_publisher=lambda: None,
        ticks=0,
        _scene_cfg={"newton": {"authored_visuals": authored_visuals}},
        shutdown=lambda: None,
    )

    class Snapshot:
        authored_visuals_from_config = staticmethod(scene_snapshot.SceneSnapshot.authored_visuals_from_config)

        def __init__(self, actual_sim, **kwargs):
            assert actual_sim is sim
            assert kwargs["authored_robot"] == {"asset_path": "/assets/robot_visual.usda"}
            assert kwargs["robot_bodies"] == {"head": "/genie/head"}
            if authored_visuals is not None:
                descriptor, prefixes = self.authored_visuals_from_config(authored_visuals, assets_root=assets_root)
                assert kwargs["authored_scene"] == descriptor
                assert kwargs["hide_shape_prefixes"] == prefixes
                events.append("authored_scene_configured")
            else:
                assert "authored_scene" not in kwargs
            if snapshot_error is not None:
                raise snapshot_error

        def capture(self, sim_time):
            events.append("snapshot")
            return {"sim_time": sim_time}

        def close(self):
            events.append("scene_closed")

    class Renderer:
        def __init__(self, *args, **kwargs):
            events.append("worker_started")

        def render(self, scene, cameras, width, height):
            names = [camera["name"] for camera in cameras]
            events.append(tuple(names))
            assert scene["sim_time"] == sim._sim_time_acc
            if names == ["observer"] and render_error is not None:
                raise render_error
            return {name: np.full((height, width, 3), 9, dtype=np.uint8) for name in names}

        def close(self):
            events.append("worker_closed")

    session = SimpleNamespace(
        sim=sim,
        physics_hz=60,
        render_hz=30,
        realtime_factor=0.5,
        scene_usda="unused",
        dt=1 / 60,
        startup=lambda **_: None,
        bind_newton_body_topology=lambda: None,
        publish_plan=object(),
        core=SimpleNamespace(shutdown=lambda: None),
        run=lambda **kwargs: loop(sim, events, **kwargs),
    )
    monkeypatch.setitem(
        sys.modules, "ament_index_python.packages", SimpleNamespace(get_package_prefix=lambda _: "/test")
    )
    monkeypatch.setitem(
        sys.modules,
        "genie_sim_engine_newton.engine.newton.visualizers.isaacsim",
        SimpleNamespace(IsaacSimRenderer=Renderer),
    )
    monkeypatch.setitem(
        sys.modules,
        "genie_sim_engine_newton.engine.newton.visualizers.scene_snapshot",
        SimpleNamespace(SceneSnapshot=Snapshot),
    )
    monkeypatch.setitem(
        sys.modules,
        "genie_sim_engine_core.schema.manifest",
        SimpleNamespace(
            Manifest=SimpleNamespace(
                read=lambda _: SimpleNamespace(
                    robot_visual_usda=Path("/assets/robot_visual.usda"),
                    robot={"body_paths": {"head": "/genie/head"}},
                    base_path=assets_root,
                )
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "genie_sim_engine_newton.engine.newton.visualizers.gl_observation",
        SimpleNamespace(
            GlObservationConfig=SimpleNamespace(from_params=lambda _: SimpleNamespace(views=views, width=4, height=2)),
            GlObservationPublisher=lambda *_: SimpleNamespace(
                publish=lambda images: events.append(("images_published", len(images))), close=lambda: None
            ),
            body_indices=lambda *_: (0, 1, 2),
            capture_views=lambda *_: pytest.fail("GL capture called"),
            fixed_render_size=lambda *_: pytest.fail("GL framebuffer used"),
            mounted_camera_pose=lambda *_: (np.array([1, 0, 1]), np.array([0, 0, 0]), np.array([0, 0, 1])),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "genie_sim_engine_newton.engine.newton.visualizers.gl_video",
        SimpleNamespace(GlVideoConfig=video.GlVideoConfig, GlVideoRecorder=lambda *args, **kwargs: instance),
    )
    monkeypatch.setitem(
        sys.modules,
        "genie_sim_engine_core.common.loop",
        SimpleNamespace(_publish_tick=lambda *args: events.append("final_state_published")),
    )
    script = PACKAGE / "scripts/genie_sim_engine_newton_node.py"
    tree = ast.parse(script.read_text())
    run_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    namespace = dict(
        SimpleLogger=lambda _: SimpleNamespace(info=lambda _: None, warn=lambda _: None),
        GenieSimEngineSession=lambda **kwargs: session,
        _VISUALIZER="isaacsim",
        _HEADLESS=False,
        simulation_app=None,
        ExitStack=ExitStack,
        time=time,
        _create_newton_viewer=lambda *_: pytest.fail("Newton GL viewer created"),
    )
    exec(compile(ast.Module(body=[run_function], type_ignores=[]), str(script), "exec"), namespace)
    return namespace["run"], sim, events


def test_isaac_hook_waits_for_complete_sensor_readback_and_records_only_observer(recorder, monkeypatch):
    instance, encoder, _ = recorder

    def loop(sim, events, *, render_hook, exit_check):
        render_hook(1, 1, 0)
        sim._pause_observation_callback()
        render_hook(100, 100, 0)
        assert exit_check()
        return 0

    run, _, events = _isaac_run_wiring(recorder, monkeypatch, loop)
    assert (
        run(
            dict(
                newton_video_path=str(instance.config.path),
                isaac_render_log_path="unused.log",
                stage_manifest="manifest.json",
            )
        )
        == 0
    )
    assert events.count(("head", "hand_left", "hand_right")) == 2  # warmup and frozen readback
    assert events.count(("observer",)) == 1
    published = events.index("final_state_published")
    assert events[published : published + 4] == [
        "final_state_published",
        "snapshot",
        ("head", "hand_left", "hand_right"),
        ("images_published", 3),
    ]
    assert len(encoder.stdin.frames) == 1
    assert encoder.stdin.frames[0] == bytes([9]) * (4 * 2 * 3)
    assert events[-2:] == ["scene_closed", "worker_closed"]


@pytest.mark.parametrize("failure_stage", ["capture", "encode"])
def test_isaac_observer_failure_swallowed_by_core_stops_before_next_tick_and_remains_fatal(
    recorder, monkeypatch, failure_stage
):
    instance, encoder, _ = recorder
    # Own an encoder with an earlier completed frame before this failing tick;
    # its cleanup must still happen when the next capture never completes.
    instance.record_rgb(np.zeros((2, 4, 3), np.uint8), -1)
    original_error = RuntimeError("worker readback failed") if failure_stage == "capture" else OSError("encoder broke")
    if failure_stage == "encode":

        def fail_write(_):
            raise original_error

        encoder.stdin.write = fail_write

    def loop(sim, events, *, render_hook, exit_check):
        # Mirror the core loop: exit_check runs before physics, and an ordinary
        # Exception from render_hook is logged/swallowed after that tick.
        for _ in range(2):
            if not exit_check():
                events.append("exit_before_next_tick")
                break
            sim.ticks += 1
            try:
                render_hook(1, 1, sim._sim_time_acc)
            except Exception:
                events.append("core_swallowed_render_error")
        assert sim.ticks == 1, "renderer failure allowed another physics tick"
        assert not exit_check()
        return 0  # A swallowed error must not turn this into a successful run.

    run, sim, events = _isaac_run_wiring(
        recorder, monkeypatch, loop, render_error=original_error if failure_stage == "capture" else None
    )
    with pytest.raises(RuntimeError, match="observer rendering failed") as raised:
        run(
            dict(
                newton_video_path=str(instance.config.path),
                isaac_render_log_path="unused.log",
                stage_manifest="manifest.json",
            )
        )
    # record_rgb wraps write failures, whereas a capture failure propagates
    # unchanged. Both causes remain visible behind the final run failure.
    if failure_stage == "capture":
        assert raised.value.__cause__ is original_error
    else:
        assert raised.value.__cause__.__cause__ is original_error
    assert events.count("core_swallowed_render_error") == 1
    assert events.count("exit_before_next_tick") == 1
    assert events.count(("observer",)) == 1
    assert not any(isinstance(event, tuple) and event[0] == "images_published" for event in events)
    assert events[-2:] == ["scene_closed", "worker_closed"]
    assert sim._pause_observation_callback is None
    assert encoder.stdin.closed and encoder.waited
    assert instance._log is None and instance._process is None


def test_isaac_node_passes_scene_yaml_visual_descriptor_with_manifest_asset_root(recorder, monkeypatch, tmp_path):
    asset = tmp_path / "scene/background.usda"
    asset.parent.mkdir()
    asset.touch()
    config = {
        "asset_path": "scene/background.usda",
        "root_prim": "/World",
        "hide_prims": ["/World/objects"],
        "use_asset_lights": True,
        "body_visuals": [
            {"body": "chef_wok", "source_prim": "/World/pot", "reference_pose": [0.9949, 0, 1.09, 0, 0, 0, 1]}
        ],
        "hide_shape_prefixes": ["chef_stove_"],
    }
    run, _, events = _isaac_run_wiring(
        recorder, monkeypatch, lambda *args, **kwargs: 0, authored_visuals=config, assets_root=tmp_path
    )
    assert run(dict(isaac_render_log_path="unused.log", stage_manifest="manifest.json")) == 0
    assert "authored_scene_configured" in events
    assert events[-2:] == ["scene_closed", "worker_closed"]


def test_isaac_node_rejects_invalid_visual_asset_before_starting_worker(recorder, monkeypatch, tmp_path):
    run, _, events = _isaac_run_wiring(
        recorder,
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("simulation started"),
        authored_visuals={"asset_path": "missing.usda"},
        assets_root=tmp_path,
    )
    with pytest.raises(ValueError, match="does not exist"):
        run(dict(isaac_render_log_path="unused.log", stage_manifest="manifest.json"))
    assert "worker_started" not in events


def test_isaac_node_closes_worker_when_snapshot_construction_fails(recorder, monkeypatch):
    run, _, events = _isaac_run_wiring(
        recorder,
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("simulation started"),
        snapshot_error=ValueError("visual body missing"),
    )
    with pytest.raises(ValueError, match="visual body missing"):
        run(dict(isaac_render_log_path="unused.log", stage_manifest="manifest.json"))
    assert events == ["worker_started", "worker_closed"]
