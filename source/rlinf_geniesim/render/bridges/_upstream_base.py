# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared base for upstream-subprocess render backends."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from ...child_lifetime import bind_to_server
from .base import (
    CameraSpec,
    RenderHealthStatus,
    RenderResult,
    VecRenderBackend,
)
from ._ros_image_bridge import RosRenderBridge

if TYPE_CHECKING:
    from rlinf_geniesim.physics.backends.base import SceneSchema


class _UpstreamRenderBridge(VecRenderBackend):
    """Common subprocess plumbing for OVRTX / IsaacSim render backends.

    Subclasses set:
      * ``UPSTREAM_PACKAGE`` / ``UPSTREAM_EXEC``
      * ``BACKEND_NAME``
      * ``OUTPUT_CHANNELS``
      * ``SUPPORTS_BATCHING``

    May override ``_extra_ros_params(self) -> dict[str, str]`` and
    ``_image_topic(self, camera: CameraSpec) -> str``.
    """

    # True when this backend's render process starts a frame merely because
    # the tick changed — then poses for an uncollected step cost a wasted
    # frame that the collected one has to queue behind. See update_poses.
    _skip_poses_when_uncollected: bool = False

    # Backends whose render stage is GridCloner-organized (every driveable prim
    # under /World/envs/env_<i>/) set this True so update_poses publishes the
    # multi-env body frames even at num_envs==1 (isaac_tiled overrides).
    _gridcloner_single_env_frames = False

    UPSTREAM_PACKAGE: str = ""
    UPSTREAM_EXEC: str = ""
    BACKEND_NAME: str = ""
    OUTPUT_CHANNELS: list[str] = ["rgb"]
    SUPPORTS_BATCHING: bool = False

    # ------------------------------------------------------------------

    def __init__(
        self,
        num_envs: int,
        cameras: list[CameraSpec],
        render_hz: float,
        *,
        stage_manifest: str = "",
        ovrtx_root: str = "",
        ros_log_level: str = "warn",
        ready_timeout_s: float = 180.0,  # OVRTX shader compile can take 90-120s on first run
        render_timeout_s: float = 2.0,
        # "strict" tick barrier | "latest" (async freshest-frame). Default
        # "latest" here: the OVRTX ROS-image path stalls under sustained strict
        # lockstep (see frame-sync doc §OVRTX), so it opts into strict explicitly
        # once tuned. The tiled backend sets "strict" itself (SHM path, verified).
        render_sync: str = "latest",
        # Multi-env layout config handed down by build_server (the generic layer
        # no longer organizes multi-env — each backend does, using these).
        env_spacing: float = 15.0,
        head_height_m: float = 1.5,
        multi_env_output_dir: str = "",
        # Barrier stall recovery (see _barrier_stall_recover). ``None`` = take
        # the value from the env knob, so neither training config needs editing.
        barrier_recovery: Optional[bool] = None,
        barrier_max_stalls: Optional[int] = None,
        **backend_kwargs: Any,
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            cameras=cameras,
            render_hz=render_hz,
            **backend_kwargs,
        )
        if num_envs < 1:
            raise ValueError(f"{type(self).__name__}: num_envs must be >= 1 (got {num_envs})")
        # The multi-env stage_manifest must already
        # be a multi-env manifest (produced by
        # ``render/scene_assembly.py``); upstream OVRTX handles
        # the per-env Camera + RenderProduct setup from the manifest's
        # ``cameras`` array directly.
        self._stage_manifest = stage_manifest
        self._ovrtx_root = ovrtx_root
        self._ros_log_level = ros_log_level
        self._ready_timeout_s = float(ready_timeout_s)
        self._render_timeout_s = float(render_timeout_s)
        self._render_sync = str(render_sync)
        if self._render_sync not in {"strict", "latest"}:
            raise ValueError(
                f"{type(self).__name__}: render_sync must be 'strict' or " f"'latest', got {render_sync!r}"
            )

        self._proc: Optional[subprocess.Popen] = None
        self._pid: Optional[int] = None
        self._bridge: Optional[RosRenderBridge] = None
        # Global monotonic tick latched by ``update_poses``; the anchor pump
        # republishes THIS (not wall-clock) so the tick-in-stamp survives.
        self._current_tick: int = 0
        self._frame_index: int = 0
        # Bounded stale-frame recovery; reset/reload requires a fresh frame.
        self._barrier_recovery = (
            os.environ.get("RLINF_TILED_BARRIER_RECOVERY", "1") not in ("0", "false", "False", "no", "off")
            if barrier_recovery is None
            else bool(barrier_recovery)
        )
        self._barrier_max_stalls = int(
            os.environ.get("RLINF_TILED_BARRIER_MAX_STALLS", "5") if barrier_max_stalls is None else barrier_max_stalls
        )
        self._recovery_frame_valid = False
        self._barrier_stall_streak: int = 0
        self._barrier_stall_total: int = 0
        self._last_render_ms: float = 0.0
        self._frame_shm = None  # opaque ShmRegion; set in attach_shm
        self._layout = None

        # Resolve image topics. Prefer the manifest as the source of truth
        # (matches what the upstream renderer actually publishes) and fall
        # back to ``/<camera_name>/image_raw`` if the manifest is missing
        # or unreadable.
        self._image_topics: list[str] = self._resolve_image_topics()
        print(
            f"[{type(self).__name__}] subscribing to {len(self._image_topics)} "
            f"image topic(s): {self._image_topics}",
            flush=True,
        )
        # Multi-env tf expansion: env_spacing is handed down by build_server
        # (authoritative). When num_envs > 1 we rebroadcast every body's
        # tf_render N times with prefixed child_frame_id and an X offset
        # (= ``i * env_spacing``) so each env's per-env prims/anchors get driven.
        self._env_spacing: float = float(env_spacing)
        self._head_height_m: float = float(head_height_m)
        self._multi_env_output_dir: str = str(multi_env_output_dir)
        # Whether update_poses bakes the +X*env_spacing offset into the driven
        # /tf_render poses. USD-stage backends (OVRTX scene_multi) DO — their env
        # prims sit at the origin. The Isaac tiled backend does NOT: it publishes
        # RAW poses and its GridCloner node adds each env's grid origin.
        self._bake_env_offset: bool = True
        # Diagnostic toggle: set RLINF_DISABLE_ANCHOR_MIRROR=1 to skip
        # publishing per-env head_link3 tf mirrors. Useful to isolate
        # whether the head-camera anchor flicker comes from our mirror
        # racing upstream's long-name tf or from elsewhere.
        self._disable_anchor_mirror = bool(int(os.environ.get("RLINF_DISABLE_ANCHOR_MIRROR", "0") or "0"))

    def _resolve_image_topics(self) -> list[str]:
        """Read camera topic names from the stage manifest.

        Manifest schema (see upstream's assemble_scene)::

            { "cameras": [
                { "topic": "/genie_sim/head_front_camera_rgb",
                  "depth_topic": "/genie_sim/head_front_camera_depth",
                  "is_free_cam": false, ... },
                ...
            ] }

        - Single-env (num_envs == 1): match each ``CameraSpec`` to a
          manifest entry by position (i-th spec gets i-th non-free
          camera).
        - Multi-env (num_envs > 1): subscribe to **every** non-free
          camera in the manifest. ``scene_assembly`` lays them
          out as env_0's cams, env_1's cams, .... The per-env routing
          in ``render()`` uses ``parse_env_from_topic`` to slot each
          frame into ``rgb_buf[env_id, cam_id]``.
        """
        try:
            import json

            with open(self._stage_manifest, "r") as f:
                manifest = json.load(f)
        except Exception:  # noqa: BLE001
            # Either no manifest path set or unreadable — caller's
            # _verify_environment will catch the first case; just fall
            # back to the legacy naming so import-time logic stays usable.
            return [self._image_topic_fallback(c) for c in self._cameras]

        man_cams = manifest.get("cameras", []) or []
        non_free = [c for c in man_cams if not c.get("is_free_cam")]

        if self._num_envs > 1:
            topics: list[str] = []
            for cam in non_free:
                if "topic" in cam:
                    base = str(cam["topic"]).rstrip("/")
                    topics.append(base + "/image_raw")
            if topics:
                return topics
            return [self._image_topic_fallback(c) for c in self._cameras]

        topics = []
        for i, cam in enumerate(self._cameras):
            if i < len(non_free) and "topic" in non_free[i]:
                base = str(non_free[i]["topic"]).rstrip("/")
                topics.append(base + "/image_raw")
            else:
                topics.append(self._image_topic_fallback(cam))
        return topics

    def _image_topic_fallback(self, camera: CameraSpec) -> str:
        """Fallback topic name when the manifest doesn't tell us."""
        return f"/{camera['name']}/image_raw"

    def _read_env_spacing(self) -> float:
        """Pull ``env_spacing`` from the manifest (written by
        ``assemble_multi_env``). Defaults to 1.5 if missing."""
        try:
            import json

            with open(self._stage_manifest, "r") as f:
                return float(json.load(f).get("env_spacing", 15.0))
        except Exception:  # noqa: BLE001
            return 1.5

    def _anchor_link_names_from_manifest(self) -> list[str]:
        """Anchor link names parsed from the manifest's env_0 cameras.

        OVRTX's anchor system uses the first slash-separated segment of
        each non-free camera's ``path`` as the body_name key in
        ``cam_body_to_render_path_``. For env_0 the manifest entries
        produced by upstream's single-env assemble_scene look like
        ``path: "head_link3/Head_Front_Camera"`` — so the anchor link
        name is ``"head_link3"``.

        For multi-env, we *only* want the env_0 set (one entry per
        anchor link). The per-env ``env_<i>_<link>`` mirror names are
        constructed in ``update_poses`` by us, not by the manifest.
        """
        try:
            import json

            with open(self._stage_manifest, "r") as f:
                manifest = json.load(f)
        except Exception:  # noqa: BLE001
            return []
        out: list[str] = []
        seen: set[str] = set()
        for cam in manifest.get("cameras", []) or []:
            if cam.get("is_free_cam"):
                continue
            path = str(cam.get("path", "") or "")
            if not path or path.startswith("/"):
                continue
            head = path.split("/", 1)[0]
            if head.startswith("env_"):
                # Skip the per-env entries we generated; we only want
                # the env_0 anchor link names.
                continue
            if head and head not in seen:
                out.append(head)
                seen.add(head)
        return out

    # ------------------------------------------------------------------
    # Subprocess management
    # ------------------------------------------------------------------

    def _build_argv(self) -> list[str]:
        if not self.UPSTREAM_PACKAGE or not self.UPSTREAM_EXEC:
            raise RuntimeError(f"{type(self).__name__}: UPSTREAM_PACKAGE / UPSTREAM_EXEC " f"must be set")

        if not self._stage_manifest:
            raise ValueError(f"{type(self).__name__}: stage_manifest is required")

        argv: list[str] = [
            "ros2",
            "run",
            self.UPSTREAM_PACKAGE,
            self.UPSTREAM_EXEC,
        ]

        ros_params: dict[str, str] = {
            "stage_manifest": str(Path(self._stage_manifest).resolve()),
            "render_fps": f"{self._render_hz}",
        }
        if self._ovrtx_root:
            ros_params["ovrtx_root"] = self._ovrtx_root
        ros_params.update(self._extra_ros_params())

        argv += ["--ros-args", "--log-level", self._ros_log_level]
        for k, v in ros_params.items():
            argv += ["-p", f"{k}:={v}"]

        return argv

    def _extra_ros_params(self) -> dict[str, str]:
        return {}

    def _verify_environment(self) -> None:
        if shutil.which("ros2") is None:
            raise RuntimeError("'ros2' not on PATH — source ROS first.")
        if not Path(self._stage_manifest).is_file():
            raise FileNotFoundError(f"stage_manifest not found: {self._stage_manifest}")

    def _assemble_multi_env_stage(self) -> None:
        """Organize multi-env by building a ``scene_multi.usda`` (N robot copies
        + per-env cameras) from the single-env ``stage_manifest`` and pointing
        this backend at it. This is the USD-stage backends' way (OVRTX / Isaac
        Kit) — the Isaac *tiled* backend does NOT use it (it clones with
        GridCloner in its own node). Called from a subclass ``start()`` BEFORE
        launching the upstream node; no-op for num_envs<=1. Runs only in the
        container (imports pxr via scene_assembly), so construction on a
        plain host stays cheap."""
        if self._num_envs <= 1 or not self._stage_manifest:
            return
        from ..scene_assembly import assemble_multi_env
        from ...server_identity import default_multi_env_dir

        out_dir = self._multi_env_output_dir or default_multi_env_dir(self._num_envs)
        multi = assemble_multi_env(
            single_env_manifest=self._stage_manifest,
            num_envs=self._num_envs,
            output_dir=out_dir,
            env_spacing=self._env_spacing,
            head_height_m=self._head_height_m,
        )
        print(
            f"[{type(self).__name__}] assembled scene_multi " f"(num_envs={self._num_envs}): {multi}",
            flush=True,
        )
        self._stage_manifest = multi
        # Re-resolve per-env image topics from the assembled multi manifest.
        self._image_topics = self._resolve_image_topics()

    def start(self) -> None:
        if self._started:
            return
        self._verify_environment()
        argv = self._build_argv()

        env = os.environ.copy()
        # Capture child stdout+stderr to a file so the operator can tail
        # it; previously we PIPE'd it into a buffer no-one read, which
        # both hid useful errors AND risked back-pressuring the child
        # once the pipe filled up.
        log_path = f"/tmp/rlinf_{self.BACKEND_NAME}_{os.getpid()}.log"
        self._log_path = log_path
        self._log_fp = open(log_path, "w", buffering=1)
        print(
            f"[{type(self).__name__}] child stdout/stderr -> {log_path}",
            flush=True,
        )
        argv, env = bind_to_server(argv, env)
        self._proc = subprocess.Popen(
            argv,
            env=env,
            stdout=self._log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # so killpg in close() kills the
            # ros2 run + child render node together
        )
        self._pid = self._proc.pid

        # Bring up the rclpy bridge before waiting — we use the first
        # frame on any camera topic as the readiness signal.
        self._bridge = RosRenderBridge(node_name=f"rlinf_{self.BACKEND_NAME}_render_bridge")
        self._bridge.start(self._image_topics)

        self._wait_ready()
        self._started = True

        # Background pump: keep the anchor mirror fresh at ~100 Hz so
        # OVRTX always has a populated anchor when it samples a frame,
        # independent of how often the trainer calls ``update_poses``.
        # See ``_latest_anchor_*`` setup in update_poses.
        import threading

        self._anchor_pump_stop = threading.Event()
        # (names, poses, tick) latched as ONE tuple so the pump can never pair a
        # fresh tick with the previous call's poses — a single attribute read is
        # atomic under the GIL, closing the sub-ms tick/poses skew window.
        self._latest_anchor = None
        self._anchor_pump_thread = threading.Thread(
            target=self._anchor_pump_loop,
            daemon=True,
            name=f"{self.BACKEND_NAME}_anchor_pump",
        )
        self._anchor_pump_thread.start()

    def _anchor_pump_loop(self) -> None:
        """Republish the latched anchor mirror at ~100 Hz."""
        period = 1.0 / 100.0
        while not self._anchor_pump_stop.is_set():
            latest = self._latest_anchor  # atomic read of (names,poses,tick)
            if latest is not None and self._bridge is not None:
                names, poses, tick = latest
                try:
                    self._bridge.publish_tf_render(
                        body_names=list(names),
                        body_xyzwxyz=poses,
                        tick=tick,
                    )
                except Exception:  # noqa: BLE001
                    pass
            self._anchor_pump_stop.wait(period)

    def _wait_ready(self) -> None:
        """Wait until the upstream renderer is "alive enough" to accept poses.

        IMPORTANT: we **do not** wait for the first image to arrive here.
        The upstream OVRTX node only emits frames after it receives
        ``/tf_render``, but ``/tf_render`` is published by
        ``sim_server`` inside the main step loop — which can only start
        after this method returns. Waiting for an image here would
        deadlock.

        Phase-1.5 readiness check: just confirm the subprocess didn't
        crash within the grace period. The first ``render()`` call in
        the main loop will then wait (with its own timeout) for the
        first frame.
        """
        grace = min(5.0, self._ready_timeout_s)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                tail = ""
                log_path = getattr(self, "_log_path", "")
                if log_path:
                    try:
                        with open(log_path, "r") as f:
                            tail = f.read()
                    except Exception:  # noqa: BLE001
                        pass
                self._cleanup_after_failure()
                raise RuntimeError(
                    f"{type(self).__name__}: upstream "
                    f"'{self.UPSTREAM_EXEC}' exited with code "
                    f"{self._proc.returncode if self._proc else '?'}.\n"
                    f"Log: {log_path}\n"
                    f"Tail:\n{tail[-2000:]}"
                )
            time.sleep(0.1)
        # Survived the grace period — assume the renderer is up and
        # waiting for /tf_render. If it actually never starts emitting,
        # render() will time out and report it.
        return

    def _cleanup_after_failure(self) -> None:
        if self._bridge is not None:
            try:
                self._bridge.close()
            except Exception:  # noqa: BLE001
                pass
            self._bridge = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3.0)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    def render_proc_alive(self) -> bool:
        """False once a spawned render subprocess has EXITED (e.g. the user
        closed the Isaac Sim GUI window, or the node crashed). True while it
        runs, or when there is no subprocess to watch (not started / torn down).
        The server's watchdog polls this to tear the whole sim down with the
        renderer."""
        proc = self._proc
        return proc is None or proc.poll() is None

    def close(self) -> None:
        if not self._started and self._proc is None and self._bridge is None:
            return
        bridge = self._bridge
        proc = self._proc
        self._bridge = None
        self._proc = None
        self._started = False
        self._shm_attached = False
        self._schema_bound = False

        if bridge is not None:
            try:
                bridge.close()
            except Exception:  # noqa: BLE001
                pass
        if proc is None:
            return
        pgid = proc.pid
        try:
            os.killpg(pgid, signal.SIGINT)
            proc.wait(timeout=10.0)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=5.0)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass

    # ------------------------------------------------------------------
    # Wiring contracts
    # ------------------------------------------------------------------

    def attach_shm(self, frame_shm: Any, layout: Any) -> None:
        """Receive the Frame SHM region.

        ``frame_shm`` is a ``comm.shm.layout.ShmRegion`` (or compatible
        object exposing a ``field(name) -> ndarray`` accessor).
        """
        self._ensure_started()
        self._frame_shm = frame_shm
        self._layout = layout
        self._shm_attached = True

    def bind_scene_schema(self, schema: "SceneSchema") -> None:
        self._ensure_started()
        self._schema = schema
        # The renderer resolves prim paths from its already-loaded manifest.
        self._schema_bound = True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @staticmethod
    def _multi_env_body_frames(
        base_names: list[str],
        body_poses: np.ndarray,
        num_envs: int,
        env_spacing: float,
        anchor_links: list[str],
        disable_anchor: bool = False,
    ) -> tuple[list[str], np.ndarray]:
        """Build (frame_names, poses) for native multi-world body separation.

        For each env ``i`` and body ``b``, emit an absolute scene_multi prim
        frame ``/World/envs/env_<i>/<label>`` carrying the body's world pose
        shifted +X by ``i * env_spacing`` (the physics worlds overlap at the
        origin; the scene_multi links carry ``omni:resetXformStack`` so
        render_node writes the pose as a WORLD transform). When a body's leaf
        name is an anchor link, ALSO emit its camera-anchor frame
        ``/RenderOVRTX/Cameras/[env_<i>_]<link>`` with the same pose so the
        per-env head camera tracks that env's robot.

        Pure function (no ROS / ``self`` state) — host unit-testable.

        Returns (names, poses_ndarray) where poses_ndarray is (M, 7) float32,
        M = total emitted frames (bodies + anchors).
        """
        n = min(len(base_names), int(body_poses.shape[1]))
        n_eff = min(int(num_envs), int(body_poses.shape[0]))
        anchors = set(anchor_links or []) if not disable_anchor else set()

        labels = [str(nm).lstrip("/") for nm in base_names[:n]]
        short_links = [lb.rsplit("/", 1)[-1] for lb in labels]
        anchor_mask = [sl in anchors for sl in short_links] if anchors else []

        poses_flat = body_poses[:n_eff, :n, :].astype(np.float32, copy=True)
        if env_spacing != 0.0:
            offsets = np.arange(n_eff, dtype=np.float32) * env_spacing
            poses_flat[:, :, 0] += offsets[:, None]

        names_out: list[str] = []
        anchor_indices: list[int] = []

        for env_id in range(n_eff):
            base_idx = env_id * n
            for bi in range(n):
                names_out.append(f"/World/envs/env_{env_id}/{labels[bi]}")
                if anchors and anchor_mask[bi]:
                    short = short_links[bi] if env_id == 0 else f"env_{env_id}_{short_links[bi]}"
                    names_out.append(f"/RenderOVRTX/Cameras/{short}")
                    anchor_indices.append(base_idx + bi)

        poses_reshaped = poses_flat.reshape(n_eff * n, -1)

        if anchor_indices:
            anchor_poses = poses_reshaped[anchor_indices]
            total = n_eff * n + len(anchor_indices)
            all_poses = np.empty((total, poses_reshaped.shape[1]), dtype=np.float32)
            body_cursor = 0
            anchor_cursor = 0
            out_cursor = 0
            for env_id in range(n_eff):
                for bi in range(n):
                    all_poses[out_cursor] = poses_reshaped[body_cursor]
                    out_cursor += 1
                    body_cursor += 1
                    if anchors and anchor_mask[bi]:
                        all_poses[out_cursor] = anchor_poses[anchor_cursor]
                        out_cursor += 1
                        anchor_cursor += 1
            poses_out = all_poses[:out_cursor]
        else:
            poses_out = poses_reshaped

        return names_out, poses_out

    def update_poses(
        self,
        body_poses: np.ndarray,
        sim_time: np.ndarray,
        camera_poses: Optional[np.ndarray] = None,
        object_poses: Optional[np.ndarray] = None,
        cloth_vertices: Optional[np.ndarray] = None,
        body_names: Optional[list[str]] = None,
        tick: int = 0,
        will_collect: bool = True,
    ) -> None:
        self._ensure_started()
        self._ensure_schema_bound()
        # ``_current_tick`` is deliberately NOT latched when skipping: it is
        # the render barrier's TARGET, and it stays satisfiable only while it
        # means "the newest tick the render process was told about". Latching
        # a tick we then decline to publish sets a target no frame can carry,
        # and the next collected step blocks until the barrier cap.
        if not will_collect and self._skip_poses_when_uncollected:
            return
        # Latch the tick BEFORE the branches so both publish sites and the
        # background anchor pump republish this tick (not wall-clock).
        self._current_tick = int(tick)
        if self._bridge is None:
            return

        # ``body_names`` should be the live names upstream is currently
        # publishing on /tf_render (absolute USD paths like
        # ``/World/envs/env_1/Geometry/.../head_link3``). The pre-bound
        # schema can be stale (built before upstream's first tf), so
        # callers should pass fresh names via ``body_names`` whenever
        # available.
        if not body_names:
            names = list(getattr(self._schema, "body_link_names", [])) if hasattr(self, "_schema") else []
            if not names:
                names = [f"body_{i}" for i in range(body_poses.shape[1])]
            body_names = names
        n = min(len(body_names), body_poses.shape[1])
        base_names = list(body_names[:n])
        base_poses = body_poses[0, :n, :]

        # Native multi-world body routing.
        #
        # With native replication, body_poses is (N, B, 7) and the labels are
        # bare per-env names rather than absolute paths. Drive each world's
        # bodies onto its own prim by publishing an absolute
        # /World/envs/env_<i>/<label>; the render node writes straight to that
        # prim and the link carries omni:resetXformStack, so it is treated as
        # a world transform. Physics worlds all sit at the origin, so the
        # visual +X spacing is baked in here. Camera anchors are mirrored per
        # env as well.
        #
        # GridCloner-based backends take this branch at any env count: even at
        # N==1 every drivable prim lives under /World/envs/env_0/, where the
        # engine's own frames do not resolve.
        gridcloner_n1 = self._num_envs == 1 and self._gridcloner_single_env_frames
        if (self._num_envs > 1 and body_poses.shape[0] > 1) or gridcloner_n1:
            anchor_links = self._anchor_link_names_from_manifest()
            names_out, poses_out = self._multi_env_body_frames(
                base_names[:n],
                body_poses,
                self._num_envs,
                self._env_spacing if self._bake_env_offset else 0.0,
                anchor_links,
                disable_anchor=getattr(self, "_disable_anchor_mirror", False),
            )
            if names_out:
                arr_out = poses_out if isinstance(poses_out, np.ndarray) else np.stack(poses_out, axis=0)
                if getattr(self, "_flat_pose_channel", False) and hasattr(self._bridge, "publish_flat_poses"):
                    # FLAT channel (isaac_tiled): one Float64MultiArray instead
                    # of ~500 TransformStamped — the node's rclpy deserialization
                    # of the TFMessage cost ~140ms/frame at 8 envs.
                    self._bridge.publish_flat_poses(names_out, arr_out, self._current_tick)
                else:
                    self._bridge.publish_tf_render(
                        body_names=names_out,
                        body_xyzwxyz=arr_out,
                        tick=self._current_tick,
                    )
                # Latch names+poses+tick atomically (same tick these poses carry).
                self._latest_anchor = (list(names_out), arr_out.copy(), self._current_tick)
            if not getattr(self, "_tf_diag_done", False):
                print(
                    f"[{type(self).__name__}] first multi-env body+anchor "
                    f"/tf_render: num_envs={self._num_envs} "
                    f"worlds={body_poses.shape[0]} bodies/env={n} "
                    f"frames={len(names_out)} anchor_links={anchor_links}",
                    flush=True,
                )
                self._tf_diag_done = True
            return

        # Camera anchor mirroring.
        #
        # The engine publishes absolute USD prim paths, while the anchor
        # mechanism matches short names only (the first segment of each
        # ``cam.path``), so absolute paths never reach a camera anchor.
        #
        # Publish a short-name tf per ``env_<i>`` anchor: ``<anchor_link>``
        # for env_0 and ``env_<i>_<anchor_link>`` beyond it, taking the
        # transform from the absolute path containing ``/envs/env_<i>/`` that
        # ends with the anchor link name.
        anchor_links = self._anchor_link_names_from_manifest()

        mirror_names: list[str] = []
        mirror_poses: list[np.ndarray] = []

        def _find_pose(env_id: int, link: str) -> Optional[np.ndarray]:
            # Two upstream path conventions:
            #   N == 1:  /<robot_prefix>/Geometry/.../<link>
            #              e.g. /genie/Geometry/.../head_link3
            #   N >  1:  /World/envs/env_<i>/Geometry/.../<link>
            # We support both — the env_id==0 branch first tries the
            # multi-env path (in case the user does run N=1 inside a
            # multi-env stage layout), and falls back to the legacy
            # single-env ``/<robot_prefix>/...`` form.
            link_suffix = f"/{link}"
            multi_env_tag = f"/envs/env_{env_id}/"
            for idx, nm in enumerate(base_names):
                if multi_env_tag in nm and nm.endswith(link_suffix):
                    return base_poses[idx]
            if env_id == 0:
                # Single-env legacy path: any absolute name ending with
                # /<link> that does NOT have an /envs/env_<j>/ segment.
                for idx, nm in enumerate(base_names):
                    if "/envs/env_" in nm:
                        continue
                    if nm.endswith(link_suffix):
                        return base_poses[idx]
            return None

        for env_id in range(self._num_envs):
            for link in anchor_links:
                pose = _find_pose(env_id, link)
                if pose is None:
                    continue
                # Send as ABSOLUTE USD path so OVRTX's render_node
                # (render_node.cpp:597-602) takes the leading-slash
                # branch and writes directly to the anchor prim,
                # bypassing its cam_body_to_render_path_ map. That map
                # is also (legitimately) hit by upstream's long-name
                # /genie/Geometry/.../head_link3 tf, and the two writes
                # were racing — head_link3 anchor flickered between
                # the correct world transform and a duplicate write
                # via the anchor map (causing the black/visible
                # alternation).
                short = link if env_id == 0 else f"env_{env_id}_{link}"
                abs_anchor_path = f"/RenderOVRTX/Cameras/{short}"
                mirror_names.append(abs_anchor_path)
                mirror_poses.append(pose)

        if mirror_names and not getattr(self, "_disable_anchor_mirror", False):
            mirror_arr = np.stack(mirror_poses, axis=0)
            self._bridge.publish_tf_render(
                body_names=mirror_names,
                body_xyzwxyz=mirror_arr,
                tick=self._current_tick,
            )
            # Latch the most recent anchor mirror so a background pump
            # can keep it fresh at upstream-tf rate. Without this the
            # anchor only gets refreshed at trainer-STEP rate (~20 Hz
            # in the demo viewer), much slower than upstream's 100 Hz
            # tf_render — which makes the head camera flicker between
            # "anchor populated" and "anchor stale" frames.
            # Latch names+poses+tick atomically (same tick these poses carry).
            self._latest_anchor = (list(mirror_names), mirror_arr.copy(), self._current_tick)

        # One-shot diagnostic — see what we're forwarding.
        if not getattr(self, "_tf_diag_done", False):
            print(
                f"[{type(self).__name__}] first anchor-mirror /tf_render: "
                f"num_envs={self._num_envs} anchor_links={anchor_links} "
                f"upstream_body_count={len(base_names)} "
                f"mirror_count={len(mirror_names)} "
                f"mirror_names={mirror_names}",
                flush=True,
            )
            self._tf_diag_done = True

    # ------------------------------------------------------------------
    # Barrier stall recovery — timeout + previous-frame substitution
    # ------------------------------------------------------------------

    # Default knob hint for the fatal message. The tiled backend's cap really
    # is RLINF_TILED_BARRIER_CAP_S; other paths pass their own hint rather than
    # naming a knob that does not govern them -- a message pointing at the
    # wrong knob makes debugging a stall much harder (see the READY_TIMEOUT
    # note in the tests).
    _BARRIER_KNOB_HINT = (
        " — to wait longer, tune RLINF_TILED_BARRIER_CAP_S; "
        "RLINF_TILED_READY_TIMEOUT governs SHM attach and has no effect here"
    )

    def on_reset(self, env_ids, seeds=None, material_bindings=None):
        self._recovery_frame_valid = False

    def _barrier_stall_recover(
        self,
        *,
        rendered_tick: Any,
        target: int,
        waited_s: float,
        cold_first: bool,
        knob_hint: Optional[str] = None,
    ) -> None:
        """Allow another physics tick after a stall within a bounded budget.

        A missed pose tick can stall the watermark until a newer tick arrives.
        Returning a dropped frame lets the caller advance, but does not promise
        recovery: persistent stalls and invalid episode frames still fail.
        """
        base = (
            f"{type(self).__name__}: barrier stuck — node alive but "
            f"rendered_tick={rendered_tick} never reached target={target} in "
            f"{waited_s:.0f}s (tick threading broken?)"
            f"{self._BARRIER_KNOB_HINT if knob_hint is None else knob_hint}"
        )

        # No previous frame exists yet: the buffer is still all zeros, so
        # substituting would feed the policy a black image AND hide a genuine
        # cold-start failure (the 300s RLINF_TILED_FIRST_FRAME_CAP_S case).
        if cold_first or self._frame_index <= 0 or not self._recovery_frame_valid:
            why = "cold start or scene generation mismatch" if cold_first else "no valid frame this episode"
            raise RuntimeError(f"{base} — {why}, no previous frame to reuse, not substituting")

        if not self._barrier_recovery:
            raise RuntimeError(f"{base} — RLINF_TILED_BARRIER_RECOVERY=0, previous-frame substitution disabled")

        if self._barrier_stall_streak + 1 > self._barrier_max_stalls:
            raise RuntimeError(
                f"{base} — {self._barrier_stall_streak} consecutive previous-frame substitutions "
                f"still have not recovered (cap {self._barrier_max_stalls}, tune "
                f"RLINF_TILED_BARRIER_MAX_STALLS) — treating the render node as genuinely dead"
            )

        self._barrier_stall_streak += 1
        self._barrier_stall_total += 1
        print(
            f"[{type(self).__name__}] BARRIER STALL "
            f"{self._barrier_stall_streak}/{self._barrier_max_stalls}: "
            f"rendered_tick={rendered_tick} target={target} "
            f"waited={waited_s:.1f}s — timed out, reusing previous frame (dropped=True); "
            f"can send tick>{target} next step"
            f" (total={self._barrier_stall_total})",
            flush=True,
        )

    def render(self) -> RenderResult:
        self._ensure_started()
        self._ensure_shm_attached()
        if self._bridge is None:
            raise RuntimeError("render bridge not running")

        t0 = time.monotonic()

        # Frame-sync barrier. ``got[1]`` is the image's timestamp, which the
        # upstream render node sets from the /tf_render stamp we publish — i.e.
        # the global render TICK (for tick < 2^31 the encoding is sec=tick,
        # nanosec=0, so the float equals the tick exactly). In "strict" lockstep
        # we block until EVERY camera has an image rendered at tick >= this
        # step's tick (never returning a stale frame; the node dying or the tick
        # getting stuck raises). "latest" keeps the legacy freshest-frame path.
        sync = getattr(self, "_render_sync", "strict")
        target = getattr(self, "_current_tick", 0)
        prev_ts = getattr(self, "_last_frame_sim_time", -1.0)
        deadline = t0 + self._render_timeout_s
        cap = t0 + max(30.0, self._render_timeout_s * 10.0)

        def _ready(got) -> bool:
            if got is None:
                return False
            if sync == "latest":
                return got[1] > prev_ts
            return got[1] >= target

        all_fresh = False
        if self._image_topics:
            while True:
                if all(_ready(self._bridge.latest_frame(tp)) for tp in self._image_topics):
                    all_fresh = True
                    self._recovery_frame_valid = True
                    self._barrier_stall_streak = 0  # only CONSECUTIVE stalls
                    break
                if sync == "latest" and time.monotonic() > deadline:
                    break
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(f"{type(self).__name__}: render node died while waiting " f"for tick {target}")
                if sync != "latest" and time.monotonic() > cap:
                    # Timeout, not a fatal error — reuse the previous frame.
                    # The copy loop below reads latest_frame(tp), i.e. whatever
                    # is already there, so breaking here IS the substitution.
                    _stale = [
                        None if (g := self._bridge.latest_frame(tp)) is None else g[1] for tp in self._image_topics
                    ]
                    self._barrier_stall_recover(
                        rendered_tick=_stale,
                        target=target,
                        waited_s=time.monotonic() - t0,
                        cold_first=self._frame_index <= 0,
                        knob_hint=(
                            " — this ROS-image path's cap is "
                            "max(30s, render_timeout_s*10); "
                            "RLINF_TILED_BARRIER_CAP_S only governs the tiled path"
                        ),
                    )
                    break
                time.sleep(0.001)
        dropped = not all_fresh

        # Copy frames into Frame SHM rgb_buf if attached.
        # Topic-name convention (see scene_assembly.env_topic_name):
        #   env 0:    /<ns>/<basename>            (no env_ prefix)
        #   env i>0:  /<ns>/env_<i>_<basename>
        # We route each frame to ``rgb_buf[env_id, cam_id]`` slot.
        # ``cam_id`` is the camera index within the env (0..num_cams_per_env-1),
        # which we derive from the position of this topic among topics
        # sharing the same env_id.
        max_sim_time = prev_ts
        min_tick = None  # slowest camera's tick -> frame_tick (the synced tick)
        if self._frame_shm is not None:
            try:
                from ._multi_env_helpers import parse_env_from_topic

                rgb_buf = self._frame_shm.field("rgb_buf")  # (N, C, H, W, 3)
                num_envs, num_cams = rgb_buf.shape[0], rgb_buf.shape[1]
                h, w = rgb_buf.shape[2], rgb_buf.shape[3]
                # Build a per-env counter so multiple cameras on the same
                # env get distinct cam_id slots.
                per_env_cam_idx: dict[int, int] = {}
                for topic in self._image_topics:
                    got = self._bridge.latest_frame(topic)
                    if got is None:
                        continue
                    img, t = got
                    env_id = parse_env_from_topic(topic)
                    if env_id >= num_envs:
                        # SHM was sized for fewer envs; skip silently.
                        continue
                    cam_id = per_env_cam_idx.get(env_id, 0)
                    per_env_cam_idx[env_id] = cam_id + 1
                    if cam_id >= num_cams:
                        continue
                    if img.shape[0] == h and img.shape[1] == w:
                        rgb_buf[env_id, cam_id] = img
                    else:
                        dropped = True
                    if t > max_sim_time:
                        max_sim_time = t
                    min_tick = t if min_tick is None else min(min_tick, t)
                # Update frame metadata fields.
                self._frame_shm.field("frame_index").fill(self._frame_index + 1)
                self._frame_shm.field("frame_sim_time")[0] = max_sim_time
                if min_tick is not None:
                    self._frame_shm.field("frame_tick").fill(int(round(min_tick)))
                self._frame_shm.field("frame_ready").fill(True)
            except KeyError:
                # Layout missing some field — shouldn't happen in
                # practice but don't crash the loop.
                pass

        self._frame_index += 1
        self._last_frame_sim_time = max_sim_time
        self._last_render_ms = (time.monotonic() - t0) * 1000.0

        return RenderResult(
            sim_time=np.array([max_sim_time], dtype=np.float64),
            frame_index=self._frame_index,
            latency_ms=self._last_render_ms,
            dropped=dropped,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def supports_batching(self) -> bool:
        return self.SUPPORTS_BATCHING

    @property
    def output_channels(self) -> list[str]:
        return list(self.OUTPUT_CHANNELS)

    @property
    def backend_name(self) -> str:
        if not self.BACKEND_NAME:
            raise RuntimeError(f"{type(self).__name__}: BACKEND_NAME not set")
        return self.BACKEND_NAME

    # ------------------------------------------------------------------

    def health_check(self) -> RenderHealthStatus:
        alive = self._started and self._proc is not None and self._proc.poll() is None and self._bridge is not None
        return RenderHealthStatus(
            alive=alive,
            pid=self._pid,
            last_render_ms=self._last_render_ms,
            cuda_oom=False,
            error_msg="" if alive else "subprocess or bridge not running",
        )

    def get_metrics(self) -> dict[str, float]:
        return {
            "last_render_ms": float(self._last_render_ms),
            "frame_index": float(self._frame_index),
        }


__all__ = ["_UpstreamRenderBridge"]
