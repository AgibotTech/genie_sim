# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Render batched observation cameras through an Isaac Sim subprocess.

The node writes de-tiled RGB into TiledFrameShm; this bridge copies frames into
the handler's observation buffer. Body and camera poses use the ROS bridge.
Reload and per-reset randomization use separate generation-tagged command/ack
files. Material candidates are scanned once per loaded scene."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from . import register_bridge
from ._hotswap_protocol import (
    clear_stale_control_files,
    poll_ack,
    write_cmd,
)
from ._render_randomization import RenderRandomizationMixin, RenderReloadError  # noqa: F401
from ._upstream_base import _UpstreamRenderBridge
from .base import RenderResult
from ..tiled_shm import TiledFrameShm
from ...randomize import RandomizationPlan


@register_bridge("isaac_tiled")
class IsaacTiledBackend(RenderRandomizationMixin, _UpstreamRenderBridge):
    UPSTREAM_PACKAGE = "genie_sim_render"  # documentation only; we launch
    UPSTREAM_EXEC = "isaacsim_tiled_render.py"  # the script directly (below)
    BACKEND_NAME = "isaac_tiled"
    OUTPUT_CHANNELS = ["rgb"]
    SUPPORTS_BATCHING = True  # one Kit process tiled-renders all envs in 1 pass
    # The node's stage is GridCloner-organized at ANY env count — its prims
    # only exist under /World/envs/env_<i>/, so update_poses must publish the
    # multi-env body frames even at N==1 (see _upstream_base.update_poses).
    _gridcloner_single_env_frames = True
    # Publishing a new tick triggers a render; skip updates for uncollected frames.
    _skip_poses_when_uncollected = True

    def __init__(
        self,
        *args: Any,
        script_path: str = "",
        python_exec: str = "",
        tiled_shm_name: str = "",
        # Isaac Sim startup + scene load is slow; the tiled-SHM only appears once
        # the node has loaded scene_multi + built the TiledCameraSensor. A COLD
        # run (empty RTX shader cache, e.g. first launch on a fresh cloud pod)
        # can take 3-5 min just to compile shaders, so the ceiling is env-tunable
        # via RLINF_TILED_READY_TIMEOUT (seconds); warm runs stay well under 180.
        ready_timeout_s: float = float(os.environ.get("RLINF_TILED_READY_TIMEOUT", "180")),
        # Frame-sync policy: "strict" lockstep-barrier (wait until the frame in
        # tiled-SHM was rendered at/after the tick we're stepping) or "latest"
        # (async, never wait; caller records lag).
        render_sync: str = "strict",
        # Normalized per-reset render domains; background selection occurs at stage build.
        render_randomization: Optional[dict] = None,
        # Timeout for acknowledgement of per-reset scene changes.
        randomize_ack_timeout_s: float = 30.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("ready_timeout_s", ready_timeout_s)
        super().__init__(*args, **kwargs)
        self._render_sync = str(render_sync)
        if self._render_sync not in {"strict", "latest"}:
            raise ValueError(
                f"{type(self).__name__}: render_sync must be 'strict' or " f"'latest', got {render_sync!r}"
            )
        self._script_path = script_path or self._default_tiled_script()
        self._python_exec = python_exec or os.environ.get("RLINF_ISAAC_PYTHON", "") or "python3"
        # Unique per bridge instance so concurrent runs never collide on a
        # global SharedMemory name. Deterministic within a run (no rng/clock).
        self._tiled_shm_name = tiled_shm_name or f"rlinf_tiled_{os.getpid()}"

        self._tiled_shm: Optional[TiledFrameShm] = None
        self._last_tiled_frame_id: int = 0
        self._last_sim_time_val: float = 0.0
        self._copy_diag_done: bool = False
        # render_barrier()'s two-phase poll: sleep coarsely for this long
        # (a frame cannot plausibly be ready sooner — the RTX pass alone is
        # ~60ms), then switch to a bare yield so the frame is picked up as
        # soon as it lands. 0 disables the coarse phase entirely.
        self._barrier_spin_after_s = float(os.environ.get("RLINF_TILED_BARRIER_SPIN_AFTER_S", "0.04"))
        # Render hot-swap (reload_stage, below): the cmd/ack control
        # directory, cached ONCE in _build_argv() at subprocess launch time,
        # and this bridge's own monotonic reload generation counter.
        self._reload_control_dir: Optional[str] = None
        self._reload_generation: int = 0
        # Render-side generalization command channel — independent from
        # the reload counter pair above, its own generation counter (see
        # the _hotswap_protocol module docstring, "the two channels never
        # interfere with each other").
        self._render_randomization: dict = dict(render_randomization or {})
        self._randomize_ack_timeout_s: float = float(randomize_ack_timeout_s)
        self._randomize_generation: int = 0
        # Material candidate scan result, lazily loaded + cached (see
        # _scan_material_candidates_once). ``None`` = not scanned yet; once
        # scanned, even an empty-dict result (the asset has no variants) is
        # never ``None`` again, so it is never rescanned or re-logged.
        self._material_candidates: Optional[dict] = None
        # The plan produced by the most recent on_reset() call, already
        # patched in place with the ack's ``applied`` readback field —
        # ``on_reset`` itself only "sends the command + validates"; this is
        # the only place where the claim "these materials were used" can
        # prove itself (see on_reset's docstring, "honesty"). Private
        # (underscore-prefixed), read the same way ``_render_prof`` is read
        # by sim_handler via getattr — not a newly invented interface.
        self._last_render_plans: dict[int, RandomizationPlan] = {}
        # Defense-in-depth for the frame-sync barrier (render(), below): once a
        # reload_stage() succeeds, this is the SHM generation render() must see
        # on ``self._tiled_shm.generation`` before it will trust ANY frame.
        # ``None`` before the first reload (no generation constraint yet — the
        # boot generation, 0, is always trustworthy). See reload_stage() and
        # render() for the full rationale.
        self._expected_generation: Optional[int] = None
        # GridCloner owns the per-env layout in the node → publish RAW physics
        # poses; the node adds each env's grid origin (see _upstream_base).
        self._bake_env_offset = False
        # Publish poses on the FLAT channel (Float64MultiArray + latched path
        # table) instead of TFMessage — the tiled node consumes it natively and
        # saves ~140ms/frame of rclpy TransformStamped deserialization at 8 envs.
        self._flat_pose_channel = True
        # phase profile printed by sim_handler's [handler-prof] line
        self._render_prof = {"pub": 0.0, "barrier": 0.0, "copy": 0.0, "n": 0}

    # ------------------------------------------------------------------
    # Launch: python3 isaacsim_tiled_render.py --manifest ... (argparse CLI)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_tiled_script() -> str:
        """Resolve the tiled render node in the geniesim_ros source tree.

        Overridable via ``RLINF_TILED_RENDER_SCRIPT``. Walks up from this file
        (``source/rlinf_geniesim/render/bridges/``) to find ``source/`` and the
        sibling ``geniesim_ros`` package."""
        env = os.environ.get("RLINF_TILED_RENDER_SCRIPT")
        if env:
            return env
        rel = (
            Path("geniesim_ros")
            / "src"
            / "ros_ws"
            / "src"
            / "genie_sim_render"
            / "scripts"
            / "isaacsim_tiled_render.py"
        )
        here = Path(__file__).resolve()
        for parent in here.parents:
            cand = parent / rel
            if cand.is_file():
                return str(cand)
        # Best-effort guess (source/ is here.parents[2]); _verify_environment
        # will raise a clear FileNotFoundError if it's wrong.
        return str(here.parents[2] / rel)

    def _resolve_image_topics(self) -> list[str]:
        # Tiled frames arrive via TiledFrameShm, not ROS image topics — we only
        # need the /tf_render publisher the base's RosRenderBridge also brings up.
        return []

    def _build_argv(self) -> list[str]:
        if not self._stage_manifest:
            raise ValueError(f"{type(self).__name__}: stage_manifest is required")
        resolved_manifest = Path(self._stage_manifest).resolve()
        # Cache the hot-swap control directory here, from the manifest path
        # placed in the subprocess argv, rather than recomputing it from
        # ``self._stage_manifest`` later: that can be rebound after startup,
        # while the render node polls cmd/ack in its original --manifest
        # directory for its whole life.
        #
        # Known limit: the directory is the manifest's parent rather than
        # per-pid, so two servers whose manifests share a parent would
        # overwrite each other's cmd/ack. Give them separate directories.
        self._reload_control_dir = str(resolved_manifest.parent)
        # The control directory is not unique to this (bridge, node) pair, so
        # a previous process may have left cmd/ack files behind while both
        # generation counters start at 0. Stale files can make the new node
        # execute a reload at startup, make the first poll succeed against the
        # wrong scene, or trigger a reload on a clean start. Remove them
        # before spawning; the bridge always starts before the node.
        clear_stale_control_files(self._reload_control_dir)
        return [
            self._python_exec,
            self._script_path,
            "--manifest",
            str(resolved_manifest),
            "--shm-name",
            self._tiled_shm_name,
            "--render-hz",
            f"{self._render_hz}",
            # Multi-env: the node organizes it via GridCloner (this backend's job).
            "--num-envs",
            f"{self._num_envs}",
            "--env-spacing",
            f"{self._env_spacing}",
            # Headless by default; set RLINF_TILED_HEADLESS=0 to open the Isaac
            # Sim GUI viewport (needs an X display in the container) — useful for
            # eyeballing the tiled render / frame-sync live.
            "--headless",
            os.environ.get("RLINF_TILED_HEADLESS", "1"),
        ]

    def _verify_environment(self) -> None:
        super()._verify_environment()  # ros2 on PATH + manifest exists
        if not Path(self._script_path).is_file():
            raise FileNotFoundError(
                f"tiled render script not found: {self._script_path} " f"(set RLINF_TILED_RENDER_SCRIPT to override)"
            )
        if shutil.which(self._python_exec) is None and not Path(self._python_exec).is_file():
            raise RuntimeError(
                f"python interpreter '{self._python_exec}' not found "
                f"(set RLINF_ISAAC_PYTHON to the isaacsim-capable python)"
            )

    # ------------------------------------------------------------------
    # Lifecycle: reuse base start() (subprocess + tf bridge + anchor pump),
    # then attach the node-created tiled-SHM.
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        super().start()  # spawn node, RosRenderBridge (tf-only), anchor pump
        try:
            self._attach_tiled_shm()
        except Exception:
            # Attach failed / timed out — tear the subprocess + bridge down so
            # we don't leak a Kit process or a half-open ROS node.
            self.close()
            raise

    def _obs_tile_dims(self) -> tuple[int, int, int]:
        """``(n_tiles, h, w)`` the node will write: ``num_envs × num_obs_cameras``
        tiles — the GridCloner node renders EVERY obs camera the task uses (head +
        wrists, from the benchmark camera_list) per env, tiled in
        ``[env_i][cam_j]`` order. Cameras share a common WIDTH but may differ in
        HEIGHT (head 640x400, wrist 640x528); the SHM is sized at the MAX height
        and each camera's shorter frames are zero-padded, so ``h`` here is the
        MAX obs-camera height."""
        h, w, n_cams = 400, 640, 1
        try:
            import json

            with open(self._stage_manifest, "r") as f:
                man = json.load(f)
            obs = [c for c in (man.get("cameras") or []) if not c.get("is_free_cam")]
            if obs:
                n_cams = len(obs)
                h = max(int(c.get("height", 0)) for c in obs) or h
                w = int(obs[0].get("width", w))
        except Exception:  # noqa: BLE001
            if self._cameras:
                n_cams = len(self._cameras)
                if "image_hw" in self._cameras[0]:
                    h = max(int(c["image_hw"][0]) for c in self._cameras if "image_hw" in c) or h
                    w = int(self._cameras[0]["image_hw"][1])
        return self._num_envs * n_cams, h, w

    def _attach_tiled_shm(self) -> None:
        n_tiles, h, w = self._obs_tile_dims()
        deadline = time.monotonic() + self._ready_timeout_s
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            # Bail immediately if the render subprocess died — no point waiting
            # the full timeout for a segment that will never appear.
            if self._proc is not None and self._proc.poll() is not None:
                tail = self._read_log_tail()
                raise RuntimeError(
                    f"{type(self).__name__}: tiled render node exited with code "
                    f"{self._proc.returncode} before creating the tiled-SHM.\n"
                    f"Log tail:\n{tail}"
                )
            try:
                self._tiled_shm = TiledFrameShm(self._tiled_shm_name, num_envs=n_tiles, height=h, width=w, create=False)
                print(
                    f"[{type(self).__name__}] attached tiled-SHM " f"'{self._tiled_shm_name}' ({n_tiles},{h},{w},3)",
                    flush=True,
                )
                return
            except FileNotFoundError as e:
                # Segment not created yet (node still loading Isaac scene).
                last_err = e
            except ValueError as e:
                # Segment exists but header not written yet (magic still 0), or a
                # genuine dim mismatch — retry; a real mismatch surfaces at the
                # deadline below.
                last_err = e
            time.sleep(0.2)
        raise RuntimeError(
            f"{type(self).__name__}: tiled-SHM '{self._tiled_shm_name}' "
            f"({n_tiles},{h},{w},3) did not appear within "
            f"{self._ready_timeout_s:.0f}s. Last error: {last_err!r}"
        )

    def _read_log_tail(self, n: int = 2000) -> str:
        log_path = getattr(self, "_log_path", "")
        if not log_path:
            return ""
        try:
            with open(log_path, "r") as f:
                return f.read()[-n:]
        except Exception:  # noqa: BLE001
            return ""

    def close(self) -> None:
        ts = self._tiled_shm
        self._tiled_shm = None
        # Kill the render subprocess FIRST (killpg SIGINT, escalating to
        # SIGTERM/SIGKILL, each step WAITED on — see _UpstreamRenderBridge.
        # close()) so unlinking the tiled-SHM segment below can never race a
        # still-exiting node: by the time we get past this call the node is
        # provably dead, one way or another.
        super().close()
        if ts is not None:
            # A node exiting on SIGINT unlinks its shared-memory segment and
            # this is a no-op. One killed with SIGTERM/SIGKILL does not, and
            # since the segment name is keyed on pid and constant across
            # reload generations, the leftover blocks the next generation from
            # creating it. Unlink unconditionally once the node is gone.
            try:
                ts.close(unlink=True)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Render hot-swap
    # ------------------------------------------------------------------

    def reload_stage(self, stage_manifest: str, timeout_s: float = 60.0) -> None:
        """Reload a compatible stage without restarting the render subprocess.

        The caller checks backend and observation compatibility. Validate acknowledgement
        dimensions against this bridge and raise RenderReloadError on process failure,
        timeout, rejection or mismatch; the caller decides whether to rebuild."""
        if not self.render_proc_alive():
            raise RenderReloadError(
                f"{type(self).__name__}.reload_stage: render subprocess is " f"not alive — cannot hot-swap"
            )
        control_dir = self._reload_control_dir
        if not control_dir:
            raise RenderReloadError(
                f"{type(self).__name__}.reload_stage: no cmd/ack control "
                f"directory cached — start() must launch the render node "
                f"before reload_stage() can be called"
            )

        # This bridge's OWN current expected dims, computed from the OLD
        # (still-active) manifest BEFORE any rebind below — the SAME
        # (n_tiles, h, w) math used to size/attach the tiled-SHM at start().
        n_tiles, h, w = self._obs_tile_dims()
        n_cams = n_tiles // self._num_envs if self._num_envs else n_tiles
        expected_dims = [self._num_envs, n_cams, h, w]

        self._reload_generation += 1
        generation = self._reload_generation
        resolved_manifest = str(Path(stage_manifest).resolve())
        write_cmd(control_dir, generation, resolved_manifest)
        try:
            ack = poll_ack(control_dir, generation, timeout_s, proc_alive=self.render_proc_alive)
        except TimeoutError as e:
            raise RenderReloadError(
                f"{type(self).__name__}.reload_stage: timed out after "
                f"{timeout_s}s waiting for ack (generation={generation}): "
                f"{e}"
            ) from e
        except RuntimeError as e:
            raise RenderReloadError(
                f"{type(self).__name__}.reload_stage: render subprocess "
                f"died while waiting for ack (generation={generation}): "
                f"{e}"
            ) from e

        if not ack.get("ok"):
            raise RenderReloadError(
                f"{type(self).__name__}.reload_stage: render node reload "
                f"failed (generation={generation}): {ack.get('error')!r}"
            )

        ack_dims = ack.get("dims")
        if list(ack_dims or []) != expected_dims:
            raise RenderReloadError(
                f"{type(self).__name__}.reload_stage: ack dims {ack_dims} "
                f"!= expected {expected_dims} (generation={generation}) — "
                f"the applicability gate should have caught this upstream"
            )

        # Success: rebind to the new manifest.
        #
        # ``_current_tick`` is bridge-side bookkeeping that the next
        # update_poses relocks; zeroing it is only a defensive default.
        #
        # ``_last_tiled_frame_id`` must NOT be cleared. It is the barrier's
        # high-water mark of consumed frames, and frame_id is monotonic across
        # generations precisely so an old frame cannot satisfy it. The node
        # acks a reload before rendering the new scene's first frame, so
        # clearing the mark would hand the new task the old task's picture.
        #
        # Record the generation this reload reached, so the barrier can also
        # reject frames from a stale one.
        self._stage_manifest = resolved_manifest
        self._current_tick = 0
        self._recovery_frame_valid = False
        self._expected_generation = generation
        # The node built a fresh driver for this generation and can only
        # learn the flat path table from the latched topic, so force a
        # republish on the next frame. Success path only: after a failed ack
        # this generation still uses the old table.
        _invalidate = getattr(self._bridge, "invalidate_flat_path_table", None)
        if _invalidate is not None:
            _invalidate()

    # ------------------------------------------------------------------
    # Render-side generalization (per-reset, whole grid shares one draw for
    # lights, independently per env for material and camera pose). Uses the
    # DEDICATED randomize_cmd.json/randomize_ack.json pair
    # (_hotswap_protocol's randomize_* functions) — NEVER reload_cmd.json:
    # that channel's read_cmd_if_new only ever returns {generation,
    # manifest} (any other key silently dropped), and reusing it would make
    # the node run a full stage rebuild just to change two UsdLux attributes.
    # ------------------------------------------------------------------

    def on_reset(
        self, env_ids: Sequence[int], seeds: Optional[Sequence[int]] = None, material_bindings: Optional[dict] = None
    ) -> None:
        """Apply seeded light, material and camera-pose changes before returning.

        Disabled dimensions and unseeded environments skip random draws; authored instance
        material bindings still apply. Lights share one draw and change only on a full-grid
        reset. Material and camera pose are sampled per environment. Background selection
        belongs to stage construction, not this reset channel.

        Wait for the generation's acknowledgement. Missing material/camera readback or
        zero applied changes for a requested dimension raise RenderReloadError; partial
        application warns. Light counts come from the node. Store plans and readback
        even on failure so callers can distinguish requested and applied changes."""
        self._recovery_frame_valid = False
        pending = self._send_render_randomization(env_ids, seeds, material_bindings)
        if pending is not None:
            self._await_render_randomization(pending)

    def update_poses(self, body_poses: np.ndarray, sim_time: np.ndarray, *args: Any, **kwargs: Any) -> None:
        # The tiled-SHM carries no timestamp, so stamp render() with the sim_time
        # we're driving poses at. Then reuse the base's /tf_render publishing
        # (multi-env body frames + camera anchors) verbatim.
        #
        # Keep this the ONLY update_poses in the class. A second one earlier in
        # the class body used to shadow it, silently, so its ``pub`` timer never
        # ran and every [handler-prof] line printed pub=0.0 — not "free", just
        # unmeasured.
        import time as _t  # noqa: PLC0415

        _p0 = _t.monotonic()
        try:
            self._last_sim_time_val = float(np.asarray(sim_time).reshape(-1)[0])
        except Exception:  # noqa: BLE001
            pass
        super().update_poses(body_poses, sim_time, *args, **kwargs)
        self._render_prof["pub"] += _t.monotonic() - _p0

    @staticmethod
    def _barrier_satisfied(rendered_tick: int, target_tick: int, mode: str) -> bool:
        """Strict lockstep: the frame in tiled-SHM must have been rendered at
        (or after) the tick we're stepping. latest: never wait (async, caller
        records lag)."""
        if mode == "latest":
            return True
        return int(rendered_tick) >= int(target_tick)

    def _barrier_cap_s(self, cold_first: bool) -> float:
        """How long ``render()``'s strict barrier waits before declaring the
        tick path broken.

        This is NOT governed by ``RLINF_TILED_READY_TIMEOUT``, which covers
        waiting for the tiled SHM to appear at startup — a different phase
        entirely. The barrier has its own knob, named in ``render()``'s error.

        ``cold_first`` uses a wider knob: the first frame in a process waits
        for RTX shader compilation, which takes minutes on a cold cache.
        """
        if cold_first:
            return float(os.environ.get("RLINF_TILED_FIRST_FRAME_CAP_S", "300"))
        return float(os.environ.get("RLINF_TILED_BARRIER_CAP_S", str(max(30.0, self._render_timeout_s * 10))))

    @staticmethod
    def _fill_rgb_from_tiled(rgb_buf: np.ndarray, tiled_rgb: np.ndarray, cam_id: int = 0) -> bool:
        """Copy each de-tiled per-env head frame into ``rgb_buf[env, cam_id]``.

        ``rgb_buf``  : ``(N, C, H, W, 3)`` uint8 — the rlinf Frame SHM view.
        ``tiled_rgb``: ``(n_tiles, h, w, 3)`` uint8 — tile ``i`` is env ``i``'s
                       head observation camera.

        Returns ``dropped=True`` if any env's slot could not be filled (missing
        tile, resolution mismatch, or ``cam_id`` out of range). Never writes a
        wrong-shaped frame into ``rgb_buf``.
        """
        n_envs = int(rgb_buf.shape[0])
        n_cams = int(rgb_buf.shape[1])
        H, W = int(rgb_buf.shape[2]), int(rgb_buf.shape[3])
        n_tiles = int(tiled_rgb.shape[0])
        # Tiles are ordered [env_i][cam_j] -> flat i*n_cams + j (matches
        # build_gridcloner_stage's cam_paths order). De-tile ALL cameras into
        # rgb_buf[env, cam]. cam_id is ignored (kept for signature compat).
        # FAST PATH: exact layout match -> ONE bulk reshape copy (the per-tile
        # loop below does strided SHM reads/writes and cost ~60ms/frame at
        # 8 envs; the bulk copy is ~5ms for the same 18MB).
        if n_tiles == n_envs * n_cams and int(tiled_rgb.shape[1]) == H and int(tiled_rgb.shape[2]) == W:
            src = tiled_rgb if tiled_rgb.shape[-1] == 3 else tiled_rgb[..., :3]
            rgb_buf[:] = src.reshape(n_envs, n_cams, H, W, 3)
            return False
        dropped = n_tiles != n_envs * n_cams
        for i in range(n_envs):
            for j in range(n_cams):
                t = i * n_cams + j
                if t >= n_tiles:
                    dropped = True
                    continue
                tile = tiled_rgb[t]
                if int(tile.shape[0]) == H and int(tile.shape[1]) == W:
                    rgb_buf[i, j] = tile[..., :3]
                else:
                    dropped = True
        return dropped

    def render_barrier(self) -> bool:
        """Block until the tiled-SHM holds a fresh frame (barrier phase only).

        Returns True if a valid frame was received (satisfied), False if
        dropped.  After this returns, call ``render_copy()`` to copy the
        frame into the Frame SHM and finalize metrics.  The split allows
        the caller to perform CPU work between update_poses and the copy
        while the GPU is still rendering — increasing the overlap window.
        """
        self._ensure_started()
        self._ensure_shm_attached()

        self._barrier_t0 = time.monotonic()
        self._barrier_dropped = False

        if self._tiled_shm is None:
            self._barrier_dropped = True
            return False

        prev = self._last_tiled_frame_id
        target = getattr(self, "_current_tick", 0)
        expected_gen = self._expected_generation
        start = self._barrier_t0
        deadline = start + self._render_timeout_s
        _cold_first = self._last_tiled_frame_id == 0 and self._expected_generation is None
        barrier_cap = start + self._barrier_cap_s(_cold_first)
        satisfied = False
        # Poll granularity: the frame we're waiting on takes ~60-70ms (the RTX
        # pass dominates), so a flat 1ms sleep both (a) wastes ~65 pointless
        # SHM reads early on and (b) costs ~0.5ms of average detection lag at
        # the end — the frame is ready but we're still asleep. Coarse while the
        # frame can't plausibly be done yet, then fine once it's due.
        _spin_after = start + self._barrier_spin_after_s
        while True:
            rt = self._tiled_shm.rendered_tick
            fid = self._tiled_shm.frame_id
            gen = self._tiled_shm.generation
            gen_ok = expected_gen is None or gen == expected_gen
            if fid > prev and gen_ok and self._barrier_satisfied(rt, target, self._render_sync):
                self._last_tiled_frame_id = fid
                satisfied = True
                self._barrier_stall_streak = 0
                self._recovery_frame_valid = True
                break
            if self._render_sync == "latest" and time.monotonic() > deadline:
                break
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"{type(self).__name__}: render node died while waiting for "
                    f"tick {target} (rendered_tick={rt}, frame_id={fid})"
                )
            if self._render_sync != "latest" and time.monotonic() > barrier_cap:
                self._barrier_stall_recover(
                    rendered_tick=rt,
                    target=target,
                    waited_s=time.monotonic() - start,
                    cold_first=_cold_first or not gen_ok,
                )
                break
            if time.monotonic() < _spin_after:
                time.sleep(0.002)
            else:
                time.sleep(0.0002)  # frame is due — poll finely, but still
                # sleep: a bare yield/busy-spin here would
                # peg a core for ~30ms of every step to
                # save a fraction of a ms, and the render
                # node shares this machine.
        self._barrier_dropped = not satisfied
        self._render_prof["barrier"] += time.monotonic() - self._barrier_t0
        return satisfied

    def render_copy(self) -> RenderResult:
        """Copy the rendered frame from tiled-SHM into Frame SHM (copy phase).

        Must be called after ``render_barrier()``.  Together
        ``render_barrier() + render_copy()`` is semantically identical to
        the original ``render()`` but allows CPU work to be scheduled in
        between.
        """
        _c0 = time.monotonic()
        dropped = self._barrier_dropped

        if self._tiled_shm is not None and self._frame_shm is not None:
            try:
                rgb_buf = self._frame_shm.field("rgb_buf")
                copy_dropped = self._fill_rgb_from_tiled(rgb_buf, self._tiled_shm.rgb, cam_id=0)
                dropped = dropped or copy_dropped
                self._frame_shm.field("frame_index").fill(self._frame_index + 1)
                self._frame_shm.field("frame_sim_time")[0] = self._last_sim_time_val
                self._frame_shm.field("frame_ready").fill(True)
                self._frame_shm.field("frame_tick").fill(int(self._tiled_shm.rendered_tick))
                if not self._copy_diag_done:
                    _tr = np.asarray(self._tiled_shm.rgb)
                    _stds = [int(_tr[i].std()) for i in range(_tr.shape[0])]
                    print(
                        f"[{type(self).__name__}] first tiled->rgb_buf copy: "
                        f"tiled={tuple(self._tiled_shm.rgb.shape)} "
                        f"rgb_buf={tuple(rgb_buf.shape)} dropped={dropped} "
                        f"per_tile_std={_stds}",
                        flush=True,
                    )
                    self._copy_diag_done = True
            except KeyError:
                pass

        self._render_prof["copy"] += time.monotonic() - _c0
        self._render_prof["n"] += 1
        self._frame_index += 1
        self._last_render_ms = (time.monotonic() - self._barrier_t0) * 1000.0
        return RenderResult(
            sim_time=np.array([self._last_sim_time_val], dtype=np.float64),
            frame_index=self._frame_index,
            latency_ms=self._last_render_ms,
            dropped=dropped,
        )

    def render(self) -> RenderResult:
        self._ensure_started()
        self._ensure_shm_attached()

        t0 = time.monotonic()
        dropped = False

        if self._tiled_shm is None:
            self._frame_index += 1
            self._last_render_ms = (time.monotonic() - t0) * 1000.0
            return RenderResult(
                sim_time=np.array([self._last_sim_time_val], dtype=np.float64),
                frame_index=self._frame_index,
                latency_ms=self._last_render_ms,
                dropped=True,
            )

        # Tick BARRIER (replaces the old "wait for a fresh frame_id" loop). In
        # strict lockstep we block until the tiled-SHM holds a NEW frame that was
        # rendered at/after the tick we're stepping (self._current_tick, latched
        # by update_poses); in latest mode we take the freshest frame and bail at
        # the timeout. The barrier is the whole point of frame-sync: a step's
        # rgb must match the poses it drove.
        prev = self._last_tiled_frame_id
        target = getattr(self, "_current_tick", 0)
        # After a reload, ``_expected_generation`` pins the SHM generation a
        # frame must carry, backing up the frame_id high-water mark. ``None``
        # means no reload has happened and imposes no constraint. A generation
        # mismatch counts as "the new frame has not arrived", not an error.
        expected_gen = self._expected_generation
        start = t0
        deadline = start + self._render_timeout_s
        # A generous absolute cap, so a node that is alive but stuck cannot
        # hang strict mode forever.
        #
        # The first frame gets a wider allowance: RTX sensor initialisation
        # compiles shaders per GPU model, which takes minutes on a cold cache
        # before the node reaches its main loop. A reload keeps the ordinary
        # cap, since the node is long past that point by then.
        _cold_first = self._last_tiled_frame_id == 0 and self._expected_generation is None
        barrier_cap = start + self._barrier_cap_s(_cold_first)
        satisfied = False
        while True:
            rt = self._tiled_shm.rendered_tick
            fid = self._tiled_shm.frame_id
            gen = self._tiled_shm.generation
            gen_ok = expected_gen is None or gen == expected_gen
            if fid > prev and gen_ok and self._barrier_satisfied(rt, target, self._render_sync):
                self._last_tiled_frame_id = fid
                satisfied = True
                self._barrier_stall_streak = 0
                self._recovery_frame_valid = True
                break
            # latest: async — never block past the timeout (frame is then dropped).
            if self._render_sync == "latest" and time.monotonic() > deadline:
                break
            # Liveness: a dead render node ends the wait at once (both modes).
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"{type(self).__name__}: render node died while waiting for "
                    f"tick {target} (rendered_tick={rt}, frame_id={fid})"
                )
            # Strict safety valve: bounded stale-frame recovery when a valid
            # current-episode frame exists; otherwise fail.
            if self._render_sync != "latest" and time.monotonic() > barrier_cap:
                self._barrier_stall_recover(
                    rendered_tick=rt,
                    target=target,
                    waited_s=time.monotonic() - start,
                    cold_first=_cold_first or not gen_ok,
                )
                break
            time.sleep(0.001)
        dropped = not satisfied
        self._render_prof["barrier"] += time.monotonic() - t0
        _c0 = time.monotonic()

        if self._frame_shm is not None:
            try:
                rgb_buf = self._frame_shm.field("rgb_buf")  # (N, C, H, W, 3)
                copy_dropped = self._fill_rgb_from_tiled(rgb_buf, self._tiled_shm.rgb, cam_id=0)
                dropped = dropped or copy_dropped
                self._frame_shm.field("frame_index").fill(self._frame_index + 1)
                self._frame_shm.field("frame_sim_time")[0] = self._last_sim_time_val
                self._frame_shm.field("frame_ready").fill(True)
                # Stamp the frame with the tick it was rendered at so consumers
                # can match rgb -> the step whose poses produced it.
                self._frame_shm.field("frame_tick").fill(int(self._tiled_shm.rendered_tick))
                if not self._copy_diag_done:
                    _tr = np.asarray(self._tiled_shm.rgb)
                    _stds = [int(_tr[i].std()) for i in range(_tr.shape[0])]
                    print(
                        f"[{type(self).__name__}] first tiled->rgb_buf copy: "
                        f"tiled={tuple(self._tiled_shm.rgb.shape)} "
                        f"rgb_buf={tuple(rgb_buf.shape)} dropped={dropped} "
                        f"per_tile_std={_stds}",
                        flush=True,
                    )
                    self._copy_diag_done = True
            except KeyError:
                pass

        self._render_prof["copy"] += time.monotonic() - _c0
        self._render_prof["n"] += 1
        self._frame_index += 1
        self._last_render_ms = (time.monotonic() - t0) * 1000.0
        return RenderResult(
            sim_time=np.array([self._last_sim_time_val], dtype=np.float64),
            frame_index=self._frame_index,
            latency_ms=self._last_render_ms,
            dropped=dropped,
        )


__all__ = ["IsaacTiledBackend", "RenderReloadError"]
