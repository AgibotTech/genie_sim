#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Isaac Sim TILED render node (SP1 — Phase 1).

Renders ALL per-env observation (head) cameras of a multi-env stage in ONE
batched RTX pass via ``isaacsim.sensors.experimental.rtx.TiledCameraSensor``
(Replicator ``render_product_tiled`` under the hood — no isaaclab needed), and
writes the de-tiled ``(N, H, W, 3)`` RGB to a ``TiledFrameShm`` segment. The
``isaac_tiled`` bridge (rlinf, Phase 2) reads that + copies each env's slice into
the rlinf Frame SHM ``rgb_buf`` — so this node stays decoupled from rlinf's Frame
SHM layout (same role ROS images play for the OVRTX node).

Physics stays Newton (external); poses are driven from ``/tf_render`` exactly
like ``isaacsim_render.py`` — this node only renders.

Verified feasible by the P0 spike: N=8 tiled ~26fps vs OVRTX ~9fps (~2.8x).

Usage (in the geniesim_ros container, after sourcing ROS):
    python3 isaacsim_tiled_render.py --manifest /tmp/rlinf_multi_2env_p5601/manifest.json \\
        --shm-name rlinf_tiled --render-hz 30
    # standalone verify (no engine): render N frames, dump PNGs, exit:
    python3 isaacsim_tiled_render.py --manifest ... --frames 30 --save-dir /tmp/tiled_out
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional


def _bootstrap_source_path() -> None:
    """Put ``source/`` on sys.path so rlinf_geniesim.render.tiled_shm imports
    (same trick rl_renderer.py uses)."""
    src = os.environ.get("SIM_REPO_ROOT")
    if src and os.path.isdir(os.path.join(src, "source")):
        sys.path.insert(0, os.path.join(src, "source"))
        return
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "source"
        if cand.is_dir():
            sys.path.insert(0, str(cand))
            return


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isaac Sim tiled RTX render node")
    p.add_argument("--manifest", required=True, help="stage manifest.json path")
    p.add_argument("--shm-name", default="rlinf_tiled", help="TiledFrameShm segment name (bridge attaches to this)")
    p.add_argument("--render-hz", type=float, default=30.0)
    p.add_argument("--num-envs", type=int, default=1, help="if >1: build the multi-env stage via GridCloner")
    p.add_argument("--env-spacing", type=float, default=15.0)
    p.add_argument("--frames", type=int, default=0, help="if >0: render this many frames then exit (standalone test)")
    p.add_argument("--save-dir", default="", help="if set: dump per-env RGB PNGs after rendering (test)")
    p.add_argument("--headless", type=int, default=1)
    return p.parse_args(argv)


# Parse args + launch SimulationApp BEFORE any omni/isaacsim import.
_ARGS = parse_args([a for a in sys.argv[1:] if not a.startswith("__")])
_bootstrap_source_path()
# Hide our CLI from Kit's SimulationApp (it parses sys.argv for --/... args).
sys.argv = [sys.argv[0]]

from genie_sim_engine_core.common.parent_lifetime import exit_with_parent  # noqa: E402

# Before Kit starts: a server killed during the slow startup must not leave this node behind.
exit_with_parent()

from isaacsim import SimulationApp  # noqa: E402
from genie_sim_engine_isaacsim.kit.launch_config import prepare_launch_config  # noqa: E402

# ``disable_viewport_updates``: headless STILL renders the invisible viewport
# every frame otherwise — a whole second render nobody looks at. The benchmark
# launcher (geniesim_benchmark/app/workflow/app_launcher.py) has always passed
# this; this node had not.
#
# Measured 2026-08-07, 8 interleaved runs (4 on / 4 off, same binary, only this
# flag toggled): 64.2 -> 46.3 ms per frame (-28%), client 8.43 -> 9.83 steps/s
# (+17%), the two groups not overlapping at all (slowest ON 49.4 vs fastest OFF
# 62.5 ms). Observation cameras never went through the viewport, so per-tile std
# is unchanged. ``RLINF_TILED_DISABLE_VIEWPORT=0`` restores the old behaviour.
simulation_app = SimulationApp(
    prepare_launch_config(
        {
            "headless": bool(_ARGS.headless),
            "disable_viewport_updates": (
                bool(_ARGS.headless) and os.environ.get("RLINF_TILED_DISABLE_VIEWPORT", "1").strip() != "0"
            ),
        }
    )
)

# --- post-SimulationApp imports ---
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

# Isaac Sim 6.2 exports the tiled camera runtime from experimental.rtx.
from isaacsim.sensors.experimental.rtx import TiledCameraSensor  # noqa: E402

import _isaac_render_common as common  # noqa: E402
from rlinf_geniesim.render.bridges._hotswap_protocol import (  # noqa: E402
    peek_generation,
    peek_randomize_generation,
    read_cmd_if_new,
    read_randomize_cmd_if_new,
    write_ack,
    write_randomize_ack,
)
from rlinf_geniesim.render.tiled_shm import TiledFrameShm  # noqa: E402

# The adaptive-subframes formula, its bounds, and the full backstory all
# live in ``_isaac_render_common`` (``subframes_for_gap``) -- once the
# fused node implements on-demand rendering it will also produce gaps, and
# each side keeping its own copy is bound to drift, with the drift showing
# up as "one backend's image is quietly blurrier than the other's". The
# environment variable name stays the same.
_SUBFRAME_LOG = bool(os.environ.get("RLINF_TILED_SUBFRAME_LOG", ""))
_subframes_for_gap = common.subframes_for_gap


def _render_once(subframes: int = 1) -> None:
    """Force one RTX render + annotator populate (bare app.update doesn't)."""
    try:
        rep.orchestrator.step(rt_subframes=max(1, int(subframes)), pause_timeline=False)
    except TypeError:
        rep.orchestrator.step()


class SceneBuildError(RuntimeError):
    """Raised ONLY by ``build_scene`` for its own "no obs camera prims
    resolved" condition — deliberately distinct from the plain
    ``RuntimeError`` (and subclasses thereof) that
    ``common.build_gridcloner_stage`` raises for unrelated stage-build
    failures (e.g. "camera link 'X' not found"). ``main()``'s boot call
    site catches only THIS type, so those unrelated errors keep propagating
    to the module-level FATAL handler (traceback + exit 1) exactly as they
    did before this node grew a ``build_scene`` wrapper; ``_do_stage_reload``
    still catches ``Exception`` broadly (any failure there -> ack ok=False,
    node stays alive)."""


def build_scene(manifest: "common.Manifest", simulation_app, num_envs: int, env_spacing: float) -> dict:
    """Build (or REBUILD) the GridCloner stage + per-env TiledCameraSensor
    group for ``manifest`` — the ONE construction path for both boot (called
    once from ``main()``, below) and hot-swap reload (called again, on a
    fresh empty stage, from ``main()``'s nested ``_do_stage_reload``) — so a
    reload can never drift from what boot would have built for the same
    manifest (see docs/superpowers/specs/2026-07-26-render-hotswap-design.md).

    Stage via GridCloner at ANY env count (N==1 included): every driveable
    prim lives under /World/envs/env_<i>/ and the bridge publishes matching
    /World/envs/env_<i>/<label> flat-pose frames. The old N==1 special case
    (init_stage + resolve_obs_camera_prims: original prim paths, cameras on
    /RenderOVRTX anchors) was never validated end-to-end — the engine's own
    /tf_render frames don't survive to the driver (the flat channel drops
    that subscription), object /entity prims lack the resetXformStack /
    physics-disable treatment, and none of the bridge's frame dialects
    matched it, so an N==1 run rendered a frozen initial stage forever.

    Returns a dict: ``stage``, ``cam_paths``, ``env_origins``, ``obs``
    (non-free-cam ``CameraConfig``s), ``n_cams``, ``n_envs``, ``n`` (total
    camera tiles == ``n_envs * n_cams``), ``W``, ``Hmax``, ``cam_hw`` (per-
    camera ``(height, width)``), ``sensors`` (list of
    ``(TiledCameraSensor, flat_idx, gh)`` — one entry per distinct obs-camera
    HEIGHT, see the grouping comment below).

    Raises:
        SceneBuildError: no obs camera prims resolved — mirrors the old
            boot-time ``"[tiled] ERROR: no obs camera prims resolved"`` +
            ``exit(2)``; the caller decides how to surface it (``main()``
            prints + returns 2 at boot, ``_do_stage_reload`` lets it
            propagate into ``ack(ok=False)`` on reload). Any OTHER exception
            (e.g. a plain ``RuntimeError`` from
            ``common.build_gridcloner_stage`` such as "camera link 'X' not
            found") propagates unchanged — ``main()``'s boot call site only
            catches ``SceneBuildError``, so those keep hitting the
            module-level FATAL handler (traceback + exit 1) as before.
    """
    stage, cam_paths, env_origins = common.build_gridcloner_stage(manifest, simulation_app, num_envs, env_spacing)
    if not cam_paths:
        raise SceneBuildError("no obs camera prims resolved")
    n = len(cam_paths)
    obs = manifest.obs_cameras()
    n_cams = len(obs)
    n_envs = (n // n_cams) if n_cams else n
    # Each obs camera shares a common WIDTH but may have its own HEIGHT (benchmark
    # aspect: head 640x400, wrist 640x528). cam_paths is [env][cam] order.
    cam_hw = [(int(c.height), int(c.width)) for c in obs]
    W = cam_hw[0][1]
    Hmax = max(h for h, _w in cam_hw)
    print(f"[tiled] {n_cams} obs cam(s) width={W} heights={[h for h, _ in cam_hw]} " f"{n_envs} env(s)", flush=True)
    # Per-camera intrinsics on env_0's prims (clones inherit the model).
    for cam, path in zip(obs, cam_paths):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            try:
                common.set_camera_intrinsics(prim, cam)
            except Exception as e:  # noqa: BLE001
                print(f"[tiled] intrinsics failed for {path}: {e!r}", flush=True)

    # Group cameras by obs HEIGHT (a TiledCameraSensor needs a uniform tile size);
    # one sensor per distinct height, ALL rendered in the same RTX pass. Each
    # group's tiles scatter into ONE uniform (Hmax) SHM at their global
    # [env*n_cams + cam] index; cameras shorter than Hmax leave the trailing rows
    # zero (cropped back to their real height at the obs). One height => one
    # sensor, bulk copy — identical to the old uniform path.
    from collections import OrderedDict  # noqa: PLC0415

    _groups: "OrderedDict[int, list[int]]" = OrderedDict()
    for j, (hh, _w) in enumerate(cam_hw):
        _groups.setdefault(hh, []).append(j)
    sensors = []  # (TiledCameraSensor, flat_idx: list[int], gh)
    for gh, cam_js in _groups.items():
        paths, flat_idx = [], []
        for i in range(n_envs):
            for j in cam_js:
                paths.append(cam_paths[i * n_cams + j])
                flat_idx.append(i * n_cams + j)
        s = TiledCameraSensor(paths, resolution=(gh, W), annotators=["rgb"])
        sensors.append((s, flat_idx, gh))
        print(f"[tiled] sensor @ {W}x{gh}: cams={cam_js} ({len(paths)} tiles)", flush=True)
    return {
        "stage": stage,
        "cam_paths": cam_paths,
        "env_origins": env_origins,
        "obs": obs,
        "n_cams": n_cams,
        "n_envs": n_envs,
        "n": n,
        "W": W,
        "Hmax": Hmax,
        "cam_hw": cam_hw,
        "sensors": sensors,
    }


# Render-side generalization (lights, material, camera pose) is applied by code
# shared with the fused node; see _render_randomize.
from _render_randomize import (  # noqa: E402
    _apply_randomize_cmd,
    _read_camera_pose_baseline,
    _read_light_baseline,
)


def main() -> int:
    manifest = common.Manifest.load(_ARGS.manifest)
    print(f"[tiled] scene: {manifest.scene_usda}", flush=True)
    # Hot-swap control channel: a FIXED directory for this whole process's
    # lifetime (the boot --manifest's own directory) — regardless of which
    # manifest.json path a LATER reload command points at. The bridge must
    # remember and reuse this SAME directory for every reload_stage() call
    # (see _hotswap_protocol.py + item 1 of the design's component-contract
    # section).
    _manifest_dir = Path(_ARGS.manifest).resolve().parent
    # Critical (final-review C1) — node-side half, belt-and-suspenders
    # alongside the bridge's own ``clear_stale_control_files`` at launch
    # (isaac_tiled.py's ``_build_argv``): this control dir is not per-pid
    # (see the M1 note there), so a cmd file can in principle still be
    # sitting here at boot (e.g. this script launched directly, outside the
    # bridge). Seed ``last_generation`` from it via ``peek_generation``
    # instead of hardcoding 0 — a freshly launched node is ALREADY on the
    # correct manifest (built straight from its own ``--manifest`` argv
    # above), so any pre-existing cmd must be treated as already consumed,
    # never acted on as a spurious boot reload.
    last_generation = peek_generation(_manifest_dir)
    # The generalization-layer render-side command channel -- a separate
    # file, separate generation, with the same belt-and-suspenders "seed
    # from whatever leftover file is in the control directory" logic as
    # the reload counter above, for the same reason (see peek_generation's
    # docstring).
    last_randomize_generation = peek_randomize_generation(_manifest_dir)
    try:
        scene = build_scene(manifest, simulation_app, _ARGS.num_envs, _ARGS.env_spacing)
    except SceneBuildError as e:
        # ONLY build_scene's own "no obs camera prims resolved" condition —
        # any other exception (e.g. a plain RuntimeError from
        # common.build_gridcloner_stage such as "camera link 'X' not found")
        # must propagate to the module-level FATAL handler below, unchanged
        # from pre-hot-swap behavior.
        print(f"[tiled] ERROR: {e}", flush=True)
        return 2
    stage, cam_paths, env_origins = (scene["stage"], scene["cam_paths"], scene["env_origins"])
    obs, n_cams, n_envs, n = (scene["obs"], scene["n_cams"], scene["n_envs"], scene["n"])
    W, Hmax, cam_hw, sensors = (scene["W"], scene["Hmax"], scene["cam_hw"], scene["sensors"])
    # The lights perturbation baseline -- held in a mutable dict (rather
    # than a plain local variable) so that _do_stage_reload (a closure,
    # which never needs nonlocal for this name) can swap in the new
    # stage's baseline with .clear()+.update(), and main()'s main loop
    # always sees the latest one.
    _light_baseline: dict = _read_light_baseline(stage)
    # The camera pose perturbation baseline -- the same "mutable dict +
    # .clear()+.update()" technique, for the same reason (see
    # _read_camera_pose_baseline's docstring).
    _camera_pose_baseline: dict = _read_camera_pose_baseline(stage, cam_paths, obs, n_cams, n_envs)

    shm = TiledFrameShm(_ARGS.shm_name, num_envs=n, height=Hmax, width=W, create=True)
    print(f"[tiled] tiled-SHM '{shm.name}' ({n},{Hmax},{W},3), {len(sensors)} sensor(s)", flush=True)

    # optional /tf_render driving (Phase 2 integration; harmless with no engine)
    driver = None
    ros_node = None
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from tf2_msgs.msg import TFMessage

        rclpy.init(args=None)
        driver = common.TfRenderDriver(stage, manifest.robot_prefix, env_origins=env_origins)
        ros_node = Node("isaacsim_tiled_render")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

        def _tf_cb(m):
            _a0 = time.monotonic()
            driver.apply(m.transforms)
            _prof["apply"] += time.monotonic() - _a0

        _tf_sub = {"sub": ros_node.create_subscription(TFMessage, "/rlinf/tf_render", _tf_cb, qos)}

        # Reset signal: once received, make **the next** render use
        # _RT_SUBFRAMES_RESET. A reset teleports the whole scene within
        # one tick, so the gap is only 1, and without special-casing it
        # that frame would only get 1 subframe -- which is exactly the
        # frame whose temporal history is completely invalidated, and
        # that the policy treats as the opening observation. Subscribes
        # to both topics: the eval path uses the seeded one (see
        # _ros_bridge.send_reset).
        _reset_pending = {"v": False}

        def _reset_cb(_m):
            _reset_pending["v"] = True

        try:
            from std_msgs.msg import Int32MultiArray, Int64MultiArray  # noqa: PLC0415

            ros_node.create_subscription(Int32MultiArray, "/rlinf/reset_envs", _reset_cb, qos)
            ros_node.create_subscription(Int64MultiArray, "/rlinf/reset_envs_seeded", _reset_cb, qos)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[tiled] reset subscription failed ({exc!r}) -- the reset frame falls back to being judged by gap",
                flush=True,
            )
        # FLAT pose channel: latched path table + per-frame float array. Once
        # live, DROP the /tf_render subscription — deserializing the engine's
        # bare-frame TFMessages (~500 TransformStamped/step, discarded anyway
        # on a GridCloner stage) was pure overhead.
        import json as _json
        from std_msgs.msg import Float64MultiArray as _Flat, String as _Str
        from rclpy.qos import DurabilityPolicy as _Dur

        def _paths_cb(m):
            driver.set_flat_paths(_json.loads(m.data))

        def _flat_cb(m):
            _a0 = time.monotonic()
            driver.apply_flat(m.data)
            _prof["apply"] += time.monotonic() - _a0
            # Drop the tf fallback only once the flat channel is ACTUALLY
            # usable (this generation's path table is installed). It used to
            # unsubscribe on the first flat message even when that message was
            # unusable — so if the path table arrived late or went missing, the
            # generation had neither flat nor tf: no pose applied, tick frozen,
            # the node down to its 2 s heartbeat, a still image still being
            # scored. Worst case now is staying on tf (slow) instead of
            # freezing silently (wrong).
            if _tf_sub["sub"] is not None and driver.has_flat_paths():
                ros_node.destroy_subscription(_tf_sub["sub"])
                _tf_sub["sub"] = None
                print("[tiled] flat pose channel active — dropped /tf_render " "subscription", flush=True)

        # Ordering dependency (undocumented in code, only here): pose_paths
        # must be subscribed BEFORE pose_flat — rclpy services callbacks in
        # registration order, so this is what lets _paths_cb install the
        # table ahead of a same-tick _flat_cb; swap them and apply_flat's
        # no-table WARNING starts firing once per reload in healthy
        # operation, turning a real warning into ignored noise.
        ros_node.create_subscription(
            _Str,
            "/rlinf/pose_paths",
            _paths_cb,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=_Dur.TRANSIENT_LOCAL,
            ),
        )
        ros_node.create_subscription(_Flat, "/rlinf/pose_flat", _flat_cb, qos)
        print("[tiled] subscribed /tf_render + /rlinf/pose_flat", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[tiled] ROS disabled ({e!r}); rendering static stage", flush=True)

    period = 1.0 / max(_ARGS.render_hz, 1e-3)
    frame_id = 0
    last_data = None
    t_start = time.monotonic()
    n_timed, t_timed0 = 0, None
    # ON-DEMAND rendering: an RTX pass only when NEW poses arrived (the driver's
    # /tf_render tick changed), plus a low-rate heartbeat for GUI/preview
    # freshness. The old unconditional loop re-rendered the identical frame
    # back-to-back (12 tiles ~600ms >> the 33ms period) and pegged the GPU at
    # 99% util even with the sim fully idle. Standalone --frames test and the
    # no-ROS path keep the legacy free-run behavior.
    heartbeat_s = 0.5 if not _ARGS.headless else 2.0
    last_tick_rendered: int | None = None
    last_render_t = 0.0
    _prof = {
        "rtx": 0.0,
        "get": 0.0,
        "tonp": 0.0,
        "copy": 0.0,
        "pub": 0.0,
        "spin": 0.0,
        "apply": 0.0,
        "n": 0,
        "t_log": time.monotonic(),
    }
    from _render_perf import RenderPerfLogger

    _perf_csv = RenderPerfLogger()
    _last_frame_wall = time.monotonic()
    _last_apply_acc = 0.0
    _sfstat = {"n": 0, "sum": 0}

    def _do_stage_reload(manifest_path: str, generation: int) -> None:
        """Render hot-swap: rebuild the stage IN PLACE onto ``manifest_path``
        (a NEW manifest.json — a different task_init) without restarting this
        process. Reuses the SAME tiled-SHM segment (dims are checked below;
        the caller's applicability gate is supposed to guarantee they match)
        and the SAME ROS node/subscriptions — only ``driver`` (the pose ->
        prim bindings, which cache prim handles from the stage being torn
        down) gets replaced.

        On ANY failure: write ``ack(ok=False, error=repr(e))`` and return —
        the node STAYS ALIVE (per the spec's failure-semantics section: the
        bridge/caller falls back to
        a full cold restart; this process just waits for the next command,
        or to be killed)."""
        nonlocal _perf_csv, _last_frame_wall, _last_apply_acc
        nonlocal stage, cam_paths, env_origins, obs, n_cams, n_envs, n
        nonlocal W, Hmax, cam_hw, sensors, driver, last_tick_rendered
        print(f"[tiled] reload: generation={generation} manifest={manifest_path}", flush=True)
        try:
            new_manifest = common.Manifest.load(manifest_path)
            # Drop the OLD sensors/annotators while the OLD stage is STILL
            # composed (TiledCameraSensor.__del__ tears down its render
            # product/annotators — do that against real prims, not orphans)
            # before wiping the stage out from under them.
            sensors = []
            import omni.usd  # noqa: PLC0415

            omni.usd.get_context().new_stage()
            simulation_app.update()
            scene = build_scene(new_manifest, simulation_app, _ARGS.num_envs, _ARGS.env_spacing)
            new_n, new_Hmax, new_W = scene["n"], scene["Hmax"], scene["W"]
            if (new_n, new_Hmax, new_W) != (shm.num_envs, shm.height, shm.width):
                raise RuntimeError(
                    f"reload dims mismatch: new manifest resolves to "
                    f"({new_n},{new_Hmax},{new_W}) tiles but the SHM segment "
                    f"is ({shm.num_envs},{shm.height},{shm.width}) — the "
                    f"applicability gate should have blocked this reload"
                )
            stage = scene["stage"]
            cam_paths = scene["cam_paths"]
            env_origins = scene["env_origins"]
            obs = scene["obs"]
            n_cams = scene["n_cams"]
            n_envs = scene["n_envs"]
            n = new_n
            W = new_W
            Hmax = new_Hmax
            cam_hw = scene["cam_hw"]
            sensors = scene["sensors"]

            # A new stage means a (possibly) new lights baseline -- must
            # re-cache, cannot keep using the old stage's values (the old
            # prim objects were already invalidated by omni.usd.new_stage();
            # continuing to use them would read stale/dead references).
            # Mutated in place (never needs nonlocal for this name) --
            # main()'s main loop holds a reference to this same dict object.
            _light_baseline.clear()
            _light_baseline.update(_read_light_baseline(stage))
            # Likewise, the camera pose baseline must also be re-cached for
            # the new stage -- the lines above have already reassigned
            # cam_paths/obs/n_cams/n_envs to the new stage's.
            _camera_pose_baseline.clear()
            _camera_pose_baseline.update(_read_camera_pose_baseline(stage, cam_paths, obs, n_cams, n_envs))

            # Reset pose-channel state: a FRESH driver bound to the new stage
            # (its path->prim cache starts empty — the next
            # /rlinf/pose_paths message rebuilds it), and re-subscribe
            # /tf_render if a PRIOR generation's flat channel already
            # dropped it (see _flat_cb above).
            if ros_node is not None:
                driver = common.TfRenderDriver(stage, new_manifest.robot_prefix, env_origins=env_origins)
                if _tf_sub["sub"] is None:
                    _tf_sub["sub"] = ros_node.create_subscription(TFMessage, "/rlinf/tf_render", _tf_cb, qos)
                    print("[tiled] reload: re-subscribed /rlinf/tf_render", flush=True)
            # Reset the main-loop render gate too — NECESSARY but not, on its
            # own, SUFFICIENT: without this, the first tick the new driver
            # reports (even a correct, unpoisoned one) would compare equal to
            # whatever huge value last_tick_rendered held from the outgoing
            # generation and never render. But last_tick_rendered gets
            # overwritten by cur_tick on the very next render regardless, so
            # this alone only bought one extra frame — the actual fix for the
            # STICKY multi-generation stall is the tick-watermark reset in
            # TfRenderDriver.set_flat_paths (fires on the paths_cb table
            # install, i.e. the real generation boundary).
            last_tick_rendered = None

            # SHM header: bump generation, zero rendered_tick. The segment
            # (and frame_id) is REUSED as-is — never unlinked/recreated.
            _perf_csv.close()
            _perf_csv = RenderPerfLogger()
            _last_frame_wall = time.monotonic()
            _last_apply_acc = _prof["apply"]
            shm.reset_for_reload(generation)
            write_ack(_manifest_dir, generation, ok=True, error=None, dims=[n_envs, n_cams, Hmax, W])
            print(f"[tiled] reload OK: generation={generation} " f"dims=({n_envs},{n_cams},{Hmax},{W})", flush=True)
        except Exception as e:  # noqa: BLE001 — node stays alive; ack carries the failure
            print(f"[tiled] reload FAILED: generation={generation}: {e!r}", flush=True)
            write_ack(_manifest_dir, generation, ok=False, error=repr(e), dims=None)

    try:
        while simulation_app.is_running():
            t0 = time.monotonic()
            # Hot-swap: cheap per-frame poll (a file-exists check + a small
            # JSON parse, never raises) — a NEW cmd triggers an in-place
            # rebuild instead of a fresh render() this tick. No cmd file ever
            # appearing (the standalone/no-hotswap-caller default) costs one
            # such check per iteration and changes nothing else.
            _cmd = read_cmd_if_new(_manifest_dir, last_generation)
            if _cmd is not None:
                last_generation = _cmd["generation"]
                _do_stage_reload(_cmd["manifest"], _cmd["generation"])
                continue
            # A second poll, equally cheap, equally never-raising -- a
            # separate file, separate generation (see the
            # _hotswap_protocol module docstring, "the two channels never
            # interfere with each other"). Only changes UsdLux
            # attributes, triggers no stage rebuild at all, needs no
            # extra overhead beyond the continue.
            _rcmd = read_randomize_cmd_if_new(_manifest_dir, last_randomize_generation)
            if _rcmd is not None:
                last_randomize_generation = _rcmd["generation"]
                _apply_randomize_cmd(
                    stage, _light_baseline, _camera_pose_baseline, _rcmd["render"], _manifest_dir, _rcmd["generation"]
                )
                continue
            if ros_node is not None:
                import rclpy

                _ts0 = time.monotonic()
                rclpy.spin_once(ros_node, timeout_sec=0.0)
                _prof["spin"] += time.monotonic() - _ts0  # contains driver.apply
            cur_tick = driver.last_tick() if driver is not None else None
            need = (
                _ARGS.frames > 0  # standalone render test
                or driver is None  # no ROS: legacy free-run
                or cur_tick != last_tick_rendered  # new poses -> new frame
                or (t0 - last_render_t) >= heartbeat_s
            )
            if not need:
                # Idle: wait for the NEXT pose message instead of sleeping a
                # fixed 5ms. The sleep ran while the spin_once above is
                # non-blocking (timeout_sec=0.0), so a pose landing just after
                # the poll sat unnoticed for the rest of the sleep — pure added
                # latency on the caller's barrier, which is the hot path
                # (handler render_barrier p50 73.8ms vs this node's own frame
                # p50 69.9ms). Blocking here wakes the loop the moment the
                # message arrives; the timeout keeps the heartbeat (and the
                # reload cmd poll above) alive.
                if ros_node is not None:
                    import rclpy

                    _ts0 = time.monotonic()
                    rclpy.spin_once(ros_node, timeout_sec=0.005)
                    _prof["spin"] += time.monotonic() - _ts0
                else:
                    time.sleep(0.005)
                continue
            # How many ticks were skipped since the last render -- see
            # _subframes_for_gap's comment. Either end may be None (a
            # free-run with no ROS / the watermark cleared after a hot
            # reload), in which case passing None makes it handled as
            # "treat accumulation as empty".
            _gap = None if (cur_tick is None or last_tick_rendered is None) else int(cur_tick) - int(last_tick_rendered)
            _after_reset = _reset_pending["v"]
            _reset_pending["v"] = False
            _sub = _subframes_for_gap(_gap, after_reset=_after_reset)
            _t0 = time.monotonic()
            _render_once(_sub)
            _t1 = time.monotonic()
            # Diagnostic logging, off by default (set
            # RLINF_TILED_SUBFRAME_LOG=1 to turn it on when troubleshooting
            # "is the subframe count ramping up as expected"). Prints every
            # time for the first 12 ticks (covering cold start and the
            # first few chunks), then once every 50 after that, to avoid
            # spamming the log.
            _sfstat["n"] += 1
            _sfstat["sum"] += _sub
            if _SUBFRAME_LOG and (_sfstat["n"] <= 12 or _sfstat["n"] % 50 == 0):
                print(
                    f"[subframes] n={_sfstat['n']} gap={_gap}"
                    f"{' RESET' if _after_reset else ''} -> {_sub} "
                    f"subframe(s), rtx={(_t1 - _t0) * 1000:.1f}ms "
                    f"(avg sub {_sfstat['sum'] / _sfstat['n']:.2f})",
                    flush=True,
                )
            last_tick_rendered = cur_tick
            last_render_t = _t1
            _got = False
            _get_s = _numpy_s = _copy_s = 0.0
            for s, flat_idx, gh in sensors:
                _get_start = time.monotonic()
                data, _info = s.get_data("rgb")
                _numpy_start = time.monotonic()
                _get_s += _numpy_start - _get_start
                if data is None:
                    continue
                arr = (data.numpy() if hasattr(data, "numpy") else np.asarray(data))[..., :3]
                # The annotator already hands back uint8 RGBA; an
                # unconditional .astype(np.uint8) therefore bought nothing but
                # a full extra host copy of the whole tile batch every frame
                # (~1.8ms at 3 tiles, ~37ms at 48). Convert only if it really
                # isn't uint8 — the SHM assignment below casts anyway.
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                _copy_start = time.monotonic()
                _numpy_s += _copy_start - _numpy_start
                if int(arr.shape[0]) != len(flat_idx):
                    continue
                # Scatter this group's tiles into the uniform (Hmax) SHM at their
                # global [env*n_cams+cam] index; rows [gh:] stay zero (pad).
                for k, gi in enumerate(flat_idx):
                    shm.rgb[gi, :gh] = arr[k]
                _copy_s += time.monotonic() - _copy_start
                _got = True
            _t4 = time.monotonic()
            if _got:
                frame_id += 1
                tick = driver.last_tick() if driver is not None else 0
                shm.publish(frame_id, rendered_tick=tick)
                _t5 = time.monotonic()
                # Per-phase profile (accumulated, logged ~every 5s): where the
                # per-frame time actually goes — RTX pass vs annotator readback
                # vs host convert vs SHM copy.
                _prof["rtx"] += _t1 - _t0
                _prof["get"] += _get_s
                _prof["tonp"] += _numpy_s
                _prof["copy"] += _copy_s
                _prof["pub"] += _t5 - _t4
                _perf_csv.log(
                    frame=frame_id,
                    monotonic_s=_t5,
                    tick=tick,
                    generation=last_generation,
                    n_tiles=len(cam_paths),
                    total_ms=(_t5 - t0) * 1000,
                    spin_ms=(_t0 - t0) * 1000,
                    apply_ms=(_prof["apply"] - _last_apply_acc) * 1000,
                    render_call_ms=(_t1 - _t0) * 1000,
                    get_data_ms=_get_s * 1000,
                    to_numpy_ms=_numpy_s * 1000,
                    shm_copy_ms=_copy_s * 1000,
                    publish_ms=(_t5 - _t4) * 1000,
                    interval_ms=(_t5 - _last_frame_wall) * 1000,
                )
                _last_frame_wall = _t5
                _last_apply_acc = _prof["apply"]
                _prof["n"] += 1
                if _t5 - _prof["t_log"] >= 5.0 and _prof["n"]:
                    n = _prof["n"]
                    print(
                        f"[tiled-prof] {n} frame(s): "
                        f"rtx={_prof['rtx']/n*1000:.1f}ms "
                        f"get_data={_prof['get']/n*1000:.1f}ms "
                        f"to_numpy={_prof['tonp']/n*1000:.1f}ms "
                        f"shm_copy={_prof['copy']/n*1000:.1f}ms "
                        f"publish={_prof['pub']/n*1000:.1f}ms "
                        f"spin={_prof['spin']/n*1000:.1f}ms "
                        f"apply_inner={_prof['apply']/n*1000:.1f}ms "
                        f"({len(cam_paths)} tiles)",
                        flush=True,
                    )
                    for k in ("rtx", "get", "tonp", "copy", "pub", "spin", "apply"):
                        _prof[k] = 0.0
                    _last_apply_acc = 0.0
                    _prof["n"] = 0
                    _prof["t_log"] = _t5
                last_data = shm.rgb
                if t_timed0 is None:  # start fps timing after first real frame
                    t_timed0 = time.monotonic()
                else:
                    n_timed += 1
            if _ARGS.frames and frame_id >= _ARGS.frames:
                break
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        _perf_csv.close()

    if n_timed and t_timed0:
        fps = n_timed / (time.monotonic() - t_timed0)
        print(f"[tiled] rendered {frame_id} frames, ~{fps:.1f} fps", flush=True)

    if _ARGS.save_dir and last_data is not None:
        d = Path(_ARGS.save_dir)
        d.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image

            for i in range(n):
                Image.fromarray(np.asarray(last_data[i])).save(d / f"tiled_env{i}.png")
            print(f"[tiled] saved {n} PNG(s) to {d}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[tiled] PNG save failed: {e!r}", flush=True)

    shm.close(unlink=True)
    if ros_node is not None:
        import rclpy

        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


try:
    _rc = main()
except Exception as e:  # noqa: BLE001
    import traceback

    print("[tiled] FATAL:", repr(e), flush=True)
    traceback.print_exc()
    _rc = 1
finally:
    simulation_app.close()

sys.exit(_rc)
