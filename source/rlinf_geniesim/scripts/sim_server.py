#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Start an idle ZMQ server for the released GenieSim tasks.

Clients select an official task/scene pair and physics/render backends in
the task-init payload. The first successful initialization fixes the
backend settings for the server process; reloads can select another task.
Restart the server to change its backend settings.

Run inside the prepared runtime after sourcing ROS:

    python3 sim_server.py [--comm-config <comm.yaml>]

``--comm-config`` selects the transport binding; the default is ZMQ at
``tcp://0.0.0.0:5555``. Task selection happens through client initialization."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import copy
import faulthandler
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Bootstrap: make the rlinf_geniesim package importable regardless of cwd.
# ---------------------------------------------------------------------------


def _bootstrap_geniesim() -> None:
    """Find ``source/`` through SIM_REPO_ROOT or this script's checkout."""
    src = os.environ.get("SIM_REPO_ROOT")
    if src:
        candidate = os.path.join(src, "source")
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
            return
    # Fallback: this file is at .../source/rlinf_geniesim/scripts/...
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "source"
        if candidate.is_dir():
            sp = str(candidate)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return


_bootstrap_geniesim()


from rlinf_geniesim.physics.backends import make_engine  # noqa: E402
from rlinf_geniesim.render.bridges import (  # noqa: E402
    make_bridge,
    bridge_supports_kwarg,
    bridges_supporting_kwarg,
)
from rlinf_geniesim.render.bridges.isaac_tiled import (  # noqa: E402
    IsaacTiledBackend,
)
from rlinf_geniesim.comm import make_transport  # noqa: E402
from rlinf_geniesim.comm.base import SimDims, SimHandler  # noqa: E402
from rlinf_geniesim.owner_watchdog import start_owner_watchdog  # noqa: E402
from rlinf_geniesim.randomize import RandomizationSpec  # noqa: E402
from rlinf_geniesim.server_identity import (  # noqa: E402
    default_multi_env_dir,
    default_stage_dir,
    set_server_tag_from_bind,
)
from rlinf_geniesim.sim_handler import GenieSimHandler  # noqa: E402
from rlinf_geniesim.task_background import resolve_use_background  # noqa: E402
from rlinf_geniesim.task_loader import (  # noqa: E402
    build_stage_manifest,
    build_superset_stage_manifest,
    last_build_randomization,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    """Load a YAML or JSON config. We avoid a hard PyYAML dependency:
    if the file is .json it loads natively; if .yaml we try PyYAML
    and fall back with a friendly error."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    text = p.read_text()
    if p.suffix in (".json",):
        return json.loads(text)
    # YAML
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Loading {path} requires PyYAML. Use a .json config instead, " f"or pip install pyyaml."
        ) from e
    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# SimDims inference
# ---------------------------------------------------------------------------


def _camera_render_dims(cameras: list) -> tuple[int, int, int, tuple]:
    """``(num_cameras, image_h, image_w, cam_heights)`` — the render-obs-shape
    slice of ``_infer_sim_dims`` (below), extracted so it can ALSO be computed
    from a task-init payload alone, with no engine/bridge built or started
    (see ``_render_dims_from_init``, used by the reload hot-swap
    applicability gate ``_hotswap_applicable``) — both call sites compute the
    SAME numbers from the SAME camera list, so they can never diverge.

    Each camera shares a common WIDTH but may have its own HEIGHT (benchmark
    aspect: head 640x400, wrist 640x528). The frame buffer is sized at the
    MAX height; per-camera heights ride in ``cam_heights`` so the obs is
    cropped back to each camera's real height.
    """
    cameras = cameras or []
    if cameras and "image_hw" in cameras[0]:
        cam_heights = tuple(int(c["image_hw"][0]) for c in cameras if "image_hw" in c)
        w = int(cameras[0]["image_hw"][1])
        h = max(cam_heights) if cam_heights else int(cameras[0]["image_hw"][0])
    else:
        cam_heights = ()
        h, w = 256, 256
    return len(cameras), int(h), int(w), cam_heights


# Fallback body-row count, used only when the engine cannot be asked (see
# ``_resolve_num_bodies``). It is NOT a guess at any particular scene: it is
# an upper bound generous enough that the fallback path still produces a
# working buffer. The old value here was 32, below this robot's 56 links, so
# the fallback was a guaranteed failure rather than a safety net.
#
# Over-allocating is close to free. A row is 7 float32 = 28 bytes, so the
# whole region is 128 x 28 = 3.5 KiB per env, and the per-step cost is that
# same memcpy — order 1 us against a ~90 ms step. Under-allocating silently
# deletes data. The asymmetry says: round up.
_NUM_BODIES_FALLBACK = 128

# Headroom over the engine's reported body count. Covers a scene whose body
# set grows slightly after sizing (a reload hot-swap into a task with a few
# more props) without a re-negotiation.
_NUM_BODIES_HEADROOM = 8


def _resolve_num_bodies(scene_cfg: dict, engine) -> int:
    """Rows to allocate for ``body_poses``: ask the engine, else fall back.

    Order of precedence, and why:

    1. An explicit ``scene.num_bodies_estimate`` wins outright. An operator
       pinning it (``host_run_teleop.sh --num-bodies=N``) is usually
       reproducing a specific wire layout, and silently "improving" on that
       would make two runs incomparable.
    2. Otherwise the engine's live body list, plus headroom. This is the
       authoritative answer — the same list the tape guard and the eval
       checkers index into.
    3. Otherwise ``_NUM_BODIES_FALLBACK``, said out loud.
    """
    pinned = scene_cfg.get("num_bodies_estimate")
    if pinned is not None:
        return int(pinned)

    waiter = getattr(engine, "wait_for_body_link_names", None)
    names = waiter() if callable(waiter) else []
    if names:
        n = len(names) + _NUM_BODIES_HEADROOM
        n_obj = sum(1 for nm in names if "/Objects/" in nm)
        print(
            f"[sim_server] num_bodies={n} (engine reported {len(names)} bodies, "
            f"including {n_obj} objects; headroom={_NUM_BODIES_HEADROOM})",
            flush=True,
        )
        return n

    # Loud on purpose. This is the branch where the wire layout stops being
    # derived from the actual scene, and a silent fallback here is what let
    # the 32-row buffer survive as long as it did.
    print(
        f"[sim_server] WARNING: engine body names unavailable; using num_bodies="
        f"{_NUM_BODIES_FALLBACK}. Bodies beyond this capacity have no pose data. "
        f"Check /tf_render or set scene.num_bodies_estimate explicitly.",
        flush=True,
    )
    return _NUM_BODIES_FALLBACK


def _infer_sim_dims(cfg: dict, engine, bridge) -> SimDims:
    """Pull the dims needed to size SHM regions from the running backend.

    ``num_bodies`` is asked of the ENGINE, not guessed from config. The old
    guess (32) was smaller than this robot's own link count (56), and objects
    sort after the robot links in ``body_link_names`` — so every object fell
    outside the buffer on every task, forever. ``sim_handler._copy_into``
    trims silently, so the wire carried all-zero object poses and the only
    symptom was a flyaway/AABB judgement quietly scoring nothing. It took a
    client-side tape guard to make it visible.

    Why a WAIT and not just a read: ``engine.start()`` blocks on the first
    ``/joint_states``, not the first ``/tf_render``, and the body names come
    from the latter — so reading here without waiting usually gets ``[]``,
    which is worse than the estimate. See
    ``_UpstreamSubprocessBackend.wait_for_body_link_names`` for why the first
    non-empty read is already the complete set.

    Falling back is still possible (backends that never enable
    ``use_tf_render`` have no ``/tf_render`` at all), so the fallback value
    must be one that actually works — hence ``_NUM_BODIES_FALLBACK`` rather
    than the old 32 — and the fallback must say so out loud.
    """
    physics = cfg.get("physics", {})
    render = cfg.get("render", {})
    cameras = render.get("cameras", []) or []

    scene_cfg = cfg.get("scene", {}) or {}
    num_objects = int(scene_cfg.get("num_objects_estimate", 8))
    num_bodies = _resolve_num_bodies(scene_cfg, engine)

    num_cameras, h, w, cam_heights = _camera_render_dims(cameras)

    return SimDims(
        num_envs=int(physics.get("num_envs", 1)),
        action_dim=engine.action_dim,
        state_dim=engine.state_dim,
        num_joints=engine.action_dim,
        num_bodies=num_bodies,
        num_objects=num_objects,
        num_cameras=num_cameras,
        image_h=h,
        image_w=w,
        cam_heights=cam_heights,
    )


# ---------------------------------------------------------------------------
# Config split: backend (platform) config + task (scene) config
# ---------------------------------------------------------------------------


def _merge_config(backends: dict, task: dict) -> dict:
    """Combine a backend (platform) config and a task (scene) config into the
    internal cfg shape ``build_handler`` / ``_infer_sim_dims`` use.

    Backend config owns *what runs and how fast* (engine/render/comm backend
    selection, num_envs, hz, device, headless, solver / ovrtx kwargs) — stable
    per machine. Task config owns *what scene* (scene_yaml/robot_urdf,
    stage_manifest, cameras, multi-env layout) — varies per experiment, and is
    what an RLinf ``GenieSimEnv`` would pass in.

    The caller's dicts are not mutated (task fields are merged into copies).
    """
    physics = dict(backends.get("physics", {}))
    render = dict(backends.get("render", {}))
    comm = dict(backends.get("comm", {"backend": "shm", "bootstrap_out": "/tmp/rlinf_bootstrap_demo.json"}))
    scene = dict(task.get("scene", {}))

    stage_manifest = task.get("stage_manifest", "")
    if stage_manifest:
        # Both backends load the SAME assembled stage.
        physics["stage_manifest"] = stage_manifest
        render["stage_manifest"] = stage_manifest
    render["cameras"] = task.get("cameras", [])

    return {"physics": physics, "render": render, "scene": scene, "comm": comm}


def _policy_step_dt(physics_cfg: dict) -> Optional[float]:
    """1 ``engine.step()`` call's sim-time span, i.e. the LOCKSTEP
    ``policy_step_dt`` the post-reset settle feature needs to convert
    ``task.reset_settle_time_s`` into a step count (see
    ``GenieSimHandler.__init__``). Derived from the CONFIG's
    ``control_hz`` (1 control step == ``round(physics_hz/control_hz)``
    physics ticks, command-driven — see ``_upstream_base.py``'s lockstep
    doc), never introspected from the engine object itself (no backend
    exposes a generic dt getter, and guessing one would silently mis-time
    the settle window). ``control_hz<=0`` (absent, or the legacy
    real-time free-run knob) -> ``None``: settle then can't compute a step
    count and stays disabled — see ``GenieSimHandler``'s no-guess
    fallback."""
    control_hz = float(physics_cfg.get("control_hz", 0.0) or 0.0)
    return (1.0 / control_hz) if control_hz > 0 else None


def _split_cfg_pieces(backends: dict, task: dict):
    """Merge (backends, task) into the internal cfg shape and hand back
    working copies of its ``physics``/``render`` sections with the
    render-layout defaults ``build_handler`` always fills in (env_spacing /
    head_height_m / multi_env_output_dir — read by the BRIDGE, not the
    engine).

    Extracted out of ``build_handler`` (below) so the reload hot-swap engine
    rebuild (``_build_reload_engine``, below — engine only, no bridge) shares
    the EXACT same cfg-shaping as the cold build's engine+bridge path; the
    two can never quietly diverge on how ``cfg``/``physics``/``render`` get
    assembled from a (backends, task) pair.
    """
    cfg = _merge_config(backends, task)
    physics = dict(cfg["physics"])
    render = dict(cfg["render"])
    scene = cfg["scene"]

    num_envs = int(physics.get("num_envs", 1))

    # Multi-env RENDER organization is the RENDERER's responsibility, not the
    # generic layer's: each backend decides how it makes N visual envs — the
    # OVRTX backend assembles a scene_multi.usda (in its own class), the Isaac
    # tiled backend uses GridCloner. So we do NOT assemble anything here; we hand
    # the bridge the SINGLE-env stage_manifest + num_envs + env_spacing /
    # head_height_m and let it organize. PHYSICS always keeps the single-env
    # manifest (the engine replicates it natively into N worlds).
    render.setdefault("env_spacing", float(scene.get("env_spacing", 15.0)))
    render.setdefault("head_height_m", float(scene.get("head_height_m", 1.5)))
    render.setdefault("multi_env_output_dir", scene.get("multi_env_output_dir") or default_multi_env_dir(num_envs))
    return cfg, physics, render, scene, num_envs


#: The one backend name that fuses physics and render into a single process.
#: Both ``backends.physics`` and ``backends.render`` must be set to it -- see
#: ``_is_fused``.
FUSED_BACKEND_NAME = "isaac_fused"


def _is_fused(physics: dict, render: dict) -> bool:
    """Require physics and render to select the fused backend together.

    Its render bridge reads frames from the physics process; mixing either
    side with another backend cannot provide a valid image stream."""
    p = str(physics.get("backend", ""))
    r = str(render.get("backend", ""))
    if p == FUSED_BACKEND_NAME and r == FUSED_BACKEND_NAME:
        return True
    if p == FUSED_BACKEND_NAME or r == FUSED_BACKEND_NAME:
        raise ValueError(
            f"backends.physics={p!r} / backends.render={r!r}: "
            f"{FUSED_BACKEND_NAME!r} fuses physics and rendering into ONE "
            f"Isaac Sim process, so BOTH must be set to it. Set the other one "
            f"to {FUSED_BACKEND_NAME!r} too, or pick a non-fused pair "
            f"(e.g. physics=newton_mujoco_warp + render=isaac_tiled)."
        )
    return False


def _render_backend_supports_randomization(name: str) -> bool:
    """Whether ``name`` honours ``render_randomization``.

    Render bridges declare it as a named constructor argument
    (``bridge_supports_kwarg``). The fused backend is not a registered render
    bridge -- one object plays engine and bridge -- so it is checked on its
    own constructor the same way.
    """
    if name == FUSED_BACKEND_NAME:
        import inspect  # noqa: PLC0415

        from rlinf_geniesim.physics.backends.isaac_fused import FusedIsaacBackend  # noqa: PLC0415

        return "render_randomization" in inspect.signature(FusedIsaacBackend.__init__).parameters
    return bridge_supports_kwarg(name, "render_randomization")


def _build_fused_backend(physics: dict, render: dict, scene: dict, num_envs: int, *, reset_settle_time_s: float = 1.0):
    """Create one backend that owns both physics and rendering.

    Forward the same scene, physics, camera and randomization configuration
    used by the split backend path; a fused handler shares this one object."""
    kwargs = {k: v for k, v in physics.items() if k not in ("backend", "num_envs")}
    # Render-owned keys, explicit: the layout/camera side of the fused object.
    for key, default in (("cameras", []), ("render_hz", 30.0), ("env_spacing", 15.0), ("multi_env_output_dir", "")):
        kwargs[key] = render.get(key, default)
    # Only present when a render-side dimension is enabled (see
    # _prepare_backends_task_cfg), so the default construction is unchanged.
    if "render_randomization" in render:
        kwargs["render_randomization"] = render["render_randomization"]
    # The node settles inside its own RESET command, before rendering the
    # reset frame, so it is told the task's settle time (see build_handler).
    kwargs["reset_settle_time_s"] = float(reset_settle_time_s)
    return make_engine(
        FUSED_BACKEND_NAME,
        scene_yaml=scene["scene_yaml"],
        robot_urdf=scene.get("robot_urdf", ""),
        num_envs=num_envs,
        **kwargs,
    )


def build_handler(backends: dict, task: dict):
    """Build physics, rendering and a handler without creating a transport.

    Task reloads retain the transport so the pending client request survives."""
    cfg, physics, render, scene, num_envs = _split_cfg_pieces(backends, task)
    # Derive settle timing from the same physics configuration used by the engine.
    policy_step_dt = _policy_step_dt(physics)
    reset_settle_time_s = float(task.get("reset_settle_time_s", 1.0))
    fused = _is_fused(physics, render)

    engine = bridge = None
    try:
        if fused:
            # FUSED backend: PhysX physics and Isaac tiled rendering run in ONE
            # Isaac Sim process, so ONE object plays both roles, handed to the
            # handler as the engine AND the bridge (see
            # docs/ISAAC_FUSED_PHYSX_DESIGN.md). ``handler.close()`` calls
            # ``close()`` on it twice (once per role); the VecPhysicsBackend ABC
            # already requires close() to be idempotent, so that is in contract.
            #
            # The fused node settles inside its RESET command and renders the
            # reset frame after that settle, so the handler must not settle
            # again -- a second settle would double the reset time and return an
            # image that predates the returned state.
            engine = bridge = _build_fused_backend(
                physics, render, scene, num_envs, reset_settle_time_s=reset_settle_time_s
            )
            engine.start()
        else:
            engine = make_engine(
                physics.pop("backend"),
                scene_yaml=scene["scene_yaml"],
                robot_urdf=scene.get("robot_urdf", ""),
                **physics,
            )
            bridge = make_bridge(
                render.pop("backend"),
                cameras=render.get("cameras", []),
                num_envs=num_envs,
                render_hz=render.get("render_hz", 30.0),
                **{k: v for k, v in render.items() if k not in ("cameras", "render_hz")},
            )
            engine.start()
            bridge.start()
        dims = _infer_sim_dims(cfg, engine, bridge)
        # Benchmark success checkers are configured per environment.
        return GenieSimHandler(
            engine,
            bridge,
            dims,
            eval_cfg=task.get("eval"),
            reset_settle_time_s=0.0 if fused else reset_settle_time_s,
            policy_step_dt=policy_step_dt,
            eval_interval=int(task.get("eval_interval", 1) or 1),
        )
    except Exception:
        for backend in (bridge, engine) if bridge is not engine else (engine,):
            if backend is not None:
                with contextlib.suppress(Exception):
                    backend.close()
        raise


def _build_reload_engine(backends: dict, task: dict):
    """Build a fresh engine and stage while retaining the existing renderer.

    Use the same scene assembly and backend configuration as build_handler.
    Start physics and return the merged configuration, engine and stage path.
    Close the new engine on failure so the caller can perform a cold rebuild."""
    cfg, physics, render, scene, _num_envs = _split_cfg_pieces(backends, task)
    engine = make_engine(
        physics.pop("backend"), scene_yaml=scene["scene_yaml"], robot_urdf=scene.get("robot_urdf", ""), **physics
    )
    try:
        engine.start()
        stage_manifest = render.get("stage_manifest", "")
        if not stage_manifest:
            raise ValueError(
                "_build_reload_engine: new cfg carries no render.stage_manifest "
                "— cannot hot-swap the render node onto it"
            )
        return cfg, engine, stage_manifest
    except Exception:
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            pass
        raise


def _requested_render_width(init_cfg: dict) -> Optional[int]:
    """``observation.render_width`` — the client telling the server "this is all
    the horizontal resolution I actually consume, don't render more".

    Only the client knows its policy's input size, and every pixel rendered
    above that is paid for again at encode and transfer time, so it is worth
    letting the client ask for less than the benchmark's native resolution.

    Absent or ``null`` means native resolution. Validated HERE, outside
    ``_resolve_task_cameras``'s ``except Exception`` — in there a bad value
    would be swallowed into the "benchmark camera_list unavailable" fallback
    and the run would quietly render at native instead of raising.
    """
    raw = (init_cfg.get("observation", {}) or {}).get("render_width")
    if raw is None:
        return None
    width = int(raw)
    if width <= 0:
        raise ValueError(f"observation.render_width must be positive, got {raw!r}")
    return width


def _resolve_task_cameras(init_cfg: dict) -> list:
    """Resolve benchmark cameras with an optional observation.render_width override.

    Unknown tasks use cameras supplied in the init payload. Known tasks use
    their benchmark camera list so observations retain the training ordering."""
    task = init_cfg.get("task", {})
    cameras = (init_cfg.get("observation", {}) or {}).get("cameras", []) or []
    obs_width = _requested_render_width(init_cfg)
    try:
        from rlinf_geniesim.task_loader import resolve_task_cameras  # noqa: PLC0415

        _bench_cams = resolve_task_cameras(task.get("task_name", ""), obs_width=obs_width)
        if _bench_cams:
            cameras = _bench_cams
            print(
                f"[sim_server] cameras from benchmark camera_list: "
                f"{[c['name'] for c in cameras]} @ "
                f"{[tuple(c['image_hw']) for c in cameras]}"
                f"{'' if obs_width is None else f' (render_width={obs_width})'}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[sim_server] benchmark camera_list unavailable ({exc!r}); using " f"observation.cameras ({len(cameras)})",
            flush=True,
        )
    # Debug freecam (env-gated): append a world-anchored look-at camera so the
    # tiled render also yields a third-person view of env_0's robot+table.
    # GENIE_DEBUG_FREECAM="ex,ey,ez:tx,ty,tz" (env-local offsets). Adds one camera
    # to num_cameras + the tiled buffer (extract as the LAST cam index).
    _fc = os.environ.get("GENIE_DEBUG_FREECAM", "").strip()
    if _fc:
        try:
            eye_s, tgt_s = _fc.split(":")
            eye = [float(x) for x in eye_s.split(",")]
            tgt = [float(x) for x in tgt_s.split(",")]
            hw = (cameras[0].get("image_hw") if cameras else None) or [400, 640]
            cameras = list(cameras) + [
                {"name": "freecam", "image_hw": list(hw), "freecam_eye": eye, "freecam_target": tgt}
            ]
            print(
                f"[sim_server] debug freecam appended eye={eye} target={tgt} " f"(cam idx {len(cameras) - 1})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[sim_server] GENIE_DEBUG_FREECAM parse failed ({exc!r})", flush=True)
    return cameras


def _render_dims_from_init(init_cfg: dict) -> Optional[tuple[int, int, int, int, tuple]]:
    """``(num_envs, num_cameras, image_h, image_w, cam_heights)`` computed
    from a task-init payload ALONE — no engine/bridge is built or started,
    no stage manifest is written. Used ONLY by the reload hot-swap
    applicability gate (``_hotswap_applicable``) to compare a candidate
    reload's obs shape against the CURRENTLY-running handler's ``SimDims``
    without booting anything. Returns ``None`` on ANY uncertainty — the gate
    then conservatively refuses the hot path (see its docstring)."""
    try:
        be = dict((init_cfg or {}).get("backends") or {})
        num_envs = int(be.get("num_envs", 1))
        cameras = _resolve_task_cameras(init_cfg)
        num_cameras, h, w, cam_heights = _camera_render_dims(cameras)
        return num_envs, num_cameras, h, w, cam_heights
    except Exception:  # noqa: BLE001
        return None


def _hotswap_applicable(old_handler, new_cfg: dict) -> bool:
    """Return whether the live renderer can safely load the next task stage.

    Requires a live Isaac tiled renderer with matching environment count,
    camera count, image dimensions and per-camera heights."""
    try:
        old_bridge = getattr(old_handler, "_bridge", None)
        old_dims = getattr(old_handler, "_dims", None)
        if old_bridge is None or old_dims is None:
            return False
        if getattr(old_bridge, "BACKEND_NAME", None) != IsaacTiledBackend.BACKEND_NAME:
            return False
        be = dict((new_cfg or {}).get("backends") or {})
        if str(be.get("render", "")) != IsaacTiledBackend.BACKEND_NAME:
            return False
        proc_alive = getattr(old_bridge, "render_proc_alive", None)
        if proc_alive is None or not proc_alive():
            return False
        new_render_dims = _render_dims_from_init(new_cfg)
        if new_render_dims is None:
            return False
        new_num_envs, new_num_cams, new_h, new_w, new_cam_heights = new_render_dims
        return (
            new_num_envs == int(old_dims.num_envs)
            and new_num_cams == int(old_dims.num_cameras)
            and new_h == int(old_dims.image_h)
            and new_w == int(old_dims.image_w)
            and tuple(new_cam_heights) == tuple(old_dims.cam_heights)
        )
    except Exception:  # noqa: BLE001
        return False


# Parallelism may change between tasks without changing the locked backend.
_BACKENDS_LOCK_EXEMPT = frozenset({"num_envs"})


def _lock_view(backends: dict) -> dict:
    return {k: v for k, v in dict(backends or {}).items() if k not in _BACKENDS_LOCK_EXEMPT}


def check_backends_locked(locked: Optional[dict], incoming: dict) -> None:
    """Reject changes to the backend configuration after the first declaration."""
    if locked is None:
        return
    old, new = _lock_view(locked), _lock_view(incoming)
    if old == new:
        return
    diffs = []
    for k in sorted(set(old) | set(new)):
        if old.get(k) != new.get(k):
            diffs.append(f"{k}: {old.get(k)!r} -> {new.get(k)!r}")
    raise ValueError(
        "backends are fixed for this server's lifetime and cannot be changed "
        f"by init; restart the sim server to change engines. Changed: {'; '.join(diffs)}"
    )


def _deep_merge_cfg(base: dict, over: dict) -> dict:
    """Recursively merge mappings; replace lists and scalars without mutating inputs."""
    out = copy.deepcopy(base or {})
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_cfg(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _merge_init_onto_active(active: Optional[dict], incoming: dict) -> dict:
    """Apply explicit init fields while preserving omitted active settings.

    Before the first init there is no active configuration to inherit."""
    if not active:
        return copy.deepcopy(incoming or {})
    return _deep_merge_cfg(active, incoming or {})


def _attach_backends_lock(transport, live: Optional[dict] = None) -> None:
    """Merge each init payload with the active configuration and lock backends.

    Validate before replacing the active configuration or rebuilding a handler,
    so rejected requests leave the running task intact. The readback provider
    uses the same configuration and the current handler from ``live``."""
    _lock: dict = {"backends": None}
    _active: dict = {"cfg": None, "randomization": None}
    _pending: dict = {"cfg": None, "randomization_before": None}

    def _validate_init(payload: dict) -> dict:
        merged = _merge_init_onto_active(_active["cfg"], payload or {})
        incoming = dict(merged.get("backends") or {})
        check_backends_locked(_lock["backends"], incoming)
        _validate_task_pairing(merged.get("task") or {})
        _pending["cfg"] = copy.deepcopy(merged)
        _pending["randomization_before"] = last_build_randomization()
        return merged

    def _commit_init() -> None:
        # Build, warmup and the init reply must succeed before settings become active.
        cfg = _pending["cfg"]
        if cfg is None:
            return
        incoming = dict(cfg.get("backends") or {})
        if _lock["backends"] is None and _lock_view(incoming):
            _lock["backends"] = incoming
        _active["cfg"] = cfg
        recorded = last_build_randomization()
        # A supplied manifest does not produce a new build record.
        _active["randomization"] = copy.deepcopy(recorded) if recorded is not _pending["randomization_before"] else None
        _pending["cfg"] = None
        _pending["randomization_before"] = None

    def _randomization_readback_provider() -> dict:
        """Return requested, resolved and applied randomization state.

        Use the successful init payload and its stage-build snapshot for
        requested/resolved values and build-time plans, and the active
        render bridge for last-reset plans. ``None`` means unavailable or not yet
        applied; an empty mapping means resolved but disabled."""
        cfg = _active["cfg"] or {}
        requested = dict(cfg.get("randomization") or {})
        # The render section can always be computed: it only has two layers ("explicit
        # yaml > built-in default"), doesn't depend on the eval task, and doesn't need a
        # stage to have been built.
        render = RandomizationSpec.normalize_render(requested.get("render"))
        recorded = _active["randomization"]
        _res = (recorded or {}).get("resolved") or {}
        physics = dict(_res["physics"]) if "physics" in _res else None
        sensor = dict(_res["sensor"]) if "sensor" in _res else None
        build_time = (recorded or {}).get("build_time_plans") if recorded else None

        # The render-side bridge hangs off the currently active handler
        # (handler._bridge). ``live`` being None, or the handler not having been
        # built yet, both count as "no active handler" -- the same "don't know" as
        # a bridge that doesn't support this or hasn't reset yet.
        handler = (live or {}).get("handler")
        bridge = getattr(handler, "_bridge", None) if handler is not None else None
        render_plans = getattr(bridge, "_last_render_plans", None) if bridge is not None else None
        if render_plans:  # None (unsupported) and {} (never reset yet) both fall to else
            last_reset = {
                "generation": getattr(bridge, "_randomize_generation", None),
                "per_env": {str(env_id): plan.to_dict() for env_id, plan in render_plans.items()},
            }
        else:
            last_reset = None

        return {
            "requested": requested,
            "resolved": {"physics": physics, "sensor": sensor, "render": render},
            "resolved_available": recorded is not None,
            "applied": {"build_time": build_time, "last_reset": last_reset},
        }

    transport._init_validator = _validate_init
    transport._init_committer = _commit_init
    _rr_setter = getattr(transport, "set_randomization_readback_provider", None)
    if _rr_setter is not None:
        _rr_setter(_randomization_readback_provider)


def _all_task_instances(sub_task_name: str) -> list:
    """List task instances with a scene file, or return [] if unavailable.

    This supplies the full superset when task.superset_instances is omitted.
    """
    try:
        from rlinf_geniesim.task_loader import _benchmark_config_root  # noqa: PLC0415

        task_dir = _benchmark_config_root() / "llm_task" / sub_task_name
        return sorted(
            int(p.name) for p in task_dir.iterdir() if p.is_dir() and p.name.isdigit() and (p / "scene.usda").exists()
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[sim_server] WARN: could not enumerate instances for "
            f"sub_task_name={sub_task_name!r} ({exc!r}); superset_instances "
            f"defaults to [] (no cross-instance switching)",
            flush=True,
        )
        return []


def _resolve_superset_name_alias(sub_task_name: str, superset: list) -> dict:
    """``{iid_str: {member_prim: slot_name}}`` for a superset instance list --
    the deduped-by-asset analog of ``_resolve_name_alias`` (which serves the
    single-topology-group path). Never raises: on failure this degrades to an
    empty alias table, but ALWAYS logs loudly first -- a silently-empty table
    here is exactly the failure mode ``name_alias`` exists to prevent (checker
    never lights up on non-build superset members, and nothing else would say
    why)."""
    from rlinf_geniesim.task_loader import resolve_superset_instances

    try:
        return resolve_superset_instances(sub_task_name, list(superset))["name_alias"]
    except Exception as exc:  # noqa: BLE001
        # All superset members use deduplicated slot names, including the build instance.
        print(
            f"[sim_server] superset name_alias unavailable ({exc!r}); "
            f"checker will not see ANY superset member's objects (post-"
            f"dedup even the build instance has only slot names, not its "
            f"own raw prim names) -- every env's score will stay at 0",
            flush=True,
        )
        return {}


def _resolve_name_alias(task: dict, instances: list) -> dict:
    """``{iid_str: {member_prim: build_prim}}`` for the task's topology group,
    or ``{}`` when it can't be resolved (missing benchmark data in a bare dev
    setup). Never raises: a missing alias degrades to the pre-existing
    behavior (checker sees only the build instance's names), it must not take
    the whole server down."""
    from rlinf_geniesim.task_loader import resolve_instance_group

    try:
        return resolve_instance_group(task["sub_task_name"], int(instances[0]))["name_alias"]
    except Exception as exc:  # noqa: BLE001
        print(
            f"[sim_server] name_alias unavailable ({exc!r}); "
            f"in-group instance switching will not re-alias checker names",
            flush=True,
        )
        return {}


_TASK_CATALOG_CACHE: Optional[dict] = None


def _build_task_catalog() -> dict:
    """Read released task pairings, instance IDs and instructions from benchmark files.

    The catalog is independent of the loaded handler. Switchability is computed
    separately, only when metadata for one specific task is requested."""
    global _TASK_CATALOG_CACHE
    if _TASK_CATALOG_CACHE is not None:
        return _TASK_CATALOG_CACHE

    from rlinf_geniesim.teleop.eval_hud import InstructionSampler  # noqa: PLC0415

    pairings: dict = {}
    for entries in _build_suite_catalog()["suites"].values():
        for entry in entries:
            pairings.setdefault(entry["sub_task_name"], []).append(
                {"robot": entry["robot"], "task_name": entry["task_name"]}
            )

    sub_tasks: dict = {}
    for sub_task_name, task_names in pairings.items():
        instance_ids = _all_task_instances(sub_task_name)
        instructions_by_instance: dict = {}
        for iid in instance_ids:
            table = InstructionSampler(sub_task_name, iid).table
            instructions_by_instance[iid] = [{"id": tid, "text": text} for tid, text in table]

        sub_tasks[sub_task_name] = {
            "task_names": task_names,
            "instance_ids": instance_ids,
            "instructions_by_instance": instructions_by_instance,
        }

    _TASK_CATALOG_CACHE = {"sub_tasks": sub_tasks}
    return _TASK_CATALOG_CACHE


_SUITE_CATALOG_CACHE: Optional[dict] = None

_RELEASE_SUITES = ("g2op_if", "g2op_manip", "g2op_spatial")
_EXCLUDED_RELEASE_TASKS = frozenset({"sorting_packages", "sorting_packages_continuous"})


def _benchmark_suite_config_root():
    """Locate benchmark suite configs, distinct from benchmark/config task data."""
    import geniesim_benchmark  # noqa: PLC0415

    return Path(geniesim_benchmark.__file__).resolve().parent / "config"


def _build_suite_catalog() -> dict:
    """Read the released G2op task pairings from their official suite configs.

    Malformed selected files raise instead of producing an incomplete catalog.
    Unrelated benchmark suites do not participate in this server's release.
    """
    global _SUITE_CATALOG_CACHE
    if _SUITE_CATALOG_CACHE is not None:
        return _SUITE_CATALOG_CACHE

    import yaml  # noqa: PLC0415

    cfg_dir = _benchmark_suite_config_root()
    suites: dict = {}
    for path in sorted(cfg_dir.glob("*.yaml")):
        suite_key = next((suite for suite in _RELEASE_SUITES if path.stem.startswith(suite + "_")), None)
        if suite_key is None:
            continue
        if path.stem.removeprefix(suite_key + "_") in _EXCLUDED_RELEASE_TASKS:
            continue
        robot, category = suite_key.split("_", 1)
        try:
            bench = (yaml.safe_load(path.read_text()) or {}).get("benchmark") or {}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"benchmark suite config {path} is unparseable ({exc!r}); "
                f"task_metadata refuses to serve a suite catalog that is "
                f"missing entries"
            ) from exc
        task_name, sub_task_name = bench.get("task_name"), bench.get("sub_task_name")
        if not task_name or not sub_task_name:
            raise RuntimeError(
                f"benchmark suite config {path} names no task_name/"
                f"sub_task_name (got {task_name!r}/{sub_task_name!r}); "
                f"task_metadata refuses to serve a suite catalog that is "
                f"missing entries"
            )
        if sub_task_name in _EXCLUDED_RELEASE_TASKS:
            continue
        suites.setdefault(suite_key, []).append(
            {
                "config": path.stem,
                "robot": robot,
                "category": category,
                "task_name": str(task_name),
                "sub_task_name": str(sub_task_name),
            }
        )

    _SUITE_CATALOG_CACHE = {"suites": suites}
    return _SUITE_CATALOG_CACHE


#: Physics backends that run PhysX rather than a newton solver. Kept as backend
#: NAMES rather than read off the classes (``UPSTREAM_SOLVER`` is None for both)
#: because this has to answer before any engine object exists — the superset
#: scale gate runs while the stage is still being assembled.
_PHYSX_BACKENDS = frozenset({"isaac_physx", "isaac_fused"})


def _superset_gate_applies(physics_backend: str) -> bool:
    """Does the superset scale gate's cost model describe ``physics_backend``?

    False for the PhysX backends (``isaac_physx`` / ``isaac_fused``), which run a
    different solver entirely: on a 4090, a superset the gate refuses outright
    (42 deduped free bodies, nv=298) costs isaac_fused 26 ms per physics tick with
    rendering included, against the 166 ms the gate's mujoco-warp cost model
    predicts for it.

    Unknown/newton-family backends keep the gate. featherstone and avbd share
    the newton engine node and have no measurements of their own, so the
    conservative answer for them is today's behaviour, not an assumption that
    they scale like PhysX.
    """
    return str(physics_backend or "").strip() not in _PHYSX_BACKENDS


def _compute_switchable_instance_ids(sub_task_name: str, physics_backend: str = "") -> list:
    """Return instances reachable by reset for the requested task and engine.

    Test the full instance set against the backend's superset scale gate.
    If it exceeds the gate, return only the first instance's topology group.
    This query is independent of the active handler and recomputes the gate
    on each call; PhysX backends are exempt from the Newton scale limit."""
    from rlinf_geniesim.task_loader import (  # noqa: PLC0415
        build_superset_stage_manifest,
        resolve_instance_group,
    )

    instance_ids = _all_task_instances(sub_task_name)
    if not instance_ids:
        return []

    task_name = _build_task_catalog()["sub_tasks"][sub_task_name]["task_names"][0]["task_name"]

    import tempfile  # noqa: PLC0415

    try:
        with tempfile.TemporaryDirectory(prefix="rlinf_switchable_gate_") as td:
            out_path = str(Path(td) / f"{sub_task_name}_gate.json")
            build_superset_stage_manifest(
                task_name,
                sub_task_name,
                instance_ids,
                cameras=[],
                out_path=out_path,
                author_render=False,
                # Switchability is a property of (task, engine): the gate below only
                # describes mujoco-warp, so reporting a task as un-switchable on a
                # PhysX server that would happily load it would contradict the very
                # next init.
                allow_oversize=not _superset_gate_applies(physics_backend),
            )
        return instance_ids
    except ValueError:
        # Over threshold: only the base instance's topology group is
        # actually switchable without a process restart.
        return list(resolve_instance_group(sub_task_name, instance_ids[0])["group_instances"])


def _task_metadata_provider(sub_task_name: Optional[str] = None, physics_backend: str = "") -> dict:
    """Return the task directory and official benchmark suite pairings.

    When sub_task_name is supplied, add its switchable_instance_ids using the
    requested physics backend. Copy that entry so request-specific data does
    not mutate the cached directory. No active handler is required."""
    catalog = _build_task_catalog()
    suites = _build_suite_catalog()["suites"]
    if not sub_task_name:
        # Return fresh directory mappings so callers cannot replace cached entries.
        return {"sub_tasks": dict(catalog["sub_tasks"]), "suites": dict(suites)}
    sub_tasks = dict(catalog["sub_tasks"])
    if sub_task_name not in sub_tasks:
        raise KeyError(f"unknown sub_task_name {sub_task_name!r}; known: {sorted(sub_tasks)}")
    entry = dict(sub_tasks[sub_task_name])
    # ``physics_backend`` decides whether the superset scale gate applies, and it
    # comes from the request because nothing else here can know it: the client
    # declares engines via set_backends (client-local, sends nothing) and queries
    # metadata before init, so the server has no engine information yet. Empty
    # means "not declared" and applies the conservative scale gate.
    entry["switchable_instance_ids"] = _compute_switchable_instance_ids(sub_task_name, physics_backend)
    sub_tasks[sub_task_name] = entry
    return {"sub_tasks": sub_tasks, "suites": dict(suites)}


def _attach_task_metadata_provider(transport) -> None:
    """Attach the load-independent metadata provider when the transport supports it."""
    setter = getattr(transport, "set_task_metadata_provider", None)
    if setter is not None:
        setter(_task_metadata_provider)


def _official_pairings(sub_task_name: str) -> list:
    """Return released (suite_key, task_name) pairs for validation and errors."""
    return sorted(
        {
            (suite, entry["task_name"])
            for suite, entries in _build_suite_catalog()["suites"].items()
            for entry in entries
            if entry["sub_task_name"] == sub_task_name
        }
    )


def _validate_task_pairing(task: dict) -> None:
    """Require a released task and its official background/robot pairing."""
    sub = str(task.get("sub_task_name") or "")
    task_name = str(task.get("task_name") or "")
    pairings = _official_pairings(sub)
    if not pairings:
        raise ValueError(
            f"sub_task_name={sub!r} is outside this server's release task catalog. "
            f"Use client.task_metadata()['suites'] for supported tasks."
        )
    allowed = sorted({name for _suite, name in pairings})
    hint = ", ".join(f"{name} ({suite})" for suite, name in pairings)
    if task_name not in allowed:
        raise ValueError(
            f"task.task_name={task_name!r} is not a released background/robot pairing "
            f"for sub_task_name={sub!r}. Expected one of {allowed}; official suite "
            f"pairings: {hint}. Use client.task_metadata()['suites'] for supported pairings."
        )


def _default_stage_manifest_out_path(task: dict) -> str:
    """The stage_manifest ``out_path`` ``_prepare_backends_task_cfg`` builds at
    when ``manifest.out_path`` is not explicitly given -- factored out so
    ``_randomization_readback_provider`` can independently but IDENTICALLY
    re-derive the same path from ``_active["cfg"]`` to find
    ``build_stage_manifest``'s randomization readback sidecar
    (``task_loader.randomization_build_readback_path``). Two
    separately-hardcoded copies of this formula could drift apart and make the
    readback provider silently look in the wrong directory."""
    return str(Path(default_stage_dir()) / f"rlinf_manifest_{task.get('sub_task_name', 'task')}.json")


def _prepare_backends_task_cfg(init_cfg: dict) -> tuple[dict, dict]:
    """Convert an init payload into backend and task configurations.

    Build the requested manifest and derive camera, instance and reward data
    from the benchmark. Missing backend declarations and invalid task pairings
    are rejected before a handler is created."""
    task = init_cfg.get("task", {})
    be = dict(init_cfg.get("backends", {}))
    # Name missing backend declarations before indexing the configuration.
    _missing = [k for k in ("physics", "render") if not be.get(k)]
    if _missing:
        raise ValueError(
            f"task-init requires backends.{'/'.join(_missing)}; no active backend "
            f"configuration is available to inherit. Call client.set_backends({{...}}) "
            f"before env.init()."
        )
    _validate_task_pairing(task)
    rates = init_cfg.get("rates", {}) or {}
    # Camera resolution (benchmark camera_list preferred over observation.cameras,
    # + opt-in debug freecam) is shared with the reload hot-swap applicability
    # gate — see ``_resolve_task_cameras``.
    cameras = _resolve_task_cameras(init_cfg)
    manifest_cfg = init_cfg.get("manifest", {}) or {}
    # Known tasks own their background requirement; reject obsolete override locations.
    for _sect, _cfg in (("manifest", manifest_cfg), ("task", task)):
        if "use_background" in _cfg:
            raise ValueError(
                f"{_sect}.use_background is unsupported. TASK_NEEDS_BACKGROUND controls "
                f"known tasks; manifest.use_background_default applies only to unlisted tasks."
            )
    if "use_background_default" in task:
        raise ValueError("use_background_default belongs in manifest.use_background_default, not task.")
    multi_env_cfg = init_cfg.get("multi_env", {}) or {}
    scene_cfg = init_cfg.get("scene", {}) or {}
    instances = task.get("instances", [0]) or [0]
    # Omitted superset_instances selects all instances; an explicit empty list opts out.
    superset_is_default = "superset_instances" not in task
    if not superset_is_default:
        superset = list(task["superset_instances"] or [])
    else:
        superset = _all_task_instances(task.get("sub_task_name", ""))

    # 1. stage_manifest: runtime-built from the benchmark task, unless a static
    #    path is supplied (lets you run sooner with a hand-built manifest).
    if manifest_cfg.get("path"):
        stage_manifest = manifest_cfg["path"]
        if superset and superset_is_default:
            # A static manifest cannot guarantee the requested superset and checker aliases.
            raise ValueError(
                f"manifest.path={stage_manifest!r} is a static stage manifest, "
                f"which cannot back the DEFAULT full superset "
                f"({len(superset)} instances, implied by the ABSENT "
                f"task.superset_instances key) — the stage would load a single "
                f"instance while eval claimed the whole superset, and a later "
                f"cross-instance reset would silently switch nothing. Say which "
                f"you mean: task.superset_instances: [] (this run does one "
                f"instance per process), or an explicit list if that static "
                f"manifest really IS a superset build, or drop manifest.path "
                f"and let the server build the superset at runtime."
            )
        # Static manifests may have instance/slot layouts different from the independently computed aliases.
        print(
            f"[sim_server] WARN: manifest.path={stage_manifest!r} is a "
            f"static stage manifest; name_alias is independently "
            f"recomputed from "
            + (f"superset_instances={superset}" if superset else f"instances[0]={instances[0]}")
            + f" and may not match the instance_poses this manifest was "
            f"actually built from",
            flush=True,
        )
    else:
        # The manifest parent also owns scene/config/render layers and must be server-local.
        out_path = manifest_cfg.get("out_path", _default_stage_manifest_out_path(task))
        # Author the render layer only when rendering is on; render: none stays
        # physics-only (empty render_layer -> null bridge).
        render_on = str(be.get("render", "none")).lower() not in ("", "none", "null")
        # Use the task-specific background requirement, falling back only for unknown tasks.
        use_bg, use_bg_src = resolve_use_background(
            str(task.get("sub_task_name", "")), bool(manifest_cfg.get("use_background_default", False))
        )
        if superset:
            # The superset builder already writes instance_switch.
            try:
                stage_manifest = build_superset_stage_manifest(
                    task["task_name"],
                    task["sub_task_name"],
                    superset,
                    cameras,
                    out_path,
                    extra_newton_cfg=task.get("newton"),
                    author_render=render_on,
                    use_background=use_bg,
                    randomization=init_cfg.get("randomization"),
                    # Per-env background room assignment: a one-time choice at build
                    # time, using the seed from the top of the task-init payload (not a
                    # reset seed -- this dimension has no reset at this point), only
                    # consumed inside build_stage_manifest when
                    # randomization.render.background.enable is true.
                    num_envs=int(be.get("num_envs", 1)),
                    seed=init_cfg.get("seed"),
                    env_spacing=float(multi_env_cfg.get("env_spacing", 15.0)),
                    # The scale gate only describes mujoco-warp; it must not block the
                    # PhysX backend (see _superset_gate_applies).
                    allow_oversize=not _superset_gate_applies(str(be.get("physics", ""))),
                )
            except ValueError as exc:
                # Report build failure rather than silently loading fewer instances than requested.
                raise ValueError(
                    f"superset build for {task['sub_task_name']!r} over "
                    f"{len(superset)} instance(s) failed: {exc}. This server "
                    f"will NOT quietly fall back to a single-instance stage — "
                    f"that is what makes a later reset(instances=[k]) switch "
                    f"nothing while the checker scores the wrong scene. To run "
                    f"this task one instance per process, say so explicitly: "
                    f"task.superset_instances: [] (and pick the instance via "
                    f"task.instances). To keep switching, pass a smaller "
                    f"task.superset_instances that fits the scale gate."
                ) from exc
        else:
            stage_manifest = build_stage_manifest(
                task["task_name"],
                task["sub_task_name"],
                instances[0],
                cameras,
                out_path,
                extra_newton_cfg=task.get("newton"),
                author_render=render_on,
                use_background=use_bg,
                randomization=init_cfg.get("randomization"),
                num_envs=int(be.get("num_envs", 1)),
                seed=init_cfg.get("seed"),
                env_spacing=float(multi_env_cfg.get("env_spacing", 15.0)),
            )
        # Print BOTH ways, with where the decision came from: "why is there no
        # room in my images" has to be answerable from the log alone — a silent
        # off looks exactly like a broken background asset.
        print(
            f"[sim_server] use_background="
            f"{'on (loading eval_task background room)' if use_bg else 'off'} "
            f"for {task.get('sub_task_name', '?')!r}, from "
            + (
                "the per-task table (task_background.py)"
                if use_bg_src == "table"
                else "manifest.use_background_default (task not in that table)"
            ),
            flush=True,
        )
        if use_bg:
            # Guardrail: full-room backgrounds are ~10m wide; with a small
            # env_spacing the neighbouring envs' room walls slice through each
            # other in the RENDER stage (physics is per-world, unaffected) —
            # looks like the robot/furniture clipping a wall.
            _sp = float((init_cfg.get("multi_env", {}) or {}).get("env_spacing", 15.0))
            _ne = int(be.get("num_envs", 1))
            if _ne > 1 and _sp < 12.0:
                print(
                    f"[sim_server] WARN: use_background with num_envs={_ne} but "
                    f"env_spacing={_sp} — full rooms will visually overlap in the "
                    f"render (neighbour walls cut through). Set "
                    f"multi_env.env_spacing >= 15.",
                    flush=True,
                )

    # Warn when superset ordering changes the initial instance requested by the caller.
    if superset and superset_is_default and int(instances[0]) != int(superset[0]):
        print(
            f"[sim_server] NOTE: task.instances={list(instances)!r} does "
            f"NOT select eval's base instance here -- superset_instances "
            f"defaulted to the full instance list (key absent from "
            f"task-init), so eval.instance_id={int(superset[0])} "
            f"(superset[0]), not instances[0]={int(instances[0])}. Pass "
            f"an explicit task.superset_instances (or reorder so "
            f"instances[0] is first in it) if a specific scene instance "
            f"must be the build. Per-episode resets are unaffected -- "
            f"episode_runner-style explicit instance_ids still win.",
            flush=True,
        )

    # The top-level ``randomization`` section in the task-init payload is the single
    # config source for render-side generalization (currently lights only), the same
    # entry point physics/sensor use (see docs/GENERALIZATION_DESIGN.md).
    #
    # This calls RandomizationSpec.normalize_render, not the full
    # RandomizationSpec.resolve() -- resolve() already runs once inside
    # build_stage_manifest (physics side), and calling it a second time here would
    # risk the physics and render sides silently drifting onto different priority
    # rules. The render dimension only has two layers ("explicit yaml key >
    # built-in default"), and normalize_render is the sole implementation of that
    # layer, so it cannot diverge from what resolve() computes.
    render_randomization = RandomizationSpec.normalize_render((init_cfg.get("randomization") or {}).get("render"))
    # normalize_render fills in the built-in default for every sub-dimension
    # (including enable: false), so the result is never empty -- "non-empty" cannot
    # mean "configured". The real test is whether any dimension's enable is true.
    _enabled_render_dims = sorted(k for k, v in render_randomization.items() if isinstance(v, dict) and v.get("enable"))
    if _enabled_render_dims:
        _render_backend_name = str(be["render"])
        # Whether a bridge supports render_randomization is judged by introspecting
        # its signature (see bridge_supports_kwarg), not by hardcoding a list of
        # bridge names -- a new bridge just needs the parameter in its __init__ to
        # be recognized, with nothing to keep in sync here.
        if not _render_backend_supports_randomization(_render_backend_name):
            _supporters = bridges_supporting_kwarg("render_randomization") + [FUSED_BACKEND_NAME]
            raise ValueError(
                f"randomization.render configures the {_enabled_render_dims} "
                f"dimension(s), but the selected render bridge "
                f"{_render_backend_name!r} does not support render_randomization -- "
                f"its constructor has no such named parameter, only **kwargs, which "
                f"would silently swallow this config: no error, logs look completely "
                f"normal, but the lights (or any other render generalization "
                f"dimension) will simply never change. Bridges that support it: "
                f"{_supporters or '(no registered bridge currently supports it)'}. "
                f"Switch to a bridge that supports it, or turn off these dimensions "
                f"in randomization.render."
            )

    # 2. map the init payload onto build_handler's (backends, task) shape.
    backends_cfg = {
        "physics": {
            "backend": be["physics"],
            "num_envs": be.get("num_envs", 1),
            "physics_hz": rates.get("physics_hz", 120.0),
            "device": be.get("device", "cuda:0"),
            # Contact-solver substeps per physics tick. The mjwarp contact
            # timeconst is 2 ms, so sub_dt must be <= 1 ms — at physics_hz=100
            # that means >= 10 substeps. substeps=1 under-integrates the contact
            # spring -> a 25 Hz limit cycle that never sleeps and slowly SPINS
            # resting objects (blocks visibly rotating). Config-overridable.
            "physics_solver_substep": int(be.get("physics_solver_substep", 10)),
            # Forward inherited solver iterations; zero keeps the engine default.
            "physics_solver_iterations": int(be.get("physics_solver_iterations", 0)),
            # mjwarp constraint-buffer overrides (0 = engine auto-scale by
            # scene body count). Needed e.g. for background rooms, whose many
            # STATIC collision shapes add contact demand without adding bodies.
            "mujoco_njmax": int(be.get("mujoco_njmax", 0)),
            "mujoco_nconmax": int(be.get("mujoco_nconmax", 0)),
            # Lockstep: 1 control step == physics_hz/control_hz ticks, command-
            # driven + deterministic (GENIESIM_LOCKSTEP_TICKS in the engine).
            # control_hz: 0 disables (legacy real-time free-run).
            "control_hz": float(rates.get("control_hz", 30.0)),
        },
        "render": {
            "backend": be["render"],
            "render_hz": rates.get("render_hz", 30.0),
            # Only present when a render generalization dimension is actually on --
            # left out entirely (not an empty dict) when everything is at its
            # default, so an unconfigured caller's behaviour is unchanged. A bridge
            # that can't take this kwarg was already rejected above, so anything
            # reaching here is a confirmed-supporting bridge.
            **({"render_randomization": render_randomization} if _enabled_render_dims else {}),
        },
        # Merge onto defaults so `backend` survives even when `comm` already
        # exists for another key (e.g. main() sets comm.bootstrap_out from
        # --bootstrap-out via setdefault, which would otherwise drop backend).
        "comm": {
            **{"backend": "shm", "bootstrap_out": "/tmp/rlinf_bootstrap_demo.json"},
            **(init_cfg.get("comm") or {}),
        },
    }
    task_cfg = {
        # The generated manifest supplies the engine scene configuration.
        "scene": {
            "scene_yaml": stage_manifest,
            "robot_urdf": "",
            # Render layout consumed by build_handler.
            "env_spacing": float(multi_env_cfg.get("env_spacing", 15.0)),
            "head_height_m": float(multi_env_cfg.get("head_height_m", 1.5)),
            "multi_env_output_dir": multi_env_cfg.get("output_dir", ""),
            # Forward an explicit SHM capacity override; otherwise infer it from the running engine.
            **(
                {"num_bodies_estimate": int(scene_cfg["num_bodies_estimate"])}
                if "num_bodies_estimate" in scene_cfg
                else {}
            ),
            **(
                {"num_objects_estimate": int(scene_cfg["num_objects_estimate"])}
                if "num_objects_estimate" in scene_cfg
                else {}
            ),
        },
        "stage_manifest": stage_manifest,
        "cameras": cameras,
        # Post-reset settle (task.reset_settle_time_s -> build_handler ->
        # GenieSimHandler): sim-seconds to free-run physics after a reset,
        # holding the just-applied init pose, before the checker rebuild /
        # obs render (objects spawn slightly above the table and fall for
        # ~1s — mirrors the official benchmark's realtime reset convergence
        # wait; see GenieSimHandler.reset). Default 1.0 when the key is
        # ABSENT from the task-init payload (applies on the training path
        # too — intentional); 0 disables (today's instant-reset behavior,
        # byte-identical).
        "reset_settle_time_s": float(task.get("reset_settle_time_s", 1.0)),
        # Forward scoring cadence because task_cfg is assembled from explicit keys.
        "eval_interval": int(task.get("eval_interval", 1) or 1),
        # Enable per-environment benchmark rewards and termination unless explicitly disabled.
        "eval": {
            "enabled": bool((init_cfg.get("reward", {}) or {}).get("success_from_checker", True)),
            "task_name": task.get("task_name", ""),
            "sub_task_name": task.get("sub_task_name", ""),
            # Use the superset build instance as the base checker identity.
            "instance_id": int(superset[0]) if superset else int(instances[0]),
            "robot_cfg": task.get("robot_cfg", "G2_omnipicker"),
            # Threaded to the handler so it can round-robin-assign each env to
            # a superset member at reset (empty -> handler behavior unchanged).
            "superset_instances": list(superset),
            # Map benchmark object names to loaded instance or superset slot names.
            "name_alias": (
                _resolve_superset_name_alias(task["sub_task_name"], superset)
                if superset
                else _resolve_name_alias(task, instances)
            ),
        },
    }
    if superset:
        from rlinf_geniesim.task_loader import resolve_superset_instances

        task_cfg["eval"]["table_materials"] = resolve_superset_instances(task["sub_task_name"], list(superset))[
            "table_materials"
        ]
    # Announce disabled scoring to distinguish unscored runs from task failures.
    if not task_cfg["eval"]["enabled"]:
        print(
            "[sim_server] WARN: success checker DISABLED "
            "(reward.success_from_checker=false) — every episode will report "
            "reward 0 and success False, with no other symptom. Remove that "
            "key (it now defaults to true) unless you deliberately want an "
            "unscored run.",
            flush=True,
        )
    _plan_rest_state_bake(task, be, manifest_cfg, stage_manifest, task_cfg)
    return backends_cfg, task_cfg


def _plan_rest_state_bake(task: dict, be: dict, manifest_cfg: dict, stage_manifest: str, task_cfg: dict) -> None:
    """Reset into each instance's baked rest state (``rlinf_geniesim.reset_bake``).

    Only for stages this server built (a static ``manifest.path`` is someone
    else's scene). On a cache hit the baked poses are already in the scene
    config and the settle after a reset drops to ``settle_after_s`` (default
    0): the objects start at rest, so there is nothing to wait for. On a miss
    the plan is left in ``task_cfg["_rest_bake"]`` for ``build_handler_from_init``
    to bake. ``task.rest_state_bake.enable: false`` keeps the authored poses
    and the configured settle.
    """
    bake_cfg = dict(task.get("rest_state_bake") or {})
    if not bake_cfg.get("enable", True) or manifest_cfg.get("path"):
        return
    from rlinf_geniesim import reset_bake  # noqa: PLC0415

    plan = reset_bake.plan(
        stage_manifest,
        physics=str(be.get("physics", "")),
        solver_substep=be.get("physics_solver_substep"),
        settings=bake_cfg,
        log=lambda m: print(m, flush=True),
    )
    if plan is None:
        return
    plan["settings"] = bake_cfg
    plan["settle_before_s"] = task_cfg["reset_settle_time_s"]
    task_cfg["_rest_bake"] = plan
    if plan["status"] == "hit":
        task_cfg["reset_settle_time_s"] = float(bake_cfg.get("settle_after_s", 0.0))
        stats = plan.get("stats") or {}
        moving = sorted(int(i) for i, st in (stats.get("instances") or {}).items() if st.get("rest_s") is None)
        print(
            f"[sim_server] rest-state bake: cache hit ({plan['path']}); resets start at rest, "
            f"reset_settle_time_s {plan['settle_before_s']} -> {task_cfg['reset_settle_time_s']}"
            + (f"; instances still moving when baked: {moving[:10]}" if moving else ""),
            flush=True,
        )


def build_handler_from_init(init_cfg: dict):
    """Build ONLY a handler (engine + bridge, no transport) from a task-init
    payload — the reload rebuild path (``run_reload_server``, below): the
    SAME transport instance keeps serving across generations while only the
    handler is torn down and rebuilt from the ``reload`` verb's task_init
    dict.

    When the task's rest-state bake is not cached yet, the first handler only
    runs the bake; it is then closed and the real one is built on the baked
    poses (see ``_bake_then_build``)."""
    backends_cfg, task_cfg = _prepare_backends_task_cfg(init_cfg)
    plan = task_cfg.get("_rest_bake")
    if plan is not None and plan["status"] == "miss":
        return _bake_then_build(backends_cfg, task_cfg, plan)
    return build_handler(backends_cfg, task_cfg)


def _bake_then_build(backends_cfg: dict, task_cfg: dict, plan: dict):
    """Bake every instance's rest state in a throwaway handler, cache it,
    write it into the scene config, then build the real handler on it.

    The throwaway handler runs with no settle -- the bake does its own
    stepping. A bake that fails is reported loudly and the task runs on the
    authored poses with the configured settle, exactly as before.
    """
    from rlinf_geniesim import reset_bake  # noqa: PLC0415

    settings = plan.get("settings") or {}
    control_hz = float((backends_cfg.get("physics") or {}).get("control_hz", 30.0) or 30.0)
    exclude = tuple(plan.get("exclude") or ("table",))
    cfg = reset_bake.BakeConfig(
        cap_s=float(settings.get("cap_s", 1.5)),
        hard_cap_s=float(settings.get("hard_cap_s", 5.0)),
        window_s=float(settings.get("window_s", 0.25)),
        tol_m=float(settings.get("tol_m", 1e-4)),
        control_hz=control_hz,
        exclude=exclude,
    )
    print(
        f"[sim_server] rest-state bake: no cache for this task on {plan['physics']}; baking "
        f"{len(plan['instances'])} instance(s) (first load only)",
        flush=True,
    )
    handler = build_handler(backends_cfg, dict(task_cfg, reset_settle_time_s=0.0))
    try:
        # Bodies the engine does not simulate (Newton's static table) keep their
        # authored pose; their boxes come from the stage USD so the position
        # constraints still see the table top.
        try:
            static_boxes = reset_bake.static_boxes_for(plan, handler._engine)
        except Exception as exc:  # noqa: BLE001 -- the layout just lacks those surfaces
            print(f"[sim_server] WARNING: rest-state bake: no boxes for static bodies ({exc!r})", flush=True)
            static_boxes = {}
        baked = reset_bake.bake_rest_states(
            handler._engine,
            plan["instances"],
            plan["table"],
            cfg,
            log=lambda m: print(m, flush=True),
            static_boxes=static_boxes,
        )
        path = reset_bake.save(plan, baked)
        reset_bake.inject(plan, baked)
    except Exception as exc:  # noqa: BLE001 -- the task still runs, on the authored poses
        print(
            f"[sim_server] WARNING: rest-state bake failed ({exc!r}); running on the authored "
            f"poses with reset_settle_time_s={task_cfg['reset_settle_time_s']}",
            flush=True,
        )
        handler.close()
        return build_handler(backends_cfg, task_cfg)
    handler.close()
    settle = float(settings.get("settle_after_s", 0.0))
    print(
        f"[sim_server] rest-state bake: cached at {path}; rebuilding on the baked poses, "
        f"reset_settle_time_s {task_cfg['reset_settle_time_s']} -> {settle}",
        flush=True,
    )
    return build_handler(backends_cfg, dict(task_cfg, reset_settle_time_s=settle))


def _hotswap_rebuild(
    old_handler,
    new_cfg: dict,
    *,
    prepare_cfg_fn=_prepare_backends_task_cfg,
    build_reload_engine_fn=_build_reload_engine,
):
    """Reuse a live renderer while rebuilding physics for the requested task.

    The old handler releases physics but retains its bridge. Repoint that
    bridge with reload_stage, then attach it to the new handler. On failure,
    close the new physics engine so the caller can perform a cold rebuild."""
    import time as _time

    _t0 = _time.monotonic()
    bridge = old_handler._bridge
    old_handler.close(keep_bridge=True)  # kills the OLD engine only
    _t_close = _time.monotonic()

    new_engine = None
    try:
        backends_cfg, task_cfg = prepare_cfg_fn(new_cfg)
        if (task_cfg.get("_rest_bake") or {}).get("status") == "miss":
            # The bake needs a cold build (it rebuilds on the baked poses);
            # the cold path below takes over after the cleanup.
            raise RuntimeError("rest-state bake not cached for this task yet")
        cfg, new_engine, stage_manifest = build_reload_engine_fn(backends_cfg, task_cfg)
        _t_engine = _time.monotonic()
        bridge.reload_stage(stage_manifest)
        # reload_stage() only rebinds stage/manifest bookkeeping; it does not touch
        # render-side generalization state, which lives on the same bridge object
        # being reused here. bridge._render_randomization is captured once at
        # bridge __init__ and never reassigned elsewhere in this class, so it must
        # be refreshed here from this generation's cfg on every reload -- otherwise
        # every render-side dimension (lights/material/background/camera_pose)
        # would stay frozen at whatever the bridge was first built with.
        # _material_candidates must also be dropped: it caches a material scan
        # keyed by mesh path against the OLD scene's USD file, which the new
        # scene's meshes will not match. The scan is lazy, so dropping it here
        # only costs one re-scan on the next reset that needs it.
        bridge._render_randomization = dict(backends_cfg.get("render", {}).get("render_randomization") or {})
        bridge._material_candidates = None
        bridge._last_render_plans = {}
        _t_stage = _time.monotonic()
        dims = _infer_sim_dims(cfg, new_engine, bridge)
        # Forward the current task configuration on every reload, including scoring cadence.
        policy_step_dt = _policy_step_dt(dict(backends_cfg.get("physics") or {}))
        handler = GenieSimHandler(
            new_engine,
            bridge,
            dims,
            eval_cfg=task_cfg.get("eval"),
            reset_settle_time_s=float(task_cfg.get("reset_settle_time_s", 1.0)),
            policy_step_dt=policy_step_dt,
            eval_interval=int(task_cfg.get("eval_interval", 1) or 1),
        )
        print(
            f"[sim_server] [reload-prof] close_old={_t_close - _t0:.1f}s "
            f"engine_build={_t_engine - _t_close:.1f}s "
            f"stage_swap={_t_stage - _t_engine:.1f}s "
            f"total={_time.monotonic() - _t0:.1f}s",
            flush=True,
        )
        return handler
    except Exception:
        try:
            bridge.close()
        except Exception:  # noqa: BLE001
            pass
        if new_engine is not None:
            try:
                new_engine.close()
            except Exception:  # noqa: BLE001
                pass
        raise


# ---------------------------------------------------------------------------
# Reload lifecycle
# ---------------------------------------------------------------------------


def _log_thread_stacks(header: str) -> None:
    names = {t.ident: t.name for t in threading.enumerate()}
    out = [header]
    for ident, frame in sys._current_frames().items():
        if ident != threading.get_ident():
            out.append(f"Thread {ident:#x} ({names.get(ident, '?')}), most recent call last:")
            out.extend(line.rstrip("\n") for line in traceback.format_stack(frame))
    print("\n".join(out), file=sys.__stderr__, flush=True)


@contextlib.contextmanager
def _build_watchdog(label: str, every_s: float = 60.0, wedged_s: float = 90.0):
    """Log every thread's stack each ``every_s`` while a build runs.

    The stacks are taken under the GIL: faulthandler's periodic dump reads
    them without it and can crash CPython 3.12 (cpython#140815). faulthandler
    is kept as a one-shot fallback that each tick pushes back, so it fires
    only once the GIL has been held for ``wedged_s``.
    """
    wedged_s = max(wedged_s, 1.5 * every_s)
    started = time.monotonic()
    stop = threading.Event()

    def tick() -> None:
        while not stop.wait(every_s):
            faulthandler.dump_traceback_later(wedged_s, file=sys.__stderr__)
            _log_thread_stacks(
                f"[sim_server] still building {label} after " f"{time.monotonic() - started:.0f}s; thread stacks:"
            )

    faulthandler.dump_traceback_later(wedged_s, file=sys.__stderr__)
    ticker = threading.Thread(target=tick, name="build-watchdog", daemon=True)
    ticker.start()
    try:
        yield
    finally:
        stop.set()
        ticker.join()
        faulthandler.cancel_dump_traceback_later()


def _rebuild_until_ready(transport, cfg: Optional[dict], build_handler_fn, live: dict) -> Optional[SimHandler]:
    """Keep answering failed rebuilds (``fail_pending_reload`` +
    ``transport.serve_idle()``) until either a handler builds successfully
    (returns it, with ``live["handler"]`` set to it) or the transport is
    told to stop (``cfg`` is/becomes ``None`` -> returns ``None``).

    ``cfg`` is the task_init to try FIRST (already dequeued from a
    ``serve()``/``serve_idle()`` return by the caller) — deliberately does
    NOT automatically retry that SAME ``cfg`` on failure; it waits for
    ``serve_idle()``'s next return (a fresh client-issued reload, or
    shutdown), per spec: a failed rebuild is reported to the client, not
    silently retried.
    """
    while cfg is not None:
        task_label = (cfg.get("task") or {}).get("sub_task_name", "unknown")
        started = time.monotonic()
        print(
            f"[sim_server] rebuilding {task_label}; if this takes over 60s, "
            "thread stacks will be written to this log",
            flush=True,
        )
        try:
            with _build_watchdog(task_label):
                handler = build_handler_fn(cfg)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[sim_server] rebuild {task_label} failed after " f"{time.monotonic() - started:.1f}s: {exc}",
                flush=True,
            )
            transport.fail_pending_reload(str(exc))
            cfg = transport.serve_idle()
            continue
        print(f"[sim_server] rebuilt {task_label} in " f"{time.monotonic() - started:.1f}s", flush=True)
        live["handler"] = handler
        return handler
    return None


def run_reload_server(
    transport, handler: SimHandler, *, build_handler_fn=build_handler_from_init, live: Optional[dict] = None
) -> None:
    """Serve tasks on one transport until shutdown.

    Tear down the previous handler before a cold rebuild. A failed build or
    initialization replies to the pending request and leaves ZMQ idle for the
    next client request. SHM serves once and propagates errors after cleanup.

    The optional live mapping tracks the current handler for signal cleanup."""
    if live is None:
        live = {}
    live["handler"] = handler
    supports_reload = hasattr(transport, "serve_idle")

    while handler is not None:
        serve_ok = True
        exc_msg = ""
        cfg: Optional[dict] = None
        try:
            cfg = transport.serve(handler)
        except Exception as exc:  # noqa: BLE001
            if not supports_reload:
                # Without reload support, close the handler before propagating the error.
                try:
                    handler.close()
                except Exception:  # noqa: BLE001
                    pass
                live["handler"] = None
                raise
            serve_ok = False
            exc_msg = str(exc)

        # Fall back to a cold rebuild if reusing the renderer fails.

        if serve_ok and supports_reload and cfg is not None and _hotswap_applicable(handler, cfg):
            try:
                new_handler = _hotswap_rebuild(handler, cfg)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[sim_server] render hot-swap failed ({exc!r}) — " f"falling back to full cold reload", flush=True
                )
            else:
                live["handler"] = new_handler
                handler = new_handler
                # Task metadata is independent of the active handler.
                continue

        # Old handler ALWAYS torn down before any new build is attempted —
        # whether serve() returned normally (shutdown/reload) or raised, or a
        # hot-swap attempt above failed partway (its bridge/engine cleanup
        # already ran inside ``_hotswap_rebuild``; closing the old handler
        # again here is a harmless idempotent no-op in that case).
        try:
            handler.close()
        except Exception:  # noqa: BLE001
            pass
        live["handler"] = None
        handler = None

        if not supports_reload:
            break  # one serve() cycle is this transport's whole lifetime.

        if not serve_ok:
            transport.fail_pending_reload(exc_msg)
            cfg = transport.serve_idle()

        handler = _rebuild_until_ready(transport, cfg, build_handler_fn, live)


# ---------------------------------------------------------------------------
# Idle startup waits for the client to select a task.


def build_idle_transport(comm_cfg: dict, live: Optional[dict] = None):
    """Build a ZMQ transport that waits for the first client init.

    SHM requires task dimensions before allocating regions and cannot serve
    this idle startup path. Establish the server tag before task builds so
    generated files remain isolated across server processes."""
    comm = dict(comm_cfg or {})
    backend = comm.get("backend", "zmq")
    # Set the bind-derived server tag before any generated stage or reload-control paths are used.
    tag = set_server_tag_from_bind(str(comm.get("bind", "")))
    print(f"[sim_server] server tag: {tag} (isolates generated files)", flush=True)
    transport = make_transport(
        backend,
        bootstrap_out=comm.get("bootstrap_out", "/tmp/rlinf_bootstrap_demo.json"),
        # pass backend-specific keys through (e.g. zmq: bind / jpeg_quality)
        **{k: v for k, v in comm.items() if k not in ("backend", "bootstrap_out")},
    )
    if not hasattr(transport, "serve_idle"):
        raise ValueError(
            f"comm backend {backend!r} has no serve_idle() — starting with "
            f"no task at all requires a transport that can answer queries "
            f"and accept the first init before any handler exists (today: "
            f"zmq only; see the design doc's structural SHM limitation)"
        )
    _attach_task_metadata_provider(transport)
    # --no-task boot: nothing has ever gone through init yet, so the backends
    # lock starts unset (seed=None) — the client's first init establishes it.
    _attach_backends_lock(transport, live=live)
    return transport


def run_server_from_idle(transport, *, build_handler_fn=build_handler_from_init, live: Optional[dict] = None) -> None:
    """Wait for init, build a handler, and serve reloads until shutdown."""
    if live is None:
        live = {}
    cfg = transport.serve_idle()
    handler = _rebuild_until_ready(transport, cfg, build_handler_fn, live)
    run_reload_server(transport, handler, build_handler_fn=build_handler_fn, live=live)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_transport_from_args(args, live: Optional[dict] = None):
    """Build the idle transport and share the active handler for readback.

    ``--comm-config`` accepts a bare comm mapping or a mapping under ``comm``.
    The first client init supplies the task and creates the handler.
    """
    comm_cfg = {"backend": "zmq", "bind": "tcp://0.0.0.0:5555"}
    if args.comm_config:
        loaded = _load_config(args.comm_config)
        comm_cfg = dict(loaded.get("comm", loaded))
    if args.bootstrap_out:
        comm_cfg["bootstrap_out"] = args.bootstrap_out
    return build_idle_transport(comm_cfg, live=live)


def main() -> int:
    parser = argparse.ArgumentParser(description="GenieSim sim_server using the pluggable backend factory")
    parser.add_argument(
        "--bootstrap-out",
        default="",
        help="Override comm.bootstrap_out: where to write the bootstrap JSON " "for the trainer.",
    )
    parser.add_argument(
        "--comm-config",
        default="",
        help="Comm-only config (JSON/YAML with a top-level 'comm' section, or "
        "a bare comm dict): which transport to bind and how (see "
        "config/comm_zmq.yaml for the shape). The server "
        "ALWAYS starts with no task and waits for the client's first "
        "init — which task runs is the client's call, not a launch "
        "flag. Defaults to zmq on tcp://0.0.0.0:5555 when omitted.",
    )
    args = parser.parse_args()

    # The readback closure and cleanup always see the current handler generation.
    _live: dict = {"handler": None}
    transport = _build_transport_from_args(args, live=_live)
    _cleaned = {"done": False}

    def _cleanup(*_args) -> None:
        if _cleaned["done"]:
            return
        _cleaned["done"] = True
        try:
            live_handler = _live.get("handler")
            if live_handler is not None:
                live_handler.close()  # -> engine.close() + bridge.close() -> killpg
        except Exception:  # noqa: BLE001
            pass
        try:
            transport.close()
        except Exception:  # noqa: BLE001
            pass

    atexit.register(_cleanup)

    def _on_signal(signum, _frame) -> None:
        # Break the serve loop even if SystemExit can't unwind through the
        # blocked semaphore wait (the loop re-checks _stop every 1s timeout).
        try:
            transport._stop = True
        except Exception:  # noqa: BLE001
            pass
        _cleanup()
        raise SystemExit(128 + signum)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _on_signal)

    watchdog = None
    try:
        watchdog = start_owner_watchdog()
        # Wait for the initial task, then serve successive handler generations.
        run_server_from_idle(transport, live=_live)
    finally:
        _cleanup()
        if watchdog is not None:
            watchdog.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
