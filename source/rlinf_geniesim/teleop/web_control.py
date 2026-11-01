# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Queued task, reset and randomization controls for the teleoperation web console.

Only the main loop accesses the request-reply client. Reload pauses stepping
until both initialization and the replacement layout callback succeed."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from rlinf_geniesim.task_pairing import official_task_name


class _Pending:
    """A queued command and its result slot."""

    __slots__ = ("kind", "args", "done", "result")

    def __init__(self, kind: str, args: dict):
        self.kind = kind
        self.args = args
        self.done = threading.Event()
        self.result: dict = {}


class TeleopControl:
    """Queue HTTP requests for execution between simulation steps.

    The reload callback must replace every cached dimension and action index.
    The reset callback receives the response, instance, instruction, text and seed."""

    def __init__(
        self,
        client,
        num_envs: int,
        on_reload: Optional[Callable[[Any, Any, str], None]] = None,
        state_fn: Optional[Callable[[], dict]] = None,
        sub_task_name: str = "",
        instance_id: Optional[int] = None,
        on_reset: Optional[Callable[[Any, int, int, str, int], None]] = None,
    ):
        self._client = client
        self._n = int(num_envs)
        self._on_reload = on_reload
        # After a reset the teleop side must re-seed: the world
        # jumped and the IK reference / hold targets are stale.
        self._on_reset = on_reset
        self._state_fn = state_fn or (lambda: {})
        self._q: list[_Pending] = []
        self._lock = threading.Lock()
        self._catalog: Optional[dict] = None
        self._task_cache: dict[str, dict] = {}
        self.current = {
            "sub_task_name": str(sub_task_name or ""),
            "task_name": "",
            "instance_id": (None if instance_id is None else int(instance_id)),
            "instruction_id": None,
            "seed": -1,
        }
        self.last_error = ""
        # A failed reload or reset can leave the cached layout or hold targets stale.
        # Resume only after a complete reload and layout rebuild.
        self.session_ready = True
        # Live HUD. The copy burned into the picture is too small to read, so
        # the same data is also served at /api/state and rendered large
        # outside it.
        self.hud: dict = {}

    def set_current(self, **kw) -> None:
        """Sync in what teleoperation is running (from CLI start or keyOne
        manual reset)."""
        for k, v in kw.items():
            if k in self.current:
                self.current[k] = v

    def set_hud(self, hud: Optional[dict]) -> None:
        """Called each frame by the teleop main loop, with the same dict
        passed to viewer.push."""
        if hud:
            self.hud = dict(hud)

    # Catalogue queries also go through the queue. They look read-only and
    # safe, but the client is a strict request-reply socket, so a request from
    # the HTTP thread while the main loop is stepping corrupts it. Only a
    # cache hit returns directly on the HTTP thread.

    def catalog(self, refresh: bool = False) -> dict:
        if self._catalog is not None and not refresh:
            return self._catalog
        r = self.submit("catalog", {"refresh": bool(refresh)})
        if not r.get("ok"):
            raise RuntimeError(r.get("error") or "catalog query failed")
        return self._catalog or {}

    def task_info(self, sub: str, refresh: bool = False) -> dict:
        if sub in self._task_cache and not refresh:
            return self._task_cache[sub]
        r = self.submit("task", {"sub": sub, "refresh": bool(refresh)})
        if not r.get("ok"):
            raise RuntimeError(r.get("error") or "task query failed")
        return self._task_cache.get(sub, {})

    # -- command queue (HTTP thread) + execution (main loop) -------------

    def submit(self, kind: str, args: dict, timeout_s: float = 600.0) -> dict:
        p = _Pending(kind, args)
        with self._lock:
            self._q.append(p)
        if not p.done.wait(timeout_s):
            return {"ok": False, "error": f"{kind} timed out after {timeout_s}s"}
        return p.result

    def pending(self) -> bool:
        with self._lock:
            return bool(self._q)

    def drain(self) -> Optional[dict]:
        """Run between two steps. Executes at most one command and returns its
        result."""
        with self._lock:
            p = self._q.pop(0) if self._q else None
        if p is None:
            return None
        try:
            if p.kind == "catalog":
                self._catalog = self._client.task_metadata()
                p.result = {"ok": True}
            elif p.kind == "task":
                sub = str(p.args.get("sub") or "")
                self._fetch_task(sub)
                p.result = {"ok": True}
            elif p.kind == "reload":
                p.result = self._do_reload(**p.args)
            elif p.kind == "reset":
                p.result = self._do_reset(**p.args)
            elif p.kind == "randomization_readback":
                p.result = {"ok": True, **self._client.randomization_readback()}
            else:
                p.result = {"ok": False, "error": f"unknown command {p.kind}"}
        except Exception as exc:  # noqa: BLE001 — a control-plane error must
            # not interrupt teleoperation
            import traceback as _tb

            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[ctl] command {p.kind} failed:\n{_tb.format_exc()}", flush=True)
            p.result = {"ok": False, "error": self.last_error}
        finally:
            p.done.set()
        return p.result

    def _fetch_task(self, sub: str) -> dict:
        """Query and cache ``sub``'s metadata, including its official ``default_task_name``."""
        md = self._client.task_metadata(sub_task_name=sub)
        info = dict((md.get("sub_tasks") or {}).get(sub) or {})
        info["default_task_name"] = official_task_name(md, sub)
        self._task_cache[sub] = info
        return info

    # -- the two verb implementations -------------------------------------

    def _do_reload(
        self,
        sub_task_name: str,
        task_name: str = "",
        num_envs: Optional[int] = None,
        build_instance: Optional[int] = None,
        full_superset: bool = True,
        randomization: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> dict:
        """Switch tasks and pass build-time seed/randomization through to init.

        The full instance list is explicit because init deep-merges the active config."""
        sub = str(sub_task_name or "").strip()
        if not sub:
            return {"ok": False, "error": "missing sub_task_name"}
        n = int(num_envs or self._n)
        backends: dict = {}
        if self._client.backends_declared():
            backends = dict(self._client.get_backends())
        backends["num_envs"] = n
        task: dict = {"sub_task_name": sub}
        info = (self._task_cache.get(sub) or self._fetch_task(sub)) if (full_superset or not task_name) else {}
        # Init merges onto the active config, so an omitted task_name keeps the
        # previous task's background.
        task_name = task_name or info.get("default_task_name") or ""
        if task_name:
            task["task_name"] = task_name
        if full_superset:
            # Send the full set explicitly rather than relying on omission:
            # init deep-merges onto the current configuration, so omitting
            # keeps the previous (possibly empty) value. The full instance
            # list taken from task_metadata equals the server's default
            # _all_task_instances path.
            allid = [int(i) for i in (info.get("instance_ids") or [])]
            if allid:
                task["superset_instances"] = allid
        else:
            task["superset_instances"] = []
        if build_instance is not None:
            # The build instance must be explicit too, or the command-line
            # one survives and reload location becomes unpredictable.
            task["instances"] = [int(build_instance)]
        payload = {"task": task, "backends": backends}
        if randomization is not None:
            payload["randomization"] = randomization
        if seed is not None:
            payload["seed"] = int(seed)
        t0 = time.monotonic()
        self.session_ready = False
        self.last_error = ""
        dims, schema = self._client.init(payload)
        self._n = int(getattr(dims, "num_envs", n) or n)
        if self._on_reload is not None:
            self._on_reload(dims, schema, sub)
        self._task_cache.pop(sub, None)
        self.current.update(sub_task_name=sub, task_name=task_name, instance_id=None, instruction_id=None, seed=-1)
        self.session_ready = True
        self.last_error = ""
        return {
            "ok": True,
            "sub_task_name": sub,
            "num_envs": self._n,
            "action_dim": int(getattr(dims, "action_dim", 0) or 0),
            "seconds": round(time.monotonic() - t0, 1),
        }

    def _do_reset(self, instance_id: int, instruction_id: int, seed: int = -1, env_ids: Optional[list] = None) -> dict:
        """Switch instance/instruction. Full-length arrays with -1 in
        non-target slots, matching RLinf exactly."""
        if not self.session_ready:
            return {"ok": False, "error": "session unavailable; reload successfully before resetting"}
        n = self._n
        ids = list(range(n)) if env_ids is None else [int(i) for i in env_ids]
        if not ids:
            return {"ok": False, "error": "env_ids must not be empty"}
        if min(ids) < 0 or max(ids) >= n:
            return {"ok": False, "error": f"env_ids out of range: {ids} " f"(num_envs={n})"}
        if int(seed) < -1:
            return {"ok": False, "error": "seed must be -1 or non-negative"}
        seeds = np.full(n, -1, dtype=np.int64)
        instances = np.full(n, -1, dtype=np.int64)
        instructions = np.full(n, -1, dtype=np.int64)
        for e in ids:
            seeds[e] = int(seed)
            instances[e] = int(instance_id)
            instructions[e] = int(instruction_id)
        mask = np.zeros(n, dtype=np.float32)
        mask[ids] = 1.0
        text = self._instruction_text(int(instance_id), int(instruction_id))
        info = self._task_cache.get(self.current.get("sub_task_name") or "")
        if info is not None:
            table = (info.get("instructions_by_instance") or {}).get(str(instance_id))
            if table is None:
                table = (info.get("instructions_by_instance") or {}).get(int(instance_id), [])
            if not any(int(item.get("id", -1)) == int(instruction_id) for item in table):
                return {"ok": False, "error": f"instruction {instruction_id} is unavailable for instance {instance_id}"}
        t0 = time.monotonic()
        self.session_ready = False
        out = self._client.reset(
            mask, seeds=seeds.tolist(), instances=instances.tolist(), instructions=instructions.tolist()
        )
        if self._on_reset is not None:
            self._on_reset(out, int(instance_id), int(instruction_id), text, int(seed))
        self.current.update(instance_id=int(instance_id), instruction_id=int(instruction_id), seed=int(seed))
        self.session_ready = True
        self.last_error = ""
        st = np.asarray(getattr(out, "states", np.zeros((1, 1))))
        return {
            "ok": True,
            "instance_id": int(instance_id),
            "instruction_id": int(instruction_id),
            "seed": int(seed),
            "states_finite": bool(np.isfinite(st).all()),
            "seconds": round(time.monotonic() - t0, 1),
        }

    def _instruction_text(self, instance_id: int, instruction_id: int) -> str:
        """Look up instruction text on the main loop, refreshing the task cache if needed."""
        sub = self.current.get("sub_task_name") or ""
        if sub and sub not in self._task_cache:
            try:
                self._fetch_task(sub)
            except Exception as exc:  # noqa: BLE001 — leave blank, do not
                # interrupt the reset
                print(
                    f"[ctl] live lookup of {sub}'s instruction table failed:" f"{type(exc).__name__}: {exc}", flush=True
                )
        tbl = (self._task_cache.get(sub) or {}).get("instructions_by_instance") or {}
        for it in tbl.get(str(instance_id)) or tbl.get(instance_id) or []:
            if isinstance(it, dict) and int(it.get("id", -1)) == instruction_id:
                return str(it.get("text") or "")
        return ""

    # -- state snapshot for HTTP routes ----------------------------------

    def state(self) -> dict:
        d = {
            "current": dict(self.current),
            "num_envs": self._n,
            "last_error": self.last_error,
            "busy": self.pending(),
            "session_ready": self.session_ready,
            "hud": dict(self.hud),
        }
        try:
            d.update(self._state_fn() or {})
        except Exception:  # noqa: BLE001
            pass
        return d


# --------------------------------------------------------------------------- #
# HTTP routes, dispatched in by the viewer's Handler
# --------------------------------------------------------------------------- #


def _generalization_defaults(sub: str) -> dict:
    """The physics-side randomization this task gets when the page sends none
    of it: the task generalization file (``randomize.task_defaults``), plus
    the engines' object-offset default. Shown as the panel's placeholders.
    Tasks with per-instance robot poses skip the base value on the server."""
    try:
        from genie_sim_engine_core.common.reset_randomization import DEFAULT_OBJECT_XY_RADIUS  # noqa: PLC0415
    except Exception:  # noqa: BLE001 -- the engine package is not always on the path
        DEFAULT_OBJECT_XY_RADIUS = 0.03
    from rlinf_geniesim.randomize import task_defaults  # noqa: PLC0415

    try:
        d = task_defaults.for_task(sub)
    except Exception:  # noqa: BLE001 -- a broken file must not break the page
        d = {}
    return {
        "object_xy_radius": d.get("object_xy_radius", DEFAULT_OBJECT_XY_RADIUS),
        "init_joint_thresh": d.get("init_joint_thresh"),
        "init_base_xy_thresh": d.get("init_base_xy_thresh"),
    }


def handle_api(ctl: "TeleopControl", path: str, body: bytes) -> tuple[int, dict]:
    """Return ``(http_status, json_dict)``; ``path`` has the ``/api`` prefix
    stripped."""
    try:
        args = json.loads(body.decode() or "{}") if body else {}
    except Exception:  # noqa: BLE001
        return 400, {"ok": False, "error": "body is not valid JSON"}

    if path == "/catalog":
        cat = ctl.catalog(refresh=bool(args.get("refresh")))
        subs = sorted((cat.get("sub_tasks") or {}).keys())
        return 200, {"ok": True, "sub_tasks": subs}

    if path == "/task":
        sub = str(args.get("sub_task_name") or "")
        if not sub:
            return 400, {"ok": False, "error": "missing sub_task_name"}
        info = ctl.task_info(sub, refresh=bool(args.get("refresh")))
        # Report only switchable instances; the full instance_ids contains
        # ones a reload can return to, and listing them invites the trap.
        insts = [int(i) for i in (info.get("switchable_instance_ids") or [])]
        raw = info.get("instructions_by_instance") or {}
        instr = {
            str(k): [
                {
                    "id": int(it.get("id", i)) if isinstance(it, dict) else i,
                    "text": (it.get("text") if isinstance(it, dict) else str(it)),
                }
                for i, it in enumerate(v or [])
            ]
            for k, v in raw.items()
        }
        return 200, {
            "ok": True,
            "sub_task_name": sub,
            "switchable_instance_ids": insts,
            "task_names": list(info.get("task_names") or []),
            "default_task_name": info.get("default_task_name"),
            "instructions_by_instance": instr,
            "generalization_defaults": _generalization_defaults(sub),
        }

    if path == "/reload":
        bi = args.get("build_instance")
        sd = args.get("seed")
        return 200, ctl.submit(
            "reload",
            {
                "sub_task_name": str(args.get("sub_task_name") or ""),
                "task_name": str(args.get("task_name") or ""),
                "build_instance": (int(bi) if bi is not None and bi != "" else None),
                "full_superset": bool(args.get("full_superset", True)),
                "randomization": args.get("randomization"),
                "seed": (int(sd) if sd is not None and sd != "" else None),
            },
        )

    if path == "/reset":
        try:
            inst = int(args.get("instance_id"))
            instr_id = int(args.get("instruction_id"))
        except (TypeError, ValueError):
            return 400, {"ok": False, "error": "instance_id / instruction_id must be " "integers"}
        return 200, ctl.submit(
            "reset", {"instance_id": inst, "instruction_id": instr_id, "seed": int(args.get("seed", -1))}
        )

    if path == "/randomization":
        # A real zmq round-trip (SimClient.randomization_readback), so this
        # goes through the queue exactly like /catalog and /task — it only
        # looks read-only, the REQ socket does not care.
        return 200, ctl.submit("randomization_readback", {})

    if path == "/state":
        return 200, {"ok": True, **ctl.state()}

    return 404, {"ok": False, "error": f"unknown endpoint {path}"}


UI_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Teleoperation console</title><style>
body{font-family:-apple-system,"PingFang SC",sans-serif;margin:0;background:#0f172a;color:#e2e8f0}
.wrap{max-width:1900px;margin:0 auto;padding:12px}
h1{font-size:.95em;margin:0 0 8px;color:#94a3b8;font-weight:600}
.panel{background:#1e293b;border:1px solid #334155;border-radius:9px;padding:12px}
.panel h2{font-size:.85em;margin:0 0 8px;color:#7dd3fc;font-weight:600}
.row{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}
.row>.panel{flex:1;min-width:290px}
label{display:block;font-size:.76em;color:#94a3b8;margin:7px 0 3px}
select,input{width:100%;padding:6px 8px;background:#0f172a;color:#e2e8f0;
  border:1px solid #475569;border-radius:5px;font-size:.85em;box-sizing:border-box}
input.ckb{width:auto;vertical-align:middle;margin-right:6px}
.pair{display:flex;gap:8px}.pair>div{flex:1}
.subhead{font-size:.78em;color:#7dd3fc;margin:10px 0 2px;font-weight:600}
label.ckl{display:flex;align-items:center;color:#e2e8f0;font-size:.82em;margin:8px 0 2px}
label.ckl.dis{color:#64748b}
button{margin-top:10px;width:100%;padding:9px;background:#2563eb;color:#fff;border:0;
  border-radius:5px;font-size:.9em;cursor:pointer}
button:hover{background:#1d4ed8} button:disabled{background:#475569;cursor:not-allowed}
.hint{font-size:.7em;color:#64748b;margin-top:5px;line-height:1.45}
#log{background:#020617;border-radius:6px;padding:8px;font-family:ui-monospace,Menlo,monospace;
  font-size:.73em;height:132px;overflow-y:auto;white-space:pre-wrap;color:#a5b4fc}
#randOut{background:#020617;border-radius:6px;padding:8px;font-family:ui-monospace,Menlo,monospace;
  font-size:.7em;max-height:260px;overflow:auto;white-space:pre-wrap;color:#a5b4fc;margin:8px 0 0}
img{width:100%;border-radius:8px;border:1px solid #334155;display:block;background:#000}
.ok{color:#4ade80}.err{color:#f87171}.dim{color:#64748b}

/* Large HUD outside the picture — the in-picture copy is too small to read. */
/* Five cells plus the instruction overflow on narrow screens, so wrap and give
   the instruction a minimum width so it is not squeezed into one column of
   text. */
#hud{display:flex;gap:12px;align-items:stretch;margin:12px 0 0;flex-wrap:wrap}
#instr{flex:1 1 420px;min-width:320px;background:#111827;border:1px solid #334155;border-radius:9px;
  padding:14px 18px;font-size:1.5em;line-height:1.35;font-weight:600;color:#fff;
  display:flex;align-items:center;min-height:66px}
.stat{background:#111827;border:1px solid #334155;border-radius:9px;padding:10px 14px;
  text-align:center;min-width:96px;flex:0 0 auto;display:flex;flex-direction:column;justify-content:center}
.stat .k{font-size:.7em;color:#94a3b8;letter-spacing:.06em}
.stat .v{font-size:1.6em;font-weight:700;font-family:ui-monospace,Menlo,monospace;line-height:1.15}
#vSucc.on{color:#4ade80}#vSucc.off{color:#64748b}#vSucc.na{color:#f87171}
#vRew.hot{color:#fbbf24}
</style></head><body><div class="wrap">
<h1>Teleoperation console — control plane identical to RLinf (task_metadata / init / reset)</h1>

<div class="panel"><img id="cam" src="/stream"></div>
<p id="sessionNotice" class="err" hidden></p>

<div id="hud">
  <div id="instr" class="dim">(waiting for instruction…)</div>
  <div class="stat"><div class="k">REWARD</div><div class="v" id="vRew">—</div></div>
  <div class="stat"><div class="k">SUCCESS</div><div class="v" id="vSucc">—</div></div>
  <div class="stat"><div class="k">STEP</div><div class="v" id="vStep">—</div></div>
  <div class="stat"><div class="k">EPISODE</div><div class="v" id="vEp">—</div></div>
</div>
<p class="hint" style="margin:6px 2px 0">
  <b>REWARD / SUCCESS</b> come straight from <code>step()</code> unmodified;
  <b>STEP / EPISODE</b> are client counters;
  <b>instruction</b> is looked up by instruction_id via <code>task_metadata</code> — step has
  no instruction field, so it is what we requested and the server accepted, not what the
  server echoed.</p>

<div class="row">
  <div class="panel">
    <h2>Switch task (reload → init verb)</h2>
    <label>subtask sub_task_name</label>
    <select id="sub"></select>
    <label>background combination task_name<span class="dim"> (defaults to the official suite pairing)</span></label>
    <select id="tname"><option value="">(auto)</option></select>
    <button id="btnReload">Reload</button>
    <div class="hint">reload sends the task's <b>full superset</b> explicitly, matching RLinf.
      You cannot rely on omitting it: <code>init</code> deep-merges onto the current
      configuration, and omission keeps the command-line <b>explicitly disabled</b> empty
      set. The instance selected on the right is sent as the <b>build instance</b>.</div>
  </div>
  <div class="panel" style="flex:1.5;min-width:340px">
    <h2>Domain randomization</h2>
    <label>build-time seed<span class="dim"> build-time seed -- drives joint_pd / background, written
      into the top-level <code>seed</code> key of the init payload; leave it empty to send
      <code>null</code>. <b>This is not</b> the "per-reset seed" on the Reset panel to the right
      (which drives object/light/material/camera pose) -- the two differ in both timing and
      meaning. If joint_pd or background below has enable checked but this field is left empty,
      the server will <b>loudly error</b> when building the stage, instead of silently falling
      back to another value.</span></label>
    <input id="rz_seed" type="number" placeholder="0">

    <div class="subhead">Physics (blank = use the task default, grey text is the default; 0 = off)</div>
    <label>object_xy_radius (m)</label>
    <input id="rz_object_xy_radius" type="number" step="0.01" placeholder="task default">
    <label>init_joint_thresh (rad)</label>
    <input id="rz_init_joint_thresh" type="number" step="0.01" placeholder="task default">
    <div class="pair">
      <div><label>init_base_xy_thresh.x (m)</label>
        <input id="rz_base_xy_x" type="number" step="0.01" placeholder="task default"></div>
      <div><label>init_base_xy_thresh.y (m)</label>
        <input id="rz_base_xy_y" type="number" step="0.01" placeholder="task default"></div>
    </div>
    <label class="ckl"><input id="rz_joint_pd_enable" class="ckb" type="checkbox">joint_pd enable</label>
    <div class="pair">
      <div><label>kp_scale lo</label><input id="rz_kp_lo" type="number" step="0.05" value="1.0"></div>
      <div><label>kp_scale hi</label><input id="rz_kp_hi" type="number" step="0.05" value="1.0"></div>
    </div>
    <div class="pair">
      <div><label>kd_scale lo</label><input id="rz_kd_lo" type="number" step="0.05" value="1.0"></div>
      <div><label>kd_scale hi</label><input id="rz_kd_hi" type="number" step="0.05" value="1.0"></div>
    </div>

    <div class="subhead">Render (each has its own enable, off by default)</div>
    <label class="ckl"><input id="rz_lights_enable" class="ckb" type="checkbox">lights enable</label>
    <div class="pair">
      <div><label>intensity_scale lo</label><input id="rz_lights_lo" type="number" step="0.05" value="0.5"></div>
      <div><label>intensity_scale hi</label><input id="rz_lights_hi" type="number" step="0.05" value="1.5"></div>
    </div>
    <label>distant_rotate_deg</label>
    <input id="rz_lights_rotate" type="number" step="1" value="45.0">

    <label class="ckl"><input id="rz_material_enable" class="ckb" type="checkbox">material enable</label>
    <label>mode</label>
    <select id="rz_material_mode">
      <option value="ml_prefix">ml_prefix</option>
      <option value="keyword">keyword</option>
      <option value="both">both</option>
    </select>
    <label>keywords<span class="dim"> (comma-separated)</span></label>
    <input id="rz_material_keywords" value="table,door">

    <label class="ckl"><input id="rz_background_enable" class="ckb" type="checkbox">background enable</label>
    <label>max_footprint_m<span class="dim"> (leave empty = null, falls back to env_spacing)</span></label>
    <input id="rz_background_footprint" type="number" step="0.1" placeholder="(auto)">

    <label class="ckl"><input id="rz_camera_enable" class="ckb" type="checkbox">camera_pose enable</label>
    <div class="pair">
      <div><label>threshold.x (m)</label><input id="rz_cam_x" type="number" step="0.01" value="0"></div>
      <div><label>threshold.y (m)</label><input id="rz_cam_y" type="number" step="0.01" value="0"></div>
      <div><label>threshold.z (m)</label><input id="rz_cam_z" type="number" step="0.01" value="0"></div>
    </div>
    <div class="pair">
      <div><label>roll (deg)</label><input id="rz_cam_roll" type="number" step="1" value="0"></div>
      <div><label>pitch (deg)</label><input id="rz_cam_pitch" type="number" step="1" value="0"></div>
      <div><label>yaw (deg)</label><input id="rz_cam_yaw" type="number" step="1" value="0"></div>
    </div>

    <div class="subhead" style="color:#64748b">Sensor (not implemented here)</div>
    <div class="hint">teleop does not support this yet -- this dimension is implemented in the
      client-side image post-processing in <code>vec_env.py</code>; teleop talks to a bare
      <code>SimClient</code> and does not have that code path. Placeholder, not interactive.</div>
    <label class="ckl dis"><input class="ckb" type="checkbox" disabled>noise</label>
    <label class="ckl dis"><input class="ckb" type="checkbox" disabled>occlusion</label>
    <label class="ckl dis"><input class="ckb" type="checkbox" disabled>drop_frame</label>

    <button id="btnReadRandom" style="margin-top:12px;background:#334155">Read applied config</button>
    <div class="hint"><code>resolved</code> = what is intended to take effect; <code>applied</code> =
      what actually took effect. This block refreshes automatically once after a successful Reset.
      Before the first reset, <code>applied.last_reset</code> is <code>null</code> -- seeing
      <code>null</code> just means "hasn't been reset yet", not an error.</div>
    <pre id="randOut">(not read yet)</pre>
  </div>
  <div class="panel">
    <h2>Switch instance / instruction (reset)</h2>
    <label>instance (switchable only)</label>
    <select id="inst"></select>
    <label>instruction id</label>
    <select id="instr_sel"></select>
    <label>seed<span class="dim"> (-1 = official no randomization; the per-reset seed, drives
      object/light/material/camera pose -- not the same as the "build-time seed" in the
      randomization panel on the left)</span></label>
    <input id="seed" type="number" value="-1">
    <button id="btnReset">Reset</button>
    <div class="hint">Full-length arrays with -1 in non-target slots, matching RLinf's
      <code>build_geniesim_reset_vectors</code>. This lists the task's <b>intrinsic switchable set</b>;
      if the server was started with a <b>narrow superset</b>, switching outside that set is <b>refused
      loudly</b> — reload once, or add <code>--superset=full</code> at startup.</div>
  </div>
  <div class="panel" style="flex:1.6">
    <h2>Status / log</h2><div id="log"></div>
  </div>
</div>
</div><script>
const $=id=>document.getElementById(id);
const log=(m,c='')=>{const d=$('log');d.innerHTML+=`<span class="${c}">[${new Date().toLocaleTimeString()}] ${m}</span>\n`;d.scrollTop=d.scrollHeight};
const api=async(p,b)=>{const r=await fetch('/api'+p,{method:'POST',body:JSON.stringify(b||{})});return r.json()};
let TASK=null;
async function loadCatalog(){
  const r=await api('/catalog');
  if(!r.ok){log('catalogue query failed: '+r.error,'err');return}
  $('sub').innerHTML=r.sub_tasks.map(s=>`<option>${s}</option>`).join('');
  // Preselect the currently running subtask. Without this, the default is the
  // first alphabetically, and the instance/instruction dropdowns follow it —
  // pressing Reset would then send another task's instruction ids.
  let st=null; try{ st=await api('/state'); }catch(e){}
  const cur=(st&&st.current)||{};
  if(cur.sub_task_name && r.sub_tasks.includes(cur.sub_task_name))
    $('sub').value=cur.sub_task_name;
  log(`catalogue: ${r.sub_tasks.length} subtasks; running: ${cur.sub_task_name||'(unknown)'}`,'ok');
  await loadTask(cur);
}
async function loadTask(cur){
  const sub=$('sub').value; if(!sub)return;
  const r=await api('/task',{sub_task_name:sub});
  if(!r.ok){log('task query failed: '+r.error,'err');return}
  TASK=r;
  // Placeholders show what a blank physics field means for this task.
  const gd=r.generalization_defaults||{};
  const ph=v=>(v===null||v===undefined)?'0 (off)':'default '+v;
  $('rz_object_xy_radius').placeholder=ph(gd.object_xy_radius);
  $('rz_init_joint_thresh').placeholder=ph(gd.init_joint_thresh);
  const gb=gd.init_base_xy_thresh;
  $('rz_base_xy_x').placeholder=ph(gb?gb[0]:null);
  $('rz_base_xy_y').placeholder=ph(gb?gb[1]:null);
  $('inst').innerHTML=r.switchable_instance_ids.map(i=>`<option>${i}</option>`).join('');
  // Preselect instance/instruction to the running values only when the shown
  // task is the one running.
  if(cur && cur.sub_task_name===sub && cur.instance_id!==null
     && cur.instance_id!==undefined) $('inst').value=String(cur.instance_id);
  const names=(r.task_names||[]).map(t=>(typeof t==='string')?{n:t,rb:''}:{n:t.task_name||'',rb:t.robot?` (${t.robot})`:''});
  const def=r.default_task_name;
  if(def && !names.some(x=>x.n===def)) names.unshift({n:def,rb:''});
  $('tname').innerHTML=(def?'':'<option value="">(auto)</option>')+
    names.map(x=>`<option value="${x.n}">${x.n}${x.rb}${x.n===def?' — official':''}</option>`).join('');
  if(def) $('tname').value=def;
  fillInstr();
  if(cur && cur.sub_task_name===sub && cur.instruction_id!==null
     && cur.instruction_id!==undefined) $('instr_sel').value=String(cur.instruction_id);
  log(`${sub}: ${r.switchable_instance_ids.length} switchable instances`,'ok');
}
function fillInstr(){
  const k=$('inst').value; const arr=(TASK&&TASK.instructions_by_instance[k])||[];
  $('instr_sel').innerHTML=arr.map(o=>`<option value="${o.id}">${o.id} — ${(o.text||'').slice(0,70)}</option>`).join('')
    ||'<option value="0">0</option>';
}
$('sub').onchange=()=>loadTask(null); $('inst').onchange=fillInstr;

// Build the FULL randomization object every time, field by field, per the
// yaml schema (GENERALIZATION_DESIGN.md §5). Never omit a dimension's key
// just because its checkbox is unchecked: init() deep-merges onto the
// server's current config, so an omitted key means "keep whatever it was
// last time", not "off". An unchecked dimension must still show up with
// enable:false (or 0 for the physics three, which have no enable field).
//
// init_base_xy_thresh is the ONE exception to "0 means off": the server's
// BaseXYChannelConflictError (randomize/spec.py) checks PRESENCE of this
// key against a per-instance robot_base_xy table, not its magnitude --
// [0,0] still counts as "set" and collides with any task that replicates
// per-instance robot base poses verbatim (e.g. the g2_90d family). The
// canonical off state for this one field is the key being None, so we send
// null instead of [0,0] whenever both fields are zero/blank.
function buildRandomization(){
  const kw=($('rz_material_keywords').value||'').split(',').map(s=>s.trim()).filter(Boolean);
  // Physics: a blank field is not sent, so the task's default applies
  // (config/task_generalization.yaml; shown as the placeholder). A typed 0 is
  // sent and switches that dimension off.
  const phys={};
  const orad=$('rz_object_xy_radius').value, jt=$('rz_init_joint_thresh').value;
  const bx=$('rz_base_xy_x').value, by=$('rz_base_xy_y').value;
  if(orad!=='') phys.object_xy_radius=+orad;
  if(jt!=='') phys.init_joint_thresh=+jt;
  if(bx!=='' || by!==''){
    const bxx=+(bx||0), bxy=+(by||0);
    phys.init_base_xy_thresh=(bxx===0 && bxy===0) ? null : [bxx, bxy];
  }
  return {
    ...phys,
    joint_pd: {
      enable: $('rz_joint_pd_enable').checked,
      kp_scale: [+($('rz_kp_lo').value || 1.0), +($('rz_kp_hi').value || 1.0)],
      kd_scale: [+($('rz_kd_lo').value || 1.0), +($('rz_kd_hi').value || 1.0)]
    },
    render: {
      lights: {
        enable: $('rz_lights_enable').checked,
        intensity_scale: [+($('rz_lights_lo').value || 0.5), +($('rz_lights_hi').value || 1.5)],
        distant_rotate_deg: +($('rz_lights_rotate').value || 45.0)
      },
      material: {
        enable: $('rz_material_enable').checked,
        mode: $('rz_material_mode').value,
        keywords: kw
      },
      background: {
        enable: $('rz_background_enable').checked,
        max_footprint_m: $('rz_background_footprint').value===''?null:+$('rz_background_footprint').value
      },
      camera_pose: {
        enable: $('rz_camera_enable').checked,
        threshold: {
          x: +($('rz_cam_x').value || 0), y: +($('rz_cam_y').value || 0), z: +($('rz_cam_z').value || 0),
          roll: +($('rz_cam_roll').value || 0), pitch: +($('rz_cam_pitch').value || 0),
          yaw: +($('rz_cam_yaw').value || 0)
        }
      }
      // sensor (noise/occlusion/drop_frame) intentionally absent: that
      // dimension lives in vec_env's client-side image post-processing,
      // which teleop's bare SimClient never runs — nothing here to enable.
    }
  };
}
function showRandomization(r){ $('randOut').textContent=JSON.stringify(r,null,2); }
$('btnReadRandom').onclick=async()=>{
  const r=await api('/randomization',{});
  showRandomization(r);
  log(r.ok?'randomization readback ok':'randomization readback failed: '+r.error, r.ok?'ok':'err');
};
$('btnReload').onclick=async()=>{
  const b=$('btnReload'); b.disabled=true; b.textContent='rebuilding… (cold can take a minute)';
  log(`reload -> ${$('sub').value} …`);
  const sd=$('rz_seed').value;
  const r=await api('/reload',{sub_task_name:$('sub').value,task_name:$('tname').value,
      build_instance:$('inst').value===''?null:+$('inst').value,
      randomization:buildRandomization(), seed: sd===''?null:+sd});
  log(r.ok?`reload ok: ${r.sub_task_name} num_envs=${r.num_envs} action_dim=${r.action_dim} (${r.seconds}s)`
        :`reload failed: ${r.error}`, r.ok?'ok':'err');
  b.disabled=false; b.textContent='Reload'; if(r.ok) loadCatalog();
};
$('btnReset').onclick=async()=>{
  const b=$('btnReset'); b.disabled=true; b.textContent='resetting…';
  const r=await api('/reset',{instance_id:+$('inst').value,
      instruction_id:+$('instr_sel').value,seed:+$('seed').value});
  log(r.ok?`reset ok: instance=${r.instance_id} instr=${r.instruction_id} seed=${r.seed} `+
        `states_finite=${r.states_finite} (${r.seconds}s)`:`reset failed: ${r.error}`,
      r.ok?'ok':'err');
  b.disabled=false; b.textContent='Reset';
  // Auto-refresh the readback after a successful reset so switching
  // instance/instruction to verify a per-reset dimension does not need a
  // separate manual click each time.
  if(r.ok){ try{ showRandomization(await api('/randomization',{})); }catch(e){} }
};
// HUD polling (large text outside the picture). 5 Hz keeps up with human
// operation without saturating the control surface.
let lastErr='';
setInterval(async()=>{
  let s; try{ s=await api('/state'); }catch(e){ return }
  if(!s||!s.ok) return;
  const notice=$('sessionNotice');
  notice.hidden=s.session_ready!==false;
  notice.textContent='Simulation paused. The previous frame and values may be stale.'+
      (s.last_error?' '+s.last_error+' — fix the task and Reload to resume.':'');
  const h=s.hud||{};
  const it=$('instr');
  if(h.instruction){ it.textContent=h.instruction; it.classList.remove('dim'); }
  $('vRew').textContent = (h.reward===undefined||h.reward===null)?'—':(+h.reward).toFixed(3);
  $('vRew').className = 'v'+((+h.reward>=1)?' hot':'');
  const sv=$('vSucc');
  if(h.success===true){ sv.textContent='YES'; sv.className='v on'; }
  else if(h.success===false){ sv.textContent='no'; sv.className='v off'; }
  else { sv.textContent='EVAL OFF'; sv.className='v na'; sv.style.fontSize='1.05em'; }
  $('vStep').textContent = (h.step===undefined)?'—':h.step;
  $('vEp').textContent   = (h.episode===undefined)?'—':h.episode;
  if(s.last_error && s.last_error!==lastErr){ lastErr=s.last_error; }
},200);
loadCatalog();
</script></body></html>"""
