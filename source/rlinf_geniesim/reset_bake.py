# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Bake per-instance resting poses and support layouts in the live engine.

Schedule instances across environments and replace authored poses with settled
env-local poses. Retain authored poses for bodies the engine does not simulate.
Record support/box relationships for reset randomization, report residual motion
or large displacement, and cache by scene, engine and physics configuration."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

BAKE_VERSION = 4  # 4: fused keeps parked superset slots in place (gravity off, woken)
_OBJECTS_PREFIX = "World/Objects/"
#: Environment variables that change contact behaviour, and so the rest state.
#: The other physics knobs live in the physics_params yaml, which is hashed as a
#: file; see genie_sim_engine_core/docs/physics_knobs.md.
_PHYSICS_ENV = ("RLINF_ROBOT_SOLREF", "RLINF_OBJECT_SOLREF")
_BAKEABLE_ENGINES = ("newton_mujoco_warp", "isaac_fused")


@dataclass
class BakeConfig:
    #: An instance still moving after ``cap_s`` is recorded as it stands when
    #: what is left is jitter (less than ``settling_mm`` per window); one that
    #: is still clearly settling runs on, up to ``hard_cap_s``.
    cap_s: float = 1.5
    hard_cap_s: float = 5.0
    settling_mm: float = 2.0
    window_s: float = 0.25
    tol_m: float = 1e-4
    control_hz: float = 30.0
    exclude: tuple = ("table",)
    #: Objects that end up farther than this sideways from their authored pose
    #: are reported: they slid, rolled or fell off their support.
    travel_report_mm: float = 20.0


def body_key(name: str) -> Optional[str]:
    """Instance table key of a body (its path below ``World/Objects/``)."""
    head, sep, tail = name.partition(_OBJECTS_PREFIX)
    return tail if sep and head in ("", "/") and tail else None


def _body_names(engine) -> list:
    schema = engine.get_scene_schema()
    names = schema.get("body_link_names") if isinstance(schema, dict) else getattr(schema, "body_link_names", None)
    return list(names or [])


def _object_of(key: str) -> str:
    return key.split("/", 1)[0]


# ---------------------------------------------------------------------------
# The bake
# ---------------------------------------------------------------------------


def bake_rest_states(
    engine,
    instances,
    table: dict,
    cfg: Optional[BakeConfig] = None,
    log: Callable[[str], None] = print,
    static_boxes: Optional[dict] = None,
) -> dict:
    """Run every instance to rest in ``engine`` and return the baked tables.

    ``engine`` is a started physics backend (``reset``/``step``/
    ``get_joint_states``/``get_body_poses``/``get_body_aabbs``/
    ``get_scene_schema``); ``table`` is the authored
    ``instance_body_poses``. Returns ``{"instance_body_poses", "instance_layout",
    "stats"}`` with env-local poses, in the authored table's format.

    Bodies in the table that the engine does not simulate (a static table in
    Newton) keep their authored pose; ``static_boxes`` (body-frame boxes, see
    ``usd_static_boxes``) puts them in the layout as surfaces.
    """
    cfg = cfg or BakeConfig()
    names = _body_names(engine)
    keys_needed = set().union(*(set(t) for t in table.values())) if table else set()
    cols = [(i, body_key(n)) for i, n in enumerate(names) if body_key(n) in keys_needed]
    if not cols:
        raise RuntimeError("rest-state bake: none of the engine's bodies are in the instance table")
    static_keys = sorted(keys_needed - {k for _, k in cols})
    static_boxes = static_boxes or {}
    idx = np.array([i for i, _ in cols])
    keys = [k for _, k in cols]

    step_kw = {}
    try:
        import inspect  # noqa: PLC0415

        if "need_images" in inspect.signature(engine.step).parameters:
            step_kw = {"need_images": False}
    except (TypeError, ValueError):
        pass

    win = max(1, int(round(cfg.window_s * cfg.control_hz)))
    cap = max(win + 1, int(round(cfg.cap_s * cfg.control_hz)))
    hard_cap = max(cap, int(round(cfg.hard_cap_s * cfg.control_hz)))
    n_env = int(np.asarray(engine.get_body_poses()).shape[0])
    queue = [int(i) for i in instances]
    assign = [None] * n_env
    origin = np.zeros((n_env, 3))
    started_at = np.zeros(n_env, dtype=int)
    last_settling = np.zeros(n_env, dtype=int)  # age at which it last moved more than settling_mm
    hist: list = [[] for _ in range(n_env)]
    hold = np.asarray(engine.get_joint_states(), np.float32)[:, :, 0].copy()
    baked, layouts, stats = {}, {}, {}
    steps = 0
    t0 = time.monotonic()

    def start(envs):
        insts = [queue.pop(0) for _ in envs]
        engine.reset(env_ids=list(envs), seeds=[-1] * len(envs), instances=insts)
        poses = np.asarray(engine.get_body_poses(), np.float64)
        js = np.asarray(engine.get_joint_states(), np.float32)[:, :, 0]
        for e, inst in zip(envs, insts):
            authored = np.array([table[str(inst)][k][:3] for k in keys], dtype=np.float64)
            origin[e] = (poses[e, idx, :3] - authored).mean(axis=0)
            assign[e] = inst
            started_at[e] = steps
            last_settling[e] = 0
            hist[e] = [poses[e, idx, :3].copy()]
            hold[e] = js[e]

    start(list(range(min(n_env, len(queue)))))
    while any(a is not None for a in assign):
        engine.step(hold, **step_kw)
        steps += 1
        poses = np.asarray(engine.get_body_poses(), np.float64)
        done = []
        for e, inst in enumerate(assign):
            if inst is None:
                continue
            if not np.isfinite(poses[e, idx, :3]).all():
                done.append((e, inst, None, float("nan")))  # the solver diverged
                continue
            hist[e].append(poses[e, idx, :3].copy())
            age = steps - started_at[e]
            if age < win:
                continue
            recent = np.stack(hist[e][-win - 1 :])
            motion = float(np.linalg.norm(recent - recent[-1], axis=-1).max())
            at_rest = motion <= cfg.tol_m
            if motion * 1000 > cfg.settling_mm:
                last_settling[e] = age
            # Past the soft cap, a body that was settling a moment ago gets one
            # more window to come to rest before it counts as jitter.
            if at_rest or age >= hard_cap or (age >= cap and age - last_settling[e] >= win):
                done.append((e, inst, (age - win) / cfg.control_hz if at_rest else None, motion))
        if not done:
            continue
        aabbs = np.asarray(engine.get_body_aabbs(), np.float64)
        for e, inst, rest_s, motion in done:
            authored_rows = table[str(inst)]
            if not np.isfinite(motion):
                # A diverged instance keeps its authored poses (and no layout):
                # writing NaN into the table stops the engine from starting.
                baked[str(inst)] = {k: list(v) for k, v in authored_rows.items()}
                stats[str(inst)] = {"rest_s": None, "residual_mm": None, "travel_mm": 0.0, "diverged": True}
                assign[e] = None
                continue
            local = poses[e, idx].copy()
            local[:, :3] -= origin[e]
            rows = {k: [float(v) for v in local[j]] for j, k in enumerate(keys)}
            rows.update({k: list(authored_rows[k]) for k in static_keys})
            baked[str(inst)] = rows
            box = aabbs[e, idx].copy()
            box[:, :3] -= origin[e]
            box[:, 3:] -= origin[e]
            box_keys = list(keys)
            extra = [_posed_box(static_boxes[k], authored_rows[k]) for k in static_keys if k in static_boxes]
            if extra:
                box = np.concatenate([box, np.asarray(extra)])
                box_keys += [k for k in static_keys if k in static_boxes]
            layouts[str(inst)] = build_layout(box_keys, box, cfg.exclude)
            # Sideways only: landing straight down onto the support is what the bake is for.
            travel = float(np.linalg.norm(local[:, :2] - np.array([authored_rows[k][:2] for k in keys]), axis=-1).max())
            stats[str(inst)] = {
                "rest_s": rest_s,
                "residual_mm": round(motion * 1000, 3),
                "travel_mm": round(travel * 1000, 2),
            }
            assign[e] = None
        refill = [e for e, *_ in done if queue][: len(queue)]
        if refill:
            start(refill)

    wall = time.monotonic() - t0
    far = sorted(int(i) for i, s in stats.items() if (s["travel_mm"] or 0) > cfg.travel_report_mm)
    if far:
        log(
            f"[rest-bake] WARNING: in instance(s) {far[:10]} an object moved more than "
            f"{cfg.travel_report_mm:g} mm sideways from its authored pose while settling (it slid, "
            f"rolled or fell off its support); the bake records where it ended up"
        )
    diverged = sorted(int(i) for i, s in stats.items() if s.get("diverged"))
    if diverged:
        log(
            f"[rest-bake] WARNING: the solver diverged (non-finite poses) in instance(s) {diverged[:10]}; "
            f"they keep their authored poses and the settle does not cover them"
        )
    capped = sorted(int(i) for i, s in stats.items() if s["rest_s"] is None and not s.get("diverged"))
    log(
        f"[rest-bake] {len(baked)} instance(s) baked in {wall:.1f}s ({steps} steps on {n_env} env(s)); "
        f"{len(capped)} still moving after {cfg.cap_s}s" + (f": {capped[:10]}" if capped else "")
    )
    return {
        "instance_body_poses": baked,
        "instance_layout": layouts,
        "stats": {"wall_s": round(wall, 2), "steps": steps, "envs": n_env, "instances": stats},
    }


def _posed_box(local_box, pose) -> np.ndarray:
    """World-aligned box of a body-frame box under ``pose`` (xyz + wxyz)."""
    lo, hi = np.asarray(local_box[:3], float), np.asarray(local_box[3:], float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w, x, y, z = (float(v) for v in pose[3:7])
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    pts = corners @ rot.T + np.asarray(pose[:3], float)
    return np.concatenate([pts.min(axis=0), pts.max(axis=0)])


def usd_static_boxes(usd_paths, keys) -> dict:
    """Body-frame boxes of ``World/Objects/<key>`` prims, from the first USD
    file in ``usd_paths`` that has each. For bodies the engine does not
    simulate (a static table), whose boxes the engine cannot report."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    out: dict = {}
    todo = list(keys)
    for path in usd_paths:
        if not todo or not path or not Path(path).is_file():
            continue
        stage = Usd.Stage.Open(str(path))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for key in list(todo):
            prim = stage.GetPrimAtPath("/World/Objects/" + key)
            if not prim or not prim.IsValid():
                continue
            rng = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            lo, hi = rng.GetMin(), rng.GetMax()
            out[key] = [float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2])]
            todo.remove(key)
    return out


def static_boxes_for(plan_: dict, engine) -> dict:
    """``usd_static_boxes`` for the instance-table bodies the engine does not
    report, looked up in the stage files this task was built from."""
    names = _body_names(engine)
    present = {body_key(n) for n in names}
    keys = sorted(set().union(*(set(t) for t in plan_["table"].values())) - present)
    if not keys:
        return {}
    manifest_path = Path(plan_["manifest"])
    manifest = json.loads(manifest_path.read_text())
    base = Path(manifest.get("base_path") or manifest_path.parent)
    candidates = [
        base / f"{manifest_path.stem}_superset_scene.usda",
        manifest.get("scene_usda"),
        manifest.get("foreground_usda"),
    ]
    return usd_static_boxes([str(c) for c in candidates if c], keys)


def build_layout(keys, boxes: np.ndarray, exclude=("table",)) -> dict:
    """Per-object boxes and carriers from rest-state body boxes (env-local).

    An object's box is the union of its bodies' boxes. Objects whose name
    contains an ``exclude`` word (never moved by the randomization) are
    surfaces. An object's carrier is the object or surface directly underneath
    or around it: its centre lies inside the carrier's footprint, the carrier
    starts lower, and the object's bottom is at most 1 cm above the carrier's
    top. Of several candidates the highest-topped wins (a bowl over the table).
    """
    merged: dict = {}
    for k, b in zip(keys, boxes):
        name = _object_of(k)
        if not np.all(np.isfinite(b)) or not np.any(b[3:] > b[:3]):
            continue
        cur = merged.get(name)
        merged[name] = (
            b.copy() if cur is None else np.concatenate([np.minimum(cur[:3], b[:3]), np.maximum(cur[3:], b[3:])])
        )
    surfaces = {n: b for n, b in merged.items() if any(x in n for x in exclude)}
    movable = {n: b for n, b in merged.items() if n not in surfaces}
    out_objects = {}
    for n, b in movable.items():
        c = (b[:2] + b[3:5]) / 2
        best, best_top = None, -np.inf
        for m, mb in list(movable.items()) + list(surfaces.items()):
            if m == n:
                continue
            if (
                mb[2] < b[2] - 1e-3
                and b[2] <= mb[5] + 0.01
                and mb[0] <= c[0] <= mb[3]
                and mb[1] <= c[1] <= mb[4]
                and mb[5] > best_top
            ):
                best, best_top = m, mb[5]
        out_objects[n] = {"aabb": [round(float(v), 6) for v in b], "carrier": best}
    return {
        "objects": out_objects,
        "surfaces": {n: {"aabb": [round(float(v), 6) for v in b]} for n, b in surfaces.items()},
    }


# ---------------------------------------------------------------------------
# Cache and scene-config plumbing
# ---------------------------------------------------------------------------


def cache_dir() -> Path:
    return Path(os.environ.get("RLINF_REST_BAKE_CACHE") or Path.home() / ".cache" / "rlinf_geniesim" / "rest_bake")


def _scene_cfg_path(manifest_path: Path, manifest: dict) -> Optional[Path]:
    p = manifest.get("scene_yaml")
    if not p:
        return None
    p = Path(p)
    if not p.is_absolute():
        p = Path(manifest.get("base_path") or manifest_path.parent) / p
    return p


def _physics_param_files() -> list:
    src = Path(__file__).resolve().parents[1]
    files = sorted((src / "geniesim_ros/src/ros_ws/src/genie_sim_engine_core/config").glob("physics_params*.yaml"))
    files += [Path(__file__).resolve().parent / "config" / "physics_overrides.yaml"]
    return [f for f in files if f.is_file()]


# The floor mesh grows with the number of envs it has to cover; its size does
# not change where anything comes to rest.
_FLOOR_BLOCK = re.compile(rb'(def Mesh "Floor".*?\{.*?)(\})', re.S)
_FLOOR_SIZE = re.compile(rb"\n[ \t]*(?:float3\[\] extent|point3f\[\] points) = [^\n]*")


def _without_floor_size(data: bytes) -> bytes:
    return _FLOOR_BLOCK.sub(lambda m: _FLOOR_SIZE.sub(b"", m.group(1)) + m.group(2), data)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _explain_miss(physics: str, task_id: str, parts: dict, log: Callable[[str], None]) -> None:
    """Name the key parts that differ from the newest cached bake of the same
    task on the same engine -- a miss costs a full bake, and a stage builder
    that writes something new on every run would otherwise miss silently."""
    newest = None
    for f in cache_dir().glob(f"{physics}_*.json"):
        try:
            old = json.loads(f.read_text()).get("key_parts") or {}
        except (OSError, ValueError):
            continue
        if old.get("task") == task_id and (newest is None or f.stat().st_mtime > newest[0]):
            newest = (f.stat().st_mtime, f.name, old)
    if newest is None:
        return
    _, name, old = newest
    changed = sorted(k for k in set(parts) | set(old) if k != "task" and parts.get(k) != old.get(k))
    log(f"[rest-bake] cache miss: this task's last bake ({name}) differs in: {', '.join(changed) or 'nothing'}")


def plan(
    manifest_path,
    *,
    physics: str,
    solver_substep=None,
    settings: Optional[dict] = None,
    log: Callable[[str], None] = print,
) -> Optional[dict]:
    """Look the task's rest-state bake up in the cache.

    Returns ``None`` when the task cannot be baked (engine not supported, no
    instance table). Otherwise a plan: ``status`` is ``"hit"`` when the cached
    bake has just been written into the scene config, ``"miss"`` when the bake
    still has to run (see ``inject`` and ``save``).
    """
    settings = dict(settings or {})
    if physics not in _BAKEABLE_ENGINES:
        return None
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text())
        cfg_path = _scene_cfg_path(manifest_path, manifest)
        if cfg_path is None or not cfg_path.is_file():
            return None
        scene_cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        return None
    sw = scene_cfg.get("instance_switch") or {}
    # A scene config that already carries a bake keeps the authored poses
    # aside; the key is always computed from those.
    table = sw.get("instance_body_poses_authored") or sw.get("instance_body_poses")
    if sw.get("body_pose_version") != 1 or not isinstance(table, dict) or not table:
        return None

    base = str(manifest.get("base_path") or manifest_path.parent)
    stem = manifest_path.stem

    def doc(v) -> str:
        return _digest(json.dumps(v, sort_keys=True, default=str).encode())

    # Everything that decides the rest state, one digest per part, so a miss
    # can say which part changed (``_explain_miss``).
    task_id = doc([stem, manifest.get("_benchmark_task")])
    parts = {
        "task": task_id,
        "engine": doc({"v": BAKE_VERSION, "physics": physics, "substep": solver_substep}),
        "robot": doc([manifest.get("robot_usda"), scene_cfg.get("robot")]),
        "newton": doc(scene_cfg.get("newton")),
        "table": doc({i: {k: [round(float(v), 6) for v in p] for k, p in t.items()} for i, t in table.items()}),
        "env": doc({k: os.environ[k] for k in _PHYSICS_ENV if k in os.environ}),
        "bake settings": doc({k: settings.get(k) for k in ("cap_s", "hard_cap_s", "window_s", "tol_m")}),
        # Which objects the layout treats as surfaces.
        "exclude": doc(list((scene_cfg.get("randomization") or {}).get("exclude", ["table"]))),
    }
    for f in sorted(Path(base).glob(f"{stem}*.usda")):
        parts[f.name.replace(stem, "<stem>")] = _digest(
            _without_floor_size(f.read_bytes().replace(base.encode(), b"<stage>"))
        )
    for f in _physics_param_files():
        parts[f.name] = _digest(f.read_bytes())
    key = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:32]
    path = cache_dir() / f"{physics}_{key}.json"
    out = {
        "key": key,
        "key_parts": parts,
        "path": str(path),
        "scene_cfg": str(cfg_path),
        "physics": physics,
        "manifest": str(manifest_path),
        "instances": sorted(int(i) for i in table),
        "table": table,
        # Surfaces in the layout are the objects the randomization never moves.
        "exclude": list((scene_cfg.get("randomization") or {}).get("exclude", ["table"])),
    }
    if path.is_file() and not settings.get("force"):
        try:
            cached = json.loads(path.read_text())
            inject(out, cached)
            out["status"] = "hit"
            out["stats"] = cached.get("stats", {})
            return out
        except Exception as exc:  # noqa: BLE001 -- a broken cache file is rebaked, not trusted
            log(f"[rest-bake] cache file {path} unusable ({exc!r}); baking again")
    else:
        _explain_miss(physics, task_id, parts, log)
    out["status"] = "miss"
    return out


def inject(plan_: dict, baked: dict) -> None:
    """Write a bake into the task's scene config, replacing the authored poses."""
    path = Path(plan_["scene_cfg"])
    scene_cfg = json.loads(path.read_text())
    sw = scene_cfg["instance_switch"]
    authored = sw.get("instance_body_poses_authored") or sw["instance_body_poses"]
    new = baked["instance_body_poses"]
    if set(new) != set(authored) or any(set(new[i]) != set(authored[i]) for i in authored):
        raise ValueError("rest-state bake does not cover the same instances and bodies as the scene")
    if not all(np.isfinite(np.asarray(v, dtype=np.float64)).all() for rows in new.values() for v in rows.values()):
        raise ValueError("rest-state bake holds non-finite poses")
    sw["instance_body_poses_authored"] = authored
    sw["instance_body_poses"] = new
    sw["instance_layout"] = baked.get("instance_layout") or {}
    sw["rest_state_baked"] = {"key": plan_["key"], "stats": baked.get("stats", {})}
    path.write_text(json.dumps(scene_cfg))


def save(plan_: dict, baked: dict) -> Path:
    path = Path(plan_["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-process temp name: two servers on one machine can bake the same task.
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps({"version": BAKE_VERSION, "key": plan_["key"], "key_parts": plan_.get("key_parts"), **baked})
    )
    tmp.replace(path)
    return path
