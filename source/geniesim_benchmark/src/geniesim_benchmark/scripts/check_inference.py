#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Send a saved corobot inference payload pkl to a model server and validate the response.

The payload is a JSON-RPC `{"method": "infer", "params": {...}}` envelope; the
server replies with `{"result": {"left_arm": ..., "right_arm": ..., ...}}` or
`{"error": ...}`. A canonical `corobot_payload.pkl` ships next to this script
and is used by default; pass a path to override it (e.g. a fresh
`debug_preview/debug_NNNN.pkl` from the corobot policy's debug dump).

Beyond key-presence and NaN/Inf checks, the response is validated per-dimension:
min/max/mean/std for each idx, plus flags for constant dims, out-of-range
values (kind-aware: JOINT_ABS expects radians, EEF_ABS expects meters+quat,
gripper expects [0,1]), NaN/Inf, and large jumps from the input state.

Usage:
    python check_inference.py --host 127.0.0.1 --port 8999      # bundled payload
    python check_inference.py debug_preview/debug_0001.pkl --host 127.0.0.1
"""

import argparse
import functools
import os
import pickle
import sys
import time

import msgpack
import numpy as np
import websockets.sync.client

# Bundled default payload, shipped next to this script.
DEFAULT_PAYLOAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corobot_payload.pkl")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

ICON_INFO = "ℹ️ "
ICON_OK = "✅"
ICON_WARN = "⚠️ "
ICON_FAIL = "❌"
ICON_LOAD = "📦"
ICON_CONN = "🔌"
ICON_RESP = "📡"
ICON_STAT = "📊"
ICON_TIME = "⏱️ "
ICON_ITER = "🔄"

RULE_HEAVY = "═" * 72
RULE_LIGHT = "─" * 72


def info(msg):
    print(f"{ICON_INFO} {msg}")


def ok(msg):
    print(f"{ICON_OK} {msg}")


def warn(msg):
    print(f"{ICON_WARN} {msg}")


def fail(msg):
    print(f"{ICON_FAIL} {msg}")


def section(icon, title):
    print(f"\n{icon} {title}")
    print(RULE_LIGHT)


# ---------------------------------------------------------------------------
# msgpack-numpy serialization
# ---------------------------------------------------------------------------


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_pack = functools.partial(msgpack.packb, default=_pack_array)
_unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array, raw=False)


# ---------------------------------------------------------------------------
# Payload loading
# ---------------------------------------------------------------------------


def load(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # The corobot policy debug hook dumps {"payload": ..., "obs": ...};
    # accept either the wrapper or a bare payload.
    payload = data["payload"] if isinstance(data, dict) and "payload" in data else data
    if not (isinstance(payload, dict) and payload.get("method") == "infer" and "params" in payload):
        keys = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"Not a corobot payload (expected method=infer / params). Got: {keys}")
    section(ICON_LOAD, "Payload")
    info(f"path : {pkl_path}")
    info(f"keys : {list(payload.keys())}")
    return payload


def connect(uri):
    section(ICON_CONN, f"Connecting to {uri}")
    ws = websockets.sync.client.connect(uri, compression=None, max_size=None, open_timeout=15)
    ok(f"connected to {uri}")
    return ws


# ---------------------------------------------------------------------------
# Per-idx range / sanity checks
# ---------------------------------------------------------------------------


# (lo, hi) soft bounds per kind. Values outside are flagged but do not fail.
# GRIPPER is intentionally absent: gripper encodings vary across policies, so we
# only report its observed range (min/max/mean/std) and skip any bounds check.
KIND_BOUNDS = {
    "JOINT_ABS": (-3.5, 3.5),  # radians, a bit over ±π
    "ABS_JOINT": (-3.5, 3.5),
    "JOINT_REL": (-1.0, 1.0),  # delta radians per chunk step
    "EEF_ABS": (-2.0, 2.0),  # meters for xyz, [-1,1] for quat — both fit
}
HARD_ABS = 1e4  # |x| above this is flagged as failure (garbage)
JUMP_FROM_STATE = 0.5  # rad — flag if action[0]_d differs from state_d by more than this


def _flatten_to_2d(a):
    """Return (n_samples, n_dims) view of arr, with last axis treated as dims."""
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a.reshape(-1, a.shape[-1])


def per_idx_summary(arr, label, *, kind=None, state=None, max_dims=64):
    """Print per-dimension stats and return True if no hard-fail issues found."""
    a = np.asarray(arr)
    if a.dtype.kind not in "fiub":
        warn(f"{label}: non-numeric dtype={a.dtype}, skipping per-idx checks")
        return True

    flat = _flatten_to_2d(a)
    n, d = flat.shape
    show = min(d, max_dims)

    soft_lo, soft_hi = KIND_BOUNDS.get(kind, (None, None))
    bounds_str = f"soft=[{soft_lo:+.2f}, {soft_hi:+.2f}]" if soft_lo is not None else "range-only (no bounds)"

    section(ICON_STAT, f"{label}")
    info(f"shape={tuple(a.shape)}  dtype={a.dtype}  samples={n}  dims={d}  kind={kind!r}  {bounds_str}")
    print()
    header = f"  {'idx':>3}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}   flags"
    print(header)
    print("  " + "─" * (len(header) - 2))

    fail_hard = False
    n_warn = 0
    for i in range(show):
        col = flat[:, i]
        if col.dtype.kind == "f":
            nan_n = int(np.isnan(col).sum())
            inf_n = int(np.isinf(col).sum())
            finite = col[np.isfinite(col)]
        else:
            nan_n = inf_n = 0
            finite = col

        flags = []
        if nan_n:
            flags.append(f"NaN×{nan_n}!")
            fail_hard = True
        if inf_n:
            flags.append(f"Inf×{inf_n}!")
            fail_hard = True

        if finite.size == 0:
            print(f"  {i:>3}  {'-':>10}  {'-':>10}  {'-':>10}  {'-':>10}   non-finite!")
            continue

        lo, hi = float(finite.min()), float(finite.max())
        mean, std = float(finite.mean()), float(finite.std())
        absmax = max(abs(lo), abs(hi))

        if absmax > HARD_ABS:
            flags.append(f"|x|>{HARD_ABS:g}!")
            fail_hard = True
        elif soft_lo is not None and (lo < soft_lo or hi > soft_hi):
            flags.append(f"OOB[{soft_lo:+.2f},{soft_hi:+.2f}]")
            n_warn += 1

        if n > 1 and std == 0:
            flags.append("const")
            n_warn += 1

        if state is not None and i < len(state):
            try:
                delta = float(col[0]) - float(state[i])
                if abs(delta) > JUMP_FROM_STATE:
                    flags.append(f"Δs={delta:+.3f}")
                    n_warn += 1
            except (TypeError, ValueError):
                pass

        print(f"  {i:>3}  {lo:>+10.4f}  {hi:>+10.4f}  {mean:>+10.4f}  {std:>10.4f}   {' '.join(flags)}")

    if d > show:
        print(f"  ... ({d - show} more dims hidden; pass --max-dims to see)")

    print()
    if fail_hard:
        fail(f"{label}: hard-fail flags present (NaN / Inf / |x|>{HARD_ABS:g})")
    elif n_warn:
        warn(f"{label}: {n_warn} soft warning(s) (OOB / const / Δstate)")
    else:
        ok(f"{label}: all {d} dims clean")
    return not fail_hard


def chunk_continuity(arrs, label, *, max_step=0.5):
    """Print and validate frame-to-frame deltas for a stack of same-shape arrays."""
    if len(arrs) < 2:
        return True
    try:
        stacked = np.stack([np.asarray(a) for a in arrs])
    except ValueError:
        warn(f"{label}: cannot stack chunks (shape mismatch)")
        return True
    if stacked.dtype.kind != "f":
        return True
    deltas = np.abs(np.diff(stacked.reshape(stacked.shape[0], -1), axis=0))
    if deltas.size == 0:
        return True
    max_delta = float(deltas.max())
    mean_delta = float(deltas.mean())
    msg = f"{label} continuity: mean Δ={mean_delta:.4f}  max Δ={max_delta:.4f}"
    (warn if max_delta > max_step * 4 else info)(msg + ("  ← jumpy!" if max_delta > max_step * 4 else ""))
    return True


# ---------------------------------------------------------------------------
# Generic NaN/Inf scan
# ---------------------------------------------------------------------------


def _flatten_arrays(obj):
    if isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _flatten_arrays(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_arrays(v)
    elif isinstance(obj, (np.ndarray, np.generic)):
        yield np.asarray(obj)
    elif isinstance(obj, (int, float)):
        yield np.asarray(obj)


def validate(arrays, label):
    arrays = list(arrays)
    if not arrays:
        fail(f"{label}: no numeric arrays in response")
        return False
    bad_nan = any(np.any(np.isnan(a)) for a in arrays if a.dtype.kind == "f")
    bad_inf = any(np.any(np.isinf(a)) for a in arrays if a.dtype.kind == "f")
    shapes = [tuple(a.shape) for a in arrays[:6]]
    info(f"{label}: {len(arrays)} arrays scanned, sample shapes={shapes}")
    if bad_nan or bad_inf:
        fail(f"{label}: NaN={bad_nan} Inf={bad_inf}")
        return False
    return True


# ---------------------------------------------------------------------------
# Response checker
# ---------------------------------------------------------------------------


def _state_for_corobot(payload):
    """Best-effort extract per-output input states from a corobot payload.

    arm_joint_states is typically left+right concatenated; we try common splits
    and just pass through whatever exists. Returns dict[name -> 1D array | None].
    """
    states = (payload.get("params") or {}).get("states") or {}
    arm = np.asarray(states.get("arm_joint_states") or [])
    waist = np.asarray(states.get("waist_joint_states") or [])
    head = np.asarray(states.get("head_joint_states") or [])
    grip = np.asarray(states.get("gripper_states") or [])

    out = {"waist": waist if waist.size else None, "head": head if head.size else None}
    if arm.size and arm.size % 2 == 0:
        half = arm.size // 2
        out["left_arm"] = arm[:half]
        out["right_arm"] = arm[half:]
    else:
        out["left_arm"] = out["right_arm"] = None
    if grip.size >= 2:
        out["left_effector"] = grip[:1]
        out["right_effector"] = grip[1:2]
    else:
        out["left_effector"] = out["right_effector"] = None
    return out


def validate_action_schema(result):
    """Check the required chunk structure consumed by CoRobotPolicy._parse_result."""
    arrays = {}
    kinds = []
    for key in ("left_arm", "right_arm", "left_effector", "right_effector"):
        values = result.get(key)
        if key.endswith("arm"):
            if not isinstance(values, dict) or "values" not in values:
                fail(f"{key}: required action block must contain 'values'")
                return False
            kind = values.get("kind", "JOINT_ABS")
            if kind not in ("JOINT_ABS", "EEF_ABS"):
                fail(f"{key}: unsupported action kind {kind!r}")
                return False
            kinds.append(kind)
            values = values["values"]
        try:
            arr = np.asarray(values)
        except (TypeError, ValueError):
            fail(f"{key}: expected a rectangular numeric action chunk")
            return False
        if arr.ndim != 2 or arr.size == 0 or arr.dtype.kind not in "fiub":
            fail(f"{key}: expected a non-empty numeric [steps, dimensions] action chunk")
            return False
        if key.endswith("arm") and kind == "EEF_ABS" and arr.shape[1] != 7:
            fail(f"{key}: EEF_ABS requires 7 dimensions (xyz + xyzw quaternion)")
            return False
        arrays[key] = arr
    if kinds[0] != kinds[1]:
        fail("left_arm/right_arm: action kinds must match")
        return False
    if len({arr.shape[0] for arr in arrays.values()}) != 1:
        fail("required action chunks must have matching step counts")
        return False
    frame = result["left_arm"].get("base_link") or result.get("base_link") or "base_link"
    if frame not in ("base_link", "arm_base_link"):
        fail(f"unsupported EEF frame: {frame!r}")
        return False
    return True


def check_corobot(ws, payload, max_dims):
    t0 = time.time()
    ws.send(_pack(payload))
    resp = ws.recv()
    dt = (time.time() - t0) * 1000
    if isinstance(resp, str):
        fail(f"server returned string: {resp[:200]}")
        return False
    result = _unpack(resp)
    section(ICON_RESP, f"corobot response  {dt:.1f} ms")
    if not isinstance(result, dict):
        fail("response must be a corobot JSON-RPC dict")
        return False
    info(f"keys: {list(result.keys())}")
    if result.get("error"):
        fail(f"server error: {result['error']}")
        return False
    inner = result.get("result")
    if not isinstance(inner, dict):
        fail(f"response missing 'result' dict. keys={list(result.keys())}")
        return False
    info(f"result keys: {list(inner.keys())}")
    if not validate_action_schema(inner):
        return False

    state_map = _state_for_corobot(payload)
    passed = True

    for key in ("left_arm", "right_arm", "waist", "head"):
        sub = inner.get(key)
        if not isinstance(sub, dict) or "values" not in sub:
            continue
        arr = np.asarray(sub["values"])
        if arr.size == 0:
            info(f"{key}: empty values, skipping")
            continue
        kind = sub.get("kind", "JOINT_ABS")
        passed &= per_idx_summary(arr, key, kind=kind, state=state_map.get(key), max_dims=max_dims)
        if arr.ndim >= 2 and arr.shape[0] > 1:
            chunk_continuity([arr[i] for i in range(arr.shape[0])], f"{key}")

    for key in ("left_effector", "right_effector"):
        sub = inner.get(key)
        if sub is None:
            continue
        arr = np.asarray(sub if not isinstance(sub, dict) else sub.get("values", []))
        if arr.size == 0:
            info(f"{key}: empty, skipping")
            continue
        # Gripper: report observed range only — no bounds (kind unset) and no
        # jump-from-state check (state=None), since encodings vary per policy.
        passed &= per_idx_summary(arr, key, kind="GRIPPER", state=None, max_dims=max_dims)

    return passed and validate(_flatten_arrays(inner), "result")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pkl_path",
        nargs="?",
        default=DEFAULT_PAYLOAD,
        help="corobot payload pkl (default: bundled corobot_payload.pkl)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8999)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--max-dims", type=int, default=64, help="max idx rows to print per array")
    args = ap.parse_args()

    payload = load(args.pkl_path)

    uri = f"ws://{args.host}:{args.port}"
    ws = connect(uri)

    ok_all = True
    latencies = []
    for i in range(args.iters):
        print()
        print(RULE_HEAVY)
        print(f"{ICON_ITER}  Iteration {i + 1} / {args.iters}")
        print(RULE_HEAVY)
        t0 = time.time()
        ok_all &= check_corobot(ws, payload, args.max_dims)
        latencies.append((time.time() - t0) * 1000)

    try:
        ws.close()
    except Exception:
        pass

    print()
    print(RULE_HEAVY)
    if len(latencies) > 1:
        lat = np.asarray(latencies)
        print(
            f"{ICON_TIME} latency over {len(latencies)} iters (ms): "
            f"min={lat.min():.1f}  mean={lat.mean():.1f}  "
            f"p95={np.percentile(lat, 95):.1f}  max={lat.max():.1f}"
        )
    if ok_all:
        ok("PASS — inference server reachable and response valid")
    else:
        fail("FAIL — see warnings above")
    print(RULE_HEAVY)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
