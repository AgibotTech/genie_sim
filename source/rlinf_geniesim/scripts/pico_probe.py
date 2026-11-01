#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Inspect raw Pico UDP packets without loading the teleoperation stack.

Stop teleoperation first, then run: python3 pico_probe.py --port 8080 --seconds 40."""
import argparse
import json
import socket
import time


def _pos_mag(sig):
    p = sig.get("position") if isinstance(sig, dict) else None
    if not isinstance(p, dict):
        return 0.0
    return max(abs(float(p.get("x", 0))), abs(float(p.get("y", 0))), abs(float(p.get("z", 0))))


def _fmt_slot(sig):
    if not isinstance(sig, dict):
        return f"<{type(sig).__name__}: {sig!r}>"
    p = sig.get("position") or {}
    ps = {k: round(float(v), 3) for k, v in p.items()} if isinstance(p, dict) else p
    return (
        f"pos={ps} handTrig={sig.get('handTrig')} "
        f"indexTrig={sig.get('indexTrig')} "
        f"keyOne={sig.get('keyOne')} keyTwo={sig.get('keyTwo')}"
    )


def _activity(sig):
    """Round-tripped snapshot of the fields that would drive teleop, for
    change detection: (|pos| bucket, handTrig>0.8, indexTrig bucket)."""
    if not isinstance(sig, dict):
        return None
    return (round(_pos_mag(sig), 2), float(sig.get("handTrig", 0)) > 0.8, round(float(sig.get("indexTrig", 0)), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seconds", type=float, default=40.0, help="wall-clock capture window")
    ap.add_argument("--idle-timeout", type=float, default=15.0, help="give up if no packet at all arrives in this long")
    a = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", a.port))
    s.settimeout(a.idle_timeout)
    print(
        f"[probe] listening UDP :{a.port} for {a.seconds:.0f}s. NOW: squeeze "
        f"the LEFT grip and move the LEFT controller for a few seconds, then "
        f"the RIGHT. I print only when a slot's values CHANGE.",
        flush=True,
    )

    t0 = time.monotonic()
    n = 0
    last_act = {0: object(), 1: object()}  # force first print
    # running maxima per slot, to answer "did this hand EVER produce data"
    mx = {0: {"pos": 0.0, "hand": 0.0, "idx": 0.0}, 1: {"pos": 0.0, "hand": 0.0, "idx": 0.0}}
    src = None
    sz_min = None
    sz_max = 0
    over_1024 = 0  # datagrams VRServer's recvfrom(1024) would truncate

    while time.monotonic() - t0 < a.seconds:
        try:
            data, addr = s.recvfrom(65535)
        except socket.timeout:
            print(
                f"[probe] {a.idle_timeout:.0f}s with NO packet at all — " f"headset stopped streaming to :{a.port}",
                flush=True,
            )
            break
        n += 1
        src = addr
        sz = len(data)
        sz_min = sz if sz_min is None else min(sz_min, sz)
        sz_max = max(sz_max, sz)
        if sz > 1024:
            over_1024 += 1
        if n == 1:
            print(
                f"[probe] first datagram is {sz} bytes "
                f"(VRServer reads only 1024 → "
                f"{'WOULD TRUNCATE' if sz > 1024 else 'fits'})",
                flush=True,
            )
        raw = data.decode("utf-8", errors="replace")
        try:
            obj = json.loads(raw.replace("False", "false").replace("True", "true"))
        except Exception as e:  # noqa: BLE001 - probe: report and continue
            print(f"[probe] #{n} NON-JSON ({e}): {raw[:300]}", flush=True)
            continue
        if not (isinstance(obj, (list, tuple)) and len(obj) >= 2):
            print(f"[probe] #{n} UNEXPECTED shape: {raw[:300]}", flush=True)
            continue

        for slot, tag in ((0, "L"), (1, "R")):
            sig = obj[slot]
            if isinstance(sig, dict):
                mx[slot]["pos"] = max(mx[slot]["pos"], _pos_mag(sig))
                mx[slot]["hand"] = max(mx[slot]["hand"], float(sig.get("handTrig", 0)))
                mx[slot]["idx"] = max(mx[slot]["idx"], float(sig.get("indexTrig", 0)))
            act = _activity(sig)
            if act != last_act[slot]:
                last_act[slot] = act
                dt = time.monotonic() - t0
                print(f"[probe] t={dt:4.1f}s {tag}(slot{slot}) CHANGED: " f"{_fmt_slot(sig)}", flush=True)

    print(f"\n[probe] ===== SUMMARY after {n} packets from {src} =====", flush=True)
    print(
        f"[probe] datagram size: min={sz_min} max={sz_max} bytes; "
        f"{over_1024}/{n} exceeded 1024 "
        f"(VRServer's recvfrom(1024) truncates those → JSON parse fails)",
        flush=True,
    )
    for slot, tag in ((0, "L"), (1, "R")):
        m = mx[slot]
        verdict = "ALL ZERO — this slot never produced any pose/trigger" if max(m.values()) == 0.0 else "produced data"
        print(
            f"[probe] {tag}(slot{slot}): max|pos|={m['pos']:.3f} "
            f"max_handTrig={m['hand']:.2f} max_indexTrig={m['idx']:.2f}  "
            f"-> {verdict}",
            flush=True,
        )
    print("[probe] done", flush=True)


if __name__ == "__main__":
    main()
