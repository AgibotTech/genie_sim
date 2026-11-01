#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pico teleoperation client for an initialized GenieSim server.

Use init_task.py to choose the task and backends before connecting.
Run with --stub --steps 20 for a smoke test without a headset."""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from rlinf_geniesim.comm import make_client
from rlinf_geniesim.teleop.safety import format_nonfinite, nonfinite_report
from rlinf_geniesim.teleop.eval_hud import InstructionSampler, ResetChecker, draw_hud, resolve_gen_pool
from rlinf_geniesim.teleop.ik_bridge import TeleopIKBridge, build_full_action
from rlinf_geniesim.teleop.pico_signal import PicoTeleopSource

# Import robot configuration in main so viewer/reset helpers remain usable
# without the teleoperation runtime.

# Used only when the schema lacks a complete measured head/waist joint set.
WAIST_INIT = [0.0] * 5
HEAD_INIT = [0.0, 0.0, 0.0]


def _head_waist_from_state(joint_names, full_q):
    """Read the measured head and waist joints used to initialize IK."""
    head = [float(q) for n, q in zip(joint_names, full_q) if "head_joint" in n]
    waist = [float(q) for n, q in zip(joint_names, full_q) if "body_joint" in n]
    if len(head) != 3 or len(waist) != 5:
        print(
            f"[teleop] WARN: head/waist joint count is {len(head)}/{len(waist)} (expected 3/5), "
            f"IK bridge is falling back to hardcoded zeros -- the base_link->arm_base_link "
            f"transform will be off",
            flush=True,
        )
        return list(HEAD_INIT), list(WAIST_INIT)
    return head, waist


def _robot_cfg_of(joint_names):
    """Select the IK robot model from the gripper joints in the server schema."""
    names = set(joint_names)
    if any("idx41_gripper" in n or "idx81_gripper" in n for n in names):
        return "G2_omnipicker"
    if any("idx32_gripper" in n or "idx72_gripper" in n for n in names):
        return "G2_90d"
    return "G2_omnipicker"


MAX_NONFINITE_RESETS = 3


# --------------------------------------------------------------------------- #
# Live MJPEG viewer using Pillow.
# --------------------------------------------------------------------------- #
class MjpegViewer:
    """Serve camera frames as MJPEG alongside an optional task-control UI.

    Push ``(C, H, W, 3)`` uint8 stacks to form one wide frame."""

    def __init__(self, port: int, control=None):
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()
        viewer = self
        self.control = control

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                if viewer.control is None or not self.path.startswith("/api/"):
                    self.send_error(404)
                    return
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    body = self.rfile.read(n) if n else b""
                    from rlinf_geniesim.teleop.web_control import handle_api

                    code, payload = handle_api(viewer.control, self.path[len("/api") :], body)
                except Exception as exc:  # noqa: BLE001
                    code, payload = 500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                raw = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):

                if viewer.control is not None and self.path in ("/", "/ui"):
                    from rlinf_geniesim.teleop.web_control import UI_HTML

                    raw = UI_HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    try:
                        self.wfile.write(raw)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while True:
                    with viewer._lock:
                        buf = viewer._jpeg
                    if buf is not None:
                        try:
                            self.wfile.write(b"--frame\r\nContent-Type: " b"image/jpeg\r\n\r\n")
                            self.wfile.write(buf)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    time.sleep(0.03)

        self._srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def push(self, cams: np.ndarray, cam_heights=None, hud: dict | None = None):
        # cams: (C, H, W, 3) uint8 -> crop per-camera height, hstack, JPEG.
        # hud supplies optional draw_hud() values; None leaves the frame unchanged.
        from PIL import Image

        tiles = []
        for c in range(cams.shape[0]):
            h = int(cam_heights[c]) if cam_heights else cams.shape[1]
            tiles.append(cams[c, :h])
        hmax = max(t.shape[0] for t in tiles)

        padded = []
        for t in tiles:
            gap = hmax - t.shape[0]
            top = gap // 2
            padded.append(np.pad(t, ((top, gap - top), (0, 0), (0, 0))))
        frame = np.concatenate(padded, axis=1)
        img = Image.fromarray(frame, "RGB")
        if hud is not None:

            _DRAW_KEYS = ("instruction", "episode", "step", "reward", "success")
            draw_hud(img, **{k: hud[k] for k in _DRAW_KEYS if k in hud})
        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=80)
        with self._lock:
            self._jpeg = bio.getvalue()


# --------------------------------------------------------------------------- #
# Stub Pico source (smoke test without a headset)
# --------------------------------------------------------------------------- #
class StubPico:
    """Yields an engaged left-hand frame nudging +x, for N steps, then None."""

    def __init__(self, steps: int):
        self._left = steps

    def poll(self):
        if self._left <= 0:
            return None
        self._left -= 1
        return {
            "left": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "right": [0.0] * 7,
            "l_on": True,
            "r_on": False,
            "l_grip": 0.0,
            "r_grip": 0.0,
            "l_reset": False,
        }

    def close(self):
        pass


# --------------------------------------------------------------------------- #
def _arm_indices(joint_names):
    from geniesim_teleop.config.robot_interface import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES

    idx = {n: i for i, n in enumerate(joint_names)}
    li = [idx[n] for n in LEFT_ARM_JOINT_NAMES if n in idx]
    ri = [idx[n] for n in RIGHT_ARM_JOINT_NAMES if n in idx]
    if len(li) != 7 or len(ri) != 7:
        raise RuntimeError(
            f"arm joints not found in schema: left={len(li)}/7 right={len(ri)}/7. "
            f"Check LEFT/RIGHT_ARM_JOINT_NAMES vs schema joint_names."
        )
    return li + ri  # 14 indices


def _qpos(states_row):
    return np.asarray(states_row)[0::2].astype(np.float64)  # qpos = states[0::2]


def _make_gen_rng(seed: int):
    """Use an independent deterministic stream for instance selection."""
    return np.random.default_rng([int(seed), 0x67656E])


def _rebuild_after_reload(client, new_dims, new_schema) -> dict:
    """Replace the joint and camera layout, then obtain a reset state for reseeding."""
    nsc = json.loads(new_schema) if isinstance(new_schema, str) else (new_schema or {})
    jn = list(nsc.get("joint_names") or [])
    if not jn:
        raise RuntimeError("reload schema has no joint_names")
    n_env = int(new_dims.num_envs)

    out = client.reset(np.ones((n_env,), np.float32))
    return {
        "joint_names": jn,
        "arm_idx": _arm_indices(jn),
        "n_env": n_env,
        "cam_heights": tuple(getattr(new_dims, "cam_heights", ()) or ()),
        "out": out,
    }


def _do_reset(client, n_env, sampler, checker, ep, args, gen_pool=None, gen_rng=None):
    """Reset with the CLI instruction sampler and optional per-episode instance selection."""
    if getattr(args, "gen", False):
        inst = int(gen_rng.choice(gen_pool))
        instr = sampler.sample_for(inst)
        out = client.reset(
            np.ones((n_env,), np.float32),
            seeds=[args.seed + ep] * n_env,
            instances=[inst] * n_env,
            instructions=[instr[0]] * n_env,
        )
        print(f"[eval] episode {ep}: instance={inst} instr id={instr[0]} " f'"{instr[1]}"', flush=True)
        return out, instr[1], inst, instr[0]

    instr = sampler.sample() if sampler else None
    seeds = [args.seed + ep if args.vary_seed else args.seed] * n_env if sampler else None
    instance = int(args.instance) if sampler else None
    out = client.reset(
        np.ones((n_env,), np.float32),
        seeds=seeds,
        instances=[instance] * n_env if instance is not None else None,
        instructions=[instr[0]] * n_env if instr else None,
    )
    if instr:
        print(f'[eval] episode {ep}: instr id={instr[0]} "{instr[1]}"', flush=True)
    if checker and not args.vary_seed:
        qpos = np.asarray(out.states)[0][0::2]
        rep = checker.observe(qpos, np.asarray(out.object_poses)[0])
        if rep:
            print(rep, flush=True)
    return out, (instr[1] if instr else ""), instance, (instr[0] if instr else None)


def _guard_recover(client, n_env, identity):
    """Repeat the current instance, instruction and seed for every environment."""

    def _fill(key):
        v = identity.get(key)
        return None if v is None else [int(v)] * n_env

    return client.reset(
        np.ones((n_env,), np.float32),
        seeds=_fill("seed"),
        instances=_fill("instance_id"),
        instructions=_fill("instruction_id"),
    )


def _handle_nonfinite(client, n_env, rep, ctx, attempts, printer=print):
    """Retry the current episode identity, stopping after repeated invalid state."""
    n = attempts + 1
    identity = ctx.get("identity") or {}
    display_ctx = {key: value for key, value in ctx.items() if key != "identity"}
    display_ctx.update(identity)
    printer(format_nonfinite(rep, display_ctx), flush=True)
    if n >= MAX_NONFINITE_RESETS:
        raise SystemExit(f"[teleop] state remains non-finite after {n} checks; reload the task before continuing")
    printer(f"[teleop] recovery reset {n}/{MAX_NONFINITE_RESETS}", flush=True)
    return _guard_recover(client, n_env, identity), n


def _step_and_check(client, action, joint_names, printer=print):
    """Step a live task and check its state before updating the IK seed."""
    try:
        out = client.step(action)
    except RuntimeError as exc:
        if "NO_ACTIVE_TASK" not in str(exc):
            raise
        printer("[teleop] server has no active task; reload through the web console", flush=True)
        return None, None
    return out, nonfinite_report(out, env=0, joint_names=joint_names, action=action[0])


def _recover_reset_until_finite(client, n_env, out, identity, ctx, attempts, joint_names, printer=print, action=None):
    """Check every reset response before reseeding IK; retries preserve the episode identity."""
    rep = nonfinite_report(out, env=0, joint_names=joint_names, action=action)
    while rep is not None:
        out, attempts = _handle_nonfinite(client, n_env, rep, {"identity": identity, **ctx}, attempts, printer=printer)
        rep = nonfinite_report(out, env=0, joint_names=joint_names)
    return out, attempts


def _check_startup_reset(out, joint_names, printer=print):
    """Reject invalid initial state before constructing the IK solver."""
    rep = nonfinite_report(out, 0, joint_names)
    if rep is None:
        return
    printer(format_nonfinite(rep, {"stage": "startup reset"}), flush=True)
    raise SystemExit("[teleop] startup reset returned non-finite state; reload the task before starting teleoperation")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--addr", default="tcp://127.0.0.1:5601", help="simulation server ZMQ address")
    ap.add_argument("--host_ip", default="", help="VR UDP bind ip ('' = all)")
    ap.add_argument("--port", type=int, default=8080, help="VR UDP port")
    ap.add_argument("--viewer-port", type=int, default=8091)
    ap.add_argument("--max-ik-iters", type=int, default=8)
    ap.add_argument("--stub", action="store_true", help="stub Pico (smoke test)")
    ap.add_argument("--steps", type=int, default=0, help="stop after N steps (0=inf)")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="trace Pico input, per-arm command changes and " "post-step tracking errors (every 15 steps)",
    )
    ap.add_argument(
        "--task",
        default="",
        help="eval mode: sub_task_name for InstructionSampler " "(empty = eval mode off, default behavior unchanged)",
    )
    ap.add_argument("--instance", type=int, default=0, help="eval mode: instance_id under llm_task/<task>/")
    ap.add_argument("--seed", type=int, default=0, help="eval mode: InstructionSampler seed + reset seed")
    ap.add_argument(
        "--vary-seed",
        action="store_true",
        help="reset seed = --seed + episode index (instead of a "
        "fixed --seed); skips the reset-consistency checker, "
        "since the reset pose legitimately differs per episode",
    )
    ap.add_argument(
        "--gen",
        action="store_true",
        help="training-style generalization on reset (requires "
        "--task): each reset draws an instance from the "
        "--instance group (widen with --superset), a "
        "per-episode seed, and that instance's own sampled "
        "instruction; disables the reset-consistency checker",
    )
    ap.add_argument(
        "--superset",
        default="",
        help="--gen only: comma-separated instance ids widening "
        "the pool across asset groups, e.g. 0,1,10 "
        "(default '' = pool is --instance's topology group)",
    )
    args = ap.parse_args()

    from geniesim_teleop.config.robot_interface import BODY_JOINT_NAMES

    superset_list: "list[int] | None" = None
    if args.superset:
        try:
            superset_list = [int(s.strip()) for s in args.superset.split(",")]
        except ValueError:
            raise SystemExit(
                f"--superset must be a comma-separated list of integers " f"(e.g. 0,1,10); got {args.superset!r}"
            )
    # Instance sampling requires the task-specific instruction table.
    if args.gen and not args.task:
        raise SystemExit(
            "--gen requires --task (the generalization pool is a task's "
            "instance group; without a task there is nothing to sample)"
        )

    # --- connect (retry: shm segment may lag the serving flag) ---
    client = dims = schema = None
    for _ in range(40):
        try:
            client = make_client("zmq", address=args.addr)
            dims, schema = client.handshake()
            break
        except Exception:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            client = None
            time.sleep(1)
    if dims is None:
        raise SystemExit(
            f"failed to attach to sim at {args.addr} " "(start the server and initialize its task with init_task.py)"
        )
    sc = json.loads(schema) if isinstance(schema, str) else (schema or {})
    joint_names = list(sc.get("joint_names") or [])
    if not joint_names:
        raise SystemExit("schema has no joint_names")
    n_env = int(dims.num_envs)
    cam_heights = tuple(getattr(dims, "cam_heights", ()) or ())
    arm_idx = _arm_indices(joint_names)

    # --- eval mode (opt-in via --task): instruction sampler + reset checker ---
    sampler = checker = None
    if args.task:
        sampler = InstructionSampler(args.task, args.instance, args.seed)
        checker = ResetChecker()
        print(
            f"[eval] task={args.task!r} instance={args.instance} "
            f"seed={args.seed} #instructions={len(sampler.table)}",
            flush=True,
        )

    # --- --gen (opt-in, requires --task): resolve the instance pool once at
    # startup + a dedicated rng for instance draws (kept separate from the
    # sampler's own rng so the instruction-draw sequence stays independent of
    # pool size) ---
    gen_pool = gen_rng = None
    if args.task and args.gen:
        gen_pool = resolve_gen_pool(args.task, args.instance, superset_list)
        gen_rng = _make_gen_rng(args.seed)
        print(f"[gen] instance pool: {gen_pool}", flush=True)
        print("[gen] reset consistency checks disabled for randomized episodes", flush=True)

    # --- init: reset, capture the held init qpos + arm seed ---
    ep = 0

    cur_task_name = args.task
    identity = {}
    use_session_reset = False
    nonfinite_attempts = 0
    out, cur_instr_text, inst, cur_instr_id = _do_reset(client, n_env, sampler, checker, ep, args, gen_pool, gen_rng)

    _check_startup_reset(out, joint_names)
    if inst is not None:
        cur_instr_text = f"inst {inst} | {cur_instr_text}"
    init_full = _qpos(np.asarray(out.states)[0])  # constant hold vector
    arm_q14 = [float(init_full[i]) for i in arm_idx]
    arm_init = list(arm_q14)  # Reset pose for per-arm debug displacement.

    # held body/waist pose -> base_link->arm_base_link frame for the delta math
    name_idx = {n: i for i, n in enumerate(joint_names)}
    body_q = {n: float(init_full[name_idx[n]]) for n in BODY_JOINT_NAMES if n in name_idx}
    _head3, _waist5 = _head_waist_from_state(joint_names, init_full)
    bridge = TeleopIKBridge(
        arm_q14, _head3, _waist5, robot_cfg=_robot_cfg_of(joint_names), body_joint_q=body_q, joint_names=joint_names
    )

    from rlinf_geniesim.teleop.web_control import TeleopControl

    _ui = {"steps": 0}

    _box = {"instr": ""}

    def _reseed_after_reset(out, instance_id=None, instruction_id=None, seed=None):
        """Replace hold targets and clear clutch references after every reset."""
        nonlocal init_full, arm_q14, arm_init, last_action, last_on, last_success
        init_full = _qpos(np.asarray(out.states)[0])
        arm_q14 = [float(init_full[i]) for i in arm_idx]
        arm_init = list(arm_q14)
        bridge._ref.clear()
        last_on = {"left": False, "right": False}
        last_action = np.tile(init_full.astype(np.float32), (n_env, 1))
        last_success = False

        identity.clear()
        identity.update(instance_id=instance_id, instruction_id=instruction_id, seed=seed)

    def _on_reload(new_dims, new_schema, new_sub_task_name):
        nonlocal joint_names, arm_idx, n_env, cam_heights, name_idx, bridge
        nonlocal cur_task_name, nonfinite_attempts, sampler, checker, gen_pool, cur_instr_text, use_session_reset
        st = _rebuild_after_reload(client, new_dims, new_schema)
        joint_names = st["joint_names"]
        arm_idx = st["arm_idx"]
        n_env = st["n_env"]
        cam_heights = st["cam_heights"]

        cur_task_name = new_sub_task_name
        # A task switch invalidates the CLI sampler and its instance pool.
        sampler = checker = gen_pool = None
        use_session_reset = True
        cur_instr_text = ""
        _box["instr"] = ""

        name_idx = {n: i for i, n in enumerate(joint_names)}

        out, nonfinite_attempts = _recover_reset_until_finite(
            client,
            n_env,
            st["out"],
            {},
            {"step": steps, "episode": ep, "sub_task": cur_task_name or "(no --task)"},
            nonfinite_attempts,
            joint_names,
        )
        _reseed_after_reset(out)

        _h3, _w5 = _head_waist_from_state(joint_names, init_full)
        bridge = TeleopIKBridge(
            arm_q14,
            _h3,
            _w5,
            robot_cfg=_robot_cfg_of(joint_names),
            body_joint_q={n: float(init_full[name_idx[n]]) for n in BODY_JOINT_NAMES if n in name_idx},
            joint_names=joint_names,
        )
        print(
            f"[ctl] reload complete: num_envs={n_env} joints={len(joint_names)} "
            f"arm_idx={len(arm_idx)}; hold targets and IK references updated",
            flush=True,
        )

    def _on_web_reset(out, instance_id, instruction_id, text, seed):
        nonlocal ep, nonfinite_attempts, use_session_reset, cur_instr_text
        use_session_reset = True
        ep += 1

        out, nonfinite_attempts = _recover_reset_until_finite(
            client,
            n_env,
            out,
            {"instance_id": instance_id, "instruction_id": instruction_id, "seed": seed},
            {"step": steps, "episode": ep, "sub_task": cur_task_name or "(no --task)"},
            nonfinite_attempts,
            joint_names,
        )
        _reseed_after_reset(out, instance_id, instruction_id, seed=seed)
        cur_instr_text = text
        _box["instr"] = f"inst {instance_id} | {text}" if text else f"inst {instance_id} | (instruction unavailable)"
        print(f"[ctl] reset -> instance={instance_id} instr id={instruction_id} " f'"{text}"', flush=True)

    control = TeleopControl(
        client,
        n_env,
        on_reload=_on_reload,
        state_fn=lambda: dict(_ui),
        sub_task_name=args.task,
        instance_id=args.instance,
        on_reset=_on_web_reset,
    )

    last_on = {"left": False, "right": False}
    last_reset = False
    last_action = np.tile(init_full.astype(np.float32), (n_env, 1))
    steps = 0
    last_success = False  # eval mode: for the False->True SUCCESS print

    def _sync_current(inst, instr_id, seed):
        """Expose the current reset identity to the web console."""
        control.set_current(
            instance_id=(inst if use_session_reset else args.instance if inst is None else inst),
            instruction_id=instr_id,
            seed=seed,
        )

    _sync_current(inst, cur_instr_id, args.seed if sampler else None)

    _reseed_after_reset(out, inst, cur_instr_id, args.seed if sampler else None)
    _box["instr"] = ""
    viewer = MjpegViewer(args.viewer_port, control=control)
    pico = StubPico(args.steps or 20) if args.stub else PicoTeleopSource(args.host_ip, args.port)

    print(
        f"[teleop] serving viewer on http://127.0.0.1:{args.viewer_port} ; "
        f"{n_env} env, {len(joint_names)} joints; grip trigger, keyOne=reset",
        flush=True,
    )

    eval_off_warned = False  # eval mode: print the "success=None" warning once
    try:
        while True:

            _ui["steps"] = steps
            _r = control.drain()
            if _r is not None and not _r.get("ok", True):
                print(f"[ctl] command failed: {_r.get('error')}", flush=True)
            if not control.session_ready:
                # Reload must replace stale layout/hold targets after an init or reset failure.
                time.sleep(0.05)
                continue
            if _r is not None and not _r.get("ok", True):
                # An ordinary command failure still only skips this tick -- it does not pause
                # the currently-live task.
                continue
            frame = pico.poll()
            debug_tick = args.debug and steps % 15 == 0
            if debug_tick:
                debug_seed = np.asarray(arm_q14, dtype=float).copy()
                # VRServer retains its last packet; neither of these values
                # establishes that a new packet arrived during this step.
                input_status = (
                    "source=stub"
                    if args.stub
                    else f"udp_ever_received={pico.udp_alive} " f"parsed_polls={pico.frames} packet_freshness=unknown"
                )
                if frame is None:
                    print(
                        f"[dbg] step={steps} {input_status} frame=None "
                        f"(no parsed Pico input on :{args.port}; holding)",
                        flush=True,
                    )
                else:
                    lp = [round(float(v), 4) for v in frame["left"]]
                    rp = [round(float(v), 4) for v in frame["right"]]
                    raw = frame.get("_raw_trig", ("?", "?", "?", "?"))
                    print(
                        f"[dbg] step={steps} {input_status} | "
                        f"engage L={frame['l_on']} R={frame['r_on']} | "
                        f"mapped_pose(xyz_m,xyzw) L={lp} R={rp} | "
                        f"raw_handTrig L={raw[2]} R={raw[3]} "
                        f"raw_indexTrig L={raw[0]} R={raw[1]} | "
                        f"grip L={frame['l_grip']:.2f} R={frame['r_grip']:.2f} "
                        f"reset={frame['l_reset']}",
                        flush=True,
                    )
            if frame is None:  # no data -> hold
                out, rep = _step_and_check(client, last_action, joint_names)
            else:
                for hand, key in (("left", "l_on"), ("right", "r_on")):
                    if frame[key] and not last_on[hand]:  # engage rising edge
                        bridge.on_engage(hand, arm_q14)
                        if args.debug:
                            print(f"[dbg] ENGAGE {hand} (grip pressed, delta now " f"applied)", flush=True)
                    last_on[hand] = frame[key]
                if frame.get("l_reset") and not last_reset:
                    if args.debug:
                        print("[dbg] RESET (keyOne)", flush=True)
                    ep += 1
                    if use_session_reset:
                        out = _guard_recover(client, n_env, identity)
                        inst = identity.get("instance_id")
                        cur_instr_id = identity.get("instruction_id")
                    else:
                        out, cur_instr_text, inst, cur_instr_id = _do_reset(
                            client, n_env, sampler, checker, ep, args, gen_pool, gen_rng
                        )
                    _seed = (
                        identity.get("seed")
                        if use_session_reset
                        else (args.seed + ep if args.vary_seed or args.gen else args.seed) if sampler else None
                    )
                    _sync_current(inst, cur_instr_id, _seed)
                    _box["instr"] = ""
                    if inst is not None:
                        cur_instr_text = f"inst {inst} | {cur_instr_text}"

                    out, nonfinite_attempts = _recover_reset_until_finite(
                        client,
                        n_env,
                        out,
                        {"instance_id": inst, "instruction_id": cur_instr_id, "seed": _seed},
                        {"step": steps, "episode": ep, "sub_task": cur_task_name or "(no --task)"},
                        nonfinite_attempts,
                        joint_names,
                    )

                    _reseed_after_reset(out, inst, cur_instr_id, _seed)
                    last_reset = True
                    continue
                last_reset = bool(frame.get("l_reset"))
                joints16 = bridge.deltas_to_joints(arm_q14, frame, max_ik_iters=args.max_ik_iters)
                act0 = build_full_action(joint_names, joints16, init_full)
                last_action = np.tile(act0.astype(np.float32), (n_env, 1))
                out, rep = _step_and_check(client, last_action, joint_names)
            if out is None:  # Wait for an explicit reload after the server loses its task.
                control.session_ready = False
                control.last_error = "server has no active task; reload to resume"
                continue
            if rep is not None:

                out, nonfinite_attempts = _recover_reset_until_finite(
                    client,
                    n_env,
                    out,
                    identity,
                    {"step": steps, "episode": ep, "sub_task": cur_task_name or "(no --task)"},
                    nonfinite_attempts,
                    joint_names,
                    action=last_action[0],
                )
                _reseed_after_reset(
                    out, identity.get("instance_id"), identity.get("instruction_id"), identity.get("seed")
                )
                steps += 1
                continue
            nonfinite_attempts = 0
            if debug_tick:
                # Use this step's response before updating the IK seed below.
                # Each value is the max absolute error over that arm's 7 joints;
                # split the arms so right-arm motion cannot conceal a dead left.
                debug_cmd = np.asarray(last_action[0])[arm_idx]
                debug_actual = _qpos(np.asarray(out.states)[0])[arm_idx]
                debug_metrics = []
                for label, error in (
                    ("cmd_delta_from_seed", debug_cmd - debug_seed),
                    ("tracking_error", debug_actual - debug_cmd),
                    ("moved_from_reset", debug_actual - np.asarray(arm_init)),
                ):
                    debug_metrics.append(
                        f"{label} L={np.max(np.abs(error[:7])):.5f} " f"R={np.max(np.abs(error[7:])):.5f}"
                    )
                ik_mode = "solve" if frame is not None and (frame.get("l_on") or frame.get("r_on")) else "hold"
                print(
                    f"[dbg] step={steps} env=0 after_step IK={ik_mode} " f"max_abs_rad: {' | '.join(debug_metrics)}",
                    flush=True,
                )
            # Freeze the hold seed while disengaged so servo tracking errors do not accumulate.
            if frame is not None and (frame.get("l_on") or frame.get("r_on")):
                arm_q14 = [float(_qpos(np.asarray(out.states)[0])[i]) for i in arm_idx]
            hud = None
            if cur_task_name:
                cur_success = None if out.success is None else bool(np.asarray(out.success)[0])
                if cur_success is None and not eval_off_warned:
                    print("[eval] WARN: server checker is disabled (success=None) — " "EVAL OFF", flush=True)
                    eval_off_warned = True
                if cur_success and not last_success:
                    print(f"[eval] SUCCESS at step {steps}", flush=True)
                last_success = bool(cur_success)

                hud = {
                    "instruction": (_box["instr"] or cur_instr_text),
                    "episode": ep,
                    "step": steps,
                    "reward": float(np.asarray(out.rewards)[0]),
                    "success": cur_success,
                }

            control.set_hud(hud)
            if out.images is not None:
                try:
                    viewer.push(np.asarray(out.images)[0], cam_heights, hud=hud)
                except Exception as e:
                    print(f"[teleop] viewer.push failed, skipping frame: {e}", flush=True)

            steps += 1
            if args.steps and steps >= args.steps:
                print(f"[teleop] done {steps} steps", flush=True)
                break
    finally:
        pico.close()
        close_client = getattr(client, "close", None)
        if close_client is not None:
            close_client()
        # Publish a farewell frame so the viewer does not retain a stale
        # reward/success HUD after the client exits.
        try:
            gone = np.zeros((240, 640, 3), np.uint8)
            from PIL import Image, ImageDraw  # noqa: PLC0415

            img = Image.fromarray(gone, "RGB")
            d = ImageDraw.Draw(img)
            d.text(
                (20, 100), "TELEOP CLIENT EXITED — this frame is frozen; " "values above are stale", fill=(255, 80, 80)
            )
            import io as _io  # noqa: PLC0415

            bio = _io.BytesIO()
            img.save(bio, format="JPEG", quality=80)
            with viewer._lock:
                viewer._jpeg = bio.getvalue()
        except Exception:  # noqa: BLE001 — exit path must never mask the real error
            pass


if __name__ == "__main__":
    main()
