#!/usr/bin/env python3
"""G1/G2 pi0.5 bridge for the Dream policy HTTP contract."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from . import stateend
from .client import OpenPIPolicy
from .profiles import observation_profile
from .types import Observation, wm_state_to_policy_state

STATE_FORMAT = "dream_g2_wm_joints16"
# How many pi05 actions each WM frame consumes, per session chunk length:
# c16: 3 actions/frame (stride-3 decimation, 16x3=48 of the 50-action horizon
#      -- the validated offline configuration); c32: the full 50 actions
#      resampled across 32 frames (50/32, smooth interpolation, no freeze).
SPEED_BY_CHUNK = {16: 3.0, 32: 50.0 / 32.0}


class Bridge:
    def __init__(
        self,
        policy_url: str,
        speed: float | None,
        stitch_frames: int,
        chunk_length: int = 16,
    ):
        self.policy = OpenPIPolicy(policy_url)
        self.policy.reset()
        self.speed = speed
        self.stitch_frames = stitch_frames
        self.chunk_length = chunk_length
        self.lock = threading.Lock()  # serializes policy + mutable MuJoCo data

    def state_to_joints(self, state, step_index: int, profile=None) -> np.ndarray:
        if state is None:
            raise ValueError(
                "The selected example has no initial_state. Export a matching example "
                "or pass initial_state when creating the session."
            )
        state_format = profile.state_format if profile is not None else STATE_FORMAT
        if not isinstance(state, dict) or state.get("format") != state_format:
            raise ValueError(
                f"observation.state must be {{'format': '{state_format}', 'joints': [16]}}; "
                "create the session with initial_state (see export script)"
            )
        joints = np.asarray(state["joints"], dtype=np.float32)
        if joints.shape != (16,) or not np.isfinite(joints).all():
            raise ValueError("state joints must contain exactly 16 finite values")
        return joints

    def encoder_for(self, profile, state):
        key, size = ("head_waist", 4) if profile.robot == "g01" else ("head_body", 8)
        if key not in state:
            raise ValueError(f"initial_state requires {key} for {profile.id}")
        tail = np.asarray(state[key], dtype=np.float32)
        if tail.shape != (size,) or not np.isfinite(tail).all():
            raise ValueError(f"state {key} must contain {size} finite values")
        cls = stateend.StateEndEncoder if profile.robot == "g01" else stateend.G2StateEndEncoder
        return cls(tail)

    def predict(self, payload: dict) -> dict:
        started = time.monotonic()
        if payload.get("protocol") != "dream.policy.v1":
            raise ValueError("Unsupported policy protocol")
        observation = payload["observation"]
        state = observation.get("state")
        profile = observation_profile(observation.get("domain"), state)
        step_index = int(payload["step_index"])
        logging.info("bridge request task=%s step=%d", profile.id, step_index)
        # observation.chunk_length is the session's own choice (16 or 32);
        # follow it per request.  The CLI default covers platforms that do
        # not report it.
        chunk_length = observation.get("chunk_length")
        if chunk_length not in (None, 16, 32):
            raise ValueError(f"session chunk_length must be 16 or 32, got {chunk_length}")
        if chunk_length is None:
            chunk_length = self.chunk_length
        speed = (
            self.speed
            if self.speed is not None
            else (SPEED_BY_CHUNK[chunk_length] if profile.robot == "g2" else 1.0)
        )

        canvas = Image.open(io.BytesIO(base64.b64decode(observation["image_png_b64"])))
        # step3's splitter handles both the 640x600 seed frame and the 640x592
        # frames the model generates (the latent crop takes 8 rows off the
        # wrist strip, never off the head band).
        views = stateend.split_canvas(canvas, head_rows=400)
        task = observation.get("prompt") or ""

        joints = self.state_to_joints(state, step_index, profile)
        encoder = self.encoder_for(profile, state)
        prepared = time.monotonic()

        # step3 semantics: query until the fresh buffer covers chunk_length
        # ROWS (c16 + 50-action policy horizon -> exactly one query), then let
        # retime_actions spend the session's speed mapping per WM frame
        # (16 frames x 3 = 48 of 50; 32 frames x 1.5625 = 50, resampled).
        with self.lock:
            locked = time.monotonic()
            fresh = np.zeros((0, 16), dtype=np.float32)
            queries = 0
            anchor = joints
            while len(fresh) < chunk_length:
                policy_state = wm_state_to_policy_state(anchor)
                if profile.robot == "g01":
                    # G01 state is recorded in 0..120 travel degrees, while
                    # action feedback is 0..1. Match the standalone G01 loop.
                    policy_state[14:16] *= stateend.GRIPPER_TRAVEL_DEG
                chunk = np.asarray(
                    self.policy.infer(
                        Observation(
                            images=views,
                            state=policy_state,
                            task=task,
                        ),
                        profile=profile.id,
                    ),
                    dtype=np.float32,
                )
                if chunk.ndim != 2 or chunk.shape[1] != 16 or len(chunk) == 0:
                    raise RuntimeError(f"pi05 returned {chunk.shape}, expected [N,16]")
                if not np.isfinite(chunk).all():
                    raise RuntimeError("pi05 returned non-finite actions")
                # Stitch each replan onto where the arm actually is.  Each
                # replan re-anchors the policy to the rendered image, and the
                # pose it believes can sit a chunk's worth of world-model drift
                # away from the pose we commanded, so its first action would
                # carry the whole gap in one step: a teleport at every replan.
                # Spread the gap over the first --stitch-frames actions
                # instead -- same as the validated c16 client's stitch.
                if self.stitch_frames > 0:
                    gap = (anchor - chunk[0]).astype(np.float32)
                    k = min(self.stitch_frames, len(chunk))
                    w = np.clip(1.0 - (np.arange(k) + 1.0) / k, 0.0, 1.0).astype(np.float32)
                    chunk[:k] += gap[None] * w[:, None]
                fresh = np.concatenate([fresh, chunk])
                anchor = fresh[-1]
                queries += 1

            predicted = time.monotonic()
            retimed, n_frozen = stateend.retime_actions(fresh, speed, chunk_length, now=joints)
            if n_frozen:
                raise ValueError(f"Policy horizon exhausted: {n_frozen} frozen frames; lower speed")
            waypoints = np.concatenate([joints[None], retimed], axis=0)
            raw = encoder(waypoints)  # (chunk_length, 29) raw stateend
            if raw.shape != (chunk_length, 29):
                raise RuntimeError(f"encoder built {raw.shape}, want ({chunk_length}, 29)")
            if not np.isfinite(raw).all():
                raise RuntimeError("encoder produced non-finite actions")

        ended = time.monotonic()
        logging.info(
            "bridge ready task=%s step=%d prep=%.3fs lock_wait=%.3fs "
            "policy=%.3fs action_conversion=%.3fs total=%.3fs",
            profile.id,
            step_index,
            prepared - started,
            locked - prepared,
            predicted - locked,
            ended - predicted,
            ended - started,
        )
        print(
            f"pi05-dream task={profile.id} ckpt={profile.checkpoint_name} "
            f"step={step_index} chunk={chunk_length} speed={speed:g}x "
            f"queries={queries} frozen={n_frozen} |action|max={np.abs(raw).max():.3f} "
            f"grip L {retimed[-1, 7]:.2f} R {retimed[-1, 15]:.2f}",
            flush=True,
        )
        return {
            "action": raw.tolist(),
            "normalized": False,
            "state": {
                **state,  # retain this episode's constant head/body and task
                "format": profile.state_format,
                "policy_profile": profile.id,
                "policy_checkpoint": profile.checkpoint_name,
                "joints": [float(x) for x in retimed[-1]],
            },
            "state_source": "commanded",
        }


class Handler(BaseHTTPRequestHandler):
    bridge: Bridge
    started: float

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/predict":
            return self._respond(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = self.bridge.predict(payload)
            return self._respond(200, result)
        except Exception as error:  # surface errors to the platform, never swallow
            logging.exception("Dream policy bridge failed")
            return self._respond(500, {"error": str(error)})

    def _respond(self, status: int, payload: dict):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8972)
    parser.add_argument("--policy-url", default="ws://127.0.0.1:8712")
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=32,
        choices=(16, 32),
        help="default action rows when the request does not specify 16 or 32",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="override actions per generated frame; default follows the session: "
        "16 -> 3, 32 -> 50/32. A speed that exceeds the policy horizon is rejected",
    )
    parser.add_argument(
        "--stitch-frames",
        type=int,
        default=8,
        help="spread each replan's re-anchoring jump over this many policy steps "
        "(0 disables; 8 matches the validated c16 client)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bridge = Bridge(args.policy_url, args.speed, args.stitch_frames, args.chunk_length)
    Handler.bridge = bridge
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"pi05 dream.policy.v1 bridge on http://{args.host}:{args.port}/predict "
        f"(policy={args.policy_url} default_chunk={args.chunk_length} "
        f"speed={args.speed or 'session mapping'})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
