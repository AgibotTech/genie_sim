"""Serve pi0.5 checkpoints through OpenPI's WebSocket protocol."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time

from .transformers_overlay import install as install_transformers_overlay

_startup_started = time.monotonic()


def _startup(message: str) -> None:
    print(f"DREAM_POLICY_STARTUP {time.monotonic() - _startup_started:.1f}s: {message}", file=sys.stderr, flush=True)


_startup("installing OpenPI transformers overlay")
install_transformers_overlay()
_startup("importing OpenPI policy modules")

from openpi.serving import websocket_policy_server  # noqa: E402
from .inference_policy import load_policy as _load_policy  # noqa: E402
from .model_config import DEFAULT_PROMPT, make_config, resolve_action_space  # noqa: E402
_startup("OpenPI imports complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="openpi checkpoint dir (static single-model mode)")
    parser.add_argument(
        "--task-profiles",
        action="store_true",
        help="switch the registered task's checkpoint per request",
    )
    parser.add_argument(
        "--default-profile", default="cloth", help="task to preload in task-profiles mode"
    )
    parser.add_argument("--asset-id", default="dream")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default: auto)")
    parser.add_argument(
        "--action-space",
        choices=["auto", "absolute", "delta"],
        default="auto",
        help=(
            "how to read the model's 16 numbers. 'auto' takes it from the checkpoint's "
            "assets/<asset-id>/action_space.json, falling back to absolute when there "
            "is none, which is right for every checkpoint predating the flag. The "
            "explicit values are an override and raise if they contradict the marker"
        ),
    )
    args = parser.parse_args()
    if args.task_profiles:
        if args.checkpoint or args.asset_id != "dream":
            parser.error("--task-profiles uses the registered checkpoints and dream assets")
    elif not args.checkpoint:
        parser.error("choose --checkpoint or --task-profiles")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def load_policy(checkpoint):
        delta = resolve_action_space(checkpoint, args.asset_id, args.action_space)
        config = make_config(
            asset_id=args.asset_id,
            action_horizon=args.action_horizon,
            delta_actions_enabled=delta,
        )
        logging.info("loading pi05 policy and normalization assets from %s", checkpoint)
        _startup(f"loading pi0.5 checkpoint: {checkpoint}")
        started = time.monotonic()
        policy = _load_policy(
            config,
            checkpoint,
            default_prompt=DEFAULT_PROMPT,
            device=args.device,
        )
        logging.info("pi05 checkpoint loaded in %.3fs: %s", time.monotonic() - started, checkpoint)
        _startup(f"pi0.5 checkpoint loaded in {time.monotonic() - started:.1f}s")
        return policy

    if args.task_profiles:
        import torch
        from .task_policy import TaskPolicy

        def release_policy():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        policy = TaskPolicy(load_policy, release_policy, args.default_profile)
    else:
        policy = load_policy(args.checkpoint)
    logging.info(
        "serving on ws://%s:%d (asset_id=%s, horizon=%d)",
        args.host,
        args.port,
        args.asset_id,
        args.action_horizon,
    )
    server_type = websocket_policy_server.WebsocketPolicyServer
    if args.task_profiles:
        from .task_policy_server import TaskPolicyServer

        server_type = TaskPolicyServer
    server = server_type(policy=policy, host=args.host, port=args.port, metadata=policy.metadata)
    _startup(f"serving policy on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
