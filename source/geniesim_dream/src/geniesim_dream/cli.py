from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_config
from .examples import discover_examples


def main(argv=None):
    parser = argparse.ArgumentParser(prog="geniesim-dream")
    parser.add_argument("command", choices=["serve", "check"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8971)
    parser.add_argument("--backend", choices=["cosmos", "recorded", "mock"])
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.backend:
        config.backend = args.backend
    if args.command == "check":
        verified = config.verify_release()
        examples, rejected = discover_examples(config)
        print(
            json.dumps(
                {
                    "release": verified,
                    "examples": [item.public() for item in examples.values()],
                    "rejected": rejected,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if rejected:
            raise SystemExit(1)
        return
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get(
        config.api_token_env
    ):
        parser.error(f"Set {config.api_token_env} before exposing a non-loopback address")
    if config.backend == "cosmos":
        from .bootstrap import prepare_cosmos_environment

        prepare_cosmos_environment(config.cosmos_repo)
    import uvicorn
    from .app import create_app

    uvicorn.run(create_app(config), host=args.host, port=args.port, workers=1)
