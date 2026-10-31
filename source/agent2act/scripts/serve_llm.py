# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Start the regular CLI with a log and trace directory laid out for one run."""

import argparse
import os
from datetime import datetime
from pathlib import Path

from agent2act.cli import add_observation_arguments, add_ui_arguments, run
from agent2act.configuration import load_config
from agent2act.service.action_trace import action_log_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "examples/g2_llm_recipe.json"
    )
    parser.add_argument("--mode", choices=("auto", "motion", "policy"), help="Override configured mode")
    parser.add_argument("--host", help="Override configured listen address")
    parser.add_argument("--port", type=int, help="Override configured listen port")
    add_observation_arguments(parser)
    add_ui_arguments(parser)
    parser.add_argument("--log-file", type=Path, help="Override the default logs/agent-<time>-<pid>.jsonl")
    parser.add_argument("--trace-dir", type=Path, help="Override the default <log-name>.trace directory")
    parser.add_argument("--no-trace", action="store_true", help="Disable full input/output artifacts")
    args = parser.parse_args()
    if args.ui and args.no_trace:
        parser.error("--ui requires trace recording and cannot be combined with --no-trace")
    settings = load_config(args.config)
    key_env = settings.planner.options.get("api_key_env")
    if key_env and not os.environ.get(key_env):
        parser.error(f"set {key_env} before launching")
    log_file = args.log_file or (
        Path(__file__).resolve().parents[1]
        / "logs"
        / f"agent-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}.jsonl"
    )
    print(f"Agent log: {log_file.resolve()}", flush=True)
    command = ["serve", "--config", str(args.config), "--log-file", str(log_file)]
    if args.ui:
        command.extend(["--ui", "--ui-port", str(args.ui_port)])
        if args.no_browser:
            command.append("--no-browser")
    if args.mode is not None:
        command.extend(["--mode", args.mode])
    if args.host is not None:
        command.extend(["--host", args.host])
    if args.port is not None:
        command.extend(["--port", str(args.port)])
    if args.depth is not None:
        command.append("--depth" if args.depth else "--no-depth")
    for flag in ("gripper_center", "height_prior"):
        value = getattr(args, flag)
        if value is not None:
            command.append("--" + ("" if value else "no-") + flag.replace("_", "-"))
    if args.height_calibration is not None:
        command.extend(["--height-calibration", str(args.height_calibration)])
    if not args.no_trace:
        trace_dir = args.trace_dir or log_file.with_suffix(".trace")
        print(f"Request artifacts: {trace_dir.resolve()}", flush=True)
        print(f"Action reasons and timing: {action_log_path(trace_dir).resolve()}", flush=True)
        command.extend(["--trace-dir", str(trace_dir)])
    return run(command)


if __name__ == "__main__":
    raise SystemExit(main())
