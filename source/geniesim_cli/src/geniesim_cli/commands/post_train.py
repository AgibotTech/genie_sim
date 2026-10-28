# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run post-training simulation services from a local GenieSim checkout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import NoReturn

from geniesim_cli import _env
from geniesim_cli._style import BOLD, CYAN, DIM, MAGENTA, RED, RST


def _error(message: str) -> NoReturn:
    print(f"{RED}❌ {message}{RST}", file=sys.stderr)
    sys.exit(2)


def _exec_script(name: str, args: list[str]) -> NoReturn:
    """Replace the CLI with a checkout script, preserving arguments and signals."""
    override = _env.repo_root() or _env.repo_path()
    cwd = Path.cwd()
    candidates = [Path(override).expanduser()] if override else [cwd, *cwd.parents]
    for root in candidates:
        script = root.resolve() / "source/rlinf_geniesim/scripts" / name
        if script.is_file():
            env = os.environ.copy()
            env["SIM_REPO_ROOT"] = str(root.resolve())
            try:
                os.execvpe("bash", ["bash", str(script), *args], env)
            except OSError as error:
                _error(f"Cannot execute {script}: {error}")
    location = f" at {override}" if override else " from the current directory"
    _error(f"Cannot find source/rlinf_geniesim/scripts/{name}{location}. Set GENIESIM_REPO_ROOT to the checkout.")


def _print_usage() -> None:
    print(f"{BOLD}{MAGENTA}geniesim post-train{RST}")
    print(f"{BOLD}Usage:{RST} geniesim post-train {CYAN}<command>{RST} [args...]")
    print(f"  {CYAN}server start|status|stop{RST}  Manage selected simulation server ports{RST}")
    print(f"  {CYAN}server logs{RST} --port PORT [--follow]  Read one server's log{RST}")
    print(f"  {CYAN}teleop{RST} [args...]  Connect Pico debugging to an initialized server{RST}")
    print(f"{DIM}Run in the prepared server/teleop environment. Clients initialize tasks.{RST}")
    print(f"{DIM}Use 'geniesim post-train server --help' or 'geniesim post-train teleop --help' for options.{RST}")


def _print_server_usage(action: str = "<start|status|stop>") -> None:
    print(f"{BOLD}Usage:{RST} geniesim post-train server {action} PLAN.conf [--only P1,P2] [--dry-run]{RST}")
    print(f"       geniesim post-train server {action} --count N [--base-port P] [--devices 0,1] [options]{RST}")
    print(f"       geniesim post-train server logs [--port PORT] [--follow]{RST}")
    print(f"{DIM}Generated-plan options: --code DIR --assets DIR --no-fabric --rebuild-ws{RST}")
    print(f"{DIM}Shared options: --only P1,P2 --dry-run --barrier-cap-s SECONDS{RST}")
    print(f"{DIM}Use the same plan or count/base-port selection for start, status and stop.{RST}")
    print(f"{DIM}Start requires stopped slots and rolls back only servers started by that invocation.{RST}")


def _logs(args: list[str]) -> NoReturn:
    parser = argparse.ArgumentParser(
        prog="geniesim post-train server logs", description="Read the last 100 server log lines."
    )
    parser.add_argument("--port", type=int, default=5601, help="ZMQ server port (default: 5601)")
    parser.add_argument("--follow", "-f", action="store_true", help="follow appended lines and log replacement")
    options = parser.parse_args(args)
    if not 1 <= options.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    forwarded = ["--port", str(options.port), "--logs"]
    if options.follow:
        forwarded.append("--follow")
    _exec_script("start_sim_server.sh", forwarded)


def run(args: list[str]) -> None:
    """Dispatch server lifecycle, log, and teleop commands without simulator imports."""
    if not args or args[0] in ("-h", "--help", "help"):
        _print_usage()
        return
    command, *rest = args
    if command == "teleop":
        _exec_script("host_run_teleop.sh", rest)
    if command != "server":
        _print_usage()
        _error(f"Unknown post-train command: {command}")
    if not rest or rest[0] in ("-h", "--help", "help"):
        _print_server_usage()
        return
    action, *forwarded = rest
    if action == "logs":
        _logs(forwarded)
    if action not in ("start", "status", "stop"):
        _print_server_usage()
        _error(f"Unknown server action: {action}")
    if any(arg in ("--status", "--stop") for arg in forwarded):
        _error("Select the server action as a subcommand; do not pass --status or --stop.")
    if "--help" in forwarded or "-h" in forwarded:
        _print_server_usage(action)
        return
    if action != "start":
        forwarded.insert(0, f"--{action}")
    _exec_script("start_sim_servers.sh", forwarded)
